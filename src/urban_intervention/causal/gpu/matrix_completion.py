"""GPU nuclear-norm matrix completion with rolling CV and jackknife refits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .contracts import EstimatorProvenance, PanelData
from .inference import (
    InferenceResult,
    inference_from_standard_error,
    jackknife_standard_error,
)
from .linalg import (
    as_panel_tensors,
    fit_fixed_effects,
    fit_fixed_effects_batched,
    fit_nuclear_norm_completion,
    fit_nuclear_norm_completion_batched,
)
from .runtime import RuntimeConfig, TorchRuntime


@dataclass(frozen=True, slots=True)
class RollingFold:
    training: npt.NDArray[np.bool_]
    score: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        training = np.asarray(self.training, dtype=bool)
        score = np.asarray(self.score, dtype=bool)
        if training.ndim != 2 or training.shape != score.shape:
            raise ValueError("MC fold masks must be equally shaped two-dimensional arrays")
        if np.any(training & score):
            raise ValueError("MC score cells cannot remain in the training mask")
        if not score.any():
            raise ValueError("an MC fold must score at least one cell")
        object.__setattr__(self, "training", training)
        object.__setattr__(self, "score", score)


@dataclass(frozen=True, slots=True)
class MatrixCompletionConfig:
    n_lambdas: int = 20
    lambda_min_ratio: float = 1e-3
    lambdas: tuple[float, ...] | None = None
    fixed_lambda: float | None = None
    folds: int = 20
    cv_prop: float = 0.1
    cv_nobs: int = 1
    buffer: int = 0
    min_pre_periods: int = 1
    cv_rule: Literal["min", "1se"] = "1se"
    max_iter: int = 5000
    tol: float = 1e-5
    inference: Literal["none", "jackknife"] = "none"
    seed: int = 20260725
    batch_cv: bool = True
    batch_inference: bool = True
    inference_batch_size: int = 16
    minimum_valid_fraction: float = 0.9

    def __post_init__(self) -> None:
        if self.n_lambdas < 2 and self.lambdas is None:
            raise ValueError("n_lambdas must be at least two")
        if not 0 < self.lambda_min_ratio < 1:
            raise ValueError("lambda_min_ratio must be in (0, 1)")
        if self.lambdas is not None and (
            not self.lambdas or min(self.lambdas) < 0 or len(set(self.lambdas)) != len(self.lambdas)
        ):
            raise ValueError("lambdas must be unique and non-negative")
        if self.fixed_lambda is not None and self.fixed_lambda < 0:
            raise ValueError("fixed_lambda must be non-negative")
        if self.folds < 2:
            raise ValueError("folds must be at least two")
        if not 0 < self.cv_prop <= 1:
            raise ValueError("cv_prop must be in (0, 1]")
        if self.cv_nobs < 1 or self.buffer < 0:
            raise ValueError("cv_nobs must be positive and buffer non-negative")
        if self.min_pre_periods < 1:
            raise ValueError("min_pre_periods must be positive")
        if self.inference_batch_size < 1:
            raise ValueError("inference_batch_size must be positive")
        if not 0 < self.minimum_valid_fraction <= 1:
            raise ValueError("minimum_valid_fraction must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class MatrixCompletionResult:
    selected_lambda: float
    counterfactual: npt.NDArray[np.float64]
    effect: npt.NDArray[np.float64]
    cv_mean_mspe: dict[float, float]
    cv_se_mspe: dict[float, float]
    jackknife_se: npt.NDArray[np.float64] | None
    jackknife_effect_draws: npt.NDArray[np.float64] | None
    inference: InferenceResult | None
    treatment_start: int
    converged: bool
    iterations: int
    effective_rank: int
    provenance: EstimatorProvenance


def make_rolling_cv_folds(
    observed_untreated: npt.NDArray[np.bool_],
    treated: npt.NDArray[np.bool_],
    *,
    folds: int,
    nobs: int,
    buffer: int,
    proportion: float = 0.1,
    min_pre_periods: int = 1,
    seed: int = 20260725,
) -> list[RollingFold]:
    """Build fect-style rolling-origin masks without future leakage."""
    available = np.asarray(observed_untreated, dtype=bool)
    treated = np.asarray(treated, dtype=bool)
    if available.ndim != 2 or treated.shape != available.shape:
        raise ValueError("available and treated must be equally shaped 2D masks")
    eligible_times: list[npt.NDArray[np.int64]] = []
    eligible_units: list[int] = []
    for unit in range(available.shape[1]):
        times = np.flatnonzero(available[:, unit]).astype(np.int64)
        treated_periods = np.flatnonzero(treated[:, unit])
        if treated_periods.size:
            times = times[times < treated_periods[0]]
        eligible_times.append(times)
        if len(times) >= min_pre_periods + nobs:
            eligible_units.append(unit)
    if not eligible_units:
        raise ValueError("no unit has enough observations for rolling MC CV")
    sampled_count = max(1, int(np.floor(proportion * len(eligible_units) + 0.5)))
    sampled_count = min(sampled_count, len(eligible_units))
    rng = np.random.default_rng(seed)
    result: list[RollingFold] = []
    for _ in range(folds):
        training = available.copy()
        score = np.zeros_like(available)
        sampled = np.sort(rng.choice(eligible_units, size=sampled_count, replace=False))
        for unit in sampled:
            times = eligible_times[int(unit)]
            first_anchor = min_pre_periods
            last_anchor = len(times) - nobs
            anchor = int(rng.integers(first_anchor, last_anchor + 1))
            holdout = times[anchor : anchor + nobs]
            score[holdout, unit] = True
            buffer_start = max(0, anchor - buffer)
            training[times[buffer_start:], unit] = False
        result.append(RollingFold(training=training, score=score))
    return result


def make_lambda_grid(y: Any, mask: Any, config: MatrixCompletionConfig) -> tuple[float, ...]:
    if config.lambdas is not None:
        return tuple(sorted((float(value) for value in config.lambdas), reverse=True))
    fixed = fit_fixed_effects(y, mask, force="two-way", tol=config.tol * 0.1)
    residual = (y - fixed.matrix()).masked_fill(~mask.bool(), 0)
    maximum = float(residual.svd().S.max().detach().cpu()) / y.numel()
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("cannot construct an MC lambda grid from a zero-rank panel")
    # fect emits nlambda-1 log-spaced positive values (10^0 through 10^-3)
    # and appends an unregularised zero candidate.
    positive = np.geomspace(maximum, maximum * config.lambda_min_ratio, config.n_lambdas - 1)
    return tuple(float(value) for value in positive) + (0.0,)


def _select_lambda(
    means: dict[float, float],
    standard_errors: dict[float, float],
    rule: str,
) -> float:
    finite = [value for value in means if np.isfinite(means[value])]
    if not finite:
        raise RuntimeError("all matrix-completion lambda candidates failed CV")
    minimum = min(finite, key=lambda value: (means[value], -value))
    if rule == "min":
        return minimum
    threshold = means[minimum] + standard_errors[minimum]
    # Larger lambda is the simpler/more strongly regularised model.
    return max(value for value in finite if means[value] <= threshold)


def _jackknife_refits(
    y: Any,
    observed: Any,
    fit_mask: Any,
    treated: Any,
    target_index: int,
    selected_lambda: float,
    full_effect: npt.NDArray[np.float64],
    config: MatrixCompletionConfig,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    unit_count = y.shape[1]
    omitted_units = [index for index in range(unit_count) if index != target_index]
    draws = np.full((len(omitted_units), y.shape[0]), np.nan, dtype=np.float64)
    target_y = y[:, target_index].detach().cpu().numpy()
    target_observed = observed[:, target_index].detach().cpu().numpy()
    target_y[~target_observed] = np.nan
    keep_sets = [
        [index for index in range(unit_count) if index != omitted]
        for omitted in omitted_units
    ]
    for start in range(0, len(keep_sets), config.inference_batch_size):
        local_keeps = keep_sets[start : start + config.inference_batch_size]
        local_omitted = omitted_units[start : start + config.inference_batch_size]
        if config.batch_inference:
            batch_shape = (len(local_keeps), y.shape[0], y.shape[1] - 1)
            y_batch = y.new_empty(batch_shape)
            mask_batch = fit_mask.new_empty(batch_shape)
            for row, keep in enumerate(local_keeps):
                y_batch[row] = y[:, keep]
                mask_batch[row] = fit_mask[:, keep]
            fits = fit_nuclear_norm_completion_batched(
                y_batch,
                mask_batch,
                penalty=selected_lambda,
                force="two-way",
                max_iter=config.max_iter,
                tol=config.tol,
            )
        else:
            fits = [
                fit_nuclear_norm_completion(
                    y[:, keep],
                    fit_mask[:, keep],
                    penalty=selected_lambda,
                    force="two-way",
                    max_iter=config.max_iter,
                    tol=config.tol,
                )
                for keep in local_keeps
            ]
        for row, (omitted, fit) in enumerate(
            zip(local_omitted, fits, strict=True), start=start
        ):
            if not fit.converged:
                continue
            local_target = target_index - int(omitted < target_index)
            counterfactual = fit.fitted[:, local_target].detach().cpu().numpy()
            draws[row] = target_y - counterfactual
    standard_error, _ = jackknife_standard_error(full_effect, draws)
    return draws, standard_error


def fit_matrix_completion(
    panel: PanelData,
    *,
    config: MatrixCompletionConfig | None = None,
    cv_folds: list[RollingFold] | None = None,
    lambda_grid: tuple[float, ...] | None = None,
    runtime: TorchRuntime | None = None,
) -> MatrixCompletionResult:
    """Fit one-target two-way fixed-effect matrix completion."""
    config = config or MatrixCompletionConfig()
    runtime = runtime or TorchRuntime(RuntimeConfig(seed=config.seed))
    target_index = panel.single_treated_unit()
    treatment_start = panel.treatment_start()
    y, observed, treated = as_panel_tensors(panel, runtime)
    fit_mask = observed & ~treated
    if int(fit_mask[:, target_index].sum()) < config.min_pre_periods:
        raise ValueError("treated target lacks enough observed pre-treatment periods")
    if bool((fit_mask.sum(dim=0) == 0).any()):
        raise ValueError("an MC unit has no observed untreated outcomes")
    if bool((fit_mask.sum(dim=1) == 0).any()):
        raise ValueError("an MC period has no observed untreated outcomes")
    if config.fixed_lambda is not None:
        lambdas: tuple[float, ...] = (config.fixed_lambda,)
        selected_lambda = config.fixed_lambda
        means: dict[float, float] = {}
        standard_errors: dict[float, float] = {}
    else:
        if cv_folds is None:
            cv_folds = make_rolling_cv_folds(
                fit_mask.detach().cpu().numpy(),
                treated.detach().cpu().numpy(),
                folds=config.folds,
                nobs=config.cv_nobs,
                buffer=config.buffer,
                proportion=config.cv_prop,
                min_pre_periods=config.min_pre_periods,
                seed=config.seed,
            )
            cv_contract = "python_rolling"
        else:
            cv_contract = "persisted_rolling"
        if len(cv_folds) != config.folds:
            raise ValueError("MC CV contract fold count does not match config.folds")
        lambdas = lambda_grid if lambda_grid is not None else make_lambda_grid(y, fit_mask, config)
        expected_lambdas = len(config.lambdas) if config.lambdas is not None else config.n_lambdas
        if len(lambdas) != expected_lambdas:
            raise ValueError("MC lambda grid length does not match the configured candidates")
        scores: dict[float, list[float]] = {penalty: [] for penalty in lambdas}
        training_tensors: list[Any] = []
        score_tensors: list[Any] = []
        for fold in cv_folds:
            if fold.training.shape != tuple(fit_mask.shape):
                raise ValueError("MC CV contract shape does not match the panel")
            training = runtime.tensor(fold.training, dtype=runtime.torch.bool)
            score = runtime.tensor(fold.score, dtype=runtime.torch.bool)
            if bool((training & ~fit_mask).any()):
                raise ValueError("MC CV contract trains on an unavailable panel cell")
            if bool((score & ~fit_mask).any()):
                raise ValueError("MC CV contract scores an unavailable panel cell")
            training_tensors.append(training)
            score_tensors.append(score)
        training_batch = runtime.torch.stack(training_tensors)
        if config.batch_cv:
            initial_fixed = fit_fixed_effects_batched(
                y,
                training_batch,
                force="two-way",
                max_iter=config.max_iter,
                tol=min(config.tol * 0.1, 1e-12),
            )
            initial_fit_batch = runtime.torch.stack([fit.matrix() for fit in initial_fixed])
        else:
            initial_fit_batch = runtime.torch.stack(
                [
                    fit_fixed_effects(
                        y,
                        training,
                        force="two-way",
                        max_iter=config.max_iter,
                        tol=min(config.tol * 0.1, 1e-12),
                    ).matrix()
                    for training in training_tensors
                ]
            )
        score_sse: dict[float, float] = {penalty: 0.0 for penalty in lambdas}
        score_count: dict[float, int] = {penalty: 0 for penalty in lambdas}
        evaluated: list[float] = []
        cv_tolerance = max(config.tol, 1e-3)
        for penalty in lambdas:
            if config.batch_cv:
                fits = fit_nuclear_norm_completion_batched(
                    y,
                    training_batch,
                    penalty=penalty,
                    force="two-way",
                    max_iter=config.max_iter,
                    tol=cv_tolerance,
                    initial_fits=initial_fit_batch,
                )
            else:
                fits = [
                    fit_nuclear_norm_completion(
                        y,
                        training,
                        penalty=penalty,
                        force="two-way",
                        max_iter=config.max_iter,
                        tol=cv_tolerance,
                        initial_fit=initial_fit,
                    )
                    for training, initial_fit in zip(
                        training_tensors, initial_fit_batch, strict=True
                    )
                ]
            for score, fit in zip(score_tensors, fits, strict=True):
                if not fit.converged:
                    raise RuntimeError(
                        f"MC CV failed to converge for lambda {penalty:.17g}"
                    )
                error = y[score] - fit.fitted[score]
                squared = error.square()
                scores[penalty].append(float(squared.mean().detach().cpu()))
                score_sse[penalty] += float(squared.sum().detach().cpu())
                score_count[penalty] += int(error.numel())
            evaluated.append(penalty)
        means = {penalty: score_sse[penalty] / score_count[penalty] for penalty in evaluated}
        standard_errors = {
            penalty: float(np.std(values, ddof=1) / np.sqrt(len(values)))
            for penalty, values in scores.items()
            if penalty in means
        }
        selected_lambda = _select_lambda(means, standard_errors, config.cv_rule)
    final_fit = fit_nuclear_norm_completion(
        y,
        fit_mask,
        penalty=selected_lambda,
        force="two-way",
        max_iter=config.max_iter,
        tol=config.tol,
    )
    counterfactual = final_fit.fitted[:, target_index].detach().cpu().numpy()
    effect = panel.y[:, target_index] - counterfactual
    jackknife_draws = None
    jackknife_se = None
    inference = None
    if config.inference == "jackknife":
        jackknife_draws, jackknife_se = _jackknife_refits(
            y,
            observed,
            fit_mask,
            treated,
            target_index,
            selected_lambda,
            effect,
            config,
        )
        valid_repetitions = np.sum(np.isfinite(jackknife_draws), axis=0).astype(
            np.int64
        )
        inference = inference_from_standard_error(
            effect,
            jackknife_se,
            method="mc_unit_jackknife",
            requested_repetitions=jackknife_draws.shape[0],
            valid_repetitions=valid_repetitions,
            replicate_estimates=jackknife_draws,
        )
        estimable_repetitions = jackknife_draws.shape[0]
        minimum = max(
            2,
            int(np.ceil(estimable_repetitions * config.minimum_valid_fraction)),
        )
        assert panel.treated is not None
        treated_periods = panel.treated[:, target_index]
        estimable_treated_periods = treated_periods & observed[:, target_index].detach().cpu().numpy()
        if np.any(valid_repetitions[estimable_treated_periods] < minimum):
            raise RuntimeError("too few converged MC jackknife refits")
    provenance = EstimatorProvenance(
        estimator="matrix_completion",
        backend="pytorch",
        formal_eligible=False,
        numerical_policy=runtime.metadata()
        | {
            "cv_method": "rolling",
            "cv_rule": config.cv_rule,
            "cv_nobs": config.cv_nobs,
            "buffer": config.buffer,
            "cv_prop": config.cv_prop,
            "cv_contract": "not_used" if config.fixed_lambda is not None else cv_contract,
            "cv_tolerance": max(config.tol, 1e-3),
            "lambda_count": len(lambdas),
            "tuning_source": "fixed" if config.fixed_lambda is not None else "gpu_cv",
            "inference": config.inference,
            "batch_cv": config.batch_cv,
            "batch_inference": config.batch_inference,
            "inference_batch_size": config.inference_batch_size,
            "minimum_valid_fraction": config.minimum_valid_fraction,
        },
        notes=(
            "lambda candidates reuse the same fixed-effect initialization within each fold",
            "compiled fect MC parity is required before production selection",
        ),
    )
    return MatrixCompletionResult(
        selected_lambda=selected_lambda,
        counterfactual=counterfactual,
        effect=effect,
        cv_mean_mspe=means,
        cv_se_mspe=standard_errors,
        jackknife_se=jackknife_se,
        jackknife_effect_draws=jackknife_draws,
        inference=inference,
        treatment_start=treatment_start,
        converged=final_fit.converged,
        iterations=final_fit.iterations,
        effective_rank=int((final_fit.singular_values > 0).sum()),
        provenance=provenance,
    )
