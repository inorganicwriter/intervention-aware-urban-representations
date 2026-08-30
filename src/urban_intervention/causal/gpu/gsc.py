"""GPU generalized synthetic control with an explicit fect-style contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .contracts import EstimatorProvenance, PanelData
from .inference import InferenceResult, normal_inference_from_replicates
from .linalg import (
    LowRankFit,
    as_panel_tensors,
    fit_interactive_fixed_effects,
    fit_interactive_fixed_effects_batched,
    fit_interactive_fixed_effects_panel_batch,
)
from .runtime import RuntimeConfig, TorchRuntime

BootstrapMode = Literal["none", "auto", "reference_empirical", "reference_ar"]


def estimate_gsc_inference_batch_bytes(
    periods: int,
    controls: int,
    batch_size: int,
) -> int:
    """Conservative live-tensor estimate for batched GSC inference.

    The factor covers the synthetic panel, masks, fitted values, residuals and
    decomposition workspaces in float64.  It is intentionally more cautious
    than counting only the input tensor.
    """
    if periods < 1 or controls < 1 or batch_size < 1:
        raise ValueError("periods, controls and batch_size must be positive")
    return int(periods * (controls + 1) * batch_size * 96)


def _resolved_inference_batch_size(
    y: Any,
    controls: int,
    requested: int,
) -> int:
    if y.device.type != "cuda":
        return requested
    torch = __import__("torch")
    free_bytes, _ = torch.cuda.mem_get_info(y.device)
    bytes_per_batch = estimate_gsc_inference_batch_bytes(y.shape[0], controls, 1)
    # Keep allocator/SVD headroom and respect memory already occupied by the
    # fitted panel.  A single replicate that does not fit is rejected before
    # PyTorch attempts a multi-gigabyte allocation.
    maximum = int((int(free_bytes) * 0.70) // bytes_per_batch)
    if maximum < 1:
        required_gib = bytes_per_batch / 1024**3
        free_gib = int(free_bytes) / 1024**3
        raise RuntimeError(
            "GSC inference panel cannot fit on the selected GPU: "
            f"estimated {required_gib:.2f} GiB per replicate with "
            f"{free_gib:.2f} GiB currently free"
        )
    return min(requested, maximum)


@dataclass(frozen=True, slots=True)
class GSCConfig:
    ranks: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    folds: int = 5
    cv_prop: float = 0.1
    cv_nobs: int = 3
    cv_buffer: int = 1
    cv_rule: Literal["min", "1se"] = "1se"
    fixed_rank: int | None = None
    min_pre_periods: int = 5
    normalize: bool = True
    max_iter: int = 5000
    tol: float = 1e-5
    bootstrap_mode: BootstrapMode = "none"
    n_bootstrap: int = 0
    # Keep the formal inference seed aligned with the R reference contract.
    # The R runner resets its stream to 20260723 before the fixed-r bootstrap.
    seed: int = 20260723
    batch_cv: bool | None = None
    inference_batch_size: int = 16
    minimum_valid_fraction: float = 0.9

    def __post_init__(self) -> None:
        if not self.ranks or min(self.ranks) < 0:
            raise ValueError("ranks must contain non-negative integers")
        if tuple(sorted(set(self.ranks))) != self.ranks:
            raise ValueError("ranks must be unique and increasing")
        if self.fixed_rank is not None and self.fixed_rank < 0:
            raise ValueError("fixed_rank must be non-negative")
        if self.folds < 2:
            raise ValueError("folds must be at least two")
        if not 0 < self.cv_prop < 1:
            raise ValueError("cv_prop must be in (0, 1)")
        if self.cv_nobs < 1:
            raise ValueError("cv_nobs must be at least one")
        if self.cv_buffer < 0:
            raise ValueError("cv_buffer cannot be negative")
        if self.min_pre_periods < 2:
            raise ValueError("min_pre_periods must be at least two")
        if self.bootstrap_mode == "none" and self.n_bootstrap:
            raise ValueError("n_bootstrap must be zero when bootstrap_mode='none'")
        if self.bootstrap_mode != "none" and self.n_bootstrap < 2:
            raise ValueError("bootstrap inference requires at least two replicates")
        if self.inference_batch_size < 1:
            raise ValueError("inference_batch_size must be positive")
        if not 0 < self.minimum_valid_fraction <= 1:
            raise ValueError("minimum_valid_fraction must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class GSCResult:
    selected_rank: int
    counterfactual: npt.NDArray[np.float64]
    effect: npt.NDArray[np.float64]
    cv_mean_mspe: dict[int, float]
    cv_se_mspe: dict[int, float]
    bootstrap_se: npt.NDArray[np.float64] | None
    bootstrap_draws: npt.NDArray[np.float64] | None
    inference: InferenceResult | None
    treatment_start: int
    converged: bool
    iterations: int
    provenance: EstimatorProvenance


@dataclass(frozen=True, slots=True)
class GSCFold:
    """One fect rolling-CV fold on the control panel."""

    removed: npt.NDArray[np.bool_]
    scored: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        removed = np.asarray(self.removed, dtype=bool)
        scored = np.asarray(self.scored, dtype=bool)
        if removed.ndim != 2 or removed.shape != scored.shape:
            raise ValueError("GSC fold masks must be equally shaped two-dimensional arrays")
        if np.any(scored & ~removed):
            raise ValueError("every scored GSC cell must also be removed from training")
        if not scored.any():
            raise ValueError("a GSC fold must score at least one cell")
        object.__setattr__(self, "removed", removed)
        object.__setattr__(self, "scored", scored)


def make_rolling_cv_folds(
    observed: npt.NDArray[np.bool_],
    *,
    folds: int,
    proportion: float,
    min_pre_periods: int,
    holdout_periods: int,
    buffer_periods: int,
    seed: int,
) -> list[GSCFold]:
    """Create fect-style forward-only rolling folds.

    R-generated fold contracts should be used for strict parity because R's
    ``sample.int`` stream is not NumPy-compatible.  This native constructor
    preserves the same masking semantics for standalone Python use.
    """
    observed = np.asarray(observed, dtype=bool)
    if observed.ndim != 2:
        raise ValueError("observed must be a two-dimensional mask")
    eligible_units = [
        unit
        for unit in range(observed.shape[1])
        if int(observed[:, unit].sum()) >= min_pre_periods + holdout_periods
    ]
    if not eligible_units:
        raise ValueError("no control unit has enough observations for rolling GSC CV")
    sampled_count = max(1, int(np.floor(proportion * len(eligible_units) + 0.5)))
    sampled_count = min(sampled_count, len(eligible_units))
    rng = np.random.default_rng(seed)
    result: list[GSCFold] = []
    for fold in range(folds):
        del fold
        sampled = rng.choice(eligible_units, size=sampled_count, replace=False)
        removed = np.zeros_like(observed)
        scored = np.zeros_like(observed)
        for unit in np.sort(sampled):
            times = np.flatnonzero(observed[:, unit])
            first_anchor = min_pre_periods
            last_anchor = len(times) - holdout_periods
            anchor = int(rng.integers(first_anchor, last_anchor + 1))
            holdout = times[anchor : anchor + holdout_periods]
            scored[holdout, unit] = True
            buffer_start = max(0, anchor - buffer_periods)
            removed[times[buffer_start:], unit] = True
        result.append(GSCFold(removed=removed, scored=scored))
    return result


def make_all_unit_cv_masks(
    observed: npt.NDArray[np.bool_],
    *,
    folds: int,
    proportion: float,
    min_observed_per_unit: int,
    seed: int,
) -> list[npt.NDArray[np.bool_]]:
    """Compatibility adapter for the superseded prototype mask API."""
    return [
        fold.scored
        for fold in make_rolling_cv_folds(
            observed,
            folds=folds,
            proportion=proportion,
            min_pre_periods=min_observed_per_unit,
            holdout_periods=1,
            buffer_periods=0,
            seed=seed,
        )
    ]


def _select_one_standard_error(
    means: dict[int, float],
    standard_errors: dict[int, float],
    rule: str,
) -> int:
    finite = [rank for rank in sorted(means) if np.isfinite(means[rank])]
    if not finite:
        raise RuntimeError("all GSC rank candidates failed cross-validation")
    minimum = min(finite, key=lambda rank: (means[rank], rank))
    if rule == "min":
        return minimum
    threshold = means[minimum] + standard_errors[minimum]
    return next(rank for rank in finite if means[rank] <= threshold)


def _max_identifiable_rank(observed: npt.ArrayLike) -> int:
    """Return the largest factor rank admitted by fect's observation checks."""
    mask = np.asarray(observed, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("GSC control observation mask must be a non-empty matrix")
    periods, units = mask.shape
    minimum_unit_observations = int(mask.sum(axis=0).min())
    dimension_limit = min(periods, units, minimum_unit_observations - 1)
    observed_count = int(mask.sum())
    valid = [
        rank
        for rank in range(max(0, dimension_limit) + 1)
        if observed_count - rank * (units + periods) + rank * rank > 0
    ]
    if not valid:
        raise ValueError("GSC control panel does not identify an additive model")
    return max(valid)


def _target_counterfactual(
    y: Any,
    observed: Any,
    treated: Any,
    target_index: int,
    control_fit: LowRankFit,
    rank: int,
) -> Any:
    """Estimate target loadings using untreated target periods only."""
    pre_mask = observed[:, target_index] & ~treated[:, target_index]
    if int(pre_mask.sum()) < rank + 1:
        raise ValueError("treated target has too few pre-periods for the selected rank")
    time_effect = control_fit.fixed.grand + control_fit.fixed.time
    if rank:
        left, singular_values, _ = control_fit.low_rank.svd()
        factors = left[:, :rank] * singular_values[:rank]
        design = y.new_ones((y.shape[0], rank + 1))
        design[:, :rank] = factors
    else:
        design = y.new_ones((y.shape[0], 1))
    response = y[:, target_index] - time_effect
    coefficients = design[pre_mask].pinverse() @ response[pre_mask]
    return time_effect + design @ coefficients


def _fit_controls(
    y: Any,
    observed: Any,
    control_indices: Any,
    rank: int,
    config: GSCConfig,
) -> LowRankFit:
    return fit_interactive_fixed_effects(
        y[:, control_indices],
        observed[:, control_indices],
        rank=rank,
        force="two-way",
        max_iter=config.max_iter,
        tol=config.tol,
    )


def _draw_covering_controls(
    rng: np.random.Generator,
    observed: npt.NDArray[np.bool_],
    count: int,
    *,
    candidates: npt.NDArray[np.int64] | None = None,
    maximum_attempts: int = 10_000,
) -> npt.NDArray[np.int64]:
    """Sample controls with replacement while retaining period support."""
    pool = (
        np.arange(observed.shape[1], dtype=np.int64)
        if candidates is None
        else np.asarray(candidates, dtype=np.int64)
    )
    if pool.size == 0:
        raise ValueError("no controls are available for bootstrap resampling")
    for _ in range(maximum_attempts):
        sampled = rng.choice(pool, size=count, replace=True).astype(np.int64)
        if np.all(observed[:, sampled].any(axis=1)):
            return sampled
    raise RuntimeError("failed to sample a bootstrap control panel with period support")


def _residual_vcov(
    residual: npt.NDArray[np.float64],
    *,
    cov_ar: int | None = None,
) -> npt.NDArray[np.float64]:
    """Port ``fect:::res.vcov`` without changing its covariance geometry.

    The R reference zeroes covariance entries beyond ``cov.ar`` and applies
    a pairwise-observation correction to the remaining entries.  It does not
    project the result to the PSD cone or add diagonal jitter.  Those
    operations alter the parametric bootstrap distribution and therefore are
    not appropriate on the formal parity path.
    """
    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("residual must be a time-by-unit matrix")
    if cov_ar is None:
        cov_ar = values.shape[0] - 1
    if cov_ar < 0:
        raise ValueError("cov_ar must be non-negative")
    missing = ~np.isfinite(values)
    filled = np.where(missing, 0.0, values)
    covariance = filled @ filled.T
    weights = np.zeros_like(covariance)
    for left in range(values.shape[0]):
        for right in range(left, values.shape[0]):
            if right - left > cov_ar:
                continue
            jointly_observed = int(np.sum(~missing[left] & ~missing[right]))
            weight = min(1.0 / jointly_observed, 1.0) if jointly_observed else 1.0
            weights[left, right] = weight
            weights[right, left] = weight
    return covariance * weights


def _fect_bootstrap_fit_out(
    fitted: npt.NDArray[np.float64],
    observed: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Build the ``fect_boot`` fitted-value matrix before error injection."""
    values = np.asarray(fitted, dtype=np.float64)
    mask = np.asarray(observed, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("fitted and observed must have the same shape")
    return np.where(mask, values, 0.0)


def _placebo_target_errors(
    y: Any,
    observed: Any,
    treated: Any,
    controls: Any,
    rank: int,
    config: GSCConfig,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], int]:
    """Generate fect-style pseudo-treated control residual paths."""
    control_y = y[:, controls]
    control_observed = observed[:, controls]
    observed_np = control_observed.detach().cpu().numpy()
    target_treated = treated[:, 0] if treated.shape[1] == 1 else treated[:, treated.any(dim=0)].squeeze(1)
    treatment_start = int(np.flatnonzero(target_treated.detach().cpu().numpy())[0])
    valid_fake = np.flatnonzero(
        (observed_np[:treatment_start].sum(axis=0) >= rank + 1)
        & (observed_np[treatment_start:].sum(axis=0) >= 1)
    ).astype(np.int64)
    if valid_fake.size == 0:
        raise ValueError("GSC bootstrap has no valid pseudo-treated control")
    unit_count = control_y.shape[1]
    inference_batch_size = _resolved_inference_batch_size(
        y, unit_count, config.inference_batch_size
    )
    errors = np.full((config.n_bootstrap, y.shape[0]), np.nan, dtype=np.float64)
    for batch_start in range(0, config.n_bootstrap, inference_batch_size):
        batch_size = min(inference_batch_size, config.n_bootstrap - batch_start)
        panels = y.new_empty((batch_size, y.shape[0], unit_count + 1))
        masks = observed.new_empty((batch_size, y.shape[0], unit_count + 1))
        local_treated = treated.new_zeros((batch_size, y.shape[0], unit_count + 1))
        local_treated[:, :, 0] = target_treated
        for row in range(batch_size):
            fake_target = int(rng.choice(valid_fake))
            donor_pool = np.asarray(
                [index for index in range(unit_count) if index != fake_target],
                dtype=np.int64,
            )
            sampled = _draw_covering_controls(
                rng,
                observed_np,
                unit_count,
                candidates=donor_pool,
            )
            panels[row, :, 0] = control_y[:, fake_target]
            panels[row, :, 1:] = control_y[:, sampled]
            masks[row, :, 0] = control_observed[:, fake_target]
            masks[row, :, 1:] = control_observed[:, sampled]
        fits = fit_interactive_fixed_effects_panel_batch(
            panels[:, :, 1:],
            masks[:, :, 1:],
            ranks=(rank,) * batch_size,
            force="two-way",
            max_iter=config.max_iter,
            tol=config.tol,
        )
        for row, fit in enumerate(fits):
            if not fit.converged:
                continue
            counterfactual = _target_counterfactual(
                panels[row],
                masks[row],
                local_treated[row],
                0,
                fit,
                rank,
            )
            draw = (panels[row, :, 0] - counterfactual).detach().cpu().numpy()
            draw[~masks[row, :, 0].detach().cpu().numpy()] = np.nan
            errors[batch_start + row] = draw
    minimum = max(2, int(np.ceil(config.n_bootstrap * config.minimum_valid_fraction)))
    if np.any(np.sum(np.isfinite(errors), axis=0)[treatment_start:] < minimum):
        raise RuntimeError("too few valid pseudo-treated GSC error paths")
    return errors, inference_batch_size


def _bootstrap_effects(
    y: Any,
    observed: Any,
    treated: Any,
    target_index: int,
    controls: Any,
    rank: int,
    fit: LowRankFit,
    config: GSCConfig,
) -> tuple[npt.NDArray[np.float64], int]:
    """Run the two-stage fect-style parametric bootstrap for one target."""
    control_observed = observed[:, controls]
    # ``fect`` resolves para.error from the raw observation mask ``I``.
    complete = bool(observed.all())
    resolved_mode = (
        "reference_empirical"
        if config.bootstrap_mode == "auto" and complete
        else "reference_ar"
        if config.bootstrap_mode == "auto"
        else config.bootstrap_mode
    )
    if resolved_mode == "reference_empirical" and not complete:
        raise ValueError("reference_empirical bootstrap requires a complete control panel")
    # CV folds are supplied separately on the formal path, so the bootstrap
    # stream must start from the contract seed itself, as the R reference does.
    rng = np.random.default_rng(config.seed)
    control_y = y[:, controls]
    residual = (control_y - fit.fitted).detach().cpu().numpy()
    observed_np = control_observed.detach().cpu().numpy()
    residual[~observed_np] = np.nan
    fitted_controls = fit.fitted.detach().cpu().numpy()
    # fect::fect_boot constructs fit.out = out$Y.ct and then sets every
    # I == 0 cell to zero before adding the simulated error.  The zero is
    # intentional: impute_Y0 receives the I/II masks and reconstructs those
    # cells during the bootstrap refit.  Supplying the already-completed
    # value here changes the EM starting matrix and systematically narrows
    # the GSC bootstrap distribution on incomplete donor panels.
    fit_out_controls = _fect_bootstrap_fit_out(fitted_controls, observed_np)
    fitted_target = _target_counterfactual(
        y,
        observed,
        treated,
        target_index,
        fit,
        rank,
    ).detach().cpu().numpy()
    target_observed = observed[:, target_index].detach().cpu().numpy()
    fit_out_target = _fect_bootstrap_fit_out(fitted_target, target_observed)
    placebo_errors, placebo_batch_size = _placebo_target_errors(
        y,
        observed,
        treated[:, target_index : target_index + 1],
        controls,
        rank,
        config,
        rng,
    )
    target_covariance = control_covariance = None
    if resolved_mode == "reference_ar":
        target_covariance = _residual_vcov(placebo_errors.T)
        control_covariance = _residual_vcov(residual)
    draws = np.full((config.n_bootstrap, y.shape[0]), np.nan, dtype=np.float64)
    target_treated = treated[:, target_index]
    inference_batch_size = _resolved_inference_batch_size(
        y, control_y.shape[1], config.inference_batch_size
    )
    for batch_start in range(0, config.n_bootstrap, inference_batch_size):
        batch_size = min(inference_batch_size, config.n_bootstrap - batch_start)
        boot_y = y.new_empty((batch_size, y.shape[0], control_y.shape[1] + 1))
        boot_observed = observed.new_empty(boot_y.shape)
        boot_treated = treated.new_zeros(boot_y.shape)
        boot_treated[:, :, 0] = target_treated
        for row in range(batch_size):
            sampled = _draw_covering_controls(rng, observed_np, control_y.shape[1])
            sampled_observed = observed_np[:, sampled]
            if resolved_mode == "reference_empirical":
                valid_placebos = np.flatnonzero(
                    np.all(np.isfinite(placebo_errors[:, target_observed]), axis=1)
                )
                if valid_placebos.size == 0:
                    raise RuntimeError("no complete pseudo-treated errors for empirical bootstrap")
                target_error = placebo_errors[int(rng.choice(valid_placebos))]
                error_columns = rng.integers(0, residual.shape[1], size=control_y.shape[1])
                control_error = residual[:, error_columns]
            else:
                assert target_covariance is not None and control_covariance is not None
                target_error = rng.multivariate_normal(
                    np.zeros(y.shape[0]),
                    target_covariance,
                    check_valid="ignore",
                    method="svd",
                ).T
                control_error = rng.multivariate_normal(
                    np.zeros(y.shape[0]),
                    control_covariance,
                    size=control_y.shape[1],
                    check_valid="ignore",
                    method="svd",
                ).T
                control_error[~sampled_observed] = 0.0
            target_error = np.where(target_observed, target_error, 0.0)
            synthetic_target = fit_out_target + target_error
            synthetic_controls = fit_out_controls[:, sampled] + control_error
            boot_y[row, :, 0] = y.new_tensor(synthetic_target)
            boot_y[row, :, 1:] = y.new_tensor(synthetic_controls)
            boot_observed[row, :, 0] = observed[:, target_index]
            boot_observed[row, :, 1:] = y.new_tensor(
                sampled_observed, dtype=observed.dtype
            )
        boot_fits = fit_interactive_fixed_effects_panel_batch(
            boot_y[:, :, 1:],
            boot_observed[:, :, 1:],
            ranks=(rank,) * batch_size,
            force="two-way",
            max_iter=config.max_iter,
            tol=config.tol,
        )
        for row, boot_fit in enumerate(boot_fits):
            if not boot_fit.converged:
                continue
            bootstrap_counterfactual = _target_counterfactual(
                boot_y[row],
                boot_observed[row],
                boot_treated[row],
                0,
                boot_fit,
                rank,
            )
            effect_draw = (
                boot_y[row, :, 0] - bootstrap_counterfactual
            ).detach().cpu().numpy()
            effect_draw[~target_observed] = np.nan
            draws[batch_start + row] = effect_draw
    return draws, min(placebo_batch_size, inference_batch_size)


def fit_gsc(
    panel: PanelData,
    *,
    config: GSCConfig | None = None,
    cv_folds: list[GSCFold] | None = None,
    runtime: TorchRuntime | None = None,
) -> GSCResult:
    """Fit one-treated-unit generalized synthetic control on CPU or GPU."""
    config = config or GSCConfig()
    runtime = runtime or TorchRuntime(RuntimeConfig(seed=config.seed))
    target_index = panel.single_treated_unit()
    treatment_start = panel.treatment_start()
    if treatment_start < config.min_pre_periods:
        raise ValueError("treated target does not meet min_pre_periods")
    y, observed, treated = as_panel_tensors(panel, runtime)
    controls = runtime.torch.tensor(
        [index for index in range(y.shape[1]) if index != target_index],
        dtype=runtime.torch.long,
        device=runtime.device,
    )
    fit_mask = observed & ~treated
    if int(fit_mask[:, target_index].sum()) < config.min_pre_periods:
        raise ValueError("treated target lacks enough observed pre-treatment periods")
    scale = y.new_ones(())
    if config.normalize:
        scale = y[fit_mask].std(unbiased=True)
        if not bool(runtime.torch.isfinite(scale)) or float(scale) <= 0:
            raise ValueError("cannot normalize a panel with zero/invalid outcome variance")
        y = y / scale

    control_observed = fit_mask[:, controls]
    if bool((control_observed.sum(dim=0) == 0).any()):
        raise ValueError("a GSC control unit has no observed untreated outcomes")
    if bool((control_observed.sum(dim=1) == 0).any()):
        raise ValueError("a GSC period has no observed control outcomes")
    target_pre_periods = int(fit_mask[:, target_index].sum())
    max_target_rank = target_pre_periods - 1
    max_control_rank = _max_identifiable_rank(
        control_observed.detach().cpu().numpy()
    )
    valid_ranks = tuple(
        rank
        for rank in config.ranks
        if rank <= min(max_control_rank, max_target_rank)
    )
    if not valid_ranks:
        raise ValueError("no requested GSC rank fits the control panel dimensions")
    if config.fixed_rank is not None:
        if config.fixed_rank > max_control_rank:
            raise ValueError("fixed_rank is not identified by the observed control panel")
        if config.fixed_rank > max_target_rank:
            raise ValueError(
                "fixed_rank requires more observed target pre-periods than available"
            )
        selected_rank = config.fixed_rank
        means: dict[int, float] = {}
        standard_errors: dict[int, float] = {}
    else:
        if cv_folds is None:
            cv_folds = make_rolling_cv_folds(
                control_observed.detach().cpu().numpy(),
                folds=config.folds,
                proportion=config.cv_prop,
                min_pre_periods=config.min_pre_periods,
                holdout_periods=config.cv_nobs,
                buffer_periods=config.cv_buffer,
                seed=config.seed,
            )
            cv_contract = "python_rolling"
        else:
            cv_contract = "persisted_rolling"
        if len(cv_folds) != config.folds:
            raise ValueError("GSC CV contract fold count does not match config.folds")
        scores: dict[int, list[float]] = {rank: [] for rank in valid_ranks}
        control_y = y[:, controls]
        use_batch_cv = (
            config.batch_cv
            if config.batch_cv is not None
            else control_y.numel() <= 250_000
        )
        score_sse: dict[int, float] = {rank: 0.0 for rank in valid_ranks}
        score_count: dict[int, int] = {rank: 0 for rank in valid_ranks}
        cv_tolerance = max(config.tol, 1e-3)
        for fold in cv_folds:
            if fold.removed.shape != tuple(control_observed.shape):
                raise ValueError("GSC CV contract shape does not match the control panel")
            removed = runtime.tensor(fold.removed, dtype=runtime.torch.bool)
            scored = runtime.tensor(fold.scored, dtype=runtime.torch.bool)
            if bool((removed & ~control_observed).any()):
                raise ValueError("GSC CV contract removes an unobserved control cell")
            training = control_observed & ~removed
            if use_batch_cv:
                fits = fit_interactive_fixed_effects_batched(
                    control_y,
                    training,
                    ranks=valid_ranks,
                    force="two-way",
                    max_iter=config.max_iter,
                    tol=cv_tolerance,
                )
            else:
                fits = [
                    fit_interactive_fixed_effects(
                        control_y,
                        training,
                        rank=rank,
                        force="two-way",
                        max_iter=config.max_iter,
                        tol=cv_tolerance,
                    )
                    for rank in valid_ranks
                ]
            for rank, fit in zip(valid_ranks, fits, strict=True):
                if not fit.converged:
                    raise RuntimeError(f"GSC CV failed to converge for rank {rank}")
                errors = control_y[scored] - fit.fitted[scored]
                squared = errors.square()
                scores[rank].append(float(squared.mean().detach().cpu()))
                score_sse[rank] += float(squared.sum().detach().cpu())
                score_count[rank] += int(errors.numel())
        # fect stores pooled-cell MSPE in CV.out and per-fold SE separately.
        means = {rank: score_sse[rank] / score_count[rank] for rank in valid_ranks}
        standard_errors = {
            rank: float(np.std(values, ddof=1) / np.sqrt(len(values)))
            for rank, values in scores.items()
        }
        selected_rank = _select_one_standard_error(means, standard_errors, config.cv_rule)
    final_fit = _fit_controls(y, fit_mask, controls, selected_rank, config)
    counterfactual = _target_counterfactual(
        y,
        observed,
        treated,
        target_index,
        final_fit,
        selected_rank,
    )
    bootstrap_draws = None
    bootstrap_se = None
    inference = None
    resolved_inference_batch_size = config.inference_batch_size
    if config.bootstrap_mode != "none":
        bootstrap_draws, resolved_inference_batch_size = _bootstrap_effects(
            y,
            observed,
            treated,
            target_index,
            controls,
            selected_rank,
            final_fit,
            config,
        )
        bootstrap_draws *= float(scale)
    counterfactual_np = counterfactual.detach().cpu().numpy() * float(scale)
    effect = panel.y[:, target_index] - counterfactual_np
    if bootstrap_draws is not None:
        assert panel.observed is not None and panel.treated is not None
        resolved_mode = (
            "reference_empirical"
            if config.bootstrap_mode == "auto"
            and bool(panel.observed.all())
            else "reference_ar"
            if config.bootstrap_mode == "auto"
            else config.bootstrap_mode
        )
        inference = normal_inference_from_replicates(
            effect,
            bootstrap_draws,
            method=f"gsc_parametric_{resolved_mode}",
            requested_repetitions=config.n_bootstrap,
        )
        bootstrap_se = inference.standard_error
        minimum = max(2, int(np.ceil(config.n_bootstrap * config.minimum_valid_fraction)))
        treated_periods = panel.treated[:, target_index]
        if np.any(inference.valid_repetitions[treated_periods] < minimum):
            raise RuntimeError("too few converged GSC bootstrap repetitions")
    notes = (
        "rank CV and target refit are implemented on the selected torch device",
        "bootstrap stochastic semantics require R parity before formal use",
    )
    provenance = EstimatorProvenance(
        estimator="gsc",
        backend="pytorch",
        formal_eligible=False,
        numerical_policy=runtime.metadata()
        | {
            "cv_rule": config.cv_rule,
            "cv_method": "rolling",
            "cv_contract": "not_used" if config.fixed_rank is not None else cv_contract,
            "cv_nobs": config.cv_nobs,
            "cv_buffer": config.cv_buffer,
            "cv_tolerance": max(config.tol, 1e-3),
            "tuning_source": "fixed" if config.fixed_rank is not None else "gpu_cv",
            "bootstrap_mode": config.bootstrap_mode,
            "bootstrap_mode_resolved": (
                "none"
                if config.bootstrap_mode == "none"
                else inference.method.removeprefix("gsc_parametric_")
                if inference is not None
                else "unavailable"
            ),
            "n_bootstrap": config.n_bootstrap,
            "minimum_valid_fraction": config.minimum_valid_fraction,
            "batch_cv": "not_used" if config.fixed_rank is not None else use_batch_cv,
            "inference_batch_size": config.inference_batch_size,
            "resolved_inference_batch_size": resolved_inference_batch_size,
        },
        notes=notes,
    )
    return GSCResult(
        selected_rank=selected_rank,
        counterfactual=counterfactual_np,
        effect=effect,
        cv_mean_mspe=means,
        cv_se_mspe=standard_errors,
        bootstrap_se=bootstrap_se,
        bootstrap_draws=bootstrap_draws,
        inference=inference,
        treatment_start=treatment_start,
        converged=final_fit.converged,
        iterations=final_fit.iterations,
        provenance=provenance,
    )
