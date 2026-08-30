"""R-free formal execution and publication for GSC and matrix completion."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from urban_intervention.utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
)

from .contracts import FORMAL_IMPLEMENTATION_VERSION, FORMAL_RESULT_SCHEMA, PanelData
from .gsc import GSCConfig, GSCResult, _target_counterfactual, fit_gsc
from .inference import jackknife_standard_error
from .io import LoadedPanel, load_estimation_panel
from .linalg import as_panel_tensors, fit_interactive_fixed_effects
from .matrix_completion import (
    MatrixCompletionConfig,
    MatrixCompletionResult,
    fit_matrix_completion,
)
from .provenance import (
    estimator_code_fingerprint,
    file_sha256,
    python_environment,
    qualified_environment_differences,
)
from .qualification import validate_formal_qualification_receipt
from .runtime import RuntimeConfig, TorchRuntime

Estimator = Literal["gsc", "mc"]
RunMode = Literal["production", "preview"]


@dataclass(frozen=True, slots=True)
class FormalRunRequest:
    estimator: Estimator
    output_directory: Path
    treatment_order: int
    city_key: str
    grid_id: str
    opening_month: str
    outcome_family: str
    outcome: str
    donor_scope: str = "same_city"
    run_mode: RunMode = "production"
    run_id: str = ""
    specification_fingerprint: str = ""
    price_measure: str = "median"
    observation_window: int = 1
    transaction_count_threshold: int = 1
    device: str = "auto"
    seed: int | None = None
    qualification_receipt: Path | None = None
    qualification_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.estimator not in {"gsc", "mc"}:
            raise ValueError("estimator must be 'gsc' or 'mc'")
        if self.run_mode not in {"production", "preview"}:
            raise ValueError("run_mode must be production or preview")
        if not 1 <= self.observation_window <= 6:
            raise ValueError("observation_window must be in 1..6")
        if self.transaction_count_threshold < 1:
            raise ValueError("transaction_count_threshold must be positive")


@dataclass(frozen=True, slots=True)
class FormalRunResult:
    labels: pd.DataFrame
    manifest: dict[str, Any]
    labels_path: Path
    manifest_path: Path


def _normal_p_value(effect: np.ndarray, standard_error: np.ndarray) -> np.ndarray:
    result = np.full(effect.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(effect) & np.isfinite(standard_error) & (standard_error > 0)
    result[valid] = [
        math.erfc(abs(float(value)) / math.sqrt(2.0))
        for value in effect[valid] / standard_error[valid]
    ]
    return result


def _panel_content_signature(frame: pd.DataFrame) -> str:
    """Hash the exact normalized estimator input independently of Parquet bytes."""
    sort_columns = [
        value
        for value in ("city_key", "grid_id", "gsc_unit_id", "mc_unit_id", "time_id")
        if value in frame.columns
    ]
    canonical = frame.sort_values(sort_columns, kind="stable") if sort_columns else frame
    canonical = canonical.reindex(sorted(canonical.columns), axis=1).reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(canonical, index=False, categorize=True)
    digest = hashlib.sha256()
    digest.update(FORMAL_IMPLEMENTATION_VERSION.encode("ascii"))
    digest.update("\x1f".join(canonical.columns).encode("utf-8"))
    digest.update("\x1f".join(map(str, canonical.dtypes)).encode("utf-8"))
    digest.update(np.asarray(row_hashes, dtype=np.uint64).tobytes())
    return digest.hexdigest()


def _event_time(
    periods: tuple[object, ...], opening_period: object, frequency: str
) -> np.ndarray:
    if frequency == "annual":
        return np.asarray(periods, dtype=np.int64) - int(str(opening_period)[:4])
    opening = pd.Timestamp(opening_period).to_period("M")
    return np.asarray(
        [
            (pd.Timestamp(period).to_period("M").year - opening.year) * 12
            + pd.Timestamp(period).to_period("M").month
            - opening.month
            for period in periods
        ],
        dtype=np.int64,
    )


def _resolve_panel_metadata(
    frame: pd.DataFrame,
    loaded: LoadedPanel,
    request: FormalRunRequest,
    panel_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(panel_metadata or {})
    metadata.setdefault(
        "frequency", "monthly" if request.outcome_family in {"housing", "viirs"} else "annual"
    )
    metadata.setdefault("opening_period_excluded", request.opening_month)
    metadata.setdefault("target_effect_scale_to_original_units", 1.0)
    metadata.setdefault("target_center_to_original_units", 0.0)
    metadata.setdefault("clean_pre_periods", loaded.panel.treatment_start())
    metadata.setdefault("post_periods", len(loaded.periods) - loaded.panel.treatment_start())
    unit_column = "gsc_unit_id" if request.estimator == "gsc" else "mc_unit_id"
    target_id = loaded.numeric_unit_ids[loaded.panel.single_treated_unit()]
    if "role" in frame:
        target_rows = frame.loc[frame[unit_column].eq(target_id) & frame["role"].eq("treated")]
    else:
        target_rows = frame.loc[frame[unit_column].eq(target_id)]
    metadata.setdefault("donors_used", int(frame[unit_column].nunique() - 1))
    if not target_rows.empty:
        metadata.setdefault("city_key", str(target_rows.iloc[0].get("city_key", request.city_key)))
        metadata.setdefault("grid_id", str(target_rows.iloc[0].get("grid_id", request.grid_id)))
    return metadata


def _target_observed_path(
    frame: pd.DataFrame,
    loaded: LoadedPanel,
    estimator: Estimator,
    *,
    scale: float,
    center: float,
) -> np.ndarray:
    unit_column = "gsc_unit_id" if estimator == "gsc" else "mc_unit_id"
    target_id = loaded.numeric_unit_ids[loaded.panel.single_treated_unit()]
    rows = frame.loc[frame[unit_column].eq(target_id)].sort_values("time_id")
    if len(rows) != len(loaded.periods):
        raise ValueError("panel does not expose one target row per period")
    if "value" in rows:
        return pd.to_numeric(rows["value"], errors="coerce").to_numpy(dtype=np.float64)
    model = pd.to_numeric(rows["model_value"], errors="coerce").to_numpy(dtype=np.float64)
    return model * scale + center


def _target_optional_path(
    frame: pd.DataFrame, loaded: LoadedPanel, estimator: Estimator, column: str
) -> np.ndarray:
    if column not in frame:
        return np.full(len(loaded.periods), np.nan, dtype=np.float64)
    unit_column = "gsc_unit_id" if estimator == "gsc" else "mc_unit_id"
    target_id = loaded.numeric_unit_ids[loaded.panel.single_treated_unit()]
    rows = frame.loc[frame[unit_column].eq(target_id)].sort_values("time_id")
    return pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=np.float64)


def _apply_observation_window(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    if window == 1 or not frame["event_time"].gt(0).any():
        return frame
    pre = frame.loc[frame["event_time"].lt(0)].copy()
    post = frame.loc[frame["event_time"].gt(0)].copy()
    rows: list[pd.Series] = []
    for horizon in sorted(post["event_time"].astype(int).unique()):
        start = max(1, horizon - window + 1)
        part = post.loc[post["event_time"].between(start, horizon)]
        result = part.loc[part["event_time"].idxmax()].copy()
        minimum = min(window, horizon)
        finite_observed = np.isfinite(part["observed"])
        finite_counterfactual = np.isfinite(part["counterfactual"])
        supported = (
            len(part) == minimum
            and int(finite_observed.sum()) == minimum
            and int(finite_counterfactual.sum()) == minimum
        )
        result["observed"] = part.loc[finite_observed, "observed"].mean()
        result["counterfactual"] = part.loc[finite_counterfactual, "counterfactual"].mean()
        result["causal_response_label"] = (
            result["observed"] - result["counterfactual"] if supported else np.nan
        )
        result["minimum_window_n"] = minimum
        result["effective_n_observed"] = int(finite_observed.sum())
        result["effective_n_counterfactual"] = int(finite_counterfactual.sum())
        result["window_supported"] = supported
        result["label_available"] = bool(
            supported and np.isfinite(result["causal_response_label"])
        )
        if "transaction_count" in part:
            counts = pd.to_numeric(part["transaction_count"], errors="coerce")
            result["transaction_count"] = counts.sum(min_count=1)
            result["transaction_count_min"] = counts.min(skipna=False)
            result["transaction_count_supported"] = bool(
                len(part) == minimum
                and part["transaction_count_supported"].fillna(False).all()
            )
        # Marginal monthly standard errors do not identify the covariance of a
        # moving-window mean.  The formal runner replaces these values using
        # the estimator's joint replicate paths below; fail closed otherwise.
        result[
            [
                "standard_error",
                "confidence_lower",
                "confidence_upper",
                "p_value",
                "valid_inference_repetitions",
            ]
        ] = np.nan
        result["uncertainty_source"] = f"{result['uncertainty_source']}_window{window}"
        rows.append(result)
    return pd.concat([pre, pd.DataFrame(rows)], ignore_index=True)


def _apply_window_replicate_inference(
    labels: pd.DataFrame,
    *,
    raw_event_time: np.ndarray,
    replicate_estimates: np.ndarray,
    estimator: Estimator,
    window: int,
    requested_repetitions: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Aggregate joint replicate paths and rebuild window uncertainty.

    GSC replicate rows are bootstrap effect paths. MC rows are leave-one-unit-
    out paths, so their window standard error is rebuilt from jackknife
    pseudo-values rather than from leave-one-out path dispersion.
    """
    draws = np.asarray(replicate_estimates, dtype=np.float64)
    event_time = np.asarray(raw_event_time, dtype=np.int64)
    if draws.ndim != 2 or draws.shape[1] != event_time.size:
        raise ValueError("replicate paths must align with the unwindowed event-time path")
    if draws.shape[0] != requested_repetitions:
        raise ValueError("replicate row count must equal requested repetitions")

    output = labels.copy()
    aggregated = np.full((draws.shape[0], len(output)), np.nan, dtype=np.float64)
    for column, (index, row) in enumerate(output.iterrows()):
        horizon = int(row["event_time"])
        if horizon < 0 or window == 1:
            indices = np.flatnonzero(event_time == horizon)
        else:
            start = max(1, horizon - window + 1)
            indices = np.flatnonzero((event_time >= start) & (event_time <= horizon))
        if indices.size == 0:
            continue
        part = draws[:, indices]
        complete = np.isfinite(part).all(axis=1)
        aggregated[complete, column] = part[complete].mean(axis=1)

        point = float(row["causal_response_label"])
        valid = int(complete.sum())
        standard_error = np.nan
        if valid >= 2 and np.isfinite(point):
            if estimator == "gsc":
                standard_error = float(np.std(aggregated[complete, column], ddof=1))
            else:
                standard_error = float(
                    jackknife_standard_error(
                        np.asarray([point]), aggregated[:, [column]]
                    )[0][0]
                )
        output.at[index, "valid_inference_repetitions"] = valid
        output.at[index, "standard_error"] = standard_error
        if np.isfinite(standard_error):
            output.at[index, "confidence_lower"] = point - 1.96 * standard_error
            output.at[index, "confidence_upper"] = point + 1.96 * standard_error
            output.at[index, "p_value"] = _normal_p_value(
                np.asarray([point]), np.asarray([standard_error])
            )[0]
        else:
            output.at[index, "confidence_lower"] = np.nan
            output.at[index, "confidence_upper"] = np.nan
            output.at[index, "p_value"] = np.nan
    return output, aggregated


def _validate_formal_post_inference(
    labels: pd.DataFrame,
    *,
    requested_repetitions: int,
    minimum_valid_fraction: float,
) -> None:
    """Require adequate inference for each publishable post-treatment label."""
    minimum_valid = max(
        2, int(np.ceil(requested_repetitions * minimum_valid_fraction))
    )
    publishable_post = labels["event_time"].gt(0) & labels[
        "label_available"
    ].fillna(False)
    post_valid = pd.to_numeric(
        labels.loc[publishable_post, "valid_inference_repetitions"],
        errors="coerce",
    )
    if post_valid.empty:
        raise RuntimeError("formal result has no publishable post-treatment labels")
    if bool(post_valid.lt(minimum_valid).any()):
        raise RuntimeError("too few complete replicate paths for formal window inference")


def _fit(
    loaded: LoadedPanel,
    request: FormalRunRequest,
    runtime: TorchRuntime,
    gsc_config: GSCConfig | None,
    mc_config: MatrixCompletionConfig | None,
) -> tuple[
    GSCResult | MatrixCompletionResult,
    dict[Any, float],
    pd.DataFrame | None,
]:
    if request.estimator == "gsc":
        resolved_gsc_config = gsc_config or GSCConfig(
            bootstrap_mode="auto" if request.run_mode == "production" else "none",
            n_bootstrap=200 if request.run_mode == "production" else 0,
            seed=request.seed or 20260723,
        )
        if (
            request.donor_scope == "all_city_standardized"
            and request.run_mode == "production"
        ):
            selection_config = replace(
                resolved_gsc_config, bootstrap_mode="none", n_bootstrap=0
            )
            if resolved_gsc_config.fixed_rank is None:
                selection = fit_gsc(
                    loaded.panel, config=selection_config, runtime=runtime
                )
                selected_rank = selection.selected_rank
                cv_mean_mspe: dict[Any, float] = dict(selection.cv_mean_mspe)
            else:
                selected_rank = resolved_gsc_config.fixed_rank
                cv_mean_mspe = {}
            placebo = _cross_city_masked_placebo(
                loaded, selection_config, runtime, request
            )
            if not bool(placebo.loc[placebo["placebo_role"].eq("target"), "target_accepted"].iloc[0]):
                target_rmspe = float(
                    placebo.loc[
                        placebo["placebo_role"].eq("target"), "masked_rmspe"
                    ].iloc[0]
                )
                threshold = float(placebo["donor_placebo_q95"].iloc[0])
                raise RuntimeError(
                    "cross-city masked-placebo gate failed: "
                    f"target RMSPE {target_rmspe} > donor q95 {threshold}"
                )
            final = fit_gsc(
                loaded.panel,
                config=replace(resolved_gsc_config, fixed_rank=selected_rank),
                runtime=runtime,
            )
            return final, cv_mean_mspe, placebo
        gsc_result = fit_gsc(
            loaded.panel, config=resolved_gsc_config, runtime=runtime
        )
        return gsc_result, dict(gsc_result.cv_mean_mspe), None
    resolved_mc_config = mc_config or MatrixCompletionConfig(
        inference="jackknife" if request.run_mode == "production" else "none",
        seed=request.seed or 20260725,
    )
    mc_result = fit_matrix_completion(
        loaded.panel, config=resolved_mc_config, runtime=runtime
    )
    return mc_result, dict(mc_result.cv_mean_mspe), None


def _validate_production_estimator_config(
    estimator: Estimator,
    gsc_config: GSCConfig | None,
    mc_config: MatrixCompletionConfig | None,
) -> None:
    """Prevent callers from weakening the qualified production contract."""
    if estimator == "gsc":
        config = gsc_config or GSCConfig(bootstrap_mode="auto", n_bootstrap=200)
        if (
            config.bootstrap_mode != "auto"
            or config.n_bootstrap < 200
            or config.max_iter != 5000
            or not math.isclose(config.tol, 1e-5, rel_tol=0.0, abs_tol=1e-15)
            or config.minimum_valid_fraction < 0.9
        ):
            raise ValueError(
                "production GSC requires auto bootstrap, at least 200 draws, "
                "max_iter=5000, tol=1e-5 and minimum_valid_fraction>=0.9"
            )
        return
    config = mc_config or MatrixCompletionConfig(inference="jackknife")
    if (
        config.inference != "jackknife"
        or not config.batch_inference
        or config.max_iter != 5000
        or not math.isclose(config.tol, 1e-5, rel_tol=0.0, abs_tol=1e-15)
        or config.minimum_valid_fraction < 0.9
    ):
        raise ValueError(
            "production MC requires batched unit jackknife, max_iter=5000, "
            "tol=1e-5 and minimum_valid_fraction>=0.9"
        )


def _cross_city_masked_placebo(
    loaded: LoadedPanel,
    config: GSCConfig,
    runtime: TorchRuntime,
    request: FormalRunRequest,
) -> pd.DataFrame:
    """Run the pre-only target-versus-20-donor masked prediction gate."""
    panel = loaded.panel
    target = panel.single_treated_unit()
    pre_count = panel.treatment_start()
    requested_holdout = 12 if request.outcome_family in {"housing", "viirs"} else 1
    holdout = min(requested_holdout, pre_count - config.min_pre_periods)
    if holdout < 1:
        raise ValueError("cross-city masked placebo lacks enough clean pre-periods")
    train_end = pre_count - holdout
    donors = np.asarray(
        [index for index in range(panel.y.shape[1]) if index != target], dtype=np.int64
    )
    if donors.size < 20:
        raise ValueError("cross-city masked placebo requires at least 20 donors")
    rng = np.random.default_rng(20260723)
    placebo_donors = np.sort(rng.choice(donors, size=20, replace=False))
    pseudo = np.concatenate([[target], placebo_donors])
    controls = np.asarray(
        [index for index in range(panel.y.shape[1]) if index not in set(pseudo)],
        dtype=np.int64,
    )
    y, observed, _ = as_panel_tensors(panel, runtime)
    y = y[:pre_count]
    observed = observed[:pre_count]
    rank_units = np.concatenate([[target], controls])
    rank_unit_tensor = runtime.tensor(rank_units, dtype=runtime.torch.long)
    rank_y = y[:, rank_unit_tensor]
    rank_observed = observed[:, rank_unit_tensor]
    rank_values = rank_y.detach().cpu().numpy().copy()
    rank_values[~rank_observed.detach().cpu().numpy()] = np.nan
    rank_treated = np.zeros(rank_values.shape, dtype=bool)
    rank_treated[train_end:, 0] = True
    masked_max_rank = max(0, (train_end - 1) // 2)
    masked_ranks = tuple(rank for rank in config.ranks if rank <= masked_max_rank)
    if not masked_ranks:
        raise ValueError("cross-city masked placebo has no identifiable rank candidate")
    masked_cv_nobs = min(
        config.cv_nobs, max(1, train_end - config.min_pre_periods)
    )
    rank_result = fit_gsc(
        PanelData(y=rank_values, treated=rank_treated),
        config=replace(
            config,
            ranks=masked_ranks,
            fixed_rank=None,
            cv_nobs=masked_cv_nobs,
            bootstrap_mode="none",
            n_bootstrap=0,
        ),
        runtime=runtime,
    )
    rank = rank_result.selected_rank
    control_indices = runtime.tensor(controls, dtype=runtime.torch.long)
    control_fit = fit_interactive_fixed_effects(
        y[:, control_indices],
        observed[:, control_indices],
        rank=rank,
        force="two-way",
        max_iter=config.max_iter,
        tol=config.tol,
    )
    if not control_fit.converged:
        raise RuntimeError("cross-city masked-placebo control fit did not converge")
    rows: list[dict[str, object]] = []
    for unit in pseudo:
        treated = observed.new_zeros(observed.shape)
        treated[train_end:, unit] = True
        counterfactual = _target_counterfactual(
            y, observed, treated, int(unit), control_fit, rank
        )
        errors = y[train_end:, unit] - counterfactual[train_end:]
        rmspe = float(errors.square().mean().sqrt().detach().cpu())
        rows.append(
            {
                "unit_id": loaded.numeric_unit_ids[int(unit)],
                "placebo_role": "target" if unit == target else "donor_placebo",
                "masked_periods": holdout,
                "masked_rmspe": rmspe,
                "masked_selected_rank": rank,
            }
        )
    result = pd.DataFrame(rows)
    threshold = float(
        np.quantile(
            result.loc[result["placebo_role"].eq("donor_placebo"), "masked_rmspe"],
            0.95,
            method="median_unbiased",
        )
    )
    target_rmspe = float(
        result.loc[result["placebo_role"].eq("target"), "masked_rmspe"].iloc[0]
    )
    result["donor_placebo_q95"] = threshold
    result["target_accepted"] = np.isfinite(target_rmspe) and target_rmspe <= threshold
    return result


def run_formal_panel(
    panel: str | Path | pd.DataFrame,
    request: FormalRunRequest,
    *,
    panel_metadata: dict[str, Any] | None = None,
    gsc_config: GSCConfig | None = None,
    mc_config: MatrixCompletionConfig | None = None,
) -> FormalRunResult:
    """Fit, validate and atomically publish one Python formal panel result."""
    code_fingerprint = estimator_code_fingerprint(request.estimator)
    qualification: dict[str, Any] = {}
    if request.run_mode == "production":
        if request.qualification_receipt is None:
            raise ValueError(
                "production Python estimation requires a formal qualification receipt"
            )
        qualification = validate_formal_qualification_receipt(
            request.qualification_receipt,
            verify_bound_sources=not bool(request.qualification_receipt_sha256),
            expected_sha256=request.qualification_receipt_sha256 or None,
        )
        _validate_production_estimator_config(
            request.estimator, gsc_config, mc_config
        )
    frame = panel.copy() if isinstance(panel, pd.DataFrame) else pd.read_parquet(panel)
    panel_signature = _panel_content_signature(frame)
    loaded = load_estimation_panel(frame, request.estimator)
    metadata = _resolve_panel_metadata(frame, loaded, request, panel_metadata)
    seed = request.seed or (20260723 if request.estimator == "gsc" else 20260725)
    runtime = TorchRuntime(RuntimeConfig(device=request.device, seed=seed))
    if qualification:
        differences = qualified_environment_differences(
            qualification.get("formal_qualification_environment"),
            python_environment(),
            runtime.metadata(),
        )
        if differences:
            raise RuntimeError(
                "production runtime differs from the qualified numerical environment: "
                + "; ".join(differences)
            )
    fit, cv_mean_mspe, masked_placebo = _fit(
        loaded, request, runtime, gsc_config, mc_config
    )
    if not fit.converged:
        raise RuntimeError(f"{request.estimator.upper()} final fit did not converge")
    if request.run_mode == "production" and fit.inference is None:
        raise RuntimeError("production formal results require complete estimator inference")

    scale = float(metadata["target_effect_scale_to_original_units"])
    center = float(metadata["target_center_to_original_units"])
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(center):
        raise ValueError("invalid target inverse-scaling parameters")
    observed = _target_observed_path(
        frame, loaded, request.estimator, scale=scale, center=center
    )
    transaction_count = _target_optional_path(
        frame, loaded, request.estimator, "transaction_count"
    )
    counterfactual = np.asarray(fit.counterfactual, dtype=np.float64) * scale + center
    effect = observed - counterfactual
    event_time = _event_time(
        loaded.periods, metadata["opening_period_excluded"], str(metadata["frequency"])
    )
    inference = fit.inference
    if inference is None:
        standard_error = confidence_lower = confidence_upper = p_value = np.full(
            effect.shape, np.nan
        )
        valid_repetitions = np.zeros(effect.shape, dtype=np.int64)
        requested_repetitions = 0
        uncertainty_source = "preview_point_estimate"
    else:
        standard_error = np.asarray(inference.standard_error, dtype=np.float64) * scale
        confidence_lower = effect - 1.96 * standard_error
        confidence_upper = effect + 1.96 * standard_error
        p_value = _normal_p_value(effect, standard_error)
        valid_repetitions = np.asarray(inference.valid_repetitions, dtype=np.int64)
        requested_repetitions = int(inference.requested_repetitions)
        uncertainty_source = inference.method
    labels = pd.DataFrame(
        {
            "treatment_order": request.treatment_order,
            "city_key": request.city_key,
            "grid_id": request.grid_id,
            "outcome_family": request.outcome_family,
            "outcome": request.outcome,
            "period": loaded.periods,
            "event_time": event_time,
            "observed": observed,
            "counterfactual": counterfactual,
            "causal_response_label": effect,
            "label_available": np.isfinite(effect),
            "method": "xu_2017_gsynth" if request.estimator == "gsc" else "athey_2021_mc",
            "standard_error": standard_error,
            "confidence_lower": confidence_lower,
            "confidence_upper": confidence_upper,
            "p_value": p_value,
            "bootstrap_repetitions": requested_repetitions,
            "valid_inference_repetitions": valid_repetitions,
            "uncertainty_source": uncertainty_source,
            "donor_scope": request.donor_scope,
            "estimator_backend": "python_pytorch",
            "implementation_version": FORMAL_IMPLEMENTATION_VERSION,
            "code_fingerprint": code_fingerprint,
            "price_measure": request.price_measure,
            "transaction_count": transaction_count,
            "transaction_count_min": transaction_count,
            "transaction_count_threshold": (
                request.transaction_count_threshold
                if request.outcome_family == "housing"
                else np.nan
            ),
            "transaction_count_supported": (
                transaction_count >= request.transaction_count_threshold
                if request.outcome_family == "housing"
                else np.full(effect.shape, False)
            ),
        }
    )
    labels = _apply_observation_window(labels, request.observation_window)
    if request.outcome_family == "housing":
        transaction_supported = labels["transaction_count_supported"].fillna(False)
        labels["label_available"] = labels["label_available"].fillna(False) & transaction_supported
        labels.loc[~labels["label_available"], "causal_response_label"] = np.nan
    output_draws: np.ndarray | None = None
    if inference is not None and inference.replicate_estimates is not None:
        raw_draws = np.asarray(inference.replicate_estimates, dtype=np.float64) * scale
        labels, output_draws = _apply_window_replicate_inference(
            labels,
            raw_event_time=event_time,
            replicate_estimates=raw_draws,
            estimator=request.estimator,
            window=request.observation_window,
            requested_repetitions=requested_repetitions,
        )
        if request.run_mode == "production":
            if request.estimator == "gsc":
                valid_population = requested_repetitions
                valid_fraction = (
                    gsc_config.minimum_valid_fraction if gsc_config is not None else 0.9
                )
            else:
                valid_population = requested_repetitions
                valid_fraction = (
                    mc_config.minimum_valid_fraction if mc_config is not None else 0.9
                )
            _validate_formal_post_inference(
                labels,
                requested_repetitions=valid_population,
                minimum_valid_fraction=valid_fraction,
            )
    pre_effect = labels.loc[labels["event_time"].lt(0), "causal_response_label"]
    labels["pre_rmspe"] = float(np.sqrt(np.nanmean(np.square(pre_effect))))
    labels["pre_observed_periods"] = int(
        np.isfinite(labels.loc[labels["event_time"].lt(0), "observed"]).sum()
    )
    pre_rows = labels.loc[
        labels["event_time"].lt(0)
        & np.isfinite(pd.to_numeric(labels["causal_response_label"], errors="coerce")),
        ["event_time", "causal_response_label"],
    ]
    labels["pre_mean_effect"] = (
        float(pre_rows["causal_response_label"].mean()) if not pre_rows.empty else np.nan
    )
    if len(pre_rows) >= 3 and pre_rows["event_time"].nunique() >= 2:
        from scipy.stats import linregress

        trend = linregress(
            pre_rows["event_time"].to_numpy(dtype=np.float64),
            pre_rows["causal_response_label"].to_numpy(dtype=np.float64),
        )
        labels["pretrend_slope"] = float(trend.slope)
        labels["pretrend_slope_p_value"] = float(trend.pvalue)
        labels["pretrend_task_flag"] = (
            "slope_flag" if trend.pvalue < 0.05 else "slope_not_detected"
        )
    else:
        labels["pretrend_slope"] = np.nan
        labels["pretrend_slope_p_value"] = np.nan
        labels["pretrend_task_flag"] = "insufficient_pre_periods"
    if request.estimator == "gsc":
        assert isinstance(fit, GSCResult)
        labels["selected_factors"] = int(fit.selected_rank)
        selected_tuning: float | int = int(fit.selected_rank)
        cv_min_mspe = (
            min(cv_mean_mspe.values())
            if cv_mean_mspe
            else float(metadata.get("cached_cv_min_mspe", np.nan))
        )
    else:
        assert isinstance(fit, MatrixCompletionResult)
        if request.run_mode == "production" and (
            not np.isfinite(fit.selected_lambda) or fit.selected_lambda < 0
        ):
            raise RuntimeError(
                "production matrix completion requires a finite non-negative selected lambda"
            )
        labels["mc_lambda"] = float(fit.selected_lambda)
        labels["mc_regularized"] = bool(fit.selected_lambda > 0)
        cv_min_mspe = (
            min(cv_mean_mspe.values())
            if cv_mean_mspe
            else float(metadata.get("cached_cv_min_mspe", np.nan))
        )
        labels["mc_cv_mspe"] = cv_min_mspe
        selected_tuning = float(fit.selected_lambda)

    output = request.output_directory.resolve()
    labels_path = output / "causal_response_labels.parquet"
    atomic_write_parquet(labels, labels_path)
    diagnostics = pd.DataFrame(
        [
            {
                "clean_pre_periods": metadata["clean_pre_periods"],
                "post_periods": metadata["post_periods"],
                "donors_used": metadata["donors_used"],
                "available_post_labels": int(
                    labels.loc[labels["event_time"].gt(0), "label_available"].sum()
                ),
                "selected_tuning": selected_tuning,
                "cv_min_mspe": cv_min_mspe,
                "converged": fit.converged,
                "iterations": fit.iterations,
            }
        ]
    )
    atomic_write_csv(diagnostics, output / "diagnostics.csv")
    if masked_placebo is not None:
        atomic_write_csv(masked_placebo, output / "cross_city_masked_placebo.csv")
    if output_draws is not None:
        draw_frame = pd.DataFrame(
            output_draws, columns=[str(value) for value in labels["period"]]
        )
        draw_frame.insert(0, "replicate", np.arange(1, len(draw_frame) + 1))
        atomic_write_parquet(draw_frame, output / "inference_draws.parquet")

    run_id = request.run_id or uuid.uuid4().hex
    manifest: dict[str, Any] = {
        "schema": FORMAL_RESULT_SCHEMA,
        "implementation_version": FORMAL_IMPLEMENTATION_VERSION,
        "code_fingerprint": code_fingerprint,
        "run_id": run_id,
        "estimator": request.estimator,
        "backend": "python_pytorch",
        "device": str(runtime.device),
        "dtype": runtime.config.dtype,
        "deterministic": runtime.config.deterministic,
        "run_mode": request.run_mode,
        "production_eligible": (
            request.run_mode == "production"
            and qualification.get("formal_qualification_eligible") is True
        ),
        "formal_validation_status": (
            "qualified_against_r_reference"
            if qualification
            else "preview_not_formally_qualified"
        ),
        **qualification,
        "treatment_order": request.treatment_order,
        "city_key": request.city_key,
        "grid_id": request.grid_id,
        "opening_month": request.opening_month,
        "outcome_family": request.outcome_family,
        "outcome": request.outcome,
        "frequency": metadata["frequency"],
        "specification_fingerprint": request.specification_fingerprint,
        "price_measure": request.price_measure,
        "observation_window": request.observation_window,
        "transaction_count_threshold": request.transaction_count_threshold,
        "donor_scope": request.donor_scope,
        "inference": uncertainty_source,
        "requested_inference_repetitions": requested_repetitions,
        "selected_tuning": selected_tuning,
        "mc_regularized": (
            bool(selected_tuning > 0) if request.estimator == "mc" else None
        ),
        "cv_min_mspe": cv_min_mspe,
        "tuning_source": metadata.get("tuning_source", "fresh_gpu_cv"),
        "tuning_cache_hit": bool(metadata.get("tuning_cache_hit", False)),
        "tuning_panel_signature": metadata.get("tuning_panel_signature", ""),
        "input_panel_signature": panel_signature,
        "labels_sha256": file_sha256(labels_path),
        "request": asdict(request) | {"output_directory": str(request.output_directory)},
        "panel_metadata": metadata,
        "runtime": runtime.metadata(),
        "provenance": {
            "estimator": fit.provenance.estimator,
            "backend": fit.provenance.backend,
            "implementation_version": fit.provenance.implementation_version,
            "code_fingerprint": code_fingerprint,
            "numerical_policy": fit.provenance.numerical_policy,
            "notes": fit.provenance.notes,
        },
    }
    manifest_path = output / "manifest.json"
    atomic_write_json(manifest, manifest_path, default=str)
    manifest_frame = pd.DataFrame(
        {
            "field": manifest.keys(),
            "value": [
                json.dumps(value, ensure_ascii=False, default=str)
                if isinstance(value, (dict, list, tuple))
                else value
                for value in manifest.values()
            ],
        }
    )
    csv_manifest_path = output / "manifest.csv"
    atomic_write_csv(manifest_frame, csv_manifest_path)
    return FormalRunResult(
        labels=labels,
        manifest=manifest,
        labels_path=labels_path,
        manifest_path=csv_manifest_path,
    )
