"""R-free pooled event studies for GSC and matrix-completion effect paths."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from urban_intervention.causal.gpu.inference import jackknife_standard_error
from urban_intervention.causal.gpu.provenance import file_sha256

LOGGER = logging.getLogger(__name__)

DEFAULT_MAIN_SPECIFICATION_FINGERPRINT = (
    r"^main_a6_r1km__a0__w1__price_(median|hedonic)(__.*)?$"
)

GROUP_COLUMNS = [
    "frequency",
    "outcome_family",
    "outcome",
    "method",
    "donor_scope",
    "observation_window",
    "reference_event_time",
]
PATH_COLUMNS = [
    "treatment_order",
    "city_key",
    *GROUP_COLUMNS,
    "event_time",
    "causal_response_label",
    "label_available",
    "standard_error",
    "contrast_inference_source",
    "specification_fingerprint",
    "run_id",
]


@dataclass(frozen=True, slots=True)
class PooledPathEventStudyResult:
    paths: pd.DataFrame
    series: pd.DataFrame
    grid_pretrend: pd.DataFrame
    city_pretrend: pd.DataFrame
    diagnostics: dict[str, Any]


def _read_manifest(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if set(frame.columns) != {"field", "value"} or frame["field"].duplicated().any():
        raise ValueError(f"malformed estimator manifest: {path}")
    return dict(zip(frame["field"], frame["value"], strict=False))


def _manifest_anticipation_months(manifest: Mapping[str, object]) -> int | None:
    """Read anticipation from structured metadata, then the full fingerprint."""
    raw = manifest.get("anticipation_months")
    try:
        if raw not in (None, ""):
            value = int(str(raw))
            if value >= 0:
                return value
    except (TypeError, ValueError):
        pass
    match = re.search(r"__a(\d+)(?:__|$)", str(manifest.get("specification_fingerprint", "")))
    return int(match.group(1)) if match else None


def _canonical_reference_event_time(
    frequency: str, anticipation_months: int | None
) -> int | None:
    """Return the reference implied by a manifest's data frequency.

    Annual outcomes use the prior calendar year regardless of the monthly
    anticipation setting embedded in a shared queue fingerprint.  Monthly
    outcomes use the month immediately before the anticipation buffer.
    """
    if frequency == "annual":
        return -1
    if frequency == "monthly" and anticipation_months is not None:
        return -(anticipation_months + 1)
    return None


def _reference_contrast_standard_errors(
    part: pd.DataFrame,
    draws: np.ndarray,
    *,
    estimator: str,
    reference_event_time: int,
) -> np.ndarray:
    """Compute uncertainty for effect(t) - effect(reference) from joint draws."""
    result = np.full(len(part), np.nan, dtype=np.float64)
    point = pd.to_numeric(part["causal_response_label"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    grouped = part.groupby(["treatment_order", "outcome"], sort=False).indices
    for positions in grouped.values():
        local_positions = np.asarray(positions, dtype=np.int64)
        local_event_time = part.iloc[local_positions]["event_time"].to_numpy(dtype=int)
        reference_positions = np.flatnonzero(local_event_time == reference_event_time)
        if reference_positions.size != 1:
            continue
        local_reference = int(reference_positions[0])
        local_draws = draws[:, local_positions]
        contrast_draws = local_draws - local_draws[:, [local_reference]]
        contrast_point = point[local_positions] - point[local_positions[local_reference]]
        if estimator == "mc":
            standard_error, _ = jackknife_standard_error(
                contrast_point, contrast_draws
            )
        else:
            standard_error = np.full(local_positions.size, np.nan, dtype=np.float64)
            valid = np.sum(np.isfinite(contrast_draws), axis=0)
            for column in np.flatnonzero(valid >= 2):
                standard_error[column] = float(
                    np.nanstd(contrast_draws[:, column], ddof=1)
                )
        result[local_positions] = standard_error
    return result


def read_production_effect_paths(
    staging_root: Path,
    *,
    specification_fingerprint: str | None = None,
    frequency: str | None = None,
    monthly_observation_window: int | None = None,
    monthly_observation_windows: Mapping[str, int] | None = None,
    reference_event_time: int = -1,
) -> pd.DataFrame:
    """Read production GSC/MC paths whose manifests prove the exact labels."""
    if reference_event_time >= 0:
        raise ValueError("reference_event_time must be negative")
    specification_pattern = None
    if specification_fingerprint is not None:
        try:
            specification_pattern = re.compile(specification_fingerprint)
        except re.error as error:
            raise ValueError(
                f"invalid specification_fingerprint regex: {specification_fingerprint!r}"
            ) from error
    if monthly_observation_windows is not None:
        monthly_observation_windows = {
            str(family): int(window)
            for family, window in monthly_observation_windows.items()
        }
        if any(window < 1 or window > 6 for window in monthly_observation_windows.values()):
            raise ValueError("monthly observation windows must be in 1..6")
    parts: list[pd.DataFrame] = []
    rejection_reasons = (
        "missing_manifest",
        "unreadable_manifest",
        "nonproduction",
        "production_ineligible",
        "unsupported_estimator",
        "invalid_manifest_frequency",
        "invalid_observation_window",
        "observation_window_filter_mismatch",
        "frequency_filter_mismatch",
        "specification_filter_mismatch",
        "missing_specification_fingerprint",
        "labels_hash_mismatch",
        "labels_unreadable",
        "missing_required_columns",
        "invalid_event_time",
        "inference_draws_hash_mismatch",
        "inference_draws_unreadable",
        "inference_draws_misaligned",
        "reference_specification_mismatch",
        "reference_period_missing",
    )
    admission = {
        key: 0
        for key in (
            "candidate_files",
            "admitted_files",
            "admitted_rows",
            "rejected_files",
            *rejection_reasons,
        )
    }
    for labels_path in sorted(Path(staging_root).rglob("causal_response_labels.parquet")):
        admission["candidate_files"] += 1
        manifest_path = labels_path.with_name("manifest.csv")
        if not manifest_path.is_file():
            admission["missing_manifest"] += 1
            continue
        try:
            manifest = _read_manifest(manifest_path)
        except (OSError, ValueError):
            admission["unreadable_manifest"] += 1
            continue
        if manifest.get("run_mode", "").lower() != "production":
            admission["nonproduction"] += 1
            continue
        if manifest.get("production_eligible", "").upper() != "TRUE":
            admission["production_ineligible"] += 1
            continue
        estimator = manifest.get("estimator", "")
        if estimator not in {"gsc", "mc"}:
            admission["unsupported_estimator"] += 1
            continue
        manifest_frequency = manifest.get("frequency", "")
        if manifest_frequency not in {"monthly", "annual"}:
            admission["invalid_manifest_frequency"] += 1
            continue
        if frequency is not None and manifest_frequency != frequency:
            admission["frequency_filter_mismatch"] += 1
            continue
        try:
            observation_window = int(manifest.get("observation_window", ""))
        except (TypeError, ValueError):
            admission["invalid_observation_window"] += 1
            continue
        if not 1 <= observation_window <= 6 or (
            manifest_frequency == "annual" and observation_window != 1
        ):
            admission["invalid_observation_window"] += 1
            continue
        if (
            manifest_frequency == "monthly"
            and monthly_observation_window is not None
            and monthly_observation_windows is None
            and observation_window != monthly_observation_window
        ):
            admission["observation_window_filter_mismatch"] += 1
            continue
        manifest_specification = manifest.get("specification_fingerprint", "")
        if not manifest_specification:
            admission["missing_specification_fingerprint"] += 1
            continue
        if (
            specification_pattern is not None
            and specification_pattern.fullmatch(manifest_specification) is None
        ):
            admission["specification_filter_mismatch"] += 1
            continue
        anticipation_months = _manifest_anticipation_months(manifest)
        canonical_reference = _canonical_reference_event_time(
            manifest_frequency, anticipation_months
        )
        if canonical_reference is None or canonical_reference != int(reference_event_time):
            admission["reference_specification_mismatch"] += 1
            continue
        manifest_reference = manifest.get("reference_event_time", "")
        if manifest_reference:
            try:
                if int(manifest_reference) != canonical_reference:
                    raise ValueError
            except (TypeError, ValueError):
                admission["reference_specification_mismatch"] += 1
                continue
        expected_hash = manifest.get("labels_sha256", "")
        if not expected_hash or file_sha256(labels_path) != expected_hash:
            admission["labels_hash_mismatch"] += 1
            continue
        try:
            labels = pd.read_parquet(labels_path)
        except (OSError, ValueError):
            admission["labels_unreadable"] += 1
            continue
        required = {
            "treatment_order",
            "city_key",
            "outcome_family",
            "outcome",
            "event_time",
            "causal_response_label",
            "label_available",
        }
        if not required.issubset(labels.columns):
            admission["missing_required_columns"] += 1
            continue
        event_time = pd.to_numeric(labels["event_time"], errors="coerce")
        if event_time.isna().any() or not np.equal(event_time, np.floor(event_time)).all():
            admission["invalid_event_time"] += 1
            continue
        part = labels.copy().reset_index(drop=True)
        part["event_time"] = event_time.astype(int)
        if manifest_frequency == "monthly" and monthly_observation_windows is not None:
            family_values = set(part["outcome_family"].astype(str))
            family = next(iter(family_values), None)
            expected_window = (
                monthly_observation_windows.get(family) if family is not None else None
            )
            if expected_window != observation_window:
                admission["observation_window_filter_mismatch"] += 1
                continue
        part["frequency"] = manifest_frequency
        part["observation_window"] = observation_window
        part["method"] = (
            part["method"].astype(str)
            if "method" in part
            else pd.Series(estimator, index=part.index, dtype="object")
        )
        part["donor_scope"] = (
            part["donor_scope"].astype(str)
            if "donor_scope" in part
            else pd.Series(manifest.get("donor_scope", ""), index=part.index, dtype="object")
        )
        part["standard_error"] = pd.to_numeric(
            part.get("standard_error", pd.Series(np.nan, index=part.index)),
            errors="coerce",
        )
        contrast_standard_error: np.ndarray | None = None
        contrast_inference_source = "unavailable_missing_joint_draws"
        expected_draws_hash = manifest.get("inference_draws_sha256", "")
        if expected_draws_hash:
            draws_path = labels_path.with_name("inference_draws.parquet")
            if not draws_path.is_file() or file_sha256(draws_path) != expected_draws_hash:
                admission["inference_draws_hash_mismatch"] += 1
                continue
            try:
                draw_frame = pd.read_parquet(draws_path)
            except (OSError, ValueError):
                admission["inference_draws_unreadable"] += 1
                continue
            expected_columns = [str(value) for value in part["period"]]
            draw_columns = [str(value) for value in draw_frame.columns if value != "replicate"]
            if draw_columns != expected_columns:
                admission["inference_draws_misaligned"] += 1
                continue
            draws = draw_frame.drop(columns=["replicate"], errors="ignore").to_numpy(
                dtype=np.float64
            )
            contrast_standard_error = _reference_contrast_standard_errors(
                part,
                draws,
                estimator=estimator,
                reference_event_time=reference_event_time,
            )
            contrast_inference_source = (
                "joint_jackknife_reference_contrast"
                if estimator == "mc"
                else "joint_bootstrap_reference_contrast"
            )
        reference_rows = part.loc[
            part["event_time"].eq(reference_event_time)
            & part["label_available"].fillna(False)
            & np.isfinite(
                pd.to_numeric(part["causal_response_label"], errors="coerce")
            )
        ]
        reference_key = ["treatment_order", "outcome"]
        expected_groups = part[reference_key].drop_duplicates()
        if (
            reference_rows.duplicated(reference_key).any()
            or len(reference_rows) != len(expected_groups)
        ):
            admission["reference_period_missing"] += 1
            continue
        reference_values = reference_rows.set_index(reference_key)[
            "causal_response_label"
        ]
        row_keys = pd.MultiIndex.from_frame(part[reference_key])
        reference_value = pd.to_numeric(
            reference_values.reindex(row_keys).reset_index(drop=True), errors="coerce"
        )
        available = part["label_available"].fillna(False)
        part["causal_response_label"] = (
            pd.to_numeric(part["causal_response_label"], errors="coerce")
            - reference_value.to_numpy(dtype=float)
        )
        part.loc[~available, "causal_response_label"] = np.nan
        # Marginal standard errors do not identify Var(effect_h-effect_ref).
        # Use joint replicate paths when available; historical artifacts that
        # predate draw hashing remain usable for point estimates but expose NA.
        part["standard_error"] = (
            contrast_standard_error
            if contrast_standard_error is not None
            else np.full(len(part), np.nan, dtype=np.float64)
        )
        part.loc[part["event_time"].eq(reference_event_time), "standard_error"] = 0.0
        part["contrast_inference_source"] = contrast_inference_source
        part["reference_event_time"] = int(reference_event_time)
        part["specification_fingerprint"] = manifest_specification
        part["run_id"] = manifest.get("run_id", "")
        parts.append(part[PATH_COLUMNS])
        admission["admitted_files"] += 1
        admission["admitted_rows"] += len(part)
    admission["rejected_files"] = admission["candidate_files"] - admission["admitted_files"]
    if admission["rejected_files"]:
        reason_summary = ", ".join(
            f"{reason}={admission[reason]}"
            for reason in rejection_reasons
            if admission[reason]
        )
        expected_filter_exclusions = (
            admission["frequency_filter_mismatch"]
            + admission["specification_filter_mismatch"]
            + admission["observation_window_filter_mismatch"]
        )
        log_admission = (
            LOGGER.info
            if admission["rejected_files"] == expected_filter_exclusions
            else LOGGER.warning
        )
        log_admission(
            "Pooled effect-path admission rejected %d/%d candidate file(s): %s",
            admission["rejected_files"],
            admission["candidate_files"],
            reason_summary,
        )
    result = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=PATH_COLUMNS)
    )
    result.attrs["admission_counts"] = admission
    return result


def restrict_to_common_event_support(paths: pd.DataFrame) -> pd.DataFrame:
    """Keep every valid event time available for cross-method comparison.

    Anticipation and opening periods remain visible as diagnostics.  The
    pre-trend test excludes them separately, and the plotting layer selects
    the longest interval shared by the methods available for each outcome.
    """
    event_time = pd.to_numeric(paths["event_time"], errors="coerce")
    valid_frequency = paths["frequency"].isin({"monthly", "annual"})
    return paths.loc[valid_frequency & np.isfinite(event_time)].copy()


def aggregate_effect_paths(paths: pd.DataFrame) -> pd.DataFrame:
    """Pool each estimator/scope separately, combining within- and between-grid variance."""
    observed = paths.loc[
        paths["label_available"].fillna(False)
        & np.isfinite(pd.to_numeric(paths["causal_response_label"], errors="coerce"))
    ].copy()
    if observed.empty:
        return pd.DataFrame(
            columns=[
                *GROUP_COLUMNS,
                "event_time",
                "n_grids",
                "n_cities",
                "mean_label",
                "sd_label",
                "se_label",
                "se_label_city_cluster",
                "ci_lower",
                "ci_upper",
                "ci_lower_city_cluster",
                "ci_upper_city_cluster",
                "within_var",
                "between_var",
                "se_available",
                "uncertainty_status",
                "specifications",
            ]
        )
    duplicate_key = ["treatment_order", *GROUP_COLUMNS, "event_time"]
    if observed.duplicated(duplicate_key).any():
        examples = observed.loc[observed.duplicated(duplicate_key, keep=False), duplicate_key]
        raise ValueError(
            "duplicate admitted event-study paths; filter to one final specification: "
            + ", ".join(examples.head(5).astype(str).agg("/".join, axis=1))
        )
    observed["causal_response_label"] = pd.to_numeric(
        observed["causal_response_label"], errors="coerce"
    )
    observed["standard_error"] = pd.to_numeric(observed["standard_error"], errors="coerce")
    group = [*GROUP_COLUMNS, "event_time"]
    rows: list[dict[str, Any]] = []
    for key, part in observed.groupby(group, sort=True, dropna=False):
        values = part["causal_response_label"].to_numpy(dtype=np.float64)
        task_se = part["standard_error"].to_numpy(dtype=np.float64)
        n = len(values)
        between_var = float(np.var(values, ddof=1)) if n > 1 else 0.0
        finite_se = task_se[np.isfinite(task_se)]
        all_task_se_available = finite_se.size == n
        within_var = (
            float(np.mean(np.square(finite_se)) / n)
            if all_task_se_available
            else np.nan
        )
        mean = float(np.mean(values))
        # A missing task-level SE is missing information, not evidence of
        # zero uncertainty.  Keep the pooled point estimate and descriptive
        # between-grid variance, but do not turn a one-grid group into a
        # degenerate [estimate, estimate] confidence interval.
        if all_task_se_available:
            pooled_variance = within_var + between_var / n
            standard_error = float(np.sqrt(max(pooled_variance, 0.0)))
            uncertainty_status = "available"
            ci_lower = mean - 1.96 * standard_error
            ci_upper = mean + 1.96 * standard_error
        else:
            standard_error = np.nan
            uncertainty_status = (
                "missing_task_standard_errors"
                if not finite_se.size
                else "partially_missing_task_standard_errors"
            )
            ci_lower = ci_upper = np.nan
        city_codes = part["city_key"].astype("string").fillna("unknown")
        city_scores = (
            pd.DataFrame({"city": city_codes.to_numpy(), "residual": values - mean})
            .groupby("city", sort=False)["residual"]
            .sum()
            .to_numpy(dtype=np.float64)
        )
        city_count = len(city_scores)
        city_cluster_se = (
            float(
                np.sqrt(
                    (city_count / (city_count - 1))
                    * np.square(city_scores).sum()
                    / (n * n)
                )
            )
            if city_count > 1
            else np.nan
        )
        if city_count > 1 and np.isfinite(city_cluster_se):
            from scipy.stats import t as student_t

            city_critical = float(student_t.ppf(0.975, city_count - 1))
            city_lower = mean - city_critical * city_cluster_se
            city_upper = mean + city_critical * city_cluster_se
        else:
            city_lower = city_upper = np.nan
        rows.append(
            {
                **dict(zip(group, key, strict=True)),
                "n_grids": n,
                "n_cities": city_count,
                "mean_label": mean,
                "sd_label": float(np.std(values, ddof=1)) if n > 1 else 0.0,
                "se_label": standard_error,
                "se_label_city_cluster": city_cluster_se,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ci_lower_city_cluster": city_lower,
                "ci_upper_city_cluster": city_upper,
                "within_var": within_var,
                "between_var": between_var,
                "se_available": int(finite_se.size),
                "uncertainty_status": uncertainty_status,
                "specifications": "|".join(
                    sorted(set(part["specification_fingerprint"].dropna().astype(str)))
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(group).reset_index(drop=True)


def select_pretrend_rows(
    paths: pd.DataFrame,
    *,
    min_pre_event_time: int | None = None,
    latest_n: int = 5,
) -> pd.DataFrame:
    """Select clean pre-period paths without assuming anticipation periods are -1..-6."""
    if min_pre_event_time is not None and min_pre_event_time >= 0:
        raise ValueError("min_pre_event_time must be negative")
    event_time = pd.to_numeric(paths["event_time"], errors="coerce")
    reference = pd.to_numeric(paths["reference_event_time"], errors="coerce")
    clean_pre = event_time.lt(reference)
    pre = paths.loc[
        paths["label_available"].fillna(False)
        & clean_pre
        & np.isfinite(pd.to_numeric(paths["causal_response_label"], errors="coerce"))
    ].copy()
    if min_pre_event_time is not None:
        return pre.loc[pre["event_time"].ge(min_pre_event_time)].copy()
    rank_group = [*GROUP_COLUMNS, "treatment_order"]
    pre["_latest_rank"] = pre.groupby(rank_group, dropna=False)["event_time"].rank(
        method="dense", ascending=False
    )
    return pre.loc[pre["_latest_rank"].le(latest_n)].drop(columns="_latest_rank")


def pretrend_tests(
    paths: pd.DataFrame,
    *,
    cluster: str,
    min_pre_event_time: int | None = None,
    latest_n: int = 5,
) -> pd.DataFrame:
    """One-sample t test on cluster-level means; diagnostics never delete labels."""
    if cluster not in {"grid", "city"}:
        raise ValueError("cluster must be grid or city")
    pre = select_pretrend_rows(
        paths, min_pre_event_time=min_pre_event_time, latest_n=latest_n
    )
    cluster_column = "treatment_order" if cluster == "grid" else "city_key"
    count_column = "n_grids" if cluster == "grid" else "n_cities"
    mean_column = "mean_grid_mean" if cluster == "grid" else "mean_city_mean"
    sd_column = "sd_grid_mean" if cluster == "grid" else "sd_city_mean"
    if pre.empty:
        return pd.DataFrame(
            columns=[
                *GROUP_COLUMNS,
                count_column,
                "n_pre_observations",
                mean_column,
                sd_column,
                "t_statistic",
                "p_value",
                "reject_5pct",
            ]
        )
    cluster_group = [*GROUP_COLUMNS, cluster_column]
    clustered = (
        pre.groupby(cluster_group, dropna=False, sort=True)
        .agg(
            n_pre_observations=("causal_response_label", "size"),
            cluster_mean=("causal_response_label", "mean"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    from scipy.stats import t as student_t

    for key, part in clustered.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        means = part["cluster_mean"].to_numpy(dtype=np.float64)
        n = len(means)
        mean = float(np.mean(means))
        sd = float(np.std(means, ddof=1)) if n > 1 else np.nan
        statistic = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
        p_value = (
            float(2 * student_t.sf(abs(statistic), n - 1))
            if np.isfinite(statistic)
            else np.nan
        )
        rows.append(
            {
                **dict(zip(GROUP_COLUMNS, key, strict=True)),
                count_column: n,
                "n_pre_observations": int(part["n_pre_observations"].sum()),
                mean_column: mean,
                sd_column: sd,
                "t_statistic": statistic,
                "p_value": p_value,
                "reject_5pct": bool(np.isfinite(p_value) and p_value < 0.05),
                "test_type": "cluster_mean_pre_gap_zero",
            }
        )
    return pd.DataFrame(rows)


def attach_pretrend_metadata(
    paths: pd.DataFrame,
    grid_tests: pd.DataFrame,
    city_tests: pd.DataFrame,
) -> pd.DataFrame:
    """Attach non-blocking pooled credibility diagnostics to every effect-path row."""
    result = paths.copy()
    for prefix, tests in (("grid", grid_tests), ("city", city_tests)):
        selected = tests[[*GROUP_COLUMNS, "p_value", "reject_5pct"]].rename(
            columns={
                "p_value": f"{prefix}_pretrend_p_value",
                "reject_5pct": f"{prefix}_pretrend_reject_5pct",
            }
        )
        result = result.merge(selected, on=GROUP_COLUMNS, how="left", validate="many_to_one")
    city_available = np.isfinite(
        pd.to_numeric(result["city_pretrend_p_value"], errors="coerce")
    )
    grid_available = np.isfinite(
        pd.to_numeric(result["grid_pretrend_p_value"], errors="coerce")
    )
    result["pretrend_flag"] = np.select(
        [
            city_available & result["city_pretrend_reject_5pct"].fillna(False),
            city_available,
            grid_available & result["grid_pretrend_reject_5pct"].fillna(False),
            grid_available,
        ],
        ["city_cluster_flag", "city_cluster_pass", "grid_cluster_flag", "grid_cluster_pass"],
        default="insufficient_clusters",
    )
    return result


def _safe_name(values: tuple[Any, ...]) -> str:
    return "__".join(re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)) for value in values)


def write_event_study_figures(series: pd.DataFrame, output_directory: Path) -> list[Path]:
    """Write one robustly scaled mean-and-95%-CI figure per method and outcome."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for key, part in series.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        part = part.sort_values("event_time")
        figure, axis = plt.subplots(figsize=(7.2, 4.3))
        axis.axhline(0, color="0.35", linewidth=0.9)
        axis.axvline(0, color="0.35", linewidth=0.9, linestyle="--")
        city_interval = (
            np.isfinite(part["ci_lower_city_cluster"])
            & np.isfinite(part["ci_upper_city_cluster"])
        )
        lower = part["ci_lower_city_cluster"].where(city_interval, part["ci_lower"])
        upper = part["ci_upper_city_cluster"].where(city_interval, part["ci_upper"])
        axis.errorbar(
            part["event_time"],
            part["mean_label"],
            yerr=np.vstack([part["mean_label"] - lower, upper - part["mean_label"]]),
            fmt="o",
            markersize=3.8,
            linewidth=1.0,
            capsize=2.0,
            color="#2C6E9F",
        )
        consecutive = part["event_time"].diff().ne(1).cumsum()
        for _, piece in part.groupby(consecutive, sort=False):
            if len(piece) > 1:
                axis.plot(
                    piece["event_time"],
                    piece["mean_label"],
                    linewidth=1.2,
                    color="#2C6E9F",
                )
        finite = np.concatenate(
            [
                lower.to_numpy(dtype=float),
                upper.to_numpy(dtype=float),
                np.asarray([0.0]),
            ]
        )
        finite = finite[np.isfinite(finite)]
        if finite.size:
            finite_lower, finite_upper = float(finite.min()), float(finite.max())
            span = max(finite_upper - finite_lower, 1e-9)
            axis.set_ylim(finite_lower - 0.08 * span, finite_upper + 0.08 * span)
        axis.set_xlabel("Event time")
        axis.set_ylabel("Causal response")
        axis.set_title(" / ".join(map(str, key)))
        figure.tight_layout()
        base = figure_directory / _safe_name(key)
        png = base.with_suffix(".png")
        pdf = base.with_suffix(".pdf")
        svg = base.with_suffix(".svg")
        tiff = base.with_suffix(".tiff")
        figure.savefig(png, dpi=300)
        figure.savefig(pdf)
        figure.savefig(svg)
        figure.savefig(tiff, dpi=600)
        plt.close(figure)
        written.extend([png, pdf, svg, tiff])
    return written


def run_pooled_path_event_study(
    staging_root: Path,
    output_directory: Path,
    *,
    specification_fingerprint: str | None = None,
    frequency: str | None = None,
    min_pre_event_time: int | None = None,
    latest_pre_periods: int = 5,
    observation_window: int = 1,
    monthly_observation_windows: Mapping[str, int] | None = None,
    figures: bool = True,
    task_keys: set[tuple[int, str]] | None = None,
    reference_event_time: int = -1,
) -> PooledPathEventStudyResult:
    if not 1 <= observation_window <= 6:
        raise ValueError("observation_window must be between 1 and 6")
    paths = read_production_effect_paths(
        staging_root,
        specification_fingerprint=specification_fingerprint,
        frequency=frequency,
        monthly_observation_window=(
            None if monthly_observation_windows is not None else observation_window
        ),
        monthly_observation_windows=monthly_observation_windows,
        reference_event_time=reference_event_time,
    )
    admission_counts = paths.attrs.get("admission_counts", {})
    if task_keys is not None and not paths.empty:
        keys = pd.MultiIndex.from_arrays(
            [paths["treatment_order"].astype(int), paths["outcome_family"].astype(str)]
        )
        paths = paths.loc[keys.isin(task_keys)].copy()
        paths.attrs["admission_counts"] = admission_counts
    admitted_before_common_support = len(paths)
    paths = restrict_to_common_event_support(paths)
    paths.attrs["admission_counts"] = admission_counts
    if paths.empty:
        count_summary = ", ".join(
            f"{key}={value}" for key, value in admission_counts.items() if value
        ) or "candidate_files=0"
        raise ValueError(
            "no production GSC/MC effect paths were admitted; "
            f"admission counts: {count_summary}"
        )
    series = aggregate_effect_paths(paths)
    grid = pretrend_tests(
        paths,
        cluster="grid",
        min_pre_event_time=min_pre_event_time,
        latest_n=latest_pre_periods,
    )
    city = pretrend_tests(
        paths,
        cluster="city",
        min_pre_event_time=min_pre_event_time,
        latest_n=latest_pre_periods,
    )
    enriched = attach_pretrend_metadata(paths, grid, city)
    output_directory.mkdir(parents=True, exist_ok=True)
    series.to_csv(output_directory / "event_study_series.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(output_directory / "pretrend_grid_cluster.csv", index=False, encoding="utf-8-sig")
    city.to_csv(output_directory / "pretrend_city_cluster.csv", index=False, encoding="utf-8-sig")
    enriched.to_parquet(output_directory / "effect_paths_with_pretrend.parquet", index=False)
    written_figures = write_event_study_figures(series, output_directory) if figures else []
    diagnostics = {
        "effect_path_rows": len(enriched),
        "treatment_orders": int(enriched["treatment_order"].nunique()),
        "cities": int(enriched["city_key"].nunique()),
        "series_rows": len(series),
        "figures": len(written_figures),
        "pretrend_policy": "diagnostic_only_not_an_automatic_label_gate",
        "requested_task_keys": len(task_keys) if task_keys is not None else None,
        "observation_window_monthly": observation_window,
        "observation_windows_monthly": (
            dict(monthly_observation_windows) if monthly_observation_windows is not None else None
        ),
        "reference_event_time": int(reference_event_time),
        "common_support_policy": (
            "all_available_periods; estimates normalized to the declared omitted "
            "reference; t0 is the real opening-period effect; "
            "longest_cross_method_overlap_selected_per_outcome_at_plot_time"
        ),
        "rows_excluded_outside_common_support": admitted_before_common_support - len(paths),
    }
    diagnostics.update(
        {f"admission_{key}": int(value) for key, value in admission_counts.items()}
    )
    pd.DataFrame([diagnostics]).to_csv(
        output_directory / "diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    return PooledPathEventStudyResult(enriched, series, grid, city, diagnostics)
