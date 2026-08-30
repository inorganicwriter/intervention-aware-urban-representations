"""Python two-way fixed-effect event studies with clustered inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from urban_intervention.causal.gpu.panel_builder import (
    read_annual_outcome,
    read_monthly_housing,
    read_monthly_viirs,
)
from urban_intervention.causal.gpu.provenance import file_sha256
from urban_intervention.causal.schemas import (
    CAUSAL_RESPONSE_LABELS_SCHEMA,
    accepts_legacy_version,
)
from urban_intervention.data.paths import (
    CONTROL_DESIGN_QUEUE,
    OUTCOME_FAMILY_QUEUE,
    OUTPUT_CAUSAL_TASKS_DIR,
    PROJECT_ROOT,
)


@dataclass(frozen=True, slots=True)
class EventStudyResult:
    coefficients: pd.DataFrame
    grid_cluster_pretrend: pd.DataFrame
    city_cluster_pretrend: pd.DataFrame
    diagnostics: dict[str, int | float | str]


def write_matching_event_study_figure(
    result: EventStudyResult,
    path: Path,
    *,
    title: str,
) -> None:
    """Render the city-clustered TWFE coefficients with robust y-axis limits."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    coefficients = result.coefficients.sort_values("event_time")
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.axhline(0, color="0.35", linewidth=0.9)
    axis.axvline(0, color="0.35", linewidth=0.9, linestyle="--")
    axis.fill_between(
        coefficients["event_time"],
        coefficients["confidence_lower_city"],
        coefficients["confidence_upper_city"],
        alpha=0.2,
    )
    axis.plot(coefficients["event_time"], coefficients["estimate"], marker="o")
    finite = np.concatenate(
        [
            coefficients["confidence_lower_city"].to_numpy(dtype=float),
            coefficients["confidence_upper_city"].to_numpy(dtype=float),
            np.asarray([0.0]),
        ]
    )
    finite = finite[np.isfinite(finite)]
    if finite.size:
        lower, upper = np.quantile(finite, [0.01, 0.99])
        span = max(float(upper - lower), 1e-9)
        axis.set_ylim(float(lower - 0.12 * span), float(upper + 0.12 * span))
    axis.set_xlabel("Event time")
    axis.set_ylabel("TWFE effect")
    axis.set_title(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _group_demean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.zeros((len(counts), values.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, values)
    return values - sums[inverse] / counts[inverse, None]


def absorb_two_way_fixed_effects(
    values: np.ndarray,
    unit: np.ndarray,
    period: np.ndarray,
    *,
    tolerance: float = 1e-11,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, int]:
    """Residualize columns against two categorical fixed effects."""
    residual = np.asarray(values, dtype=np.float64).copy()
    if residual.ndim != 2 or len(unit) != len(residual) or len(period) != len(residual):
        raise ValueError("fixed-effect arrays must align")
    for iteration in range(1, max_iterations + 1):
        previous = residual.copy()
        residual = _group_demean(residual, unit)
        residual = _group_demean(residual, period)
        change = float(np.max(np.abs(residual - previous)))
        if change <= tolerance:
            return residual, iteration
    raise RuntimeError("two-way fixed-effect absorption did not converge")


def _cluster_covariance(
    design: np.ndarray,
    residual: np.ndarray,
    clusters: np.ndarray,
    *,
    fixed_effects: tuple[np.ndarray, ...] = (),
) -> tuple[np.ndarray, int, int]:
    unique, inverse = np.unique(clusters, return_inverse=True)
    cluster_count = len(unique)
    nobs, parameter_count = design.shape
    effective_parameter_count = parameter_count
    for fixed_effect in fixed_effects:
        fixed_effect = np.asarray(fixed_effect)
        if fixed_effect.shape != clusters.shape:
            raise ValueError("fixed-effect and cluster arrays must align")
        _, fixed_inverse = np.unique(fixed_effect, return_inverse=True)
        nested = True
        for level in range(int(fixed_inverse.max()) + 1):
            if np.unique(clusters[fixed_inverse == level]).size != 1:
                nested = False
                break
        if not nested:
            effective_parameter_count += int(np.unique(fixed_effect).size)
    if cluster_count < 2 or nobs <= effective_parameter_count:
        return (
            np.full((parameter_count, parameter_count), np.nan),
            cluster_count,
            effective_parameter_count,
        )
    scores = np.zeros((cluster_count, parameter_count), dtype=np.float64)
    np.add.at(scores, inverse, design * residual[:, None])
    bread = np.linalg.pinv(design.T @ design, hermitian=True)
    correction = (cluster_count / (cluster_count - 1)) * (
        (nobs - 1) / (nobs - effective_parameter_count)
    )
    covariance = correction * bread @ (scores.T @ scores) @ bread
    return (
        (covariance + covariance.T) / 2,
        cluster_count,
        effective_parameter_count,
    )


def _wald_zero(
    coefficients: np.ndarray,
    covariance: np.ndarray,
    positions: np.ndarray,
    cluster_count: int,
) -> pd.DataFrame:
    if positions.size == 0 or cluster_count < 2:
        return pd.DataFrame(
            [{"statistic": np.nan, "df1": 0, "df2": max(cluster_count - 1, 0), "p_value": np.nan}]
        )
    selected = coefficients[positions]
    selected_covariance = covariance[np.ix_(positions, positions)]
    statistic = float(selected @ np.linalg.pinv(selected_covariance) @ selected / len(positions))
    from scipy.stats import f

    p_value = float(f.sf(statistic, len(positions), cluster_count - 1))
    return pd.DataFrame(
        [
            {
                "statistic": statistic,
                "df1": int(len(positions)),
                "df2": int(cluster_count - 1),
                "p_value": p_value,
            }
        ]
    )


def fit_twfe_event_study(
    panel: pd.DataFrame,
    *,
    reference_event_time: int,
    outcome_column: str = "outcome",
    unit_column: str = "unit",
    period_column: str = "period",
    role_column: str = "role",
    grid_cluster_column: str = "grid_cluster",
    city_cluster_column: str = "city_cluster",
) -> EventStudyResult:
    """Fit one matched-pair TWFE event study and two clustered VCOVs."""
    required = {
        outcome_column,
        unit_column,
        period_column,
        role_column,
        "event_time",
        grid_cluster_column,
        city_cluster_column,
        "treatment_order",
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"event-study panel lacks columns: {sorted(missing)}")
    frame = panel.loc[np.isfinite(pd.to_numeric(panel[outcome_column], errors="coerce"))].copy()
    frame[outcome_column] = pd.to_numeric(frame[outcome_column], errors="raise")
    treated = frame[role_column].eq("treated")
    event_times = sorted(
        set(frame.loc[treated, "event_time"].astype(int)) - {reference_event_time}
    )
    if not event_times:
        raise ValueError("event-study panel has no estimable event-time coefficients")
    design = np.column_stack(
        [
            (treated & frame["event_time"].eq(event_time)).to_numpy(dtype=np.float64)
            for event_time in event_times
        ]
    )
    matrix = np.column_stack([frame[outcome_column].to_numpy(dtype=float), design])
    residualized, absorption_iterations = absorb_two_way_fixed_effects(
        matrix,
        frame[unit_column].astype(str).to_numpy(),
        frame[period_column].astype(str).to_numpy(),
    )
    y = residualized[:, 0]
    x_all = residualized[:, 1:]
    norms = np.sqrt(np.square(x_all).sum(axis=0))
    active = norms > np.sqrt(np.finfo(float).eps)
    if not active.any():
        raise ValueError("all event-time indicators are collinear with fixed effects")
    x = x_all[:, active]
    active_event_times = np.asarray(event_times, dtype=int)[active]
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        raise ValueError("event-study design remains rank deficient after collinearity removal")
    regression_residual = y - x @ coefficients
    fixed_effects = (
        frame[unit_column].astype(str).to_numpy(),
        frame[period_column].astype(str).to_numpy(),
    )
    grid_covariance, grid_clusters, grid_ssc_parameters = _cluster_covariance(
        x,
        regression_residual,
        frame[grid_cluster_column].astype(str).to_numpy(),
        fixed_effects=fixed_effects,
    )
    city_covariance, city_clusters, city_ssc_parameters = _cluster_covariance(
        x,
        regression_residual,
        frame[city_cluster_column].astype(str).to_numpy(),
        fixed_effects=fixed_effects,
    )
    grid_se = np.sqrt(np.maximum(np.diag(grid_covariance), 0))
    city_se = np.sqrt(np.maximum(np.diag(city_covariance), 0))

    def p_values(standard_error: np.ndarray, cluster_count: int) -> np.ndarray:
        result = np.full(len(coefficients), np.nan)
        valid = np.isfinite(standard_error) & (standard_error > 0) & (cluster_count > 1)
        if valid.any():
            from scipy.stats import t

            result[valid] = 2 * t.sf(
                np.abs(coefficients[valid] / standard_error[valid]), cluster_count - 1
            )
        return result

    coefficient_frame = pd.DataFrame(
        {
            "term": [f"event_time::{value}" for value in active_event_times],
            "event_time": active_event_times,
            "estimate": coefficients,
            "standard_error_grid": grid_se,
            "p_value_grid": p_values(grid_se, grid_clusters),
            "confidence_lower_grid": coefficients,
            "confidence_upper_grid": coefficients,
            "standard_error_city": city_se,
            "p_value_city": p_values(city_se, city_clusters),
            "confidence_lower_city": coefficients,
            "confidence_upper_city": coefficients,
        }
    )
    from scipy.stats import t

    grid_critical = t.ppf(0.975, grid_clusters - 1) if grid_clusters > 1 else np.nan
    city_critical = t.ppf(0.975, city_clusters - 1) if city_clusters > 1 else np.nan
    coefficient_frame["confidence_lower_grid"] -= grid_critical * grid_se
    coefficient_frame["confidence_upper_grid"] += grid_critical * grid_se
    coefficient_frame["confidence_lower_city"] -= city_critical * city_se
    coefficient_frame["confidence_upper_city"] += city_critical * city_se
    pre_positions = np.flatnonzero(active_event_times < 0)
    return EventStudyResult(
        coefficients=coefficient_frame,
        grid_cluster_pretrend=_wald_zero(
            coefficients, grid_covariance, pre_positions, grid_clusters
        ),
        city_cluster_pretrend=_wald_zero(
            coefficients, city_covariance, pre_positions, city_clusters
        ),
        diagnostics={
            "nobs": len(frame),
            "units": int(frame[unit_column].nunique()),
            "treated_events": int(frame["treatment_order"].nunique()),
            "grid_clusters": grid_clusters,
            "city_clusters": city_clusters,
            "parameters": len(coefficients),
            "grid_ssc_parameters": grid_ssc_parameters,
            "city_ssc_parameters": city_ssc_parameters,
            "absorption_iterations": absorption_iterations,
            "reference_event_time": reference_event_time,
            "variance": "CRV1",
        },
    )


def _monthly_pair_panel(
    row: pd.Series,
    family: str,
    root: Path,
    min_pre: int,
    max_post: int,
    price_measure: str,
) -> pd.DataFrame:
    opening = pd.Timestamp(str(row["opening_month"]) + "-01")
    periods = pd.date_range(
        opening + pd.DateOffset(months=min_pre),
        opening + pd.DateOffset(months=max_post),
        freq="MS",
    )
    def reader(city: str) -> pd.DataFrame:
        if family == "housing":
            return read_monthly_housing(root, city, price_measure)
        return read_monthly_viirs(root, city, periods)
    outcome = "housing_log_price" if family == "housing" else "viirs_avg_asinh"
    parts: list[pd.DataFrame] = []
    for role, city_column, grid_column in (
        ("treated", "city_key", "grid_id"),
        ("control", "control_city_key", "control_grid_id"),
    ):
        city = str(row[city_column])
        grid = str(row[grid_column])
        part = reader(city)
        part = part.loc[part["grid_id"].eq(grid), ["period", outcome]].rename(
            columns={outcome: "outcome"}
        )
        part["role"] = role
        part["source_city"] = city
        part["source_grid"] = grid
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _annual_pair_panel(
    row: pd.Series, family: str, root: Path, min_pre: int, max_post: int
) -> pd.DataFrame:
    opening_year = int(str(row["opening_month"])[:4])
    outcome = "poi_count_log" if family == "poi" else "population_log"
    parts: list[pd.DataFrame] = []
    for role, city_column, grid_column in (
        ("treated", "city_key", "grid_id"),
        ("control", "control_city_key", "control_grid_id"),
    ):
        city = str(row[city_column])
        grid = str(row[grid_column])
        part = read_annual_outcome(root, city, family)
        part = part.loc[part["grid_id"].eq(grid), ["year", outcome]].rename(
            columns={"year": "period", outcome: "outcome"}
        )
        part = part.loc[part["period"].between(opening_year + min_pre, opening_year + max_post)]
        part["role"] = role
        part["source_city"] = city
        part["source_grid"] = grid
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def build_matching_event_study_panel(
    outcome_family: str,
    *,
    root: Path = PROJECT_ROOT,
    control_queue: Path | pd.DataFrame | None = None,
    family_queue: Path | pd.DataFrame | None = None,
    task_root: Path | None = None,
    donor_scope: str = "same_city",
    min_pre: int | None = None,
    max_post: int | None = None,
) -> tuple[pd.DataFrame, int]:
    """Build the matched pair panel without dropping positive event times."""
    annual = outcome_family in {"poi", "population"}
    if outcome_family not in {"housing", "viirs", "poi", "population"}:
        raise ValueError("unsupported event-study family")
    minimum = (-4 if annual else -42) if min_pre is None else min_pre
    maximum = (3 if annual else 24) if max_post is None else max_post
    reference = -1 if annual else -7
    if donor_scope not in {"same_city", "all_city_standardized"}:
        raise ValueError("unsupported event-study donor scope")
    queue = (
        control_queue.copy()
        if isinstance(control_queue, pd.DataFrame)
        else pd.read_csv(control_queue or (root / CONTROL_DESIGN_QUEUE.relative_to(PROJECT_ROOT)))
    )
    families = (
        family_queue.copy()
        if isinstance(family_queue, pd.DataFrame)
        else pd.read_csv(
            family_queue or (root / OUTCOME_FAMILY_QUEUE.relative_to(PROJECT_ROOT))
        )
    )
    candidates = families.loc[
        families["outcome_family"].eq(outcome_family)
        & families["status"].eq("matched_labelled")
    ]
    task_directory = task_root or (root / OUTPUT_CAUSAL_TASKS_DIR.relative_to(PROJECT_ROOT))
    controls = queue.set_index("treatment_order", drop=False)
    accepted: list[dict[str, object]] = []
    for candidate in candidates.to_dict("records"):
        order = int(candidate["treatment_order"])
        manifest_path = task_directory / f"{order:05d}" / outcome_family / "manifest.json"
        labels_path = manifest_path.parent / "labels.parquet"
        if not manifest_path.is_file() or not labels_path.is_file():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        details = payload.get("details")
        try:
            valid_hash = payload.get("labels_sha256") == file_sha256(labels_path)
        except OSError:
            continue
        if (
            not accepts_legacy_version(
                payload.get("schema"), CAUSAL_RESPONSE_LABELS_SCHEMA
            )
            or payload.get("status") != "matched_labelled"
            or not str(payload.get("method", "")).startswith("frozen_matched_change")
            or payload.get("outcome_family") != outcome_family
            or payload.get("run_mode") != "production"
            or payload.get("production_eligible") is not True
            or not valid_hash
            or not isinstance(details, dict)
            or details.get("donor_scope") != donor_scope
        ):
            continue
        try:
            task_labels = pd.read_parquet(labels_path, columns=["estimator_backend"])
        except (OSError, KeyError, ValueError):
            continue
        python_task = task_labels["estimator_backend"].astype(str).str.contains(
            "python", case=False, na=False
        ).any()
        if python_task and (
            details.get("formal_qualification_eligible") is not True
            or not str(details.get("formal_qualification_receipt_sha256", ""))
        ):
            continue
        control_key = str(details.get("control_unit_key", ""))
        control_city, separator, control_grid = control_key.partition("::")
        if not separator or not control_city or not control_grid:
            continue
        if donor_scope == "same_city":
            if order not in controls.index:
                continue
            frozen = controls.loc[order]
            if isinstance(frozen, pd.DataFrame):
                continue
            if (
                str(frozen.get("status")) != "matched"
                or str(frozen.get("control_unit_key")) != control_key
            ):
                continue
        candidate["control_city_key"] = control_city
        candidate["control_grid_id"] = control_grid
        candidate["price_measure"] = str(payload.get("price_measure", "median"))
        accepted.append(candidate)
    matched = pd.DataFrame(accepted)
    parts: list[pd.DataFrame] = []
    for row in matched.to_dict("records"):
        record = pd.Series(row)
        part = (
            _annual_pair_panel(record, outcome_family, root, minimum, maximum)
            if annual
            else _monthly_pair_panel(
                record,
                outcome_family,
                root,
                minimum,
                maximum,
                str(record.get("price_measure", "median")),
            )
        )
        if part.empty:
            continue
        opening = str(record["opening_month"])
        if annual:
            part["event_time"] = part["period"].astype(int) - int(opening[:4])
            part["period"] = part["period"].astype(int)
        else:
            opening_period = pd.Period(opening, freq="M")
            values = pd.to_datetime(part["period"]).dt.to_period("M")
            opening_year = opening_period.year
            opening_month = opening_period.month
            part["event_time"] = (
                (values.dt.year - opening_year) * 12
                + values.dt.month
                - opening_month
            )
        part = part.loc[
            part["event_time"].between(minimum, reference)
            | part["event_time"].between(1, maximum)
        ].copy()
        order = int(record["treatment_order"])
        part["treatment_order"] = order
        part["unit"] = part["role"] + "_" + str(order) + "_" + part["source_city"] + "::" + part["source_grid"]
        part["grid_cluster"] = part["source_city"] + "::" + part["source_grid"]
        part["city_cluster"] = part["source_city"].astype(str)
        parts.append(part)
    if not parts:
        raise ValueError("no matched event-study outcome panels could be built")
    return pd.concat(parts, ignore_index=True), reference


def run_matching_event_study(
    outcome_family: str,
    output_directory: Path,
    *,
    root: Path = PROJECT_ROOT,
    control_queue: Path | pd.DataFrame | None = None,
    family_queue: Path | pd.DataFrame | None = None,
    task_root: Path | None = None,
    donor_scope: str = "same_city",
    min_pre: int | None = None,
    max_post: int | None = None,
) -> EventStudyResult:
    panel, reference = build_matching_event_study_panel(
        outcome_family,
        root=root,
        control_queue=control_queue,
        family_queue=family_queue,
        task_root=task_root,
        donor_scope=donor_scope,
        min_pre=min_pre,
        max_post=max_post,
    )
    result = fit_twfe_event_study(panel, reference_event_time=reference)
    output_directory.mkdir(parents=True, exist_ok=True)
    result.coefficients.to_csv(
        output_directory / "event_study_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    result.grid_cluster_pretrend.to_csv(
        output_directory / "parallel_trends_wald.csv", index=False, encoding="utf-8-sig"
    )
    result.city_cluster_pretrend.to_csv(
        output_directory / "parallel_trends_wald_city_cluster.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([result.diagnostics]).to_csv(
        output_directory / "diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    grid_p = pd.to_numeric(result.grid_cluster_pretrend["p_value"], errors="coerce").iloc[0]
    city_p = pd.to_numeric(result.city_cluster_pretrend["p_value"], errors="coerce").iloc[0]
    preferred_p = city_p if np.isfinite(city_p) else grid_p
    pd.DataFrame(
        [
            {
                "outcome_family": outcome_family,
                "donor_scope": donor_scope,
                "grid_pretrend_p_value": grid_p,
                "city_pretrend_p_value": city_p,
                "pretrend_flag": (
                    "cluster_flag"
                    if np.isfinite(preferred_p) and preferred_p < 0.05
                    else "cluster_not_detected"
                    if np.isfinite(preferred_p)
                    else "insufficient_clusters"
                ),
                "admission_policy": "diagnostic_only_not_an_automatic_label_gate",
            }
        ]
    ).to_csv(output_directory / "pretrend_metadata.csv", index=False, encoding="utf-8-sig")
    write_matching_event_study_figure(
        result,
        output_directory / "event_study_matching_python.png",
        title=f"Matching event study: {outcome_family} / {donor_scope}",
    )
    return result
