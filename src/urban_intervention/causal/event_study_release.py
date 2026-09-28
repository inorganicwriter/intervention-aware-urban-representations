"""Assemble the frozen four-family event-study release.

The release is the analysis-stage closure for the causal labels: it combines
the Matching TWFE event studies and the pooled GSC/MC response paths into one
window-consistent, method-consistent long panel, records availability of every
family/method/donor-scope cell, and documents which families enter downstream
supervision.

Design rules:

- Every coefficient row is filtered to the pre-registered canonical window of
  its family (``canonical_event_window``); the source files stay untouched.
- Matching rows are same-city (frozen controls); pooled GSC/MC rows may carry a
  same-city or cross-city donor scope, and cross-city cells with fewer than
  three clusters are kept but flagged.
- An unavailable cell is recorded as ``no_data`` rather than dropped silently.
- The manifest states the downstream selection; it is configuration, not a
  post-hoc test result.
"""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .event_study import canonical_event_window
from .gpu.provenance import file_sha256

FAMILIES: tuple[str, ...] = ("housing", "viirs", "poi", "population")
FINAL_SCHEMA = "mit_event_study_final"
FREQUENCIES = {
    "housing": "monthly",
    "viirs": "monthly",
    "poi": "annual",
    "population": "annual",
}
MAIN_OUTCOMES = {
    "housing": "housing_log_price",
    "viirs": "viirs_avg_asinh",
    "poi": "poi_count_log",
    "population": "population_log",
}
METHODS: tuple[str, ...] = ("matching", "gsc", "mc")
DONOR_SCOPES: tuple[str, ...] = ("same_city", "all_city_standardized")
REFERENCE_EVENT_TIME = -1
MIN_INTERPRETABLE_CLUSTERS = 3

PANEL_COLUMNS = [
    "outcome_family",
    "method",
    "donor_scope",
    "frequency",
    "outcome",
    "event_time",
    "event_phase",
    "estimate",
    "se_grid",
    "se_city",
    "p_grid",
    "p_city",
    "ci_lower_grid",
    "ci_upper_grid",
    "ci_lower_city",
    "ci_upper_city",
    "n_units",
    "n_cities",
    "se_type",
    "p_value_method",
    "n_city_clusters",
    "uncertainty_status",
    "specification",
    "source_file",
]

AVAILABILITY_COLUMNS = [
    "outcome_family",
    "analysis_type",
    "method",
    "donor_scope",
    "status",
    "reason",
    "availability_treated_events",
    "event_study_max_n_treated",
    "max_supported_treated_grids",
    "coverage",
    "rows",
    "n_cities",
    "n_cities_min",
    "canonical_pre_min",
    "canonical_post_max",
    "source_pre_min",
    "source_pre_max",
    "source_post_min",
    "source_post_max",
    "t0_available",
    "warnings",
    "source_file",
]

PRETREND_COLUMNS = [
    "outcome_family",
    "method",
    "donor_scope",
    "frequency",
    "outcome",
    "cluster",
    "test_type",
    "statistic",
    "df1",
    "df2",
    "covariance_rank",
    "n_cities",
    "n_pre_observations",
    "mean_city_mean",
    "sd_city_mean",
    "t_statistic",
    "p_value",
    "reject_5pct",
    "pretrend_flag",
    "source_file",
]

SELECTION_COLUMNS = [
    "outcome_family",
    "analyzed",
    "selected_family",
    "selection_reason",
    "frequency",
    "canonical_outcome",
    "availability_treated_events",
    "event_study_max_n_treated",
    "max_supported_treated_grids",
    "n_treated_gap",
    "coverage",
    "matching_n_cities",
    "matching_city_pretrend_p",
    "pooled_min_n_cities",
    "post_mean_matching",
    "post_share_significant_city",
    "method_sign_consistency",
]

CAVEATS = [
    "Pooled GSC/MC estimates are label-scale means per method and donor scope; "
    "compare them within a method/scope, not across standardization choices.",
    "The opening period (event_time = 0) is a real impact period, not a clean "
    "lead; pre-trend tests use event_time < 0 only.",
    "Cross-city donor cells with fewer than three city clusters are retained "
    "but flagged insufficient_clusters; their p-values are not interpretable.",
    "Only the pre-registered canonical windows are exported; full-window "
    "event-study output is intentionally excluded from this release.",
    "226 zero-anchor MC tasks were demoted to skipped before this release; "
    "pooled coverage reflects the post-demotion production queue.",
    "Pooled p-values are two-sided normal approximations from city-cluster "
    "standard errors (se_type=city_cluster, p_value_method=normal_approximation, "
    "n_city_clusters recorded in event_study_panel.csv).",
    "Treated-event counts are stage-specific and never collapsed into one field: "
    "availability_treated_events (admission stage), event_study_max_n_treated "
    "(canonical-window support) and max_supported_treated_grids (pooled cells).",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _optional_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return _read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def normalize_treated_event_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map legacy ambiguous ``treated_events`` columns to the current schema.

    Older release tables carried a single ``treated_events`` column with
    different meanings per analysis stage.  Reading such a table maps the
    legacy value by analysis type: matching rows to
    ``availability_treated_events`` and pooled rows to
    ``max_supported_treated_grids``.  New releases never write the ambiguous
    column again.
    """
    if "treated_events" not in frame.columns:
        return frame
    normalized = frame.copy()
    legacy = pd.to_numeric(normalized["treated_events"], errors="coerce")
    if "analysis_type" in normalized.columns:
        analysis = normalized["analysis_type"].astype(str)
    else:
        method = normalized.get(
            "method", pd.Series("", index=normalized.index)
        ).astype(str)
        analysis = pd.Series(
            np.where(method.eq("matching"), "matching", "pooled"),
            index=normalized.index,
        )
    matching_mask = analysis.eq("matching")
    if "availability_treated_events" not in normalized.columns:
        normalized["availability_treated_events"] = np.nan
    if "max_supported_treated_grids" not in normalized.columns:
        normalized["max_supported_treated_grids"] = np.nan
    normalized.loc[matching_mask, "availability_treated_events"] = normalized.loc[
        matching_mask, "availability_treated_events"
    ].fillna(legacy[matching_mask])
    normalized.loc[~matching_mask, "max_supported_treated_grids"] = normalized.loc[
        ~matching_mask, "max_supported_treated_grids"
    ].fillna(legacy[~matching_mask])
    return normalized.drop(columns=["treated_events"])


def _normal_p_value(t_stat: pd.Series) -> pd.Series:
    values = pd.to_numeric(t_stat, errors="coerce")
    result = values.map(
        lambda value: math.erfc(abs(float(value)) / math.sqrt(2.0))
        if np.isfinite(value)
        else np.nan
    )
    return result.astype(float)


def _event_phase(event_time: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(event_time, errors="coerce")
    phase = pd.Series("post", index=numeric.index, dtype="object")
    phase = phase.mask(numeric.eq(REFERENCE_EVENT_TIME), "reference")
    phase = phase.mask(numeric.lt(0) & numeric.ne(REFERENCE_EVENT_TIME), "pre")
    phase = phase.mask(numeric.eq(0), "t0")
    phase = phase.mask(numeric.isna(), "unknown")
    return phase


def find_matching_directory(
    root: Path, family: str, donor_scope: str
) -> Path | None:
    """Locate a family's Matching event-study directory for one donor scope."""
    candidates: tuple[Path, ...]
    if donor_scope == "same_city":
        candidates = (
            root / "matching_twfe" / family,
            root / f"matching_{family}",
            root / family,
        )
    else:
        candidates = (
            root / "matching_twfe_cross" / family,
            root / f"matching_{family}_cross",
        )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def find_family_directory(root: Path, family: str) -> Path | None:
    """Backward-compatible same-city lookup."""
    return find_matching_directory(root, family, "same_city")


def _window_bounds(family: str) -> tuple[int, int]:
    return canonical_event_window(family)


def _filter_window(frame: pd.DataFrame, family: str) -> tuple[pd.DataFrame, int]:
    bounds = _window_bounds(family)
    times = pd.to_numeric(frame["event_time"], errors="coerce")
    keep = times.ge(bounds[0]) & times.le(bounds[1])
    removed = int((~keep).sum())
    return frame.loc[keep].copy(), removed


def _standardize_matching(
    family: str, directory: Path, donor_scope: str = "same_city"
) -> tuple[pd.DataFrame, int]:
    coefficients = _read_csv(directory / "event_study_coefficients.csv")
    diagnostics = {}
    diagnostics_path = directory / "diagnostics.csv"
    if diagnostics_path.is_file():
        diagnostics_frame = _read_csv(diagnostics_path)
        if not diagnostics_frame.empty:
            diagnostics = diagnostics_frame.iloc[0].to_dict()
    n_cities = diagnostics.get("city_clusters")
    frame = pd.DataFrame(
        {
            "outcome_family": family,
            "method": "matching",
            "donor_scope": donor_scope,
            "frequency": FREQUENCIES[family],
            "outcome": MAIN_OUTCOMES[family],
            "event_time": _numeric_column(coefficients, "event_time"),
            "estimate": _numeric_column(coefficients, "estimate"),
            "se_grid": _numeric_column(coefficients, "standard_error_grid"),
            "se_city": _numeric_column(coefficients, "standard_error_city"),
            "p_grid": _numeric_column(coefficients, "p_value_grid"),
            "p_city": _numeric_column(coefficients, "p_value_city"),
            "ci_lower_grid": _numeric_column(coefficients, "confidence_lower_grid"),
            "ci_upper_grid": _numeric_column(coefficients, "confidence_upper_grid"),
            "ci_lower_city": _numeric_column(coefficients, "confidence_lower_city"),
            "ci_upper_city": _numeric_column(coefficients, "confidence_upper_city"),
            "n_units": _numeric_column(coefficients, "n_treated"),
            "n_cities": pd.to_numeric(n_cities, errors="coerce"),
            "se_type": "city_cluster",
            "p_value_method": "twfe_cluster_robust",
            "n_city_clusters": pd.to_numeric(n_cities, errors="coerce"),
            "uncertainty_status": np.where(
                _numeric_column(coefficients, "standard_error_city").notna(),
                "available",
                "missing",
            ),
            "specification": "",
            "source_file": f"{directory.name}/event_study_coefficients.csv",
        }
    )
    frame["event_phase"] = _event_phase(frame["event_time"])
    frame, removed = _filter_window(frame, family)
    frame["rows_before_window_filter"] = len(frame) + removed
    return frame, removed


def _pooled_uncertainty(frame: pd.DataFrame) -> pd.Series:
    if "uncertainty_status" in frame.columns:
        status = frame["uncertainty_status"].astype(str)
        status = status.mask(status.isin({"", "nan", "None"}), "available")
        return status
    se = pd.to_numeric(frame.get("se_label_city_cluster"), errors="coerce")
    return pd.Series(np.where(se.notna(), "available", "missing"), index=frame.index)


def _standardize_pooled(pooled_directory: Path) -> tuple[pd.DataFrame, int]:
    series = _read_csv(pooled_directory / "event_study_series.csv")
    frame = pd.DataFrame(
        {
            "outcome_family": series["outcome_family"].astype(str),
            "method": series["method"].map(_method_family),
            "donor_scope": series["donor_scope"].astype(str),
            "frequency": series["frequency"].astype(str),
            "outcome": series["outcome"].astype(str),
            "event_time": pd.to_numeric(series["event_time"], errors="coerce"),
            "estimate": pd.to_numeric(series["mean_label"], errors="coerce"),
            "se_grid": pd.to_numeric(series.get("se_label"), errors="coerce"),
            "se_city": pd.to_numeric(series.get("se_label_city_cluster"), errors="coerce"),
            "p_grid": np.nan,
            "p_city": np.nan,
            "ci_lower_grid": pd.to_numeric(series.get("ci_lower"), errors="coerce"),
            "ci_upper_grid": pd.to_numeric(series.get("ci_upper"), errors="coerce"),
            "ci_lower_city": pd.to_numeric(
                series.get("ci_lower_city_cluster"), errors="coerce"
            ),
            "ci_upper_city": pd.to_numeric(
                series.get("ci_upper_city_cluster"), errors="coerce"
            ),
            "n_units": pd.to_numeric(series.get("n_grids"), errors="coerce"),
            "n_cities": pd.to_numeric(series.get("n_cities"), errors="coerce"),
            "se_type": "city_cluster",
            "p_value_method": "normal_approximation",
            "n_city_clusters": pd.to_numeric(series.get("n_cities"), errors="coerce"),
            "uncertainty_status": _pooled_uncertainty(series),
            "specification": series.get("specifications", pd.Series("", index=series.index))
            .fillna("")
            .astype(str),
            "source_file": f"{pooled_directory.name}/event_study_series.csv",
        }
    )
    frame["p_grid"] = _normal_p_value(
        frame["estimate"] / frame["se_grid"].where(frame["se_grid"] > 0)
    )
    frame["p_city"] = _normal_p_value(
        frame["estimate"] / frame["se_city"].where(frame["se_city"] > 0)
    )
    frame["event_phase"] = _event_phase(frame["event_time"])
    filtered_parts: list[pd.DataFrame] = []
    removed_total = 0
    for family in sorted(frame["outcome_family"].unique()):
        if family not in FAMILIES:
            removed_total += int(frame["outcome_family"].eq(family).sum())
            continue
        part, removed = _filter_window(frame.loc[frame["outcome_family"].eq(family)], family)
        part["rows_before_window_filter"] = len(part) + removed
        filtered_parts.append(part)
        removed_total += removed
    if not filtered_parts:
        return pd.DataFrame(columns=PANEL_COLUMNS), removed_total
    return pd.concat(filtered_parts, ignore_index=True), removed_total


def _method_family(value: object) -> str:
    text = str(value)
    if "gsynth" in text or "gsc" in text:
        return "gsc"
    if "mc" in text:
        return "mc"
    return text


def _method_outcome_key(row: pd.Series) -> tuple[str, str] | None:
    family = str(row["outcome_family"])
    method = str(row["method"])
    scope = str(row["donor_scope"])
    if method not in METHODS or scope not in DONOR_SCOPES or family not in FAMILIES:
        return None
    return method, scope


def _matching_availability(
    family: str,
    directory: Path | None,
    support: pd.DataFrame | None,
    matching_panel: pd.DataFrame,
    donor_scope: str = "same_city",
    component_reason: str | None = None,
) -> dict[str, object]:
    bounds = _window_bounds(family)
    panel_rows = int(len(matching_panel))
    window_max = np.nan
    if panel_rows > 0:
        n_treated = _numeric_column(matching_panel, "n_units")
        if n_treated.notna().any():
            window_max = float(n_treated.max())
    row: dict[str, object] = {
        "outcome_family": family,
        "analysis_type": "matching",
        "method": "matching",
        "donor_scope": donor_scope,
        "status": "no_data",
        "reason": "",
        "availability_treated_events": np.nan,
        "event_study_max_n_treated": window_max if panel_rows > 0 else np.nan,
        "max_supported_treated_grids": np.nan,
        "coverage": np.nan,
        "rows": panel_rows,
        "n_cities": np.nan,
        "n_cities_min": np.nan,
        "canonical_pre_min": bounds[0],
        "canonical_post_max": bounds[1],
        "source_pre_min": np.nan,
        "source_pre_max": np.nan,
        "source_post_min": np.nan,
        "source_post_max": np.nan,
        "t0_available": np.nan,
        "warnings": "",
        "source_file": "",
    }
    if directory is None:
        row["reason"] = component_reason or (
            "insufficient_samples_for_event_study"
            if donor_scope != "same_city"
            else "matching_directory_missing"
        )
        return row
    diagnostics_path = directory / "diagnostics.csv"
    coefficients_path = directory / "event_study_coefficients.csv"
    if not diagnostics_path.is_file() or not coefficients_path.is_file():
        row["reason"] = component_reason or "matching_outputs_missing"
        return row
    diagnostics = _read_csv(diagnostics_path).iloc[0].to_dict()
    availability_treated = pd.to_numeric(
        diagnostics.get("treated_events"), errors="coerce"
    )
    row.update(
        {
            "status": "available" if panel_rows > 0 else "no_data",
            "reason": "" if panel_rows > 0 else "no_coefficients_in_canonical_window",
            "availability_treated_events": availability_treated,
            "n_cities": diagnostics.get("city_clusters"),
            "source_file": f"{directory.name}/event_study_coefficients.csv",
        }
    )
    if pd.notna(availability_treated) and np.isfinite(window_max) and availability_treated > 0:
        row["coverage"] = window_max / float(availability_treated)
    if support is not None:
        match = support.loc[
            support["outcome_family"].eq(family) & support["method"].eq("matching")
        ]
        if not match.empty:
            supported = match.iloc[0]
            row.update(
                {
                    "source_pre_min": supported.get("method_pre_min"),
                    "source_pre_max": supported.get("method_pre_max"),
                    "source_post_min": supported.get("method_post_min"),
                    "source_post_max": supported.get("method_post_max"),
                    "t0_available": supported.get("t0_available"),
                }
            )
            if not bool(supported.get("available", False)):
                row["status"] = "no_data"
                row["reason"] = "not_available_in_event_time_support"
    return row


def _pooled_availability(
    panel: pd.DataFrame,
    support: pd.DataFrame | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        bounds = _window_bounds(family)
        for method in ("gsc", "mc"):
            for scope in DONOR_SCOPES:
                subset = panel.loc[
                    panel["outcome_family"].eq(family)
                    & panel["method"].eq(method)
                    & panel["donor_scope"].eq(scope)
                ]
                row: dict[str, object] = {
                    "outcome_family": family,
                    "analysis_type": "pooled",
                    "method": method,
                    "donor_scope": scope,
                    "status": "no_data",
                    "reason": "",
                    "availability_treated_events": np.nan,
                    "event_study_max_n_treated": np.nan,
                    "max_supported_treated_grids": np.nan,
                    "coverage": np.nan,
                    "rows": int(len(subset)),
                    "n_cities": np.nan,
                    "n_cities_min": np.nan,
                    "canonical_pre_min": bounds[0],
                    "canonical_post_max": bounds[1],
                    "source_pre_min": np.nan,
                    "source_pre_max": np.nan,
                    "source_post_min": np.nan,
                    "source_post_max": np.nan,
                    "t0_available": np.nan,
                    "warnings": "",
                    "source_file": "",
                }
                if support is not None:
                    match = support.loc[
                        support["outcome_family"].eq(family) & support["method"].eq(method)
                    ]
                    if not match.empty:
                        supported = match.iloc[0]
                        row.update(
                            {
                                "source_pre_min": supported.get("method_pre_min"),
                                "source_pre_max": supported.get("method_pre_max"),
                                "source_post_min": supported.get("method_post_min"),
                                "source_post_max": supported.get("method_post_max"),
                                "t0_available": supported.get("t0_available"),
                            }
                        )
                if subset.empty:
                    row["reason"] = "no_pooled_rows_in_canonical_window"
                    rows.append(row)
                    continue
                n_cities = pd.to_numeric(subset["n_cities"], errors="coerce")
                row.update(
                    {
                        "status": "available",
                        "max_supported_treated_grids": pd.to_numeric(
                            subset["n_units"], errors="coerce"
                        ).max(),
                        "n_cities": n_cities.max(),
                        "n_cities_min": n_cities.min(),
                        "source_file": str(subset["source_file"].iloc[0]),
                    }
                )
                if np.isfinite(n_cities.min()) and n_cities.min() < MIN_INTERPRETABLE_CLUSTERS:
                    row["warnings"] = "insufficient_clusters"
                if support is not None:
                    match = support.loc[
                        support["outcome_family"].eq(family) & support["method"].eq(method)
                    ]
                    if not match.empty and not bool(match.iloc[0].get("available", False)):
                        row["status"] = "no_data"
                        row["reason"] = "not_available_in_event_time_support"
                rows.append(row)
    return rows


def _support_table(suite_root: Path | None, matching_root: Path, pooled_root: Path) -> pd.DataFrame | None:
    candidates = [matching_root, pooled_root]
    if suite_root is not None:
        candidates.insert(0, suite_root)
    for candidate in candidates:
        path = candidate / "event_time_support.csv"
        frame = _optional_read_csv(path)
        if frame is not None:
            return frame
    return None


def _suite_status_table(
    suite_root: Path | None, matching_root: Path, pooled_root: Path
) -> dict[str, tuple[str, str]]:
    """Map component name -> (status, details) from suite_status.csv when present."""
    candidates: list[Path] = []
    if suite_root is not None:
        candidates.append(suite_root)
    candidates.extend([matching_root, pooled_root])
    for candidate in candidates:
        frame = _optional_read_csv(candidate / "suite_status.csv")
        if frame is None or not {"component", "status"}.issubset(frame.columns):
            continue
        lookup: dict[str, tuple[str, str]] = {}
        for row in frame.itertuples():
            details = getattr(row, "details", "")
            lookup[str(row.component)] = (
                str(row.status),
                "" if pd.isna(details) else str(details),
            )
        return lookup
    return {}


def _pretrend_rows(
    matching_root: Path,
    pooled_root: Path,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for family in FAMILIES:
        for donor_scope in ("same_city", "all_city_standardized"):
            directory = find_matching_directory(matching_root, family, donor_scope)
            if directory is None:
                continue
            for cluster, filename in (
                ("grid", "parallel_trends_wald.csv"),
                ("city", "parallel_trends_wald_city_cluster.csv"),
            ):
                frame = _optional_read_csv(directory / filename)
                if frame is None or frame.empty:
                    continue
                row = frame.iloc[0]
                p_value = pd.to_numeric(row.get("p_value"), errors="coerce")
                records.append(
                    {
                        "outcome_family": family,
                        "method": "matching",
                        "donor_scope": donor_scope,
                        "frequency": FREQUENCIES[family],
                        "outcome": MAIN_OUTCOMES[family],
                        "cluster": cluster,
                        "test_type": "joint_wald_pretrend",
                        "statistic": row.get("statistic"),
                        "df1": row.get("df1"),
                        "df2": row.get("df2"),
                        "covariance_rank": row.get("covariance_rank"),
                        "n_cities": np.nan,
                        "n_pre_observations": np.nan,
                        "mean_city_mean": np.nan,
                        "sd_city_mean": np.nan,
                        "t_statistic": np.nan,
                        "p_value": p_value,
                        "reject_5pct": bool(p_value < 0.05) if pd.notna(p_value) else False,
                        "pretrend_flag": (
                            "cluster_flag"
                            if pd.notna(p_value) and p_value < 0.05
                            else "cluster_not_detected"
                            if pd.notna(p_value)
                            else "insufficient_clusters"
                        ),
                        "source_file": f"{directory.name}/{filename}",
                    }
                )
    pooled_pretrend_parts: list[pd.DataFrame] = []
    for filename in ("pretrend_city_cluster.csv", "pretrend_grid_cluster.csv"):
        frame = _optional_read_csv(pooled_root / filename)
        if frame is not None and not frame.empty:
            frame = frame.copy()
            frame["cluster"] = "city" if "city" in filename else "grid"
            frame["source_file"] = f"{pooled_root.name}/{filename}"
            pooled_pretrend_parts.append(frame)
    if pooled_pretrend_parts:
        pooled = pd.concat(pooled_pretrend_parts, ignore_index=True)
        p_values = pd.to_numeric(pooled["p_value"], errors="coerce")
        pooled["reject_5pct"] = p_values < 0.05
        pooled["pretrend_flag"] = np.where(
            p_values.isna(),
            "insufficient_clusters",
            np.where(p_values < 0.05, "cluster_flag", "cluster_not_detected"),
        )
        pooled["method"] = pooled["method"].map(_method_family)
        records.extend(pooled.to_dict("records"))
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=PRETREND_COLUMNS)
    for column in PRETREND_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def _post_rows(panel: pd.DataFrame, family: str, method: str, scope: str, outcome: str) -> pd.DataFrame:
    return panel.loc[
        panel["outcome_family"].eq(family)
        & panel["method"].eq(method)
        & panel["donor_scope"].eq(scope)
        & panel["outcome"].eq(outcome)
        & panel["event_phase"].eq("post"),
    ]


def _selection_table(
    panel: pd.DataFrame,
    pretrend: pd.DataFrame,
    availability: pd.DataFrame,
    downstream: tuple[str, ...],
) -> pd.DataFrame:
    """Record the analysis-stage selection with explicit two-stage sample counts.

    ``availability_treated_events`` is the treated-event count identified at the
    availability/admission stage; ``event_study_max_n_treated`` is the largest
    per-event-time ``n_treated`` that actually enters the canonical-window
    event-study coefficients.  The two are kept separate (with gap and coverage)
    instead of being forced to agree, because they describe different stages.
    """
    records: list[dict[str, object]] = []
    for family in FAMILIES:
        outcome = MAIN_OUTCOMES[family]
        matching_post = _post_rows(panel, family, "matching", "same_city", outcome)
        matching_city_p = np.nan
        matching_city_row = pretrend.loc[
            pretrend["outcome_family"].eq(family)
            & pretrend["method"].eq("matching")
            & pretrend["cluster"].eq("city")
        ]
        if not matching_city_row.empty:
            matching_city_p = pd.to_numeric(
                matching_city_row.iloc[0]["p_value"], errors="coerce"
            )
        matching_row = panel.loc[
            panel["outcome_family"].eq(family)
            & panel["method"].eq("matching")
        ]
        matching_n_cities = np.nan
        if not matching_row.empty:
            matching_n_cities = pd.to_numeric(
                matching_row["n_cities"], errors="coerce"
            ).max()

        availability_row = availability.loc[
            availability["outcome_family"].eq(family)
            & availability["method"].eq("matching")
            & availability["donor_scope"].eq("same_city")
        ]
        availability_treated_events = np.nan
        if not availability_row.empty:
            value = pd.to_numeric(
                availability_row.iloc[0]["availability_treated_events"], errors="coerce"
            )
            if pd.notna(value):
                availability_treated_events = float(value)

        event_study_max_n_treated = np.nan
        if not matching_row.empty:
            n_treated = _numeric_column(matching_row, "n_units")
            if n_treated.notna().any():
                event_study_max_n_treated = float(n_treated.max())

        n_treated_gap = np.nan
        coverage = np.nan
        if np.isfinite(availability_treated_events) and np.isfinite(
            event_study_max_n_treated
        ):
            n_treated_gap = availability_treated_events - event_study_max_n_treated
            if availability_treated_events > 0:
                coverage = event_study_max_n_treated / availability_treated_events

        pooled_family = panel.loc[
            panel["outcome_family"].eq(family)
            & panel["method"].isin(["gsc", "mc"])
            & panel["outcome"].eq(outcome)
        ]
        pooled_min_cities = np.nan
        if not pooled_family.empty:
            pooled_min_cities = pd.to_numeric(
                pooled_family["n_cities"], errors="coerce"
            ).min()
        max_supported_grids = np.nan
        if not pooled_family.empty:
            supported = _numeric_column(pooled_family, "n_units")
            if supported.notna().any():
                max_supported_grids = float(supported.max())
        post_mean_matching = (
            float(matching_post["estimate"].mean())
            if not matching_post.empty
            else np.nan
        )
        t_city = matching_post["estimate"] / matching_post["se_city"].where(
            matching_post["se_city"] > 0
        )
        post_share_sig = (
            float((t_city.abs() >= 1.96).mean()) if not matching_post.empty else np.nan
        )
        signs: list[int] = []
        if not matching_post.empty:
            signs.append(1 if float(matching_post["estimate"].mean()) >= 0 else -1)
        for method in ("gsc", "mc"):
            for scope in DONOR_SCOPES:
                subset = _post_rows(panel, family, method, scope, outcome)
                if subset.empty:
                    continue
                mean_value = float(subset["estimate"].mean())
                signs.append(1 if mean_value >= 0 else -1)
        consistency = (
            max(signs.count(1), signs.count(-1)) / len(signs) if signs else np.nan
        )
        selected = family in downstream
        records.append(
            {
                "outcome_family": family,
                "analyzed": True,
                "selected_family": selected,
                "selection_reason": (
                    "selected for downstream causal supervision"
                    if selected
                    else "analyzed but excluded from downstream causal supervision"
                ),
                "frequency": FREQUENCIES[family],
                "canonical_outcome": outcome,
                "availability_treated_events": availability_treated_events,
                "event_study_max_n_treated": event_study_max_n_treated,
                "max_supported_treated_grids": max_supported_grids,
                "n_treated_gap": n_treated_gap,
                "coverage": coverage,
                "matching_n_cities": matching_n_cities,
                "matching_city_pretrend_p": matching_city_p,
                "pooled_min_n_cities": pooled_min_cities,
                "post_mean_matching": post_mean_matching,
                "post_share_significant_city": post_share_sig,
                "method_sign_consistency": consistency,
            }
        )
    return pd.DataFrame(records, columns=SELECTION_COLUMNS)


def _selection_markdown(selection: pd.DataFrame, downstream: tuple[str, ...]) -> str:
    lines = [
        "# Event-study family selection",
        "",
        "The four families are all analyzed and exported.  Downstream causal",
        "supervision uses **" + ", ".join(downstream) + "** only.  Housing and",
        "population are retained as analyzed outcomes but excluded from downstream",
        "supervision; this is a fixed scope decision recorded here, not a",
        "post-hoc significance filter.",
        "",
        "Two sample counts are reported separately because they belong to",
        "different stages:",
        "",
        "- `availability_treated_events`: treated events identified at the",
        "  availability/admission stage;",
        "- `event_study_max_n_treated`: largest per-event-time `n_treated` entering",
        "  the canonical-window event-study coefficients;",
        "- `n_treated_gap = availability_treated_events - event_study_max_n_treated`;",
        "- `coverage = event_study_max_n_treated / availability_treated_events`;",
        "  coverage above 1 signals mixed releases/versions or inconsistent",
        "  de-duplication keys and is flagged by `validation.json`.",
        "",
        "Criteria recorded for transparency:",
        "",
        "- signal stability: share of post periods with |t| >= 1.96 (city cluster);",
        "- pre-trends: joint city-cluster pre-trend test p-value (diagnostic, not a gate);",
        "- effective sample: availability treated events, canonical-window support",
        "  and city clusters;",
        "- method consistency: sign agreement of post-period means across methods",
        "  and donor scopes for the family's canonical outcome.",
        "",
        "| family | selected | freq | outcome | availability events | window max n_treated | gap | coverage | city clusters | pre-trend p | pooled min cities | post mean (matching) | share |t|>=1.96 | sign consistency |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selection.itertuples():
        def _fmt(value: object, digits: int = 3) -> str:
            if value is None:
                return "n/a"
            if not isinstance(value, (int, float, np.integer, np.floating)):
                return str(value)
            number = float(value)
            if not np.isfinite(number):
                return "n/a"
            return f"{number:.{digits}f}"

        lines.append(
            "| {family} | {selected} | {freq} | {outcome} | {events} | {max_n} | "
            "{gap} | {coverage} | {cities} | {p} | {min_cities} | {post} | "
            "{share} | {consistency} |".format(
                family=row.outcome_family,
                selected="yes" if row.selected_family else "no",
                freq=row.frequency,
                outcome=row.canonical_outcome,
                events=_fmt(row.availability_treated_events, 0),
                max_n=_fmt(row.event_study_max_n_treated, 0),
                gap=_fmt(row.n_treated_gap, 0),
                coverage=_fmt(row.coverage),
                cities=_fmt(row.matching_n_cities, 0),
                p=_fmt(row.matching_city_pretrend_p),
                min_cities=_fmt(row.pooled_min_n_cities, 0),
                post=_fmt(row.post_mean_matching),
                share=_fmt(row.post_share_significant_city),
                consistency=_fmt(row.method_sign_consistency),
            )
        )
    lines.extend(
        [
            "",
            "Cross-city pooled cells with fewer than three city clusters are",
            "flagged `insufficient_clusters` in `availability.csv`; their cluster",
            "p-values are not interpretable.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_manifest(
    output_directory: Path,
    *,
    matching_root: Path,
    pooled_root: Path,
    suite_root: Path | None,
    downstream: tuple[str, ...],
    panel: pd.DataFrame,
    availability: pd.DataFrame,
    pretrend: pd.DataFrame,
    selection: pd.DataFrame,
    input_hashes: list[dict[str, object]],
    output_files: list[Path],
    rows_filtered: int,
) -> dict[str, object]:
    windows = {
        family: {
            "frequency": FREQUENCIES[family],
            "canonical_pre_min": _window_bounds(family)[0],
            "canonical_post_max": _window_bounds(family)[1],
            "reference_event_time": REFERENCE_EVENT_TIME,
        }
        for family in FAMILIES
    }
    outputs = []
    for path in sorted(output_files):
        outputs.append(
            {
                "path": path.relative_to(output_directory).as_posix(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    status_counts = (
        availability["status"].value_counts().to_dict() if not availability.empty else {}
    )
    matching_scopes = sorted(
        set(panel.loc[panel["method"].eq("matching"), "donor_scope"].astype(str))
    )
    manifest: dict[str, object] = {
        "schema": FINAL_SCHEMA,
        "created_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "windows": windows,
        "methods": {
            "matching": matching_scopes or ["same_city"],
            "gsc": list(DONOR_SCOPES),
            "mc": list(DONOR_SCOPES),
        },
        "canonical_outcomes": dict(MAIN_OUTCOMES),
        "downstream_training": {
            "included_families": list(downstream),
            "excluded_families": [
                family for family in FAMILIES if family not in downstream
            ],
            "rationale_file": "family_selection.md",
            "selection_file": "family_selection.csv",
        },
        "sources": {
            "matching_root": str(matching_root),
            "pooled_root": str(pooled_root),
            "suite_root": str(suite_root) if suite_root is not None else None,
        },
        "input_hashes": input_hashes,
        "outputs": outputs,
        "row_counts": {
            "panel": int(len(panel)),
            "availability": int(len(availability)),
            "pretrend": int(len(pretrend)),
            "selection": int(len(selection)),
        },
        "window_filtered_rows": int(rows_filtered),
        "availability_status_counts": {
            str(key): int(value) for key, value in status_counts.items()
        },
        "caveats": list(CAVEATS),
    }
    return manifest


def _validate(
    panel: pd.DataFrame,
    availability: pd.DataFrame,
    pretrend: pd.DataFrame,
    downstream: tuple[str, ...],
) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, severity: str, detail: str) -> None:
        checks.append(
            {"name": name, "passed": bool(passed), "severity": severity, "detail": detail}
        )

    present_families = set(panel["outcome_family"].astype(str))
    missing = sorted(set(FAMILIES) - present_families)
    add(
        "all_families_available",
        not missing,
        "error",
        f"missing families: {missing}" if missing else "all four families present",
    )
    matching = availability.loc[
        availability["method"].eq("matching")
        & availability["donor_scope"].eq("same_city")
    ]
    missing_matching = sorted(set(FAMILIES) - set(matching["outcome_family"]))
    unavailable_matching = sorted(
        matching.loc[~matching["status"].eq("available"), "outcome_family"].astype(str)
    )
    add(
        "matching_available",
        not missing_matching and not unavailable_matching,
        "error",
        (
            f"missing: {missing_matching}; no_data: {unavailable_matching}"
            if missing_matching or unavailable_matching
            else "matching available for all four families"
        ),
    )
    downstream_missing = sorted(set(downstream) - present_families)
    add(
        "downstream_families_available",
        not downstream_missing,
        "error",
        (
            f"missing downstream families: {downstream_missing}"
            if downstream_missing
            else "downstream families present"
        ),
    )
    outside = pd.DataFrame()
    for family in FAMILIES:
        subset = panel.loc[panel["outcome_family"].eq(family)]
        if subset.empty:
            continue
        bounds = _window_bounds(family)
        times = pd.to_numeric(subset["event_time"], errors="coerce")
        outside = pd.concat(
            [
                outside,
                subset.loc[times.lt(bounds[0]) | times.gt(bounds[1])],
            ]
        )
    add(
        "canonical_window_only",
        outside.empty,
        "error",
        f"{len(outside)} rows outside canonical windows" if not outside.empty else "ok",
    )
    key_nulls = {
        column: int(panel[column].isna().sum())
        for column in ("outcome_family", "method", "donor_scope", "event_time", "estimate")
    }
    add(
        "panel_key_fields",
        all(value == 0 for value in key_nulls.values()),
        "warning",
        str(key_nulls),
    )
    add(
        "panel_columns",
        all(column in panel.columns for column in PANEL_COLUMNS),
        "error",
        "unified schema complete"
        if all(column in panel.columns for column in PANEL_COLUMNS)
        else "columns missing",
    )
    pretrend_families = set(pretrend.get("outcome_family", pd.Series(dtype=str)).astype(str))
    add(
        "pretrend_recorded",
        set(FAMILIES).issubset(pretrend_families),
        "warning",
        f"pretrend families: {sorted(pretrend_families)}",
    )
    violations: list[str] = []
    matching_rows = availability.loc[availability["analysis_type"].eq("matching")]
    for row in matching_rows.itertuples():
        available_events = pd.to_numeric(
            row.availability_treated_events, errors="coerce"
        )
        window_max = pd.to_numeric(row.event_study_max_n_treated, errors="coerce")
        if not (pd.notna(available_events) and pd.notna(window_max)):
            continue
        if window_max > available_events + 1e-9:
            violations.append(
                f"{row.outcome_family}: window max {window_max:g} > "
                f"availability {available_events:g}"
            )
    add(
        "treated_event_consistency",
        not violations,
        "error",
        (
            "event_study_max_n_treated > availability_treated_events for "
            f"{'; '.join(violations)}; check mixed releases/versions, "
            "de-duplication keys, events outside the availability table, and "
            "whether treated events and treated units share a statistical unit"
        )
        if violations
        else "canonical-window support never exceeds the availability count",
    )
    return {
        "checks": checks,
        "ok": all(
            check["passed"] for check in checks if check["severity"] == "error"
        ),
    }


def collect_event_study_release(
    matching_root: Path,
    pooled_root: Path,
    *,
    suite_root: Path | None = None,
    downstream: tuple[str, ...] = ("viirs", "poi"),
) -> dict[str, Any]:
    """Collect, standardize and validate the four-family release in memory."""
    support = _support_table(suite_root, matching_root, pooled_root)
    suite_status_lookup = _suite_status_table(suite_root, matching_root, pooled_root)
    matching_parts: list[pd.DataFrame] = []
    matching_removed = 0
    matching_directories: dict[tuple[str, str], Path] = {}
    for donor_scope in ("same_city", "all_city_standardized"):
        for family in FAMILIES:
            directory = find_matching_directory(matching_root, family, donor_scope)
            if directory is None or not (
                directory / "event_study_coefficients.csv"
            ).is_file():
                continue
            frame, removed = _standardize_matching(family, directory, donor_scope)
            matching_parts.append(frame)
            matching_removed += removed
            matching_directories[(family, donor_scope)] = directory
    matching_panel = (
        pd.concat(matching_parts, ignore_index=True)
        if matching_parts
        else pd.DataFrame(columns=PANEL_COLUMNS)
    )
    pooled_panel, pooled_removed = _standardize_pooled(pooled_root)
    panel = pd.concat([matching_panel, pooled_panel], ignore_index=True)
    for column in PANEL_COLUMNS:
        if column not in panel.columns:
            panel[column] = np.nan
    panel = panel[PANEL_COLUMNS]

    availability_rows: list[dict[str, object]] = []
    for family in FAMILIES:
        for donor_scope in ("same_city", "all_city_standardized"):
            directory = matching_directories.get((family, donor_scope))
            component = (
                f"matching_twfe:{family}"
                if donor_scope == "same_city"
                else f"matching_twfe_cross:{family}"
            )
            component_reason: str | None = None
            status_entry = suite_status_lookup.get(component)
            if status_entry is not None and status_entry[0] == "no_data":
                component_reason = (
                    status_entry[1] or "insufficient_samples_for_event_study"
                )
            family_panel = matching_panel.loc[
                matching_panel["outcome_family"].eq(family)
                & matching_panel["donor_scope"].eq(donor_scope)
            ]
            availability_rows.append(
                _matching_availability(
                    family,
                    directory,
                    support if donor_scope == "same_city" else None,
                    family_panel,
                    donor_scope,
                    component_reason,
                )
            )
    availability_rows.extend(_pooled_availability(pooled_panel, support))
    availability = pd.DataFrame(availability_rows)
    for column in AVAILABILITY_COLUMNS:
        if column not in availability.columns:
            availability[column] = np.nan
    availability = availability[AVAILABILITY_COLUMNS]

    pretrend = _pretrend_rows(matching_root, pooled_root)
    selection = _selection_table(panel, pretrend, availability, downstream)
    validation = _validate(panel, availability, pretrend, downstream)
    return {
        "panel": panel,
        "availability": availability,
        "pretrend": pretrend,
        "selection": selection,
        "validation": validation,
        "rows_filtered": matching_removed + pooled_removed,
        "support": support,
    }


def export_event_study_final(
    matching_root: Path,
    pooled_root: Path,
    output_directory: Path,
    *,
    suite_root: Path | None = None,
    downstream: tuple[str, ...] = ("viirs", "poi"),
    make_zip: bool = False,
    zip_name: str = "mit_event_study_final.zip",
) -> dict[str, Any]:
    """Write the frozen release directory (and optionally a zip)."""
    collected = collect_event_study_release(
        matching_root, pooled_root, suite_root=suite_root, downstream=downstream
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    panel = collected["panel"]
    availability = normalize_treated_event_columns(collected["availability"])
    pretrend = collected["pretrend"]
    selection = normalize_treated_event_columns(collected["selection"])
    validation = collected["validation"]

    panel.to_csv(output_directory / "event_study_panel.csv", index=False, encoding="utf-8-sig")
    availability.to_csv(
        output_directory / "availability.csv", index=False, encoding="utf-8-sig"
    )
    pretrend.to_csv(output_directory / "pretrend_summary.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(
        output_directory / "family_selection.csv", index=False, encoding="utf-8-sig"
    )
    (output_directory / "family_selection.md").write_text(
        _selection_markdown(selection, downstream), encoding="utf-8"
    )
    (output_directory / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_directory / "README.md").write_text(
        _release_readme(), encoding="utf-8"
    )

    figures_directory = output_directory / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)
    _copy_matching_figures(matching_root, figures_directory)
    _copy_pooled_figures(suite_root, pooled_root, figures_directory)

    source_tables = output_directory / "source_tables"
    source_tables.mkdir(parents=True, exist_ok=True)
    input_hashes = _copy_source_tables(
        matching_root, pooled_root, suite_root, source_tables
    )
    for family in FAMILIES:
        for donor_scope in ("same_city", "all_city_standardized"):
            directory = find_matching_directory(matching_root, family, donor_scope)
            if directory is None:
                continue
            for filename in (
                "event_study_coefficients.csv",
                "diagnostics.csv",
                "parallel_trends_wald.csv",
                "parallel_trends_wald_city_cluster.csv",
                "pretrend_metadata.csv",
            ):
                path = directory / filename
                if path.is_file():
                    input_hashes.append(
                        {
                            "path": f"{directory.name}/{filename}",
                            "sha256": file_sha256(path),
                            "bytes": path.stat().st_size,
                        }
                    )
    for filename in (
        "event_study_series.csv",
        "diagnostics.csv",
        "pretrend_city_cluster.csv",
        "pretrend_grid_cluster.csv",
    ):
        path = pooled_root / filename
        if path.is_file():
            input_hashes.append(
                {
                    "path": f"{pooled_root.name}/{filename}",
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
            )

    output_files = [
        output_directory / "event_study_panel.csv",
        output_directory / "availability.csv",
        output_directory / "pretrend_summary.csv",
        output_directory / "family_selection.csv",
        output_directory / "family_selection.md",
        output_directory / "validation.json",
        output_directory / "README.md",
    ]
    manifest = _write_manifest(
        output_directory,
        matching_root=matching_root,
        pooled_root=pooled_root,
        suite_root=suite_root,
        downstream=downstream,
        panel=panel,
        availability=availability,
        pretrend=pretrend,
        selection=selection,
        input_hashes=input_hashes,
        output_files=output_files,
        rows_filtered=int(collected["rows_filtered"]),
    )
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    zip_path: Path | None = None
    if make_zip:
        zip_path = output_directory.parent / zip_name
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output_directory.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        arcname=(
                            Path(output_directory.name) / path.relative_to(output_directory)
                        ).as_posix(),
                    )
    return {
        "output_directory": str(output_directory),
        "zip": str(zip_path) if zip_path is not None else None,
        "manifest": manifest,
        "validation": validation,
    }


def _copy_matching_figures(matching_root: Path, figures_directory: Path) -> None:
    for family in FAMILIES:
        for donor_scope, suffix in (
            ("same_city", ""),
            ("all_city_standardized", "_cross"),
        ):
            directory = find_matching_directory(matching_root, family, donor_scope)
            if directory is None:
                continue
            target = figures_directory / f"matching_{family}{suffix}"
            for path in sorted(directory.glob("event_study_matching_python.*")):
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target / path.name)


def _copy_pooled_figures(
    suite_root: Path | None, pooled_root: Path, figures_directory: Path
) -> None:
    candidates = []
    if suite_root is not None:
        candidates.append(suite_root / "pooled_outcomes")
    candidates.append(pooled_root / "pooled_outcomes")
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        target = figures_directory / "pooled_outcomes"
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(candidate.iterdir()):
            if path.is_file():
                shutil.copy2(path, target / path.name)
        break


def _copy_source_tables(
    matching_root: Path,
    pooled_root: Path,
    suite_root: Path | None,
    source_tables: Path,
) -> list[dict[str, object]]:
    hashes: list[dict[str, object]] = []
    search_roots: list[Path] = []
    if suite_root is not None:
        search_roots.append(suite_root)
    search_roots.extend([matching_root, pooled_root])
    copied: set[str] = set()
    for filename in (
        "suite_status.csv",
        "event_time_support.csv",
        "event_time_coverage.csv",
        "sample_aggregation_coverage.csv",
    ):
        for root in search_roots:
            path = root / filename
            if path.is_file() and path.name not in copied:
                shutil.copy2(path, source_tables / path.name)
                copied.add(path.name)
                hashes.append(
                    {
                        "path": f"{root.name}/{filename}",
                        "sha256": file_sha256(path),
                        "bytes": path.stat().st_size,
                    }
                )
                break
    return hashes


def _release_readme() -> str:
    return "\n".join(
        [
            "# Four-family event-study final release",
            "",
            "Files:",
            "",
            "- `event_study_panel.csv`: unified long panel (family x method x donor scope",
            "  x outcome x event time) with estimates, grid/city-cluster SEs, CIs, p-values,",
            "  sample sizes and inference metadata (`se_type`, `p_value_method`,",
            "  `n_city_clusters`); filtered to the pre-registered canonical windows.",
            "- `availability.csv`: one row per family x method x donor scope with status",
            "  (`available` / `no_data`), reasons, stage-specific treated counts",
            "  (`availability_treated_events`, `event_study_max_n_treated`, `coverage` for",
            "  matching; `max_supported_treated_grids` for pooled), source ranges and",
            "  warnings (`insufficient_clusters`).",
            "- `pretrend_summary.csv`: joint pre-trend tests (Matching Wald by grid and",
            "  city cluster; pooled GSC/MC cluster-mean tests).",
            "- `family_selection.csv` / `family_selection.md`: two-stage sample counts",
            "  (`availability_treated_events` vs `event_study_max_n_treated`, with",
            "  `n_treated_gap` and `coverage`), signal stability, pre-trends and the",
            "  downstream selection record; downstream causal supervision uses VIIRS",
            "  and POI.",
            "- `manifest.json`: windows, methods, input/output hashes, row counts,",
            "  downstream selection and caveats.",
            "- `validation.json`: schema and window checks (errors block a clean run).",
            "- `figures/` and `source_tables/`: the underlying suite figures and audit",
            "  tables, copied unchanged.",
            "",
            "Caveats are stored verbatim in `manifest.json:caveats`.",
            "",
        ]
    )
