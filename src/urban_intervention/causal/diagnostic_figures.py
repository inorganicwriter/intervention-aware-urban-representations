"""Publication-ready diagnostic figures for final causal-label releases."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from urban_intervention.causal.event_study import EventStudyResult
from urban_intervention.causal.pooled_event_study import PooledPathEventStudyResult

FAMILIES = ("housing", "viirs", "poi", "population")
PRIMARY_OUTCOMES = {
    "housing": "housing_log_price",
    "viirs": "viirs_avg_asinh",
    "poi": "poi_count_log",
    "population": "population_log",
}
FAMILY_TITLES = {
    "housing": "Housing price",
    "viirs": "Night-time light",
    "poi": "POI count",
    "population": "Population",
}
METHOD_TITLES = {"matching": "Matching TWFE", "gsc": "GSC", "mc": "Matrix completion"}
METHOD_COLORS = {"matching": "#3B6C8E", "gsc": "#C47A45", "mc": "#668F80"}


def _method_family(value: object) -> str | None:
    text = str(value).lower()
    if "gsynth" in text or "gsc" in text:
        return "gsc"
    if "matrix" in text or "athey" in text or text == "mc" or "_mc_" in text:
        return "mc"
    return None


def _segments(frame: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = frame.sort_values("event_time").reset_index(drop=True)
    if ordered.empty:
        return []
    difference = ordered["event_time"].diff()
    breaks = difference.ne(1)
    return [part for _, part in ordered.groupby(breaks.cumsum())]


def _format_p(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return f"{number:.3f}" if np.isfinite(number) else "NA"


def _style_axis(axis: Any, *, family: str, reference_event_time: int) -> None:
    monthly = family in {"housing", "viirs"}
    if monthly and reference_event_time < -1:
        axis.axvspan(
            reference_event_time + 0.5,
            -0.5,
            color="#D9D9D9",
            alpha=0.18,
            linewidth=0,
        )
    axis.axvspan(-0.5, 0.5, color="#E7B45A", alpha=0.16, linewidth=0)
    axis.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
    axis.axvline(0, color="#333333", linewidth=0.9)
    axis.axvline(reference_event_time, color="#888888", linewidth=0.7, linestyle=":")
    axis.grid(axis="y", color="#E3E3E3", linewidth=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xlabel(f"Event time ({'months' if monthly else 'years'})")


def _draw_path(
    axis: Any,
    frame: pd.DataFrame,
    *,
    color: str,
    label: str | None = None,
) -> tuple[float, float] | None:
    if frame.empty:
        return None
    ordered = frame.sort_values("event_time").reset_index(drop=True)
    estimate = pd.to_numeric(ordered["estimate"], errors="coerce")
    lower = pd.to_numeric(ordered["lower"], errors="coerce")
    upper = pd.to_numeric(ordered["upper"], errors="coerce")
    finite = np.isfinite(estimate) & np.isfinite(lower) & np.isfinite(upper)
    ordered = ordered.loc[finite].copy()
    if ordered.empty:
        return None
    estimate = estimate.loc[finite].to_numpy(dtype=float)
    lower = lower.loc[finite].to_numpy(dtype=float)
    upper = upper.loc[finite].to_numpy(dtype=float)
    axis.errorbar(
        ordered["event_time"],
        estimate,
        yerr=np.vstack([estimate - lower, upper - estimate]),
        fmt="o",
        markersize=3.4,
        capsize=1.8,
        linewidth=0.9,
        color=color,
        label=label,
    )
    for piece in _segments(ordered):
        if len(piece) > 1:
            axis.plot(
                piece["event_time"],
                piece["estimate"],
                color=color,
                linewidth=1.15,
            )
    return float(lower.min()), float(upper.max())


def _matching_frame(result: EventStudyResult, *, family: str) -> pd.DataFrame:
    frame = result.coefficients[
        ["event_time", "estimate", "confidence_lower_city", "confidence_upper_city"]
    ].rename(
        columns={
            "confidence_lower_city": "lower",
            "confidence_upper_city": "upper",
        }
    )
    reference = int(result.diagnostics["reference_event_time"])
    return pd.concat(
        [
            frame,
            pd.DataFrame(
                {"event_time": [reference], "estimate": [0.0], "lower": [0.0], "upper": [0.0]}
            ),
        ],
        ignore_index=True,
    )


def _pooled_frame(series: pd.DataFrame) -> pd.DataFrame:
    city_available = np.isfinite(
        pd.to_numeric(series["ci_lower_city_cluster"], errors="coerce")
    ) & np.isfinite(pd.to_numeric(series["ci_upper_city_cluster"], errors="coerce"))
    return pd.DataFrame(
        {
            "event_time": series["event_time"],
            "estimate": series["mean_label"],
            "lower": series["ci_lower_city_cluster"].where(city_available, series["ci_lower"]),
            "upper": series["ci_upper_city_cluster"].where(city_available, series["ci_upper"]),
        }
    )


def _pooled_pretrend_p(
    pooled: PooledPathEventStudyResult,
    *,
    family: str,
    outcome: str,
    method: str,
    donor_scope: str,
    observation_window: object = None,
    reference_event_time: object = None,
) -> object:
    tests = pooled.city_pretrend
    hit = tests.loc[
        tests["outcome_family"].eq(family)
        & tests["outcome"].eq(outcome)
        & tests["method"].eq(method)
        & tests["donor_scope"].eq(donor_scope)
    ]
    # Several observation windows can share one outcome/method/scope key; the
    # first row is not necessarily the window this figure is plotting.
    if observation_window is not None and "observation_window" in hit.columns:
        hit = hit.loc[hit["observation_window"].eq(observation_window)]
    if reference_event_time is not None and "reference_event_time" in hit.columns:
        hit = hit.loc[hit["reference_event_time"].eq(reference_event_time)]
    return hit["p_value"].iloc[0] if len(hit) else np.nan


def _save_bundle(figure: Any, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = [base.with_suffix(extension) for extension in (".png", ".pdf", ".svg", ".tiff")]
    for path in paths:
        dpi = 600 if path.suffix == ".tiff" else 300
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return paths


def _family_method_supports(
    matching_results: Mapping[str, EventStudyResult],
    pooled: PooledPathEventStudyResult | None,
    family: str,
    *,
    donor_scope: str,
) -> dict[str, set[int]]:
    """Return observed event times for each available estimator family."""
    supports: dict[str, set[int]] = {}
    matching = matching_results.get(family)
    if matching is not None:
        matching_frame = _matching_frame(matching, family=family)
        finite = np.isfinite(pd.to_numeric(matching_frame["estimate"], errors="coerce"))
        supports["matching"] = set(
            pd.to_numeric(matching_frame.loc[finite, "event_time"], errors="raise")
            .astype(int)
            .tolist()
        )
    if pooled is None or pooled.series.empty:
        return supports
    outcome = PRIMARY_OUTCOMES[family]
    candidates = pooled.series.loc[
        pooled.series["outcome_family"].eq(family)
        & pooled.series["outcome"].eq(outcome)
        & pooled.series["donor_scope"].eq(donor_scope)
    ].copy()
    candidates["method_family"] = candidates["method"].map(_method_family)
    for method_family, part in candidates.groupby("method_family", dropna=True, sort=True):
        finite = np.isfinite(pd.to_numeric(part["mean_label"], errors="coerce"))
        supports[str(method_family)] = set(
            pd.to_numeric(part.loc[finite, "event_time"], errors="raise").astype(int).tolist()
        )
    return supports


def _longest_common_bounds(supports: Mapping[str, set[int]]) -> dict[str, int | None]:
    """Find the longest pre/post interval covered by every available method."""
    result: dict[str, int | None] = {
        "common_pre_min": None,
        "common_pre_max": None,
        "common_post_min": None,
        "common_post_max": None,
    }
    pre = [sorted(value for value in times if value < 0) for times in supports.values()]
    if pre and all(pre):
        lower = max(values[0] for values in pre)
        upper = min(values[-1] for values in pre)
        if lower <= upper:
            result["common_pre_min"] = lower
            result["common_pre_max"] = upper
    post = [sorted(value for value in times if value > 0) for times in supports.values()]
    if post and all(post):
        lower = max(values[0] for values in post)
        upper = min(values[-1] for values in post)
        if lower <= upper:
            result["common_post_min"] = lower
            result["common_post_max"] = upper
    return result


def event_study_support_table(
    matching_results: Mapping[str, EventStudyResult],
    pooled: PooledPathEventStudyResult | None,
    *,
    donor_scope: str = "same_city",
) -> pd.DataFrame:
    """Describe method-specific and longest common event-time support."""
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        supports = _family_method_supports(
            matching_results, pooled, family, donor_scope=donor_scope
        )
        common = _longest_common_bounds(supports)
        for method in ("matching", "gsc", "mc"):
            times = supports.get(method, set())
            pre = [value for value in times if value < 0]
            post = [value for value in times if value > 0]
            rows.append(
                {
                    "outcome_family": family,
                    "method": method,
                    "available": bool(times),
                    "method_pre_min": min(pre) if pre else None,
                    "method_pre_max": max(pre) if pre else None,
                    "method_post_min": min(post) if post else None,
                    "method_post_max": max(post) if post else None,
                    "t0_available": 0 in times,
                    **common,
                }
            )
    return pd.DataFrame(rows)


def event_study_coverage_table(
    matching_results: Mapping[str, EventStudyResult],
    pooled: PooledPathEventStudyResult | None,
    *,
    donor_scope: str = "same_city",
) -> pd.DataFrame:
    """Report changing treated-grid support at every plotted event time."""
    rows: list[dict[str, object]] = []
    support = event_study_support_table(
        matching_results, pooled, donor_scope=donor_scope
    )
    for family, matching_result in matching_results.items():
        frame = _matching_frame(matching_result, family=family)
        counts = (
            matching_result.coefficients.set_index("event_time")["n_treated"].to_dict()
            if "n_treated" in matching_result.coefficients
            else {}
        )
        reference = int(matching_result.diagnostics["reference_event_time"])
        for event_time in sorted(set(frame["event_time"].astype(int))):
            rows.append(
                {
                    "outcome_family": family,
                    "method": "matching",
                    "event_time": event_time,
                    "treated_grids": int(
                        counts.get(event_time, matching_result.diagnostics["reference_treated"])
                        if event_time == reference
                        else counts.get(event_time, 0)
                    ),
                }
            )
    if pooled is not None and not pooled.paths.empty:
        paths = pooled.paths.loc[
            pooled.paths["donor_scope"].eq(donor_scope)
            & pooled.paths["label_available"].fillna(False)
        ].copy()
        paths["method"] = paths["method"].map(_method_family)
        paths = paths.loc[paths["method"].notna()]
        grouped = (
            paths.groupby(["outcome_family", "method", "event_time"], sort=True)
            ["treatment_order"]
            .nunique()
        )
        for (family, method, event_time), count in grouped.items():
            rows.append(
                {
                    "outcome_family": str(family),
                    "method": str(method),
                    "event_time": int(event_time),
                    "treated_grids": int(count),
                }
            )
    coverage = pd.DataFrame(rows)
    if coverage.empty:
        return pd.DataFrame(
            columns=[
                "outcome_family",
                "method",
                "event_time",
                "treated_grids",
                "included_in_comparison_figure",
            ]
        )
    bounds = support.drop_duplicates("outcome_family").set_index("outcome_family")

    def included(row: pd.Series) -> bool:
        family_bounds = bounds.loc[str(row["outcome_family"])]
        event_time = int(row["event_time"])
        if event_time == 0:
            return True
        side = "pre" if event_time < 0 else "post"
        lower = family_bounds[f"common_{side}_min"]
        upper = family_bounds[f"common_{side}_max"]
        return bool(pd.notna(lower) and pd.notna(upper) and lower <= event_time <= upper)

    coverage["included_in_comparison_figure"] = coverage.apply(included, axis=1)
    return coverage.sort_values(["outcome_family", "method", "event_time"]).reset_index(
        drop=True
    )


def _restrict_to_bounds(
    frame: pd.DataFrame,
    bounds: Mapping[str, int | None],
    *,
    keep_t0: bool,
) -> pd.DataFrame:
    event_time = pd.to_numeric(frame["event_time"], errors="coerce")
    selected = pd.Series(False, index=frame.index)
    pre_min = bounds["common_pre_min"]
    pre_max = bounds["common_pre_max"]
    if pre_min is not None and pre_max is not None:
        selected |= event_time.between(pre_min, pre_max)
    post_min = bounds["common_post_min"]
    post_max = bounds["common_post_max"]
    if post_min is not None and post_max is not None:
        selected |= event_time.between(post_min, post_max)
    if keep_t0:
        selected |= event_time.eq(0)
    return frame.loc[selected].copy()


def write_pooled_outcome_event_studies(
    matching_results: Mapping[str, EventStudyResult],
    pooled: PooledPathEventStudyResult | None,
    output_directory: Path,
    *,
    donor_scope: str = "same_city",
) -> list[Path]:
    """Write one three-method event-study figure for each primary outcome family."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
        }
    )
    written: list[Path] = []
    support_table = event_study_support_table(
        matching_results, pooled, donor_scope=donor_scope
    )
    reference_candidates = {
        int(result.diagnostics["reference_event_time"])
        for result in matching_results.values()
    }
    if pooled is not None and not pooled.paths.empty:
        reference_candidates.update(
            pd.to_numeric(
                pooled.paths["reference_event_time"], errors="raise"
            ).astype(int)
        )
    if len(reference_candidates) > 1:
        raise ValueError("pooled comparison mixes different event-study references")
    reference_event_time = next(iter(reference_candidates), -1)
    for family in FAMILIES:
        outcome = PRIMARY_OUTCOMES[family]
        family_support = support_table.loc[support_table["outcome_family"].eq(family)]
        common = {
            column: (
                None
                if family_support.empty or pd.isna(family_support[column].iloc[0])
                else int(family_support[column].iloc[0])
            )
            for column in (
                "common_pre_min",
                "common_pre_max",
                "common_post_min",
                "common_post_max",
            )
        }
        figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.05), sharey=True)
        limits: list[tuple[float, float]] = []
        matching = matching_results.get(family)
        axis = axes[0]
        _style_axis(
            axis, family=family, reference_event_time=reference_event_time
        )
        axis.set_title(METHOD_TITLES["matching"])
        if matching is None:
            axis.text(
                0.5,
                0.5,
                "No production labels",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        else:
            limit = _draw_path(
                axis,
                _restrict_to_bounds(
                    _matching_frame(matching, family=family), common, keep_t0=True
                ),
                color=METHOD_COLORS["matching"],
            )
            if limit:
                limits.append(limit)
            p_value = matching.city_cluster_pretrend["p_value"].iloc[0]
            axis.text(
                0.02,
                0.98,
                "Joint clean-lead test\n"
                f"city p={_format_p(p_value)}; G={matching.diagnostics['city_clusters']}",
                transform=axis.transAxes,
                va="top",
                fontsize=7,
            )

        for panel_index, method_family in enumerate(("gsc", "mc"), start=1):
            axis = axes[panel_index]
            _style_axis(
                axis, family=family, reference_event_time=reference_event_time
            )
            axis.set_title(METHOD_TITLES[method_family])
            if pooled is None or pooled.series.empty:
                axis.text(
                    0.5,
                    0.5,
                    "No production labels",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                )
                continue
            candidates = pooled.series.loc[
                pooled.series["outcome_family"].eq(family)
                & pooled.series["outcome"].eq(outcome)
                & pooled.series["donor_scope"].eq(donor_scope)
                & pooled.series["method"].map(_method_family).eq(method_family)
            ]
            if candidates.empty:
                axis.text(
                    0.5,
                    0.5,
                    "No production labels",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                )
                continue
            for method, part in candidates.groupby("method", sort=True):
                limit = _draw_path(
                    axis,
                    _restrict_to_bounds(_pooled_frame(part), common, keep_t0=True),
                    color=METHOD_COLORS[method_family],
                    label=str(method) if candidates["method"].nunique() > 1 else None,
                )
                if limit:
                    limits.append(limit)
                p_value = _pooled_pretrend_p(
                    pooled,
                    family=family,
                    outcome=outcome,
                    method=str(method),
                    donor_scope=donor_scope,
                    observation_window=(
                        part["observation_window"].iloc[0]
                        if "observation_window" in part.columns and len(part)
                        else None
                    ),
                    reference_event_time=(
                        part["reference_event_time"].iloc[0]
                        if "reference_event_time" in part.columns and len(part)
                        else None
                    ),
                )
                city_count = int(pd.to_numeric(part["n_cities"], errors="coerce").max())
                axis.text(
                    0.02,
                    0.98,
                    f"Mean clean-pre gap test\ncity p={_format_p(p_value)}; G≤{city_count}",
                    transform=axis.transAxes,
                    va="top",
                    fontsize=7,
                )
            if candidates["method"].nunique() > 1:
                axis.legend(fontsize=6, loc="best")

        if limits:
            lower = min(value[0] for value in limits)
            upper = max(value[1] for value in limits)
            span = max(upper - lower, 1e-9)
            for axis in axes:
                axis.set_ylim(lower - 0.08 * span, upper + 0.08 * span)
        x_values = [0]
        for key in ("common_pre_min", "common_pre_max", "common_post_min", "common_post_max"):
            value = common[key]
            if value is not None:
                x_values.append(int(value))
        x_min, x_max = min(x_values), max(x_values)
        padding = max((x_max - x_min) * 0.02, 0.5)
        for axis in axes:
            axis.set_xlim(x_min - padding, x_max + padding)
        axes[0].set_ylabel("Causal response (transformed outcome scale)")
        figure.suptitle(f"{FAMILY_TITLES[family]} pooled event study", fontsize=11, y=0.97)
        figure.text(
            0.5,
            0.035,
            "For monthly robustness results, gray marks anticipation months when present; "
            "amber marks "
            "the opening period (t=0, shown but excluded from the pre-trend test).\n"
            "Each outcome uses its longest overlap across available methods; anticipation is "
            "shown but excluded from the clean-lead test. Error bars are 95% city-cluster "
            "intervals when available.",
            ha="center",
            va="bottom",
            fontsize=7,
        )
        figure.subplots_adjust(left=0.10, right=0.99, bottom=0.25, top=0.80, wspace=0.08)
        written.extend(_save_bundle(figure, output_directory / f"{family}_pooled_event_study"))
        plt.close(figure)
    return written
