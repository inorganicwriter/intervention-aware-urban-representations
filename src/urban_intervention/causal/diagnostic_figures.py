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
MONTHLY_LABEL_HORIZONS = frozenset({1, 3, 6, 12, 18, 24})


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
    return [part for _, part in ordered.groupby(ordered["event_time"].diff().ne(1).cumsum())]


def _format_p(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return f"{number:.3f}" if np.isfinite(number) else "NA"


def _style_axis(axis: Any, *, family: str) -> None:
    monthly = family in {"housing", "viirs"}
    if monthly:
        axis.axvspan(-6.5, -0.5, color="#D9D9D9", alpha=0.18, linewidth=0)
        axis.set_xlim(-42.5, 24.5)
    else:
        axis.set_xlim(-4.5, 3.5)
    axis.axvspan(-0.5, 0.5, color="#E7B45A", alpha=0.16, linewidth=0)
    axis.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
    axis.axvline(0, color="#333333", linewidth=0.9)
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
    if family in {"housing", "viirs"}:
        event_time = pd.to_numeric(frame["event_time"], errors="coerce")
        frame = frame.loc[event_time.le(0) | event_time.isin(MONTHLY_LABEL_HORIZONS)].copy()
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
) -> object:
    tests = pooled.city_pretrend
    hit = tests.loc[
        tests["outcome_family"].eq(family)
        & tests["outcome"].eq(outcome)
        & tests["method"].eq(method)
        & tests["donor_scope"].eq(donor_scope)
    ]
    return hit["p_value"].iloc[0] if len(hit) else np.nan


def _save_bundle(figure: Any, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = [base.with_suffix(extension) for extension in (".png", ".pdf", ".svg", ".tiff")]
    for path in paths:
        dpi = 600 if path.suffix == ".tiff" else 300
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return paths


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
    for family in FAMILIES:
        outcome = PRIMARY_OUTCOMES[family]
        figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.05), sharey=True)
        limits: list[tuple[float, float]] = []
        matching = matching_results.get(family)
        axis = axes[0]
        _style_axis(axis, family=family)
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
                _matching_frame(matching, family=family),
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
            _style_axis(axis, family=family)
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
                    _pooled_frame(part),
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
        axes[0].set_ylabel("Causal response (transformed outcome scale)")
        figure.suptitle(f"{FAMILY_TITLES[family]} pooled event study", fontsize=11, y=0.97)
        figure.text(
            0.5,
            0.035,
            "For monthly outcomes, gray marks excluded anticipation months; amber marks "
            "the opening period (t=0, shown but excluded from the pre-trend test).\n"
            "Error bars are 95% city-cluster intervals when available.",
            ha="center",
            va="bottom",
            fontsize=7,
        )
        figure.subplots_adjust(left=0.10, right=0.99, bottom=0.25, top=0.80, wspace=0.08)
        written.extend(_save_bundle(figure, output_directory / f"{family}_pooled_event_study"))
        plt.close(figure)
    return written
