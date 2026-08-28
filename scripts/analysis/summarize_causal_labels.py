"""Summarize a released Response Artifact into the sample-run three deliverables.

Given a release directory from ``build_response_artifact.py`` (with
``--allow-partial`` for samples), this writes:

- success/failure breakdown per outcome family (quality grades, failure
  reasons grouped into research / data-truncation / execution-state / code classes);
- distribution statistics of ``causal_response_label`` per family (count,
  mean, median, quartiles, Tukey-outlier count, share of zeros/NA);
- three scope views of the label availability: same-city main table,
  cross-city companion table, merged table.

All outputs go to ``outputs/causal_labels/summary_<release_id>/``.

Usage:
    python scripts/analysis/summarize_causal_labels.py --response-release <release_dir>
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "causal_labels"

FAMILIES = ["housing", "viirs", "poi", "population"]

FAILURE_CLASSES: dict[str, str] = {
    "preonly_placebo_quality_gate_failed": "research",
    "same_city:preonly_placebo_quality_gate_failed": "research",
    "all_city_standardized:preonly_placebo_quality_gate_failed": "research",
    "fewer_than_1_complete_pre_treatment_families": "research",
    "no_complete_pre_treatment_families": "research",
    "family_no_observed_support": "research",
    "viirs_insufficient_clean_pre_periods_for_gsc": "data_truncation",
    "monthly_viirs_cache_unavailable": "data_truncation",
    "target_period_outcome_or_counterfactual_missing": "data_truncation",
    "task_skipped": "research",
    "task_not_terminal": "execution_state",
}


def classify_failure(reason: object) -> str:
    text = str(reason or "").strip().lower()
    if not text or text in {"nan", "none"}:
        return "none"
    for key, cls in FAILURE_CLASSES.items():
        if key.lower() in text:
            return cls
    if any(token in text for token in ("error", "exception", "traceback", "runtime")):
        return "code"
    if any(token in text for token in ("missing", "unavailable", "insufficient", "censor", "truncat")):
        return "data_truncation"
    return "research"


def scope_of(row: pd.Series) -> str:
    donor_scope = str(row.get("donor_scope", "") or "")
    if donor_scope == "all_city_standardized":
        return "cross_city"
    if donor_scope == "same_city":
        return "same_city"
    method = str(row.get("method", "") or "")
    if "all_city" in method:
        return "cross_city"
    return "same_city" if "same_city" in method else "unknown"


def as_bool(values: pd.Series) -> pd.Series:
    """Parse parquet/CSV booleans without treating the string ``False`` as true."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype("string").str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    )


def summarize_family(
    frame: pd.DataFrame,
) -> dict[str, object]:
    available_mask = as_bool(frame["label_available"])
    available = frame.loc[available_mask]
    raw_labels = pd.to_numeric(available["causal_response_label"], errors="coerce")
    labels = raw_labels.replace([np.inf, -np.inf], np.nan).dropna()
    nonfinite_available = int(raw_labels.notna().sum() - len(labels))
    q = labels.quantile([0.25, 0.5, 0.75])
    iqr = float(q[0.75] - q[0.25])
    lo, hi = float(q[0.25]) - 1.5 * iqr, float(q[0.75]) + 1.5 * iqr
    return {
        "tasks": int(len(frame)),
        "grids_with_any_label": int(available["treatment_order"].nunique()),
        "grids_without_any_label": int(
            frame["treatment_order"].nunique() - available["treatment_order"].nunique()
        ),
        "label_available_cells": int(available.shape[0]),
        "label_share": round(float(available.shape[0] / max(len(frame), 1)), 4),
        "n": int(len(labels)),
        "nonfinite_available_labels": nonfinite_available,
        "mean": round(float(labels.mean()), 6) if len(labels) else None,
        "median": round(float(labels.median()), 6) if len(labels) else None,
        "q25": round(float(q[0.25]), 6) if len(labels) else None,
        "q75": round(float(q[0.75]), 6) if len(labels) else None,
        "min": round(float(labels.min()), 6) if len(labels) else None,
        "max": round(float(labels.max()), 6) if len(labels) else None,
        "tukey_outliers": int(((labels < lo) | (labels > hi)).sum()) if len(labels) else 0,
        "zero_share": round(float((labels == 0).mean()), 4) if len(labels) else None,
        "grades": (
            frame["quality_grade"].value_counts(dropna=False).astype(int).to_dict()
            if "quality_grade" in frame.columns
            else {}
        ),
        "failure_reasons": (
            frame.loc[~available_mask, "failure_reason"]
            .astype(str)
            .value_counts()
            .head(15)
            .astype(int)
            .to_dict()
            if "failure_reason" in frame.columns
            else {}
        ),
        "failure_classes": (
            frame.loc[~available_mask]
            .assign(_cls=lambda df: df["failure_reason"].map(classify_failure))
            .groupby("_cls")
            .size()
            .astype(int)
            .to_dict()
            if "failure_reason" in frame.columns
            else {}
        ),
        "periods_available": (
            available.groupby("treatment_order")["event_time"].nunique().describe().to_dict()
            if len(available)
            else {}
        ),
    }


def _save_figure_bundle(fig: plt.Figure, base_path: Path) -> list[str]:
    """Save publication-friendly raster and vector versions of one figure."""
    paths = [base_path.with_suffix(ext) for ext in (".png", ".pdf", ".svg")]
    for path in paths:
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    return [str(path) for path in paths]


def render_summary_figures(
    artifact: pd.DataFrame,
    family_summary: dict[str, dict[str, object]],
    out_dir: Path,
) -> list[str]:
    """Render the two hand-off figures for label coverage and distributions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    available_mask = as_bool(artifact["label_available"])
    written: list[str] = []

    # Four native-scale boxplots make outliers and the median/IQR immediately
    # visible without forcing incomparable outcome families onto one axis.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    rng = np.random.default_rng(20260819)
    for index, family in enumerate(FAMILIES):
        axis = axes.flat[index]
        values = pd.to_numeric(
            artifact.loc[available_mask & artifact["outcome_family"].eq(family),
                         "causal_response_label"],
            errors="coerce",
        )
        values = values[np.isfinite(values)]
        if len(values):
            axis.boxplot(
                values.to_numpy(),
                vert=True,
                patch_artist=True,
                showfliers=True,
                widths=0.42,
                boxprops={"facecolor": "#DCEAF7", "edgecolor": "#1F4E79"},
                medianprops={"color": "#C0392B", "linewidth": 2},
                whiskerprops={"color": "#1F4E79"},
                capprops={"color": "#1F4E79"},
                flierprops={"marker": "o", "markersize": 3,
                            "markerfacecolor": "#C0392B", "alpha": 0.45},
            )
            # A bounded jitter layer shows sample density while avoiding a
            # slow rendering path when a release contains many event cells.
            plot_values = values.to_numpy()
            if len(plot_values) > 3000:
                plot_values = plot_values[np.linspace(0, len(plot_values) - 1, 3000).astype(int)]
            axis.scatter(
                rng.normal(1.0, 0.035, len(plot_values)),
                plot_values,
                s=5,
                color="#1F78B4",
                alpha=0.18,
                linewidths=0,
                zorder=1,
            )
            outliers = family_summary.get(family, {}).get("tukey_outliers", 0)
            axis.set_title(f"{family} (n={len(values):,}; Tukey outliers={outliers})")
            axis.set_xticks([])
            axis.axhline(0, color="#666666", linestyle="--", linewidth=0.8)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        else:
            axis.text(0.5, 0.5, "No available labels", ha="center", va="center")
            axis.set_title(family)
            axis.set_xticks([])
        axis.set_ylabel("Causal response label")
    fig.suptitle("Causal label distributions by outcome family", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    written.extend(_save_figure_bundle(fig, out_dir / "causal_label_distributions"))

    # Failure reasons are shown as exact top reasons, stacked by family. This
    # preserves the distinction between a research-quality gate and a data
    # truncation issue while keeping rare reasons from producing an unreadable
    # wall of bars.
    failure = artifact.loc[~available_mask].copy()
    if len(failure):
        reasons = failure["failure_reason"].astype("string").fillna("unknown")
        reasons = reasons.str.strip().replace({"": "unknown", "nan": "unknown"})
        reasons = reasons.str.slice(0, 80)
        failure["_failure_reason"] = reasons
        counts = failure["_failure_reason"].value_counts().head(12)
        table = pd.crosstab(failure["_failure_reason"], failure["outcome_family"])
        table = table.reindex(counts.index).fillna(0)
        table = table.reindex(columns=FAMILIES, fill_value=0)
        fig, axis = plt.subplots(figsize=(12, max(5, 0.42 * len(table) + 2)))
        table.iloc[::-1].plot(
            kind="barh",
            stacked=True,
            ax=axis,
            color=["#2C7FB8", "#F28E2B", "#59A14F", "#B07AA1"],
            edgecolor="white",
            linewidth=0.4,
        )
        axis.set_title("Top causal-label failure reasons")
        axis.set_xlabel("Failed label cells")
        axis.set_ylabel("Failure reason")
        axis.legend(title="Outcome family", loc="lower right", frameon=False)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        fig.tight_layout()
        written.extend(_save_figure_bundle(fig, out_dir / "causal_label_failure_reasons"))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-release", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--orders-file",
        type=Path,
        help="Optional CSV with treatment_order values; restricts the report to those grids.",
    )
    args = parser.parse_args()

    artifact_path = args.response_release / "response_artifact.parquet"
    if not artifact_path.exists():
        raise FileNotFoundError(f"No response_artifact.parquet under {args.response_release}")
    artifact = pd.read_parquet(artifact_path)
    if args.orders_file is not None:
        orders_frame = pd.read_csv(args.orders_file)
        if "treatment_order" not in orders_frame.columns:
            raise ValueError(f"Orders file lacks treatment_order: {args.orders_file}")
        orders = pd.to_numeric(orders_frame["treatment_order"], errors="coerce")
        if orders.isna().any() or orders.duplicated().any():
            raise ValueError("Orders file must contain unique integer treatment_order values")
        order_set = set(int(value) for value in orders)
        artifact = artifact.loc[artifact["treatment_order"].astype(int).isin(order_set)].copy()

    if "main_spec" not in artifact.columns and "donor_scope" in artifact.columns:
        artifact["main_spec"] = artifact["donor_scope"].eq("same_city").astype(int)

    release_id = args.response_release.name
    out_dir = args.output_root / f"summary_{release_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    family_summary: dict[str, dict[str, object]] = {}
    for family in FAMILIES:
        sub = artifact.loc[artifact["outcome_family"].eq(family)]
        if sub.empty:
            family_summary[family] = {"tasks": 0}
            continue
        family_summary[family] = summarize_family(sub)

    scope_rows: list[dict[str, object]] = []
    for family in FAMILIES:
        sub = artifact.loc[artifact["outcome_family"].eq(family)]
        for scope in ("same_city", "cross_city", "merged"):
            view = sub if scope == "merged" else sub.loc[sub.apply(scope_of, axis=1).eq(scope)]
            scope_rows.append(
                {
                    "outcome_family": family,
                    "scope": scope,
                    "tasks": int(len(view)),
                    "label_available_cells": int(as_bool(view["label_available"]).sum()),
                    "label_share": round(
                        float(as_bool(view["label_available"]).mean()), 4
                    ),
                    "grids_with_any_label": int(
                        view.loc[as_bool(view["label_available"]), "treatment_order"].nunique()
                    ),
                }
            )
    scope_table = pd.DataFrame(scope_rows)

    grid_available = (
        artifact.assign(_label_available=as_bool(artifact["label_available"]))
        .groupby("treatment_order", as_index=False)["_label_available"]
        .any()
    )
    failed_grid_orders = set(
        grid_available.loc[~grid_available["_label_available"], "treatment_order"]
    )
    failed_grid_rows = artifact.loc[artifact["treatment_order"].isin(failed_grid_orders)]
    grid_failure_reasons = {}
    if len(failed_grid_rows) and "failure_reason" in failed_grid_rows.columns:
        grid_failure_reasons = (
            failed_grid_rows["failure_reason"]
            .astype("string")
            .fillna("unknown")
            .replace({"": "unknown", "nan": "unknown"})
            .value_counts()
            .head(15)
            .astype(int)
            .to_dict()
        )

    report: dict[str, object] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "release_id": release_id,
        "artifact_rows": int(len(artifact)),
        "grids": int(artifact["treatment_order"].nunique()),
        "families": family_summary,
        "scope_views": scope_table.to_dict(orient="records"),
        "grid_generation": {
            "total_treated_grids": int(len(grid_available)),
            "grids_with_any_label": int(grid_available["_label_available"].sum()),
            "grids_without_any_label": int((~grid_available["_label_available"]).sum()),
            "failure_reasons_on_failed_grids": grid_failure_reasons,
        },
    }
    figure_paths = render_summary_figures(artifact, family_summary, out_dir / "figures")
    report["figures"] = figure_paths
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    scope_table.to_csv(out_dir / "scope_views.csv", index=False, encoding="utf-8-sig")
    family_table = pd.DataFrame.from_dict(
        {k: {"outcome_family": k, **v} for k, v in family_summary.items()},
        orient="index",
    )
    family_table.to_csv(out_dir / "family_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote summary to {out_dir}")
    print(scope_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
