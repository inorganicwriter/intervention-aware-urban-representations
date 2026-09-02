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
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
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
GENERATION_STATES = ("complete", "partial", "failed")

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
    donor_scope_value = row.get("donor_scope", "")
    donor_scope = "" if pd.isna(donor_scope_value) else str(donor_scope_value)
    if donor_scope == "all_city_standardized":
        return "cross_city"
    if donor_scope == "same_city":
        return "same_city"
    method_value = row.get("method", "")
    method = "" if pd.isna(method_value) else str(method_value)
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


def primary_label_rows(artifact: pd.DataFrame) -> pd.DataFrame:
    """Keep the four primary outcomes without mixing auxiliary POI scales."""
    selected = [
        artifact.loc[
            artifact["outcome_family"].eq(family)
            & artifact["outcome"].eq(PRIMARY_OUTCOMES[family])
        ]
        for family in FAMILIES
    ]
    return pd.concat(selected, ignore_index=True) if selected else artifact.iloc[0:0].copy()


def analysis_label_mask(frame: pd.DataFrame) -> pd.Series:
    """Select released labels eligible for quantitative summaries."""
    available = as_bool(frame["label_available"])
    if "production_eligible" in frame.columns:
        available &= as_bool(frame["production_eligible"])
    return available


def _clean_reason(values: pd.Series) -> str | None:
    cleaned = (
        values.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "<NA>": pd.NA})
    )
    cleaned = cleaned.loc[cleaned.notna()]
    if cleaned.empty:
        return None
    return str(cleaned.value_counts().index[0])


def build_generation_tables(
    primary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize one generation outcome per treated-grid and family task."""
    task_rows: list[dict[str, object]] = []
    for (order, family), part in primary.groupby(
        ["treatment_order", "outcome_family"], sort=True, dropna=False
    ):
        available = analysis_label_mask(part)
        expected = len(part)
        obtained = int(available.sum())
        state = "complete" if obtained == expected and expected else "partial" if obtained else "failed"
        reason = None
        for column in ("task_failure_reason", "failure_reason"):
            if column in part.columns:
                reason = _clean_reason(part.loc[~available, column])
                if reason:
                    break
        if reason is None and state == "partial":
            reason = "partial_target_path_missing_periods"
        if reason is None and state == "failed":
            reason = "no_primary_outcome_label"
        task_rows.append(
            {
                "treatment_order": int(order),
                "outcome_family": str(family),
                "expected_label_cells": expected,
                "available_label_cells": obtained,
                "generation_state": state,
                "failure_reason": reason,
            }
        )
    tasks = pd.DataFrame(task_rows)
    if tasks.empty:
        generation = pd.DataFrame(
            columns=["outcome_family", "generation_state", "treated_grids", "total_grids", "share"]
        )
        failures = pd.DataFrame(columns=["outcome_family", "failure_reason", "treated_grids"])
        return tasks, generation, failures
    generation = (
        tasks.groupby(["outcome_family", "generation_state"], sort=True)
        .size()
        .rename("treated_grids")
        .reset_index()
    )
    complete_index = pd.MultiIndex.from_product(
        [FAMILIES, GENERATION_STATES], names=["outcome_family", "generation_state"]
    )
    generation = (
        generation.set_index(["outcome_family", "generation_state"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )
    totals = tasks.groupby("outcome_family").size().rename("total_grids")
    generation = generation.merge(totals, on="outcome_family", how="left")
    generation["total_grids"] = generation["total_grids"].fillna(0).astype(int)
    generation["share"] = np.divide(
        generation["treated_grids"],
        generation["total_grids"],
        out=np.zeros(len(generation), dtype=float),
        where=generation["total_grids"].to_numpy() > 0,
    )
    failures = (
        tasks.loc[tasks["generation_state"].ne("complete")]
        .groupby(["outcome_family", "failure_reason"], dropna=False, sort=True)
        .size()
        .rename("treated_grids")
        .reset_index()
        .sort_values(["treated_grids", "failure_reason"], ascending=[False, True])
    )
    return tasks, generation, failures


def build_distribution_table(primary: pd.DataFrame) -> pd.DataFrame:
    """Compute native-scale distribution diagnostics for the four primary labels."""
    rows: list[dict[str, object]] = []
    selected = primary.loc[analysis_label_mask(primary)].copy()
    selected["causal_response_label"] = pd.to_numeric(
        selected["causal_response_label"], errors="coerce"
    )
    selected = selected.loc[np.isfinite(selected["causal_response_label"])]
    for family in FAMILIES:
        part = selected.loc[selected["outcome_family"].eq(family)]
        values = part["causal_response_label"]
        if values.empty:
            rows.append({"outcome_family": family, "outcome": PRIMARY_OUTCOMES[family], "n": 0})
            continue
        q25, median, q75 = (float(value) for value in values.quantile([0.25, 0.5, 0.75]))
        iqr = q75 - q25
        lower_fence, upper_fence = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        outliers = (values < lower_fence) | (values > upper_fence)
        rows.append(
            {
                "outcome_family": family,
                "outcome": PRIMARY_OUTCOMES[family],
                "n": int(len(values)),
                "treated_grids": int(part["treatment_order"].nunique()),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "min": float(values.min()),
                "q25": q25,
                "median": median,
                "q75": q75,
                "max": float(values.max()),
                "iqr": iqr,
                "lower_tukey_fence": lower_fence,
                "upper_tukey_fence": upper_fence,
                "tukey_outliers": int(outliers.sum()),
                "tukey_outlier_share": float(outliers.mean()),
            }
        )
    return pd.DataFrame(rows)


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
        "tasks": int(frame[["treatment_order", "outcome_family"]].drop_duplicates().shape[0]),
        "label_cells": int(len(frame)),
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
    paths = [base_path.with_suffix(ext) for ext in (".png", ".pdf", ".svg", ".tiff")]
    for path in paths:
        dpi = 600 if path.suffix == ".tiff" else 300
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    return [str(path) for path in paths]


def render_summary_figures(
    artifact: pd.DataFrame,
    generation_summary: pd.DataFrame,
    failure_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    out_dir: Path,
) -> list[str]:
    """Render generation, failure-reason and primary-label distribution figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = primary_label_rows(artifact)
    available_mask = analysis_label_mask(primary)
    written: list[str] = []

    # One task-level state per treated grid and family prevents missing horizons
    # from being counted repeatedly as separate failed treatments.
    coverage = (
        generation_summary.pivot(
            index="outcome_family", columns="generation_state", values="treated_grids"
        )
        .reindex(index=FAMILIES, columns=GENERATION_STATES, fill_value=0)
        .fillna(0)
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    bottom = np.zeros(len(coverage), dtype=float)
    state_colors = {"complete": "#4E8098", "partial": "#E1A65A", "failed": "#B75D69"}
    for state in GENERATION_STATES:
        values = coverage[state].to_numpy(dtype=float)
        bars = axis.bar(
            np.arange(len(coverage)),
            values,
            bottom=bottom,
            color=state_colors[state],
            edgecolor="white",
            linewidth=0.6,
            label=state.capitalize(),
        )
        for bar, count, base in zip(bars, values, bottom, strict=True):
            if count > 0:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + count / 2,
                    f"{int(count):,}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if state != "partial" else "#333333",
                )
        bottom += values
    axis.set_xticks(np.arange(len(coverage)), [FAMILY_TITLES[value] for value in coverage.index])
    axis.set_ylabel("Treated grids")
    axis.set_title("Causal-label generation by primary outcome")
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axis.grid(axis="y", color="#E3E3E3", linewidth=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    written.extend(_save_figure_bundle(fig, out_dir / "causal_label_generation"))

    # Four native-scale boxplots keep transformed outcomes on separate axes.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), squeeze=False)
    for index, family in enumerate(FAMILIES):
        axis = axes.flat[index]
        values = pd.to_numeric(
            primary.loc[
                available_mask & primary["outcome_family"].eq(family),
                "causal_response_label",
            ],
            errors="coerce",
        )
        values = values[np.isfinite(values)]
        if len(values):
            axis.boxplot(
                values.to_numpy(),
                orientation="horizontal",
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
                plot_values,
                1.0 + 0.035 * np.sin(np.arange(len(plot_values)) * 2.399963),
                s=5,
                color="#1F78B4",
                alpha=0.18,
                linewidths=0,
                zorder=1,
            )
            stats = distribution_summary.loc[
                distribution_summary["outcome_family"].eq(family)
            ].iloc[0]
            axis.scatter(
                [stats["mean"]],
                [1.0],
                marker="D",
                s=30,
                color="#C47A45",
                zorder=4,
                label="Mean",
            )
            axis.axvline(0, color="#666666", linestyle="--", linewidth=0.8)
            axis.set_title(
                f"{FAMILY_TITLES[family]}  "
                f"(n={len(values):,}; outliers={int(stats['tukey_outliers']):,})"
            )
            axis.text(
                0.01,
                0.04,
                f"Q1={stats['q25']:.3g}   median={stats['median']:.3g}   "
                f"Q3={stats['q75']:.3g}   mean={stats['mean']:.3g}",
                transform=axis.transAxes,
                fontsize=8,
            )
            axis.set_yticks([])
            axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
            axis.legend(loc="upper right", fontsize=7)
        else:
            axis.text(0.5, 0.5, "No available labels", ha="center", va="center")
            axis.set_title(FAMILY_TITLES[family])
            axis.set_yticks([])
        axis.set_xlabel("Causal response label (native transformed scale)")
        axis.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Primary causal-label distributions", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    written.extend(_save_figure_bundle(fig, out_dir / "causal_label_distributions"))

    # Failure reasons are counted once per treated-grid/family task.
    if len(failure_summary):
        top_reasons = (
            failure_summary.groupby("failure_reason")["treated_grids"]
            .sum()
            .nlargest(12)
            .index
        )
        selected = failure_summary.loc[failure_summary["failure_reason"].isin(top_reasons)]
        table = (
            selected.pivot_table(
                index="failure_reason",
                columns="outcome_family",
                values="treated_grids",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(index=top_reasons, columns=FAMILIES, fill_value=0)
        )
        table.index = [textwrap.fill(str(value).replace("_", " "), width=42) for value in table.index]
        fig, axis = plt.subplots(figsize=(7.2, max(4.5, 0.42 * len(table) + 1.7)))
        table.iloc[::-1].plot(
            kind="barh",
            stacked=True,
            ax=axis,
            color=["#2C7FB8", "#F28E2B", "#59A14F", "#B07AA1"],
            edgecolor="white",
            linewidth=0.4,
        )
        axis.set_title("Top causal-label failure reasons")
        axis.set_xlabel("Affected treated-grid × outcome-family tasks")
        axis.set_ylabel("Failure reason")
        axis.legend(title="Outcome family", loc="lower right", frameon=False)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    else:
        fig, axis = plt.subplots(figsize=(7.2, 3.8))
        axis.text(0.5, 0.5, "No failed or partial primary-label tasks", ha="center", va="center")
        axis.set_axis_off()
        axis.set_title("Causal-label failure reasons")
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
    parser.add_argument(
        "--tasks-file",
        type=Path,
        help=(
            "Optional CSV with treatment_order and outcome_family; restricts the "
            "report to those exact independently sampled tasks."
        ),
    )
    args = parser.parse_args()
    if args.orders_file is not None and args.tasks_file is not None:
        parser.error("--orders-file and --tasks-file are mutually exclusive")

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
    elif args.tasks_file is not None:
        tasks = pd.read_csv(args.tasks_file)
        required = {"treatment_order", "outcome_family"}
        missing = required - set(tasks.columns)
        if missing:
            raise ValueError(f"Tasks file lacks columns: {sorted(missing)}")
        orders = pd.to_numeric(tasks["treatment_order"], errors="coerce")
        families = tasks["outcome_family"].astype(str)
        keys = pd.MultiIndex.from_arrays([orders, families])
        if orders.isna().any() or keys.duplicated().any() or len(keys) == 0:
            raise ValueError(
                "Tasks file must contain unique treatment_order/outcome_family keys"
            )
        artifact_keys = pd.MultiIndex.from_arrays(
            [
                artifact["treatment_order"].astype(int),
                artifact["outcome_family"].astype(str),
            ]
        )
        artifact = artifact.loc[artifact_keys.isin(keys)].copy()

    if "main_spec" not in artifact.columns and "donor_scope" in artifact.columns:
        artifact["main_spec"] = artifact["donor_scope"].eq("same_city").astype(int)

    release_id = args.response_release.name
    out_dir = args.output_root / f"summary_{release_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    primary = primary_label_rows(artifact)
    family_summary: dict[str, dict[str, object]] = {}
    for family in FAMILIES:
        sub = primary.loc[primary["outcome_family"].eq(family)]
        if sub.empty:
            family_summary[family] = {"tasks": 0}
            continue
        family_summary[family] = summarize_family(sub)
    generation_tasks, generation_summary, failure_summary = build_generation_tables(primary)
    distribution_summary = build_distribution_table(primary)

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
        "primary_outcomes": PRIMARY_OUTCOMES,
        "primary_generation": generation_summary.to_dict(orient="records"),
        "primary_distributions": distribution_summary.to_dict(orient="records"),
    }
    figure_paths = render_summary_figures(
        artifact,
        generation_summary,
        failure_summary,
        distribution_summary,
        out_dir / "figures",
    )
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
    generation_tasks.to_csv(
        out_dir / "label_generation_by_task.csv", index=False, encoding="utf-8-sig"
    )
    generation_summary.to_csv(
        out_dir / "label_generation_by_family.csv", index=False, encoding="utf-8-sig"
    )
    failure_summary.to_csv(
        out_dir / "failure_reasons_by_family.csv", index=False, encoding="utf-8-sig"
    )
    distribution_summary.to_csv(
        out_dir / "label_distribution_by_family.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Wrote summary to {out_dir}")
    print(scope_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
