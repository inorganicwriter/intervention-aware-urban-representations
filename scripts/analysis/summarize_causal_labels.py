"""Summarize a released Response Artifact into the sample-run three deliverables.

Given a release directory from ``build_response_artifact.py`` (with
``--allow-partial`` for samples), this writes:

- success/failure breakdown per outcome family (quality grades, failure
  reasons grouped into research / data-truncation / code classes);
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


def summarize_family(
    frame: pd.DataFrame,
) -> dict[str, object]:
    available = frame.loc[frame["label_available"].astype(bool)]
    labels = pd.to_numeric(available["causal_response_label"], errors="coerce").dropna()
    q = labels.quantile([0.25, 0.5, 0.75])
    iqr = float(q[0.75] - q[0.25])
    lo, hi = float(q[0.25]) - 1.5 * iqr, float(q[0.75]) + 1.5 * iqr
    return {
        "tasks": int(len(frame)),
        "grids_with_any_label": int(available["treatment_order"].nunique()),
        "label_available_cells": int(available.shape[0]),
        "label_share": round(float(available.shape[0] / max(len(frame), 1)), 4),
        "n": int(len(labels)),
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
            frame.loc[~frame["label_available"].astype(bool), "failure_reason"]
            .astype(str)
            .value_counts()
            .head(15)
            .astype(int)
            .to_dict()
            if "failure_reason" in frame.columns
            else {}
        ),
        "failure_classes": (
            frame.loc[~frame["label_available"].astype(bool)]
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-release", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    artifact_path = args.response_release / "response_artifact.parquet"
    if not artifact_path.exists():
        raise FileNotFoundError(f"No response_artifact.parquet under {args.response_release}")
    artifact = pd.read_parquet(artifact_path)

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
                    "label_available_cells": int(view["label_available"].astype(bool).sum()),
                    "label_share": round(
                        float(view["label_available"].astype(bool).mean()), 4
                    ),
                    "grids_with_any_label": int(
                        view.loc[view["label_available"].astype(bool), "treatment_order"].nunique()
                    ),
                }
            )
    scope_table = pd.DataFrame(scope_rows)

    report: dict[str, object] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "release_id": release_id,
        "artifact_rows": int(len(artifact)),
        "grids": int(artifact["treatment_order"].nunique()),
        "families": family_summary,
        "scope_views": scope_table.to_dict(orient="records"),
    }
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
