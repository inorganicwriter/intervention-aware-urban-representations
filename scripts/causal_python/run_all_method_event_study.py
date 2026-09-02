#!/usr/bin/env python3
"""Run the R-free Matching, GSC and MC pooled event-study suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from urban_intervention.causal.diagnostic_figures import (  # noqa: E402
    write_pooled_outcome_event_studies,
)
from urban_intervention.causal.event_study import run_matching_event_study  # noqa: E402
from urban_intervention.causal.pooled_event_study import (  # noqa: E402
    PooledPathEventStudyResult,
    run_pooled_path_event_study,
)
from urban_intervention.data.paths import OUTPUT_COMPLETE_STAGING_DIR, OUTPUT_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, default=OUTPUT_COMPLETE_STAGING_DIR)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIR / "event_study_python")
    parser.add_argument(
        "--families",
        default="housing,viirs,poi,population",
        help="Comma-separated Matching families.",
    )
    parser.add_argument("--specification-fingerprint")
    parser.add_argument("--frequency", choices=("monthly", "annual"))
    parser.add_argument("--min-pre-event-time", type=int)
    parser.add_argument("--latest-pre-periods", type=int, default=5)
    parser.add_argument(
        "--observation-window",
        type=int,
        default=3,
        choices=range(1, 7),
        metavar="{1,...,6}",
        help="Monthly post-treatment moving-window width used by production labels.",
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        help=(
            "CSV with treatment_order and outcome_family; restrict every pooled "
            "event study to the exact independently sampled tasks."
        ),
    )
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    task_keys: set[tuple[int, str]] | None = None
    if args.tasks_file is not None:
        tasks = pd.read_csv(args.tasks_file)
        required = {"treatment_order", "outcome_family"}
        missing = required - set(tasks.columns)
        if missing:
            parser.error(f"tasks file lacks columns: {sorted(missing)}")
        orders = pd.to_numeric(tasks["treatment_order"], errors="coerce")
        families_in_file = tasks["outcome_family"].astype(str)
        if orders.isna().any():
            parser.error("tasks file contains a non-numeric treatment_order")
        task_keys = set(zip(orders.astype(int), families_in_file, strict=False))
        if not task_keys or len(task_keys) != len(tasks):
            parser.error("tasks file is empty or contains duplicate task keys")

    statuses: list[dict[str, object]] = []
    path_result: PooledPathEventStudyResult | None = None
    matching_results = {}
    path_output = args.output_directory / "gsc_mc_paths"
    try:
        path_result = run_pooled_path_event_study(
            args.staging_root,
            path_output,
            specification_fingerprint=args.specification_fingerprint,
            frequency=args.frequency,
            min_pre_event_time=args.min_pre_event_time,
            latest_pre_periods=args.latest_pre_periods,
            observation_window=args.observation_window,
            figures=not args.no_figures,
            task_keys=task_keys,
        )
        statuses.append(
            {
                "component": "gsc_mc_paths",
                "status": "completed",
                "details": f"{path_result.diagnostics['treatment_orders']} treatments",
            }
        )
    except ValueError as error:
        statuses.append({"component": "gsc_mc_paths", "status": "no_data", "details": str(error)})

    allowed = {"housing", "viirs", "poi", "population"}
    families = [value.strip() for value in args.families.split(",") if value.strip()]
    unknown = set(families) - allowed
    if unknown:
        parser.error(f"unknown families: {sorted(unknown)}")
    for family in families:
        output = args.output_directory / "matching_twfe" / family
        try:
            result = run_matching_event_study(
                family,
                output,
                pretrend_window=args.latest_pre_periods,
                observation_window=args.observation_window,
                treatment_orders=(
                    {order for order, task_family in task_keys if task_family == family}
                    if task_keys is not None
                    else None
                ),
            )
            matching_results[family] = result
            statuses.append(
                {
                    "component": f"matching_twfe:{family}",
                    "status": "completed",
                    "details": f"{result.diagnostics['treated_events']} treatments",
                }
            )
        except ValueError as error:
            statuses.append(
                {
                    "component": f"matching_twfe:{family}",
                    "status": "no_data",
                    "details": str(error),
                }
            )
    combined = (
        []
        if args.no_figures
        else write_pooled_outcome_event_studies(
            matching_results,
            path_result,
            args.output_directory / "pooled_outcomes",
        )
    )
    coverage_rows: list[dict[str, object]] = []
    requested_by_family = (
        pd.Series(
            [family for _, family in task_keys],
            dtype="string",
        ).value_counts()
        if task_keys is not None
        else pd.Series(dtype="int64")
    )
    for family in families:
        matching_count = int(
            matching_results[family].diagnostics["treated_events"]
            if family in matching_results
            else 0
        )
        family_paths = (
            path_result.paths.loc[path_result.paths["outcome_family"].eq(family)]
            if path_result is not None
            else pd.DataFrame()
        )
        method_counts = (
            family_paths.assign(
                method_family=family_paths["method"].map(
                    lambda value: "gsc" if "gsynth" in str(value) else "mc"
                )
            )
            .groupby("method_family")["treatment_order"]
            .nunique()
            .to_dict()
            if not family_paths.empty
            else {}
        )
        requested = int(requested_by_family.get(family, 0)) if task_keys is not None else None
        for method, count in (
            ("matching", matching_count),
            ("gsc", int(method_counts.get("gsc", 0))),
            ("mc", int(method_counts.get("mc", 0))),
        ):
            coverage_rows.append(
                {
                    "outcome_family": family,
                    "method": method,
                    "treated_grids_admitted": count,
                    "tasks_requested_for_family": requested,
                    "aggregation": "all_admitted_sample_tasks_within_family_and_method",
                }
            )
    statuses.append(
        {
            "component": "pooled_outcome_figures",
            "status": "completed" if combined else "no_data",
            "details": f"{len(combined) // 4} outcome figures",
        }
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage_rows).to_csv(
        args.output_directory / "sample_aggregation_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    status = pd.DataFrame(statuses)
    status.to_csv(args.output_directory / "suite_status.csv", index=False, encoding="utf-8-sig")
    completed = int(status["status"].eq("completed").sum())
    print(f"Completed {completed}/{len(status)} Python event-study components")
    return int(completed == 0)


if __name__ == "__main__":
    raise SystemExit(main())
