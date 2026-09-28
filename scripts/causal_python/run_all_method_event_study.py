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
    event_study_coverage_table,
    event_study_support_table,
    write_pooled_outcome_event_studies,
)
from urban_intervention.causal.event_study import (  # noqa: E402
    canonical_event_window,
    run_matching_event_study,
)
from urban_intervention.causal.pooled_event_study import (  # noqa: E402
    DEFAULT_MAIN_SPECIFICATION_FINGERPRINT,
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
    parser.add_argument(
        "--specification-fingerprint",
        default=DEFAULT_MAIN_SPECIFICATION_FINGERPRINT,
        help="Exact fingerprint or full-match regex for the main/robustness specification.",
    )
    parser.add_argument("--frequency", choices=("monthly", "annual"))
    parser.add_argument("--min-pre-event-time", type=int)
    parser.add_argument("--latest-pre-periods", type=int, default=5)
    parser.add_argument(
        "--reference-event-time",
        type=int,
        default=-1,
        help=(
            "Omitted event-time reference (main: -1 for every family; monthly "
            "robustness: -7, paired with Phase 2 --anticipation-months 6)."
        ),
    )
    parser.add_argument(
        "--observation-window",
        type=int,
        default=1,
        choices=range(1, 7),
        metavar="{1,...,6}",
        help=(
            "Monthly post-treatment observation-window width; the main specification "
            "uses window=1 for both housing and VIIRS."
        ),
    )
    parser.add_argument(
        "--transaction-count-threshold",
        type=int,
        default=1,
        help="Minimum monthly housing transaction count used by labels and TWFE.",
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        help=(
            "CSV with treatment_order and outcome_family; restrict every pooled "
            "event study to the exact independently sampled tasks."
        ),
    )
    parser.add_argument(
        "--matching-family-queue",
        type=Path,
        help="Optional isolated outcome-family queue used by Matching event studies.",
    )
    parser.add_argument(
        "--matching-control-queue",
        type=Path,
        help="Optional isolated frozen-control queue used by Matching event studies.",
    )
    parser.add_argument(
        "--matching-task-root",
        type=Path,
        help="Optional isolated Matching task artifact root.",
    )
    parser.add_argument(
        "--matching-donor-scopes",
        default="same_city,all_city_standardized",
        help=(
            "Comma-separated Matching donor scopes. same_city writes "
            "matching_twfe/<family>; all_city_standardized writes "
            "matching_twfe_cross/<family>. Each family is analyzed under every "
            "requested scope independently."
        ),
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--matching-window-preset",
        choices=("canonical", "full"),
        default="canonical",
        help=(
            "canonical: horizon-aligned pre-registered event windows for the "
            "Matching components; full: every available period."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Exit 0 when a component reports no_data (bounded tests/demos). "
            "Without this flag an incomplete GSC/MC or Matching component fails "
            "the process instead of only writing suite_status.csv."
        ),
    )
    args = parser.parse_args()
    if args.reference_event_time >= 0:
        parser.error("--reference-event-time must be negative")
    if args.transaction_count_threshold < 1:
        parser.error("--transaction-count-threshold must be positive")

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
    cross_matching_results = {}
    path_output = args.output_directory / "gsc_mc_paths"
    monthly_observation_windows = {"housing": args.observation_window, "viirs": 1}
    try:
        path_result = run_pooled_path_event_study(
            args.staging_root,
            path_output,
            specification_fingerprint=args.specification_fingerprint,
            frequency=args.frequency,
            min_pre_event_time=args.min_pre_event_time,
            latest_pre_periods=args.latest_pre_periods,
            observation_window=args.observation_window,
            monthly_observation_windows=monthly_observation_windows,
            figures=not args.no_figures,
            task_keys=task_keys,
            reference_event_time=args.reference_event_time,
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
    scopes = [value.strip() for value in args.matching_donor_scopes.split(",") if value.strip()]
    allowed_scopes = {"same_city", "all_city_standardized"}
    unknown_scopes = set(scopes) - allowed_scopes
    if unknown_scopes:
        parser.error(f"unknown matching donor scopes: {sorted(unknown_scopes)}")
    for scope in scopes:
        for family in families:
            if scope == "same_city":
                output = args.output_directory / "matching_twfe" / family
                component = f"matching_twfe:{family}"
            else:
                output = args.output_directory / "matching_twfe_cross" / family
                component = f"matching_twfe_cross:{family}"
            if args.matching_window_preset == "canonical":
                min_pre, max_post = canonical_event_window(family)
            else:
                min_pre, max_post = None, None
            try:
                result = run_matching_event_study(
                    family,
                    output,
                    donor_scope=scope,
                    min_pre=min_pre,
                    max_post=max_post,
                    pretrend_window=args.latest_pre_periods,
                    observation_window=(1 if family == "viirs" else args.observation_window),
                    treatment_orders=(
                        {order for order, task_family in task_keys if task_family == family}
                        if task_keys is not None
                        else None
                    ),
                    reference_event_time=args.reference_event_time,
                    family_queue=args.matching_family_queue,
                    control_queue=args.matching_control_queue,
                    task_root=args.matching_task_root,
                    transaction_count_threshold=args.transaction_count_threshold,
                    specification_fingerprint=args.specification_fingerprint,
                )
                if scope == "same_city":
                    matching_results[family] = result
                else:
                    cross_matching_results[family] = result
                statuses.append(
                    {
                        "component": component,
                        "status": "completed",
                        "details": f"{result.diagnostics['treated_events']} treatments",
                    }
                )
            except ValueError as error:
                statuses.append(
                    {
                        "component": component,
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
        cross_count = int(
            cross_matching_results[family].diagnostics["treated_events"]
            if family in cross_matching_results
            else 0
        )
        for method, count in (
            ("matching", matching_count),
            ("matching_cross", cross_count),
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
    if args.no_figures:
        statuses[-1]["status"] = "skipped_by_flag"
    args.output_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage_rows).to_csv(
        args.output_directory / "sample_aggregation_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    event_study_support_table(matching_results, path_result).to_csv(
        args.output_directory / "event_time_support.csv",
        index=False,
        encoding="utf-8-sig",
    )
    event_study_coverage_table(matching_results, path_result).to_csv(
        args.output_directory / "event_time_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    status = pd.DataFrame(statuses)
    status.to_csv(args.output_directory / "suite_status.csv", index=False, encoding="utf-8-sig")
    completed = int(status["status"].eq("completed").sum())
    print(f"Completed {completed}/{len(status)} Python event-study components")
    acceptable = status["status"].isin({"completed", "skipped_by_flag"})
    # A family with no cross-city matched labels is a valid outcome, not a
    # failure: families are analyzed independently, and the cross scope is
    # simply empty for that family.
    cross_no_data = status["component"].str.startswith("matching_twfe_cross:") & status[
        "status"
    ].eq("no_data")
    acceptable = acceptable | cross_no_data
    if not args.allow_partial and not acceptable.all():
        incomplete = status.loc[
            ~acceptable, ["component", "status"]
        ]
        details = ", ".join(
            f"{row.component}={row.status}" for row in incomplete.itertuples()
        )
        print(
            f"ERROR: incomplete event-study components: {details}; "
            "rerun with --allow-partial for bounded tests",
            file=sys.stderr,
        )
        return 1
    return int(completed == 0)


if __name__ == "__main__":
    raise SystemExit(main())
