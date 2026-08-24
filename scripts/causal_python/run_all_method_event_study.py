#!/usr/bin/env python3
"""Run the R-free Matching, GSC and MC pooled event-study suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from urban_intervention.causal.event_study import run_matching_event_study  # noqa: E402
from urban_intervention.causal.pooled_event_study import (  # noqa: E402
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
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    statuses: list[dict[str, object]] = []
    path_output = args.output_directory / "gsc_mc_paths"
    try:
        result = run_pooled_path_event_study(
            args.staging_root,
            path_output,
            specification_fingerprint=args.specification_fingerprint,
            frequency=args.frequency,
            min_pre_event_time=args.min_pre_event_time,
            latest_pre_periods=args.latest_pre_periods,
            figures=not args.no_figures,
        )
        statuses.append(
            {
                "component": "gsc_mc_paths",
                "status": "completed",
                "details": f"{result.diagnostics['treatment_orders']} treatments",
            }
        )
    except ValueError as error:
        statuses.append(
            {"component": "gsc_mc_paths", "status": "no_data", "details": str(error)}
        )

    allowed = {"housing", "viirs", "poi", "population"}
    families = [value.strip() for value in args.families.split(",") if value.strip()]
    unknown = set(families) - allowed
    if unknown:
        parser.error(f"unknown families: {sorted(unknown)}")
    for family in families:
        output = args.output_directory / "matching_twfe" / family
        try:
            result = run_matching_event_study(family, output)
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
    args.output_directory.mkdir(parents=True, exist_ok=True)
    status = pd.DataFrame(statuses)
    status.to_csv(args.output_directory / "suite_status.csv", index=False, encoding="utf-8-sig")
    completed = int(status["status"].eq("completed").sum())
    print(f"Completed {completed}/{len(status)} Python event-study components")
    return int(completed == 0)


if __name__ == "__main__":
    raise SystemExit(main())
