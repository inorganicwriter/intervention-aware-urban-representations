#!/usr/bin/env python3
"""Run the R-free matched-pair TWFE event study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from urban_intervention.causal.event_study import run_matching_event_study  # noqa: E402
from urban_intervention.data.paths import OUTPUT_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outcome_family", choices=("housing", "viirs", "poi", "population"))
    parser.add_argument("--min-pre", type=int)
    parser.add_argument("--max-post", type=int)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--donor-scope",
        choices=("same_city", "all_city_standardized"),
        default="same_city",
    )
    args = parser.parse_args()
    output = args.output_directory or (
        OUTPUT_DIR / "event_study" / "matching_python" / args.outcome_family
    )
    result = run_matching_event_study(
        args.outcome_family,
        output,
        min_pre=args.min_pre,
        max_post=args.max_post,
        donor_scope=args.donor_scope,
    )
    print(
        f"Python TWFE event study: {len(result.coefficients)} coefficients, "
        f"{result.diagnostics['treated_events']} treated events -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
