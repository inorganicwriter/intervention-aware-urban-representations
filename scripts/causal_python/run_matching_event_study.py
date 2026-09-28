#!/usr/bin/env python3
"""Run the R-free matched-pair TWFE event study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from urban_intervention.causal.event_study import (  # noqa: E402
    canonical_event_window,
    run_matching_event_study,
)
from urban_intervention.data.paths import OUTPUT_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outcome_family", choices=("housing", "viirs", "poi", "population"))
    parser.add_argument("--min-pre", type=int)
    parser.add_argument("--max-post", type=int)
    parser.add_argument(
        "--window-preset",
        choices=("canonical", "full"),
        default="canonical",
        help=(
            "canonical: horizon-aligned pre-registered event window; "
            "full: every available period (not the main specification)."
        ),
    )
    parser.add_argument("--pretrend-window", type=int, default=5)
    parser.add_argument("--transaction-count-threshold", type=int, default=1)
    parser.add_argument(
        "--specification-fingerprint",
        help="Exact fingerprint or full-match regex used to admit production task manifests.",
    )
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
    min_pre, max_post = args.min_pre, args.max_post
    if args.window_preset == "canonical" and min_pre is None and max_post is None:
        min_pre, max_post = canonical_event_window(args.outcome_family)
    result = run_matching_event_study(
        args.outcome_family,
        output,
        min_pre=min_pre,
        max_post=max_post,
        donor_scope=args.donor_scope,
        pretrend_window=args.pretrend_window,
        transaction_count_threshold=args.transaction_count_threshold,
        specification_fingerprint=args.specification_fingerprint,
    )
    print(
        f"Python TWFE event study: {len(result.coefficients)} coefficients, "
        f"{result.diagnostics['treated_events']} treated events -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
