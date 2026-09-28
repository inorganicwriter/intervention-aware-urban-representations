#!/usr/bin/env python3
"""Export the frozen four-family event-study release.

Typical usage on a machine that holds the downloaded suite output:

    python scripts/causal_python/export_event_study_final.py \
        --suite-root outputs/event_study_python \
        --output-directory outputs/event_study_final \
        --zip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from urban_intervention.causal.event_study_release import (  # noqa: E402
    FAMILIES,
    export_event_study_final,
)
from urban_intervention.data.paths import OUTPUT_DIR  # noqa: E402


def _resolve_roots(
    suite_root: Path | None, matching_root: Path | None, pooled_root: Path | None
) -> tuple[Path, Path, Path | None]:
    if suite_root is None and (matching_root is None or pooled_root is None):
        raise SystemExit("provide --suite-root or both --matching-root and --pooled-root")
    matching = matching_root or suite_root
    assert matching is not None
    if pooled_root is None:
        assert suite_root is not None
        pooled = (
            suite_root / "gsc_mc_paths"
            if (suite_root / "gsc_mc_paths").is_dir()
            else suite_root
        )
    else:
        pooled = pooled_root
    inferred_suite = suite_root
    if (pooled / "gsc_mc_paths").is_dir():
        if inferred_suite is None:
            inferred_suite = pooled
        pooled = pooled / "gsc_mc_paths"
    return matching, pooled, inferred_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path)
    parser.add_argument("--matching-root", type=Path)
    parser.add_argument("--pooled-root", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=OUTPUT_DIR / "event_study_final",
    )
    parser.add_argument(
        "--downstream-families",
        default="viirs,poi",
        help="Comma-separated families entering downstream causal supervision.",
    )
    parser.add_argument("--zip", action="store_true", help="Also write mit_event_study_final.zip.")
    parser.add_argument("--zip-name", default="mit_event_study_final.zip")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Exit 0 even when validation reports blocking errors (bounded tests/demos).",
    )
    args = parser.parse_args()

    matching_root, pooled_root, suite_root = _resolve_roots(
        args.suite_root, args.matching_root, args.pooled_root
    )
    downstream = tuple(
        value.strip() for value in args.downstream_families.split(",") if value.strip()
    )
    unknown = sorted(set(downstream) - set(FAMILIES))
    if unknown:
        parser.error(f"unknown downstream families: {unknown}")

    result = export_event_study_final(
        matching_root,
        pooled_root,
        args.output_directory,
        suite_root=suite_root,
        downstream=downstream,
        make_zip=args.zip,
        zip_name=args.zip_name,
    )
    validation = result["validation"]
    errors = [
        check
        for check in validation["checks"]
        if check["severity"] == "error" and not check["passed"]
    ]
    print(f"Wrote {result['output_directory']}")
    if result["zip"] is not None:
        print(f"Archive: {result['zip']}")
    print(
        f"Validation: {'ok' if validation['ok'] else 'FAILED'} "
        f"({len(errors)} blocking errors)"
    )
    for check in errors:
        print(f"  ERROR {check['name']}: {check['detail']}", file=sys.stderr)
    print(json.dumps(result["manifest"]["row_counts"], ensure_ascii=False))
    if errors and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
