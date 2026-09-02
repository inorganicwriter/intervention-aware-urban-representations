"""Audit or repair fect-normalized R GSC qualification references."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.causal.gpu.gsc_reference import (  # noqa: E402
    qualification_gsc_references,
    repair_gsc_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qualification-root",
        type=Path,
        default=Path("outputs/complete_estimators/gpu_formal_qualification"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write corrected labels and audited manifest fields",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    references = qualification_gsc_references(args.qualification_root)
    for panel, labels in references:
        result = repair_gsc_reference(panel, labels, apply=args.apply)
        print(
            f"{result.status}: {result.labels} "
            f"scale={result.normalization_scale:.12g} rows={result.corrected_rows}"
        )
    action = "Repaired" if args.apply else "Audited"
    print(f"{action} {len(references)} GSC reference artifact(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
