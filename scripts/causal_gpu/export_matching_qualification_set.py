#!/usr/bin/env python3
"""Export at least three R Matching references, including final label paths."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def reference_quality_passed(path: Path) -> bool:
    """Return whether one exported R reference passed its frozen quality gate."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return False
    return len(rows) == 1 and str(rows[0].get("accepted", "")).upper() == "TRUE"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orders",
        required=True,
        help="Comma-separated representative treatment orders; at least three are required.",
    )
    parser.add_argument(
        "--scope",
        choices=("same_city", "all_city_standardized"),
        default="same_city",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "complete_estimators" / "gpu_matching_inputs",
    )
    parser.add_argument("--rscript", default=os.environ.get("MIT_RSCRIPT", "Rscript"))
    args = parser.parse_args()
    try:
        orders = sorted({int(value) for value in args.orders.split(",") if value.strip()})
    except ValueError as error:
        parser.error(f"--orders contains a non-integer value: {error}")
    if len(orders) < 3:
        parser.error("formal qualification requires at least three Matching orders")

    exporter = ROOT / "scripts" / "causal_gpu" / "export_matching_reference.R"
    for order in orders:
        output = args.output_root / f"{order:05d}"
        completed = subprocess.run(
            [
                args.rscript,
                str(exporter),
                str(order),
                str(output),
                args.scope,
                "reference",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(completed.stdout or "", end="")
        if completed.returncode != 0:
            raise RuntimeError(f"Matching reference export failed for treatment {order}")
        required = (
            "matching_input.parquet",
            "metadata.csv",
            "reference_candidates.csv",
            "reference_selection.csv",
            "reference_labels.parquet",
        )
        missing = [name for name in required if not (output / name).is_file()]
        if missing:
            raise RuntimeError(
                f"Matching reference {order} is incomplete: {', '.join(missing)}"
            )
        if not reference_quality_passed(output / "reference_selection.csv"):
            raise RuntimeError(
                f"Matching reference {order} failed the frozen holdout/placebo quality gate"
            )
    print(f"Exported {len(orders)} Matching qualification references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
