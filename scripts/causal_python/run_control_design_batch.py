#!/usr/bin/env python3
"""Run Python/GPU frozen-control design for one batch of treatment orders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from urban_intervention.causal.gpu.control_design import (  # noqa: E402
    design_grid_control,
    write_control_design,
)
from urban_intervention.data.paths import OUTPUT_CONTROL_TASKS_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", required=True, help="Comma-separated treatment orders")
    parser.add_argument(
        "--scope", choices=("same_city", "all_city_standardized"), default="same_city"
    )
    parser.add_argument("--task-root", type=Path, default=OUTPUT_CONTROL_TASKS_DIR)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    orders = [int(value) for value in args.orders.split(",") if value.strip()]
    if not orders:
        parser.error("--orders must contain at least one treatment order")
    for order in orders:
        suffix = "" if args.scope == "same_city" else "cross_city"
        output = args.task_root / f"{order:05d}"
        if suffix:
            output = output / suffix
        result = design_grid_control(order, scope=args.scope, device=args.device)
        write_control_design(result, output)
        record = result.record.iloc[0]
        print(
            f"control {order}: status={record['status']} "
            f"scope={args.scope} reason={record.get('failure_reason')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
