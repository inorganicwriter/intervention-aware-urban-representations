"""Publish terminal causal-label tasks into one immutable Response Artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.causal.response_artifact import (  # noqa: E402
    ArtifactInputs,
    publish_response_artifact,
)
from urban_intervention.data.paths import (  # noqa: E402
    CAUSAL_RELEASES_DIR,
    CONTROL_DESIGN_QUEUE,
    ELIGIBLE_DONORS,
    FORMAL_TARGET_SUPPORT,
    OUTCOME_FAMILY_QUEUE,
    OUTPUT_CAUSAL_TASKS_DIR,
    TREATMENT_UNIT_LIST,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Canary/audit only: allow unfinished queues; output is not a production release.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CAUSAL_RELEASES_DIR,
    )
    parser.add_argument(
        "--orders-file",
        type=Path,
        help="Optional CSV with treatment_order values; restricts a canary release to those grids.",
    )
    args = parser.parse_args()
    treatment_orders = None
    if args.orders_file is not None:
        orders_frame = pd.read_csv(args.orders_file)
        if "treatment_order" not in orders_frame.columns:
            raise ValueError(f"Orders file lacks treatment_order: {args.orders_file}")
        parsed_orders = pd.to_numeric(orders_frame["treatment_order"], errors="coerce")
        if parsed_orders.isna().any() or parsed_orders.duplicated().any():
            raise ValueError("Orders file must contain unique integer treatment_order values")
        treatment_orders = tuple(int(value) for value in parsed_orders)
    inputs = ArtifactInputs(
        treatments=TREATMENT_UNIT_LIST,
        family_queue=OUTCOME_FAMILY_QUEUE,
        control_queue=CONTROL_DESIGN_QUEUE,
        task_root=OUTPUT_CAUSAL_TASKS_DIR,
        donor_universe=ELIGIBLE_DONORS,
        target_support=FORMAL_TARGET_SUPPORT,
        treatment_orders=treatment_orders,
    )
    destination = publish_response_artifact(
        inputs,
        args.output_root,
        release_id=args.release_id,
        strict_production=not args.allow_partial,
        workers=args.workers,
        project_root=ROOT,
    )
    print(f"Published Response Artifact at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
