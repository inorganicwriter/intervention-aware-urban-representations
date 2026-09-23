"""Publish terminal causal-label tasks into one immutable Response Artifact."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.causal.gpu.control_design_waiver import (  # noqa: E402
    CONTROL_DESIGN_WAIVER_ENV_VAR,
    load_control_design_waiver,
)
from urban_intervention.causal.gpu.qualification import (  # noqa: E402
    WAIVER_ENV_VAR as QUALIFICATION_WAIVER_ENV_VAR,
)
from urban_intervention.causal.response_artifact import (  # noqa: E402
    ArtifactInputs,
    load_release_qualification_waiver,
    publish_response_artifact,
)
from urban_intervention.data.paths import (  # noqa: E402
    CAUSAL_RELEASES_DIR,
    CONTROL_DESIGN_QUEUE,
    ELIGIBLE_DONORS,
    FORMAL_TARGET_SUPPORT,
    OUTCOME_FAMILY_QUEUE,
    OUTPUT_CAUSAL_TASKS_DIR,
    OUTPUT_CONTROL_TASKS_DIR,
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
    parser.add_argument(
        "--tasks-file",
        type=Path,
        help=(
            "Optional CSV with treatment_order and outcome_family; restricts a "
            "canary release to those exact independently sampled tasks."
        ),
    )
    parser.add_argument(
        "--control-design-waiver",
        type=Path,
        help=(
            "Operator-attested spot-check waiver for control records whose stored "
            "matching fingerprint differs from the current source tree."
        ),
    )
    parser.add_argument(
        "--qualification-waiver",
        type=Path,
        help=(
            "Operator-attested drift waiver letting receipt-revision tasks keep "
            "their formal qualification proof in the release."
        ),
    )
    args = parser.parse_args()
    if args.orders_file is not None and args.tasks_file is not None:
        parser.error("--orders-file and --tasks-file are mutually exclusive")
    waiver_path = args.control_design_waiver
    if waiver_path is None:
        environment_waiver = os.environ.get(CONTROL_DESIGN_WAIVER_ENV_VAR, "").strip()
        if environment_waiver:
            waiver_path = Path(environment_waiver)
    control_design_waiver = (
        load_control_design_waiver(waiver_path) if waiver_path is not None else None
    )
    qualification_waiver_path = args.qualification_waiver
    if qualification_waiver_path is None:
        environment_qualification = os.environ.get(
            QUALIFICATION_WAIVER_ENV_VAR, ""
        ).strip()
        if environment_qualification:
            qualification_waiver_path = Path(environment_qualification)
    qualification_waiver = (
        load_release_qualification_waiver(qualification_waiver_path)
        if qualification_waiver_path is not None
        else None
    )
    treatment_orders = None
    task_keys = None
    if args.orders_file is not None:
        orders_frame = pd.read_csv(args.orders_file)
        if "treatment_order" not in orders_frame.columns:
            raise ValueError(f"Orders file lacks treatment_order: {args.orders_file}")
        parsed_orders = pd.to_numeric(orders_frame["treatment_order"], errors="coerce")
        if parsed_orders.isna().any() or parsed_orders.duplicated().any():
            raise ValueError("Orders file must contain unique integer treatment_order values")
        treatment_orders = tuple(int(value) for value in parsed_orders)
    elif args.tasks_file is not None:
        tasks_frame = pd.read_csv(args.tasks_file)
        required = {"treatment_order", "outcome_family"}
        missing = required - set(tasks_frame.columns)
        if missing:
            raise ValueError(f"Tasks file lacks columns: {sorted(missing)}")
        parsed_orders = pd.to_numeric(tasks_frame["treatment_order"], errors="coerce")
        families = tasks_frame["outcome_family"].astype(str)
        parsed = tuple(zip(parsed_orders, families, strict=False))
        if parsed_orders.isna().any() or len(set(parsed)) != len(parsed) or not parsed:
            raise ValueError(
                "Tasks file must contain unique treatment_order/outcome_family keys"
            )
        task_keys = tuple((int(order), family) for order, family in parsed)
    inputs = ArtifactInputs(
        treatments=TREATMENT_UNIT_LIST,
        family_queue=OUTCOME_FAMILY_QUEUE,
        control_queue=CONTROL_DESIGN_QUEUE,
        task_root=OUTPUT_CAUSAL_TASKS_DIR,
        control_task_root=OUTPUT_CONTROL_TASKS_DIR,
        donor_universe=ELIGIBLE_DONORS,
        target_support=FORMAL_TARGET_SUPPORT,
        treatment_orders=treatment_orders,
        task_keys=task_keys,
        control_design_waiver=control_design_waiver,
        qualification_waiver=qualification_waiver,
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
