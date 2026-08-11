"""Publish terminal causal-label tasks into one immutable Response Artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    args = parser.parse_args()
    inputs = ArtifactInputs(
        treatments=TREATMENT_UNIT_LIST,
        family_queue=OUTCOME_FAMILY_QUEUE,
        control_queue=CONTROL_DESIGN_QUEUE,
        task_root=OUTPUT_CAUSAL_TASKS_DIR,
        donor_universe=ELIGIBLE_DONORS,
        target_support=FORMAL_TARGET_SUPPORT,
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
