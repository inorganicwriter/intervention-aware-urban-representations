"""Build leakage-safe model inputs from a published Response Artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.causal.pretraining_dataset import (  # noqa: E402
    publish_pretraining_dataset,
)
from urban_intervention.data.paths import (  # noqa: E402
    MODEL_INPUTS_DIR,
    TREATMENT_UNIT_LIST,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-release", type=Path, required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument("--split-seed", default="mit-urban-v1")
    parser.add_argument("--min-modalities", type=int, default=2)
    parser.add_argument("--streetview-index", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--scope-view",
        choices=("all", "same_city", "cross_city"),
        default="all",
        help="Restrict final_training_mask to a donor scope: all (default, "
        "cross-city labels included), same_city (main-specification view), "
        "or cross_city (extension-only view).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=MODEL_INPUTS_DIR,
    )
    args = parser.parse_args()
    destination = publish_pretraining_dataset(
        args.response_release,
        TREATMENT_UNIT_LIST,
        ROOT,
        args.output_root,
        dataset_id=args.dataset_id,
        split_seed=args.split_seed,
        min_modalities=args.min_modalities,
        streetview_index=args.streetview_index,
        strict_production=not args.allow_partial,
        scope_view=args.scope_view,
    )
    print(f"Published pretraining dataset at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
