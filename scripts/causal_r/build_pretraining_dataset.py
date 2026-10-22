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
    parser.add_argument(
        "--scope-view",
        choices=("all", "same_city", "cross_city"),
        default="same_city",
        help="Restrict final_training_mask to a donor scope: same_city (default, "
        "main-specification view), all (cross-city labels included), "
        "or cross_city (extension-only view).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a non-strict/exploratory release. Strict production defaults "
        "to the frozen VIIRS+POI / same-city Matching+GSC supervision boundary.",
    )
    parser.add_argument(
        "--families",
        help="Comma-separated outcome families to export. Strict production "
        "defaults to viirs,poi; exploratory releases may request others.",
    )
    parser.add_argument(
        "--methods",
        help="Comma-separated estimator methods to keep: matched, gsc, mc. "
        "Strict production defaults to matched,gsc; filtering happens before "
        "weight calibration.",
    )
    parser.add_argument(
        "--weight-gamma",
        type=float,
        default=None,
        help="Design-score steepness for the label reliability weights (default 1.0; 0 = equal design weights).",
    )
    parser.add_argument(
        "--weight-alpha",
        type=float,
        default=None,
        help="Precision-multiplier tempering for the label reliability weights (default 0.5; 0 or 1 for sensitivity).",
    )
    parser.add_argument(
        "--satellite-source",
        type=Path,
        default=None,
        help="Path to AlphaEarth satellite embeddings parquet file or directory.",
    )
    parser.add_argument(
        "--allow-satellite-proxy",
        action="store_true",
        default=False,
        help="Allow contemporary/post-treatment satellite proxy embeddings (default: False, strict pre-treatment only).",
    )
    parser.add_argument(
        "--require-multimodal",
        action="store_true",
        default=False,
        help="Enforce mandatory multimodal inputs: require both valid pre-treatment satellite and streetview for eligible training units.",
    )
    parser.add_argument(
        "--require-satellite",
        action="store_true",
        default=False,
        help="Require valid pre-treatment satellite embeddings for eligible training units.",
    )
    parser.add_argument(
        "--require-streetview",
        action="store_true",
        default=False,
        help="Require valid pre-treatment street-view images for eligible training units.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=MODEL_INPUTS_DIR,
    )
    parser.add_argument(
        "--stc",
        action="store_true",
        default=False,
        help="Publish S/T/C role-based multimodal dataset package instead of legacy schema.",
    )
    parser.add_argument(
        "--stc-inputs-dir",
        type=Path,
        default=None,
        help="Directory containing S/T/C role tables (default: data/active/multimodal_inputs_v1).",
    )
    args = parser.parse_args()
    families = None
    if args.families:
        families = tuple(
            part.strip() for part in args.families.split(",") if part.strip()
        )
    methods = None
    if args.methods:
        methods = tuple(
            part.strip() for part in args.methods.split(",") if part.strip()
        )
    weight_kwargs: dict[str, float] = {}
    if args.weight_gamma is not None:
        weight_kwargs["weight_gamma"] = args.weight_gamma
    if args.weight_alpha is not None:
        weight_kwargs["weight_alpha"] = args.weight_alpha
    destination = publish_pretraining_dataset(
        args.response_release,
        TREATMENT_UNIT_LIST,
        ROOT,
        args.output_root,
        dataset_id=args.dataset_id,
        split_seed=args.split_seed,
        min_modalities=args.min_modalities,
        streetview_index=args.streetview_index,
        satellite_source=args.satellite_source,
        strict_satellite_pretreatment=not args.allow_satellite_proxy,
        strict_production=not args.allow_partial,
        scope_view=args.scope_view,
        families=families,
        methods=methods,
        require_multimodal=args.require_multimodal,
        require_satellite=args.require_satellite,
        require_streetview=args.require_streetview,
        stc=args.stc,
        stc_inputs_dir=args.stc_inputs_dir,
        **weight_kwargs,
    )
    print(f"Published pretraining dataset at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
