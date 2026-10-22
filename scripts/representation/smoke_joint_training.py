"""Non-production multimodal smoke using synthetic data and cached real DINOv2."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch

from tests.helpers.joint_stc import build_joint_fixture
from urban_intervention.representation.joint_experiments import RECIPES, run_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(RECIPES),
        default=["viirs_only", "poi_only", "joint"],
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    hub = Path(os.environ.get("TORCH_HUB", torch.hub.get_dir()))
    repo = hub / "facebookresearch_dinov2_main"
    weights = hub / "checkpoints/dinov2_vits14_pretrain.pth"
    if not repo.is_dir() or not weights.is_file():
        raise FileNotFoundError(
            "Smoke requires the existing local DINOv2 repo and weights; no downloads are attempted"
        )
    backbone = torch.hub.load(str(repo), "dinov2_vits14", source="local", pretrained=True)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)
    data = build_joint_fixture(args.output / "input", n_units=args.events, images=True)
    evidence = {
        "production_eligible": False,
        "labels_and_images": "synthetic_fixture",
        "backbone": "real_frozen_cached_dinov2_vits14",
        "backbone_weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "events": args.events,
        "epochs": args.epochs,
    }
    (args.output / "smoke_evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    with patch("urban_intervention.representation.encoder._ensure_dinov2", return_value=backbone):
        result = run_comparison(
            data,
            args.output / "experiments",
            names=args.experiments,
            allow_fixture=True,
            epochs=args.epochs,
            batch_size=4,
            device="cpu",
            seed=42,
            max_grad_norm=1.0,
            gradient_diagnostics_interval=1,
            eval_n_perm=4,
            eval_n_boot=4,
            run_baselines=True,
            run_transfer=True,
        )
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "support_consistent": result["support_consistent"],
                "runs": {k: v["status"] for k, v in result["runs"].items()},
            },
            indent=2,
        )
    )
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
