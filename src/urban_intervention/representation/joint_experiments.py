"""Controlled shared-STC experiments on one immutable joint data bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .export import export_embeddings
from .trainer import train_representation
from .training.joint import write_json

RECIPES = {
    "viirs_only": {"training_mode": "viirs_only"},
    "poi_only": {"training_mode": "poi_only"},
    "joint": {"training_mode": "joint"},
    "anchor_only": {"training_mode": "joint", "loss_align_weight": 0.0},
    "tc_only": {"training_mode": "joint", "prediction_inputs": "tc_only"},
    "z_only": {"training_mode": "joint", "prediction_inputs": "z_only"},
    "no_streetview": {"training_mode": "joint", "use_images": False},
    "no_satellite": {"training_mode": "joint", "use_satellite": False},
    "tabular_only": {"training_mode": "joint", "use_images": False, "use_satellite": False},
}


def run_comparison(data_dir: Path, output: Path, names=None, allow_fixture=False, **common):
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    if not manifest.get("strict_production", False) and not allow_fixture:
        raise ValueError(
            "Formal comparisons require a strict-production bundle; fixture runs must be explicit"
        )
    names = list(names or ("viirs_only", "poi_only", "joint"))
    unknown = set(names) - RECIPES.keys()
    if unknown:
        raise ValueError(f"Unknown experiment recipes: {sorted(unknown)}")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "comparison_status.json"
    if state_path.exists():
        raise FileExistsError(f"Refusing to overwrite comparison {output}")
    result = {
        "dataset_id": manifest.get("dataset_id"),
        "production_evidence": bool(manifest.get("strict_production", False)),
        "source_kind": manifest.get("source_kind", "formal_joint_bundle"),
        "runs": {name: {"status": "pending"} for name in names},
        "support_consistent": None,
    }
    write_json(state_path, result)
    for name in names:
        options = {
            "embedding_dim": 128,
            "use_images": True,
            "use_satellite": True,
            **common,
            **RECIPES[name],
        }
        out = output / name
        result["runs"][name] = {"status": "running", "options": options}
        write_json(state_path, result)
        try:
            ckpt = train_representation(data_dir, out, **options)
            exported = export_embeddings(
                ckpt,
                data_dir,
                out / "embeddings.parquet",
                device=options.get("device"),
                streetview_index_path=options.get("streetview_index_path"),
                satellite_root=options.get("satellite_root"),
            )
            report = json.loads((out / "evaluation_report.json").read_text(encoding="utf-8"))
            config = json.loads((out / "training_config.json").read_text(encoding="utf-8"))
            result["runs"][name] = {
                "status": "completed",
                "checkpoint": str(ckpt),
                "export": str(exported),
                "test": report["test"]["tasks"],
                "data_splits": config["data_splits"],
                "bundle_shas": config["bundle_shas"],
                "code_sha256": config["code_sha256"],
            }
        except Exception as exc:
            result["runs"][name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        write_json(state_path, result)
    completed = [r for r in result["runs"].values() if r["status"] == "completed"]
    if completed:
        first = completed[0]
        result["support_consistent"] = all(
            r["data_splits"] == first["data_splits"]
            and r["bundle_shas"] == first["bundle_shas"]
            and r["code_sha256"] == first["code_sha256"]
            and all(
                r["test"][f]["support_sha256"] == first["test"][f]["support_sha256"]
                for f in ("viirs", "poi")
            )
            for r in completed
        )
    result["complete"] = len(completed) == len(names) and result["support_consistent"] is True
    write_json(state_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(RECIPES),
        default=["viirs_only", "poi_only", "joint"],
    )
    parser.add_argument("--allow-fixture", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-diagnostics-interval", type=int, default=1)
    parser.add_argument("--eval-n-perm", type=int, default=100)
    parser.add_argument("--eval-n-boot", type=int, default=200)
    args = vars(parser.parse_args())
    args["names"] = args.pop("experiments")
    args["max_grad_norm"] = args["max_grad_norm"] or None
    result = run_comparison(**args)
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
