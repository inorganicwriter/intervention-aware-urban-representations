"""Reevaluate response-direction AUC from a frozen checkpoint, without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch

from urban_intervention.representation.dataset import RepresentationDataset
from urban_intervention.representation.export import export_embeddings
from urban_intervention.representation.transfer import predictive_transfer_report


def reevaluate_direction_auc(
    checkpoint_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    *,
    station_attributes_path: Path | None = None,
    allow_unbound_station_attributes: bool = False,
    ridge: float | None = None,
    min_obs: int | None = None,
) -> dict[str, object]:
    config_path = checkpoint_path.parent / "training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    evaluation = config.get("evaluation", {})
    ridge = float(evaluation.get("probe_ridge", 1.0) if ridge is None else ridge)
    min_obs = int(evaluation.get("probe_min_obs", 16) if min_obs is None else min_obs)
    if ridge <= 0 or min_obs < 2:
        raise ValueError("ridge must be positive and min_obs must be at least 2")

    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    schema = str(
        checkpoint.get("conditioning_schema") or ("v1" if checkpoint.get("conditioning") else "v2")
    )
    dataset = RepresentationDataset(
        dataset_dir,
        only_training_mask=True,
        load_images=False,
        station_attributes_path=station_attributes_path,
        station_schema=schema,
    )
    samples = [dataset[i] for i in range(len(dataset))]
    if not samples:
        raise ValueError("No eligible units in dataset")
    split_rows = {
        split: torch.tensor([s.split == split for s in samples], dtype=torch.bool)
        for split in ("train", "validation", "test")
    }
    if any(int(mask.sum()) < 2 for mask in split_rows.values()):
        raise ValueError("Each split needs at least two eligible units")

    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "embeddings_for_auc.parquet"
    export_embeddings(
        checkpoint_path,
        dataset_dir,
        embedding_path,
        device="cpu",
        station_attributes_path=station_attributes_path,
        allow_unbound_station_attributes=allow_unbound_station_attributes,
    )
    frame = pd.read_parquet(embedding_path)
    station_metadata = frame.attrs["station_attributes"]
    if station_metadata["status"] != "not_used" and (
        station_metadata["binding"] != dataset.station_attributes_binding
    ):
        raise ValueError("Station attributes changed during reevaluation")
    frame = frame.set_index("treatment_order").loc[[s.treatment_order for s in samples]]
    columns = [f"emb_{i}" for i in range(int(checkpoint["embedding_dim"]))]
    embeddings = torch.from_numpy(frame[columns].to_numpy(dtype="float32"))
    features = torch.stack([s.feature_vector for s in samples])
    responses = torch.stack([s.response_vector for s in samples])
    masks = torch.stack([s.response_mask for s in samples])
    train = split_rows["train"]
    report: dict[str, object] = {
        "schema": "urban_intervention_direction_auc_reevaluation_v1",
        "classifier": "per_cell_centered_ridge_v2",
        "checkpoint_sha256": checkpoint_hash,
        "dataset_id": dataset.manifest.get("dataset_id", ""),
        "ridge": ridge,
        "min_obs": min_obs,
        "retrained": False,
        "station_attributes": station_metadata,
        "eligible_split_units": {name: int(rows.sum()) for name, rows in split_rows.items()},
    }
    for name in ("validation", "test"):
        rows = split_rows[name]
        report[name] = predictive_transfer_report(
            embeddings[rows],
            features[rows],
            embeddings[train],
            features[train],
            responses[train],
            masks[train],
            responses[rows],
            masks[rows],
            ridge=ridge,
            min_obs=min_obs,
        )
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_hash:
        raise ValueError("Checkpoint changed during reevaluation")
    (output_dir / "direction_auc_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--station-attributes", type=Path)
    parser.add_argument("--allow-unbound-station-attributes", action="store_true")
    parser.add_argument("--ridge", type=float, default=None)
    parser.add_argument("--min-obs", type=int, default=None)
    args = parser.parse_args()
    reevaluate_direction_auc(
        args.checkpoint,
        args.dataset,
        args.output_dir,
        station_attributes_path=args.station_attributes,
        allow_unbound_station_attributes=args.allow_unbound_station_attributes,
        ridge=args.ridge,
        min_obs=args.min_obs,
    )
    print(f"Wrote {args.output_dir / 'direction_auc_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
