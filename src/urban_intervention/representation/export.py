"""Export learned embeddings and their metadata to a parquet file.

Usage:
    urban-export-embeddings CHECKPOINT DATA_DIR --output embeddings.parquet [options]

The output has one row per grid in the dataset (all splits) with identity
columns (``treatment_order``, ``city_key``, ``split``, ``quality_grade``,
``final_training_mask``) plus ``emb_0..emb_{d-1}`` embedding columns, ready for
downstream retrieval, clustering, or figure scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import RepresentationDataset, collate_samples
from .model import ResponseEmbeddingModel

EMBEDDING_COLUMN_PREFIX = "emb_"


def export_embeddings(
    checkpoint_path: Path,
    model_inputs_dir: Path,
    output_path: Path,
    device: str | None = None,
    batch_size: int = 64,
    use_images: bool | None = None,
    max_images_per_grid: int | None = None,
) -> Path:
    """Load a trained checkpoint and write per-grid embeddings to parquet."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_name)

    checkpoint_uses_images = bool(checkpoint.get("use_image_encoder", False))
    if use_images is not None and use_images != checkpoint_uses_images:
        raise ValueError(
            "Image setting does not match checkpoint architecture: "
            f"checkpoint use_images={checkpoint_uses_images}, requested={use_images}"
        )
    effective_use_images = checkpoint_uses_images
    effective_max_images = int(
        max_images_per_grid
        if max_images_per_grid is not None
        else checkpoint.get("max_images_per_grid", 4)
    )

    dataset = RepresentationDataset(
        model_inputs_dir,
        split=None,
        max_images_per_grid=effective_max_images,
        load_images=effective_use_images,
        only_training_mask=False,
    )
    expected_columns = checkpoint.get("feature_columns")
    if not isinstance(expected_columns, (list, tuple)):
        raise ValueError(
            "Checkpoint does not contain feature_columns; refusing an unvalidated export"
        )
    if list(expected_columns) != dataset.feature_columns:
        raise ValueError(
            "Dataset feature schema does not match checkpoint feature_columns "
            "(names or ordering differ)"
        )
    checkpoint_dataset_id = str(checkpoint.get("dataset_id", ""))
    current_dataset_id = str(dataset.manifest.get("dataset_id", ""))
    if checkpoint_dataset_id and checkpoint_dataset_id != current_dataset_id:
        raise ValueError(
            "Dataset id does not match checkpoint: "
            f"{current_dataset_id!r} != {checkpoint_dataset_id!r}"
        )
    expected_feature_hash = str(checkpoint.get("feature_artifact_sha256", ""))
    current_feature_hash = str(
        dataset.manifest.get("outputs", {}).get("unit_features.parquet", "")
    )
    if expected_feature_hash and expected_feature_hash != current_feature_hash:
        raise ValueError("unit_features artifact hash does not match checkpoint")
    expected_normalization_hash = str(checkpoint.get("normalization_sha256", ""))
    current_normalization_hash = hashlib.sha256(
        json.dumps(
            dataset.normalization,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    if expected_normalization_hash and expected_normalization_hash != current_normalization_hash:
        raise ValueError("Dataset normalization metadata does not match checkpoint")
    if int(checkpoint["feature_dim"]) != dataset.feature_dim():
        raise ValueError("Dataset feature dimension does not match checkpoint")
    loader: DataLoader[object] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_samples(
            b,
            load_images_fn=dataset._get_images if effective_use_images else None,
            max_images_per_grid=dataset.max_images_per_grid,
            use_images=effective_use_images,
        ),
    )

    model = ResponseEmbeddingModel(
        feature_dim=int(checkpoint["feature_dim"]),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        dropout=float(checkpoint.get("dropout", 0.1)),
        use_image_encoder=effective_use_images,
        image_pooling=str(checkpoint.get("image_pooling", "max")),
        conditioning=checkpoint.get("conditioning"),
        learnable_temperature=bool(checkpoint.get("learnable_temperature", False)),
        uncertainty_weighted=bool(checkpoint.get("uncertainty_weighted", False)),
    ).to(dev)
    # Legacy image checkpoints accidentally serialized the frozen lazy
    # backbone. It is loaded independently by ImageEncoder and must not be
    # treated as trainable model state.
    model_state = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if not key.startswith("image_encoder._backbone.")
    }
    model.load_state_dict(model_state)
    model.eval()

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(dev)
            images = batch.get("images")
            image_mask = batch.get("image_mask")
            conditioning_tokens = batch.get("conditioning_tokens")
            if images is not None:
                images = images.to(dev)
            if image_mask is not None:
                image_mask = image_mask.to(dev)
            if conditioning_tokens is not None:
                conditioning_tokens = conditioning_tokens.to(dev)
            embedding, _ = model(
                features, images, image_mask, conditioning_tokens=conditioning_tokens
            )
            embedding = embedding.cpu()
            dim = embedding.shape[1]
            for index, order in enumerate(batch["treatment_order"].tolist()):
                row: dict[str, object] = {
                    "treatment_order": int(order),
                    "city_key": str(batch["city_keys"][index]),
                    "split": str(batch["splits"][index]),
                    "quality_grade": str(batch["quality_grades"][index]),
                    "final_training_mask": bool(batch["final_training_mask"][index].item()),
                }
                for d in range(dim):
                    row[f"{EMBEDDING_COLUMN_PREFIX}{d}"] = float(embedding[index, d].item())
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("treatment_order").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False, compression="zstd")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="Path to best_model.pt")
    parser.add_argument("data_dir", type=Path, help="Path to data/model_inputs/<dataset_id>/")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/representation/embeddings.parquet")
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--use-images",
        action="store_true",
        dest="use_images",
        help="Require an image-enabled checkpoint (normally inferred)",
    )
    image_group.add_argument(
        "--no-images",
        action="store_false",
        dest="use_images",
        help="Require a tabular-only checkpoint (normally inferred)",
    )
    parser.set_defaults(use_images=None)
    parser.add_argument(
        "--max-images", type=int, default=None, help="Override checkpoint image-count setting"
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.data_dir.is_dir():
        print(f"Error: data directory not found: {args.data_dir}", file=sys.stderr)
        return 1
    try:
        path = export_embeddings(
            args.checkpoint,
            args.data_dir,
            args.output,
            device=args.device,
            batch_size=args.batch_size,
            use_images=args.use_images,
            max_images_per_grid=args.max_images,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote embeddings for {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
