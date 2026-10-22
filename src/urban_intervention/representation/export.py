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
from .model import JointSTCResponseModel, ResponseEmbeddingModel, STCResponseModel

EMBEDDING_COLUMN_PREFIX = "emb_"


def export_embeddings(
    checkpoint_path: Path,
    model_inputs_dir: Path,
    output_path: Path,
    device: str | None = None,
    batch_size: int = 64,
    use_images: bool | None = None,
    max_images_per_grid: int | None = None,
    station_attributes_path: Path | None = None,
    allow_unbound_station_attributes: bool = False,
    streetview_index_path: Path | str | None = None,
    satellite_root: Path | str | None = None,
) -> Path:
    """Load a trained checkpoint and write per-grid embeddings to parquet."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_name)

    checkpoint_uses_images = bool(checkpoint.get("use_image_encoder", False))
    checkpoint_uses_satellite = bool(checkpoint.get("use_satellite_encoder", False))
    checkpoint_uses_synergy = bool(checkpoint.get("use_cross_view_synergy", False))
    synergy_heads = int(checkpoint.get("synergy_heads", 4))
    allow_satellite_proxy = bool(checkpoint.get("allow_satellite_proxy", False))

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

    checkpoint_conditioning = checkpoint.get("conditioning")
    if (
        checkpoint_conditioning in {"station", "opening_year_station"}
        and station_attributes_path is None
    ):
        raise ValueError("Checkpoint uses station conditioning; pass --station-attributes")
    conditioning_schema = str(
        checkpoint.get("conditioning_schema") or ("v1" if checkpoint_conditioning else "v2")
    )

    if streetview_index_path is None:
        raw_idx = checkpoint.get("streetview_index_path") or checkpoint.get("stc_contract", {}).get("streetview_index_path")
        if raw_idx is not None:
            streetview_index_path = Path(raw_idx)

    if satellite_root is None:
        raw_sat = checkpoint.get("satellite_root") or checkpoint.get("stc_contract", {}).get("satellite_root")
        if raw_sat is not None:
            satellite_root = Path(raw_sat)

    dataset = RepresentationDataset(
        model_inputs_dir,
        split=None,
        max_images_per_grid=effective_max_images,
        load_images=effective_use_images,
        only_training_mask=False,
        station_attributes_path=station_attributes_path,
        station_schema=conditioning_schema,
        allow_satellite_proxy=allow_satellite_proxy,
        streetview_index_path=streetview_index_path,
        satellite_root=satellite_root,
    )
    station_binding_status = "not_used"
    if checkpoint_conditioning in {"station", "opening_year_station"}:
        expected_binding = checkpoint.get("station_attributes_binding")
        if expected_binding is None:
            if not allow_unbound_station_attributes:
                raise ValueError(
                    "Legacy station-conditioned checkpoint has no station attribute binding; "
                    "recover the original input or use --allow-unbound-station-attributes "
                    "for an explicitly unverified legacy export"
                )
            station_binding_status = "unverified_legacy"
        elif expected_binding != dataset.station_attributes_binding:
            raise ValueError(
                "Station attribute file, schema or token coverage does not match checkpoint"
            )
        else:
            station_binding_status = "verified"
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
    # Full training data bundle integrity verification on disk for STC
    expected_bundle = checkpoint.get("bundle_shas") or checkpoint.get("stc_contract", {}).get("bundle_shas")
    if isinstance(expected_bundle, dict) and expected_bundle:
        for fname, exp_sha in expected_bundle.items():
            disk_f = dataset.root / fname
            if not disk_f.is_file():
                raise FileNotFoundError(f"Checkpoint bundle file {fname} not found in {dataset.root}")
            actual_file_sha = hashlib.sha256(disk_f.read_bytes()).hexdigest()
            if actual_file_sha != exp_sha:
                raise ValueError(
                    f"Dataset bundle file {fname} SHA256 on disk ({actual_file_sha}) does not match checkpoint bundle SHA ({exp_sha})"
                )
    elif getattr(dataset, "is_stc", False) and isinstance(dataset.manifest.get("outputs"), dict):
        # If STC checkpoint lacks bundle_shas, verify manifest outputs on disk
        for fname, exp_sha in dataset.manifest["outputs"].items():
            disk_f = dataset.root / fname
            if disk_f.is_file():
                actual_file_sha = hashlib.sha256(disk_f.read_bytes()).hexdigest()
                if actual_file_sha != exp_sha:
                    raise ValueError(
                        f"Dataset bundle file {fname} SHA256 on disk ({actual_file_sha}) does not match dataset manifest outputs ({exp_sha})"
                    )

    expected_feature_hash = str(checkpoint.get("feature_artifact_sha256", ""))
    feat_output_key = "state_inputs.parquet" if getattr(dataset, "is_stc", False) else "unit_features.parquet"
    if getattr(dataset, "is_stc", False):
        feat_disk = dataset.root / feat_output_key
        if feat_disk.is_file():
            actual_feat_sha = hashlib.sha256(feat_disk.read_bytes()).hexdigest()
            if expected_feature_hash and expected_feature_hash != actual_feat_sha:
                raise ValueError(f"{feat_output_key} artifact hash on disk does not match checkpoint")
    current_feature_hash = str(dataset.manifest.get("outputs", {}).get(feat_output_key, ""))
    if expected_feature_hash and expected_feature_hash != current_feature_hash:
        raise ValueError(f"{feat_output_key} artifact hash does not match checkpoint")
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
    expected_sv_assets_sha = checkpoint.get("stc_contract", {}).get("streetview_assets_sha256")
    if (
        expected_sv_assets_sha
        and getattr(dataset, "streetview_assets_sha256", None)
        and expected_sv_assets_sha != dataset.streetview_assets_sha256
    ):
        raise ValueError(
            f"Streetview selection assets SHA ({dataset.streetview_assets_sha256}) "
            f"does not match checkpoint contract ({expected_sv_assets_sha})"
        )
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

    if checkpoint.get("is_stc") or getattr(dataset, "is_stc", False):
        if not (checkpoint.get("is_stc") and getattr(dataset, "is_stc", False)):
            raise ValueError("STC checkpoint and STC dataset must both be present")
        stc_contract = checkpoint.get("stc_contract", {})
        if "state_tabular_columns" in stc_contract and list(dataset.state_tabular_columns) != list(
            stc_contract["state_tabular_columns"]
        ):
            raise ValueError(
                f"Dataset state_tabular_columns do not match checkpoint contract: "
                f"{list(dataset.state_tabular_columns)} != {list(stc_contract['state_tabular_columns'])}"
            )
        if "intervention_columns" in stc_contract and list(dataset.intervention_columns) != list(
            stc_contract["intervention_columns"]
        ):
            raise ValueError(
                f"Dataset intervention_columns do not match checkpoint contract: "
                f"{list(dataset.intervention_columns)} != {list(stc_contract['intervention_columns'])}"
            )
        if "context_tabular_columns" in stc_contract and list(
            dataset.context_tabular_columns
        ) != list(stc_contract["context_tabular_columns"]):
            raise ValueError(
                f"Dataset context_tabular_columns do not match checkpoint contract: "
                f"{list(dataset.context_tabular_columns)} != {list(stc_contract['context_tabular_columns'])}"
            )
        if "satellite_relative_years" in stc_contract and list(
            dataset.satellite_relative_years
        ) != list(stc_contract["satellite_relative_years"]):
            raise ValueError(
                f"Dataset satellite_relative_years do not match checkpoint contract: "
                f"{list(dataset.satellite_relative_years)} != {list(stc_contract['satellite_relative_years'])}"
            )
        if "response_dim" in stc_contract and dataset.response_dim() != int(
            stc_contract["response_dim"]
        ):
            raise ValueError(
                f"Dataset response_dim ({dataset.response_dim()}) does not match checkpoint contract ({stc_contract['response_dim']})"
            )

        model = STCResponseModel(
            state_tabular_dim=len(dataset.state_tabular_columns),
            intervention_dim=len(dataset.intervention_columns),
            context_tabular_dim=len(dataset.context_tabular_columns),
            alphaearth_dim=64,
            z_dim=int(checkpoint["embedding_dim"]),
            t_dim=32,
            b_dim=64,
            predictor_hidden_dim=128,
            outcome_dim=dataset.response_dim(),
            outcome_family=stc_contract.get("outcome_family"),
            use_streetview=effective_use_images,
            use_satellite=checkpoint_uses_satellite,
            streetview_time_aware=bool(stc_contract.get("streetview_time_aware", False)),
            dropout=float(checkpoint.get("dropout", 0.1)),
            learnable_temperature=bool(checkpoint.get("learnable_temperature", False)),
            uncertainty_weighted=bool(checkpoint.get("uncertainty_weighted", False)),
        ).to(dev)
        if checkpoint.get("architecture_version") == JointSTCResponseModel.architecture_version:
            if stc_contract.get("dimensions") != {
                "z": 128, "t": 32, "b": 96, "predictor_input": 256, "viirs": 6, "poi": 12
            }:
                raise ValueError("Joint checkpoint has an invalid dimension contract")
            model = JointSTCResponseModel(
                state_tabular_dim=len(dataset.state_tabular_columns),
                context_tabular_dim=len(dataset.context_tabular_columns),
                intervention_dim=len(dataset.intervention_columns),
                use_streetview=effective_use_images,
                use_satellite=checkpoint_uses_satellite,
                dropout=float(checkpoint.get("dropout", 0.1)),
                prediction_inputs=checkpoint.get("prediction_inputs", "full"),
            ).to(dev)
    else:
        model = ResponseEmbeddingModel(
            feature_dim=int(checkpoint["feature_dim"]),
            hidden_dims=tuple(checkpoint["hidden_dims"]),
            embedding_dim=int(checkpoint["embedding_dim"]),
            dropout=float(checkpoint.get("dropout", 0.1)),
            use_image_encoder=effective_use_images,
            image_pooling=str(checkpoint.get("image_pooling", "max")),
            use_satellite_encoder=checkpoint_uses_satellite,
            use_cross_view_synergy=checkpoint_uses_synergy,
            synergy_heads=synergy_heads,
            conditioning=checkpoint.get("conditioning"),
            conditioning_schema=conditioning_schema,
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
    identity_lookup = (dataset.state_inputs.set_index("treatment_order")
                       if checkpoint.get("model_type") == "stc_joint" else None)
    with torch.no_grad():
        for batch in loader:
            if getattr(model, "is_stc_model", False):
                device_batch = {
                    k: v.to(dev) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                embedding, predictions, fused = model.forward_batch(device_batch)
                predictions = {k: v.cpu() for k, v in predictions.items()}
                embedding = embedding.cpu()
                fused = fused.cpu()
            else:
                features = batch["features"].to(dev)
                images = batch.get("images")
                image_mask = batch.get("image_mask")
                conditioning_tokens = batch.get("conditioning_tokens")
                station_tokens = batch.get("station_tokens")
                satellite_vectors = batch.get("satellite_vectors")
                satellite_mask = batch.get("satellite_mask")
                if images is not None:
                    images = images.to(dev)
                if image_mask is not None:
                    image_mask = image_mask.to(dev)
                if conditioning_tokens is not None:
                    conditioning_tokens = conditioning_tokens.to(dev)
                if station_tokens is not None:
                    station_tokens = station_tokens.to(dev)
                if satellite_vectors is not None:
                    satellite_vectors = satellite_vectors.to(dev)
                if satellite_mask is not None:
                    satellite_mask = satellite_mask.to(dev)
                embedding, _ = model(
                    features,
                    images,
                    image_mask,
                    conditioning_tokens=conditioning_tokens,
                    station_tokens=station_tokens,
                    satellite_vectors=satellite_vectors,
                    satellite_mask=satellite_mask,
                )
                embedding = embedding.cpu()
                fused = None

            dim = embedding.shape[1]
            fused_dim = fused.shape[1] if fused is not None else 0
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
                if fused is not None:
                    for d in range(fused_dim):
                        row[f"fused_{d}"] = float(fused[index, d].item())
                if checkpoint.get("model_type") == "stc_joint":
                    for family, prediction in predictions.items():
                        start, end, _ = dataset.response_offsets()[family]
                        for d in range(prediction.shape[1]):
                            row[f"tau_hat_{family}_{d}"] = float(prediction[index, d])
                        row[f"{family}_training_mask"] = bool(
                            (batch["response_mask"][index, start:end]
                             & torch.isfinite(batch["response_weight"][index, start:end])
                             & (batch["response_weight"][index, start:end] > 0)).any()
                        )
                    identity = identity_lookup.loc[int(order)]
                    for key in ("grid_id", "station_event_id", "opening_month"):
                        row[key] = str(identity[key])
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("treatment_order").reset_index(drop=True)
    frame.attrs["station_attributes"] = {
        "status": station_binding_status,
        "binding": dataset.station_attributes_binding
        if checkpoint_conditioning in {"station", "opening_year_station"}
        else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False, compression="zstd")
    if checkpoint.get("model_type") == "stc_joint":
        output_path.with_suffix(".metadata.json").write_text(json.dumps({
            "architecture_version": checkpoint["architecture_version"],
            "training_mode": checkpoint["training_mode"],
            "dataset_id": checkpoint["dataset_id"],
            "dimensions": checkpoint["stc_contract"]["dimensions"],
            "embedding_columns": [f"emb_{i}" for i in range(128)],
            "bundle_shas": checkpoint.get("bundle_shas", {}),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
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
    parser.add_argument(
        "--station-attributes",
        type=Path,
        default=None,
        help="Station attribute parquet (required for station-conditioned checkpoints)",
    )
    parser.add_argument(
        "--allow-unbound-station-attributes",
        action="store_true",
        help="Allow legacy station checkpoints; mark the export as unverified",
    )
    parser.add_argument(
        "--streetview-index-path",
        type=Path,
        default=None,
        help="Path to streetview state assets parquet table",
    )
    parser.add_argument(
        "--satellite-root",
        type=Path,
        default=None,
        help="Root directory containing satellite images and sequences",
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
            station_attributes_path=args.station_attributes,
            allow_unbound_station_attributes=args.allow_unbound_station_attributes,
            streetview_index_path=args.streetview_index_path,
            satellite_root=args.satellite_root,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote embeddings for {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
