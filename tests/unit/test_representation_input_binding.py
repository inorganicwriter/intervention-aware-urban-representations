from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from tests.helpers.representation import build_synthetic_model_inputs
from urban_intervention.representation.dataset import RepresentationDataset
from urban_intervention.representation.export import export_embeddings
from urban_intervention.representation.model import ResponseEmbeddingModel
from urban_intervention.representation.trainer import train_representation
from urban_intervention.representation.training.runner import train_representation as train_modular
from urban_intervention.representation.transfer import predictive_auc


def _attributes(path: Path, count: int) -> Path:
    pd.DataFrame(
        {
            "treatment_order": range(1, count + 1),
            "is_transfer_at_opening": [1] * count,
            "is_new_line_opening": [0] * count,
            "is_extension_opening": [0] * count,
            "is_terminal_at_opening": [0] * count,
        }
    ).to_parquet(path, index=False)
    return path


@pytest.mark.parametrize("shift", [-100.0, 0.0, 100.0])
def test_direction_auc_is_invariant_to_feature_origin_and_missing_rows(shift: float) -> None:
    x_train = torch.arange(1, 11, dtype=torch.float64).reshape(-1, 1)
    x_train[8:] = 10000  # These rows have no observed training labels.
    y_train = torch.tensor([1] * 4 + [-1] * 4 + [0] * 2).reshape(-1, 1)
    mask = torch.tensor([True] * 8 + [False] * 2).reshape(-1, 1)
    x_test = torch.tensor([[1.5], [3.5], [5.5], [7.5]], dtype=torch.float64)
    y_test = torch.tensor([[1], [1], [-1], [-1]])
    test_mask = torch.ones_like(y_test, dtype=torch.bool)
    result = predictive_auc(
        x_train + shift, y_train, mask, x_test + shift, y_test, test_mask, min_obs=8
    )
    assert result["auc_overall"] == 1.0


@pytest.mark.parametrize("trainer", [train_representation, train_modular])
def test_station_training_binds_attributes_and_export_rejects_replacement(
    tmp_path: Path, trainer
) -> None:
    dataset_dir, count = build_synthetic_model_inputs(tmp_path)
    attrs = _attributes(tmp_path / "attributes.parquet", count)
    checkpoint_path = trainer(
        dataset_dir,
        tmp_path / "run",
        hidden_dims=(8,),
        embedding_dim=4,
        epochs=1,
        batch_size=4,
        dropout=0.0,
        device="cpu",
        conditioning="station",
        station_attributes_path=attrs,
        run_baselines=False,
        run_transfer=False,
        eval_n_perm=2,
        eval_n_boot=2,
    )
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    binding = checkpoint["station_attributes_binding"]
    assert binding["sha256"] == hashlib.sha256(attrs.read_bytes()).hexdigest()
    assert binding["schema"] == "v2"
    assert binding["dataset_units"] == binding["covered_units"] == count
    config = json.loads((tmp_path / "run" / "training_config.json").read_text())
    assert config["station_attributes_binding"] == binding

    output = tmp_path / "embeddings.parquet"
    export_embeddings(checkpoint_path, dataset_dir, output, station_attributes_path=attrs)
    assert pd.read_parquet(output).attrs["station_attributes"]["status"] == "verified"
    # A different path with identical bytes must still be accepted.
    copied = tmp_path / "copied.parquet"
    copied.write_bytes(attrs.read_bytes())
    export_embeddings(checkpoint_path, dataset_dir, output, station_attributes_path=copied)

    changed = pd.read_parquet(attrs)
    changed.loc[0, "is_transfer_at_opening"] = 0
    changed.to_parquet(attrs, index=False)
    with pytest.raises(ValueError, match="Station attribute"):
        export_embeddings(
            checkpoint_path,
            dataset_dir,
            tmp_path / "invalid.parquet",
            station_attributes_path=attrs,
        )
    # The legacy override cannot bypass a known binding mismatch.
    with pytest.raises(ValueError, match="Station attribute"):
        export_embeddings(
            checkpoint_path,
            dataset_dir,
            tmp_path / "invalid.parquet",
            station_attributes_path=attrs,
            allow_unbound_station_attributes=True,
        )
    assert not (tmp_path / "invalid.parquet").exists()


@pytest.mark.parametrize("schema", ["v1", "v2"])
def test_legacy_station_export_requires_explicit_unverified_mode(
    tmp_path: Path, schema: str
) -> None:
    dataset_dir, count = build_synthetic_model_inputs(tmp_path)
    attrs = _attributes(tmp_path / "attributes.parquet", count)
    dataset = RepresentationDataset(
        dataset_dir, station_attributes_path=attrs, station_schema=schema
    )
    model = ResponseEmbeddingModel(
        feature_dim=dataset.feature_dim(),
        hidden_dims=(8,),
        embedding_dim=4,
        use_image_encoder=False,
        conditioning="station",
        conditioning_schema=schema,
    )
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_dim": dataset.feature_dim(),
            "feature_columns": dataset.feature_columns,
            "hidden_dims": (8,),
            "embedding_dim": 4,
            "conditioning": "station",
            "conditioning_schema": schema,
        },
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="no station attribute binding"):
        export_embeddings(
            checkpoint_path,
            dataset_dir,
            tmp_path / "invalid.parquet",
            station_attributes_path=attrs,
        )
    output = tmp_path / "legacy.parquet"
    export_embeddings(
        checkpoint_path,
        dataset_dir,
        output,
        station_attributes_path=attrs,
        allow_unbound_station_attributes=True,
    )
    assert pd.read_parquet(output).attrs["station_attributes"]["status"] == "unverified_legacy"


def test_station_coverage_is_bound_before_split_filtering(tmp_path: Path) -> None:
    dataset_dir, count = build_synthetic_model_inputs(tmp_path)
    attrs = _attributes(tmp_path / "attributes.parquet", count - 1)
    train = RepresentationDataset(dataset_dir, split="train", station_attributes_path=attrs)
    full = RepresentationDataset(dataset_dir, station_attributes_path=attrs)
    assert train.station_attributes_binding == full.station_attributes_binding
    assert full.station_attributes_binding["covered_units"] == count - 1
    assert train.station_token_count == len(train)


def test_station_split_loading_rejects_attribute_changes(tmp_path: Path, monkeypatch) -> None:
    dataset_dir, count = build_synthetic_model_inputs(tmp_path)
    attrs = _attributes(tmp_path / "attributes.parquet", count)

    class ChangingDataset(RepresentationDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if kwargs.get("split") == "train":
                frame = pd.read_parquet(attrs)
                frame.loc[0, "is_transfer_at_opening"] = 0
                frame.to_parquet(attrs, index=False)

    monkeypatch.setattr(
        "urban_intervention.representation.trainer.RepresentationDataset", ChangingDataset
    )
    with pytest.raises(ValueError, match="changed while loading"):
        train_representation(
            dataset_dir, tmp_path / "run", conditioning="station", station_attributes_path=attrs
        )
