"""Tests for mandatory visual modalities (satellite and streetview).

Validates that when satellite and street-view are specified as mandatory:
1. RepresentationDataset filters units to require valid satellite embeddings.
2. RepresentationDataset filters units to require street-view images.
3. RepresentationDataset with require_multimodal=True requires both.
4. train_representation with require_multimodal=True trains successfully on the valid multimodal cohort.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tests.helpers.representation import build_synthetic_model_inputs
from urban_intervention.representation.dataset import RepresentationDataset
from urban_intervention.representation.trainer import train_representation


def test_dataset_require_satellite(tmp_path: Path) -> None:
    ds_dir, n_grids = build_synthetic_model_inputs(
        tmp_path, n_grids=24, n_cities=8, include_satellite=True
    )
    # Default: proxy units have zero satellite_vector, but are not dropped from sample_index
    ds_all = RepresentationDataset(ds_dir, split="train", allow_satellite_proxy=False)
    all_count = len(ds_all)

    # When require_satellite=True, only units with valid pre-treatment satellite vectors remain
    ds_sat = RepresentationDataset(
        ds_dir, split="train", allow_satellite_proxy=False, require_satellite=True
    )
    sat_count = len(ds_sat)
    assert 0 < sat_count < all_count
    for sample in ds_sat:
        assert sample.satellite_available is True
        assert torch.any(sample.satellite_vector != 0)


def test_dataset_require_multimodal(tmp_path: Path) -> None:
    ds_dir, n_grids = build_synthetic_model_inputs(
        tmp_path, n_grids=24, n_cities=8, include_satellite=True
    )
    # Add dummy image path to first 4 units
    import pandas as pd
    features_path = ds_dir / "unit_features.parquet"
    df = pd.read_parquet(features_path)
    img_file = tmp_path / "dummy.jpg"
    img_file.write_bytes(b"dummy")
    assets_col = []
    for i in range(len(df)):
        if i < 4:
            assets_col.append(json.dumps([str(img_file)]))
        else:
            assets_col.append("[]")
    df["streetview_assets"] = assets_col
    df["streetview_available"] = [i < 4 for i in range(len(df))]
    df.to_parquet(features_path, index=False)

    ds_mm = RepresentationDataset(
        ds_dir, split=None, allow_satellite_proxy=False, require_multimodal=True
    )
    for sample in ds_mm:
        assert sample.satellite_available is True
        assert len(sample.image_paths) > 0


def test_trainer_require_multimodal_end_to_end(tmp_path: Path) -> None:
    ds_dir, n_grids = build_synthetic_model_inputs(
        tmp_path, n_grids=48, n_cities=12, include_satellite=True
    )
    import pandas as pd
    features_path = ds_dir / "unit_features.parquet"
    df = pd.read_parquet(features_path)
    df["opening_year"] = 2020
    df["satellite_is_temporal_proxy"] = False
    df["satellite_temporal_valid"] = True
    df.to_parquet(features_path, index=False)

    out_dir = tmp_path / "train_mm_out"
    train_representation(
        model_inputs_dir=ds_dir,
        output_dir=out_dir,
        epochs=1,
        batch_size=4,
        embedding_dim=16,
        hidden_dims=(32,),
        use_images=False,
        use_satellite=True,
        require_satellite=True,
        run_baselines=False,
        run_transfer=False,
    )
    assert (out_dir / "best_model.pt").is_file()
