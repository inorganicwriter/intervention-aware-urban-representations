from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tests.helpers.representation import build_synthetic_model_inputs
from urban_intervention.representation.dataset import (
    RESPONSE_DIM,
    RepresentationDataset,
    collate_samples,
)


class TestRepresentationDataset:
    def test_load_full_dataset(self, tmp_path: Path) -> None:
        ds_dir, n_grids = build_synthetic_model_inputs(tmp_path)
        ds = RepresentationDataset(ds_dir)

        assert len(ds) == n_grids
        assert ds.feature_dim() > 0
        assert ds.response_dim() == RESPONSE_DIM
        assert len(ds.treatment_orders()) == n_grids

    def test_split_filtering(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)
        full_ds = RepresentationDataset(ds_dir)
        train_ds = RepresentationDataset(ds_dir, split="train")
        val_ds = RepresentationDataset(ds_dir, split="validation")
        test_ds = RepresentationDataset(ds_dir, split="test")

        assert len(train_ds) == 6
        assert len(val_ds) == 3
        assert len(test_ds) == 3
        assert len(train_ds) + len(val_ds) + len(test_ds) == len(full_ds)

    def test_sample_structure(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)
        ds = RepresentationDataset(ds_dir)
        sample = ds[0]

        assert isinstance(sample.treatment_order, int)
        assert isinstance(sample.city_key, str)
        assert sample.split in {"train", "validation", "test"}
        assert isinstance(sample.feature_vector, torch.Tensor)
        assert sample.feature_vector.ndim == 1
        assert sample.feature_vector.shape[0] == ds.feature_dim()
        assert isinstance(sample.response_vector, torch.Tensor)
        assert sample.response_vector.shape[0] == RESPONSE_DIM
        assert sample.response_mask.shape[0] == RESPONSE_DIM
        assert sample.response_mask.all()
        assert isinstance(sample.modality_available, dict)

    def test_training_mask_boundary(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)
        ds = RepresentationDataset(ds_dir)

        training = [ds[i] for i in range(len(ds)) if ds[i].final_training_mask]
        assert len(training) == 8
        non_training = [ds[i] for i in range(len(ds)) if not ds[i].final_training_mask]
        assert len(non_training) == 4

    def test_response_offsets_sanity(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)
        ds = RepresentationDataset(ds_dir)
        offsets = ds.response_offsets()

        assert len(offsets) == 4
        assert offsets["housing"][0] == 0
        housing_dim = len(offsets["housing"][2])
        assert housing_dim == 6
        total_dim = sum(end - start for start, end, _ in offsets.values())
        assert total_dim == RESPONSE_DIM

    def test_response_similarity_identical(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)
        ds = RepresentationDataset(ds_dir)

        sim = ds.response_similarity(0, 0)
        assert np.isfinite(sim)

    def test_response_similarity_within_family(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)
        ds = RepresentationDataset(ds_dir)

        sim = ds.response_similarity(0, 1, within_family="housing")
        assert -1.0 <= sim <= 1.0

    def test_training_orders_subset(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)
        ds = RepresentationDataset(ds_dir)

        training = ds.training_orders()
        assert len(training) == 8
        assert all(o in ds.treatment_orders() for o in training)

    def test_collate_batch(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)
        ds = RepresentationDataset(ds_dir, split="train")

        batch = collate_samples([ds[i] for i in range(2)])
        features = batch["features"]
        responses = batch["responses"]
        city_keys = batch["city_keys"]
        final_mask = batch["final_training_mask"]
        assert isinstance(features, torch.Tensor)
        assert isinstance(responses, torch.Tensor)
        assert isinstance(city_keys, list)
        assert isinstance(final_mask, torch.Tensor)
        assert features.shape == (2, ds.feature_dim())
        assert responses.shape == (2, RESPONSE_DIM)
        assert len(city_keys) == 2
        assert final_mask.dtype == torch.bool

    def test_training_only_split_ignores_holdout(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)
        train_ds = RepresentationDataset(ds_dir, split="train")

        for i in range(len(train_ds)):
            assert train_ds[i].split == "train"
