from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from urban_intervention.causal.pretraining_dataset import (
    STC_CONTEXT_TABULAR_COLUMNS,
    STC_INTERVENTION_COLUMNS,
    STC_STATE_TABULAR_COLUMNS,
)
from urban_intervention.representation.dataset import (
    GridSample,
    RepresentationDataset,
    collate_samples,
)
from urban_intervention.representation.encoder import (
    StateEncoder,
)
from urban_intervention.representation.export import export_embeddings
from urban_intervention.representation.model import STCResponseModel
from urban_intervention.representation.training.runner import train_representation


def test_stc_encoder_invariance_fixed_s_changes_tc() -> None:
    """Fixed S, changing T/C leaves the state embedding unchanged."""
    torch.manual_seed(42)
    model = STCResponseModel(
        state_tabular_dim=15,
        intervention_dim=7,
        context_tabular_dim=len(STC_CONTEXT_TABULAR_COLUMNS),
        alphaearth_dim=64,
        z_dim=64,
        t_dim=32,
        b_dim=64,
        outcome_dim=6,
        outcome_family="viirs",
        use_streetview=True,
        use_satellite=True,
    )
    model.eval()

    B = 4
    state_tab = torch.randn(B, 15)
    state_mask = torch.ones(B, 15, dtype=torch.bool)
    sv_features = torch.randn(B, 2, 384)
    sv_mask = torch.ones(B, 2, dtype=torch.bool)
    sat_images = torch.rand(B, 6, 6, 17, 17)
    sat_pixel_mask = torch.ones(B, 6, 1, 17, 17, dtype=torch.bool)
    sat_year_mask = torch.ones(B, 6, dtype=torch.bool)

    # Base T and C
    t1 = torch.randn(B, 7)
    t_mask1 = torch.ones(B, 7, dtype=torch.bool)
    c1 = torch.randn(B, len(STC_CONTEXT_TABULAR_COLUMNS))
    c_mask1 = torch.ones(B, len(STC_CONTEXT_TABULAR_COLUMNS), dtype=torch.bool)
    ae1 = torch.randn(B, 64)
    ae_avail1 = torch.ones(B, dtype=torch.bool)

    # Altered T and C
    t2 = t1 + 100.0
    c2 = c1 - 50.0
    ae2 = ae1 * 3.0

    with torch.no_grad():
        z1, pred1, fused1 = model(
            state_tabular=state_tab,
            state_field_mask=state_mask,
            intervention=t1,
            intervention_mask=t_mask1,
            context_tabular=c1,
            context_field_mask=c_mask1,
            alphaearth=ae1,
            alphaearth_available=ae_avail1,
            streetview_features=sv_features,
            streetview_mask=sv_mask,
            satellite_images=sat_images,
            satellite_pixel_mask=sat_pixel_mask,
            satellite_year_mask=sat_year_mask,
        )

        z2, pred2, fused2 = model(
            state_tabular=state_tab,
            state_field_mask=state_mask,
            intervention=t2,
            intervention_mask=t_mask1,
            context_tabular=c2,
            context_field_mask=c_mask1,
            alphaearth=ae2,
            alphaearth_available=ae_avail1,
            streetview_features=sv_features,
            streetview_mask=sv_mask,
            satellite_images=sat_images,
            satellite_pixel_mask=sat_pixel_mask,
            satellite_year_mask=sat_year_mask,
        )

    assert torch.allclose(z1, z2, atol=1e-6)
    assert not torch.allclose(fused1, fused2, atol=1e-3)
    assert not torch.allclose(pred1["viirs"], pred2["viirs"], atol=1e-3)


def test_stc_mask_perturbation_invariance() -> None:
    """Modifying masked-out pixels, masked years, or masked tabular/context fields must produce 0.0 change."""
    torch.manual_seed(123)
    model = STCResponseModel(
        state_tabular_dim=15,
        intervention_dim=7,
        context_tabular_dim=len(STC_CONTEXT_TABULAR_COLUMNS),
        alphaearth_dim=64,
        z_dim=64,
        t_dim=32,
        b_dim=64,
        outcome_dim=6,
        outcome_family="viirs",
        use_streetview=True,
        use_satellite=True,
    )
    model.eval()

    B = 2
    state_tab = torch.randn(B, 15)
    state_mask = torch.ones(B, 15, dtype=torch.bool)
    # Mask out the last 5 fields
    state_mask[:, 10:] = False

    t = torch.randn(B, 7)
    t_mask = torch.ones(B, 7, dtype=torch.bool)
    # Mask out the last 2 intervention features
    t_mask[:, 5:] = False

    c = torch.randn(B, len(STC_CONTEXT_TABULAR_COLUMNS))
    c_mask = torch.ones(B, len(STC_CONTEXT_TABULAR_COLUMNS), dtype=torch.bool)
    # Mask out first 4 context features
    c_mask[:, :4] = False

    ae = torch.randn(B, 64)
    # Mask out alphaearth for row 1
    ae_avail = torch.tensor([True, False], dtype=torch.bool)

    sat_images = torch.rand(B, 6, 6, 17, 17)
    sat_pixel_mask = torch.ones(B, 6, 1, 17, 17, dtype=torch.bool)
    # Mask out pixels in top corner of year 0
    sat_pixel_mask[:, 0, :, :5, :5] = False
    sat_year_mask = torch.tensor([[True, True, False, True, False, True], [False, True, True, True, True, False]], dtype=torch.bool)

    with torch.no_grad():
        z_orig, pred_orig, fused_orig = model(
            state_tabular=state_tab,
            state_field_mask=state_mask,
            intervention=t,
            intervention_mask=t_mask,
            context_tabular=c,
            context_field_mask=c_mask,
            alphaearth=ae,
            alphaearth_available=ae_avail,
            satellite_images=sat_images,
            satellite_pixel_mask=sat_pixel_mask,
            satellite_year_mask=sat_year_mask,
        )

        # Perturb the masked-out values with wild values
        state_tab_perturbed = state_tab.clone()
        state_tab_perturbed[:, 10:] += 9999.0

        t_perturbed = t.clone()
        t_perturbed[:, 5:] -= 8888.0

        c_perturbed = c.clone()
        c_perturbed[:, :4] += 7777.0

        ae_perturbed = ae.clone()
        ae_perturbed[1] = ae_perturbed[1] + 6666.0  # ae_avail is False for row 1

        sat_images_perturbed = sat_images.clone()
        # Perturb masked pixels
        sat_images_perturbed[:, 0, :, :5, :5] += 5555.0
        # Perturb masked years
        for b in range(B):
            for yr in range(6):
                if not sat_year_mask[b, yr]:
                    sat_images_perturbed[b, yr] += 4444.0

        z_pert, pred_pert, fused_pert = model(
            state_tabular=state_tab_perturbed,
            state_field_mask=state_mask,
            intervention=t_perturbed,
            intervention_mask=t_mask,
            context_tabular=c_perturbed,
            context_field_mask=c_mask,
            alphaearth=ae_perturbed,
            alphaearth_available=ae_avail,
            satellite_images=sat_images_perturbed,
            satellite_pixel_mask=sat_pixel_mask,
            satellite_year_mask=sat_year_mask,
        )

    # Outputs must remain completely unchanged
    diff_z = (z_orig - z_pert).abs().max().item()
    diff_fused = (fused_orig - fused_pert).abs().max().item()
    diff_pred = (pred_orig["viirs"] - pred_pert["viirs"]).abs().max().item()

    assert diff_z < 1e-5, f"Mask perturbation affected z! diff: {diff_z}"
    assert diff_fused < 1e-5, f"Mask perturbation affected fused! diff: {diff_fused}"
    assert diff_pred < 1e-5, f"Mask perturbation affected prediction! diff: {diff_pred}"


def test_stc_missing_modalities_and_gradients() -> None:
    """Missing streetview, satellite years, alphaearth, or tabular features produces finite outputs and finite gradients."""
    torch.manual_seed(999)
    model = STCResponseModel(
        state_tabular_dim=15,
        intervention_dim=7,
        context_tabular_dim=len(STC_CONTEXT_TABULAR_COLUMNS),
        alphaearth_dim=64,
        z_dim=64,
        t_dim=32,
        b_dim=64,
        outcome_dim=6,
        outcome_family="viirs",
        use_streetview=True,
        use_satellite=True,
    )
    model.train()

    B = 3
    # Case 1: All modalities missing / masked
    state_tab = torch.randn(B, 15)
    state_mask = torch.zeros(B, 15, dtype=torch.bool)  # all missing
    t = torch.randn(B, 7)
    t_mask = torch.zeros(B, 7, dtype=torch.bool)
    c = torch.randn(B, len(STC_CONTEXT_TABULAR_COLUMNS))
    c_mask = torch.zeros(B, len(STC_CONTEXT_TABULAR_COLUMNS), dtype=torch.bool)
    ae = torch.randn(B, 64)
    ae_avail = torch.zeros(B, dtype=torch.bool)
    sv_mask = torch.zeros(B, 2, dtype=torch.bool)
    sv_features = torch.randn(B, 2, 384)
    sat_images = torch.randn(B, 6, 6, 17, 17)
    sat_pixel_mask = torch.zeros(B, 6, 1, 17, 17, dtype=torch.bool)
    sat_year_mask = torch.zeros(B, 6, dtype=torch.bool)  # no satellite years

    z, preds, fused = model(
        state_tabular=state_tab,
        state_field_mask=state_mask,
        intervention=t,
        intervention_mask=t_mask,
        context_tabular=c,
        context_field_mask=c_mask,
        alphaearth=ae,
        alphaearth_available=ae_avail,
        streetview_features=sv_features,
        streetview_mask=sv_mask,
        satellite_images=sat_images,
        satellite_pixel_mask=sat_pixel_mask,
        satellite_year_mask=sat_year_mask,
    )

    assert torch.isfinite(z).all(), "z contains non-finite values!"
    assert torch.isfinite(fused).all(), "fused contains non-finite values!"
    assert torch.isfinite(preds["viirs"]).all(), "predictions contain non-finite values!"

    # Test backward pass
    dummy_targets = torch.randn(B, 6)
    loss = (preds["viirs"] - dummy_targets).pow(2).mean()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains non-finite values!"


def _create_mock_stc_dataset_on_disk(tmp_path: Path, n_units: int = 16) -> Path:
    """Helper to build a complete synthetically valid STC dataset directory."""
    root = tmp_path / "stc_dataset"
    root.mkdir(parents=True, exist_ok=True)

    cities = ["city_a", "city_b", "city_c", "city_d"]
    orders = list(range(1, n_units + 1))
    city_keys = [cities[i % len(cities)] for i in range(n_units)]
    splits = ["train" if i % 4 < 2 else "validation" if i % 4 == 2 else "test" for i in range(n_units)]

    sample_index = pd.DataFrame({
        "treatment_order": orders,
        "city_key": city_keys,
        "split": splits,
        "final_training_mask": [True] * n_units,
        "quality_grade": ["gsc_same_city_pass"] * n_units,
    })

    # State inputs
    state_data = {
        "treatment_order": orders,
        "city_key": city_keys,
        "grid_id": [f"grid_{o}" for o in orders],
        "station_event_id": [f"evt_{o}" for o in orders],
        "opening_month": ["2018-05"] * n_units,
        "streetview_available": [False] * n_units,
        "streetview_assets": ["[]"] * n_units,
    }
    for col in STC_STATE_TABULAR_COLUMNS:
        state_data[col] = np.random.uniform(0, 10, n_units)
        state_data[f"z__{col}"] = np.random.normal(0, 1, n_units)
        state_data[f"{col}_available"] = True
    state_inputs = pd.DataFrame(state_data)

    # Intervention inputs
    interv_data = {
        "treatment_order": orders,
        "city_key": city_keys,
        "grid_id": [f"grid_{o}" for o in orders],
        "station_event_id": [f"evt_{o}" for o in orders],
        "opening_month": ["2018-05"] * n_units,
    }
    for col in STC_INTERVENTION_COLUMNS:
        interv_data[col] = np.random.uniform(0, 5, n_units)
        interv_data[f"{col}_available"] = True
    intervention_inputs = pd.DataFrame(interv_data)

    # Context inputs
    context_data = {
        "treatment_order": orders,
        "city_key": city_keys,
        "grid_id": [f"grid_{o}" for o in orders],
        "station_event_id": [f"evt_{o}" for o in orders],
        "opening_month": ["2018-05"] * n_units,
        "alphaearth_available": [True] * n_units,
    }
    for i in range(64):
        context_data[f"A{i:02d}"] = np.random.normal(0, 1, n_units)
    for col in STC_CONTEXT_TABULAR_COLUMNS:
        context_data[col] = np.random.uniform(0, 10, n_units)
        context_data[f"z__{col}"] = np.random.normal(0, 1, n_units)
        context_data[f"{col}_available"] = True
    context_inputs = pd.DataFrame(context_data)

    # Satellite state sequences
    seq_records = []
    for o, c in zip(orders, city_keys, strict=False):
        for yr in [-6, -5, -4, -3, -2, -1]:
            seq_records.append({
                "treatment_order": o,
                "city_key": c,
                "grid_id": f"grid_{o}",
                "station_event_id": f"evt_{o}",
                "event_time": yr,
                "target_year": 2018 + yr,
                "state_temporal_allowed": True,
                "satellite_image_available": False,
                "state_image_available": False,
                "image_path": "",
            })
    satellite_sequences = pd.DataFrame(seq_records)

    # Response targets (VIIRS 6 horizons: 1, 3, 6, 12, 18, 24)
    target_records = []
    for o in orders:
        for h in [1, 3, 6, 12, 18, 24]:
            target_records.append({
                "treatment_order": o,
                "outcome_family": "viirs",
                "outcome": "viirs_avg_asinh",
                "event_time": h,
                "causal_response_label": float(np.random.normal(0.5, 0.2)),
                "standard_error": 0.1,
                "response_weight": 1.0,
                "final_training_mask": True,
            })
    response_targets = pd.DataFrame(target_records)

    normalization = {
        "state": {col: {"mean": 0.0, "sd": 1.0, "usable": True} for col in STC_STATE_TABULAR_COLUMNS},
        "context": {col: {"mean": 0.0, "sd": 1.0, "usable": True} for col in STC_CONTEXT_TABULAR_COLUMNS},
        "intervention": {col: {"mean": 0.0, "sd": 1.0, "usable": True} for col in ["opening_year", "n_lines_at_opening", "same_month_openings"]},
    }

    manifest = {
        "schema": "urban_intervention_stc_dataset",
        "dataset_id": "test_stc_dataset_v1",
        "strict_production": False,
        "state_tabular_columns": list(STC_STATE_TABULAR_COLUMNS),
        "intervention_columns": list(STC_INTERVENTION_COLUMNS),
        "context_tabular_columns": list(STC_CONTEXT_TABULAR_COLUMNS),
        "satellite_relative_years": [-6, -5, -4, -3, -2, -1],
        "families": ["viirs"],
    }
    sample_index.to_parquet(root / "sample_index.parquet", index=False)
    state_inputs.to_parquet(root / "state_inputs.parquet", index=False)
    context_inputs.to_parquet(root / "context_inputs.parquet", index=False)
    intervention_inputs.to_parquet(root / "intervention_inputs.parquet", index=False)
    satellite_sequences.to_parquet(root / "satellite_state_sequences.parquet", index=False)
    response_targets.to_parquet(root / "response_targets.parquet", index=False)
    (root / "normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")

    import hashlib
    manifest["outputs"] = {
        fname: hashlib.sha256((root / fname).read_bytes()).hexdigest()
        for fname in [
            "state_inputs.parquet",
            "context_inputs.parquet",
            "intervention_inputs.parquet",
            "satellite_state_sequences.parquet",
            "response_targets.parquet",
            "sample_index.parquet",
            "normalization.json",
        ]
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return root


def test_stc_dataset_loading_and_collate(tmp_path: Path) -> None:
    """Verify RepresentationDataset correctly identifies STC schema and collate_samples batches all fields."""
    root = _create_mock_stc_dataset_on_disk(tmp_path, n_units=16)

    ds = RepresentationDataset(root)
    assert ds.is_stc is True
    assert len(ds) == 16
    assert ds.response_dim() == 6
    assert list(ds.response_offsets().keys()) == ["viirs"]

    sample = ds[0]
    assert sample.is_stc is True
    assert sample.state_tabular.shape == (15,)
    assert sample.intervention.shape == (7,)
    assert sample.context_tabular.shape == (len(STC_CONTEXT_TABULAR_COLUMNS),)
    assert sample.alphaearth.shape == (64,)
    assert sample.response_vector.shape == (6,)

    # Test collate
    batch = collate_samples([ds[0], ds[1], ds[2]])
    assert batch["is_stc"] is True
    assert batch["state_tabular"].shape == (3, 15)
    assert batch["intervention"].shape == (3, 7)
    assert batch["context"].shape == (3, len(STC_CONTEXT_TABULAR_COLUMNS))
    assert batch["alphaearth"].shape == (3, 64)
    assert batch["responses"].shape == (3, 6)


def test_stc_require_modalities_filter_and_fail(tmp_path: Path) -> None:
    root = _create_mock_stc_dataset_on_disk(tmp_path, n_units=8)
    with pytest.raises(ValueError, match="mandatory modality"):
        RepresentationDataset(root, load_images=False, require_streetview=True)
    with pytest.raises(ValueError, match="mandatory modality"):
        RepresentationDataset(root, load_images=False, require_satellite=True)
    with pytest.raises(ValueError, match="mandatory modality"):
        RepresentationDataset(root, load_images=False, require_multimodal=True)


def test_stc_end_to_end_train_checkpoint_export(tmp_path: Path) -> None:
    """Full end-to-end: training STC model, saving best_model.pt, restoring, and exporting embeddings."""
    root = _create_mock_stc_dataset_on_disk(tmp_path, n_units=16)
    out_dir = tmp_path / "training_output"

    ckpt_path = train_representation(
        model_inputs_dir=root,
        output_dir=out_dir,
        embedding_dim=64,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        run_baselines=False,
        run_transfer=False,
        seed=42,
    )

    assert ckpt_path.is_file()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert ckpt["is_stc"] is True
    assert ckpt["model_type"] == "stc"
    assert "stc_contract" in ckpt

    export_path = out_dir / "embeddings.parquet"
    res = export_embeddings(
        checkpoint_path=ckpt_path,
        model_inputs_dir=root,
        output_path=export_path,
        batch_size=4,
    )
    assert res.is_file()

    df = pd.read_parquet(export_path)
    assert len(df) == 16
    assert "treatment_order" in df.columns
    assert "emb_0" in df.columns
    assert "emb_63" in df.columns
    assert "fused_0" in df.columns


def test_stc_checkpoint_records_actual_streetview_branch(tmp_path: Path) -> None:
    root = _create_mock_stc_dataset_on_disk(tmp_path, n_units=8)
    out_dir = tmp_path / "image_branch_output"
    ckpt_path = train_representation(
        model_inputs_dir=root,
        output_dir=out_dir,
        embedding_dim=16,
        epochs=1,
        batch_size=4,
        use_images=True,
        run_baselines=False,
        run_transfer=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert ckpt["use_image_encoder"] is True
    export_path = out_dir / "embeddings.parquet"
    export_embeddings(
        checkpoint_path=ckpt_path,
        model_inputs_dir=root,
        output_path=export_path,
        batch_size=4,
    )
    assert export_path.is_file()


def test_regression_streetview_image_sensitivity() -> None:
    """When streetview images are provided without precomputed features, changing image pixels must alter z."""
    model = StateEncoder(tabular_dim=15, embedding_dim=32, use_satellite=False, dropout=0).eval()
    B = 2
    state_tab = torch.randn(B, 15)
    state_mask = torch.ones(B, 15, dtype=torch.bool)
    zero_feats = torch.zeros(B, 2, 384)
    sv_mask = torch.ones(B, 2, dtype=torch.bool)
    imgs1 = torch.rand(B, 2, 3, 224, 224)
    imgs2 = torch.rand(B, 2, 3, 224, 224)

    with torch.no_grad():
        z1 = model(
            state_tabular=state_tab,
            state_field_mask=state_mask,
            streetview_features=zero_feats,
            streetview_mask=sv_mask,
            streetview_images=imgs1,
        )
        z2 = model(
            state_tabular=state_tab,
            state_field_mask=state_mask,
            streetview_features=zero_feats,
            streetview_mask=sv_mask,
            streetview_images=imgs2,
        )

    diff = (z1 - z2).abs().max().item()
    assert diff > 1e-4, f"State representation z did not change when streetview images changed (diff={diff})"


def test_stc_collate_preserves_all_streetview_images_and_month_gaps() -> None:
    def sample(order: int, count: int) -> GridSample:
        gaps = torch.arange(1, count + 1, dtype=torch.float32)
        return GridSample(
            treatment_order=order,
            city_key=f"city_{order}",
            split="train",
            feature_vector=torch.zeros(15),
            response_vector=torch.zeros(1),
            response_mask=torch.ones(1, dtype=torch.bool),
            response_se=torch.zeros(1),
            response_weight=torch.ones(1),
            modality_available={"streetview": count > 0},
            final_training_mask=True,
            quality_grade="A",
            is_stc=True,
            streetview_features=torch.zeros(count, 384),
            streetview_mask=torch.ones(count, dtype=torch.bool),
            streetview_images=torch.ones(count, 3, 224, 224),
            streetview_gap_months=gaps,
            streetview_gap_compressed=gaps / (gaps + 12.0),
        )

    batch = collate_samples([sample(1, 3), sample(2, 1)])

    assert batch["streetview_images"].shape == (2, 3, 3, 224, 224)
    assert batch["streetview_mask"].tolist() == [[True, True, True], [True, False, False]]
    assert batch["streetview_gap_months"].tolist() == [[1.0, 2.0, 3.0], [1.0, 0.0, 0.0]]
    assert torch.allclose(
        batch["streetview_gap_compressed"],
        torch.tensor([[1 / 13, 2 / 14, 3 / 15], [1 / 13, 0, 0]]),
    )


def test_state_encoder_uses_per_image_opening_time_gap() -> None:
    class FeaturePassthrough(torch.nn.Module):
        def forward_features(self, features: torch.Tensor) -> torch.Tensor:
            return features[..., :16]

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            raise AssertionError("This test supplies pre-extracted image features")

    torch.manual_seed(23)
    model = StateEncoder(
        tabular_dim=15,
        embedding_dim=16,
        use_satellite=False,
        dropout=0,
        streetview_time_aware=True,
    ).eval()
    model.streetview_encoder = FeaturePassthrough()
    state = torch.zeros(1, 15)
    features = torch.randn(1, 2, 384)
    mask = torch.ones(1, 2, dtype=torch.bool)

    with torch.no_grad():
        recent_first = model(
            state_tabular=state,
            streetview_features=features,
            streetview_mask=mask,
            streetview_gap_compressed=torch.tensor([[1 / 13, 12 / 24]]),
        )
        recent_second = model(
            state_tabular=state,
            streetview_features=features,
            streetview_mask=mask,
            streetview_gap_compressed=torch.tensor([[12 / 24, 1 / 13]]),
        )

    assert torch.isfinite(recent_first).all()
    assert not torch.allclose(recent_first, recent_second, atol=1e-6)


def test_regression_satellite_tiff_normalized_convolution_mask_invariance(tmp_path: Path) -> None:
    """Modifying masked-out pixels in raw satellite TIFF must have 0.0 impact on valid resampled pixels."""
    import rasterio
    from rasterio.transform import from_origin

    H = W = 50
    valid = np.ones((H, W), dtype=bool)
    valid[:, :25] = False

    images = []
    masks = []
    ds = RepresentationDataset.__new__(RepresentationDataset)
    ds.satellite_image_size = (17, 17)

    for value in (0.0, 10000.0):
        bands = np.ones((8, H, W), dtype=np.float32)
        bands[:6, :, :25] = value
        bands[6] = valid.astype(np.float32)
        path = tmp_path / f"masked_{int(value)}.tif"
        with rasterio.open(
            path, "w", driver="GTiff", width=W, height=H, count=8, dtype="float32",
            crs="EPSG:4326", transform=from_origin(116, 40, .0001, .0001)
        ) as dst:
            dst.write(bands)
        img, mask = ds._read_satellite_tiff(path)
        images.append(img)
        masks.append(mask)

    max_change = float(((images[0] - images[1]).abs() * masks[0]).max())
    assert max_change == 0.0, f"Invalid pixels contaminated valid resampled pixels (max_change={max_change})"
    assert masks[0].sum() > 0, "Resampled valid mask is unexpectedly empty"


def test_regression_stc_dataset_rejects_identity_mismatch(tmp_path: Path) -> None:
    """RepresentationDataset must raise ValueError if role tables disagree on event identity."""
    root = _create_mock_stc_dataset_on_disk(tmp_path / "data", n_units=8)

    # Tamper city_key in context_inputs
    context_file = root / "context_inputs.parquet"
    ctx_df = pd.read_parquet(context_file)
    ctx_df.loc[0, "city_key"] = "WRONG_CITY"
    ctx_df.to_parquet(context_file, index=False)

    import pytest
    with pytest.raises(ValueError, match="context_inputs city_key does not match sample_index"):
        RepresentationDataset(root)


def test_regression_stc_dataset_rejects_future_target_year(tmp_path: Path) -> None:
    """RepresentationDataset must reject frames with target_year >= opening_year even if state_temporal_allowed is True."""
    root = _create_mock_stc_dataset_on_disk(tmp_path / "data", n_units=8)

    seq_file = root / "satellite_state_sequences.parquet"
    seq_df = pd.read_parquet(seq_file)
    # Set future year 2099 for rel_yr = -1
    pos = seq_df.index[seq_df.event_time.eq(-1)][0]
    seq_df.loc[pos, "target_year"] = 2099
    seq_df.loc[pos, "state_temporal_allowed"] = True
    seq_df.loc[pos, "state_image_available"] = True
    seq_df.to_parquet(seq_file, index=False)

    ds = RepresentationDataset(root)
    order = int(seq_df.loc[pos, "treatment_order"])
    sample = ds[ds._treatment_orders.index(order)]
    assert bool(sample.satellite_year_mask[-1]) is False, "Future target_year was accepted into satellite_year_mask"


def test_regression_export_rejects_reordered_stc_contract(tmp_path: Path) -> None:
    """export_embeddings must verify full STC contract and reject reordered context columns."""
    root = _create_mock_stc_dataset_on_disk(tmp_path / "dataset", n_units=8)
    out_dir = tmp_path / "training_output"
    ckpt_path = train_representation(
        model_inputs_dir=root,
        output_dir=out_dir,
        embedding_dim=32,
        epochs=1,
        batch_size=4,
        run_baselines=False,
        run_transfer=False,
    )

    # Reorder context columns in manifest
    manifest_file = root / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["context_tabular_columns"] = list(reversed(manifest["context_tabular_columns"]))
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    import pytest
    with pytest.raises(ValueError, match="context_tabular_columns do not match checkpoint contract"):
        export_embeddings(
            checkpoint_path=ckpt_path,
            model_inputs_dir=root,
            output_path=out_dir / "bad_export.parquet",
            batch_size=4,
        )


def test_regression_export_rejects_altered_context_values(tmp_path: Path) -> None:
    """export_embeddings must verify file hashes on disk and reject altered context values."""
    root = _create_mock_stc_dataset_on_disk(tmp_path / "dataset", n_units=8)
    out_dir = tmp_path / "training_output"
    ckpt_path = train_representation(
        model_inputs_dir=root,
        output_dir=out_dir,
        embedding_dim=32,
        epochs=1,
        batch_size=4,
        run_baselines=False,
        run_transfer=False,
    )

    ctx_file = root / "context_inputs.parquet"
    ctx_df = pd.read_parquet(ctx_file)
    ctx_df.loc[0, "z__population_log_lag1"] = 999.0
    ctx_df.to_parquet(ctx_file, index=False)

    import pytest
    with pytest.raises(ValueError, match="does not match"):
        export_embeddings(
            checkpoint_path=ckpt_path,
            model_inputs_dir=root,
            output_path=out_dir / "altered_export.parquet",
            batch_size=4,
        )


def test_regression_satellite_image_hash_mismatch_rejected(tmp_path: Path) -> None:
    """RepresentationDataset must reject satellite frames whose physical file SHA disagrees with image_sha256."""
    root = _create_mock_stc_dataset_on_disk(tmp_path / "dataset", n_units=8)
    seq_file = root / "satellite_state_sequences.parquet"
    seq_df = pd.read_parquet(seq_file)
    pos = seq_df.index[seq_df.event_time.eq(-1)][0]
    seq_df.loc[pos, "image_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
    seq_df.to_parquet(seq_file, index=False)

    ds = RepresentationDataset(root)
    order = int(seq_df.loc[pos, "treatment_order"])
    sample = ds[ds._treatment_orders.index(order)]
    assert bool(sample.satellite_year_mask[-1]) is False, "Frame with mismatched image_sha256 was accepted"


def test_strict_satellite_hash_mismatch_raises(tmp_path: Path) -> None:
    import hashlib

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    root = _create_mock_stc_dataset_on_disk(tmp_path / "dataset", n_units=8)
    seq_file = root / "satellite_state_sequences.parquet"
    seq_df = pd.read_parquet(seq_file)
    pos = seq_df.index[seq_df.event_time.eq(-1)][0]
    order = int(seq_df.loc[pos, "treatment_order"])
    image_path = tmp_path / "fixture_2017_landsat.tif"
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=17,
        height=17,
        count=8,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(116, 40, 0.0001, 0.0001),
    ) as image:
        image.write(np.ones((8, 17, 17), dtype=np.float32))
    seq_df.loc[pos, "image_path"] = str(image_path)
    seq_df.loc[pos, "state_image_available"] = True
    seq_df.loc[pos, "satellite_image_available"] = True
    seq_df.loc[pos, "image_sha256"] = "0" * 64
    seq_df.to_parquet(seq_file, index=False)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["strict_production"] = True
    manifest["outputs"]["satellite_state_sequences.parquet"] = hashlib.sha256(
        seq_file.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ds = RepresentationDataset(root)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ds[ds._treatment_orders.index(order)]


def test_regression_satellite_post_opening_filename_rejected(tmp_path: Path) -> None:
    """RepresentationDataset must reject frames pointing to files whose filename indicates post-opening year."""
    root = _create_mock_stc_dataset_on_disk(tmp_path / "dataset", n_units=8)
    seq_file = root / "satellite_state_sequences.parquet"
    seq_df = pd.read_parquet(seq_file)
    pos = seq_df.index[seq_df.event_time.eq(-1)][0]

    # Create dummy post-opening tiff: e.g. 000001_2015_landsat.tif for event opening in 2010
    post_tiff = tmp_path / "000001_2015_landsat.tif"
    post_tiff.write_bytes(b"dummy_tiff_content")
    seq_df.loc[pos, "image_path"] = str(post_tiff)
    seq_df.loc[pos, "target_year"] = 2009  # metadata claims pre-opening 2009
    seq_df.to_parquet(seq_file, index=False)

    ds = RepresentationDataset(root)
    order = int(seq_df.loc[pos, "treatment_order"])
    sample = ds[ds._treatment_orders.index(order)]
    assert bool(sample.satellite_year_mask[-1]) is False, "Frame pointing to post-opening filename was accepted"


def test_regression_streetview_alias_rejected(tmp_path: Path) -> None:
    """RepresentationDataset must reject streetview image when referenced asset belongs to different city/event or post-opening."""
    import hashlib
    root = _create_mock_stc_dataset_on_disk(tmp_path / "dataset", n_units=8)

    # Update event 1 identity consistently across tables: Shanghai, opening 2010-02
    for fname in ["sample_index.parquet", "state_inputs.parquet", "context_inputs.parquet", "intervention_inputs.parquet"]:
        df = pd.read_parquet(root / fname)
        df.loc[0, "city_key"] = "shanghai"
        if "opening_month" in df.columns:
            df.loc[0, "opening_month"] = "2010-02"
        df.to_parquet(root / fname, index=False)

    # Create dummy image on disk
    img_dir = tmp_path / "images"
    img_dir.mkdir(exist_ok=True)
    img_file = img_dir / "beijing_2013_pano.jpg"
    img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    img_sha = hashlib.sha256(img_file.read_bytes()).hexdigest()

    # Create mock streetview registry table: records image as belonging to Beijing in 2013-08
    registry_file = tmp_path / "streetview_state_assets.parquet"
    reg_df = pd.DataFrame([{
        "treatment_order": 586,
        "city_key": "beijing",
        "grid_id": "bj_grid_586",
        "opening_month": "2015-12",
        "capture_month": "2013-08",
        "asset_sha256": img_sha,
        "asset_path": str(img_file),
    }])
    reg_df.to_parquet(registry_file, index=False)

    # Point event 1 to the Beijing image
    state_file = root / "state_inputs.parquet"
    state_df = pd.read_parquet(state_file)
    state_df.loc[0, "streetview_assets"] = json.dumps([str(img_file)])
    state_df.loc[0, "streetview_available"] = True
    state_df.to_parquet(state_file, index=False)

    # Load dataset with custom streetview index path
    ds = RepresentationDataset(root, load_images=True, streetview_index_path=registry_file)
    sample = ds[0]
    # Shanghai 2010 event referencing Beijing 2013 image must have 0 loaded images
    assert int(sample.streetview_mask.sum().item()) == 0, "Cross-city post-opening streetview alias was accepted"


@pytest.fixture
def role_input_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create disposable role inputs, labels and tiny images without project outputs."""
    import rasterio
    from PIL import Image
    from rasterio.transform import from_origin

    project_root = tmp_path / "assets"
    inputs_dir = _create_mock_stc_dataset_on_disk(project_root / "role_inputs", n_units=96)
    state = pd.read_parquet(inputs_dir / "state_inputs.parquet")
    identity_columns = [
        "treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"
    ]
    treatments_dir = project_root / "data/active/causal"
    treatments_dir.mkdir(parents=True)
    state[identity_columns].to_parquet(treatments_dir / "treatment_unit_list.parquet", index=False)

    response_dir = project_root / "fixture_response"
    response_dir.mkdir()
    targets = pd.read_parquet(inputs_dir / "response_targets.parquet")
    targets = targets.drop(columns=["final_training_mask", "response_weight"])
    targets = targets.merge(state[identity_columns], on="treatment_order", validate="many_to_one")
    targets["training_mask"] = True
    targets["main_spec"] = True
    targets["quality_grade"] = "synthetic_fixture"
    targets["method"] = "frozen_matched_change"
    targets.to_parquet(response_dir / "response_artifact.parquet", index=False)

    satellite_dir = project_root / "data/active/satellite_event_sequences_v2/images"
    satellite_dir.mkdir(parents=True)
    sequences = pd.read_parquet(inputs_dir / "satellite_state_sequences.parquet")
    for year in range(2012, 2018):
        image_path = satellite_dir / f"fixture_{year}_landsat.tif"
        with rasterio.open(
            image_path, "w", driver="GTiff", width=17, height=17, count=8,
            dtype="float32", crs="EPSG:4326", transform=from_origin(116, 40, 0.0001, 0.0001),
        ) as image:
            image.write(np.ones((8, 17, 17), dtype=np.float32))
        selected = sequences.treatment_order.eq(1) & sequences.target_year.eq(year)
        sequences.loc[selected, "image_path"] = f"images/{image_path.name}"
        sequences.loc[selected, "image_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        sequences.loc[selected, "state_image_available"] = True
        sequences.loc[selected, "satellite_image_available"] = True
    sequences.to_parquet(inputs_dir / "satellite_state_sequences.parquet", index=False)

    streetview_dir = project_root / "data/active/streetview_grid_v2"
    streetview_dir.mkdir(parents=True)
    streetview_path = streetview_dir / "fixture.jpg"
    Image.new("RGB", (16, 16), color=(70, 120, 180)).save(streetview_path)
    state.loc[0, "streetview_assets"] = json.dumps([str(streetview_path)])
    state.loc[0, "streetview_available"] = True
    state.to_parquet(inputs_dir / "state_inputs.parquet", index=False)
    registry = state.loc[[0], identity_columns].copy()
    registry["capture_month"] = "2017-06"
    registry["asset_path"] = str(streetview_path)
    registry["asset_sha256"] = hashlib.sha256(streetview_path.read_bytes()).hexdigest()
    registry.to_parquet(streetview_dir / "streetview_state_assets.parquet", index=False)
    return project_root, inputs_dir, response_dir


def test_regression_subset_normalization_consistency(
    tmp_path: Path, role_input_fixture: tuple[Path, Path, Path]
) -> None:
    """Slicing a published STC dataset must preserve exact normalization and yield zero error between raw and z__* columns."""
    from scripts.analysis.verify_stc_server_smoke import build_smoke_slice

    project_root, inputs_dir, response_dir = role_input_fixture

    candidate = tmp_path / "candidate_slice"
    built = build_smoke_slice(
        stc_inputs_dir=inputs_dir,
        response_release=response_dir,
        candidate_dir=candidate,
        outcome_family="viirs",
        n_samples=64,
        project_root=project_root,
    )

    normalization = json.loads((built / "normalization.json").read_text(encoding="utf-8"))
    for role in ("state", "context", "intervention"):
        table = pd.read_parquet(built / f"{role}_inputs.parquet")
        for col, norm in normalization[role].items():
            if not norm.get("usable") or f"z__{col}" not in table:
                continue
            expected = (pd.to_numeric(table[col], errors="coerce") - norm["mean"]) / norm["sd"]
            actual_z = pd.to_numeric(table[f"z__{col}"], errors="coerce")
            valid = np.isfinite(expected) & np.isfinite(actual_z)
            if valid.any():
                max_diff = float((expected[valid] - actual_z[valid]).abs().max())
                assert max_diff < 1e-5, f"Normalization mismatch in {role}.{col}: diff={max_diff}"


def test_regression_separate_code_and_data_roots(
    tmp_path: Path, role_input_fixture: tuple[Path, Path, Path]
) -> None:
    """When code directory does not contain data/, assets are cleanly located via MIT_PROJECT_ROOT or explicit paths."""
    from scripts.analysis import verify_stc_server_smoke as smoke
    from urban_intervention.representation import dataset as dataset_mod

    project_root, inputs_dir, response_dir = role_input_fixture

    code_root = tmp_path / "candidate_code"
    code_root.mkdir(exist_ok=True)
    candidate_smoke = tmp_path / "candidate_smoke"

    saved_repo_root = dataset_mod._REPO_ROOT
    saved_registry = dataset_mod._STREETVIEW_REGISTRY
    saved_smoke_root = smoke.ROOT
    saved_env = os.environ.get("MIT_PROJECT_ROOT")
    try:
        os.environ["MIT_PROJECT_ROOT"] = str(project_root)
        dataset_mod._REPO_ROOT = code_root
        dataset_mod._STREETVIEW_REGISTRY = None
        smoke.ROOT = code_root

        # build_smoke_slice must succeed when smoke.ROOT has no data/ directory
        slice_dir = smoke.build_smoke_slice(
            stc_inputs_dir=inputs_dir,
            response_release=response_dir,
            candidate_dir=candidate_smoke,
            outcome_family="viirs",
            n_samples=64,
        )
        assert (slice_dir / "state_inputs.parquet").is_file()

        # Dataset must load streetviews without silent 0-image fallback
        ds = RepresentationDataset(slice_dir, load_images=True, verify_bundle=True)
        sv_orders = [o for o in ds._treatment_orders if ds._stc_streetview_avail.get(o, False)]
        if sv_orders:
            sample = ds[ds._treatment_orders.index(sv_orders[0])]
            assert int(sample.streetview_mask.sum().item()) > 0

        # Dataset must load satellite sequences without silent 0-year fallback
        sample0 = ds[0]
        assert int(sample0.satellite_year_mask.sum().item()) == 6
    finally:
        dataset_mod._REPO_ROOT = saved_repo_root
        dataset_mod._STREETVIEW_REGISTRY = saved_registry
        smoke.ROOT = saved_smoke_root
        if saved_env is None:
            os.environ.pop("MIT_PROJECT_ROOT", None)
        else:
            os.environ["MIT_PROJECT_ROOT"] = saved_env


def test_regression_export_rejects_altered_streetview_assets(tmp_path: Path) -> None:
    """export_embeddings must reject checkpoints if streetview_state_assets has been tampered with."""
    root = _create_mock_stc_dataset_on_disk(tmp_path / "dataset", n_units=8)

    # Write a mock streetview assets table
    sv_table = root / "streetview_state_assets.parquet"
    sv_df = pd.DataFrame([{
        "treatment_order": 1,
        "city_key": "city_0",
        "grid_id": "g1",
        "opening_month": "2015-01",
        "capture_month": "2014-06",
        "asset_sha256": "abcdef123456",
        "asset_path": str(tmp_path / "mock.jpg"),
    }])
    sv_df.to_parquet(sv_table, index=False)
    sv_sha = hashlib.sha256(sv_table.read_bytes()).hexdigest()

    manifest_file = root / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["outputs"]["streetview_state_assets.parquet"] = sv_sha
    manifest["streetview_assets_sha256"] = sv_sha
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    out_dir = tmp_path / "training_output"
    ckpt_path = train_representation(
        model_inputs_dir=root,
        output_dir=out_dir,
        embedding_dim=32,
        epochs=1,
        batch_size=4,
        run_baselines=False,
        run_transfer=False,
        streetview_index_path=sv_table,
    )

    # Tamper with streetview_state_assets on disk
    sv_df.loc[0, "asset_sha256"] = "tampered_sha"
    sv_df.to_parquet(sv_table, index=False)

    with pytest.raises(ValueError, match="does not match checkpoint"):
        export_embeddings(
            checkpoint_path=ckpt_path,
            model_inputs_dir=root,
            output_path=out_dir / "tampered_export.parquet",
            batch_size=4,
            streetview_index_path=sv_table,
        )


def test_regression_satellite_relative_paths_separate_process(
    tmp_path: Path, role_input_fixture: tuple[Path, Path, Path]
) -> None:
    """Subprocess with candidate code directory and no local data/ resolves satellite images via MIT_PROJECT_ROOT."""
    import shutil
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    project_root, inputs_dir, response_dir = role_input_fixture

    candidate_code = tmp_path / "candidate_code"
    candidate_code.mkdir(exist_ok=True)
    # Copy source only into candidate_code checkout (no data)
    shutil.copytree(
        repo_root / "src/urban_intervention",
        candidate_code / "src/urban_intervention",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (candidate_code / "scripts/analysis").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        repo_root / "scripts/analysis/verify_stc_server_smoke.py",
        candidate_code / "scripts/analysis/verify_stc_server_smoke.py",
    )

    from scripts.analysis import verify_stc_server_smoke as smoke
    bundle_dir = tmp_path / "bundle64"
    smoke.build_smoke_slice(
        stc_inputs_dir=inputs_dir,
        response_release=response_dir,
        candidate_dir=bundle_dir,
        outcome_family="viirs",
        n_samples=64,
        project_root=project_root,
    )

    probe_script = candidate_code / "probe_sat.py"
    probe_script.write_text(
        """import json, sys, torch
from pathlib import Path
from urban_intervention.representation.dataset import RepresentationDataset
bundle = Path(sys.argv[1])
ds = RepresentationDataset(bundle, load_images=True, verify_bundle=True)
sample0 = ds[0]
n_sat = int(sample0.satellite_year_mask.sum().item())
print(json.dumps({"loaded_sat_years": n_sat}))
""",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["MIT_PROJECT_ROOT"] = str(project_root)
    env["PYTHONPATH"] = str(candidate_code / "src")
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    res = subprocess.run(
        [sys.executable, str(probe_script), str(bundle_dir)],
        cwd=candidate_code,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, f"Subprocess probe failed: {res.stderr}"
    data = json.loads(res.stdout.strip().splitlines()[-1])
    assert data["loaded_sat_years"] == 6, f"Expected 6 satellite years, got {data}"





