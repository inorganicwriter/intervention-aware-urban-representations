"""End-to-end integration test for multimodal causal representation learning.

Validates the complete pipeline:
1. Data loading with strictly pre-treatment AlphaEarth satellite inputs.
2. Temporal proxy masking verification (events opened <= 2017 masked, events opened >= 2018 active).
3. Formal training with Soft Causal Anchoring Huber loss (multimodal_causal).
4. Checkpoint saving with loss_config metadata.
5. Checkpoint loading and inference verification.
6. Post-training evaluation report generation (Level 1 trajectory prediction + Level 2 semantic classification).
7. Legacy contrastive baseline execution side-by-side.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tests.helpers.representation import build_synthetic_model_inputs
from urban_intervention.representation.dataset import RepresentationDataset, collate_samples
from urban_intervention.representation.model import ResponseEmbeddingModel
from urban_intervention.representation.trainer import train_representation


def test_multimodal_dataset_satellite_temporal_masking(tmp_path: Path) -> None:
    """Verify that temporal proxy satellite inputs are strictly masked when allow_satellite_proxy=False."""
    ds_dir, n_grids = build_synthetic_model_inputs(
        tmp_path, n_grids=24, n_cities=8, include_satellite=True
    )

    # 1. Strict mode (default: allow_satellite_proxy=False)
    ds_strict = RepresentationDataset(ds_dir, split="train", allow_satellite_proxy=False)
    assert len(ds_strict) > 0

    # Ensure satellite bands A00..A63 are NOT leaked into tabular feature columns
    for col in ds_strict.feature_columns:
        assert not col.startswith("A")
        assert not col.startswith("satellite_")

    proxy_count = 0
    valid_count = 0
    for sample in ds_strict:
        if sample.satellite_is_temporal_proxy:
            proxy_count += 1
            # Proxy must be masked out
            assert sample.satellite_available is False
            assert torch.all(sample.satellite_vector == 0.0)
        else:
            valid_count += 1
            # Valid must be preserved
            assert sample.satellite_available is True
            assert torch.any(sample.satellite_vector != 0.0)

    assert proxy_count > 0, "Synthetic dataset must contain proxy units"
    assert valid_count > 0, "Synthetic dataset must contain valid units"

    # Verify collate handles satellite fields properly
    batch = collate_samples([ds_strict[0], ds_strict[1]])
    assert "satellite_vectors" in batch
    assert "satellite_mask" in batch
    assert "satellite_is_temporal_proxy" in batch
    assert batch["satellite_vectors"].shape == (2, 64)
    assert batch["satellite_mask"].shape == (2,)

    # 2. Retrospective exploration mode (allow_satellite_proxy=True)
    ds_permissive = RepresentationDataset(ds_dir, split="train", allow_satellite_proxy=True)
    for sample in ds_permissive:
        assert sample.satellite_available is True
        assert torch.any(sample.satellite_vector != 0.0)


def test_multimodal_causal_end_to_end_pipeline(tmp_path: Path) -> None:
    """Complete end-to-end verification: Data Loading -> Training -> Save -> Load -> Evaluation."""
    ds_dir, n_grids = build_synthetic_model_inputs(
        tmp_path, n_grids=24, n_cities=8, include_satellite=True
    )
    output_dir = tmp_path / "checkpoints_multimodal_causal"

    # Step 1: Train model with Soft Causal Anchoring Huber loss and multimodal encoders
    best_ckpt = train_representation(
        model_inputs_dir=ds_dir,
        output_dir=output_dir,
        embedding_dim=32,
        hidden_dims=(64,),
        dropout=0.0,
        batch_size=8,
        epochs=3,
        use_images=False,
        use_satellite=True,
        use_cross_view_synergy=True,
        loss_type="multimodal_causal",
        loss_anchor_weight=0.85,
        loss_align_weight=0.15,
        loss_huber_delta=1.0,
        allow_satellite_proxy=False,
        run_baselines=True,
        run_transfer=True,
        seed=42,
    )

    # Step 2: Verify saved checkpoint and configuration artifacts
    assert best_ckpt.exists()
    assert (output_dir / "training_config.json").exists()
    assert (output_dir / "training_history.json").exists()
    assert (output_dir / "test_metrics.json").exists()
    assert (output_dir / "evaluation_report.json").exists()

    config = json.loads((output_dir / "training_config.json").read_text(encoding="utf-8"))
    assert config["architecture"]["use_satellite_encoder"] is True
    assert config["architecture"]["use_cross_view_synergy"] is True
    assert config["loss_config"]["loss_type"] == "multimodal_causal"
    assert config["loss_config"]["anchor_weight"] == 0.85
    assert config["loss_config"]["align_weight"] == 0.15
    assert config["loss_config"]["huber_delta"] == 1.0
    assert config["loss_config"]["allow_satellite_proxy"] is False

    history = json.loads((output_dir / "training_history.json").read_text(encoding="utf-8"))
    assert len(history) == 3
    for epoch_record in history:
        assert "train_total" in epoch_record
        assert "val_total" in epoch_record
        assert "train_pred" in epoch_record
        assert "train_rep" in epoch_record

    # Step 3: Load checkpoint and verify inference
    ckpt_data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    assert ckpt_data["embedding_dim"] == 32
    assert ckpt_data["use_satellite_encoder"] is True
    assert ckpt_data["use_cross_view_synergy"] is True
    assert ckpt_data["loss_config"]["loss_type"] == "multimodal_causal"

    loaded_model = ResponseEmbeddingModel(
        feature_dim=ckpt_data["feature_dim"],
        embedding_dim=ckpt_data["embedding_dim"],
        hidden_dims=(64,),
        dropout=0.0,
        use_image_encoder=False,
        use_satellite_encoder=True,
        use_cross_view_synergy=True,
    )
    loaded_model.load_state_dict(ckpt_data["model_state_dict"])
    loaded_model.eval()

    # Inference forward pass check
    test_ds = RepresentationDataset(ds_dir, split="test", allow_satellite_proxy=False)
    test_batch = collate_samples([test_ds[i] for i in range(min(4, len(test_ds)))])
    with torch.no_grad():
        emb, preds = loaded_model(
            features=test_batch["features"],
            satellite_vectors=test_batch["satellite_vectors"],
            satellite_mask=test_batch["satellite_mask"],
        )
    assert emb.shape == (len(test_batch["features"]), 32)
    norms = emb.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert "viirs" in preds
    assert "poi" in preds
    assert "housing" in preds
    assert "population" in preds

    # Step 4: Verify post-training evaluation report (Level 1 & Level 2 metrics)
    eval_report = json.loads((output_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    assert "config" in eval_report

    # Check validation or test split report
    eval_split_key = "validation" if "validation" in eval_report else "test"
    split_entry = eval_report[eval_split_key]
    assert split_entry["n_units"] > 0
    assert "retrieval" in split_entry
    assert "raw_feature_baseline" in split_entry

    # Level 1: Trajectory Prediction Metrics
    assert "trajectory_prediction" in split_entry
    traj_metrics = split_entry["trajectory_prediction"]
    assert "overall" in traj_metrics
    assert "mean_mse" in traj_metrics["overall"]
    assert "mean_pearson_r" in traj_metrics["overall"]
    assert "directional_accuracy" in traj_metrics["overall"]
    for family in ("viirs", "poi", "housing", "population"):
        if family in traj_metrics:
            assert "mse" in traj_metrics[family]
            assert "pearson_r" in traj_metrics[family]
            assert "directional_accuracy" in traj_metrics[family]

    # Level 2: Planning Semantic Classification Metrics
    assert "semantic_classification" in split_entry
    sem_metrics = split_entry["semantic_classification"]
    for family in ("viirs", "poi", "housing", "population"):
        if family in sem_metrics:
            assert "accuracy" in sem_metrics[family]
            assert "macro_f1" in sem_metrics[family]
            assert "cohens_kappa" in sem_metrics[family]
            assert "class_f1" in sem_metrics[family]
            # Canonical planning regimes
            assert "positive" in sem_metrics[family]["class_f1"]
            assert "neutral_or_uncertain" in sem_metrics[family]["class_f1"]
            assert "negative" in sem_metrics[family]["class_f1"]


def test_legacy_contrastive_baseline_comparison_run(tmp_path: Path) -> None:
    """Verify that legacy InfoNCE total_loss baseline can run in a separate directory as a control."""
    ds_dir, n_grids = build_synthetic_model_inputs(
        tmp_path, n_grids=16, n_cities=4, include_satellite=False
    )
    output_dir_baseline = tmp_path / "checkpoints_baseline_contrastive"

    ckpt = train_representation(
        model_inputs_dir=ds_dir,
        output_dir=output_dir_baseline,
        embedding_dim=16,
        hidden_dims=(32,),
        dropout=0.0,
        batch_size=8,
        epochs=2,
        use_images=False,
        use_satellite=False,
        loss_type="contrastive_legacy",
        seed=101,
    )
    assert ckpt.exists()
    config = json.loads((output_dir_baseline / "training_config.json").read_text(encoding="utf-8"))
    assert config["loss_config"]["loss_type"] == "contrastive_legacy"
