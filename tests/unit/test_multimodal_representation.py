"""Unit tests for multimodal representation learning components.

Tests:
- SatelliteAlphaEarthEncoder
- CrossViewSynergyModule
- ResponseEmbeddingModel with multimodal cross-view synergy
- soft_causal_anchoring_loss (Huber)
- multimodal_causal_loss
- Level 1: trajectory_prediction_metrics
- Level 2: planning_semantic_classification_metrics
- Level 4: latent_manifold_umap
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from urban_intervention.representation.encoder import (
    CrossViewSynergyModule,
    SatelliteAlphaEarthEncoder,
)
from urban_intervention.representation.evaluation import (
    latent_manifold_umap,
    planning_semantic_classification_metrics,
    trajectory_prediction_metrics,
)
from urban_intervention.representation.loss import (
    multimodal_causal_loss,
    soft_causal_anchoring_loss,
)
from urban_intervention.representation.model import (
    FAMILY_DIMS,
    ResponseEmbeddingModel,
)


def test_satellite_alphaearth_encoder_shapes():
    encoder = SatelliteAlphaEarthEncoder(input_dim=64, embedding_dim=128)
    # Test 2D input (B, 64)
    x2d = torch.randn(8, 64)
    out2d = encoder(x2d)
    assert out2d.shape == (8, 128)
    norms = torch.norm(out2d, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Test 3D input (B, S, 64)
    x3d = torch.randn(4, 5, 64)
    out3d = encoder(x3d)
    assert out3d.shape == (4, 5, 128)
    norms3d = torch.norm(out3d, p=2, dim=-1)
    assert torch.allclose(norms3d, torch.ones_like(norms3d), atol=1e-5)


def test_cross_view_synergy_module():
    synergy = CrossViewSynergyModule(embedding_dim=128, num_heads=4)
    pano = torch.randn(4, 3, 128)
    sat = torch.randn(4, 2, 128)

    # Fully available
    out = synergy(pano, sat)
    assert out.shape == (4, 128)
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Partial availability with masks
    pano_mask = torch.tensor([True, False, True, False])
    sat_mask = torch.tensor([True, True, False, False])
    out_masked = synergy(pano, sat, pano_mask=pano_mask, sat_mask=sat_mask)
    assert out_masked.shape == (4, 128)
    norms_m = torch.norm(out_masked, p=2, dim=-1)
    assert torch.allclose(norms_m, torch.ones_like(norms_m), atol=1e-5)


def test_multimodal_model_forward():
    model = ResponseEmbeddingModel(
        feature_dim=27,
        embedding_dim=128,
        use_image_encoder=False,
        use_satellite_encoder=True,
        use_cross_view_synergy=True,
    )
    batch_size = 4
    features = torch.randn(batch_size, 27)
    sat = torch.randn(batch_size, 64)
    sat_mask = torch.ones(batch_size, dtype=torch.bool)

    emb, preds = model(
        features=features,
        satellite_embeddings=sat,
        satellite_mask=sat_mask,
    )
    assert emb.shape == (batch_size, 128)
    assert "viirs" in preds
    assert "poi" in preds
    assert preds["viirs"].shape == (batch_size, FAMILY_DIMS["viirs"])
    assert preds["poi"].shape == (batch_size, FAMILY_DIMS["poi"])


def test_soft_causal_anchoring_loss_huber():
    torch.manual_seed(42)
    batch_size = 8
    # Preds dictionary
    preds = {
        "viirs": torch.randn(batch_size, FAMILY_DIMS["viirs"], requires_grad=True),
        "poi": torch.randn(batch_size, FAMILY_DIMS["poi"], requires_grad=True),
    }
    # Response targets (27-d)
    responses = torch.randn(batch_size, 27)
    masks = torch.ones(batch_size, 27, dtype=torch.bool)
    weights = torch.ones(batch_size, 27, dtype=torch.float32)

    loss = soft_causal_anchoring_loss(
        predictions=preds,
        responses=responses,
        response_mask=masks,
        response_weight=weights,
        delta=1.0,
    )
    assert loss.ndim == 0
    assert loss.item() > 0.0

    # Test backpropagation
    loss.backward()
    assert preds["viirs"].grad is not None
    assert preds["poi"].grad is not None


def test_multimodal_causal_loss_balance():
    batch_size = 6
    emb = torch.randn(batch_size, 128)
    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    preds = {
        "viirs": torch.randn(batch_size, FAMILY_DIMS["viirs"]),
        "poi": torch.randn(batch_size, FAMILY_DIMS["poi"]),
    }
    responses = torch.randn(batch_size, 27)
    masks = torch.ones(batch_size, 27, dtype=torch.bool)
    weights = torch.ones(batch_size, 27, dtype=torch.float32)

    loss_dict = multimodal_causal_loss(
        embeddings=emb,
        predictions=preds,
        responses=responses,
        response_mask=masks,
        response_weight=weights,
        anchor_weight=0.85,
        align_weight=0.15,
    )
    assert "total" in loss_dict
    assert "anchor_loss" in loss_dict
    assert "align_loss" in loss_dict
    expected = 0.85 * loss_dict["anchor_loss"] + 0.15 * loss_dict["align_loss"]
    assert torch.allclose(loss_dict["total"], expected, atol=1e-5)


def test_level1_trajectory_prediction_metrics():
    from urban_intervention.representation.dataset import RESPONSE_OFFSETS

    start, end, _ = RESPONSE_OFFSETS["viirs"]
    preds = {
        "viirs": torch.tensor([[0.5, 0.2, 0.1, 0.4, 0.3, 0.6], [0.1, -0.2, 0.3, 0.1, 0.2, 0.3]]),
    }
    responses = torch.zeros(2, 27)
    responses[:2, start:end] = torch.tensor(
        [[0.4, 0.2, 0.1, 0.5, 0.3, 0.6], [0.2, -0.1, 0.3, 0.2, 0.2, 0.4]]
    )
    masks = torch.zeros(2, 27, dtype=torch.bool)
    masks[:2, start:end] = True

    metrics = trajectory_prediction_metrics(preds, responses, masks)
    assert "viirs" in metrics
    assert "overall" in metrics
    assert metrics["viirs"]["directional_accuracy"] >= 0.8
    assert metrics["overall"]["mean_mse"] < 0.1


def test_level2_planning_semantic_classification_metrics():
    from urban_intervention.representation.dataset import RESPONSE_OFFSETS

    start, end, _ = RESPONSE_OFFSETS["viirs"]
    preds = {
        "viirs": torch.tensor([
            [0.2, 0.3, 0.2, 0.3, 0.2, 0.3],     # mean ~ 0.25 (class 0: Activation)
            [-0.2, -0.3, -0.2, -0.3, -0.2, -0.3], # mean ~ -0.25 (class 2: Displacement)
            [0.01, 0.0, -0.01, 0.0, 0.01, 0.0],  # mean ~ 0.002 (class 1: No Response)
        ]),
    }
    responses = torch.zeros(3, 27)
    responses[:3, start:end] = preds["viirs"]
    masks = torch.zeros(3, 27, dtype=torch.bool)
    masks[:3, start:end] = True

    results = planning_semantic_classification_metrics(preds, responses, masks, threshold=0.05)
    assert "viirs" in results
    viirs_res = results["viirs"]
    assert viirs_res["accuracy"] == 1.0
    assert viirs_res["macro_f1"] == 1.0
    assert viirs_res["cohens_kappa"] == 1.0
    assert viirs_res["sample_count"] == 3


def test_level4_latent_manifold_umap():
    pytest.importorskip("umap", reason="Real UMAP verification requires optional umap-learn")
    embeddings = np.random.default_rng(42).normal(size=(20, 128))
    orders = list(range(1, 21))
    cities = ["beijing"] * 10 + ["shanghai"] * 10

    res = latent_manifold_umap(embeddings, orders, cities)
    assert res["method"] == "umap"
    coords = np.asarray(res["coords_2d"])
    assert coords.shape == (20, 2)
    assert np.isfinite(coords).all()
    assert res["treatment_orders"] == orders
    assert res["city_keys"] == cities


def test_level4_latent_manifold_pca_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "umap", None)
    embeddings = np.random.default_rng(42).normal(size=(20, 128))
    orders = list(range(1, 21))
    cities = ["beijing"] * 10 + ["shanghai"] * 10

    res = latent_manifold_umap(embeddings, orders, cities)
    assert res["method"] == "pca_fallback"
    coords = np.asarray(res["coords_2d"])
    assert coords.shape == (20, 2)
    assert np.isfinite(coords).all()
    assert res["treatment_orders"] == orders
    assert res["city_keys"] == cities


def test_dataset_alphaearth_loading_and_collate(tmp_path):
    from urban_intervention.representation.dataset import GridSample, collate_samples

    samples = [
        GridSample(
            treatment_order=1,
            city_key="beijing",
            split="train",
            feature_vector=torch.zeros(20),
            response_vector=torch.zeros(27),
            response_mask=torch.ones(27, dtype=torch.bool),
            response_se=torch.ones(27),
            response_weight=torch.ones(27),
            modality_available={"housing": True},
            final_training_mask=True,
            quality_grade="pass",
            satellite_vector=torch.ones(64),
            satellite_available=True,
        ),
        GridSample(
            treatment_order=2,
            city_key="shanghai",
            split="train",
            feature_vector=torch.zeros(20),
            response_vector=torch.zeros(27),
            response_mask=torch.ones(27, dtype=torch.bool),
            response_se=torch.ones(27),
            response_weight=torch.ones(27),
            modality_available={"housing": True},
            final_training_mask=True,
            quality_grade="pass",
            satellite_vector=torch.zeros(64),
            satellite_available=False,
        ),
    ]

    batch = collate_samples(samples)
    assert "satellite_vectors" in batch
    assert "satellite_mask" in batch
    assert batch["satellite_vectors"].shape == (2, 64)
    assert batch["satellite_mask"].shape == (2,)
    assert batch["satellite_mask"][0].item() is True
    assert batch["satellite_mask"][1].item() is False
