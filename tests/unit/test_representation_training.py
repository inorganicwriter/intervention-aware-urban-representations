from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tests.helpers.representation import build_synthetic_model_inputs
from urban_intervention.representation.dataset import RepresentationDataset, collate_samples
from urban_intervention.representation.encoder import TabularEncoder
from urban_intervention.representation.loss import (
    _pairwise_response_similarity,
    combined_representation_loss,
    embedding_distance_loss,
    prediction_loss,
    response_alignment_loss,
    total_loss,
)
from urban_intervention.representation.model import MultiHeadPredictor, ResponseEmbeddingModel


class TestTabularEncoder:
    def test_forward_shape(self) -> None:
        encoder = TabularEncoder(feature_dim=18, embedding_dim=64)
        x = torch.randn(4, 18)
        out = encoder(x)
        assert out.shape == (4, 64)

    def test_output_normalized(self) -> None:
        encoder = TabularEncoder(feature_dim=12, embedding_dim=32)
        x = torch.randn(8, 12)
        out = encoder(x)
        norms = out.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_deterministic_in_eval(self) -> None:
        encoder = TabularEncoder(feature_dim=10, embedding_dim=16)
        encoder.eval()
        x = torch.randn(3, 10)
        out1 = encoder(x)
        out2 = encoder(x)
        assert torch.allclose(out1, out2)

    def test_different_batch_sizes(self) -> None:
        encoder = TabularEncoder(feature_dim=20, embedding_dim=128)
        encoder.eval()
        for n in (1, 4, 16):
            x = torch.randn(n, 20)
            out = encoder(x)
            assert out.shape == (n, 128)


class TestResponseSimilarity:
    def test_identical_vectors_perfect_correlation(self) -> None:
        responses = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float32)
        mask = torch.ones(2, 3, dtype=torch.bool)
        sim = _pairwise_response_similarity(responses, mask)
        assert sim.shape == (2, 2)
        assert abs(sim[0, 1] - 1.0) < 0.05

    def test_anti_correlated_vectors(self) -> None:
        responses = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=torch.float32)
        mask = torch.ones(2, 3, dtype=torch.bool)
        sim = _pairwise_response_similarity(responses, mask)
        assert sim[0, 1] < 0.0

    def test_diagonal_is_one(self) -> None:
        responses = torch.randn(4, 27) * 0.5 + 0.1
        mask = torch.ones(4, 27, dtype=torch.bool)
        sim = _pairwise_response_similarity(responses, mask)
        diag = sim.diag()
        assert torch.allclose(diag, torch.ones(4), atol=0.05)

    def test_symmetric(self) -> None:
        responses = torch.randn(8, 27)
        mask = torch.ones(8, 27, dtype=torch.bool)
        sim = _pairwise_response_similarity(responses, mask)
        assert torch.allclose(sim, sim.T, atol=1e-5)


class TestLossFunctions:
    @pytest.fixture
    def batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = torch.randn(16, 128)
        emb = emb / emb.norm(p=2, dim=1, keepdim=True)
        resp = torch.randn(16, 27) * 0.3 + 0.1
        mask = torch.ones(16, 27, dtype=torch.bool)
        se = torch.full((16, 27), 0.05)
        return emb, resp, mask, se

    def test_alignment_loss_finite(self, batch: tuple) -> None:
        emb, resp, mask, _ = batch
        loss = response_alignment_loss(emb, resp, mask)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_combined_rep_loss_finite(self, batch: tuple) -> None:
        emb, resp, mask, _ = batch
        loss = combined_representation_loss(emb, resp, mask)
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_prediction_loss_finite(self, batch: tuple) -> None:
        emb, resp, mask, se = batch
        predictor = MultiHeadPredictor(128, 64)
        preds = predictor(emb)
        loss = prediction_loss(preds, resp, mask, se)
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_total_loss_components(self, batch: tuple) -> None:
        emb, resp, mask, se = batch
        predictor = MultiHeadPredictor(128, 64)
        preds = predictor(emb)
        losses = total_loss(emb, preds, resp, mask, se)
        assert "total" in losses
        assert "representation" in losses
        assert "prediction" in losses
        assert losses["total"] > 0

    def test_total_loss_without_se_shrinkage(self, batch: tuple) -> None:
        emb, resp, mask, se = batch
        predictions = MultiHeadPredictor(128, 64)(emb)
        losses = total_loss(
            emb,
            predictions,
            resp,
            mask,
            se,
            se_shrinkage=False,
        )
        assert torch.isfinite(losses["total"])

    def test_self_logit_is_excluded(self) -> None:
        embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        responses = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
        mask = torch.ones_like(responses, dtype=torch.bool)
        loss = response_alignment_loss(embeddings, responses, mask)
        assert abs(float(loss)) < 1e-6

    def test_incomparable_pairs_do_not_enter_distance_loss(self) -> None:
        embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        responses = torch.tensor([[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]])
        mask = torch.tensor(
            [[True, True, False, False], [False, False, True, True]]
        )
        loss = embedding_distance_loss(embeddings, responses, mask)
        assert abs(float(loss)) < 1e-6

    def test_singleton_representation_loss_is_finite(self) -> None:
        embeddings = torch.tensor([[1.0, 0.0]], requires_grad=True)
        responses = torch.tensor([[1.0, 2.0]])
        mask = torch.ones_like(responses, dtype=torch.bool)
        loss = combined_representation_loss(embeddings, responses, mask)
        assert torch.isfinite(loss)
        loss.backward()

    def test_alignment_loss_lower_for_response_clusters(self) -> None:
        responses = torch.tensor(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.1], [3.0, 2.0, 1.0], [3.1, 2.0, 1.0]]
        )
        mask = torch.ones_like(responses, dtype=torch.bool)
        aligned = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        )
        scattered = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        )
        assert response_alignment_loss(aligned, responses, mask) < response_alignment_loss(
            scattered, responses, mask
        )


class TestResponseEmbeddingModel:
    def test_forward_tabular_only(self) -> None:
        model = ResponseEmbeddingModel(feature_dim=18, embedding_dim=64, use_image_encoder=False)
        x = torch.randn(4, 18)
        emb, preds = model(x)
        assert emb.shape == (4, 64)
        assert emb.norm(p=2, dim=1).sub(1.0).abs().max() < 1e-4
        assert len(preds) == 4
        assert "housing" in preds
        assert "viirs" in preds
        assert "population" in preds
        assert "poi" in preds

    def test_prediction_head_dims(self) -> None:
        model = ResponseEmbeddingModel(feature_dim=18, embedding_dim=64, use_image_encoder=False)
        x = torch.randn(8, 18)
        _, preds = model(x)
        assert preds["housing"].shape == (8, 6)
        assert preds["viirs"].shape == (8, 6)
        assert preds["population"].shape == (8, 3)
        assert preds["poi"].shape == (8, 12)

    def test_forward_features_only(self) -> None:
        model = ResponseEmbeddingModel(feature_dim=18, embedding_dim=64, use_image_encoder=False)
        x = torch.randn(4, 18)
        emb = model.forward_features_only(x)
        assert emb.shape == (4, 64)

    def test_encode_tabular_and_image_separate(self) -> None:
        model = ResponseEmbeddingModel(feature_dim=18, embedding_dim=64, use_image_encoder=False)
        x = torch.randn(4, 18)
        tab = model.encode_tabular(x)
        assert tab.shape == (4, 64)
        img = torch.randn(4, 3, 224, 224)
        img_emb = model.encode_image(img)
        assert img_emb.shape == (4, 64)


class TestTrainer:
    def test_training_loop_runs(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)

        from urban_intervention.representation.trainer import train_representation

        ckpt = train_representation(
            model_inputs_dir=ds_dir,
            output_dir=tmp_path / "checkpoints",
            embedding_dim=32,
            hidden_dims=(64,),
            dropout=0.0,
            batch_size=8,
            epochs=5,
            use_images=False,
            seed=123,
        )
        assert ckpt.exists()
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        assert "model_state_dict" in state
        assert state["feature_dim"] > 0
        assert state["embedding_dim"] == 32

        history_path = tmp_path / "checkpoints" / "training_history.json"
        assert history_path.exists()

    def test_training_with_prediction_loss(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)

        from urban_intervention.representation.trainer import train_representation

        ckpt = train_representation(
            model_inputs_dir=ds_dir,
            output_dir=tmp_path / "checkpoints2",
            embedding_dim=24,
            hidden_dims=(32,),
            dropout=0.0,
            batch_size=8,
            epochs=5,
            pred_weight=0.7,
            rep_alpha=0.5,
            use_images=False,
            seed=42,
        )
        assert ckpt.exists()
        history_path = tmp_path / "checkpoints2" / "training_history.json"
        assert history_path.exists()

    def test_test_split_evaluation(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)

        from urban_intervention.representation.trainer import train_representation

        train_representation(
            model_inputs_dir=ds_dir,
            output_dir=tmp_path / "checkpoints3",
            embedding_dim=16,
            hidden_dims=(32,),
            dropout=0.0,
            batch_size=8,
            epochs=3,
            use_images=False,
            seed=42,
        )
        test_path = tmp_path / "checkpoints3" / "test_metrics.json"
        assert test_path.exists()
        metrics = json.loads(test_path.read_text())
        assert "total" in metrics

    def test_image_batch_shapes(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)
        ds = RepresentationDataset(ds_dir, split="train", load_images=False)

        batch = collate_samples(
            [ds[0], ds[1]],
            load_images_fn=None,
            max_images_per_grid=4,
            use_images=False,
        )
        assert "images" not in batch
        assert "image_mask" not in batch

    def test_sample_has_image_paths(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)
        ds = RepresentationDataset(ds_dir, split="train")
        sample = ds[0]
        assert isinstance(sample.image_paths, list)

    def test_training_mask_only_filtering(self, tmp_path: Path) -> None:
        ds_dir, n_grids = build_synthetic_model_inputs(tmp_path)
        ds = RepresentationDataset(ds_dir, split="train", only_training_mask=True)
        orders = ds.treatment_orders()
        assert orders
        for order in orders:
            assert bool(ds._records.loc[order].get("final_training_mask", False))
        assert len(orders) < n_grids
