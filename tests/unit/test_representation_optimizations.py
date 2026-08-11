"""Tests for the algorithmic optimizations of representation learning.

Covers: SE-aware similarity shrinkage, the MemoryBank-style negative queue,
the learnable InfoNCE temperature, and uncertainty-weighted multi-task loss.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.helpers.representation import build_synthetic_model_inputs
from urban_intervention.representation.export import export_embeddings
from urban_intervention.representation.loss import (
    combined_representation_loss,
    embedding_distance_loss,
    response_alignment_loss,
    total_loss,
)
from urban_intervention.representation.model import MultiHeadPredictor, ResponseEmbeddingModel
from urban_intervention.representation.queue import EmbeddingQueue


def _normalized(rows: int, dim: int, seed: int = 1) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    emb = torch.from_numpy(rng.randn(rows, dim).astype(np.float32))
    return emb / emb.norm(p=2, dim=1, keepdim=True)


class TestSeShrinkage:
    def test_high_se_lowers_reliability(self) -> None:
        mask = torch.ones(2, 4, dtype=torch.bool)
        precise = torch.full((2, 4), 0.01)
        noisy = torch.full((2, 4), 10.0)
        from urban_intervention.representation.loss import _pair_reliability

        r_precise = _pair_reliability(precise, mask)
        r_noisy = _pair_reliability(noisy, mask)
        assert float(r_noisy[0, 1]) < float(r_precise[0, 1])
        assert float(r_precise[0, 1]) > 0.99

    def test_missing_and_nonpositive_se_are_neutral(self) -> None:
        mask = torch.ones(2, 4, dtype=torch.bool)
        se = torch.tensor([[float("nan"), 0.0, -1.0, float("inf")]] * 2)
        from urban_intervention.representation.loss import _pair_reliability

        reliability = _pair_reliability(se, mask)
        assert torch.allclose(reliability, torch.ones(2, 2), atol=1e-6)

    def test_noisy_pair_distance_target_is_shrunk(self) -> None:
        emb = torch.ones(2, 8) / torch.tensor(8.0**0.5)
        responses = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float32)
        mask = torch.ones(2, 3, dtype=torch.bool)
        precise = torch.full((2, 3), 0.01)
        noisy = torch.full((2, 3), 10.0)
        loss_precise = embedding_distance_loss(emb, responses, mask, response_se=precise)
        loss_noisy = embedding_distance_loss(emb, responses, mask, response_se=noisy)
        assert loss_noisy.item() > loss_precise.item()
        # Without SE the loss is exactly the standard one.
        loss_plain = embedding_distance_loss(emb, responses, mask)
        assert abs(loss_precise.item() - loss_plain.item()) < 1e-3

    def test_uniform_se_leaves_alignment_loss_invariant(self) -> None:
        emb = _normalized(8, 16, seed=5)
        responses = torch.from_numpy(np.random.RandomState(6).randn(8, 27).astype(np.float32))
        mask = torch.ones(8, 27, dtype=torch.bool)
        uniform_se = torch.full((8, 27), 1.0)
        base = response_alignment_loss(emb, responses, mask)
        shrunk = response_alignment_loss(emb, responses, mask, response_se=uniform_se)
        assert abs(base.item() - shrunk.item()) < 1e-5

    def test_combined_loss_accepts_se(self) -> None:
        emb = _normalized(8, 16, seed=7)
        responses = torch.from_numpy(np.random.RandomState(8).randn(8, 27).astype(np.float32))
        mask = torch.ones(8, 27, dtype=torch.bool)
        se = torch.rand(8, 27) * 2
        loss = combined_representation_loss(emb, responses, mask, response_se=se)
        assert torch.isfinite(loss)


class TestEmbeddingQueue:
    def test_empty_state_is_none(self) -> None:
        queue = EmbeddingQueue(dim=16, capacity=32)
        assert queue.state() is None

    def test_fifo_drop_oldest(self) -> None:
        queue = EmbeddingQueue(dim=4, capacity=12)
        for batch_index in range(4):
            emb = torch.full((4, 4), batch_index, dtype=torch.float32)
            queue.enqueue(emb)
        assert len(queue) == 12
        state = queue.state()
        assert state is not None and state.shape == (12, 4)
        assert torch.equal(state[0], torch.full((4,), 1.0))  # batch 0 dropped
        assert torch.equal(state[-1], torch.full((4,), 3.0))  # last is newest

    def test_batch_larger_than_capacity_keeps_tail(self) -> None:
        queue = EmbeddingQueue(dim=3, capacity=6)
        emb = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        queue.enqueue(emb)
        assert len(queue) == 6
        state = queue.state()
        assert state is not None and torch.equal(state, emb[-6:])

    def test_partial_batches_fill_exact_capacity(self) -> None:
        queue = EmbeddingQueue(dim=2, capacity=10)
        queue.enqueue(torch.zeros(6, 2))
        queue.enqueue(torch.ones(6, 2))
        state = queue.state()
        assert len(queue) == 10
        assert state is not None
        assert torch.equal(state[:4], torch.zeros(4, 2))
        assert torch.equal(state[4:], torch.ones(6, 2))

    def test_labeled_state_stays_aligned(self) -> None:
        queue = EmbeddingQueue(dim=2, capacity=3)
        embeddings = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        responses = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        mask = torch.ones(4, 3, dtype=torch.bool)
        se = torch.full((4, 3), 0.1)
        ids = torch.arange(4)
        queue.enqueue(embeddings, responses, mask, se, ids)
        state = queue.labeled_state()
        assert state is not None
        assert torch.equal(state["embeddings"], embeddings[-3:])
        assert torch.equal(state["responses"], responses[-3:])
        assert torch.equal(state["ids"], ids[-3:])

    def test_enqueue_detaches(self) -> None:
        queue = EmbeddingQueue(dim=4, capacity=8)
        emb = torch.randn(2, 4, requires_grad=True)
        queue.enqueue(emb)
        state = queue.state()
        assert state is not None and not state.requires_grad

    def test_dim_mismatch_rejected(self) -> None:
        queue = EmbeddingQueue(dim=4, capacity=8)
        with pytest.raises(ValueError):
            queue.enqueue(torch.randn(2, 5))

    def test_negative_capacity_rejected(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingQueue(dim=4, capacity=-1)

    def test_queue_negatives_increase_alignment_loss(self) -> None:
        emb = _normalized(8, 16, seed=9)
        responses = torch.from_numpy(np.random.RandomState(10).randn(8, 27).astype(np.float32))
        mask = torch.ones(8, 27, dtype=torch.bool)
        queue_emb = _normalized(64, 16, seed=11)
        base = response_alignment_loss(emb, responses, mask)
        with_queue = response_alignment_loss(
            emb, responses, mask, queue_embeddings=queue_emb
        )
        assert torch.isfinite(with_queue)
        assert with_queue.item() >= base.item() - 1e-6


class TestLearnableTemperature:
    def test_initial_temperature_matches_default(self) -> None:
        model = ResponseEmbeddingModel(
            feature_dim=12, embedding_dim=16, use_image_encoder=False, learnable_temperature=True
        )
        assert model.logit_scale is not None
        assert abs(float(model.temperature().detach()) - 0.07) < 1e-5

    def test_temperature_clamped_floor(self) -> None:
        model = ResponseEmbeddingModel(
            feature_dim=12, embedding_dim=16, use_image_encoder=False, learnable_temperature=True
        )
        model.logit_scale.data.copy_(torch.tensor(20.0))
        assert abs(float(model.temperature().detach()) - 0.01) < 1e-6

    def test_fixed_temperature_returns_float(self) -> None:
        model = ResponseEmbeddingModel(
            feature_dim=12, embedding_dim=16, use_image_encoder=False
        )
        assert model.logit_scale is None
        assert model.temperature() == 0.07

    def test_forward_and_loss_with_tensor_temperature(self) -> None:
        model = ResponseEmbeddingModel(
            feature_dim=12, embedding_dim=16, use_image_encoder=False, learnable_temperature=True
        )
        emb, preds = model(torch.randn(8, 12))
        responses = torch.from_numpy(np.random.RandomState(12).randn(8, 27).astype(np.float32))
        mask = torch.ones(8, 27, dtype=torch.bool)
        se = torch.full((8, 27), 0.05)
        losses = total_loss(emb, preds, responses, mask, se, temperature=model.temperature())
        assert torch.isfinite(losses["total"])


class TestUncertaintyWeighted:
    def test_initial_weights_match_default_balance(self) -> None:
        model = ResponseEmbeddingModel(
            feature_dim=12,
            embedding_dim=16,
            use_image_encoder=False,
            uncertainty_weighted=True,
        )
        assert model.log_var_rep is not None and model.log_var_pred is not None
        assert abs(float(torch.exp(-model.log_var_rep).detach()) - 0.5) < 1e-5
        assert abs(float(torch.exp(-model.log_var_pred).detach()) - 0.5) < 1e-5

    def test_total_loss_uses_learned_weights_and_regularizer(self) -> None:
        emb = _normalized(8, 16, seed=13)
        responses = torch.from_numpy(np.random.RandomState(14).randn(8, 27).astype(np.float32))
        mask = torch.ones(8, 27, dtype=torch.bool)
        se = torch.full((8, 27), 0.05)
        preds = MultiHeadPredictor(16, 8)(emb)
        log_var_rep = torch.tensor(np.log(2.0), dtype=torch.float32, requires_grad=True)
        log_var_pred = torch.tensor(np.log(2.0), dtype=torch.float32, requires_grad=True)
        losses = total_loss(
            emb,
            preds,
            responses,
            mask,
            se,
            pred_weight=0.99,
            log_var_rep=log_var_rep,
            log_var_pred=log_var_pred,
        )
        assert "rep_weight" in losses and "pred_weight" in losses
        assert abs(float(losses["rep_weight"]) - 0.5) < 1e-3
        expected = (
            torch.exp(-log_var_rep) * losses["representation"]
            + torch.exp(-log_var_pred) * losses["prediction"]
            + log_var_rep
            + log_var_pred
        )
        assert abs(losses["total"].item() - expected.item()) < 1e-6

    def test_log_vars_receive_gradients(self) -> None:
        model = ResponseEmbeddingModel(
            feature_dim=12,
            embedding_dim=16,
            use_image_encoder=False,
            uncertainty_weighted=True,
        )
        assert model.log_var_rep is not None
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        emb, preds = model(torch.randn(8, 12))
        responses = torch.from_numpy(np.random.RandomState(15).randn(8, 27).astype(np.float32))
        mask = torch.ones(8, 27, dtype=torch.bool)
        se = torch.full((8, 27), 0.05)
        losses = total_loss(
            emb,
            preds,
            responses,
            mask,
            se,
            log_var_rep=model.log_var_rep,
            log_var_pred=model.log_var_pred,
        )
        optimizer.zero_grad()
        losses["total"].backward()
        assert model.log_var_rep.grad is not None
        optimizer.step()
        assert abs(float(torch.exp(-model.log_var_rep).detach()) - 0.5) < 0.1


class TestOptimizedTrainingEndToEnd:
    def test_train_export_with_all_optimizations(self, tmp_path: Path) -> None:
        ds_dir, n_grids = build_synthetic_model_inputs(tmp_path)
        run_dir = tmp_path / "run_opt"
        train_representation_kwargs = {
            "model_inputs_dir": ds_dir,
            "output_dir": run_dir,
            "embedding_dim": 16,
            "hidden_dims": (32,),
            "dropout": 0.0,
            "batch_size": 8,
            "epochs": 2,
            "use_images": False,
            "seed": 42,
            "eval_n_perm": 10,
            "eval_n_boot": 10,
            "se_shrinkage": True,
            "queue_size": 64,
            "learnable_temperature": True,
            "uncertainty_weighted": True,
        }
        from urban_intervention.representation.trainer import train_representation

        train_representation(**train_representation_kwargs)

        checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
        assert checkpoint["learnable_temperature"] is True
        assert checkpoint["uncertainty_weighted"] is True

        config = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
        assert config["algorithm"] == {
            "se_shrinkage": True,
            "queue_size": 64,
            "learnable_temperature": True,
            "uncertainty_weighted": True,
        }

        history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
        assert "train_rep_weight" in history[0]

        report = json.loads((run_dir / "evaluation_report.json").read_text(encoding="utf-8"))
        assert "baselines" in report

        out_path = tmp_path / "emb_opt.parquet"
        export_embeddings(run_dir / "best_model.pt", ds_dir, out_path, batch_size=8)
        frame = __import__("pandas").read_parquet(out_path)
        assert len(frame) == n_grids
        embedding_columns = [c for c in frame.columns if c.startswith("emb_")]
        assert len(embedding_columns) == 16
        assert frame[embedding_columns].to_numpy().dtype == "float64"
