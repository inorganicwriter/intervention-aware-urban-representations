from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.helpers.representation import build_synthetic_model_inputs
from urban_intervention.representation import evaluation
from urban_intervention.representation.dataset import RESPONSE_DIM, RESPONSE_OFFSETS
from urban_intervention.representation.loss import _pairwise_response_similarity


def _cluster_pool(
    n_per_cluster: int = 8,
    dims: int = 12,
    seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two response clusters with cluster-coded embeddings (clear signal)."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, dims)
    p1 = np.sin(t).astype(np.float32)
    p2 = np.cos(t).astype(np.float32)
    responses: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    for cluster, pattern in ((0, p1), (1, p2)):
        for _ in range(n_per_cluster):
            responses.append(pattern + rng.normal(0, 0.05, dims).astype(np.float32))
            code = np.zeros(2, dtype=np.float32)
            code[cluster] = 1.0
            embeddings.append(code)
    resp = torch.from_numpy(np.stack(responses))
    emb = torch.from_numpy(np.stack(embeddings)).float()
    emb = emb / emb.norm(p=2, dim=1, keepdim=True)
    return emb, resp


class TestResponseSimilarity:
    def test_matches_legacy_implementation(self) -> None:
        rng = np.random.RandomState(0)
        resp = torch.from_numpy(rng.randn(12, RESPONSE_DIM).astype(np.float32))
        mask = torch.from_numpy(rng.rand(12, RESPONSE_DIM) > 0.3)
        legacy = _pairwise_response_similarity(resp, mask)
        refactored = evaluation.response_similarity(resp, mask)
        assert torch.allclose(legacy, refactored, atol=1e-6)

    def test_family_subset_averages_that_family_only(self) -> None:
        rng = np.random.RandomState(1)
        resp = torch.from_numpy(rng.randn(8, RESPONSE_DIM).astype(np.float32))
        mask = torch.ones(8, RESPONSE_DIM, dtype=torch.bool)
        overall = evaluation.response_similarity(resp, mask)
        poi = evaluation.response_similarity(resp, mask, families=["poi"])
        assert poi.shape == (8, 8)
        assert not torch.allclose(overall, poi, atol=1e-4)

    def test_unknown_family_rejected(self) -> None:
        resp = torch.zeros(4, RESPONSE_DIM)
        mask = torch.ones(4, RESPONSE_DIM, dtype=torch.bool)
        with pytest.raises(ValueError):
            evaluation.response_similarity(resp, mask, families=["nope"])

    def test_ragged_dimension_uses_single_segment(self) -> None:
        resp = torch.randn(5, 7)
        mask = torch.ones(5, 7, dtype=torch.bool)
        sim = evaluation.response_similarity(resp, mask)
        assert sim.shape == (5, 5)
        assert torch.allclose(sim.diag(), torch.ones(5), atol=1e-5)


class TestRetrievalMetrics:
    def test_structure_overall_and_families(self) -> None:
        emb, resp = _cluster_pool()
        mask = torch.ones_like(resp, dtype=torch.bool)
        metrics = evaluation.retrieval_metrics(emb, resp, mask, k=4)
        assert "overall" in metrics
        assert set(RESPONSE_OFFSETS) <= set(metrics)
        for block in metrics.values():
            nn_corr = block["nn_corr@k"]
            assert isinstance(nn_corr, (int, float))
            assert -1.0 <= nn_corr <= 1.0
            assert "baseline_corr" in block
            assert block["n_units"] == 16

    def test_cluster_signal_beats_random_neighbour_baseline(self) -> None:
        emb, resp = _cluster_pool()
        mask = torch.ones_like(resp, dtype=torch.bool)
        overall = evaluation.retrieval_metrics(emb, resp, mask, k=4)["overall"]
        nn_corr = overall["nn_corr@k"]
        baseline_corr = overall["baseline_corr"]
        assert isinstance(nn_corr, (int, float))
        assert isinstance(baseline_corr, (int, float))
        assert nn_corr >= 0.5
        assert nn_corr > baseline_corr

    def test_random_embeddings_hit_chance(self) -> None:
        rng = np.random.RandomState(3)
        emb = torch.from_numpy(rng.randn(32, 8).astype(np.float32))
        emb = emb / emb.norm(p=2, dim=1, keepdim=True)
        resp = torch.from_numpy(rng.randn(32, 6).astype(np.float32))
        mask = torch.ones_like(resp, dtype=torch.bool)
        overall = evaluation.retrieval_metrics(emb, resp, mask, k=5)["overall"]
        nn_corr = overall["nn_corr@k"]
        baseline_corr = overall["baseline_corr"]
        assert isinstance(nn_corr, (int, float))
        assert isinstance(baseline_corr, (int, float))
        assert abs(nn_corr - baseline_corr) < 0.05


class TestBootstrapCi:
    def test_ci_shapes_and_bounds(self) -> None:
        emb, resp = _cluster_pool(n_per_cluster=12)
        mask = torch.ones_like(resp, dtype=torch.bool)
        emb_cos = evaluation.cosine_similarity(emb)
        sim = evaluation.response_similarity(resp, mask)
        ci = evaluation.bootstrap_ci(emb_cos, sim, k=5, n_boot=100, seed=11)
        assert ci["n_boot"] == 100
        assert ci["lower"] < ci["upper"]
        point = evaluation.nn_corr_at_k(emb_cos, sim, k=5)
        assert ci["lower"] - 0.05 <= point <= ci["upper"] + 0.05

    def test_empty_pool_returns_zero(self) -> None:
        emb_cos = torch.zeros(1, 1)
        sim = torch.zeros(1, 1)
        ci = evaluation.bootstrap_ci(emb_cos, sim, k=5, n_boot=10)
        assert ci["n_boot"] == 0


class TestPermutationTest:
    def test_chance_not_significant_for_random_embeddings(self) -> None:
        rng = np.random.RandomState(5)
        emb = torch.from_numpy(rng.randn(48, 8).astype(np.float32))
        emb = emb / emb.norm(p=2, dim=1, keepdim=True)
        resp = torch.from_numpy(rng.randn(48, 10).astype(np.float32))
        mask = torch.ones_like(resp, dtype=torch.bool)
        emb_cos = evaluation.cosine_similarity(emb)
        result = evaluation.permutation_test(emb_cos, resp, mask, k=5, n_perm=99, seed=5)
        p_value = result["p_value"]
        assert isinstance(p_value, (int, float))
        assert p_value > 0.05

    def test_signal_is_significant(self) -> None:
        emb, resp = _cluster_pool(n_per_cluster=16)
        mask = torch.ones_like(resp, dtype=torch.bool)
        emb_cos = evaluation.cosine_similarity(emb)
        result = evaluation.permutation_test(emb_cos, resp, mask, k=5, n_perm=199, seed=7)
        observed = result["observed"]
        mean_null = result["mean_null"]
        p_value = result["p_value"]
        assert isinstance(observed, (int, float))
        assert isinstance(mean_null, (int, float))
        assert isinstance(p_value, (int, float))
        assert observed > mean_null
        assert p_value <= 0.05

    def test_subsampling_caps_pool(self) -> None:
        emb, resp = _cluster_pool(n_per_cluster=40)
        mask = torch.ones_like(resp, dtype=torch.bool)
        emb_cos = evaluation.cosine_similarity(emb)
        result = evaluation.permutation_test(
            emb_cos, resp, mask, k=5, n_perm=20, seed=1, max_units=32
        )
        assert result["n_units"] == 32


class TestProbeRmse:
    def test_perfect_linear_map(self) -> None:
        rng = np.random.RandomState(9)
        x = torch.from_numpy(rng.randn(40, 6).astype(np.float32))
        w = torch.from_numpy(rng.randn(6, 4).astype(np.float32))
        y = x @ w
        mask = torch.ones(40, 4, dtype=torch.bool)
        result = evaluation.probe_rmse(x, y, mask, x, y, mask, ridge=1e-6, min_obs=4)
        assert result["fitted_cells"] == 4
        rmse_overall = result["rmse_overall"]
        assert isinstance(rmse_overall, (int, float))
        assert rmse_overall < 1e-3

    def test_insufficient_support_reports_none(self) -> None:
        rng = np.random.RandomState(10)
        x = torch.from_numpy(rng.randn(6, 6).astype(np.float32))
        y = torch.from_numpy(rng.randn(6, 4).astype(np.float32))
        mask = torch.ones(6, 4, dtype=torch.bool)
        result = evaluation.probe_rmse(x, y, mask, x, y, mask, ridge=1.0, min_obs=16)
        assert result["fitted_cells"] == 0
        assert result["rmse_overall"] is None


class TestTrainerEvaluation:
    def test_evaluation_report_written(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)

        from urban_intervention.representation.trainer import train_representation

        train_representation(
            model_inputs_dir=ds_dir,
            output_dir=tmp_path / "ckpt_eval",
            embedding_dim=16,
            hidden_dims=(32,),
            dropout=0.0,
            batch_size=8,
            epochs=2,
            use_images=False,
            seed=42,
            eval_n_perm=20,
            eval_n_boot=20,
        )
        report_path = tmp_path / "ckpt_eval" / "evaluation_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert set(report) >= {"config", "validation", "test"}
        for split in ("validation", "test"):
            entry = report[split]
            assert entry["n_units"] >= 2
            assert "retrieval" in entry
            assert "overall" in entry["retrieval"]
            assert set(RESPONSE_OFFSETS) <= set(entry["retrieval"])
            assert "bootstrap_ci" in entry
            assert "permutation" in entry
            assert "raw_feature_baseline" in entry
            assert "probe" in entry
