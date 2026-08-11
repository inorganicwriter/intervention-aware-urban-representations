from __future__ import annotations

import numpy as np
import pytest
import torch

from urban_intervention.representation.baselines import (
    baseline_metric_entry,
    dinov2_image_baseline,
    headline_metric,
    pca_projection_baseline,
    random_projection_baseline,
    run_baselines,
)
from urban_intervention.representation.dataset import RESPONSE_DIM


def _pool(n: int = 16, dim: int = 12, seed: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    features = torch.from_numpy(rng.randn(n, dim).astype(np.float32))
    responses = torch.from_numpy(rng.randn(n, RESPONSE_DIM).astype(np.float32))
    masks = torch.from_numpy(rng.rand(n, RESPONSE_DIM) > 0.4)
    return features, responses, masks


class TestBaselineMetricEntry:
    def test_structure_matches_model_entries(self) -> None:
        features, responses, masks = _pool()
        entry = baseline_metric_entry(
            torch.nn.functional.normalize(features, p=2, dim=1), responses, masks, k=4, n_perm=20
        )
        assert set(entry) == {"n_units", "retrieval", "bootstrap_ci", "permutation"}
        assert entry["n_units"] == 16
        retrieval = entry["retrieval"]
        bootstrap = entry["bootstrap_ci"]
        permutation = entry["permutation"]
        assert isinstance(retrieval, dict)
        assert isinstance(bootstrap, dict)
        assert isinstance(permutation, dict)
        assert "overall" in retrieval
        assert "lower" in bootstrap
        assert "p_value" in permutation

    def test_small_pool_returns_note(self) -> None:
        features, responses, masks = _pool(n=1)
        entry = baseline_metric_entry(features, responses, masks)
        assert "note" in entry


class TestRandomProjection:
    def test_permutation_not_significant_at_chance(self) -> None:
        features, responses, masks = _pool(n=32, seed=11)
        entry = random_projection_baseline(features, responses, masks, k=5, n_perm=49, seed=1)
        permutation = entry["permutation"]
        assert isinstance(permutation, dict)
        p_value = permutation["p_value"]
        assert isinstance(p_value, (int, float))
        assert p_value > 0.05

    def test_deterministic_given_seed(self) -> None:
        features, responses, masks = _pool()
        first = random_projection_baseline(features, responses, masks, k=4, n_perm=20)
        second = random_projection_baseline(features, responses, masks, k=4, n_perm=20)
        assert first["retrieval"] == second["retrieval"]


class TestPcaProjection:
    def test_reports_components_and_finite_metrics(self) -> None:
        features, responses, masks = _pool(n=32, seed=7)
        entry = pca_projection_baseline(features, responses, masks, k=5, n_perm=30)
        retrieval = entry["retrieval"]
        assert isinstance(retrieval, dict)
        overall = retrieval["overall"]
        assert isinstance(overall, dict)
        nn_corr = overall["nn_corr@k"]
        assert isinstance(nn_corr, (int, float))
        assert -1.0 <= nn_corr <= 1.0
        components = entry["components"]
        assert isinstance(components, int)
        assert components > 0

    def test_cluster_structure_survives_pca(self) -> None:
        rng = np.random.RandomState(5)
        n = 40
        features = torch.zeros(n, 20)
        features[:20, :10] = 1.0
        features[20:, 10:] = 1.0
        features += torch.from_numpy(rng.randn(n, 20).astype(np.float32)) * 0.2
        t = np.linspace(0, 2 * np.pi, RESPONSE_DIM)
        p1 = np.sin(t).astype(np.float32)
        p2 = np.cos(t).astype(np.float32)
        responses = torch.from_numpy(
            np.stack([p1 + rng.normal(0, 0.05, RESPONSE_DIM) for _ in range(20)]
                     + [p2 + rng.normal(0, 0.05, RESPONSE_DIM) for _ in range(20)])
        ).float()
        masks = torch.ones(n, RESPONSE_DIM, dtype=torch.bool)
        entry = pca_projection_baseline(features, responses, masks, k=5, n_perm=20)
        retrieval = entry["retrieval"]
        assert isinstance(retrieval, dict)
        overall = retrieval["overall"]
        assert isinstance(overall, dict)
        nn_corr = overall["nn_corr@k"]
        assert isinstance(nn_corr, (int, float))
        assert nn_corr > 0.5


class TestDinov2ImageBaseline:
    def test_no_image_batches_returns_note(self) -> None:
        entry = dinov2_image_baseline([], torch.device("cpu"))
        assert entry.get("note") == "no_image_batches_available"

    def test_batches_without_images_returns_note(self) -> None:
        batch: dict[str, object] = {
            "responses": torch.zeros(2, RESPONSE_DIM),
            "response_mask": torch.ones(2, RESPONSE_DIM, dtype=torch.bool),
        }
        entry = dinov2_image_baseline([batch], torch.device("cpu"))
        assert entry.get("note") == "no_image_batches_available"

    def test_all_masked_grid_rows_never_nan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class DummyBackbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.randn(x.shape[0], 384)

        monkeypatch.setattr(
            "urban_intervention.representation.encoder._ensure_dinov2",
            lambda *args, **kwargs: DummyBackbone(),
        )
        batch: dict[str, object] = {
            "images": torch.zeros(2, 2, 3, 224, 224),
            "image_mask": torch.tensor([[True, True], [False, False]]),
            "responses": torch.randn(2, RESPONSE_DIM),
            "response_mask": torch.ones(2, RESPONSE_DIM, dtype=torch.bool),
        }
        entry = dinov2_image_baseline([batch], torch.device("cpu"), k=1, n_perm=5, n_boot=5)
        retrieval = entry["retrieval"]
        assert isinstance(retrieval, dict)
        overall = retrieval["overall"]
        assert isinstance(overall, dict)
        nn_corr = overall["nn_corr@k"]
        assert isinstance(nn_corr, (int, float))
        assert np.isfinite(nn_corr)


class TestRunBaselines:
    def test_includes_random_and_pca(self) -> None:
        features, responses, masks = _pool(n=24)
        report = run_baselines(features, responses, masks, k=4, n_perm=20, n_boot=20)
        assert set(report) == {"random_projection", "pca_features"}
        random_entry = report["random_projection"]
        pca_entry = report["pca_features"]
        assert isinstance(random_entry, dict)
        assert isinstance(pca_entry, dict)
        assert "retrieval" in random_entry
        assert "retrieval" in pca_entry

    def test_skips_dinov2_without_batches(self) -> None:
        features, responses, masks = _pool(n=24)
        report = run_baselines(features, responses, masks, image_batches=None)
        assert "dinov2_images" not in report


class TestHeadlineMetric:
    def test_extracts_nn_corr(self) -> None:
        features, responses, masks = _pool(n=12)
        entry = random_projection_baseline(features, responses, masks, k=4)
        value = headline_metric(entry)
        assert isinstance(value, float)

    def test_none_for_unavailable(self) -> None:
        assert headline_metric({"note": "pool smaller than 2 units"}) is None
