from __future__ import annotations

import numpy as np
import torch

from urban_intervention.representation.dataset import RESPONSE_DIM
from urban_intervention.representation.transfer import (
    few_shot_probe,
    per_city_retrieval,
    transfer_report,
)


def _pool_with_cities(
    n: int = 12, seed: int = 4, cities: tuple[str, str] = ("test_city_a", "test_city_b")
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    rng = np.random.RandomState(seed)
    embeddings = torch.from_numpy(rng.randn(n, 8).astype(np.float32))
    embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)
    responses = torch.from_numpy(rng.randn(n, RESPONSE_DIM).astype(np.float32))
    masks = torch.ones(n, RESPONSE_DIM, dtype=torch.bool)
    city_keys = [cities[0]] * (n // 2) + [cities[1]] * (n - n // 2)
    return embeddings, responses, masks, city_keys


class TestPerCityRetrieval:
    def test_splits_metrics_by_city(self) -> None:
        embeddings, responses, masks, city_keys = _pool_with_cities(n=12)
        result = per_city_retrieval(embeddings, responses, masks, city_keys, k=3)
        assert set(result) == {"test_city_a", "test_city_b"}
        for city_entry in result.values():
            overall = city_entry["overall"]
            assert overall["n_units"] == 6

    def test_skips_single_unit_cities(self) -> None:
        embeddings, responses, masks, city_keys = _pool_with_cities(n=12)
        city_keys[0] = "lonely"
        result = per_city_retrieval(embeddings, responses, masks, city_keys, k=3)
        assert "lonely" not in result
        assert len(result) == 2

    def test_misaligned_keys_rejected(self) -> None:
        embeddings, responses, masks, _ = _pool_with_cities(n=12)
        try:
            per_city_retrieval(embeddings, responses, masks, ["a"], k=3)
        except ValueError:
            return
        raise AssertionError("expected ValueError for misaligned city_keys")


class TestFewShotProbe:
    def test_returns_curve_with_structure(self) -> None:
        rng = np.random.RandomState(9)
        n = 40
        signal = torch.from_numpy(rng.randn(n, 4).astype(np.float32))
        embeddings = torch.cat([signal, torch.from_numpy(rng.randn(n, 4).astype(np.float32) * 0.05)], dim=1)
        embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)
        responses = torch.cat([signal, signal[:, :1]], dim=1)
        responses = torch.cat([responses, torch.from_numpy(rng.randn(n, RESPONSE_DIM - 5).astype(np.float32) * 0.05)], dim=1)
        masks = torch.ones(n, RESPONSE_DIM, dtype=torch.bool)
        curve = few_shot_probe(embeddings, responses, masks, shot_sizes=(4, 8, 16), ridge=1e-3, min_obs=2)
        shots = [step["shot"] for step in curve]
        assert shots == [4, 8, 16]
        for step in curve:
            rmse = step["rmse_overall"]
            eval_units = step["eval_units"]
            assert isinstance(rmse, (int, float))
            assert isinstance(eval_units, int)
            assert eval_units > 0

    def test_shot_at_least_pool_size_skipped(self) -> None:
        rng = np.random.RandomState(1)
        embeddings = torch.from_numpy(rng.randn(6, 8).astype(np.float32))
        responses = torch.from_numpy(rng.randn(6, RESPONSE_DIM).astype(np.float32))
        masks = torch.ones(6, RESPONSE_DIM, dtype=torch.bool)
        curve = few_shot_probe(embeddings, responses, masks, shot_sizes=(4, 8), ridge=1.0)
        assert [step["shot"] for step in curve] == [4]

    def test_tiny_pool_returns_empty(self) -> None:
        rng = np.random.RandomState(2)
        embeddings = torch.from_numpy(rng.randn(1, 8).astype(np.float32))
        responses = torch.from_numpy(rng.randn(1, RESPONSE_DIM).astype(np.float32))
        masks = torch.ones(1, RESPONSE_DIM, dtype=torch.bool)
        assert few_shot_probe(embeddings, responses, masks) == []


class TestTransferReport:
    def test_full_structure(self) -> None:
        embeddings, responses, masks, city_keys = _pool_with_cities(n=16)
        features = embeddings.clone()
        report = transfer_report(
            embeddings, features, responses, masks, city_keys, k=3, shot_sizes=(4,), ridge=1.0
        )
        per_city = report["per_city"]
        few_shot = report["few_shot_probe"]
        assert isinstance(per_city, dict)
        assert isinstance(few_shot, dict)
        assert set(per_city) == {"test_city_a", "test_city_b"}
        assert few_shot["protocol"] == "within_city"
        cities = few_shot["cities"]
        assert isinstance(cities, dict)
        assert set(cities) == {"test_city_a", "test_city_b"}
        first_city = cities["test_city_a"]
        assert isinstance(first_city, dict)
        embedding_curve = first_city["embeddings"]
        assert isinstance(embedding_curve, list)
        assert embedding_curve[0]["shot"] == 4
        assert "full_supervision_probe" not in report
        assert report["cross_validated_probe"]["protocol"] == "target_pool_disjoint_folds"

    def test_small_pool_skips_probes(self) -> None:
        embeddings, responses, masks, city_keys = _pool_with_cities(n=16)
        report = transfer_report(
            embeddings[:1], embeddings[:1], responses[:1], masks[:1], city_keys[:1], k=3
        )
        assert "few_shot_probe" not in report
        assert report["per_city"] == {}
