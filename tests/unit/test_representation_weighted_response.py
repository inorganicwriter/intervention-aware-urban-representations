"""Tests for per-label design reliability weights in response losses (Rev3.4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tests.helpers.representation import build_synthetic_model_inputs
from urban_intervention.representation.dataset import (
    RESPONSE_DIM,
    RESPONSE_OFFSETS,
    RepresentationDataset,
    collate_samples,
)
from urban_intervention.representation.evaluation import (
    response_similarity_between,
    response_similarity_with_validity,
)
from urban_intervention.representation.loss import prediction_loss, response_alignment_loss
from urban_intervention.representation.queue import EmbeddingQueue

HOUSING_START, HOUSING_END, _ = RESPONSE_OFFSETS["housing"]


def _random_response_pool(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    responses = torch.randn(n, RESPONSE_DIM, generator=generator)
    masks = torch.rand(n, RESPONSE_DIM, generator=generator) > 0.25
    weights = torch.rand(n, RESPONSE_DIM, generator=generator) + 0.05
    return responses, masks, weights


def _housing_pair(
    values_a: list[float], values_b: list[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    responses = torch.zeros(2, RESPONSE_DIM)
    masks = torch.zeros(2, RESPONSE_DIM, dtype=torch.bool)
    responses[0, HOUSING_START : HOUSING_START + len(values_a)] = torch.tensor(values_a)
    responses[1, HOUSING_START : HOUSING_START + len(values_b)] = torch.tensor(values_b)
    masks[0, HOUSING_START : HOUSING_START + len(values_a)] = True
    masks[1, HOUSING_START : HOUSING_START + len(values_b)] = True
    return responses, masks


def test_unit_weights_reproduce_unweighted_similarity() -> None:
    responses, masks, _ = _random_response_pool(6, seed=1)
    sim_plain, valid_plain = response_similarity_with_validity(responses, masks)
    sim_weighted, valid_weighted = response_similarity_with_validity(
        responses, masks, weights=torch.ones_like(responses)
    )
    assert torch.allclose(sim_plain, sim_weighted, atol=1e-6)
    assert torch.equal(valid_plain, valid_weighted)


def test_zero_weight_cell_leaves_the_pair_statistic() -> None:
    responses, masks = _housing_pair([1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, 5.0])
    weights = torch.ones_like(responses)
    weights[:, HOUSING_START + 2] = 0.0
    sim_weighted, valid_weighted = response_similarity_with_validity(
        responses, masks, weights=weights
    )

    dropped_responses = responses.clone()
    dropped_masks = masks.clone()
    dropped_responses[:, HOUSING_START + 2] = 0.0
    dropped_masks[:, HOUSING_START + 2] = False
    sim_dropped, valid_dropped = response_similarity_with_validity(
        dropped_responses, dropped_masks
    )

    assert torch.allclose(sim_weighted, sim_dropped, atol=1e-6)
    assert torch.equal(valid_weighted, valid_dropped)


def test_asymmetric_masks_ignore_one_sided_cells() -> None:
    responses = torch.zeros(2, RESPONSE_DIM)
    masks = torch.zeros(2, RESPONSE_DIM, dtype=torch.bool)
    responses[0, HOUSING_START : HOUSING_START + 3] = torch.tensor([1.0, 2.0, 3.0])
    responses[1, HOUSING_START + 1 : HOUSING_START + 4] = torch.tensor([4.0, 5.0, 6.0])
    masks[0, HOUSING_START : HOUSING_START + 3] = True
    masks[1, HOUSING_START + 1 : HOUSING_START + 4] = True

    sim_plain, valid_plain = response_similarity_between(responses, masks, responses, masks)
    sim_weighted, valid_weighted = response_similarity_between(
        responses,
        masks,
        responses,
        masks,
        weights_a=torch.ones_like(responses),
        weights_b=torch.ones_like(responses),
    )
    assert torch.allclose(sim_plain, sim_weighted, atol=1e-6)
    assert torch.equal(valid_plain, valid_weighted)


def test_no_common_reliable_support_is_invalid_not_zero() -> None:
    responses = torch.zeros(2, RESPONSE_DIM)
    masks = torch.zeros(2, RESPONSE_DIM, dtype=torch.bool)
    responses[0, HOUSING_START] = 1.0
    responses[1, HOUSING_START + 1] = 2.0
    masks[0, HOUSING_START] = True
    masks[1, HOUSING_START + 1] = True

    _, valid = response_similarity_with_validity(responses, masks)
    assert not bool(valid.any())

    weights = masks.to(responses.dtype)
    _, valid_weighted = response_similarity_with_validity(
        responses, masks, weights=weights
    )
    assert not bool(valid_weighted.any())

    embeddings = torch.randn(2, 4)
    loss = response_alignment_loss(
        embeddings, responses, masks, response_weight=weights
    )
    assert float(loss) == 0.0


def test_invalid_zero_weight_pair_is_masked_from_the_softmax() -> None:
    n_alive = 3
    responses = torch.zeros(n_alive + 1, RESPONSE_DIM)
    masks = torch.zeros(n_alive + 1, RESPONSE_DIM, dtype=torch.bool)
    for i in range(n_alive):
        responses[i, HOUSING_START : HOUSING_START + 3] = torch.tensor([1.0, 2.0, 3.0]) * (
            i + 1
        )
        masks[i, HOUSING_START : HOUSING_START + 3] = True
    masks[n_alive] = masks[0]
    weights = torch.ones(n_alive + 1, RESPONSE_DIM)
    weights[n_alive, :] = 0.0

    generator = torch.Generator().manual_seed(7)
    embeddings = torch.randn(n_alive + 1, 5, generator=generator)
    reference = response_alignment_loss(
        embeddings[:n_alive],
        responses[:n_alive],
        masks[:n_alive],
        response_weight=weights[:n_alive],
    )
    with_dead_pair = response_alignment_loss(
        embeddings, responses, masks, response_weight=weights
    )
    assert float(reference) > 0.0
    assert torch.allclose(reference, with_dead_pair, atol=1e-6)


def test_weight_scale_does_not_change_similarity() -> None:
    responses, masks, weights = _random_response_pool(5, seed=3)
    sim_base, valid_base = response_similarity_with_validity(
        responses, masks, weights=weights
    )
    sim_scaled, valid_scaled = response_similarity_with_validity(
        responses, masks, weights=weights * 7.5
    )
    assert torch.allclose(sim_base, sim_scaled, atol=1e-6)
    assert torch.equal(valid_base, valid_scaled)


def test_weighted_similarity_is_symmetric() -> None:
    responses_a, masks_a, weights_a = _random_response_pool(3, seed=11)
    responses_b, masks_b, weights_b = _random_response_pool(4, seed=12)
    sim_ab, valid_ab = response_similarity_between(
        responses_a,
        masks_a,
        responses_b,
        masks_b,
        weights_a=weights_a,
        weights_b=weights_b,
    )
    sim_ba, valid_ba = response_similarity_between(
        responses_b,
        masks_b,
        responses_a,
        masks_a,
        weights_a=weights_b,
        weights_b=weights_a,
    )
    assert torch.allclose(sim_ab, sim_ba.T, atol=1e-6)
    assert torch.equal(valid_ab, valid_ba.T)


def test_square_and_rectangular_paths_agree() -> None:
    responses, masks, weights = _random_response_pool(5, seed=13)
    square_sim, square_valid = response_similarity_with_validity(
        responses, masks, weights=weights
    )
    rect_sim, rect_valid = response_similarity_between(
        responses, masks, responses, masks, weights_a=weights, weights_b=weights
    )
    assert torch.allclose(square_sim, rect_sim, atol=1e-6)
    assert torch.equal(square_valid, rect_valid)


def test_labeled_queue_round_trips_response_weights() -> None:
    responses, masks, weights = _random_response_pool(2, seed=17)
    response_se = torch.full_like(responses, 0.1)
    ids = torch.tensor([1, 2])
    embeddings = torch.randn(2, 6)

    queue = EmbeddingQueue(dim=6, capacity=8)
    queue.enqueue(embeddings, responses, masks, response_se, ids, response_weight=weights)
    state = queue.labeled_state()
    assert state is not None
    assert torch.allclose(state["response_weight"], weights)

    legacy = EmbeddingQueue(dim=6, capacity=8)
    legacy.enqueue(embeddings, responses, masks, response_se, ids)
    legacy_state = legacy.labeled_state()
    assert legacy_state is not None
    assert torch.allclose(
        legacy_state["response_weight"], torch.ones_like(responses)
    )


def _housing_prediction_case(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    responses = torch.zeros(1, RESPONSE_DIM)
    masks = torch.zeros(1, RESPONSE_DIM, dtype=torch.bool)
    responses[0, HOUSING_START : HOUSING_START + 2] = targets
    masks[0, HOUSING_START : HOUSING_START + 2] = True
    return {"housing": predictions}, responses, masks


def test_prediction_loss_uses_design_weights_not_standard_errors() -> None:
    predictions, responses, masks = _housing_prediction_case(
        torch.zeros(1, HOUSING_END - HOUSING_START), torch.tensor([1.0, 0.0])
    )
    standard_errors = torch.full((1, RESPONSE_DIM), 100.0)
    standard_errors[0, HOUSING_START + 1] = 0.1
    weights = torch.zeros(1, RESPONSE_DIM)
    weights[0, HOUSING_START] = 1.0
    weights[0, HOUSING_START + 1] = 1.0

    weighted = prediction_loss(
        predictions, responses, masks, standard_errors, response_weight=weights
    )
    assert float(weighted) == pytest.approx(0.5, abs=1e-6)

    precision = prediction_loss(predictions, responses, masks, standard_errors)
    assert float(precision) != pytest.approx(0.5, abs=1e-3)


def test_prediction_loss_weights_each_cell_independently() -> None:
    predictions, responses, masks = _housing_prediction_case(
        torch.zeros(1, HOUSING_END - HOUSING_START), torch.tensor([1.0, 2.0])
    )
    weights = torch.zeros(1, RESPONSE_DIM)
    weights[0, HOUSING_START] = 1.0
    weights[0, HOUSING_START + 1] = 3.0

    loss = prediction_loss(predictions, responses, masks, None, weights)
    assert float(loss) == pytest.approx((1.0 * 1.0 + 3.0 * 4.0) / 4.0, abs=1e-6)


def test_prediction_loss_skips_families_with_no_weighted_cells() -> None:
    predictions, responses, masks = _housing_prediction_case(
        torch.zeros(1, HOUSING_END - HOUSING_START), torch.tensor([1.0, 2.0])
    )
    weights = torch.zeros(1, RESPONSE_DIM)
    loss = prediction_loss(predictions, responses, masks, None, weights)
    assert float(loss.detach()) == 0.0


def test_dataset_loader_reads_response_weights(tmp_path: Path) -> None:
    output, _ = build_synthetic_model_inputs(tmp_path, n_grids=6, n_cities=2)

    targets = pd.read_parquet(output / "response_targets.parquet")
    targets["response_weight"] = np.where(targets["final_training_mask"], 0.25, 0.0)
    targets.to_parquet(output / "response_targets.parquet", index=False)

    dataset = RepresentationDataset(output)
    sample = dataset[0]
    assert float(sample.response_weight.sum()) == pytest.approx(0.25 * RESPONSE_DIM)
    assert float(sample.response_weight.min()) >= 0.0

    batch = collate_samples([dataset[0], dataset[1]])
    weights = batch["response_weight"]
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (2, RESPONSE_DIM)

    legacy_output, _ = build_synthetic_model_inputs(tmp_path / "legacy", n_grids=6, n_cities=2)
    legacy = RepresentationDataset(legacy_output)
    legacy_sample = legacy[0]
    assert float(legacy_sample.response_weight.sum()) == pytest.approx(RESPONSE_DIM)
