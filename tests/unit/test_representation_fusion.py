from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "representation_fusion_evaluation.py"
)
_spec = importlib.util.spec_from_file_location("representation_fusion_evaluation", SCRIPT)
assert _spec is not None and _spec.loader is not None
fusion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fusion)


def test_shape_transform_removes_per_outcome_temporal_means() -> None:
    predictions = np.array([[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]])
    groups = {"a": [0, 1, 2], "b": [3, 4, 5]}
    fitted = np.ones(6, dtype=bool)

    transformed = fusion.shape_transform(predictions, groups, fitted)

    assert transformed[0, :3].mean() == pytest.approx(0.0)
    assert transformed[0, 3:].mean() == pytest.approx(0.0)
    assert transformed[0, 0] == pytest.approx(-1.0)


def test_shape_transform_skips_unfitted_cells_without_nan_propagation() -> None:
    predictions = np.array([[1.0, 2.0, 3.0, 10.0, np.nan, 30.0]])
    groups = {"a": [0, 1, 2], "b": [3, 4, 5]}
    fitted = np.array([True, True, True, True, False, True])

    transformed = fusion.shape_transform(predictions, groups, fitted)

    assert np.isfinite(transformed[0, :3]).all()
    assert np.isnan(transformed[0, 4])
    assert transformed[0, 3] == pytest.approx(-10.0)
    assert transformed[0, 5] == pytest.approx(10.0)


def test_build_block_zeroes_unfitted_cells() -> None:
    references = np.array(
        [
            [1.0, 2.0, 3.0, 10.0, np.nan, 30.0],
            [2.0, 4.0, 6.0, 20.0, np.nan, 60.0],
        ]
    )
    targets = np.array([[0.5, 3.5, 5.0, 5.0, np.nan, 55.0]])
    groups = {"a": [0, 1, 2], "b": [3, 4, 5]}
    fitted = np.array([True, True, True, True, False, True])

    block, valid, fitted_cells = fusion.build_block(
        targets, references, "shape", groups, fitted
    )

    assert fitted_cells == 5
    assert valid[0]
    assert np.isfinite(block[0]).all()
    assert block[0, 4] == pytest.approx(0.0)


def test_fuse_endpoints_and_zero_response_block() -> None:
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    h = np.array([[0.0, 0.0], [0.0, 0.0]])

    z0 = fusion.fuse(x, h, 0.0)
    z1 = fusion.fuse(x, h, 1.0)
    z_mid = fusion.fuse(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), 0.5)

    assert np.allclose(z0, np.concatenate([x, np.zeros_like(h)], axis=1))
    assert np.allclose(z1, np.concatenate([np.zeros_like(x), h], axis=1))
    assert np.allclose(np.linalg.norm(z_mid, axis=1), 1.0)


def test_unit_normalize_zero_row_handling() -> None:
    block = np.array([[0.0, 0.0], [3.0, 4.0]])

    normalized, valid = fusion.unit_normalize(block)

    assert not valid[0]
    assert valid[1]
    assert np.allclose(normalized[1], [0.6, 0.8])


def test_amplitude_uses_refit_baseline_not_test_labels() -> None:
    responses = np.full((4, 1), 10.0)
    masks = np.ones((4, 1), dtype=bool)
    predictions = np.zeros((4, 1))

    record = fusion.amplitude_direction(
        predictions, responses, masks, [0], {0: 0.0}, {0: -1.0}
    )

    assert record["rmse_pooled"] == pytest.approx(10.0)
    assert record["rmse_pooled_baseline"] == pytest.approx(10.0)
    assert record["direction_accuracy_pooled"] == pytest.approx(0.0)


def test_rmse_denominator_uses_observations_not_cells() -> None:
    responses = np.ones((4, 1))
    masks = np.ones((4, 1), dtype=bool)
    predictions = np.zeros((4, 1))
    baselines = {0: 0.0}

    record = fusion.amplitude_direction(
        predictions, responses, masks, [0], baselines, {0: 1.0}
    )
    repeated = fusion.amplitude_direction(
        np.zeros((16, 1)),
        np.ones((16, 1)),
        np.ones((16, 1), dtype=bool),
        [0],
        baselines,
        {0: 1.0},
    )

    assert record["rmse_pooled"] == pytest.approx(1.0)
    assert record["rmse_cell_macro"] == pytest.approx(1.0)
    assert repeated["rmse_pooled"] == pytest.approx(1.0)


def test_city_deltas_use_full_pool_for_every_anchor_city() -> None:
    responses = np.array(
        [
            [1.0, 0.0, 0.0],  # a1
            [1.0, 0.0, 0.0],  # b1 (identical to a1)
            [2.0, 1.0, 3.0],  # b2 (zero correlation with a1)
        ]
    )
    masks = np.ones_like(responses, dtype=bool)
    cities = np.asarray(["A", "B", "B"], dtype=object)
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    raw = np.stack([e1, e1, e2])
    scheme = np.stack([e2, e1, e2])

    per_city, macro = fusion.per_anchor_city_deltas(
        raw, scheme, responses, masks, cities, [0, 1, 2], k=1
    )

    # a1's raw nearest is b1 (similarity 1); its scheme nearest is b2
    # (similarity 0), so city A must appear with delta -1 even though it has a
    # single anchor and no same-city candidate.
    assert "A" in per_city
    assert per_city["A"]["n_anchors"] == 1
    assert per_city["A"]["delta"] == pytest.approx(-1.0)
    assert macro == pytest.approx([-1.0, 0.0])


def test_bootstrap_ci_brackets_the_mean() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]

    mean, low, high = fusion.bootstrap_mean_ci(values, seed=1, n_boot=500)

    assert mean == pytest.approx(0.3)
    assert low <= mean <= high
