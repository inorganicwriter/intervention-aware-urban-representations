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

    transformed = fusion.shape_transform(predictions, groups)

    assert transformed[0, :3].mean() == pytest.approx(0.0)
    assert transformed[0, 3:].mean() == pytest.approx(0.0)
    assert transformed[0, 0] == pytest.approx(-1.0)


def test_fuse_endpoints_match_blocks_and_middle_is_unit_norm() -> None:
    x = np.array([[1.0, 0.0]])
    h = np.array([[0.0, 1.0]])

    z0 = fusion.fuse(x, h, 0.0)
    z1 = fusion.fuse(x, h, 1.0)
    z_mid = fusion.fuse(x, h, 0.5)

    assert np.allclose(z0, np.concatenate([x, np.zeros_like(h)], axis=1))
    assert np.allclose(z1, np.concatenate([np.zeros_like(x), h], axis=1))
    assert np.allclose(np.linalg.norm(z_mid, axis=1), 1.0)


def test_unit_normalize_zero_row_handling() -> None:
    block = np.array([[0.0, 0.0], [3.0, 4.0]])

    normalized, valid = fusion.unit_normalize(block)

    assert not valid[0]
    assert valid[1]
    assert np.allclose(normalized[1], [0.6, 0.8])


def test_bootstrap_ci_brackets_the_mean() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]

    mean, low, high = fusion.bootstrap_mean_ci(values, seed=1, n_boot=500)

    assert mean == pytest.approx(0.3)
    assert low <= mean <= high
