from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "representation_controls.py"
)
_spec = importlib.util.spec_from_file_location("representation_controls", SCRIPT)
assert _spec is not None and _spec.loader is not None
controls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(controls)


def _arrays(responses: np.ndarray, cities: list[str]) -> dict[str, object]:
    n = responses.shape[0]
    return {
        "features": np.zeros((n, 2), dtype=np.float64),
        "responses": responses,
        "masks": np.ones_like(responses, dtype=bool),
        "cities": cities,
        "primary_family": "viirs",
    }


def test_refit_mean_baseline_matches_refit_model() -> None:
    """X constant: model and mean baseline both predict the refit mean."""
    responses = np.zeros((10, 1), dtype=np.float64)
    responses[0:4, 0] = 0.0  # fit labels
    responses[4:8, 0] = 10.0  # inner-validation labels
    responses[8:10, 0] = 5.0  # outer-test labels
    cities = ["a"] * 4 + ["b"] * 4 + ["c"] * 2
    arrays = _arrays(responses, cities)
    fit_idx = np.array([0, 1, 2, 3])
    val_idx = np.array([4, 5, 6, 7])
    test_idx = np.array([8, 9])
    standardized = arrays["features"]

    result = controls.fold_controls(arrays, standardized, fit_idx, val_idx, test_idx, min_obs=2)

    metric = result["ridge"][0]
    # Refit mean over fit+val labels is 5.0, exactly the test label, so the
    # model and its fair mean baseline both have zero error.
    assert metric["rmse"] == pytest.approx(0.0)
    assert metric["baseline"] == pytest.approx(0.0)
    # Without the refit baseline (fit-only mean 0) improvement would be 5.
    assert metric["baseline"] - metric["rmse"] == pytest.approx(0.0)


def test_majority_sign_uses_label_counts_not_mean() -> None:
    values = np.array([-1.0] * 9 + [100.0])
    assert controls._majority_sign(values) == -1.0
    assert np.sign(values.mean()) == 1.0
    # Tie resolves to +1.
    assert controls._majority_sign(np.array([1.0, -1.0])) == 1.0


def test_direction_baseline_uses_majority_sign() -> None:
    prediction = np.full(10, 5.0)
    target = np.array([-1.0] * 9 + [100.0])
    accuracy, baseline = controls._direction_metrics(prediction, target, -1.0)
    assert accuracy is not None and baseline is not None
    assert accuracy == pytest.approx(0.1)
    assert baseline == pytest.approx(0.9)
