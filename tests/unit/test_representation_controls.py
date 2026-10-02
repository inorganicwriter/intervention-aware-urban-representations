from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers.representation import build_synthetic_model_inputs

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

    result = controls.fold_controls(
        arrays, standardized, standardized, fit_idx, val_idx, test_idx, min_obs=2
    )

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


def test_city_summary_keeps_grid_and_cell_counts() -> None:
    responses = np.zeros((12, 1), dtype=np.float64)
    cities = ["a"] * 4 + ["b"] * 4 + ["c"] * 2 + ["d"] * 2
    arrays = _arrays(responses, cities)
    fit_idx = np.arange(0, 4)
    val_idx = np.arange(4, 8)
    test_idx = np.arange(8, 12)
    standardized = arrays["features"]

    result = controls.fold_controls(
        arrays, standardized, standardized, fit_idx, val_idx, test_idx, min_obs=2
    )

    per_city = result["per_city"]
    assert per_city["c"]["n_units"] == 2
    assert per_city["c"]["ridge"]["n_cells"] == 1
    assert per_city["d"]["knn"]["n_cells"] == 1


def test_preprocessor_depends_on_reference_rows_only() -> None:
    features = np.array([[1.0], [2.0], [3.0], [100.0]])
    reference = np.array([0, 1])
    first = controls._Preprocessor(features, reference)
    modified = features.copy()
    modified[2, 0] = -999.0  # inner-validation row must not influence the scaler
    second = controls._Preprocessor(modified, reference)
    assert first.mean[0] == pytest.approx(second.mean[0])
    assert first.scale[0] == pytest.approx(second.scale[0])


def test_load_arrays_preserves_missing_after_inverting_z_scores(tmp_path: Path) -> None:
    ds_dir, _ = build_synthetic_model_inputs(tmp_path)
    unit_path = ds_dir / "unit_features.parquet"
    unit = pd.read_parquet(unit_path)
    z_column = next(column for column in unit.columns if column.startswith("z__"))
    unit.loc[unit.index[0], z_column] = np.nan
    unit.to_parquet(unit_path, index=False)

    from urban_intervention.representation.dataset import RepresentationDataset

    ds = RepresentationDataset(ds_dir, load_images=False)
    column_index = ds.feature_columns.index(z_column)
    arrays = controls.load_arrays(ds_dir)

    features = arrays["features"]
    assert np.isnan(features[:, column_index]).sum() == 1
