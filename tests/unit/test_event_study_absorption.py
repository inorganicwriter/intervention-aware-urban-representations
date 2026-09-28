from __future__ import annotations

import numpy as np
import pytest

from urban_intervention.causal.event_study import absorb_two_way_fixed_effects


def _naive_absorb(
    values: np.ndarray,
    unit: np.ndarray,
    period: np.ndarray,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 5_000,
) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float64).copy()
    for _ in range(max_iterations):
        previous = residual.copy()
        for groups in (unit, period):
            _, inverse = np.unique(groups, return_inverse=True)
            counts = np.bincount(inverse).astype(np.float64)
            for column in range(residual.shape[1]):
                sums = np.bincount(inverse, weights=residual[:, column])
                residual[:, column] -= sums[inverse] / counts[inverse]
        if np.max(np.abs(residual - previous)) <= tolerance:
            return residual
    raise AssertionError("reference absorption did not converge")


def test_absorption_matches_naive_reference() -> None:
    rng = np.random.default_rng(7)
    rows = 600
    units = rng.integers(0, 40, size=rows)
    periods = rng.integers(0, 12, size=rows)
    values = rng.normal(size=(rows, 4))

    residual, iterations = absorb_two_way_fixed_effects(values, units, periods)
    reference = _naive_absorb(values, units, periods)

    assert iterations <= 2_000
    assert np.max(np.abs(residual - reference)) < 1e-9


def test_absorption_converges_on_large_scale_columns() -> None:
    rng = np.random.default_rng(11)
    rows = 2_000
    units = rng.integers(0, 200, size=rows)
    periods = rng.integers(0, 20, size=rows)
    values = np.column_stack(
        [1_000.0 + rng.normal(scale=50.0, size=rows), np.ones(rows)]
    )

    residual, iterations = absorb_two_way_fixed_effects(values, units, periods)

    assert iterations < 500
    assert np.isfinite(residual).all()
    assert np.max(np.abs(residual[:, 1])) < 1e-6


def test_absorption_rejects_non_finite_values() -> None:
    values = np.array([[1.0], [np.nan]])
    with pytest.raises(ValueError, match="finite"):
        absorb_two_way_fixed_effects(values, np.array([0, 1]), np.array([0, 1]))


@pytest.mark.parametrize("tolerance", [0.0, 1.0, -0.5])
def test_absorption_requires_a_relative_tolerance(tolerance: float) -> None:
    values = np.ones((4, 1))
    with pytest.raises(ValueError, match="tolerance_relative"):
        absorb_two_way_fixed_effects(
            values,
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 0, 1]),
            tolerance_relative=tolerance,
        )


def test_absorption_handles_empty_array() -> None:
    values = np.zeros((0, 2))
    residual, iterations = absorb_two_way_fixed_effects(
        values, np.array([]), np.array([])
    )
    assert residual.shape == (0, 2)
    assert iterations == 0

