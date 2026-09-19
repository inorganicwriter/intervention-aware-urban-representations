from __future__ import annotations

import numpy as np

from urban_intervention.causal.gpu.inference import (
    inference_from_standard_error,
    normal_inference_from_replicates,
)


def test_normal_inference_records_period_specific_valid_repetitions() -> None:
    draws = np.array([[0.0, 1.0], [2.0, np.nan], [4.0, 3.0]])
    result = normal_inference_from_replicates(
        np.array([2.0, 2.0]),
        draws,
        method="test_bootstrap",
    )
    np.testing.assert_array_equal(result.valid_repetitions, [3, 2])
    np.testing.assert_allclose(result.standard_error, [2.0, np.sqrt(2.0)])
    assert np.all(result.confidence_lower < result.estimate)
    assert np.all(result.confidence_upper > result.estimate)
    assert np.all((result.p_value >= 0) & (result.p_value <= 1))


def test_precomputed_standard_error_fails_closed_with_one_replicate() -> None:
    result = inference_from_standard_error(
        np.array([1.0, 0.0]),
        np.array([0.2, 0.0]),
        method="test_jackknife",
        requested_repetitions=2,
        valid_repetitions=np.array([2, 1]),
    )
    assert np.isfinite(result.confidence_lower[0])
    assert np.isnan(result.confidence_lower[1])
    assert np.isnan(result.p_value[1])


def test_analytic_inference_does_not_invent_bootstrap_repetitions() -> None:
    result = inference_from_standard_error(
        np.array([2.0]),
        np.array([0.5]),
        method="abadie_imbens_analytic",
        requested_repetitions=0,
        valid_repetitions=np.array([0]),
    )
    assert result.requested_repetitions == 0
    assert result.valid_repetitions.tolist() == [0]
    assert np.isfinite(result.confidence_lower[0])
    assert np.isfinite(result.p_value[0])
