from __future__ import annotations

import numpy as np
import pytest

from urban_intervention.causal.gpu.abadie_imbens import (
    AbadieImbensConfig,
    AbadieImbensInput,
    fit_abadie_imbens,
)
from urban_intervention.causal.gpu.matching import CommonSupportError
from urban_intervention.causal.gpu.runtime import RuntimeConfig, TorchRuntime


def _reference_input() -> AbadieImbensInput:
    covariates = np.array(
        [
            [-0.8, -0.6],
            [-0.4, 0.4],
            [0.0, 0.0],
            [0.4, -0.4],
            [0.8, 0.6],
            [-2.0, -2.0],
            [-1.0, -0.5],
            [-0.5, 0.5],
            [-0.1, 0.1],
            [0.2, -0.2],
            [0.6, 0.4],
            [1.0, 1.0],
            [2.0, -1.5],
        ],
        dtype=np.float64,
    )
    treated = np.r_[np.ones(5, dtype=bool), np.zeros(8, dtype=bool)]
    noise = np.array(
        [0.10, -0.05, 0.02, 0.08, -0.03, -0.04, 0.03, -0.02, 0.01, -0.01, 0.04, -0.05, 0.02]
    )
    outcome = 1 + 0.7 * covariates[:, 0] - 0.3 * covariates[:, 1] + 2 * treated + noise
    return AbadieImbensInput(outcome, treated, covariates)


def _runtime() -> TorchRuntime:
    return TorchRuntime(RuntimeConfig(device="cpu", seed=61))


def test_abadie_imbens_matches_matching_4_10_15_reference() -> None:
    result = fit_abadie_imbens(_reference_input(), runtime=_runtime())
    np.testing.assert_array_equal(result.treated_indices + 1, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(result.control_indices + 1, [7, 8, 9, 10, 11])
    np.testing.assert_allclose(result.pair_weights, 1.0)
    np.testing.assert_allclose(result.estimate, 2.0106058394160584, rtol=1e-13)
    np.testing.assert_allclose(
        result.analytic_standard_error, 0.025602767545806086, rtol=1e-13
    )
    np.testing.assert_allclose(
        result.pair_standard_error, 0.046955297890653384, rtol=1e-13
    )
    np.testing.assert_allclose(result.unadjusted_estimate, 2.144, rtol=1e-13)
    np.testing.assert_allclose(
        result.conditional_variances,
        [
            0.045,
            0.11045,
            0.21623333333332972,
            0.1058,
            0.05445,
            0.0512,
            0.09245,
            0.09245,
            0.0392,
            0.0392,
            0.09245,
            0.00005,
            1.4112,
        ],
        rtol=1e-12,
        atol=1e-14,
    )
    assert result.inference.method == "abadie_imbens_bias_adjusted_analytic"
    assert result.inference.requested_repetitions == 0
    assert np.isfinite(result.inference.p_value[0])


def test_abadie_imbens_enforces_explicit_donor_common_support() -> None:
    data = _reference_input()
    covariates = data.covariates.copy()
    covariates[0, 0] = 10.0
    with pytest.raises(CommonSupportError, match="outside explicit donor support"):
        fit_abadie_imbens(
            AbadieImbensInput(data.outcome, data.treated, covariates),
            runtime=_runtime(),
        )


def test_abadie_imbens_gpu_batching_preserves_formal_result() -> None:
    scalar = fit_abadie_imbens(
        _reference_input(),
        config=AbadieImbensConfig(query_batch_size=1, chunk_size=2),
        runtime=_runtime(),
    )
    combined = fit_abadie_imbens(
        _reference_input(),
        config=AbadieImbensConfig(query_batch_size=16, chunk_size=64),
        runtime=_runtime(),
    )
    np.testing.assert_array_equal(combined.treated_indices, scalar.treated_indices)
    np.testing.assert_array_equal(combined.control_indices, scalar.control_indices)
    np.testing.assert_allclose(combined.estimate, scalar.estimate, atol=1e-14)
    np.testing.assert_allclose(
        combined.analytic_standard_error,
        scalar.analytic_standard_error,
        atol=1e-14,
    )
