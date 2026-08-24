from __future__ import annotations

import numpy as np
import pytest

from urban_intervention.causal.gpu.contracts import MatchingInput
from urban_intervention.causal.gpu.matching import (
    CommonSupportError,
    MatchingConfig,
    _topk_mahalanobis,
    fit_matching,
    stable_covariance_inverse,
)
from urban_intervention.causal.gpu.runtime import RuntimeConfig, TorchRuntime


def _runtime() -> TorchRuntime:
    return TorchRuntime(RuntimeConfig(device="cpu", chunk_size=7, seed=13))


def test_stable_covariance_inverse_drops_collinear_direction() -> None:
    values = np.column_stack((np.arange(8, dtype=float), np.arange(8, dtype=float) * 2))
    inverse, rank = stable_covariance_inverse(values)
    assert rank == 1
    np.testing.assert_allclose(inverse, inverse.T)


def test_matching_uses_static_refinement_only_inside_top_m() -> None:
    donors = np.column_stack((np.arange(12, dtype=float), np.arange(12, dtype=float) ** 2))
    target = np.array([5.1, 25.5])
    static = np.arange(12, dtype=float)[:, None]
    static[4] = 100
    static[5] = 50
    static[6] = 0
    result = fit_matching(
        MatchingInput(
            target=target,
            donors=donors,
            target_static=np.array([0.1]),
            donor_static=static,
        ),
        config=MatchingConfig(candidates=3),
        runtime=_runtime(),
    )
    assert result.selected_index in result.donor_indices
    assert result.selected_index == 6


def test_matching_rejects_target_outside_donor_support() -> None:
    donors = np.column_stack((np.arange(6, dtype=float), np.arange(6, dtype=float) ** 2))
    with pytest.raises(CommonSupportError):
        fit_matching(
            MatchingInput(
                target=np.array([20.0, 2.0]),
                donors=donors,
                support_feature_indices=(0,),
            ),
            runtime=_runtime(),
        )


def test_matching_placebo_gate_is_deterministic() -> None:
    x = np.linspace(-2, 2, 30)
    donors = np.column_stack((x, x**2, np.sin(x)))
    holdout = np.column_stack((np.cos(x), x**3))
    data = MatchingInput(
        target=donors[12] + 0.001,
        donors=donors,
        target_holdout=holdout[12] + 0.001,
        donor_holdout=holdout,
    )
    config = MatchingConfig(candidates=5, placebo_sample=20)
    first = fit_matching(data, config=config, runtime=_runtime())
    second = fit_matching(data, config=config, runtime=_runtime())
    assert first.selected_index == second.selected_index
    assert first.placebo_thresholds == second.placebo_thresholds
    assert first.quality_passed is True


def test_exact_topk_boundary_ties_use_original_donor_order() -> None:
    runtime = TorchRuntime(RuntimeConfig(device="cpu", chunk_size=2))
    indices, distances = _topk_mahalanobis(
        np.zeros((1, 1)),
        np.ones((9, 1)),
        np.ones((1, 1)),
        k=4,
        runtime=runtime,
        chunk_size=2,
    )
    assert indices[0].tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(distances, 1.0)
