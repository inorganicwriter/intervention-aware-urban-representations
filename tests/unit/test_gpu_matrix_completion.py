from __future__ import annotations

import numpy as np

from urban_intervention.causal.gpu.contracts import PanelData
from urban_intervention.causal.gpu.matrix_completion import (
    MatrixCompletionConfig,
    fit_matrix_completion,
    make_rolling_cv_folds,
)
from urban_intervention.causal.gpu.runtime import RuntimeConfig, TorchRuntime


def _panel(effect: float = 5.0) -> PanelData:
    rng = np.random.default_rng(22)
    periods = 15
    units = 7
    factor = np.cos(np.linspace(-1, 2, periods))
    loading = np.linspace(-1.5, 1.5, units)
    y = 2 + np.arange(periods)[:, None] * 0.03
    y = y + np.linspace(-0.2, 0.2, units)[None, :]
    y = y + factor[:, None] * loading[None, :]
    y = y + rng.normal(scale=0.002, size=(periods, units))
    treated = np.zeros_like(y, dtype=bool)
    treated[10:, 0] = True
    y[10:, 0] += effect
    return PanelData(y=y, treated=treated)


def _runtime() -> TorchRuntime:
    return TorchRuntime(RuntimeConfig(device="cpu", seed=51))


def test_rolling_cv_masks_all_future_cells_after_score_origin() -> None:
    panel = _panel()
    folds = make_rolling_cv_folds(
        panel.untreated_observed,
        panel.treated,
        folds=3,
        nobs=1,
        buffer=0,
    )
    for fold in folds:
        assert not np.any(fold.training & fold.score)
        for unit in range(panel.y.shape[1]):
            scored = np.flatnonzero(fold.score[:, unit])
            if scored.size:
                assert not fold.training[scored[0] :, unit].any()


def test_matrix_completion_does_not_leak_post_treatment_outcome() -> None:
    config = MatrixCompletionConfig(
        lambdas=(1.0, 0.1, 0.01),
        folds=2,
        max_iter=150,
        inference="none",
    )
    baseline = fit_matrix_completion(_panel(effect=0), config=config, runtime=_runtime())
    treated = fit_matrix_completion(_panel(effect=80), config=config, runtime=_runtime())
    np.testing.assert_allclose(
        baseline.counterfactual[10:],
        treated.counterfactual[10:],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        treated.effect[10:] - baseline.effect[10:],
        80,
        atol=1e-8,
    )


def test_matrix_completion_jackknife_matches_one_target_shape() -> None:
    config = MatrixCompletionConfig(
        lambdas=(0.5, 0.05),
        folds=2,
        max_iter=100,
        inference="jackknife",
    )
    result = fit_matrix_completion(_panel(), config=config, runtime=_runtime())
    assert result.jackknife_effect_draws is not None
    assert result.jackknife_effect_draws.shape == (7, 15)
    assert np.isnan(result.jackknife_effect_draws[0]).all()
    assert result.jackknife_se is not None
    assert np.isfinite(result.jackknife_se).all()
    assert result.inference is not None
    assert result.inference.method == "mc_unit_jackknife"
    assert np.all(result.inference.valid_repetitions == 6)
    assert np.isfinite(result.inference.confidence_lower).all()
    assert np.isfinite(result.inference.confidence_upper).all()
    assert np.isfinite(result.inference.p_value).all()
    assert result.provenance.formal_eligible is False


def test_matrix_completion_jackknife_batching_preserves_draws() -> None:
    common = dict(
        fixed_lambda=0.05,
        max_iter=100,
        inference="jackknife",
        inference_batch_size=3,
    )
    batched = fit_matrix_completion(
        _panel(),
        config=MatrixCompletionConfig(**common, batch_inference=True),
        runtime=_runtime(),
    )
    scalar = fit_matrix_completion(
        _panel(),
        config=MatrixCompletionConfig(**common, batch_inference=False),
        runtime=_runtime(),
    )
    np.testing.assert_allclose(
        batched.jackknife_effect_draws,
        scalar.jackknife_effect_draws,
        atol=1e-10,
        rtol=1e-10,
        equal_nan=True,
    )
    np.testing.assert_allclose(batched.jackknife_se, scalar.jackknife_se, atol=1e-10)


def test_fixed_zero_lambda_is_a_valid_fect_grid_endpoint() -> None:
    result = fit_matrix_completion(
        _panel(),
        config=MatrixCompletionConfig(
            fixed_lambda=0.0,
            max_iter=100,
            inference="none",
        ),
        runtime=_runtime(),
    )
    assert result.selected_lambda == 0.0
    assert np.isfinite(result.counterfactual).all()
