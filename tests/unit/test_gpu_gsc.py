from __future__ import annotations

import numpy as np

from urban_intervention.causal.gpu.contracts import PanelData
from urban_intervention.causal.gpu.gsc import (
    GSCConfig,
    estimate_gsc_inference_batch_bytes,
    fit_gsc,
    make_all_unit_cv_masks,
)
from urban_intervention.causal.gpu.runtime import RuntimeConfig, TorchRuntime


def test_large_gsc_inference_memory_estimate_exposes_unsafe_default_batch() -> None:
    one = estimate_gsc_inference_batch_bytes(66, 3_771_800, 1)
    sixteen = estimate_gsc_inference_batch_bytes(66, 3_771_800, 16)
    assert 10 * 1024**3 < one < 24 * 1024**3
    assert sixteen == one * 16


def _synthetic_panel(post_effect: float = 4.0) -> PanelData:
    rng = np.random.default_rng(7)
    periods = 18
    units = 9
    factor = np.sin(np.linspace(-1.2, 1.8, periods))
    loadings = np.linspace(-2.0, 2.0, units)
    y = 3.0 + np.linspace(-0.5, 0.7, periods)[:, None]
    y = y + np.linspace(-0.3, 0.3, units)[None, :]
    y = y + factor[:, None] * loadings[None, :]
    y = y + rng.normal(scale=0.005, size=(periods, units))
    treated = np.zeros_like(y, dtype=bool)
    treated[12:, 0] = True
    y[12:, 0] += post_effect
    return PanelData(y=y, treated=treated)


def test_cv_masks_preserve_training_support_and_are_reproducible() -> None:
    observed = np.ones((12, 7), dtype=bool)
    first = make_all_unit_cv_masks(
        observed,
        folds=5,
        proportion=0.1,
        min_observed_per_unit=5,
        seed=19,
    )
    second = make_all_unit_cv_masks(
        observed,
        folds=5,
        proportion=0.1,
        min_observed_per_unit=5,
        seed=19,
    )
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert all(((observed & ~held).sum(axis=0) >= 5).all() for held in first)
    assert all(((observed & ~held).sum(axis=1) >= 1).all() for held in first)


def test_gsc_recovers_counterfactual_without_post_treatment_leakage() -> None:
    runtime = TorchRuntime(RuntimeConfig(device="cpu", seed=31))
    config = GSCConfig(ranks=(0, 1, 2), folds=3, seed=31, max_iter=200)
    baseline = fit_gsc(_synthetic_panel(post_effect=0), config=config, runtime=runtime)
    treated = fit_gsc(_synthetic_panel(post_effect=100), config=config, runtime=runtime)
    np.testing.assert_allclose(
        treated.counterfactual[12:],
        baseline.counterfactual[12:],
        atol=1e-8,
    )
    assert abs(np.mean(treated.effect[12:]) - 100) < 0.1
    assert treated.selected_rank in config.ranks
    assert treated.provenance.formal_eligible is False


def test_gsc_empirical_bootstrap_has_expected_shape() -> None:
    runtime = TorchRuntime(RuntimeConfig(device="cpu", seed=41))
    config = GSCConfig(
        ranks=(1,),
        folds=2,
        seed=41,
        max_iter=100,
        bootstrap_mode="auto",
        n_bootstrap=3,
    )
    result = fit_gsc(_synthetic_panel(), config=config, runtime=runtime)
    assert result.bootstrap_draws is not None
    assert result.bootstrap_draws.shape == (3, 18)
    assert result.bootstrap_se is not None
    assert np.isfinite(result.bootstrap_se).all()
    assert result.inference is not None
    assert result.inference.method == "gsc_parametric_reference_empirical"
    assert np.isfinite(result.inference.confidence_lower).all()
    assert np.isfinite(result.inference.p_value).all()


def test_gsc_auto_bootstrap_uses_ar_when_panel_has_missing_cells() -> None:
    panel = _synthetic_panel()
    y = panel.y.copy()
    y[3, 2] = np.nan
    incomplete = PanelData(y=y, treated=panel.treated)
    result = fit_gsc(
        incomplete,
        config=GSCConfig(
            fixed_rank=1,
            ranks=(1,),
            folds=2,
            seed=42,
            max_iter=100,
            bootstrap_mode="auto",
            n_bootstrap=3,
        ),
        runtime=TorchRuntime(RuntimeConfig(device="cpu", seed=42)),
    )
    assert result.inference is not None
    assert result.inference.method == "gsc_parametric_reference_ar"
    assert np.isfinite(result.inference.standard_error).all()


def test_gsc_bootstrap_batching_preserves_draws() -> None:
    common = dict(
        ranks=(1,),
        folds=2,
        seed=43,
        max_iter=100,
        bootstrap_mode="reference_empirical",
        n_bootstrap=3,
    )
    scalar_batches = fit_gsc(
        _synthetic_panel(),
        config=GSCConfig(**common, inference_batch_size=1),
        runtime=TorchRuntime(RuntimeConfig(device="cpu", seed=43)),
    )
    combined_batch = fit_gsc(
        _synthetic_panel(),
        config=GSCConfig(**common, inference_batch_size=3),
        runtime=TorchRuntime(RuntimeConfig(device="cpu", seed=43)),
    )
    np.testing.assert_allclose(
        combined_batch.bootstrap_draws,
        scalar_batches.bootstrap_draws,
        atol=1e-10,
        rtol=1e-10,
    )
