from __future__ import annotations

import numpy as np

from urban_intervention.causal.gpu.linalg import (
    fit_fixed_effects,
    fit_fixed_effects_batched,
    fit_interactive_fixed_effects,
    fit_interactive_fixed_effects_batched,
    fit_nuclear_norm_completion,
    fit_nuclear_norm_completion_batched,
)
from urban_intervention.causal.gpu.runtime import RuntimeConfig, TorchRuntime


def test_two_way_fixed_effects_recover_additive_panel() -> None:
    runtime = TorchRuntime(RuntimeConfig(device="cpu"))
    y = runtime.tensor(
        np.array([1.0, 2.0, 4.0])[:, None] + np.array([-2.0, 0.5, 3.0])[None, :]
    )
    mask = runtime.torch.ones_like(y, dtype=runtime.torch.bool)
    fit = fit_fixed_effects(y, mask)
    np.testing.assert_allclose(fit.matrix().cpu(), y.cpu(), atol=1e-9)


def test_rank_one_interactive_fit_recovers_complete_panel() -> None:
    runtime = TorchRuntime(RuntimeConfig(device="cpu"))
    time = np.linspace(-1, 1, 8)
    loading = np.array([-1.5, 0.2, 0.8, 2.0])
    y_np = 2.0 + time[:, None] * loading[None, :]
    y = runtime.tensor(y_np)
    mask = runtime.torch.ones_like(y, dtype=runtime.torch.bool)
    fit = fit_interactive_fixed_effects(y, mask, rank=1)
    np.testing.assert_allclose(fit.fitted.cpu(), y_np, atol=1e-7)
    assert fit.converged


def test_nuclear_completion_does_not_use_masked_cell() -> None:
    runtime = TorchRuntime(RuntimeConfig(device="cpu"))
    y_np = np.add.outer(np.arange(6, dtype=float), np.arange(4, dtype=float))
    y = runtime.tensor(y_np)
    mask = runtime.torch.ones_like(y, dtype=runtime.torch.bool)
    mask[5, 0] = False
    poisoned = y.clone()
    poisoned[5, 0] = 1e9
    fit = fit_nuclear_norm_completion(poisoned, mask, penalty=0.0)
    assert abs(float(fit.fitted[5, 0]) - y_np[5, 0]) < 1e-5


def test_masked_nan_does_not_poison_fixed_effects_or_completion() -> None:
    runtime = TorchRuntime(RuntimeConfig(device="cpu"))
    y = runtime.tensor([[1.0, np.nan], [2.0, 3.0], [3.0, 4.0]])
    mask = runtime.torch.isfinite(y)
    fixed = fit_fixed_effects(y, mask)
    completion = fit_nuclear_norm_completion(y, mask, penalty=0.1)
    assert runtime.torch.isfinite(fixed.matrix()).all()
    assert runtime.torch.isfinite(completion.fitted).all()


def test_nuclear_completion_uses_fect_normalized_lambda() -> None:
    runtime = TorchRuntime(RuntimeConfig(device="cpu"))
    y = runtime.tensor(np.diag([8.0, 2.0]))
    mask = runtime.torch.ones_like(y, dtype=runtime.torch.bool)
    # T*N = 4, so lambda=.5 removes 2 from every singular value after
    # additive effects have been accounted for.
    fit = fit_nuclear_norm_completion(y, mask, penalty=0.5, force="none", max_iter=5)
    assert float(fit.singular_values.max()) <= 6.0 + 1e-10


def test_batched_ife_matches_independent_rank_fits() -> None:
    runtime = TorchRuntime(RuntimeConfig(device="cpu"))
    y = runtime.tensor(
        2
        + np.linspace(-1, 1, 9)[:, None]
        * np.array([-1.0, -0.2, 0.7, 1.4])[None, :]
    )
    mask = runtime.torch.ones_like(y, dtype=runtime.torch.bool)
    mask[-2:, -1] = False
    batched = fit_interactive_fixed_effects_batched(
        y, mask, ranks=(0, 1, 2), max_iter=100, tol=1e-5
    )
    scalar = [
        fit_interactive_fixed_effects(y, mask, rank=rank, max_iter=100, tol=1e-5)
        for rank in (0, 1, 2)
    ]
    for accelerated, reference in zip(batched, scalar, strict=True):
        np.testing.assert_allclose(
            accelerated.fitted.cpu(), reference.fitted.cpu(), atol=1e-10, rtol=1e-10
        )
        assert accelerated.converged == reference.converged
        assert accelerated.iterations == reference.iterations


def test_batched_mc_matches_independent_fold_fits() -> None:
    runtime = TorchRuntime(RuntimeConfig(device="cpu"))
    y = runtime.tensor(np.add.outer(np.arange(8, dtype=float), np.arange(5, dtype=float)))
    masks = runtime.torch.ones((3, 8, 5), dtype=runtime.torch.bool)
    masks[0, -1, 0] = False
    masks[1, -2:, 1] = False
    masks[2, -3:, 2] = False
    fixed = fit_fixed_effects_batched(y, masks, max_iter=100, tol=1e-12)
    initial = runtime.torch.stack([value.matrix() for value in fixed])
    batched = fit_nuclear_norm_completion_batched(
        y,
        masks,
        penalty=0.01,
        initial_fits=initial,
        max_iter=100,
        tol=1e-5,
    )
    scalar = [
        fit_nuclear_norm_completion(
            y,
            mask,
            penalty=0.01,
            initial_fit=start,
            max_iter=100,
            tol=1e-5,
        )
        for mask, start in zip(masks, initial, strict=True)
    ]
    for accelerated, reference in zip(batched, scalar, strict=True):
        np.testing.assert_allclose(
            accelerated.fitted.cpu(), reference.fitted.cpu(), atol=1e-10, rtol=1e-10
        )
        assert accelerated.converged == reference.converged
        assert accelerated.iterations == reference.iterations
