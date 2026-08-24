"""Masked fixed-effect and low-rank primitives shared by GSC and MC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .runtime import TorchRuntime

Force = Literal["none", "unit", "time", "two-way"]


@dataclass(slots=True)
class FixedEffects:
    grand: Any
    unit: Any
    time: Any

    def matrix(self) -> Any:
        return self.grand + self.time[:, None] + self.unit[None, :]


@dataclass(slots=True)
class LowRankFit:
    fitted: Any
    fixed: FixedEffects
    low_rank: Any
    singular_values: Any
    iterations: int
    converged: bool
    objective: float


def _masked_average(values: Any, mask: Any, dim: int) -> Any:
    mask_bool = mask.bool()
    numerator = values.masked_fill(~mask_bool, 0).sum(dim=dim)
    denominator = mask.sum(dim=dim).clamp_min(1)
    return numerator / denominator


def fit_fixed_effects(
    y: Any,
    mask: Any,
    *,
    force: Force = "two-way",
    max_iter: int = 100,
    tol: float = 1e-10,
) -> FixedEffects:
    """Least-squares additive effects for an arbitrary observation mask."""
    if y.ndim != 2 or mask.shape != y.shape:
        raise ValueError("y and mask must be equally shaped two-dimensional tensors")
    if not bool(mask.any()):
        raise ValueError("at least one observed cell is required")
    if force not in {"none", "unit", "time", "two-way"}:
        raise ValueError(f"unsupported force mode: {force}")
    mask_f = mask.to(dtype=y.dtype)
    grand = y.masked_fill(~mask.bool(), 0).sum() / mask_f.sum()
    unit = y.new_zeros(y.shape[1])
    time = y.new_zeros(y.shape[0])
    if force == "none":
        # ``fect(force = "none")`` still estimates the overall intercept.
        return FixedEffects(grand=grand, unit=unit, time=time)
    if force == "unit":
        unit = _masked_average(y, mask_f, dim=0)
        return FixedEffects(grand=y.new_zeros(()), unit=unit, time=time)
    if force == "time":
        time = _masked_average(y, mask_f, dim=1)
        return FixedEffects(grand=y.new_zeros(()), unit=unit, time=time)

    for _ in range(max_iter):
        previous_unit = unit.clone()
        previous_time = time.clone()
        unit = _masked_average(y - grand - time[:, None], mask_f, dim=0)
        unit_weight = mask_f.sum(dim=0)
        shift = (unit * unit_weight).sum() / unit_weight.sum().clamp_min(1)
        unit = unit - shift
        grand = grand + shift
        time = _masked_average(y - grand - unit[None, :], mask_f, dim=1)
        time_weight = mask_f.sum(dim=1)
        shift = (time * time_weight).sum() / time_weight.sum().clamp_min(1)
        time = time - shift
        grand = grand + shift
        delta = max(
            float((unit - previous_unit).abs().max()),
            float((time - previous_time).abs().max()),
        )
        if delta <= tol:
            break
    return FixedEffects(grand=grand, unit=unit, time=time)


def _relative_change(current: Any, previous: Any) -> float:
    denominator = previous.norm().clamp_min(1)
    return float((current - previous).norm() / denominator)


def _fect_complete_fixed_effects(y: Any, force: Force) -> tuple[FixedEffects, Any]:
    """Replicate ``fect::Y_demean`` and ``fect::fe_add`` on a full matrix."""
    grand = y.mean()
    unit = y.new_zeros(y.shape[1])
    time = y.new_zeros(y.shape[0])
    if force == "none":
        residual = y - grand
    elif force == "unit":
        column_mean = y.mean(dim=0)
        unit = column_mean - grand
        residual = y - column_mean[None, :]
    elif force == "time":
        row_mean = y.mean(dim=1)
        time = row_mean - grand
        residual = y - row_mean[:, None]
    elif force == "two-way":
        column_mean = y.mean(dim=0)
        row_mean = y.mean(dim=1)
        unit = column_mean - grand
        time = row_mean - grand
        residual = y - column_mean[None, :] - row_mean[:, None] + grand
    else:
        raise ValueError(f"unsupported force mode: {force}")
    return FixedEffects(grand=grand, unit=unit, time=time), residual


def _fect_complete_fixed_effects_batched(
    y: Any,
    force: Force,
) -> tuple[Any, Any, Any, Any]:
    """Vectorised complete-panel demeaning for ``batch x time x unit`` tensors."""
    if y.ndim != 3:
        raise ValueError("batched complete fixed effects require a three-dimensional tensor")
    grand = y.mean(dim=(-2, -1))
    unit = y.new_zeros((y.shape[0], y.shape[2]))
    time = y.new_zeros((y.shape[0], y.shape[1]))
    if force == "none":
        residual = y - grand[:, None, None]
    elif force == "unit":
        column_mean = y.mean(dim=-2)
        unit = column_mean - grand[:, None]
        residual = y - column_mean[:, None, :]
    elif force == "time":
        row_mean = y.mean(dim=-1)
        time = row_mean - grand[:, None]
        residual = y - row_mean[:, :, None]
    elif force == "two-way":
        column_mean = y.mean(dim=-2)
        row_mean = y.mean(dim=-1)
        unit = column_mean - grand[:, None]
        time = row_mean - grand[:, None]
        residual = (
            y
            - column_mean[:, None, :]
            - row_mean[:, :, None]
            + grand[:, None, None]
        )
    else:
        raise ValueError(f"unsupported force mode: {force}")
    return grand, unit, time, residual


def fit_fixed_effects_batched(
    y: Any,
    masks: Any,
    *,
    force: Force = "two-way",
    max_iter: int = 100,
    tol: float = 1e-10,
) -> list[FixedEffects]:
    """Fit independent additive models while synchronising once per batch iteration."""
    if masks.ndim != 3 or masks.shape[1:] != y.shape[-2:]:
        raise ValueError("masks must have shape batch x time x unit")
    if y.ndim == 2:
        y_batch = y.unsqueeze(0).expand(masks.shape[0], -1, -1)
    elif y.ndim == 3 and y.shape == masks.shape:
        y_batch = y
    else:
        raise ValueError("y must be time x unit or match the batched masks")
    mask_bool = masks.bool()
    mask_f = mask_bool.to(dtype=y_batch.dtype)
    if bool((mask_f.sum(dim=(-2, -1)) == 0).any()):
        raise ValueError("every fixed-effect batch member needs an observed cell")
    grand = y_batch.masked_fill(~mask_bool, 0).sum(dim=(-2, -1)) / mask_f.sum(
        dim=(-2, -1)
    )
    unit = y_batch.new_zeros((masks.shape[0], masks.shape[2]))
    time = y_batch.new_zeros((masks.shape[0], masks.shape[1]))
    if force == "none":
        return [
            FixedEffects(grand=grand[index], unit=unit[index], time=time[index])
            for index in range(masks.shape[0])
        ]
    if force == "unit":
        unit = (
            y_batch.masked_fill(~mask_bool, 0).sum(dim=-2)
            / mask_f.sum(dim=-2).clamp_min(1)
        )
        grand = y_batch.new_zeros(grand.shape)
        return [
            FixedEffects(grand=grand[index], unit=unit[index], time=time[index])
            for index in range(masks.shape[0])
        ]
    if force == "time":
        time = (
            y_batch.masked_fill(~mask_bool, 0).sum(dim=-1)
            / mask_f.sum(dim=-1).clamp_min(1)
        )
        grand = y_batch.new_zeros(grand.shape)
        return [
            FixedEffects(grand=grand[index], unit=unit[index], time=time[index])
            for index in range(masks.shape[0])
        ]
    if force != "two-way":
        raise ValueError(f"unsupported force mode: {force}")

    active = y_batch.new_ones(masks.shape[0], dtype=mask_bool.dtype)
    for _ in range(max_iter):
        previous_unit = unit
        previous_time = time
        candidate_unit = (
            (y_batch - grand[:, None, None] - time[:, :, None])
            .masked_fill(~mask_bool, 0)
            .sum(dim=-2)
            / mask_f.sum(dim=-2).clamp_min(1)
        )
        unit_weight = mask_f.sum(dim=-2)
        shift = (candidate_unit * unit_weight).sum(dim=-1) / unit_weight.sum(
            dim=-1
        ).clamp_min(1)
        candidate_unit = candidate_unit - shift[:, None]
        candidate_grand = grand + shift
        candidate_time = (
            (y_batch - candidate_grand[:, None, None] - candidate_unit[:, None, :])
            .masked_fill(~mask_bool, 0)
            .sum(dim=-1)
            / mask_f.sum(dim=-1).clamp_min(1)
        )
        time_weight = mask_f.sum(dim=-1)
        shift = (candidate_time * time_weight).sum(dim=-1) / time_weight.sum(
            dim=-1
        ).clamp_min(1)
        candidate_time = candidate_time - shift[:, None]
        candidate_grand = candidate_grand + shift
        delta = (candidate_unit - previous_unit).abs().amax(dim=-1)
        delta = delta.maximum((candidate_time - previous_time).abs().amax(dim=-1))
        active_matrix = active[:, None]
        unit = unit.where(~active_matrix, candidate_unit)
        time = time.where(~active_matrix, candidate_time)
        grand = grand.where(~active, candidate_grand)
        active = active & (delta > tol)
        if not bool(active.any()):
            break
    return [
        FixedEffects(grand=grand[index], unit=unit[index], time=time[index])
        for index in range(masks.shape[0])
    ]


def _fixed_matrix(grand: Any, unit: Any, time: Any) -> Any:
    return grand[:, None, None] + time[:, :, None] + unit[:, None, :]


def _batched_results(
    *,
    y: Any,
    masks: Any,
    fitted: Any,
    grand: Any,
    unit: Any,
    time: Any,
    low_rank: Any,
    singular_values: Any,
    iterations: Any,
    converged: Any,
    penalties: Any | None = None,
) -> list[LowRankFit]:
    residual = (y - fitted).masked_fill(~masks.bool(), 0)
    mse = residual.square().sum(dim=(-2, -1)) / masks.sum(dim=(-2, -1))
    if penalties is not None:
        mse = mse + penalties * singular_values.sum(dim=-1)
    return [
        LowRankFit(
            fitted=fitted[index],
            fixed=FixedEffects(
                grand=grand[index],
                unit=unit[index],
                time=time[index],
            ),
            low_rank=low_rank[index],
            singular_values=singular_values[index],
            iterations=int(iterations[index]),
            converged=bool(converged[index]),
            objective=float(mse[index].detach().cpu()),
        )
        for index in range(fitted.shape[0])
    ]


def fit_interactive_fixed_effects_panel_batch(
    y: Any,
    masks: Any,
    *,
    ranks: tuple[int, ...],
    force: Force = "two-way",
    max_iter: int = 500,
    tol: float = 1e-8,
) -> list[LowRankFit]:
    """Fit independent panels/ranks as one batched SVD stream."""
    if masks.ndim != 3 or not ranks or len(ranks) != masks.shape[0]:
        raise ValueError("ranks and batch x time x unit masks must have equal batch size")
    batch = masks.shape[0]
    y_batch = y.unsqueeze(0).expand(batch, -1, -1) if y.ndim == 2 else y
    if y_batch.shape != masks.shape:
        raise ValueError("y does not match the batched IFE masks")
    matrix_rank = min(y_batch.shape[-2:])
    if min(ranks) < 0 or max(ranks) > matrix_rank:
        raise ValueError("a rank is outside the panel dimensions")
    initial_fixed = fit_fixed_effects_batched(
        y_batch,
        masks.bool(),
        force=force,
        max_iter=max_iter,
        tol=min(tol * 0.1, 1e-12),
    )
    fitted = y_batch.new_empty(y_batch.shape)
    for index, fixed in enumerate(initial_fixed):
        fitted[index] = fixed.matrix()
    previous_fit = fitted.clone()
    low_rank = y_batch.new_zeros(y_batch.shape)
    previous_low_rank = low_rank.clone()
    grand, unit, time, _ = _fect_complete_fixed_effects_batched(fitted, force)
    singular_values = y_batch.new_zeros((batch, matrix_rank))
    rank_tensor = y_batch.new_tensor(ranks).long()
    active = y_batch.new_ones(batch, dtype=masks.dtype)
    iterations = y_batch.new_zeros(batch, dtype=rank_tensor.dtype)
    component = y_batch.new_tensor(range(matrix_rank)).long()
    for iteration in range(1, max_iter + 2):
        completed = y_batch.where(masks, fitted)
        candidate_grand, candidate_unit, candidate_time, residual = (
            _fect_complete_fixed_effects_batched(completed, force)
        )
        left, raw_values, right = residual.svd()
        right_t = right.transpose(-2, -1)
        kept = raw_values * (component[None, :] < rank_tensor[:, None])
        candidate_low_rank = (left * kept[:, None, :]) @ right_t
        candidate_fitted = (
            _fixed_matrix(candidate_grand, candidate_unit, candidate_time)
            + candidate_low_rank
        )
        change = (candidate_fitted - previous_fit).norm(dim=(-2, -1)) / (
            previous_fit.norm(dim=(-2, -1)) + 1e-10
        )
        previous_norm = previous_low_rank.norm(dim=(-2, -1))
        interactive_change = (candidate_low_rank - previous_low_rank).norm(
            dim=(-2, -1)
        ) / previous_norm.clamp_min(1e-10)
        change = change.maximum(
            interactive_change.where(
                (rank_tensor > 0) & (previous_norm > 1e-10),
                interactive_change.new_zeros(interactive_change.shape),
            )
        )
        active_matrix = active[:, None, None]
        fitted = fitted.where(~active_matrix, candidate_fitted)
        low_rank = low_rank.where(~active_matrix, candidate_low_rank)
        singular_values = singular_values.where(~active[:, None], raw_values)
        grand = grand.where(~active, candidate_grand)
        unit = unit.where(~active[:, None], candidate_unit)
        time = time.where(~active[:, None], candidate_time)
        iterations = iterations.where(~active, iterations.new_full(iterations.shape, iteration))
        previous_fit = previous_fit.where(~active_matrix, candidate_fitted)
        previous_low_rank = previous_low_rank.where(~active_matrix, candidate_low_rank)
        active = active & (change > tol)
        if not bool(active.any()):
            break
    return _batched_results(
        y=y_batch,
        masks=masks,
        fitted=fitted,
        grand=grand,
        unit=unit,
        time=time,
        low_rank=low_rank,
        singular_values=singular_values,
        iterations=iterations,
        converged=~active,
    )


def fit_interactive_fixed_effects_batched(
    y: Any,
    mask: Any,
    *,
    ranks: tuple[int, ...],
    force: Force = "two-way",
    max_iter: int = 500,
    tol: float = 1e-8,
) -> list[LowRankFit]:
    """Fit all rank candidates for one CV fold as one batched SVD stream."""
    masks = mask.bool().unsqueeze(0).expand(len(ranks), -1, -1)
    return fit_interactive_fixed_effects_panel_batch(
        y,
        masks,
        ranks=ranks,
        force=force,
        max_iter=max_iter,
        tol=tol,
    )


def fit_nuclear_norm_completion_batched(
    y: Any,
    masks: Any,
    *,
    penalty: float,
    initial_fits: Any,
    force: Force = "two-way",
    max_iter: int = 500,
    tol: float = 1e-8,
) -> list[LowRankFit]:
    """Fit one MC lambda over all CV folds using batched SVDs."""
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    if masks.ndim != 3 or initial_fits.shape != masks.shape:
        raise ValueError("masks and initial_fits must be batch x time x unit")
    batch = masks.shape[0]
    y_batch = y.unsqueeze(0).expand(batch, -1, -1) if y.ndim == 2 else y
    if y_batch.shape != masks.shape:
        raise ValueError("y does not match the batched MC masks")
    fitted = initial_fits.clone()
    previous_fit = fitted.clone()
    low_rank = y_batch.new_zeros(y_batch.shape)
    grand, unit, time, _ = _fect_complete_fixed_effects_batched(fitted, force)
    singular_values = y_batch.new_zeros((batch, min(y_batch.shape[-2:])))
    active = y_batch.new_ones(batch, dtype=masks.bool().dtype)
    iterations = y_batch.new_zeros(batch).long()
    threshold = penalty * y_batch.shape[-2] * y_batch.shape[-1]
    for iteration in range(1, max_iter + 2):
        completed = y_batch.where(masks.bool(), fitted)
        candidate_grand, candidate_unit, candidate_time, residual = (
            _fect_complete_fixed_effects_batched(completed, force)
        )
        left, raw_values, right = residual.svd()
        right_t = right.transpose(-2, -1)
        shrunk = (raw_values - threshold).clamp_min(0)
        candidate_low_rank = (left * shrunk[:, None, :]) @ right_t
        candidate_fitted = (
            _fixed_matrix(candidate_grand, candidate_unit, candidate_time)
            + candidate_low_rank
        )
        change = (candidate_fitted - previous_fit).norm(dim=(-2, -1)) / (
            previous_fit.norm(dim=(-2, -1)) + 1e-10
        )
        active_matrix = active[:, None, None]
        fitted = fitted.where(~active_matrix, candidate_fitted)
        low_rank = low_rank.where(~active_matrix, candidate_low_rank)
        singular_values = singular_values.where(~active[:, None], shrunk)
        grand = grand.where(~active, candidate_grand)
        unit = unit.where(~active[:, None], candidate_unit)
        time = time.where(~active[:, None], candidate_time)
        iterations = iterations.where(~active, iterations.new_full(iterations.shape, iteration))
        previous_fit = previous_fit.where(~active_matrix, candidate_fitted)
        active = active & (change > tol)
        if not bool(active.any()):
            break
    penalties = y_batch.new_full((batch,), penalty)
    return _batched_results(
        y=y_batch,
        masks=masks,
        fitted=fitted,
        grand=grand,
        unit=unit,
        time=time,
        low_rank=low_rank,
        singular_values=singular_values,
        iterations=iterations,
        converged=~active,
        penalties=penalties,
    )


def fit_interactive_fixed_effects(
    y: Any,
    mask: Any,
    *,
    rank: int,
    force: Force = "two-way",
    max_iter: int = 500,
    tol: float = 1e-8,
) -> LowRankFit:
    """Fit fect-compatible additive and rank-constrained interactive effects.

    This follows ``fect::fe_ad_inter_iter`` rather than alternating fixed
    effects only over observed cells.  The E step fills every missing cell
    with the previous *overall* fit; the M step then applies complete-panel
    demeaning followed by a hard rank-``r`` SVD.  That ordering and the
    upstream convergence denominator are material for rolling-CV parity.
    """
    if rank < 0 or rank > min(y.shape):
        raise ValueError("rank is outside the panel dimensions")
    mask_f = mask.to(dtype=y.dtype)
    mask_bool = mask.bool()
    low_rank = y.new_zeros(y.shape)
    singular_values = y.new_zeros(0)
    converged = False
    iterations = 0
    initial_fixed = fit_fixed_effects(
        y,
        mask_bool,
        force=force,
        max_iter=max_iter,
        tol=min(tol * 0.1, 1e-12),
    )
    fitted = initial_fixed.matrix()
    previous_fit = fitted.clone()
    previous_low_rank = low_rank.clone()
    fixed, _ = _fect_complete_fixed_effects(fitted, force)
    # The C++ loop condition is ``niter <= max_iter``.
    for iteration in range(1, max_iter + 2):
        iterations = iteration
        completed = y.where(mask_bool, fitted)
        fixed, residual = _fect_complete_fixed_effects(completed, force)
        if rank:
            left, singular_values, right = residual.svd()
            right_t = right.transpose(0, 1)
            keep = min(rank, singular_values.numel())
            low_rank = (left[:, :keep] * singular_values[:keep]) @ right_t[:keep, :]
        else:
            low_rank = y.new_zeros(y.shape)
            singular_values = y.new_zeros(0)
        fitted = fixed.matrix() + low_rank
        change = float(
            (fitted - previous_fit).norm() / (previous_fit.norm() + 1e-10)
        )
        if rank:
            previous_interactive_norm = float(previous_low_rank.norm())
            if previous_interactive_norm > 1e-10:
                interactive_change = float(
                    (low_rank - previous_low_rank).norm() / previous_interactive_norm
                )
                change = max(change, interactive_change)
        previous_fit = fitted.clone()
        previous_low_rank = low_rank.clone()
        if change <= tol:
            converged = True
            break
    residual_observed = (y - fitted).masked_fill(~mask_bool, 0)
    objective = float((residual_observed.square().sum() / mask_f.sum()).detach().cpu())
    return LowRankFit(
        fitted=fitted,
        fixed=fixed,
        low_rank=low_rank,
        singular_values=singular_values,
        iterations=iterations,
        converged=converged,
        objective=objective,
    )


def fit_nuclear_norm_completion(
    y: Any,
    mask: Any,
    *,
    penalty: float,
    force: Force = "two-way",
    max_iter: int = 500,
    tol: float = 1e-8,
    initial_fit: Any | None = None,
) -> LowRankFit:
    """Fit additive effects plus fect-compatible nuclear-norm completion.

    ``fect::panel_FE`` applies SVD to ``E / (T * N)``, subtracts ``lambda``,
    and multiplies the reconstruction by ``T * N``.  Applying the equivalent
    threshold ``lambda * T * N`` to the unscaled matrix avoids two extra
    full-matrix operations while preserving the package's lambda contract.

    The EM order intentionally follows ``fect::fe_ad_inter_iter``: initialise
    from an additive fixed-effect regression, fill missing cells with the
    previous *overall* fit, then perform complete-panel demeaning and singular
    value shrinkage.  A warm start therefore contains the overall fit rather
    than only its low-rank component.
    """
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    mask_f = mask.to(dtype=y.dtype)
    mask_bool = mask.bool()
    singular_value_threshold = penalty * y.numel()
    if initial_fit is not None and initial_fit.shape != y.shape:
        raise ValueError("initial_fit must match y")
    if initial_fit is None:
        initial_fixed = fit_fixed_effects(
            y,
            mask_bool,
            force=force,
            max_iter=max_iter,
            tol=min(tol * 0.1, 1e-12),
        )
        fitted = initial_fixed.matrix()
    else:
        fitted = initial_fit.detach().clone()
    previous_fit = fitted.clone()
    low_rank = y.new_zeros(y.shape)
    singular_values = y.new_zeros(0)
    converged = False
    iterations = 0
    fixed, _ = _fect_complete_fixed_effects(fitted, force)
    # The upstream C++ condition is ``niter <= max_iter`` and can therefore
    # execute max_iter + 1 updates.  Preserve that detail for parity.
    for iteration in range(1, max_iter + 2):
        iterations = iteration
        completed = y.where(mask_bool, fitted)
        fixed, residual = _fect_complete_fixed_effects(completed, force)
        # ``panel_FE`` receives the demeaned matrix, not the completed outcome.
        left, raw_values, right = residual.svd()
        right_t = right.transpose(0, 1)
        singular_values = (raw_values - singular_value_threshold).clamp_min(0)
        keep = int((singular_values > 0).sum())
        if keep:
            low_rank = (left[:, :keep] * singular_values[:keep]) @ right_t[:keep, :]
        else:
            low_rank = y.new_zeros(y.shape)
        fitted = fixed.matrix() + low_rank
        change = float(
            (fitted - previous_fit).norm() / (previous_fit.norm() + 1e-10)
        )
        previous_fit = fitted.clone()
        if change <= tol:
            converged = True
            break
    residual_observed = (y - fitted).masked_fill(~mask_bool, 0)
    mse = residual_observed.square().sum() / mask_f.sum()
    objective = float((mse + penalty * singular_values.sum()).detach().cpu())
    return LowRankFit(
        fitted=fitted,
        fixed=fixed,
        low_rank=low_rank,
        singular_values=singular_values,
        iterations=iterations,
        converged=converged,
        objective=objective,
    )


def as_panel_tensors(panel: Any, runtime: TorchRuntime) -> tuple[Any, Any, Any]:
    """Move a validated :class:`PanelData` to the configured device."""
    observed = runtime.tensor(panel.observed, dtype=runtime.torch.bool)
    raw_y = runtime.tensor(panel.y)
    # IEEE arithmetic keeps NaN when multiplied by zero.  Replace genuinely
    # unobserved cells before any masked linear algebra so missing panels do
    # not poison otherwise valid fixed-effect/SVD fits.
    y = raw_y.masked_fill(~observed, 0)
    treated = runtime.tensor(panel.treated, dtype=runtime.torch.bool)
    return y, observed, treated
