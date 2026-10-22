"""Statistical evaluation of response-aligned urban representations.

Provides:

- `response_similarity`: vectorized pairwise correlation over commonly-observed
  response cells (shared with the training loss).
- `retrieval_metrics`: nearest-neighbour retrieval quality (`nn_corr@k`) against
  response similarity, overall and per outcome family, with a random-neighbour
  baseline.
- `bootstrap_ci`: unit-resampling confidence interval for `nn_corr@k`.
- `permutation_test`: chance-level distribution obtained by shuffling response
  rows across units; the p-value answers "is the association between embedding
  proximity and response similarity better than random?".
- `probe_rmse`: linear-probe transfer metric (per-cell ridge fit on the train
  pool, evaluated on the target pool) that compares representation vs raw
  features directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .dataset import RESPONSE_DIM, RESPONSE_OFFSETS


def response_similarity(
    responses: torch.Tensor,
    masks: torch.Tensor,
    families: list[str] | None = None,
    weights: torch.Tensor | None = None,
    precision: str = "float32",
    response_offsets: dict[str, tuple[int, int, list[str]]] | None = None,
) -> torch.Tensor:
    """Vectorized pairwise correlation over commonly-observed response cells.

    With ``families=None`` and ``responses`` of the full response dimension,
    the per-family correlations are averaged exactly like the training loss.
    Passing a subset of family names restricts the average to those segments.
    ``weights`` applies the per-label design reliability: cell ``k`` of pair
    ``(i, j)`` carries weight ``sqrt(w_i w_j)``.  Without weights the plain
    correlation is returned.

    ``response_offsets`` is the dataset-specific family layout.  It is required
    for per-family releases whose response dimension is not the full 27-cell
    layout; otherwise ``families`` cannot be resolved without silently using
    the wrong slices.

    ``precision="float64"`` recomputes the covariances in double precision,
    which avoids the cancellation error of the one-pass sum-of-squares formula
    on response paths whose level dominates their variation.
    """
    similarity, _ = response_similarity_with_validity(
        responses,
        masks,
        families,
        weights,
        precision=precision,
        response_offsets=response_offsets,
    )
    return similarity


def response_similarity_with_validity(
    responses: torch.Tensor,
    masks: torch.Tensor,
    families: list[str] | None = None,
    weights: torch.Tensor | None = None,
    precision: str = "float32",
    response_offsets: dict[str, tuple[int, int, list[str]]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return response correlation and whether each pair is comparable.

    A pair is comparable when at least one requested outcome family has two
    commonly observed, non-constant response cells.  Correlations are averaged
    over comparable families only; missing families are not interpreted as
    zero response similarity.
    """
    return response_similarity_between(
        responses,
        masks,
        responses,
        masks,
        families,
        weights,
        weights,
        precision=precision,
        response_offsets=response_offsets,
    )


def response_similarity_between(
    responses_a: torch.Tensor,
    masks_a: torch.Tensor,
    responses_b: torch.Tensor,
    masks_b: torch.Tensor,
    families: list[str] | None = None,
    weights_a: torch.Tensor | None = None,
    weights_b: torch.Tensor | None = None,
    precision: str = "float32",
    response_offsets: dict[str, tuple[int, int, list[str]]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rectangular response similarity without constructing a square joint pool.

    With per-label design weights the per-family statistic becomes the weighted
    correlation over commonly observed cells with pairwise weight
    ``sqrt(w_i w_j)``; a pair whose weighted common support is below two cells
    (or whose weighted variance is degenerate) is reported as invalid instead
    of receiving a zero similarity, because zero would mean "orthogonal", not
    "no evidence".  ``precision="float64"`` upcasts the inputs so the
    one-pass covariance formula cannot lose the signal to cancellation.
    """
    if precision not in {"float32", "float64"}:
        raise ValueError(f"Unknown precision: {precision}")
    n_a, dim = responses_a.shape
    n_b = responses_b.shape[0]
    if responses_b.shape[1] != dim:
        raise ValueError("Response pools must have the same response dimension")
    if masks_a.shape != responses_a.shape or masks_b.shape != responses_b.shape:
        raise ValueError("Each response mask must match its response tensor")
    if responses_a.device != responses_b.device:
        raise ValueError("Response pools must be on the same device")
    if precision == "float64" and responses_a.dtype != torch.float64:
        responses_a = responses_a.to(torch.float64)
        responses_b = responses_b.to(torch.float64)
        if weights_a is not None:
            weights_a = weights_a.to(torch.float64)
        if weights_b is not None:
            weights_b = weights_b.to(torch.float64)
    weighted = weights_a is not None or weights_b is not None
    if weighted:
        if weights_a is None:
            weights_a = torch.ones_like(responses_a)
        if weights_b is None:
            weights_b = torch.ones_like(responses_b)
        if weights_a.shape != responses_a.shape or weights_b.shape != responses_b.shape:
            raise ValueError("Each response weight must match its response tensor")
        if weights_a.device != responses_a.device or weights_b.device != responses_b.device:
            raise ValueError("Response weights must be on the same device")
        weights_a = torch.nan_to_num(weights_a, nan=0.0).clamp_min(0.0)
        weights_b = torch.nan_to_num(weights_b, nan=0.0).clamp_min(0.0)
    sim_sum = responses_a.new_zeros(n_a, n_b)
    valid_count = torch.zeros(n_a, n_b, dtype=torch.int32, device=responses_a.device)

    offsets = response_offsets if response_offsets is not None else (
        RESPONSE_OFFSETS if dim == RESPONSE_DIM else None
    )
    if offsets is None:
        if families is not None:
            raise ValueError(
                "families requires response_offsets when the response dimension is "
                "not the full 27-cell layout"
            )
        segments = [(0, dim)]
    else:
        family_names = list(families) if families is not None else list(offsets)
        for name in family_names:
            if name not in offsets:
                raise ValueError(f"Unknown outcome family: {name}")
        segments = [(offsets[name][0], offsets[name][1]) for name in family_names]
        if any(end > dim for _, end in segments):
            raise ValueError(
                f"response_offsets exceed the response dimension {dim}: {segments}"
            )

    eps = 1e-10
    for start, end in segments:
        ma = masks_a[:, start:end]
        mb = masks_b[:, start:end]
        if not ma.any() or not mb.any():
            continue
        xa = torch.where(
            ma, responses_a[:, start:end], torch.zeros_like(responses_a[:, start:end])
        )
        xb = torch.where(
            mb, responses_b[:, start:end], torch.zeros_like(responses_b[:, start:end])
        )
        if weighted:
            assert weights_a is not None and weights_b is not None
            pair_weight = (ma.unsqueeze(1) & mb.unsqueeze(0)).to(responses_a.dtype) * (
                torch.sqrt(
                    weights_a[:, start:end].unsqueeze(1)
                    * weights_b[:, start:end].unsqueeze(0)
                )
            )
            reliable = (pair_weight > 0).sum(dim=2)
            support = pair_weight.sum(dim=2).clamp_min(eps)
            xa3 = xa.unsqueeze(1)
            xb3 = xb.unsqueeze(0)
            wx = pair_weight * xa3
            wy = pair_weight * xb3
            sx = wx.sum(dim=2)
            sy = wy.sum(dim=2)
            sxy = (wx * xb3).sum(dim=2)
            sx2 = (wx * xa3).sum(dim=2)
            sy2 = (wy * xb3).sum(dim=2)
            num = sxy - sx * sy / support
            var_x = (sx2 - sx * sx / support).clamp(min=0.0)
            var_y = (sy2 - sy * sy / support).clamp(min=0.0)
            # Reliability counts cells with positive pair weight, so the
            # validity rule is invariant under re-scaling all weights; the
            # variance is compared per unit of average weight for the same
            # reason.
            mean_weight = support / reliable.clamp_min(1).to(support.dtype)
            valid = (
                (reliable >= 2)
                & (var_x > eps * mean_weight)
                & (var_y > eps * mean_weight)
            )
            denom_std = (var_x * var_y).sqrt().clamp(min=eps)
            corr = torch.where(valid, (num / denom_std).clamp(-1.0, 1.0), 0.0)
        else:
            mfa = ma.to(responses_a.dtype)
            mfb = mb.to(responses_a.dtype)
            xma = xa * mfa
            xmb = xb * mfb
            common = mfa @ mfb.T
            sx = xma @ mfb.T
            sy = mfa @ xmb.T
            sxy = xma @ xmb.T
            sx2 = (xa * xma) @ mfb.T
            sy2 = mfa @ (xb * xmb).T

            denom = common.clamp(min=eps)
            num = sxy - (sx * sy) / denom
            var_x = (sx2 - sx * sx / denom).clamp(min=0.0)
            var_y = (sy2 - sy * sy / denom).clamp(min=0.0)
            denom_std = (var_x * var_y).sqrt().clamp(min=eps)
            valid = (common >= 2) & (var_x > eps) & (var_y > eps)
            corr = torch.where(valid, (num / denom_std).clamp(-1.0, 1.0), 0.0)
        sim_sum = sim_sum + corr
        valid_count = valid_count + valid.to(valid_count.dtype)

    valid_pairs = valid_count > 0
    similarity = torch.where(
        valid_pairs,
        sim_sum / valid_count.clamp(min=1).to(responses_a.dtype),
        torch.zeros_like(sim_sum),
    )
    return similarity, valid_pairs


def _prediction_slice(
    predictions: dict[str, torch.Tensor],
    family: str,
    start: int,
    end: int,
    response_dim: int,
) -> torch.Tensor | None:
    """Resolve a family segment from per-family or full-vector predictions."""
    if family in predictions:
        prediction = predictions[family]
        if prediction.shape[1] == end - start:
            return prediction
        if prediction.shape[1] == response_dim:
            return prediction[:, start:end]
        return None
    if len(predictions) == 1:
        only = next(iter(predictions.values()))
        if only.shape[1] == response_dim:
            return only[:, start:end]
    return None


def _normalize_rows(vectors: torch.Tensor) -> torch.Tensor:
    norm = vectors.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return vectors / norm


def cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    normalized = _normalize_rows(embeddings)
    return normalized @ normalized.T


def _per_anchor_nn_corr(
    emb_cos: torch.Tensor,
    sim_resp: torch.Tensor,
    k: int,
    valid_pairs: torch.Tensor | None = None,
) -> np.ndarray:
    n = emb_cos.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if valid_pairs is None:
        valid_pairs = torch.ones_like(emb_cos, dtype=torch.bool)
    for i in range(n):
        candidates = valid_pairs[i].clone()
        candidates[i] = False
        candidate_count = int(candidates.sum().item())
        if candidate_count == 0:
            continue
        k_eff = min(int(k), candidate_count)
        scores = emb_cos[i].clone()
        scores[~candidates] = -float("inf")
        idx = scores.topk(k_eff).indices
        out[i] = float(sim_resp[i, idx].mean())
    return out


def nn_corr_at_k(
    emb_cos: torch.Tensor,
    sim_resp: torch.Tensor,
    k: int = 5,
    valid_pairs: torch.Tensor | None = None,
) -> float:
    """Mean response similarity between each unit and its k embedding neighbours."""
    n = emb_cos.shape[0]
    if n < 2:
        return 0.0
    per_anchor = _per_anchor_nn_corr(emb_cos, sim_resp, k, valid_pairs)
    return float(np.nanmean(per_anchor)) if np.isfinite(per_anchor).any() else 0.0


def _metric_block(
    emb_cos: torch.Tensor,
    sim_resp: torch.Tensor,
    valid_pairs: torch.Tensor,
    k: int,
) -> dict[str, object]:
    n = emb_cos.shape[0]
    unmeasured: dict[str, object] = {
        "nn_corr@k": None,
        "baseline_corr": None,
        "n_units": n,
        "n_comparable_units": 0,
        "ratio": None,
    }
    if n < 2:
        return unmeasured
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=emb_cos.device)
    valid_off_diagonal = valid_pairs & off_diagonal
    row_count = valid_off_diagonal.sum(dim=1)
    valid_rows = row_count > 0
    if not bool(valid_rows.any()):
        # A family without a single comparable pair (typically the inactive
        # families of a per-family dataset) is reported as unmeasured rather
        # than as a measured zero.
        return unmeasured
    nn = nn_corr_at_k(emb_cos, sim_resp, k, valid_pairs)
    row_sum = (sim_resp * valid_off_diagonal.to(sim_resp.dtype)).sum(dim=1)
    baseline = float((row_sum[valid_rows] / row_count[valid_rows]).mean())
    ratio = round(nn / baseline, 3) if baseline > 0 else None
    return {
        "nn_corr@k": round(nn, 6),
        "baseline_corr": round(baseline, 6),
        "n_units": n,
        "n_comparable_units": int(valid_rows.sum().item()),
        "ratio": ratio,
    }


def retrieval_metrics(
    embeddings: torch.Tensor,
    responses: torch.Tensor,
    masks: torch.Tensor,
    k: int = 5,
    response_offsets: dict[str, tuple[int, int, list[str]]] | None = None,
) -> dict[str, dict[str, object]]:
    """Retrieval quality overall and per outcome family.

    Neighbours are always chosen from embedding cosine similarity; the score is
    the response similarity with those neighbours, evaluated on all families
    jointly (``overall``) and on each family separately.  For per-family
    releases, pass the dataset-specific ``response_offsets`` so each family
    block is computed from its own response segment.
    """
    emb_cos = cosine_similarity(embeddings)
    overall_sim, overall_valid = response_similarity_with_validity(
        responses, masks, response_offsets=response_offsets
    )
    result: dict[str, dict[str, object]] = {
        "overall": _metric_block(emb_cos, overall_sim, overall_valid, k),
    }
    offsets = response_offsets if response_offsets is not None else (
        RESPONSE_OFFSETS if responses.shape[1] == RESPONSE_DIM else None
    )
    if offsets is None:
        return result
    for family in offsets:
        family_sim, family_valid = response_similarity_with_validity(
            responses,
            masks,
            families=[family],
            response_offsets=offsets,
        )
        result[family] = _metric_block(
            emb_cos, family_sim, family_valid, k
        )
    return result


def bootstrap_ci(
    emb_cos: torch.Tensor,
    sim_resp: torch.Tensor,
    k: int = 5,
    n_boot: int = 200,
    seed: int = 42,
    ci: tuple[float, float] = (2.5, 97.5),
    valid_pairs: torch.Tensor | None = None,
) -> dict[str, float]:
    """Unit-resampling bootstrap confidence interval for ``nn_corr@k``."""
    n = emb_cos.shape[0]
    if n < 2 or n_boot <= 0:
        return {"lower": 0.0, "upper": 0.0, "n_boot": 0}
    per_anchor = _per_anchor_nn_corr(emb_cos, sim_resp, k, valid_pairs)
    per_anchor = per_anchor[np.isfinite(per_anchor)]
    if per_anchor.size == 0:
        return {"lower": 0.0, "upper": 0.0, "n_boot": 0}
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, per_anchor.size, size=per_anchor.size)
        means[b] = per_anchor[idx].mean()
    lower, upper = np.percentile(means, list(ci))
    return {
        "lower": round(float(lower), 6),
        "upper": round(float(upper), 6),
        "n_boot": n_boot,
    }


def permutation_test(
    emb_cos: torch.Tensor,
    responses: torch.Tensor,
    masks: torch.Tensor,
    k: int = 5,
    n_perm: int = 200,
    seed: int = 42,
    max_units: int = 512,
    families: list[str] | None = None,
    response_offsets: dict[str, tuple[int, int, list[str]]] | None = None,
) -> dict[str, object]:
    """Chance distribution of ``nn_corr@k`` under shuffled response labels.

    The null hypothesis is that embedding proximity carries no information
    about response similarity. ``max_units`` caps the pool (units are
    subsampled deterministically) to keep large-pool permutations tractable.
    """
    n = responses.shape[0]
    if n < 2 or n_perm <= 0:
        return {
            "p_value": 1.0,
            "observed": 0.0,
            "mean_null": 0.0,
            "sd_null": 0.0,
            "n_perm": 0,
            "n_units": n,
        }
    rng = np.random.RandomState(seed)
    if n > max_units:
        keep = rng.choice(n, max_units, replace=False)
        emb_cos = emb_cos[keep][:, keep]
        responses = responses[keep]
        masks = masks[keep]
        n = max_units

    observed_sim, observed_valid = response_similarity_with_validity(
        responses, masks, families, response_offsets=response_offsets
    )
    observed = nn_corr_at_k(emb_cos, observed_sim, k, observed_valid)
    nulls = np.empty(n_perm, dtype=np.float64)
    for b in range(n_perm):
        idx = rng.permutation(n)
        null_sim, null_valid = response_similarity_with_validity(
            responses[idx], masks[idx], families, response_offsets=response_offsets
        )
        nulls[b] = nn_corr_at_k(emb_cos, null_sim, k, null_valid)
    p_value = float((1 + int((nulls >= observed).sum())) / (1 + n_perm))
    return {
        "p_value": round(p_value, 4),
        "observed": round(float(observed), 6),
        "mean_null": round(float(nulls.mean()), 6),
        "sd_null": round(float(nulls.std()), 6),
        "n_perm": n_perm,
        "n_units": n,
    }


def probe_rmse(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    m_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    m_test: torch.Tensor,
    ridge: float = 1.0,
    min_obs: int = 16,
    families: list[str] | None = None,
    center: bool = True,
    response_offsets: dict[str, tuple[int, int, list[str]]] | None = None,
) -> dict[str, object]:
    """Linear-probe transfer metric: per-response-cell ridge, fitted on train.

    For every response cell with at least ``min_obs`` train observations, a
    ridge regressor is fit from the frozen input features (embeddings or raw
    features) to that cell, then applied to the target pool. The reported RMSE
    is over observed test cells only; cells with insufficient training support
    are excluded and counted in ``fitted_cells``.

    With ``center`` (the default) train-pool feature and target means are
    removed before the ridge solve and re-added to predictions, which is
    equivalent to fitting an intercept.  The report also carries the
    train-mean prediction baseline (``mean_baseline_rmse``) computed on the
    same test cells, so a probe that does not beat predicting the training mean
    cannot be read as evidence of predictability.
    """
    n_train, dim = y_train.shape
    if x_train.shape[0] != n_train or y_test.shape[1] != dim:
        raise ValueError("Probe inputs have inconsistent shapes")
    offsets = response_offsets if response_offsets is not None else (
        RESPONSE_OFFSETS if dim == RESPONSE_DIM else None
    )
    if offsets is None:
        if families is not None:
            raise ValueError(
                "families requires response_offsets when the response dimension is "
                "not the full 27-cell layout"
            )
        family_cells = {"raw": list(range(dim))}
    else:
        family_names = list(families) if families is not None else list(offsets)
        for name in family_names:
            if name not in offsets:
                raise ValueError(f"Unknown outcome family: {name}")
        family_cells = {
            name: list(range(offsets[name][0], offsets[name][1]))
            for name in family_names
        }
        if any(cell >= dim for cells in family_cells.values() for cell in cells):
            raise ValueError(
                f"response_offsets exceed the response dimension {dim}: {family_cells}"
            )
    all_cells = [cell for cells in family_cells.values() for cell in cells]

    mse_by_cell: dict[int, float] = {}
    baseline_by_cell: dict[int, float] = {}
    for j in all_cells:
        train_rows = m_train[:, j]
        fitted_n = int(train_rows.sum())
        if fitted_n < min_obs:
            continue
        test_rows = m_test[:, j]
        tested_n = int(test_rows.sum())
        if tested_n == 0:
            continue
        x_raw = x_train[train_rows]
        if center:
            # The intercept must be built from the rows that actually fit this
            # cell; centering on all training rows would mix in unobserved
            # labels and stop being equivalent to an intercept ridge.
            x_mean = x_raw.mean(dim=0, keepdim=True)
            x_j = x_raw - x_mean
            x_target = x_test[test_rows] - x_mean
        else:
            x_j = x_raw
            x_target = x_test[test_rows]
        y_j = y_train[train_rows, j].unsqueeze(1)
        y_mean = y_j.mean()
        y_target = y_test[test_rows, j].unsqueeze(1)
        baseline_by_cell[j] = float(((y_mean.expand_as(y_target) - y_target) ** 2).mean())
        design = x_j.T @ x_j + ridge * torch.eye(x_j.shape[1], dtype=x_j.dtype, device=x_j.device)
        try:
            if center:
                beta = torch.linalg.solve(design, x_j.T @ (y_j - y_mean))
            else:
                beta = torch.linalg.solve(design, x_j.T @ y_j)
        except RuntimeError:
            continue
        pred = x_target @ beta
        if center:
            pred = pred + y_mean
        mse = float(((pred - y_target) ** 2).mean())
        mse_by_cell[j] = mse

    def _cell_mse(cells: list[int], values_by_cell: dict[int, float]) -> float | None:
        values = [values_by_cell[cell] for cell in cells if cell in values_by_cell]
        if not values:
            return None
        return round(float(np.sqrt(np.mean(values))), 6)

    per_family: dict[str, dict[str, object]] = {}
    for name, cells in family_cells.items():
        per_family[name] = {
            "rmse": _cell_mse(cells, mse_by_cell),
            "mean_baseline_rmse": _cell_mse(cells, baseline_by_cell),
            "fitted_cells": sum(1 for cell in cells if cell in mse_by_cell),
            "total_cells": len(cells),
        }
    overall = _cell_mse(all_cells, mse_by_cell)
    return {
        "rmse_overall": overall,
        "mean_baseline_overall": _cell_mse(all_cells, baseline_by_cell),
        "fitted_cells": len(mse_by_cell),
        "total_cells": len(all_cells),
        "centered": bool(center),
        "ridge": float(ridge),
        "per_family": per_family,
    }


def select_probe_ridge(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    m_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    m_val: torch.Tensor,
    candidates: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    min_obs: int = 16,
    families: list[str] | None = None,
    response_offsets: dict[str, tuple[int, int, list[str]]] | None = None,
) -> dict[str, object]:
    """Pick the ridge strength with the lowest validation probe RMSE.

    The comparison is per representation (raw features or an embedding):
    callers should select a separate ridge for each so that the probe
    regularisation is not silently favouring one input scale.
    """
    scores: dict[str, float] = {}
    for candidate in candidates:
        result = probe_rmse(
            x_train,
            y_train,
            m_train,
            x_val,
            y_val,
            m_val,
            ridge=float(candidate),
            min_obs=min_obs,
            families=families,
            response_offsets=response_offsets,
        )
        rmse = result["rmse_overall"]
        if isinstance(rmse, (int, float)):
            scores[str(float(candidate))] = float(rmse)
    if not scores:
        return {"selected_ridge": None, "validation_rmse": {}}
    selected = min(scores, key=scores.__getitem__)
    return {"selected_ridge": float(selected), "validation_rmse": scores}


# ─────────────────────────────────────────────────────────────────────────────
# Level 1 to Level 4 Evaluation Hierarchy
# ─────────────────────────────────────────────────────────────────────────────


def trajectory_prediction_metrics(
    predictions: dict[str, torch.Tensor],
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    response_weight: torch.Tensor | None = None,
    delta: float = 1.0,
    response_offsets: dict[str, tuple[int, int, list[str]]] | None = None,
) -> dict[str, object]:
    """Level 1: Continuous causal trajectory prediction and concordance metrics."""
    results: dict[str, object] = {}
    overall_huber: list[float] = []
    overall_mse: list[float] = []
    overall_mae: list[float] = []
    overall_pearson: list[float] = []
    dir_matches: list[int] = []
    dir_totals: list[int] = []

    if response_offsets is not None:
        offsets_map = response_offsets
    elif responses.shape[1] == 27:
        offsets_map = RESPONSE_OFFSETS
    elif len(predictions) == 1:
        fam = next(iter(predictions.keys()))
        offsets_map = {fam: (0, responses.shape[1], [])}
    else:
        offsets_map = {}
        curr = 0
        for fam, p in predictions.items():
            dim = p.shape[1]
            offsets_map[fam] = (curr, curr + dim, [])
            curr += dim

    for family, (start, end, *_) in offsets_map.items():
        pred = _prediction_slice(
            predictions, family, start, end, responses.shape[1]
        )
        if pred is None:
            continue
        target = responses[:, start:end]
        mask = response_mask[:, start:end]
        weight_slice = response_weight[:, start:end] if response_weight is not None else None

        if mask.sum() == 0:
            continue

        pred_valid = pred[mask]
        target_valid = target[mask]
        diff = pred_valid - target_valid
        abs_diff = torch.abs(diff)

        huber = torch.where(
            abs_diff <= delta,
            0.5 * (diff**2),
            delta * (abs_diff - 0.5 * delta),
        )

        if response_weight is not None and weight_slice is not None:
            w = torch.nan_to_num(weight_slice, nan=0.0).clamp_min(0.0)[mask]
            w_sum = w.sum().item()
            if w_sum > 0:
                huber_val = float((huber * w).sum().item() / w_sum)
                mse_val = float(((diff**2) * w).sum().item() / w_sum)
                mae_val = float((abs_diff * w).sum().item() / w_sum)
            else:
                huber_val = float(huber.mean().item())
                mse_val = float((diff**2).mean().item())
                mae_val = float(abs_diff.mean().item())
        else:
            huber_val = float(huber.mean().item())
            mse_val = float((diff**2).mean().item())
            mae_val = float(abs_diff.mean().item())

        sign_pred = torch.sign(pred_valid)
        sign_target = torch.sign(target_valid)
        sig_mask = abs_diff > 1e-4
        match = (sign_pred[sig_mask] == sign_target[sig_mask]).sum().item()
        total_sig = sig_mask.sum().item()

        if len(pred_valid) >= 2 and float(torch.std(pred_valid)) > 1e-6 and float(torch.std(target_valid)) > 1e-6:
            vx = pred_valid - pred_valid.mean()
            vy = target_valid - target_valid.mean()
            denom = float(torch.norm(vx, 2) * torch.norm(vy, 2))
            pearson_val = float((vx * vy).sum().item() / denom) if denom > 1e-12 else 0.0
        else:
            pearson_val = 0.0

        overall_huber.append(huber_val)
        overall_mse.append(mse_val)
        overall_mae.append(mae_val)
        overall_pearson.append(pearson_val)
        dir_matches.append(int(match))
        dir_totals.append(int(total_sig))

        results[family] = {
            "huber_loss": huber_val,
            "mse": mse_val,
            "rmse": float(np.sqrt(mse_val)),
            "mae": mae_val,
            "pearson_r": pearson_val,
            "directional_accuracy": float(match / total_sig) if total_sig > 0 else 1.0,
            "evaluated_cells": int(mask.sum().item()),
        }

    total_dir_matches = sum(dir_matches)
    total_dir_cells = sum(dir_totals)
    results["overall"] = {
        "mean_huber_loss": float(np.mean(overall_huber)) if overall_huber else 0.0,
        "mean_mse": float(np.mean(overall_mse)) if overall_mse else 0.0,
        "mean_rmse": float(np.sqrt(np.mean(overall_mse))) if overall_mse else 0.0,
        "mean_mae": float(np.mean(overall_mae)) if overall_mae else 0.0,
        "mean_pearson_r": float(np.mean(overall_pearson)) if overall_pearson else 0.0,
        "directional_accuracy": (
            float(total_dir_matches / total_dir_cells) if total_dir_cells > 0 else 1.0
        ),
    }
    return results


def planning_semantic_classification_metrics(
    predictions: dict[str, torch.Tensor],
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    threshold: float = 0.05,
    response_offsets: dict[str, tuple[int, int, list[str]]] | None = None,
) -> dict[str, object]:
    """Level 2: Planning Semantic Classification (Activation, Displacement, No Response).

    Discretizes continuous response trajectories into 3 canonical urban planning regimes:
    0: Activation (+)      - mean response > threshold
    1: No Response (0)     - |mean response| <= threshold
    2: Displacement (-)    - mean response < -threshold
    """
    results: dict[str, object] = {}

    if response_offsets is not None:
        offsets_map = response_offsets
    elif responses.shape[1] == RESPONSE_DIM:
        offsets_map = RESPONSE_OFFSETS
    elif len(predictions) == 1:
        family = next(iter(predictions))
        offsets_map = {family: (0, responses.shape[1], [])}
    else:
        offsets_map = {}
        cursor = 0
        for family, pred in predictions.items():
            offsets_map[family] = (cursor, cursor + pred.shape[1], [])
            cursor += pred.shape[1]

    for family, (start, end, *_) in offsets_map.items():
        pred = _prediction_slice(
            predictions, family, start, end, responses.shape[1]
        )
        if pred is None:
            continue
        target = responses[:, start:end]
        mask = response_mask[:, start:end]
        if pred.shape[1] != target.shape[1]:
            raise ValueError(
                f"Prediction for {family} has {pred.shape[1]} cells but the response "
                f"target has {target.shape[1]}"
            )

        valid_units = mask.sum(dim=1) > 0
        if not valid_units.any():
            continue

        # Masked response cells are stored as NaN.  Multiply-by-mask leaves
        # those NaNs in the sum, so zero them explicitly before averaging.
        zero = torch.zeros_like(target)
        target = torch.where(mask, target, zero)
        pred = torch.where(mask, pred, zero)
        pred_mean = pred.sum(dim=1)[valid_units] / mask.sum(dim=1)[valid_units].clamp_min(1)
        target_mean = target.sum(dim=1)[valid_units] / mask.sum(dim=1)[valid_units].clamp_min(1)

        y_true = np.where(
            target_mean.cpu().numpy() > threshold,
            0,
            np.where(target_mean.cpu().numpy() < -threshold, 2, 1),
        )
        y_pred = np.where(
            pred_mean.cpu().numpy() > threshold,
            0,
            np.where(pred_mean.cpu().numpy() < -threshold, 2, 1),
        )

        classes = [0, 1, 2]
        cm = np.zeros((3, 3), dtype=int)
        for t, p in zip(y_true, y_pred, strict=False):
            cm[t, p] += 1

        accuracy = float((y_true == y_pred).mean())

        # Per-class F1 & Macro-F1
        f1_list = []
        for c in classes:
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            f1_list.append(f1)

        macro_f1 = float(np.mean(f1_list))

        # Cohen's Kappa
        n_samples = len(y_true)
        p_o = accuracy
        p_e = float(sum((cm[c, :].sum() * cm[:, c].sum()) for c in classes) / (n_samples**2)) if n_samples > 0 else 0.0
        kappa = float((p_o - p_e) / (1.0 - p_e)) if (1.0 - p_e) > 1e-12 else 0.0

        results[family] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "cohens_kappa": kappa,
            "class_f1": {
                "positive": float(f1_list[0]),
                "neutral_or_uncertain": float(f1_list[1]),
                "negative": float(f1_list[2]),
                "activation": float(f1_list[0]),
                "no_response": float(f1_list[1]),
                "displacement": float(f1_list[2]),
            },
            "confusion_matrix": cm.tolist(),
            "sample_count": int(n_samples),
        }

    return results


def latent_manifold_umap(
    embeddings: np.ndarray,
    treatment_orders: list[int],
    city_keys: list[str],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> dict[str, object]:
    """Level 4: Dimensionality reduction for latent space manifold visualization."""
    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,
        )
        proj_2d = reducer.fit_transform(embeddings)
        method = "umap"
    except ImportError:
        # Fallback to pure numpy SVD-based PCA without external dependencies
        x_centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
        proj_2d = x_centered @ vt[:2].T
        method = "pca_fallback"

    return {
        "method": method,
        "treatment_orders": treatment_orders,
        "city_keys": city_keys,
        "coords_2d": proj_2d.tolist(),
    }
