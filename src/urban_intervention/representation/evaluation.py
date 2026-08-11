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

import numpy as np
import torch

from .dataset import RESPONSE_DIM, RESPONSE_OFFSETS


def response_similarity(
    responses: torch.Tensor,
    masks: torch.Tensor,
    families: list[str] | None = None,
) -> torch.Tensor:
    """Vectorized pairwise correlation over commonly-observed response cells.

    With ``families=None`` and ``responses`` of the full response dimension,
    the per-family correlations are averaged exactly like the training loss.
    Passing a subset of family names restricts the average to those segments.
    """
    similarity, _ = response_similarity_with_validity(responses, masks, families)
    return similarity


def response_similarity_with_validity(
    responses: torch.Tensor,
    masks: torch.Tensor,
    families: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return response correlation and whether each pair is comparable.

    A pair is comparable when at least one requested outcome family has two
    commonly observed, non-constant response cells.  Correlations are averaged
    over comparable families only; missing families are not interpreted as
    zero response similarity.
    """
    return response_similarity_between(responses, masks, responses, masks, families)


def response_similarity_between(
    responses_a: torch.Tensor,
    masks_a: torch.Tensor,
    responses_b: torch.Tensor,
    masks_b: torch.Tensor,
    families: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rectangular response similarity without constructing a square joint pool."""
    n_a, dim = responses_a.shape
    n_b = responses_b.shape[0]
    if responses_b.shape[1] != dim:
        raise ValueError("Response pools must have the same response dimension")
    if masks_a.shape != responses_a.shape or masks_b.shape != responses_b.shape:
        raise ValueError("Each response mask must match its response tensor")
    if responses_a.device != responses_b.device:
        raise ValueError("Response pools must be on the same device")
    sim_sum = responses_a.new_zeros(n_a, n_b)
    valid_count = torch.zeros(n_a, n_b, dtype=torch.int32, device=responses_a.device)

    if dim == RESPONSE_DIM:
        family_names = list(families) if families is not None else list(RESPONSE_OFFSETS)
        for name in family_names:
            if name not in RESPONSE_OFFSETS:
                raise ValueError(f"Unknown outcome family: {name}")
        segments = [(RESPONSE_OFFSETS[name][0], RESPONSE_OFFSETS[name][1]) for name in family_names]
    else:
        segments = [(0, dim)]

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
    if n < 2:
        return {"nn_corr@k": 0.0, "baseline_corr": 0.0, "n_units": n, "ratio": None}
    nn = nn_corr_at_k(emb_cos, sim_resp, k, valid_pairs)
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=emb_cos.device)
    valid_off_diagonal = valid_pairs & off_diagonal
    row_count = valid_off_diagonal.sum(dim=1)
    valid_rows = row_count > 0
    row_sum = (sim_resp * valid_off_diagonal.to(sim_resp.dtype)).sum(dim=1)
    baseline = (
        float((row_sum[valid_rows] / row_count[valid_rows]).mean())
        if valid_rows.any()
        else 0.0
    )
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
) -> dict[str, dict[str, object]]:
    """Retrieval quality overall and per outcome family.

    Neighbours are always chosen from embedding cosine similarity; the score is
    the response similarity with those neighbours, evaluated on all families
    jointly (``overall``) and on each family separately.
    """
    emb_cos = cosine_similarity(embeddings)
    overall_sim, overall_valid = response_similarity_with_validity(responses, masks)
    result: dict[str, dict[str, object]] = {
        "overall": _metric_block(emb_cos, overall_sim, overall_valid, k),
    }
    for family in RESPONSE_OFFSETS:
        family_sim, family_valid = response_similarity_with_validity(
            responses, masks, families=[family]
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

    observed_sim, observed_valid = response_similarity_with_validity(responses, masks, families)
    observed = nn_corr_at_k(emb_cos, observed_sim, k, observed_valid)
    nulls = np.empty(n_perm, dtype=np.float64)
    for b in range(n_perm):
        idx = rng.permutation(n)
        null_sim, null_valid = response_similarity_with_validity(
            responses[idx], masks[idx], families
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
) -> dict[str, object]:
    """Linear-probe transfer metric: per-response-cell ridge, fitted on train.

    For every response cell with at least ``min_obs`` train observations, a
    ridge regressor is fit from the frozen input features (embeddings or raw
    features) to that cell, then applied to the target pool. The reported RMSE
    is over observed test cells only; cells with insufficient training support
    are excluded and counted in ``fitted_cells``.
    """
    n_train, dim = y_train.shape
    if x_train.shape[0] != n_train or y_test.shape[1] != dim:
        raise ValueError("Probe inputs have inconsistent shapes")
    if dim == RESPONSE_DIM:
        family_names = list(families) if families is not None else list(RESPONSE_OFFSETS)
        for name in family_names:
            if name not in RESPONSE_OFFSETS:
                raise ValueError(f"Unknown outcome family: {name}")
        family_cells = {
            name: list(range(RESPONSE_OFFSETS[name][0], RESPONSE_OFFSETS[name][1]))
            for name in family_names
        }
    else:
        family_cells = {"raw": list(range(dim))}
    all_cells = [cell for cells in family_cells.values() for cell in cells]

    mse_by_cell: dict[int, float] = {}
    for j in all_cells:
        train_rows = m_train[:, j]
        fitted_n = int(train_rows.sum())
        if fitted_n < min_obs:
            continue
        test_rows = m_test[:, j]
        tested_n = int(test_rows.sum())
        if tested_n == 0:
            continue
        x_j = x_train[train_rows]
        y_j = y_train[train_rows, j].unsqueeze(1)
        design = x_j.T @ x_j + ridge * torch.eye(x_j.shape[1], dtype=x_j.dtype, device=x_j.device)
        try:
            beta = torch.linalg.solve(design, x_j.T @ y_j)
        except RuntimeError:
            continue
        pred = x_test[test_rows] @ beta
        mse = float(((pred - y_test[test_rows, j].unsqueeze(1)) ** 2).mean())
        mse_by_cell[j] = mse

    def _cell_mse(cells: list[int]) -> float | None:
        values = [mse_by_cell[cell] for cell in cells if cell in mse_by_cell]
        if not values:
            return None
        return round(float(np.sqrt(np.mean(values))), 6)

    per_family: dict[str, dict[str, object]] = {}
    for name, cells in family_cells.items():
        per_family[name] = {
            "rmse": _cell_mse(cells),
            "fitted_cells": sum(1 for cell in cells if cell in mse_by_cell),
            "total_cells": len(cells),
        }
    overall = _cell_mse(all_cells)
    return {
        "rmse_overall": overall,
        "fitted_cells": len(mse_by_cell),
        "total_cells": len(all_cells),
        "per_family": per_family,
    }
