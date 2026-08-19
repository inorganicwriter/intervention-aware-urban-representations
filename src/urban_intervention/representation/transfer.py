"""Cross-city transfer evaluation for the response representation.

Two complementary pieces of evidence that the representation generalises to
cities it never saw during training:

- ``per_city_retrieval``: retrieval quality (``nn_corr@k``) computed separately
  for every city in a held-out pool.  A transferable representation keeps its
  quality in every unseen city instead of concentrating it in one.
- ``few_shot_probe``: a per-cell ridge probe fitted on ``shot`` units of one
  target city and evaluated on the remaining units of that same city. The
  report keeps cities separate so target-city information is never mixed.
"""

from __future__ import annotations

import numpy as np
import torch

from .dataset import RESPONSE_DIM, RESPONSE_OFFSETS
from .evaluation import probe_rmse, retrieval_metrics


def per_city_retrieval(
    embeddings: torch.Tensor,
    responses: torch.Tensor,
    masks: torch.Tensor,
    city_keys: list[str],
    k: int = 5,
) -> dict[str, dict[str, dict[str, object]]]:
    """Retrieval metrics per city; cities with fewer than 2 units are skipped."""
    if len(city_keys) != embeddings.shape[0]:
        raise ValueError("city_keys must be aligned with the pool rows")
    result: dict[str, dict[str, dict[str, object]]] = {}
    for city in sorted(set(city_keys)):
        keep = [index for index, value in enumerate(city_keys) if value == city]
        if len(keep) < 2:
            continue
        index = torch.tensor(keep, dtype=torch.long)
        result[city] = retrieval_metrics(
            embeddings[index], responses[index], masks[index], k=k
        )
    return result


def few_shot_probe(
    target_embeddings: torch.Tensor,
    target_responses: torch.Tensor,
    target_masks: torch.Tensor,
    shot_sizes: tuple[int, ...] = (4, 8, 16, 32),
    ridge: float = 1.0,
    min_obs: int = 2,
    seed: int = 7,
    n_seeds: int = 3,
) -> list[dict[str, object]]:
    """Probe RMSE on the target pool as the probe training set grows.

    For each shot size, ``shot`` target units are drawn per seed (seeds are
    ``seed + s * 7919``), a per-cell ridge probe is fitted on their observed
    response cells, and RMSE is reported over the remaining target units.
    With ``n_seeds > 1`` the RMSE is averaged across seeds and the standard
    deviation is reported, so the few-shot curve carries an uncertainty
    estimate.  Only shot sizes strictly smaller than the pool are reported.
    """
    n = target_embeddings.shape[0]
    if n < 2:
        return []
    results: list[dict[str, object]] = []
    for shot in shot_sizes:
        if shot <= 0 or shot >= n:
            continue
        rmse_values: list[float] = []
        first_probe: dict[str, object] | None = None
        for seed_index in range(max(1, n_seeds)):
            rng = np.random.RandomState(seed + seed_index * 7919)
            probe_idx = rng.choice(n, shot, replace=False)
            probe_units = torch.from_numpy(np.asarray(probe_idx, dtype=np.int64))
            probe_set = set(int(value) for value in probe_units.tolist())
            rest = [index for index in range(n) if index not in probe_set]
            if not rest:
                continue
            rest_idx = torch.tensor(rest, dtype=torch.long)
            probe = probe_rmse(
                target_embeddings[probe_units],
                target_responses[probe_units],
                target_masks[probe_units],
                target_embeddings[rest_idx],
                target_responses[rest_idx],
                target_masks[rest_idx],
                ridge=ridge,
                min_obs=min_obs,
            )
            if first_probe is None:
                first_probe = probe
            rmse_value = probe.get("rmse_overall")
            if isinstance(rmse_value, (int, float)):
                rmse_values.append(float(rmse_value))
        if not rmse_values:
            continue
        row: dict[str, object] = {
            "shot": int(shot),
            "probe_units": int(shot),
            "eval_units": n - shot,
            "rmse_overall": round(float(np.mean(rmse_values)), 6),
            "rmse_overall_std": round(float(np.std(rmse_values, ddof=1)), 6)
            if len(rmse_values) > 1
            else 0.0,
            "seeds": len(rmse_values),
        }
        if first_probe is not None:
            row["fitted_cells"] = first_probe.get("fitted_cells")
            row["total_cells"] = first_probe.get("total_cells")
        results.append(row)
    return results


def predictive_auc(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    m_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    m_test: torch.Tensor,
    ridge: float = 1.0,
    min_obs: int = 16,
) -> dict[str, object]:
    """Per-cell response-direction classification AUC.

    For every response cell with at least ``min_obs`` train observations and
    both response signs present, a ridge classifier is fit on the train pool to
    predict ``sign(cell)`` and evaluated as AUC on the target pool (rank-based
    Mann-Whitney statistic).  Reported overall and per outcome family, exactly
    like ``probe_rmse``, so embeddings and raw features can be compared cell
    for cell.
    """
    n_train, dim = y_train.shape
    if x_train.shape[0] != n_train or y_test.shape[1] != dim:
        raise ValueError("Predictive inputs have inconsistent shapes")
    if dim == RESPONSE_DIM:
        family_cells = {
            name: list(range(RESPONSE_OFFSETS[name][0], RESPONSE_OFFSETS[name][1]))
            for name in RESPONSE_OFFSETS
        }
    else:
        family_cells = {"raw": list(range(dim))}
    all_cells = [cell for cells in family_cells.values() for cell in cells]

    auc_by_cell: dict[int, float] = {}
    for j in all_cells:
        train_rows = m_train[:, j]
        fitted_n = int(train_rows.sum())
        if fitted_n < min_obs:
            continue
        test_rows = m_test[:, j]
        tested_n = int(test_rows.sum())
        if tested_n < 2:
            continue
        y_j = y_train[train_rows, j]
        labels = (y_j > 0).to(torch.int8)
        if int(labels.sum()) == 0 or int((1 - labels).sum()) == 0:
            continue
        x_j = x_train[train_rows]
        design = x_j.T @ x_j + ridge * torch.eye(x_j.shape[1], dtype=x_j.dtype, device=x_j.device)
        try:
            beta = torch.linalg.solve(design, x_j.T @ labels.to(x_j.dtype))
        except RuntimeError:
            continue
        scores = x_test[test_rows] @ beta
        auc = _rank_auc(scores, (y_test[test_rows, j] > 0).to(torch.int8))
        if auc is not None:
            auc_by_cell[j] = auc

    def _cell_auc(cells: list[int]) -> float | None:
        values = [auc_by_cell[cell] for cell in cells if cell in auc_by_cell]
        if not values:
            return None
        return round(float(np.mean(values)), 6)

    per_family: dict[str, dict[str, object]] = {}
    for name, cells in family_cells.items():
        per_family[name] = {
            "auc": _cell_auc(cells),
            "fitted_cells": sum(1 for cell in cells if cell in auc_by_cell),
            "total_cells": len(cells),
        }
    return {
        "auc_overall": _cell_auc(all_cells),
        "fitted_cells": len(auc_by_cell),
        "total_cells": len(all_cells),
        "per_family": per_family,
    }


def _rank_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    """Rank-based AUC (Mann-Whitney U) for binary labels in {0, 1}."""
    n = scores.shape[0]
    if n < 2:
        return None
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ends = counts.cumsum(dim=0).to(torch.float64)
    starts = ends - counts.to(torch.float64) + 1.0
    midranks = (starts + ends) / 2.0
    sorted_ranks = torch.repeat_interleave(midranks, counts)
    ranks = torch.empty(n, dtype=torch.float64, device=scores.device)
    ranks[order] = sorted_ranks.to(scores.device)
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def cross_validated_probe(
    inputs: torch.Tensor,
    responses: torch.Tensor,
    masks: torch.Tensor,
    ridge: float = 1.0,
    min_obs: int = 16,
    n_folds: int = 5,
    seed: int = 17,
) -> dict[str, object]:
    """Disjoint-fold target-pool probe used as a supervised adaptation ceiling."""
    n = inputs.shape[0]
    folds = min(max(2, n_folds), n)
    if n < 4:
        return {"note": "pool smaller than 4 units", "folds": 0, "rmse_overall": None}
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(n)
    fold_indices = np.array_split(shuffled, folds)
    values: list[float] = []
    fitted_cells: list[int] = []
    for test_index in fold_indices:
        train_index = np.setdiff1d(shuffled, test_index, assume_unique=True)
        result = probe_rmse(
            inputs[torch.from_numpy(train_index)],
            responses[torch.from_numpy(train_index)],
            masks[torch.from_numpy(train_index)],
            inputs[torch.from_numpy(test_index)],
            responses[torch.from_numpy(test_index)],
            masks[torch.from_numpy(test_index)],
            ridge=ridge,
            min_obs=min_obs,
        )
        value = result.get("rmse_overall")
        if isinstance(value, (int, float)):
            values.append(float(value))
            fitted = result.get("fitted_cells", 0)
            fitted_cells.append(int(fitted) if isinstance(fitted, (int, float)) else 0)
    if not values:
        return {"rmse_overall": None, "folds": 0, "requested_folds": folds}
    return {
        "rmse_overall": round(float(np.mean(values)), 6),
        "rmse_overall_std": round(float(np.std(values, ddof=1)), 6) if len(values) > 1 else 0.0,
        "folds": len(values),
        "requested_folds": folds,
        "mean_fitted_cells": round(float(np.mean(fitted_cells)), 3),
    }


def predictive_transfer_report(
    embeddings: torch.Tensor,
    features: torch.Tensor,
    train_embeddings: torch.Tensor,
    train_features: torch.Tensor,
    train_responses: torch.Tensor,
    train_masks: torch.Tensor,
    responses: torch.Tensor,
    masks: torch.Tensor,
    ridge: float = 1.0,
    min_obs: int = 16,
) -> dict[str, object]:
    """Response-direction AUC on a held-out pool: embeddings vs raw features.

    Both classifiers are fitted on the train pool only, so the AUC measures
    whether the representation predicts the *direction* of a new city's
    response from features alone.
    """
    return {
        "embeddings": predictive_auc(
            train_embeddings, train_responses, train_masks,
            embeddings, responses, masks, ridge=ridge, min_obs=min_obs,
        ),
        "raw_features": predictive_auc(
            train_features, train_responses, train_masks,
            features, responses, masks, ridge=ridge, min_obs=min_obs,
        ),
    }


def transfer_report(
    embeddings: torch.Tensor,
    features: torch.Tensor,
    responses: torch.Tensor,
    masks: torch.Tensor,
    city_keys: list[str],
    k: int = 5,
    shot_sizes: tuple[int, ...] = (4, 8, 16, 32),
    ridge: float = 1.0,
    probe_min_obs: int = 16,
    shot_min_obs: int = 2,
    seed: int = 7,
    n_seeds: int = 3,
) -> dict[str, object]:
    """One transfer section for a held-out split (per-city + few-shot)."""
    report: dict[str, object] = {
        "per_city": per_city_retrieval(embeddings, responses, masks, city_keys, k=k),
    }
    if embeddings.shape[0] >= 2:
        city_curves: dict[str, object] = {}
        for city in sorted(set(city_keys)):
            keep = [index for index, value in enumerate(city_keys) if value == city]
            if len(keep) < 2:
                continue
            index = torch.tensor(keep, dtype=torch.long)
            city_curves[city] = {
                "n_units": len(keep),
                "embeddings": few_shot_probe(
                    embeddings[index],
                    responses[index],
                    masks[index],
                    shot_sizes,
                    ridge=ridge,
                    min_obs=shot_min_obs,
                    seed=seed,
                    n_seeds=n_seeds,
                ),
                "raw_features": few_shot_probe(
                    features[index],
                    responses[index],
                    masks[index],
                    shot_sizes,
                    ridge=ridge,
                    min_obs=shot_min_obs,
                    seed=seed,
                    n_seeds=n_seeds,
                ),
            }
        report["few_shot_probe"] = {
            "protocol": "within_city",
            "cities": city_curves,
        }
        report["cross_validated_probe"] = {
            "protocol": "target_pool_disjoint_folds",
            "embeddings": cross_validated_probe(
                embeddings,
                responses,
                masks,
                ridge=ridge,
                min_obs=probe_min_obs,
                seed=seed,
            ),
            "raw_features": cross_validated_probe(
                features,
                responses,
                masks,
                ridge=ridge,
                min_obs=probe_min_obs,
                seed=seed,
            ),
        }
    return report
