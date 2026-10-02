"""Minimal controls: can raw pre-treatment features predict the frozen labels?

City-level K-fold cross-validation with an inner city-validation split for
hyperparameter selection; outer folds are used for evaluation only.  The
predictors are:

1. raw-feature retrieval (nearest-neighbour ``nn_corr`` on the model input
   features, i.e. the raw-feature baseline);
2. ridge regression with intercept (linear control);
3. k-NN regression (nonlinear control);

plus the train-mean prediction baseline, so a predictor that does not beat
predicting the training mean cannot be read as evidence of predictability.
The primary outcome family is detected from the labels; for POI the four
outcomes are also reported separately.

The current representation model is intentionally not retrained here: with a
different fold assignment its existing checkpoint would have seen many test
cities, so a fold-wise neural comparison only makes sense after this ceiling
test shows that the features carry signal at all.

Run on the server from the repository root::

    conda run -n mit python scripts/analysis/representation_controls.py \
        --dataset data/active/model_inputs/viirs-mg-design \
        --output-directory outputs/representation_controls/viirs
    conda run -n mit python scripts/analysis/representation_controls.py \
        --dataset data/active/model_inputs/poi-mg-design \
        --output-directory outputs/representation_controls/poi

The script only reads the built dataset; it never retrains or rewrites it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.representation.dataset import (  # noqa: E402
    RESPONSE_OFFSETS,
    RepresentationDataset,
)
from urban_intervention.representation.evaluation import retrieval_metrics  # noqa: E402

RIDGE_CANDIDATES = (0.01, 0.1, 1.0, 10.0, 100.0)
KNN_CANDIDATES = (5, 15, 50, 100)


class _Arrays(TypedDict):
    features: np.ndarray
    responses: np.ndarray
    masks: np.ndarray
    cities: list[str]
    primary_family: str


class _FoldResult(TypedDict):
    fold: int
    fit_units: int
    val_units: int
    test_units: int
    ridge: dict[int, dict[str, float]]
    knn: dict[int, dict[str, float]]
    retrieval: float | None
    retrieval_per_outcome: dict[str, float | None]


def load_arrays(dataset_dir: Path) -> _Arrays:
    ds = RepresentationDataset(dataset_dir, only_training_mask=True, load_images=False)
    features, responses, masks, cities = [], [], [], []
    for index in range(len(ds)):
        sample = ds[index]
        features.append(sample.feature_vector.numpy())
        responses.append(sample.response_vector.numpy())
        masks.append(sample.response_mask.numpy())
        cities.append(sample.city_key)
    features_arr = np.stack(features).astype(np.float64)
    responses_arr = np.stack(responses).astype(np.float64)
    masks_arr = np.stack(masks).astype(bool)
    primary = None
    for family, (start, end, _) in RESPONSE_OFFSETS.items():
        if masks_arr[:, start:end].any():
            primary = family
            break
    if primary is None:
        raise ValueError("Dataset has no observed response labels")
    return {
        "features": features_arr,
        "responses": responses_arr,
        "masks": masks_arr,
        "cities": list(cities),
        "primary_family": primary,
    }


def outcome_groups(family: str) -> dict[str, list[int]]:
    start, _, columns = RESPONSE_OFFSETS[family]
    groups: dict[str, list[int]] = {}
    for offset, name in enumerate(columns):
        groups.setdefault(name.split("__t")[0], []).append(start + offset)
    return groups


def make_city_folds(cities: list[str], n_folds: int, seed: int) -> dict[str, int]:
    """Greedy balanced fold assignment over cities (deterministic)."""
    counts: dict[str, int] = {}
    for city in cities:
        counts[city] = counts.get(city, 0) + 1
    rng = np.random.default_rng(seed)
    order = sorted(counts, key=lambda c: (-counts[c], float(rng.random()), c))
    totals = [0] * n_folds
    assignment: dict[str, int] = {}
    for city in order:
        fold = int(np.argmin(totals))
        assignment[city] = fold
        totals[fold] += counts[city]
    return assignment


def inner_city_split(
    train_cities: list[str], seed: int, ratio: float = 0.2
) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    shuffled = [str(value) for value in rng.permutation(sorted(train_cities))]
    n_val = max(1, int(round(len(shuffled) * ratio)))
    return shuffled[n_val:], shuffled[:n_val]


def _ridge_fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_target: np.ndarray, ridge: float
) -> np.ndarray:
    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean()
    x_centered = x_train - x_mean
    design = x_centered.T @ x_centered + ridge * np.eye(x_centered.shape[1])
    beta = np.linalg.solve(design, x_centered.T @ (y_train - y_mean))
    return (x_target - x_mean) @ beta + y_mean


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def _knn_predict(
    distance_matrix: np.ndarray, y_fit: np.ndarray, k: int
) -> np.ndarray:
    k_eff = min(k, distance_matrix.shape[1])
    neighbours = np.argsort(distance_matrix, axis=1)[:, :k_eff]
    return y_fit[neighbours].mean(axis=1)


def fold_controls(
    arrays: _Arrays,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    min_obs: int,
) -> dict[str, dict[int, dict[str, float]]]:
    features = arrays["features"]
    responses = arrays["responses"]
    masks = arrays["masks"]
    distance_val_fit = cdist(features[val_idx], features[fit_idx])
    refit_idx = np.concatenate([fit_idx, val_idx])
    distance_test_refit = cdist(features[test_idx], features[refit_idx])
    results: dict[str, dict[int, dict[str, float]]] = {"ridge": {}, "knn": {}, "baseline": {}}
    for cell in range(responses.shape[1]):
        fit_obs = masks[fit_idx, cell]
        val_obs = masks[val_idx, cell]
        test_obs = masks[test_idx, cell]
        n_fit = int(fit_obs.sum())
        n_val = int(val_obs.sum())
        n_test = int(test_obs.sum())
        if n_fit < min_obs or n_test == 0:
            continue
        y_fit = responses[fit_idx, cell][fit_obs]
        y_val = responses[val_idx, cell][val_obs]
        y_test = responses[test_idx, cell][test_obs]
        baseline = _rmse(np.full(n_test, y_fit.mean()), y_test)
        results["baseline"][cell] = {"rmse": baseline}
        ridge_param: float | None = None
        if n_val >= max(min_obs // 2, 4):
            best_ridge: tuple[float, float] | None = None
            for ridge in RIDGE_CANDIDATES:
                rmse = _rmse(
                    _ridge_fit_predict(
                        features[fit_idx][fit_obs], y_fit, features[val_idx][val_obs], ridge
                    ),
                    y_val,
                )
                if best_ridge is None or rmse < best_ridge[1]:
                    best_ridge = (ridge, rmse)
            ridge_param = float(best_ridge[0]) if best_ridge is not None else None
        if ridge_param is not None:
            refit_obs = masks[refit_idx, cell]
            prediction = _ridge_fit_predict(
                features[refit_idx][refit_obs],
                responses[refit_idx, cell][refit_obs],
                features[test_idx][test_obs],
                ridge_param,
            )
            results["ridge"][cell] = {
                "rmse": _rmse(prediction, y_test),
                "baseline": baseline,
                "param": ridge_param,
            }
        knn_param: float | None = None
        if n_val >= max(min_obs // 2, 4):
            best_knn: tuple[float, float] | None = None
            for k in KNN_CANDIDATES:
                prediction = _knn_predict(
                    distance_val_fit[:, fit_obs], y_fit, k
                )
                rmse = _rmse(prediction[val_obs], y_val)
                if best_knn is None or rmse < best_knn[1]:
                    best_knn = (float(k), rmse)
            knn_param = best_knn[0] if best_knn is not None else None
        if knn_param is not None:
            refit_obs = masks[refit_idx, cell]
            prediction = _knn_predict(
                distance_test_refit[:, refit_obs],
                responses[refit_idx, cell][refit_obs],
                int(knn_param),
            )
            results["knn"][cell] = {
                "rmse": _rmse(prediction, y_test),
                "baseline": baseline,
                "param": knn_param,
            }
    return results


def fold_retrieval(
    arrays: _Arrays, test_idx: np.ndarray, columns: list[int], k: int
) -> float | None:
    import torch

    features = torch.from_numpy(arrays["features"][test_idx]).float()
    responses = torch.from_numpy(arrays["responses"][test_idx][:, columns]).float()
    masks = torch.from_numpy(arrays["masks"][test_idx][:, columns])
    if masks.sum() < 2:
        return None
    block = retrieval_metrics(features, responses, masks, k=k)["overall"]
    value = block.get("nn_corr@k")
    return float(value) if isinstance(value, (int, float)) else None


def aggregate(
    fold_results: list[_FoldResult], cells: list[int], method: str
) -> dict[str, object]:
    rmses, baselines, params = [], [], []
    for fold in fold_results:
        method_map = fold["ridge"] if method == "ridge" else fold["knn"]
        for cell in cells:
            if cell in method_map:
                rmses.append(method_map[cell]["rmse"])
                baselines.append(method_map[cell]["baseline"])
                params.append(method_map[cell]["param"])
    if not rmses:
        return {"n": 0}
    return {
        "n": len(rmses),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_sd": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        "baseline_mean": float(np.mean(baselines)),
        "improvement_vs_baseline": float(np.mean(np.subtract(baselines, rmses))),
        "selected_params": sorted({float(value) for value in params}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-obs", type=int, default=16)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    arrays = load_arrays(args.dataset)
    cities = list(arrays["cities"])
    primary = str(arrays["primary_family"])
    groups = outcome_groups(primary)
    assignment = make_city_folds(cities, args.folds, args.seed)
    fold_results: list[_FoldResult] = []
    for fold in range(args.folds):
        test_cities = [city for city in set(cities) if assignment[city] == fold]
        train_cities = [city for city in set(cities) if assignment[city] != fold]
        inner_train, inner_val = inner_city_split(train_cities, args.seed + fold)
        train_set, val_set, test_set = set(inner_train), set(inner_val), set(test_cities)
        fit_idx = np.array([i for i, city in enumerate(cities) if city in train_set])
        val_idx = np.array([i for i, city in enumerate(cities) if city in val_set])
        test_idx = np.array([i for i, city in enumerate(cities) if city in test_set])
        controls = fold_controls(arrays, fit_idx, val_idx, test_idx, args.min_obs)
        all_cells = [cell for cells in groups.values() for cell in cells]
        retrieval = fold_retrieval(arrays, test_idx, all_cells, args.k)
        retrieval_per_outcome = {
            outcome: fold_retrieval(arrays, test_idx, cells, args.k)
            for outcome, cells in groups.items()
        }
        fold_results.append(
            {
                "fold": fold,
                "fit_units": int(len(fit_idx)),
                "val_units": int(len(val_idx)),
                "test_units": int(len(test_idx)),
                "ridge": controls["ridge"],
                "knn": controls["knn"],
                "retrieval": retrieval,
                "retrieval_per_outcome": retrieval_per_outcome,
            }
        )
        print(
            f"fold {fold}: fit {len(fit_idx)} / val {len(val_idx)} / test {len(test_idx)} | "
            f"retrieval {retrieval if retrieval is not None else float('nan'):.4f} | "
            f"cities {sorted(test_set)}"
        )
    all_cells = [cell for cells in groups.values() for cell in cells]
    retrieval_folds = [float(v) for v in (f["retrieval"] for f in fold_results) if v is not None]
    overall: dict[str, object] = {
        "retrieval": {
            "mean": float(np.mean(retrieval_folds)) if retrieval_folds else None,
            "n": len(retrieval_folds),
        },
    }
    for method in ("ridge", "knn"):
        overall[method] = aggregate(fold_results, all_cells, method)
    per_outcome: dict[str, object] = {}
    for outcome, cells in groups.items():
        entry: dict[str, object] = {}
        for method in ("ridge", "knn"):
            entry[method] = aggregate(fold_results, cells, method)
        retrieval_values = [
            float(v)
            for v in (f["retrieval_per_outcome"][outcome] for f in fold_results)
            if v is not None
        ]
        entry["retrieval"] = {
            "mean": float(np.mean(retrieval_values)) if retrieval_values else None,
            "n": len(retrieval_values),
        }
        per_outcome[outcome] = entry
    summary: dict[str, object] = {
        "primary_family": primary,
        "overall": overall,
        "per_outcome": per_outcome,
    }
    report = {
        "dataset": str(args.dataset),
        "primary_family": primary,
        "folds": args.folds,
        "seed": args.seed,
        "fold_sizes": [
            {"fold": f["fold"], "fit": f["fit_units"], "val": f["val_units"], "test": f["test_units"]}
            for f in fold_results
        ],
        "summary": summary,
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    out_path = args.output_directory / "controls.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
