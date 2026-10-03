"""Minimal controls: can raw pre-treatment features predict the frozen labels?

City-level K-fold cross-validation with an inner city-validation split for
hyperparameter selection; outer folds are used for evaluation only.

Preprocessing is fit inside each fold and in two stages:

* inner hyperparameter selection fits imputation/scaling on the fit cities
  only (the inner-validation cities must not influence the scaler);
* after a parameter is chosen, imputation/scaling and the model are refit on
  fit + inner-validation cities, and that same refit preprocessor produces the
  matrix used for the outer-test retrieval.

The feature matrix is read from the raw (unstandardized) lag columns with
missing values preserved, so the old training split's standardization and
zero-filling cannot leak into a new fold.

The predictors are:

1. raw-feature retrieval (shape: nearest-neighbour ``nn_corr``);
2. ridge regression with intercept (linear control);
3. k-NN regression (nonlinear control);

plus the refit train-mean prediction baseline.  Direction (sign agreement
against the majority sign of the refit labels), amplitude (RMSE) and shape
retrieval are reported separately, never collapsed into one conclusion.  The
primary outcome family is detected from the labels; for POI the four outcomes
are reported separately, and every city's fold result is kept with its true
grid count so paired increments, sample support and between-city differences
can be inspected.

This script intentionally does not evaluate existing checkpoints: a new fold's
test cities may have trained an old checkpoint, so the current representation
model (and the raw+response-block variant) must be retrained inside each fold;
that is a formal experiment and is gated on this ceiling test showing stable
out-of-city signal.

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
import pandas as pd
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


class _CellMetric(TypedDict):
    rmse: float
    baseline: float
    direction: float | None
    direction_baseline: float | None
    param: float


class _CityMetric(TypedDict):
    n_cells: int
    rmse_mean: float
    improvement: float
    direction_mean: float | None
    direction_baseline_mean: float | None


class _FoldResult(TypedDict):
    fold: int
    fit_units: int
    val_units: int
    test_units: int
    ridge: dict[int, _CellMetric]
    knn: dict[int, _CellMetric]
    retrieval: dict[str, float | None]
    retrieval_per_outcome: dict[str, dict[str, float | None]]
    per_city: dict[str, object]


class _Preprocessor:
    """Impute and standardize using one fitted reference row set only."""

    def __init__(self, features: np.ndarray, reference_idx: np.ndarray) -> None:
        reference = features[reference_idx]
        finite = np.isfinite(reference)
        counts = finite.sum(axis=0)
        sums = np.where(finite, reference, 0.0).sum(axis=0)
        mean = sums / np.maximum(counts, 1)
        mean = np.where(np.isfinite(mean) & (counts > 0), mean, 0.0)
        filled = np.where(finite, reference, mean)
        scale = filled.std(axis=0, ddof=0)
        self.mean = mean
        self.scale = np.where(scale < 1e-12, 1.0, scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(features), features, self.mean)
        return (filled - self.mean) / self.scale


def load_arrays(dataset_dir: Path) -> _Arrays:
    ds = RepresentationDataset(dataset_dir, only_training_mask=True, load_images=False)
    orders = ds.treatment_orders()
    unit_frame = ds.unit_features.set_index("treatment_order").reindex(orders)
    normalization = ds.normalization if isinstance(ds.normalization, dict) else {}
    raw_matrix = np.empty((len(orders), len(ds.feature_columns)), dtype=np.float64)
    for column_index, column in enumerate(ds.feature_columns):
        if column.startswith("z__"):
            raw_name = column[3:]
            if raw_name in unit_frame.columns:
                values = pd.to_numeric(unit_frame[raw_name], errors="coerce").to_numpy(
                    dtype=np.float64
                )
            else:
                # Only the z column is stored: invert the old training-split
                # affine transform so observed values return to the raw scale
                # and missing values stay missing instead of becoming zeros.
                parameters = normalization.get(raw_name)
                mean = 0.0
                scale = 1.0
                if isinstance(parameters, dict):
                    mean_value = parameters.get("mean")
                    if isinstance(mean_value, (int, float)) and np.isfinite(mean_value):
                        mean = float(mean_value)
                    scale_value = parameters.get("sd")
                    if isinstance(scale_value, (int, float)) and np.isfinite(scale_value) and scale_value > 0:
                        scale = float(scale_value)
                z_values = pd.to_numeric(unit_frame[column], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                values = z_values * scale + mean
                values[~np.isfinite(z_values)] = np.nan
        else:
            values = pd.to_numeric(unit_frame[column], errors="coerce").to_numpy(dtype=np.float64)
        raw_matrix[:, column_index] = values
    features_arr = raw_matrix
    responses, masks, cities = [], [], []
    for index in range(len(ds)):
        sample = ds[index]
        responses.append(sample.response_vector.numpy())
        masks.append(sample.response_mask.numpy())
        cities.append(sample.city_key)
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


def _knn_predict(distance_matrix: np.ndarray, y_fit: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(k, distance_matrix.shape[1])
    neighbours = np.argsort(distance_matrix, axis=1)[:, :k_eff]
    return y_fit[neighbours].mean(axis=1)


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def _majority_sign(values: np.ndarray) -> float:
    """Majority sign of non-zero labels; ties resolve to +1."""
    nonzero = values[np.abs(values) > 1e-12]
    if nonzero.size == 0:
        return 1.0
    positives = int((nonzero > 0).sum())
    negatives = int((nonzero < 0).sum())
    return 1.0 if positives >= negatives else -1.0


def _direction_metrics(
    prediction: np.ndarray, target: np.ndarray, majority_sign: float
) -> tuple[float | None, float | None]:
    nonzero = np.abs(target) > 1e-12
    if not nonzero.any():
        return None, None
    truth = np.sign(target[nonzero])
    accuracy = float(np.mean(np.sign(prediction[nonzero]) == truth))
    baseline = float(np.mean(np.full(truth.shape, majority_sign) == truth))
    return accuracy, baseline


def _cell_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    baseline_value: float,
    majority_sign: float,
    param: float,
) -> _CellMetric:
    direction, direction_baseline = _direction_metrics(prediction, target, majority_sign)
    return {
        "rmse": _rmse(prediction, target),
        "baseline": _rmse(np.full(target.shape, baseline_value), target),
        "direction": direction,
        "direction_baseline": direction_baseline,
        "param": param,
    }


def _city_metrics(metrics: list[_CellMetric]) -> _CityMetric:
    rmses = [m["rmse"] for m in metrics]
    baselines = [m["baseline"] for m in metrics]
    directions = [m["direction"] for m in metrics if m["direction"] is not None]
    direction_baselines = [
        m["direction_baseline"] for m in metrics if m["direction_baseline"] is not None
    ]
    return {
        "n_cells": len(metrics),
        "rmse_mean": float(np.mean(rmses)) if rmses else float("nan"),
        "improvement": float(np.mean(np.subtract(baselines, rmses))) if rmses else float("nan"),
        "direction_mean": float(np.mean(directions)) if directions else None,
        "direction_baseline_mean": float(np.mean(direction_baselines))
        if direction_baselines
        else None,
    }


def fold_controls(
    arrays: _Arrays,
    standardized_fit: np.ndarray,
    standardized_refit: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    min_obs: int,
) -> dict[str, object]:
    responses = arrays["responses"]
    masks = arrays["masks"]
    cities = np.asarray(arrays["cities"], dtype=object)
    refit_idx = np.concatenate([fit_idx, val_idx])
    distance_val_fit = cdist(standardized_fit[val_idx], standardized_fit[fit_idx])
    distance_test_refit = cdist(standardized_refit[test_idx], standardized_refit[refit_idx])
    test_cities = cities[test_idx]
    unique_cities = sorted({str(city) for city in test_cities})
    ridge_map: dict[int, _CellMetric] = {}
    knn_map: dict[int, _CellMetric] = {}
    per_city: dict[str, dict[str, list[_CellMetric]]] = {
        city: {"ridge": [], "knn": []} for city in unique_cities
    }
    for cell in range(responses.shape[1]):
        fit_obs = masks[fit_idx, cell]
        val_obs = masks[val_idx, cell]
        test_obs = masks[test_idx, cell]
        refit_obs = masks[refit_idx, cell]
        n_fit = int(fit_obs.sum())
        n_val = int(val_obs.sum())
        n_test = int(test_obs.sum())
        if n_fit < min_obs or n_test == 0:
            continue
        if int(refit_obs.sum()) == 0:
            continue
        y_fit = responses[fit_idx, cell][fit_obs]
        y_val = responses[val_idx, cell][val_obs]
        y_test = responses[test_idx, cell][test_obs]
        refit_values = responses[refit_idx, cell][refit_obs]
        # The regression is refit on fit+val, so its fair mean baseline and the
        # majority-sign baseline must also come from the refit labels.
        baseline_value = float(refit_values.mean())
        majority_sign = _majority_sign(refit_values)
        city_rows = test_cities[test_obs]
        ridge_param: float | None = None
        if n_val >= max(min_obs // 2, 4):
            best_ridge: tuple[float, float] | None = None
            for ridge in RIDGE_CANDIDATES:
                rmse = _rmse(
                    _ridge_fit_predict(
                        standardized_fit[fit_idx][fit_obs],
                        y_fit,
                        standardized_fit[val_idx][val_obs],
                        ridge,
                    ),
                    y_val,
                )
                if best_ridge is None or rmse < best_ridge[1]:
                    best_ridge = (ridge, rmse)
            ridge_param = float(best_ridge[0]) if best_ridge is not None else None
        if ridge_param is not None:
            prediction = _ridge_fit_predict(
                standardized_refit[refit_idx][refit_obs],
                refit_values,
                standardized_refit[test_idx][test_obs],
                ridge_param,
            )
            metric = _cell_metrics(prediction, y_test, baseline_value, majority_sign, ridge_param)
            ridge_map[cell] = metric
            for city in unique_cities:
                rows = city_rows == city
                if rows.any():
                    per_city[city]["ridge"].append(
                        _cell_metrics(
                            prediction[rows], y_test[rows], baseline_value, majority_sign, ridge_param
                        )
                    )
        knn_param: float | None = None
        if n_val >= max(min_obs // 2, 4):
            best_knn: tuple[float, float] | None = None
            for k in KNN_CANDIDATES:
                prediction = _knn_predict(distance_val_fit[:, fit_obs], y_fit, k)
                rmse = _rmse(prediction[val_obs], y_val)
                if best_knn is None or rmse < best_knn[1]:
                    best_knn = (float(k), rmse)
            knn_param = best_knn[0] if best_knn is not None else None
        if knn_param is not None:
            knn_prediction = _knn_predict(
                distance_test_refit[:, refit_obs], refit_values, int(knn_param)
            )
            # ``distance_test_refit`` covers every outer-test row; restrict to
            # the rows where this cell is observed, exactly like the ridge path.
            prediction = knn_prediction[test_obs]
            metric = _cell_metrics(prediction, y_test, baseline_value, majority_sign, knn_param)
            knn_map[cell] = metric
            for city in unique_cities:
                rows = city_rows == city
                if rows.any():
                    per_city[city]["knn"].append(
                        _cell_metrics(
                            prediction[rows], y_test[rows], baseline_value, majority_sign, knn_param
                        )
                    )
    per_city_summary: dict[str, object] = {}
    for city, bucket in per_city.items():
        per_city_summary[city] = {
            "n_units": int(np.sum(test_cities == city)),
            "ridge": _city_metrics(bucket["ridge"]),
            "knn": _city_metrics(bucket["knn"]),
        }
    return {"ridge": ridge_map, "knn": knn_map, "per_city": per_city_summary}


def fold_retrieval(
    standardized: np.ndarray,
    test_idx: np.ndarray,
    columns: list[int],
    responses: np.ndarray,
    masks: np.ndarray,
    k: int,
) -> dict[str, float | None]:
    import torch

    features = torch.from_numpy(standardized[test_idx]).float()
    response_tensor = torch.from_numpy(responses[test_idx][:, columns]).float()
    mask_tensor = torch.from_numpy(masks[test_idx][:, columns])
    if mask_tensor.sum() < 2:
        return {"nn_corr": None, "baseline": None, "n_units": 0.0}
    block = retrieval_metrics(features, response_tensor, mask_tensor, k=k)["overall"]
    nn_corr = block.get("nn_corr@k")
    baseline = block.get("baseline_corr")
    n_units = block.get("n_units", 0)
    return {
        "nn_corr": float(nn_corr) if isinstance(nn_corr, (int, float)) else None,
        "baseline": float(baseline) if isinstance(baseline, (int, float)) else None,
        "n_units": float(n_units) if isinstance(n_units, (int, float)) else 0.0,
    }


def aggregate(fold_results: list[_FoldResult], cells: list[int], method: str) -> dict[str, object]:
    rmses, baselines, params, directions, direction_baselines = [], [], [], [], []
    for fold in fold_results:
        method_map = fold["ridge"] if method == "ridge" else fold["knn"]
        for cell in cells:
            if cell in method_map:
                metric = method_map[cell]
                rmses.append(metric["rmse"])
                baselines.append(metric["baseline"])
                params.append(metric["param"])
                if metric["direction"] is not None:
                    directions.append(metric["direction"])
                if metric["direction_baseline"] is not None:
                    direction_baselines.append(metric["direction_baseline"])
    if not rmses:
        return {"n": 0}
    return {
        "n": len(rmses),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_sd": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        "baseline_mean": float(np.mean(baselines)),
        "amplitude_improvement": float(np.mean(np.subtract(baselines, rmses))),
        "direction_accuracy": float(np.mean(directions)) if directions else None,
        "direction_baseline": float(np.mean(direction_baselines)) if direction_baselines else None,
        "direction_improvement": float(np.mean(np.subtract(directions, direction_baselines)))
        if directions and direction_baselines
        else None,
        "selected_params": sorted({float(value) for value in params}),
    }


def aggregate_cities(fold_results: list[_FoldResult]) -> dict[str, object]:
    per_city: dict[str, dict[str, object]] = {}
    for fold in fold_results:
        for city, entry in fold["per_city"].items():
            assert isinstance(entry, dict)
            if city not in per_city:
                per_city[city] = {"ridge": [], "knn": [], "n_units": entry.get("n_units")}
            bucket = per_city[city]
            for method in ("ridge", "knn"):
                summary = entry.get(method)
                if isinstance(summary, dict):
                    values = bucket.get(method)
                    if isinstance(values, list):
                        values.append(summary)
    result: dict[str, object] = {}
    for city, bucket in sorted(per_city.items()):
        city_entry: dict[str, object] = {}
        n_units = bucket.get("n_units")
        if isinstance(n_units, (int, float)):
            city_entry["n_units"] = int(n_units)
        for method in ("ridge", "knn"):
            values = bucket.get(method)
            if not isinstance(values, list) or not values:
                continue
            directions = [
                value["direction_mean"] for value in values if value["direction_mean"] is not None
            ]
            direction_baselines = [
                value["direction_baseline_mean"]
                for value in values
                if value["direction_baseline_mean"] is not None
            ]
            city_entry[method] = {
                "n_cells": int(np.mean([value["n_cells"] for value in values])),
                "rmse_mean": float(np.mean([value["rmse_mean"] for value in values])),
                "amplitude_improvement": float(np.mean([value["improvement"] for value in values])),
                "direction_accuracy": float(np.mean(directions)) if directions else None,
                "direction_baseline": float(np.mean(direction_baselines))
                if direction_baselines
                else None,
            }
        result[city] = city_entry
    return result


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
    cities = arrays["cities"]
    primary = arrays["primary_family"]
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
        refit_idx = np.concatenate([fit_idx, val_idx])
        # Inner selection: preprocessing fitted on fit cities only.
        standardized_fit = _Preprocessor(arrays["features"], fit_idx).transform(arrays["features"])
        # Final refit: preprocessing fitted on fit + inner-validation cities.
        standardized_refit = _Preprocessor(arrays["features"], refit_idx).transform(
            arrays["features"]
        )
        controls = fold_controls(
            arrays, standardized_fit, standardized_refit, fit_idx, val_idx, test_idx, args.min_obs
        )
        all_cells = [cell for cells in groups.values() for cell in cells]
        retrieval = fold_retrieval(
            standardized_refit, test_idx, all_cells, arrays["responses"], arrays["masks"], args.k
        )
        retrieval_per_outcome = {
            outcome: fold_retrieval(
                standardized_refit,
                test_idx,
                cells,
                arrays["responses"],
                arrays["masks"],
                args.k,
            )
            for outcome, cells in groups.items()
        }
        fold_result: _FoldResult = {
            "fold": fold,
            "fit_units": int(len(fit_idx)),
            "val_units": int(len(val_idx)),
            "test_units": int(len(test_idx)),
            "ridge": controls["ridge"],  # type: ignore[typeddict-item]
            "knn": controls["knn"],  # type: ignore[typeddict-item]
            "retrieval": retrieval,
            "retrieval_per_outcome": retrieval_per_outcome,
            "per_city": controls["per_city"],  # type: ignore[typeddict-item]
        }
        fold_results.append(fold_result)
        print(
            f"fold {fold}: fit {len(fit_idx)} / val {len(val_idx)} / test {len(test_idx)} | "
            f"shape nn_corr {retrieval['nn_corr'] if retrieval['nn_corr'] is not None else float('nan'):.4f} | "
            f"cities {sorted(test_set)}"
        )
    all_cells = [cell for cells in groups.values() for cell in cells]
    retrieval_folds = [
        float(value)
        for value in (fold["retrieval"]["nn_corr"] for fold in fold_results)
        if value is not None
    ]
    retrieval_baselines = [
        float(value)
        for value in (fold["retrieval"]["baseline"] for fold in fold_results)
        if value is not None
    ]
    overall: dict[str, object] = {
        "shape_retrieval": {
            "nn_corr_mean": float(np.mean(retrieval_folds)) if retrieval_folds else None,
            "baseline_mean": float(np.mean(retrieval_baselines)) if retrieval_baselines else None,
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
        values = [
            float(value)
            for value in (
                fold["retrieval_per_outcome"][outcome]["nn_corr"] for fold in fold_results
            )
            if value is not None
        ]
        baselines = [
            float(value)
            for value in (
                fold["retrieval_per_outcome"][outcome]["baseline"] for fold in fold_results
            )
            if value is not None
        ]
        entry["shape_retrieval"] = {
            "nn_corr_mean": float(np.mean(values)) if values else None,
            "baseline_mean": float(np.mean(baselines)) if baselines else None,
            "n": len(values),
        }
        per_outcome[outcome] = entry
    summary: dict[str, object] = {
        "primary_family": primary,
        "overall": overall,
        "per_outcome": per_outcome,
        "per_city": aggregate_cities(fold_results),
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
