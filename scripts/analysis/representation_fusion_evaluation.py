"""Step-1 fusion evaluation: can a response block improve response retrieval?

This script implements the reviewed Step-1 protocol only; it does not retrain
any neural model and does not touch the frozen causal labels.

Protocol (per fold, identical city folds to ``representation_controls.py``):

1. preprocess features and fit per-cell ridge on the fit cities only;
2. select the ridge penalty alpha (per cell) and the fusion scheme
   (response-block version x lambda) on the inner-validation cities;
3. lock the parameters, refit preprocessing and ridge on fit + validation;
4. evaluate the outer-test pools with the locked scheme.

Fusion: ``z_i(lambda) = [sqrt(1-lambda) * xbar_i, sqrt(lambda) * hbar_i]``
with both blocks unit-normalized.  ``lambda = 0`` recovers the raw feature
geometry exactly (verified numerically here).  The two response-block versions
are the raw per-cell standardized ridge predictions and a per-outcome
temporally de-meaned shape block; all standardization statistics come from the
training side only, and per-cell standardization is not assumed to preserve
Pearson shape (both versions are reported).

Diagnostics: the full lambda curve is reported on the outer test only as a
diagnostic; the headline result is the inner-selected scheme.  City-level
paired increments (scheme minus raw) are macro-averaged with a bootstrap
interval, per outcome and overall, with and without same-city candidates
excluded.  Amplitude and direction use the un-unitized ridge predictions.

Run on the server::

    conda run -n mit python scripts/analysis/representation_fusion_evaluation.py \
        --dataset data/active/model_inputs/viirs-mg-design \
        --output-directory outputs/representation_fusion/viirs
    conda run -n mit python scripts/analysis/representation_fusion_evaluation.py \
        --dataset data/active/model_inputs/poi-mg-design \
        --output-directory outputs/representation_fusion/poi
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.representation.evaluation import (  # noqa: E402
    _metric_block,
    cosine_similarity,
    response_similarity_with_validity,
    retrieval_metrics,
)

_CONTROLS_PATH = Path(__file__).with_name("representation_controls.py")
_SPEC = importlib.util.spec_from_file_location("representation_controls", _CONTROLS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
controls = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(controls)

LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
BLOCK_VERSIONS = ("raw", "shape")
ZERO_NORM_EPS = 1e-12


def shape_transform(predictions: np.ndarray, groups: dict[str, list[int]]) -> np.ndarray:
    """Remove each outcome's horizon mean per grid (keeps the temporal shape)."""
    transformed = predictions.copy()
    for columns in groups.values():
        block = transformed[:, columns]
        transformed[:, columns] = block - block.mean(axis=1, keepdims=True)
    return transformed


def unit_normalize(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    valid = norms[:, 0] > ZERO_NORM_EPS
    safe = np.where(valid[:, None], norms, 1.0)
    return block / safe, valid


def build_block(
    target_predictions: np.ndarray,
    reference_predictions: np.ndarray,
    version: str,
    groups: dict[str, list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize on the training side (per cell), then unit-normalize rows."""
    target_values = target_predictions
    reference_values = reference_predictions
    if version == "shape":
        target_values = shape_transform(target_values, groups)
        reference_values = shape_transform(reference_values, groups)
    mean = reference_values.mean(axis=0)
    scale = reference_values.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    standardized = np.nan_to_num((target_values - mean) / scale, nan=0.0)
    return unit_normalize(standardized)


def fuse(x_block: np.ndarray, h_block: np.ndarray, lam: float) -> np.ndarray:
    return np.concatenate(
        [np.sqrt(1.0 - lam) * x_block, np.sqrt(lam) * h_block], axis=1
    )


def retrieval_block(
    embedding: np.ndarray,
    responses: np.ndarray,
    masks: np.ndarray,
    columns: list[int],
    k: int,
    city_keys: np.ndarray | None = None,
    exclude_same_city: bool = False,
) -> dict[str, object]:
    emb = torch.from_numpy(np.ascontiguousarray(embedding)).float()
    resp = torch.from_numpy(np.ascontiguousarray(responses[:, columns])).float()
    mask = torch.from_numpy(np.ascontiguousarray(masks[:, columns]))
    sim_resp, valid = response_similarity_with_validity(resp, mask)
    if exclude_same_city and city_keys is not None:
        same_city = torch.from_numpy(
            np.ascontiguousarray(city_keys[:, None] == city_keys[None, :])
        )
        valid = valid & ~same_city
    return _metric_block(cosine_similarity(emb), sim_resp, valid, k)


def per_city_deltas(
    raw_embedding: np.ndarray,
    scheme_embedding: np.ndarray,
    responses: np.ndarray,
    masks: np.ndarray,
    city_keys: np.ndarray,
    columns: list[int],
    k: int,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for city in sorted({str(city) for city in city_keys}):
        rows = np.where(city_keys == city)[0]
        if len(rows) < k + 1:
            continue
        raw_block = retrieval_block(
            raw_embedding[rows], responses[rows], masks[rows], columns, k
        )
        scheme_block = retrieval_block(
            scheme_embedding[rows], responses[rows], masks[rows], columns, k
        )
        raw_nn = raw_block.get("nn_corr@k")
        scheme_nn = scheme_block.get("nn_corr@k")
        if not isinstance(raw_nn, (int, float)) or not isinstance(scheme_nn, (int, float)):
            continue
        result[city] = {
            "n_units": float(len(rows)),
            "raw_nn_corr": float(raw_nn),
            "scheme_nn_corr": float(scheme_nn),
            "delta": float(scheme_nn) - float(raw_nn),
        }
    return result


def amplitude_direction(
    predictions: np.ndarray,
    responses: np.ndarray,
    masks: np.ndarray,
    columns: list[int],
) -> dict[str, float | None]:
    """Un-unitized ridge predictions: RMSE vs refit mean, direction vs majority sign."""
    mse_model = mse_baseline = 0.0
    cells = 0
    direction_hits = 0.0
    direction_baseline_hits = 0.0
    direction_total = 0
    for cell in columns:
        observed = masks[:, cell]
        if not observed.any():
            continue
        y = responses[:, cell][observed]
        prediction = predictions[observed, cell]
        baseline_value = float(y.mean())
        mse_model += float(((prediction - y) ** 2).sum())
        mse_baseline += float(((baseline_value - y) ** 2).sum())
        cells += 1
        nonzero = np.abs(y) > 1e-12
        if nonzero.any():
            truth = np.sign(y[nonzero])
            direction_hits += float((np.sign(prediction[nonzero]) == truth).sum())
            direction_baseline_hits += float(
                (np.full(truth.shape, controls._majority_sign(y)) == truth).sum()
            )
            direction_total += int(nonzero.sum())
    if cells == 0:
        return {
            "rmse": None,
            "rmse_baseline": None,
            "relative_sq_error_reduction": None,
            "direction_accuracy": None,
            "direction_baseline": None,
            "direction_improvement": None,
        }
    rmse = float(np.sqrt(mse_model / cells))
    rmse_baseline = float(np.sqrt(mse_baseline / cells))
    relative = 1.0 - mse_model / mse_baseline if mse_baseline > 0 else None
    direction_accuracy = direction_hits / direction_total if direction_total else None
    direction_baseline = (
        direction_baseline_hits / direction_total if direction_total else None
    )
    return {
        "rmse": rmse,
        "rmse_baseline": rmse_baseline,
        "relative_sq_error_reduction": relative,
        "direction_accuracy": direction_accuracy,
        "direction_baseline": direction_baseline,
        "direction_improvement": (
            None
            if direction_accuracy is None or direction_baseline is None
            else direction_accuracy - direction_baseline
        ),
    }


def bootstrap_mean_ci(
    values: list[float], seed: int = 0, n_boot: int = 2000
) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(n_boot, array.size), replace=True).mean(axis=1)
    return float(array.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def mean_of(values: Sequence[object]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return float(np.mean(numbers)) if numbers else None


def metric_value(block: dict[str, object], key: str) -> float | None:
    value = block.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-obs", type=int, default=16)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    arrays = controls.load_arrays(args.dataset)
    cities = arrays["cities"]
    family = arrays["primary_family"]
    features = arrays["features"]
    responses = arrays["responses"]
    masks = arrays["masks"]
    groups = controls.outcome_groups(family)
    family_columns = [column for columns in groups.values() for column in columns]
    city_array = np.asarray(cities, dtype=object)
    assignment = controls.make_city_folds(cities, args.folds, args.seed)

    fold_records: list[dict[str, object]] = []
    outer: dict[str, list[float | None]] = {"raw": [], "h_only": [], "fusion": []}
    outer_baseline: dict[str, list[float | None]] = {"raw": [], "h_only": [], "fusion": []}
    no_same_city: dict[str, list[float | None]] = {"raw": [], "fusion": []}
    lambda_curve: dict[str, dict[str, float | None]] = {}
    per_outcome: dict[str, dict[str, list[float | None]]] = {
        outcome: {"raw": [], "h_only": [], "fusion": []} for outcome in groups
    }
    per_city_all: dict[str, dict[str, float]] = {}
    amplitude_records: dict[str, list[dict[str, float | None]]] = {
        "overall": [],
        **{outcome: [] for outcome in groups},
    }
    lambda_zero_folds = 0
    lambda_zero_diffs: list[float] = []
    excluded_total = 0

    for fold in range(args.folds):
        test_cities = [city for city in set(cities) if assignment[city] == fold]
        train_cities = [city for city in set(cities) if assignment[city] != fold]
        inner_fit, inner_val = controls.inner_city_split(train_cities, args.seed + fold)
        fit_set, val_set, test_set = set(inner_fit), set(inner_val), set(test_cities)
        fit_idx = np.array([i for i, city in enumerate(cities) if city in fit_set])
        val_idx = np.array([i for i, city in enumerate(cities) if city in val_set])
        test_idx = np.array([i for i, city in enumerate(cities) if city in test_set])
        refit_idx = np.concatenate([fit_idx, val_idx])

        x_fit_all = controls._Preprocessor(features, fit_idx).transform(features)
        x_refit_all = controls._Preprocessor(features, refit_idx).transform(features)

        # --- Inner: per-cell alpha on fit -> val ---
        alphas: list[float | None] = []
        for cell in range(responses.shape[1]):
            fit_obs = masks[fit_idx, cell]
            val_obs = masks[val_idx, cell]
            if int(fit_obs.sum()) < args.min_obs or int(val_obs.sum()) == 0:
                alphas.append(None)
                continue
            best: tuple[float, float] | None = None
            for alpha in controls.RIDGE_CANDIDATES:
                prediction = controls._ridge_fit_predict(
                    x_fit_all[fit_idx][fit_obs],
                    responses[fit_idx, cell][fit_obs],
                    x_fit_all[val_idx],
                    alpha,
                )
                rmse = controls._rmse(
                    prediction[val_obs], responses[val_idx, cell][val_obs]
                )
                if best is None or rmse < best[1]:
                    best = (float(alpha), rmse)
            alphas.append(best[0] if best is not None else None)

        def predict(
            x_all: np.ndarray,
            reference_idx: np.ndarray,
            target_idx: np.ndarray,
            alphas_default: list[float | None] = alphas,
        ) -> np.ndarray:
            predictions = np.full((len(target_idx), responses.shape[1]), np.nan)
            for cell, alpha in enumerate(alphas_default):
                if alpha is None:
                    continue
                observed = masks[reference_idx, cell]
                if int(observed.sum()) < 2:
                    continue
                predictions[:, cell] = controls._ridge_fit_predict(
                    x_all[reference_idx][observed],
                    responses[reference_idx, cell][observed],
                    x_all[target_idx],
                    alpha,
                )
            return predictions

        fit_predictions = predict(x_fit_all, fit_idx, fit_idx)
        val_predictions = predict(x_fit_all, fit_idx, val_idx)

        # --- Inner selection on val: block version x lambda ---
        x_val_block, x_val_valid = unit_normalize(x_fit_all[val_idx])
        val_h: dict[str, np.ndarray] = {}
        val_h_valid: dict[str, np.ndarray] = {}
        for version in BLOCK_VERSIONS:
            h_block, h_block_valid = build_block(
                val_predictions, fit_predictions, version, groups
            )
            val_h[version] = h_block
            val_h_valid[version] = h_block_valid
        inner_valid = x_val_valid.copy()
        for version in BLOCK_VERSIONS:
            inner_valid &= val_h_valid[version]
        best_selection: tuple[float, str, float] | None = None
        best_key: tuple[float, float, int] | None = None
        for version in BLOCK_VERSIONS:
            for lam in LAMBDA_GRID:
                fused = fuse(x_val_block, val_h[version], lam)[inner_valid]
                metrics = retrieval_block(
                    fused,
                    responses[val_idx][inner_valid],
                    masks[val_idx][inner_valid],
                    family_columns,
                    args.k,
                )
                value = metrics.get("nn_corr@k")
                if not isinstance(value, (int, float)):
                    continue
                key = (float(value), -lam, 0 if version == "raw" else 1)
                if best_key is None or key > best_key:
                    best_key = key
                    best_selection = (float(lam), version, float(value))
        if best_selection is None:
            raise RuntimeError(f"fold {fold}: no valid inner selection")
        selected_lambda, selected_version, val_score = best_selection
        raw_val_block = retrieval_block(
            fuse(x_val_block, val_h[selected_version], 0.0)[inner_valid],
            responses[val_idx][inner_valid],
            masks[val_idx][inner_valid],
            family_columns,
            args.k,
        )
        raw_val_value = raw_val_block.get("nn_corr@k")

        # --- Outer: refit on fit+val, evaluate on test ---
        refit_predictions = predict(x_refit_all, refit_idx, refit_idx)
        test_predictions = predict(x_refit_all, refit_idx, test_idx)
        test_predictions_all = test_predictions.copy()
        test_responses_all = responses[test_idx]
        test_masks_all = masks[test_idx]
        x_test_block, x_test_valid = unit_normalize(x_refit_all[test_idx])
        h_test_block, h_test_valid = build_block(
            test_predictions, refit_predictions, selected_version, groups
        )
        common_valid = x_test_valid & h_test_valid
        excluded_total += int((~common_valid).sum())
        test_responses = responses[test_idx][common_valid]
        test_masks = masks[test_idx][common_valid]
        test_city = city_array[test_idx][common_valid]
        x_test_block = x_test_block[common_valid]
        h_test_block = h_test_block[common_valid]
        test_predictions = test_predictions[common_valid]

        embeddings = {
            "raw": fuse(x_test_block, h_test_block, 0.0),
            "h_only": fuse(x_test_block, h_test_block, 1.0),
            "fusion": fuse(x_test_block, h_test_block, selected_lambda),
        }
        for name, embedding in embeddings.items():
            metrics = retrieval_block(
                embedding, test_responses, test_masks, family_columns, args.k
            )
            outer[name].append(metric_value(metrics, "nn_corr@k"))
            outer_baseline[name].append(metric_value(metrics, "baseline_corr"))

        curve: dict[str, float | None] = {}
        for lam in LAMBDA_GRID:
            metrics = retrieval_block(
                fuse(x_test_block, h_test_block, lam),
                test_responses,
                test_masks,
                family_columns,
                args.k,
            )
            value = metrics.get("nn_corr@k")
            curve[str(lam)] = float(value) if isinstance(value, (int, float)) else None
        lambda_curve[str(fold)] = curve

        for name, embedding in (("raw", embeddings["raw"]), ("fusion", embeddings["fusion"])):
            metrics = retrieval_block(
                embedding,
                test_responses,
                test_masks,
                family_columns,
                args.k,
                city_keys=test_city,
                exclude_same_city=True,
            )
            no_same_city[name].append(metric_value(metrics, "nn_corr@k"))

        for outcome, columns in groups.items():
            for name, embedding in embeddings.items():
                metrics = retrieval_block(
                    embedding, test_responses, test_masks, columns, args.k
                )
                per_outcome[outcome][name].append(metric_value(metrics, "nn_corr@k"))

        for city, entry in per_city_deltas(
            embeddings["raw"],
            embeddings["fusion"],
            test_responses,
            test_masks,
            test_city,
            family_columns,
            args.k,
        ).items():
            per_city_all[city] = entry

        amplitude_records["overall"].append(
            amplitude_direction(
                test_predictions_all, test_responses_all, test_masks_all, family_columns
            )
        )
        for outcome, columns in groups.items():
            amplitude_records[outcome].append(
                amplitude_direction(
                    test_predictions_all, test_responses_all, test_masks_all, columns
                )
            )

        reference_block = retrieval_metrics(
            torch.from_numpy(x_refit_all[test_idx][common_valid]).float(),
            torch.from_numpy(test_responses[:, family_columns]).float(),
            torch.from_numpy(test_masks[:, family_columns]),
            k=args.k,
        )["overall"]
        reference_nn = reference_block.get("nn_corr@k")
        zero_nn = outer["raw"][-1]
        if isinstance(reference_nn, (int, float)) and isinstance(zero_nn, (int, float)):
            lambda_zero_diffs.append(abs(float(reference_nn) - float(zero_nn)))

        lambda_zero = selected_lambda == 0.0
        lambda_zero_folds += int(lambda_zero)
        fold_records.append(
            {
                "fold": fold,
                "fit_units": int(len(fit_idx)),
                "val_units": int(len(val_idx)),
                "test_units": int(len(test_idx)),
                "selected_version": selected_version,
                "selected_lambda": float(selected_lambda),
                "lambda_zero_selected": lambda_zero,
                "val_nn_corr_selected": val_score,
                "val_nn_corr_raw": (
                    float(raw_val_value)
                    if isinstance(raw_val_value, (int, float))
                    else None
                ),
                "per_cell_ridge": [
                    float(alpha) if alpha is not None else None for alpha in alphas
                ],
                "excluded_zero_norm": int((~common_valid).sum()),
            }
        )
        print(
            f"fold {fold}: fit {len(fit_idx)} / val {len(val_idx)} / test {len(test_idx)} | "
            f"selected {selected_version}, lambda={selected_lambda:g} | "
            f"val nn_corr {val_score:.4f} "
            f"(raw {raw_val_value if isinstance(raw_val_value, (int, float)) else float('nan'):.4f})"
        )

    city_values = [entry["delta"] for entry in per_city_all.values()]
    macro_mean, macro_low, macro_high = bootstrap_mean_ci(city_values, n_boot=args.bootstrap)
    per_outcome_summary = {
        outcome: {
            name: {"nn_corr_mean": mean_of(values)}
            for name, values in entry.items()
        }
        for outcome, entry in per_outcome.items()
    }
    amplitude_summary = {
        "overall": {
            "relative_sq_error_reduction_mean": mean_of(
                [record.get("relative_sq_error_reduction") for record in amplitude_records["overall"]]
            ),
            "direction_accuracy_mean": mean_of(
                [record.get("direction_accuracy") for record in amplitude_records["overall"]]
            ),
            "direction_baseline_mean": mean_of(
                [record.get("direction_baseline") for record in amplitude_records["overall"]]
            ),
        },
        "per_outcome": {
            outcome: {
                "relative_sq_error_reduction_mean": mean_of(
                    [record.get("relative_sq_error_reduction") for record in records]
                ),
                "direction_accuracy_mean": mean_of(
                    [record.get("direction_accuracy") for record in records]
                ),
                "direction_baseline_mean": mean_of(
                    [record.get("direction_baseline") for record in records]
                ),
            }
            for outcome, records in amplitude_records.items()
            if outcome != "overall"
        },
    }
    report = {
        "dataset": str(args.dataset),
        "primary_family": family,
        "folds": args.folds,
        "seed": args.seed,
        "parameters": {
            "k": args.k,
            "min_obs": args.min_obs,
            "lambda_grid": list(LAMBDA_GRID),
            "block_versions": list(BLOCK_VERSIONS),
            "ridge_candidates": list(controls.RIDGE_CANDIDATES),
        },
        "city_folds": {city: int(assignment[city]) for city in sorted(assignment)},
        "fold_records": fold_records,
        "lambda_zero_share": lambda_zero_folds / max(args.folds, 1),
        "lambda_zero_equivalence_max_abs_diff": (
            max(lambda_zero_diffs) if lambda_zero_diffs else None
        ),
        "excluded_zero_norm_total": excluded_total,
        "overall": {
            name: {
                "nn_corr_mean": mean_of(outer[name]),
                "baseline_mean": mean_of(outer_baseline[name]),
            }
            for name in ("raw", "h_only", "fusion")
        },
        "no_same_city": {
            name: {"nn_corr_mean": mean_of(values)}
            for name, values in no_same_city.items()
        },
        "lambda_curve_diagnostic": lambda_curve,
        "per_outcome": per_outcome_summary,
        "city_macro": {
            "n_cities": len(city_values),
            "mean_delta": macro_mean,
            "bootstrap_95_ci": [macro_low, macro_high],
        },
        "per_city": per_city_all,
        "amplitude_direction": amplitude_summary,
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    out_path = args.output_directory / "fusion_evaluation.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["overall"], indent=2, ensure_ascii=False, default=str))
    print("city macro:", json.dumps(report["city_macro"], ensure_ascii=False, default=str))
    print("lambda zero share:", report["lambda_zero_share"])
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
