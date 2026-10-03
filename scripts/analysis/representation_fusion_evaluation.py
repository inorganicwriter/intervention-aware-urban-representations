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
geometry exactly (verified against the repository retrieval helper on both the
full pool and the common response-block support).

Response blocks: the raw version standardizes per-cell ridge predictions with
training-side statistics; the shape version first removes each outcome's
horizon mean over the *fitted* cells and then standardizes per cell, so the
final block is a per-cell rescaled shape deviation, not a representation that
is exactly zero-mean along time and not guaranteed to align with Pearson
directly.  Cells without a training-side ridge fit stay missing (never
contaminate the rest of the outcome) and the response-block support is
reported.

Diagnostics: the full lambda curve is reported on the outer test only as a
diagnostic; the headline result is the inner-selected scheme.  City-level
paired increments are computed per anchor on the full evaluation pool and then
macro-averaged by the anchor city (same-city-excluded candidates are a separate
report), with a bootstrap interval over cities; amplitude and direction use the
un-unitized ridge predictions with fit+val baselines.

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
    _per_anchor_nn_corr,
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
MIN_FITTED_CELLS = 2


def shape_transform(
    predictions: np.ndarray, groups: dict[str, list[int]], fitted_mask: np.ndarray
) -> np.ndarray:
    """Remove the fitted horizon means per outcome; unfitted cells stay NaN."""
    transformed = np.full_like(predictions, np.nan, dtype=np.float64)
    for columns in groups.values():
        fitted = [column for column in columns if fitted_mask[column]]
        if not fitted:
            continue
        block = predictions[:, fitted]
        transformed[:, fitted] = block - np.nanmean(block, axis=1, keepdims=True)
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
    fitted_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Training-side per-cell standardization; unfitted cells are zeroed."""
    target_values = target_predictions
    reference_values = reference_predictions
    if version == "shape":
        target_values = shape_transform(target_values, groups, fitted_mask)
        reference_values = shape_transform(reference_values, groups, fitted_mask)
    standardized = np.zeros_like(target_values, dtype=np.float64)
    fitted_cells = int(fitted_mask.sum())
    for cell in np.where(fitted_mask)[0]:
        reference = reference_values[:, cell]
        reference = reference[np.isfinite(reference)]
        if reference.size < 2:
            continue
        mean = float(reference.mean())
        scale = float(reference.std(ddof=0))
        scale = scale if scale > 1e-12 else 1.0
        standardized[:, cell] = (
            np.nan_to_num(target_values[:, cell], nan=0.0) - mean
        ) / scale
    block, norm_valid = unit_normalize(standardized)
    valid = norm_valid & (fitted_cells >= MIN_FITTED_CELLS)
    return block, valid, fitted_cells


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


def per_anchor_city_deltas(
    raw_embedding: np.ndarray,
    scheme_embedding: np.ndarray,
    responses: np.ndarray,
    masks: np.ndarray,
    cities: np.ndarray,
    columns: list[int],
    k: int,
    exclude_same_city: bool = False,
) -> tuple[dict[str, dict[str, float]], list[float]]:
    """Per-anchor paired increments on the given pool, macro-averaged by city."""
    raw_cos = cosine_similarity(
        torch.from_numpy(np.ascontiguousarray(raw_embedding)).float()
    )
    scheme_cos = cosine_similarity(
        torch.from_numpy(np.ascontiguousarray(scheme_embedding)).float()
    )
    resp = torch.from_numpy(np.ascontiguousarray(responses[:, columns])).float()
    mask = torch.from_numpy(np.ascontiguousarray(masks[:, columns]))
    sim_resp, valid = response_similarity_with_validity(resp, mask)
    if exclude_same_city:
        same_city = torch.from_numpy(
            np.ascontiguousarray(cities[:, None] == cities[None, :])
        )
        valid = valid & ~same_city
    raw_scores = _per_anchor_nn_corr(raw_cos, sim_resp, k, valid)
    scheme_scores = _per_anchor_nn_corr(scheme_cos, sim_resp, k, valid)
    per_city: dict[str, dict[str, float]] = {}
    for city in sorted({str(city) for city in cities}):
        rows = np.where(cities == city)[0]
        deltas = scheme_scores[rows] - raw_scores[rows]
        deltas = deltas[np.isfinite(deltas)]
        if deltas.size == 0:
            continue
        per_city[city] = {
            "n_anchors": int(deltas.size),
            "delta": float(deltas.mean()),
        }
    macro = [entry["delta"] for entry in per_city.values()]
    return per_city, macro


def baseline_from_training(
    responses: np.ndarray, masks: np.ndarray, reference_idx: np.ndarray
) -> tuple[dict[int, float], dict[int, float]]:
    """Per-cell mean and majority sign from reference (fit+val) labels only."""
    means: dict[int, float] = {}
    signs: dict[int, float] = {}
    for cell in range(responses.shape[1]):
        observed = masks[reference_idx, cell]
        if not observed.any():
            continue
        values = responses[reference_idx, cell][observed]
        means[cell] = float(values.mean())
        signs[cell] = controls._majority_sign(values)
    return means, signs


def amplitude_direction(
    predictions: np.ndarray,
    responses: np.ndarray,
    masks: np.ndarray,
    columns: list[int],
    baseline_means: dict[int, float],
    baseline_signs: dict[int, float],
) -> dict[str, float | int | None]:
    """Un-unitized ridge predictions against fixed fit+val baselines."""
    sse_model = sse_baseline = 0.0
    n_observations = 0
    per_cell_model: list[float] = []
    per_cell_baseline: list[float] = []
    direction_hits = 0
    direction_baseline_hits = 0
    direction_total = 0
    per_cell_direction: list[float] = []
    per_cell_direction_baseline: list[float] = []
    fitted_cells = 0
    for cell in columns:
        if cell not in baseline_means:
            continue
        observed = masks[:, cell] & np.isfinite(predictions[:, cell])
        if not observed.any():
            continue
        fitted_cells += 1
        y = responses[:, cell][observed]
        prediction = predictions[observed, cell]
        baseline_value = baseline_means[cell]
        baseline_prediction = np.full(y.shape, baseline_value)
        model_sq = float(((prediction - y) ** 2).sum())
        baseline_sq = float(((baseline_prediction - y) ** 2).sum())
        sse_model += model_sq
        sse_baseline += baseline_sq
        n_observations += int(y.size)
        per_cell_model.append(model_sq / y.size)
        per_cell_baseline.append(baseline_sq / y.size)
        nonzero = np.abs(y) > 1e-12
        if nonzero.any():
            truth = np.sign(y[nonzero])
            hits = int((np.sign(prediction[nonzero]) == truth).sum())
            baseline_hits = int(
                (np.sign(np.full(truth.shape, baseline_signs[cell])) == truth).sum()
            )
            direction_hits += hits
            direction_baseline_hits += baseline_hits
            direction_total += int(nonzero.sum())
            per_cell_direction.append(hits / nonzero.sum())
            per_cell_direction_baseline.append(baseline_hits / nonzero.sum())
    if fitted_cells == 0 or n_observations == 0:
        return {
            "fitted_cells": fitted_cells,
            "n_observations": n_observations,
            "rmse_pooled": None,
            "rmse_pooled_baseline": None,
            "rmse_cell_macro": None,
            "rmse_cell_macro_baseline": None,
            "relative_sq_error_reduction_pooled": None,
            "relative_sq_error_reduction_cell_macro": None,
            "direction_accuracy_pooled": None,
            "direction_baseline_pooled": None,
            "direction_improvement_pooled": None,
            "direction_accuracy_cell_macro": None,
            "direction_baseline_cell_macro": None,
        }
    rmse_pooled = float(np.sqrt(sse_model / n_observations))
    rmse_pooled_baseline = float(np.sqrt(sse_baseline / n_observations))
    mean_cell_model = float(np.mean(per_cell_model))
    mean_cell_baseline = float(np.mean(per_cell_baseline))
    direction_accuracy = (
        direction_hits / direction_total if direction_total else None
    )
    direction_baseline = (
        direction_baseline_hits / direction_total if direction_total else None
    )
    return {
        "fitted_cells": fitted_cells,
        "n_observations": n_observations,
        "rmse_pooled": rmse_pooled,
        "rmse_pooled_baseline": rmse_pooled_baseline,
        "rmse_cell_macro": float(np.sqrt(mean_cell_model)),
        "rmse_cell_macro_baseline": float(np.sqrt(mean_cell_baseline)),
        "relative_sq_error_reduction_pooled": (
            1.0 - sse_model / sse_baseline if sse_baseline > 0 else None
        ),
        "relative_sq_error_reduction_cell_macro": (
            1.0 - mean_cell_model / mean_cell_baseline
            if mean_cell_baseline > 0
            else None
        ),
        "direction_accuracy_pooled": direction_accuracy,
        "direction_baseline_pooled": direction_baseline,
        "direction_improvement_pooled": (
            direction_accuracy - direction_baseline
            if direction_accuracy is not None and direction_baseline is not None
            else None
        ),
        "direction_accuracy_cell_macro": (
            float(np.mean(per_cell_direction)) if per_cell_direction else None
        ),
        "direction_baseline_cell_macro": (
            float(np.mean(per_cell_direction_baseline))
            if per_cell_direction_baseline
            else None
        ),
    }


def bootstrap_mean_ci(
    values: Sequence[float], seed: int = 0, n_boot: int = 2000
) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(n_boot, array.size), replace=True).mean(axis=1)
    return (
        float(array.mean()),
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


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
    outer: dict[str, list[float | None]] = {"raw_full": [], "raw_common": [], "h_only": [], "fusion": []}
    outer_baseline: dict[str, list[float | None]] = {"raw_full": [], "raw_common": [], "h_only": [], "fusion": []}
    no_same_city: dict[str, list[float | None]] = {"raw": [], "fusion": []}
    lambda_curve: dict[str, dict[str, float | None]] = {}
    per_outcome: dict[str, dict[str, list[float | None]]] = {
        outcome: {"raw": [], "h_only": [], "fusion": [], "city_macro": []} for outcome in groups
    }
    per_city_all: dict[str, dict[str, float]] = {}
    amplitude_records: dict[str, list[dict[str, float | int | None]]] = {
        "overall": [],
        **{outcome: [] for outcome in groups},
    }
    city_macro_overall: list[float] = []
    city_macro_no_same_city: list[float] = []
    per_outcome_city_macro: dict[str, list[float]] = {outcome: [] for outcome in groups}
    lambda_zero_folds = 0
    equivalence_full: list[float] = []
    equivalence_common: list[float] = []
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

        fit_fitted = np.array(
            [
                alphas[cell] is not None and int(masks[fit_idx, cell].sum()) >= 2
                for cell in range(responses.shape[1])
            ]
        )
        fit_predictions = predict(x_fit_all, fit_idx, fit_idx)
        val_predictions = predict(x_fit_all, fit_idx, val_idx)

        # --- Inner selection on val: block version x lambda ---
        x_val_block, x_val_valid = unit_normalize(x_fit_all[val_idx])
        val_h: dict[str, np.ndarray] = {}
        val_h_valid: dict[str, np.ndarray] = {}
        for version in BLOCK_VERSIONS:
            h_block, h_block_valid, _ = build_block(
                val_predictions, fit_predictions, version, groups, fit_fitted
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
                value = metric_value(metrics, "nn_corr@k")
                if value is None:
                    continue
                key = (value, -lam, 0 if version == "raw" else 1)
                if best_key is None or key > best_key:
                    best_key = key
                    best_selection = (float(lam), version, value)
        if best_selection is None:
            raise RuntimeError(f"fold {fold}: no valid inner selection")
        selected_lambda, selected_version, val_score = best_selection
        raw_val_metrics = retrieval_block(
            fuse(x_val_block, val_h[selected_version], 0.0)[inner_valid],
            responses[val_idx][inner_valid],
            masks[val_idx][inner_valid],
            family_columns,
            args.k,
        )
        raw_val_value = metric_value(raw_val_metrics, "nn_corr@k")

        # --- Outer: refit on fit+val, evaluate on test ---
        refit_fitted = np.array(
            [
                alphas[cell] is not None and int(masks[refit_idx, cell].sum()) >= 2
                for cell in range(responses.shape[1])
            ]
        )
        refit_predictions = predict(x_refit_all, refit_idx, refit_idx)
        test_predictions_all = predict(x_refit_all, refit_idx, test_idx)
        x_test_block, x_test_valid = unit_normalize(x_refit_all[test_idx])
        h_test_block, h_test_valid, fitted_cells = build_block(
            test_predictions_all, refit_predictions, selected_version, groups, refit_fitted
        )
        common_valid = x_test_valid & h_test_valid
        excluded_total += int((~common_valid).sum())

        test_responses_all = responses[test_idx]
        test_masks_all = masks[test_idx]
        test_cities_all = city_array[test_idx]

        # Full raw pool: lambda = 0 only needs the feature block.
        raw_full = fuse(x_test_block, h_test_block, 0.0)[x_test_valid]
        raw_full_metrics = retrieval_block(
            raw_full,
            test_responses_all[x_test_valid],
            test_masks_all[x_test_valid],
            family_columns,
            args.k,
        )
        outer["raw_full"].append(metric_value(raw_full_metrics, "nn_corr@k"))
        outer_baseline["raw_full"].append(metric_value(raw_full_metrics, "baseline_corr"))

        if not common_valid.any():
            fold_records.append(
                {
                    "fold": fold,
                    "fit_units": int(len(fit_idx)),
                    "val_units": int(len(val_idx)),
                    "test_units": int(len(test_idx)),
                    "selected_version": selected_version,
                    "selected_lambda": float(selected_lambda),
                    "lambda_zero_selected": bool(selected_lambda == 0.0),
                    "val_nn_corr_selected": val_score,
                    "val_nn_corr_raw": raw_val_value,
                    "fitted_cells": fitted_cells,
                    "excluded_from_common": int((~common_valid).sum()),
                    "per_cell_ridge": [
                        float(alpha) if alpha is not None else None for alpha in alphas
                    ],
                }
            )
            continue

        x_common = x_test_block[common_valid]
        h_common = h_test_block[common_valid]
        test_responses = test_responses_all[common_valid]
        test_masks = test_masks_all[common_valid]
        test_city_array = test_cities_all[common_valid]

        embeddings = {
            "raw_common": fuse(x_common, h_common, 0.0),
            "h_only": fuse(x_common, h_common, 1.0),
            "fusion": fuse(x_common, h_common, selected_lambda),
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
                fuse(x_common, h_common, lam),
                test_responses,
                test_masks,
                family_columns,
                args.k,
            )
            curve[str(lam)] = metric_value(metrics, "nn_corr@k")
        lambda_curve[str(fold)] = curve

        for name, embedding in (
            ("raw", embeddings["raw_common"]),
            ("fusion", embeddings["fusion"]),
        ):
            metrics = retrieval_block(
                embedding,
                test_responses,
                test_masks,
                family_columns,
                args.k,
                city_keys=test_city_array,
                exclude_same_city=True,
            )
            no_same_city[name].append(metric_value(metrics, "nn_corr@k"))

        city_deltas, macro = per_anchor_city_deltas(
            embeddings["raw_common"],
            embeddings["fusion"],
            test_responses,
            test_masks,
            test_city_array,
            family_columns,
            args.k,
        )
        city_macro_overall.extend(macro)
        for city, entry in city_deltas.items():
            per_city_all[city] = entry
        _, macro_no_same = per_anchor_city_deltas(
            embeddings["raw_common"],
            embeddings["fusion"],
            test_responses,
            test_masks,
            test_city_array,
            family_columns,
            args.k,
            exclude_same_city=True,
        )
        city_macro_no_same_city.extend(macro_no_same)

        for outcome, columns in groups.items():
            for name, embedding in (
                ("raw", embeddings["raw_common"]),
                ("h_only", embeddings["h_only"]),
                ("fusion", embeddings["fusion"]),
            ):
                metrics = retrieval_block(
                    embedding, test_responses, test_masks, columns, args.k
                )
                per_outcome[outcome][name].append(metric_value(metrics, "nn_corr@k"))
            _, outcome_macro = per_anchor_city_deltas(
                embeddings["raw_common"],
                embeddings["fusion"],
                test_responses,
                test_masks,
                test_city_array,
                columns,
                args.k,
            )
            per_outcome_city_macro[outcome].extend(outcome_macro)

        baseline_means, baseline_signs = baseline_from_training(
            responses, masks, refit_idx
        )
        amplitude_records["overall"].append(
            amplitude_direction(
                test_predictions_all,
                test_responses_all,
                test_masks_all,
                family_columns,
                baseline_means,
                baseline_signs,
            )
        )
        for outcome, columns in groups.items():
            amplitude_records[outcome].append(
                amplitude_direction(
                    test_predictions_all,
                    test_responses_all,
                    test_masks_all,
                    columns,
                    baseline_means,
                    baseline_signs,
                )
            )

        reference_full = retrieval_metrics(
            torch.from_numpy(x_refit_all[test_idx][x_test_valid]).float(),
            torch.from_numpy(test_responses_all[x_test_valid][:, family_columns]).float(),
            torch.from_numpy(test_masks_all[x_test_valid][:, family_columns]),
            k=args.k,
        )["overall"]
        reference_common = retrieval_metrics(
            torch.from_numpy(x_refit_all[test_idx][common_valid]).float(),
            torch.from_numpy(test_responses[:, family_columns]).float(),
            torch.from_numpy(test_masks[:, family_columns]),
            k=args.k,
        )["overall"]
        full_value = metric_value(reference_full, "nn_corr@k")
        common_value = metric_value(reference_common, "nn_corr@k")
        zero_full = outer["raw_full"][-1]
        zero_common = outer["raw_common"][-1]
        if full_value is not None and isinstance(zero_full, (int, float)):
            equivalence_full.append(abs(full_value - float(zero_full)))
        if common_value is not None and isinstance(zero_common, (int, float)):
            equivalence_common.append(abs(common_value - float(zero_common)))

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
                "val_nn_corr_raw": raw_val_value,
                "fitted_cells": fitted_cells,
                "excluded_from_common": int((~common_valid).sum()),
                "per_cell_ridge": [
                    float(alpha) if alpha is not None else None for alpha in alphas
                ],
            }
        )
        print(
            f"fold {fold}: fit {len(fit_idx)} / val {len(val_idx)} / test {len(test_idx)} | "
            f"selected {selected_version}, lambda={selected_lambda:g} | "
            f"val nn_corr {val_score:.4f} "
            f"(raw {raw_val_value if raw_val_value is not None else float('nan'):.4f})"
        )

    macro_mean, macro_low, macro_high = bootstrap_mean_ci(
        city_macro_overall, n_boot=args.bootstrap
    )
    no_same_mean, no_same_low, no_same_high = bootstrap_mean_ci(
        city_macro_no_same_city, n_boot=args.bootstrap
    )
    per_outcome_summary: dict[str, object] = {}
    for outcome, outcome_entry in per_outcome.items():
        macro_values = per_outcome_city_macro[outcome]
        mean_value, low, high = bootstrap_mean_ci(macro_values, n_boot=args.bootstrap)
        per_outcome_summary[outcome] = {
            **{
                name: {"nn_corr_mean": mean_of(outcome_entry[name])}
                for name in ("raw", "h_only", "fusion")
            },
            "city_macro": {
                "n_cities": len(macro_values),
                "mean_delta": mean_value,
                "bootstrap_95_ci": [low, high],
            },
        }
    amplitude_summary = {
        "overall": {
            "relative_sq_error_reduction_pooled_mean": mean_of(
                [
                    record.get("relative_sq_error_reduction_pooled")
                    for record in amplitude_records["overall"]
                ]
            ),
            "relative_sq_error_reduction_cell_macro_mean": mean_of(
                [
                    record.get("relative_sq_error_reduction_cell_macro")
                    for record in amplitude_records["overall"]
                ]
            ),
            "direction_accuracy_pooled_mean": mean_of(
                [
                    record.get("direction_accuracy_pooled")
                    for record in amplitude_records["overall"]
                ]
            ),
            "direction_baseline_pooled_mean": mean_of(
                [
                    record.get("direction_baseline_pooled")
                    for record in amplitude_records["overall"]
                ]
            ),
            "fitted_cells_mean": mean_of(
                [record.get("fitted_cells") for record in amplitude_records["overall"]]
            ),
        },
        "per_outcome": {
            outcome: {
                "relative_sq_error_reduction_pooled_mean": mean_of(
                    [
                        record.get("relative_sq_error_reduction_pooled")
                        for record in records
                    ]
                ),
                "direction_accuracy_pooled_mean": mean_of(
                    [
                        record.get("direction_accuracy_pooled")
                        for record in records
                    ]
                ),
                "direction_baseline_pooled_mean": mean_of(
                    [
                        record.get("direction_baseline_pooled")
                        for record in records
                    ]
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
            "min_fitted_cells": MIN_FITTED_CELLS,
        },
        "city_folds": {city: int(assignment[city]) for city in sorted(assignment)},
        "fold_records": fold_records,
        "lambda_zero_share": lambda_zero_folds / max(args.folds, 1),
        "lambda_zero_equivalence": {
            "full_pool_max_abs_diff": max(equivalence_full) if equivalence_full else None,
            "common_pool_max_abs_diff": (
                max(equivalence_common) if equivalence_common else None
            ),
        },
        "excluded_from_common_total": excluded_total,
        "overall": {
            name: {
                "nn_corr_mean": mean_of(outer[name]),
                "baseline_mean": mean_of(outer_baseline[name]),
            }
            for name in ("raw_full", "raw_common", "h_only", "fusion")
        },
        "no_same_city": {
            "nn_corr": {
                name: {"nn_corr_mean": mean_of(values)}
                for name, values in no_same_city.items()
            },
            "city_macro": {
                "n_cities": len(city_macro_no_same_city),
                "mean_delta": no_same_mean,
                "bootstrap_95_ci": [no_same_low, no_same_high],
            },
        },
        "lambda_curve_diagnostic": lambda_curve,
        "per_outcome": per_outcome_summary,
        "city_macro": {
            "n_cities": len(city_macro_overall),
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
