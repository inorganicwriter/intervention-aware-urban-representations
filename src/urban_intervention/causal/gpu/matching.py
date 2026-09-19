"""Chunked GPU Mahalanobis matching and deterministic placebo calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .contracts import EstimatorProvenance, MatchingInput, MatchingResult
from .runtime import RuntimeConfig, TorchRuntime


class CommonSupportError(ValueError):
    """Raised when the treated target lies outside donor common support."""


@dataclass(frozen=True, slots=True)
class MatchingConfig:
    candidates: int = 5
    placebo_sample: int = 200
    placebo_quantile: float = 0.95
    covariance_cutoff: float = float(np.sqrt(np.finfo(np.float64).eps))
    chunk_size: int | None = None
    distance_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.candidates < 1:
            raise ValueError("candidates must be positive")
        if self.placebo_sample < 3:
            raise ValueError("placebo_sample must be at least three")
        if not 0 < self.placebo_quantile < 1:
            raise ValueError("placebo_quantile must be in (0, 1)")
        if self.distance_tolerance != 0:
            raise ValueError("GPU formal matching requires exact distance_tolerance=0")


def stable_covariance_inverse(
    matrix: npt.ArrayLike,
    *,
    cutoff_scale: float = float(np.sqrt(np.finfo(np.float64).eps)),
) -> tuple[npt.NDArray[np.float64], int]:
    """R-compatible symmetric eigenvalue pseudo-inverse of sample covariance."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("covariance matrix input needs at least two rows")
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2)
    cutoff = np.max(np.abs(eigenvalues)) * cutoff_scale
    keep = eigenvalues > cutoff
    if not keep.any():
        raise ValueError("training covariance has zero numerical rank")
    inverse = (eigenvectors[:, keep] / eigenvalues[keep]) @ eigenvectors[:, keep].T
    return inverse, int(keep.sum())


def _topk_mahalanobis(
    targets: npt.NDArray[np.float64],
    donors: npt.NDArray[np.float64],
    inverse: npt.NDArray[np.float64],
    *,
    k: int,
    runtime: TorchRuntime,
    chunk_size: int,
    excluded_indices: npt.NDArray[np.int64] | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Find exact global top-k distances without materialising Q×N×D deltas."""
    torch = runtime.torch
    target_tensor = runtime.tensor(targets)
    inverse_tensor = runtime.tensor(inverse)
    target_transformed = target_tensor @ inverse_tensor
    target_quadratic = (target_transformed * target_tensor).sum(dim=1, keepdim=True)
    best_distances = target_tensor.new_full((targets.shape[0], 0), float("inf"))
    best_indices = torch.empty(
        (targets.shape[0], 0),
        dtype=torch.long,
        device=runtime.device,
    )
    keep_count = min(k, donors.shape[0])
    for start in range(0, donors.shape[0], chunk_size):
        stop = min(start + chunk_size, donors.shape[0])
        donor_chunk = runtime.tensor(donors[start:stop])
        donor_transformed = donor_chunk @ inverse_tensor
        donor_quadratic = (donor_transformed * donor_chunk).sum(dim=1)
        squared = target_quadratic + donor_quadratic[None, :] - 2 * (
            target_transformed @ donor_chunk.T
        )
        squared = squared.clamp_min(0)
        if excluded_indices is not None:
            local = excluded_indices - start
            rows = np.flatnonzero((local >= 0) & (local < stop - start))
            if rows.size:
                row_tensor = torch.as_tensor(rows, dtype=torch.long, device=runtime.device)
                col_tensor = torch.as_tensor(local[rows], dtype=torch.long, device=runtime.device)
                squared[row_tensor, col_tensor] = float("inf")
        indices = torch.arange(start, stop, dtype=torch.long, device=runtime.device)
        indices = indices.expand(targets.shape[0], -1)
        combined_distances = torch.cat((best_distances, squared), dim=1)
        combined_indices = torch.cat((best_indices, indices), dim=1)
        selected_distances, positions = torch.topk(
            combined_distances,
            k=min(keep_count, combined_distances.shape[1]),
            dim=1,
            largest=False,
            sorted=False,
        )
        selected_indices = combined_indices.gather(1, positions)
        # ``torch.topk`` does not promise which elements are returned when the
        # kth distance is tied.  Repair only those boundary rows so exact ties
        # are resolved by original donor index without sorting every Q×chunk
        # distance matrix.
        for row in range(combined_distances.shape[0]):
            boundary = selected_distances[row].max()
            tied = combined_distances[row] == boundary
            if int(tied.sum()) <= 1:
                continue
            below = combined_distances[row] < boundary
            below_distances = combined_distances[row, below]
            below_indices = combined_indices[row, below]
            needed = selected_distances.shape[1] - below_distances.numel()
            tied_indices = combined_indices[row, tied]
            tied_indices = tied_indices.sort().values[:needed]
            selected_distances[row] = torch.cat(
                (below_distances, boundary.expand(needed))
            )
            selected_indices[row] = torch.cat((below_indices, tied_indices))
        best_distances = selected_distances
        best_indices = selected_indices
    distance_np = np.sqrt(best_distances.detach().cpu().numpy())
    index_np = best_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    # The formal R contract sets distance.tolerance=0. Exact equal distances
    # are resolved by original donor order for deterministic reruns.
    for row in range(index_np.shape[0]):
        order = np.lexsort((index_np[row], distance_np[row]))
        index_np[row] = index_np[row, order]
        distance_np[row] = distance_np[row, order]
    return index_np, distance_np


def _active_columns(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    standard_deviation = np.std(matrix, axis=0, ddof=1)
    return np.isfinite(standard_deviation) & (
        standard_deviation > np.sqrt(np.finfo(np.float64).eps)
    )


def _static_choice(
    target: npt.NDArray[np.float64],
    donors: npt.NDArray[np.float64],
    candidates: npt.NDArray[np.int64],
    inverse: npt.NDArray[np.float64] | None,
) -> int:
    if inverse is None:
        return int(candidates[0])
    delta = donors[candidates] - target
    distances = np.sqrt(np.maximum(np.sum((delta @ inverse) * delta, axis=1), 0))
    return int(candidates[int(np.argmin(distances))])


def _placebo_calibration(
    training: npt.NDArray[np.float64],
    holdout: npt.NDArray[np.float64],
    static: npt.NDArray[np.float64] | None,
    *,
    config: MatchingConfig,
    runtime: TorchRuntime,
) -> tuple[dict[str, float], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if training.shape[0] < 3:
        raise ValueError("at least three donors are required for placebo calibration")
    inverse, covariance_rank = stable_covariance_inverse(
        training,
        cutoff_scale=config.covariance_cutoff,
    )
    holdout_sd = np.std(holdout, axis=0, ddof=1)
    informative = np.isfinite(holdout_sd) & (
        holdout_sd > np.sqrt(np.finfo(np.float64).eps)
    )
    if not informative.any():
        raise ValueError("all holdout features have near-zero donor variance")
    static_inverse = None
    static_active = None
    if static is not None:
        static_active = _active_columns(static)
        if static_active.any():
            static_inverse, _ = stable_covariance_inverse(
                static[:, static_active],
                cutoff_scale=config.covariance_cutoff,
            )
        else:
            static_active = None
    sample_n = min(config.placebo_sample, training.shape[0])
    sampled = np.unique(
        np.rint(np.linspace(0, training.shape[0] - 1, num=sample_n)).astype(np.int64)
    )
    candidate_indices, candidate_distances = _topk_mahalanobis(
        training[sampled],
        training,
        inverse,
        k=min(config.candidates, training.shape[0] - 1),
        runtime=runtime,
        chunk_size=config.chunk_size or runtime.config.chunk_size,
        excluded_indices=sampled,
    )
    metrics = np.empty((sampled.size, 3), dtype=np.float64)
    for row, pseudo_target in enumerate(sampled):
        candidates = candidate_indices[row]
        chosen = int(candidates[0])
        if static_inverse is not None and static_active is not None and static is not None:
            chosen = _static_choice(
                static[pseudo_target, static_active],
                static[:, static_active],
                candidates,
                static_inverse,
            )
        candidate_position = int(np.flatnonzero(candidates == chosen)[0])
        gap = (holdout[pseudo_target] - holdout[chosen]) / holdout_sd
        gap = gap[informative]
        metrics[row] = (
            candidate_distances[row, candidate_position],
            np.sqrt(np.mean(np.square(gap))),
            np.max(np.abs(gap)),
        )
    quantile = config.placebo_quantile
    thresholds = {
        "training_distance": float(np.quantile(metrics[:, 0], quantile, method="median_unbiased")),
        "holdout_rms_standardized_gap": float(
            np.quantile(metrics[:, 1], quantile, method="median_unbiased")
        ),
        "holdout_max_abs_standardized_gap": float(
            np.quantile(metrics[:, 2], quantile, method="median_unbiased")
        ),
        "calibration_pairs": float(sampled.size),
        "quantile_probability": quantile,
        "covariance_rank": float(covariance_rank),
    }
    return thresholds, inverse, holdout_sd


def fit_matching(
    data: MatchingInput,
    *,
    config: MatchingConfig | None = None,
    runtime: TorchRuntime | None = None,
) -> MatchingResult:
    """Run M-candidate outcome matching, static refinement, and q95 gate."""
    config = config or MatchingConfig()
    runtime = runtime or TorchRuntime(RuntimeConfig())
    combined = np.vstack((data.target, data.donors))
    active = _active_columns(combined)
    if not active.any():
        raise ValueError("all pre-treatment matching covariates have zero variance")
    target = data.target[active]
    donors = data.donors[:, active]
    if data.support_feature_indices is None:
        support_original = np.arange(data.target.size)
    else:
        support_original = np.asarray(data.support_feature_indices, dtype=np.int64)
    active_original = np.flatnonzero(active)
    support_positions = np.flatnonzero(np.isin(active_original, support_original))
    if support_positions.size:
        lower = donors[:, support_positions].min(axis=0)
        upper = donors[:, support_positions].max(axis=0)
        if not np.all((target[support_positions] >= lower) & (target[support_positions] <= upper)):
            raise CommonSupportError("treated target is outside explicit donor common support")

    matching_inverse, _ = stable_covariance_inverse(
        np.vstack((target, donors)),
        cutoff_scale=config.covariance_cutoff,
    )
    candidate_indices, candidate_distances = _topk_mahalanobis(
        target[None, :],
        donors,
        matching_inverse,
        k=min(config.candidates, donors.shape[0]),
        runtime=runtime,
        chunk_size=config.chunk_size or runtime.config.chunk_size,
    )
    candidates = candidate_indices[0]
    distances = candidate_distances[0]
    static_inverse = None
    target_static = None
    donor_static = None
    if data.target_static is not None and data.donor_static is not None:
        static_active = _active_columns(data.donor_static)
        if static_active.any():
            target_static = data.target_static[static_active]
            donor_static = data.donor_static[:, static_active]
            static_inverse, _ = stable_covariance_inverse(
                donor_static,
                cutoff_scale=config.covariance_cutoff,
            )
    selected = _static_choice(target_static, donor_static, candidates, static_inverse) if (
        target_static is not None and donor_static is not None
    ) else int(candidates[0])
    selected_position = int(np.flatnonzero(candidates == selected)[0])

    training_distance = None
    holdout_rms = None
    holdout_max = None
    thresholds = None
    quality_passed = None
    if data.target_holdout is not None and data.donor_holdout is not None:
        thresholds, calibration_inverse, holdout_sd = _placebo_calibration(
            donors,
            data.donor_holdout,
            donor_static,
            config=config,
            runtime=runtime,
        )
        training_gap = target - donors[selected]
        training_distance = float(
            np.sqrt(max(float(training_gap @ calibration_inverse @ training_gap), 0))
        )
        informative = np.isfinite(holdout_sd) & (
            holdout_sd > np.sqrt(np.finfo(np.float64).eps)
        )
        gap = (data.target_holdout - data.donor_holdout[selected]) / holdout_sd
        gap = gap[informative]
        holdout_rms = float(np.sqrt(np.mean(np.square(gap))))
        holdout_max = float(np.max(np.abs(gap)))
        quality_passed = bool(
            training_distance <= thresholds["training_distance"]
            and holdout_rms <= thresholds["holdout_rms_standardized_gap"]
            and holdout_max <= thresholds["holdout_max_abs_standardized_gap"]
        )
    provenance = EstimatorProvenance(
        estimator="matching",
        backend="pytorch",
        formal_eligible=False,
        numerical_policy=runtime.metadata()
        | {
            "candidates": config.candidates,
            "distance_tolerance": config.distance_tolerance,
            "tie_policy": "distance_then_original_donor_index",
            "placebo_quantile_type": 8,
        },
        notes=("R Matching parity is required before production selection",),
    )
    return MatchingResult(
        donor_indices=candidates,
        distances=distances,
        support_count=int(support_positions.size),
        selected_index=selected,
        selected_distance=float(distances[selected_position]),
        training_distance=training_distance,
        holdout_rms_standardized_gap=holdout_rms,
        holdout_max_abs_standardized_gap=holdout_max,
        placebo_thresholds=thresholds,
        quality_passed=quality_passed,
        provenance=provenance,
    )
