"""GPU Abadie--Imbens ATT matching with bias correction and analytic variance.

This is a focused port of the formal project contract implemented by
``Matching::Match`` 4.10-15: ATT, matching with replacement, ties retained,
Mahalanobis ``Weight=2``, bias adjustment on ``Z``, and ``Var.calc > 0``.
The pair search and the same-treatment variance-neighbour searches run on the
configured torch device; the small weighted regressions and scalar variance
assembly remain in NumPy float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .contracts import EstimatorProvenance
from .inference import InferenceResult, inference_from_standard_error
from .matching import CommonSupportError, _topk_mahalanobis
from .runtime import RuntimeConfig, TorchRuntime

_R_NUMERICAL_TOLERANCE = float(np.sqrt(np.finfo(np.float64).eps))
_MATCHING_CPP_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class AbadieImbensInput:
    """One complete cohort risk set for the formal ATT estimator."""

    outcome: npt.NDArray[np.float64]
    treated: npt.NDArray[np.bool_]
    covariates: npt.NDArray[np.float64]
    bias_covariates: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        outcome = np.asarray(self.outcome, dtype=np.float64)
        treated = np.asarray(self.treated, dtype=bool)
        covariates = np.asarray(self.covariates, dtype=np.float64)
        bias = (
            covariates.copy()
            if self.bias_covariates is None
            else np.asarray(self.bias_covariates, dtype=np.float64)
        )
        if outcome.ndim != 1 or treated.shape != outcome.shape:
            raise ValueError("outcome and treated must be aligned one-dimensional arrays")
        if covariates.ndim != 2 or covariates.shape[0] != outcome.size:
            raise ValueError("covariates must be observations by features")
        if bias.ndim != 2 or bias.shape[0] != outcome.size:
            raise ValueError("bias_covariates must be observations by features")
        if covariates.shape[1] < 1 or bias.shape[1] < 1:
            raise ValueError("matching and bias adjustment require at least one feature")
        if not np.all(np.isfinite(outcome)):
            raise ValueError("formal matching outcomes must all be finite")
        if not np.all(np.isfinite(covariates)) or not np.all(np.isfinite(bias)):
            raise ValueError("formal matching covariates must all be finite")
        if not treated.any() or treated.all():
            raise ValueError("formal matching requires treated and control observations")
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "treated", treated)
        object.__setattr__(self, "covariates", covariates)
        object.__setattr__(self, "bias_covariates", bias)


@dataclass(frozen=True, slots=True)
class AbadieImbensConfig:
    matches: int = 1
    variance_neighbors: int = 1
    bias_adjust: bool = True
    enforce_common_support: bool = True
    distance_tolerance: float = 0.0
    cpp_tolerance: float = _MATCHING_CPP_TOLERANCE
    numerical_tolerance: float = _R_NUMERICAL_TOLERANCE
    weight_matrix_eigen_cutoff: float = 1e-7
    chunk_size: int | None = None
    query_batch_size: int = 1024

    def __post_init__(self) -> None:
        if self.matches < 1:
            raise ValueError("matches must be positive")
        if self.variance_neighbors < 1:
            raise ValueError("variance_neighbors must be positive")
        if self.distance_tolerance != 0:
            raise ValueError("the frozen formal contract requires distance_tolerance=0")
        if self.cpp_tolerance < 0 or self.numerical_tolerance <= 0:
            raise ValueError("matching tolerances must be non-negative/positive")
        if self.query_batch_size < 1:
            raise ValueError("query_batch_size must be positive")


@dataclass(frozen=True, slots=True)
class AbadieImbensResult:
    estimate: float
    analytic_standard_error: float
    pair_standard_error: float
    unadjusted_estimate: float
    inference: InferenceResult
    treated_indices: npt.NDArray[np.int64]
    control_indices: npt.NDArray[np.int64]
    pair_weights: npt.NDArray[np.float64]
    bias_coefficients: npt.NDArray[np.float64]
    reuse_count: npt.NDArray[np.float64]
    reuse_squared_count: npt.NDArray[np.float64]
    conditional_variances: npt.NDArray[np.float64]
    active_matching_features: npt.NDArray[np.bool_]
    provenance: EstimatorProvenance


def _matching_metric(
    covariates: npt.NDArray[np.float64],
    config: AbadieImbensConfig,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Reproduce ``RmatchLoop``'s standardisation and ``Weight=2`` matrix."""
    count = covariates.shape[0]
    mean = np.mean(covariates, axis=0)
    second_moment_sum = np.sum(np.square(covariates), axis=0)
    variance = np.maximum(config.numerical_tolerance, second_moment_sum) / count
    variance = variance - np.square(mean)
    if np.any(variance <= 0) or not np.all(np.isfinite(variance)):
        raise ValueError("matching covariates have invalid standardized variance")
    scale = np.sqrt(variance) * np.sqrt(count / (count - 1))
    scale = np.maximum(scale, config.numerical_tolerance)
    standardized = (covariates - mean) / scale
    moment = standardized.T @ standardized / count
    eigenvalues = np.linalg.eigvalsh((moment + moment.T) / 2)
    if float(np.min(eigenvalues)) > config.weight_matrix_eigen_cutoff:
        metric = np.linalg.inv(moment)
    else:
        metric = np.eye(covariates.shape[1], dtype=np.float64)
    if float(np.min(np.linalg.eigvalsh((metric + metric.T) / 2))) < config.numerical_tolerance:
        metric = metric + np.eye(metric.shape[0]) * config.numerical_tolerance
    return standardized, metric


def _matches_with_ties(
    targets: npt.NDArray[np.float64],
    donors: npt.NDArray[np.float64],
    metric: npt.NDArray[np.float64],
    *,
    matches: int,
    config: AbadieImbensConfig,
    runtime: TorchRuntime,
    excluded_indices: npt.NDArray[np.int64] | None = None,
) -> list[npt.NDArray[np.int64]]:
    """Return every donor within the C++ kth-distance tie boundary."""
    if donors.shape[0] < matches:
        raise ValueError("fewer donors than requested matches")
    result: list[npt.NDArray[np.int64]] = []
    donor_chunk = config.chunk_size or runtime.config.chunk_size
    for query_start in range(0, targets.shape[0], config.query_batch_size):
        query_stop = min(query_start + config.query_batch_size, targets.shape[0])
        local_targets = targets[query_start:query_stop]
        local_excluded = (
            None
            if excluded_indices is None
            else excluded_indices[query_start:query_stop]
        )
        _, top_distances = _topk_mahalanobis(
            local_targets,
            donors,
            metric,
            k=matches,
            runtime=runtime,
            chunk_size=donor_chunk,
            excluded_indices=local_excluded,
        )
        boundaries = np.square(top_distances[:, -1])
        local_matches: list[list[int]] = [[] for _ in range(local_targets.shape[0])]
        target_tensor = runtime.tensor(local_targets)
        metric_tensor = runtime.tensor(metric)
        transformed_targets = target_tensor @ metric_tensor
        target_quadratic = (transformed_targets * target_tensor).sum(
            dim=1, keepdim=True
        )
        boundary_tensor = runtime.tensor(
            boundaries + config.distance_tolerance + config.cpp_tolerance
        )
        for donor_start in range(0, donors.shape[0], donor_chunk):
            donor_stop = min(donor_start + donor_chunk, donors.shape[0])
            donor_tensor = runtime.tensor(donors[donor_start:donor_stop])
            donor_transformed = donor_tensor @ metric_tensor
            donor_quadratic = (donor_transformed * donor_tensor).sum(dim=1)
            squared = target_quadratic + donor_quadratic[None, :] - 2 * (
                transformed_targets @ donor_tensor.T
            )
            squared = squared.clamp_min(0)
            if local_excluded is not None:
                local = local_excluded - donor_start
                rows = np.flatnonzero((local >= 0) & (local < donor_stop - donor_start))
                if rows.size:
                    row_tensor = runtime.torch.as_tensor(
                        rows, dtype=runtime.torch.long, device=runtime.device
                    )
                    column_tensor = runtime.torch.as_tensor(
                        local[rows], dtype=runtime.torch.long, device=runtime.device
                    )
                    squared[row_tensor, column_tensor] = float("inf")
            selected = (squared <= boundary_tensor[:, None]).detach().cpu().numpy()
            for row in range(selected.shape[0]):
                positions = np.flatnonzero(selected[row]) + donor_start
                local_matches[row].extend(positions.tolist())
        for values in local_matches:
            if len(values) < matches:
                raise RuntimeError("matching tie pass lost a kth-distance donor")
            result.append(np.asarray(values, dtype=np.int64))
    return result


def _bias_regression(
    outcome: npt.NDArray[np.float64],
    bias_covariates: npt.NDArray[np.float64],
    control_weights: npt.NDArray[np.float64],
    tolerance: float,
) -> npt.NDArray[np.float64]:
    design = np.column_stack(
        (np.ones(outcome.size, dtype=np.float64), bias_covariates)
    )
    root_weight = np.sqrt(control_weights)
    weighted_design = design * root_weight[:, None]
    weighted_outcome = outcome * root_weight
    cross = weighted_design.T @ weighted_design
    ridge = 0.0
    if float(np.min(np.linalg.eigvalsh((cross + cross.T) / 2))) <= tolerance:
        column_sd = np.std(weighted_design, axis=0, ddof=1)
        ridge = tolerance * float(np.max(column_sd))
    adjusted = cross + np.eye(cross.shape[0]) * ridge
    right = weighted_design.T @ weighted_outcome
    try:
        coefficients = np.linalg.solve(adjusted, right)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(adjusted) @ right
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError("unable to calculate the formal matching bias adjustment")
    return coefficients[1:]


def _conditional_variances(
    outcome: npt.NDArray[np.float64],
    treated: npt.NDArray[np.bool_],
    standardized: npt.NDArray[np.float64],
    metric: npt.NDArray[np.float64],
    *,
    config: AbadieImbensConfig,
    runtime: TorchRuntime,
) -> npt.NDArray[np.float64]:
    result = np.full(outcome.shape, np.nan, dtype=np.float64)
    for group_value in (False, True):
        group = np.flatnonzero(treated == group_value).astype(np.int64)
        if group.size <= config.variance_neighbors:
            raise ValueError(
                "each treatment arm needs more observations than variance_neighbors"
            )
        neighbours = _matches_with_ties(
            standardized[group],
            standardized[group],
            metric,
            matches=config.variance_neighbors,
            config=config,
            runtime=runtime,
            excluded_indices=np.arange(group.size, dtype=np.int64),
        )
        for local_index, local_neighbours in enumerate(neighbours):
            values = np.concatenate(
                (
                    outcome[group[local_index] : group[local_index] + 1],
                    outcome[group[local_neighbours]],
                )
            )
            result[group[local_index]] = float(np.var(values, ddof=1))
    if not np.all(np.isfinite(result)):
        raise RuntimeError("analytic matching variance contains invalid local variances")
    return result


def fit_abadie_imbens(
    data: AbadieImbensInput,
    *,
    config: AbadieImbensConfig | None = None,
    runtime: TorchRuntime | None = None,
) -> AbadieImbensResult:
    """Estimate the formal bias-adjusted ATT and Abadie--Imbens variance."""
    config = config or AbadieImbensConfig()
    runtime = runtime or TorchRuntime(RuntimeConfig())
    feature_sd = np.std(data.covariates, axis=0, ddof=1)
    active = np.isfinite(feature_sd) & (feature_sd > config.numerical_tolerance)
    if not active.any():
        raise ValueError("all formal matching covariates have zero variance")
    covariates = data.covariates[:, active]
    bias_covariates = data.bias_covariates
    assert bias_covariates is not None
    treated_rows = np.flatnonzero(data.treated).astype(np.int64)
    control_rows = np.flatnonzero(~data.treated).astype(np.int64)
    if config.enforce_common_support:
        lower = np.min(covariates[control_rows], axis=0)
        upper = np.max(covariates[control_rows], axis=0)
        supported = np.all(
            (covariates[treated_rows] >= lower) & (covariates[treated_rows] <= upper),
            axis=1,
        )
        if not supported.all():
            unsupported = treated_rows[~supported].tolist()
            raise CommonSupportError(
                f"treated observations outside explicit donor support: {unsupported}"
            )
    if treated_rows.size <= bias_covariates.shape[1] and config.bias_adjust:
        raise ValueError("bias adjustment is not identified by the treated cohort")
    if control_rows.size <= config.variance_neighbors:
        raise ValueError("analytic variance is not identified in the control cohort")

    standardized, metric = _matching_metric(covariates, config)
    matched_controls = _matches_with_ties(
        standardized[treated_rows],
        standardized[control_rows],
        metric,
        matches=min(config.matches, control_rows.size),
        config=config,
        runtime=runtime,
    )
    pair_treated: list[int] = []
    pair_controls: list[int] = []
    pair_weights: list[float] = []
    for treated_row, local_controls in zip(treated_rows, matched_controls, strict=True):
        weight = 1.0 / local_controls.size
        pair_treated.extend([int(treated_row)] * local_controls.size)
        pair_controls.extend(control_rows[local_controls].tolist())
        pair_weights.extend([weight] * local_controls.size)
    treated_index = np.asarray(pair_treated, dtype=np.int64)
    control_index = np.asarray(pair_controls, dtype=np.int64)
    weights = np.asarray(pair_weights, dtype=np.float64)

    reuse = np.zeros(data.outcome.size, dtype=np.float64)
    reuse_squared = np.zeros(data.outcome.size, dtype=np.float64)
    np.add.at(reuse, control_index, weights)
    np.add.at(reuse_squared, control_index, np.square(weights))
    unit_effect = np.zeros(data.outcome.size, dtype=np.float64)
    unit_bias_gap = np.zeros(
        (data.outcome.size, bias_covariates.shape[1]), dtype=np.float64
    )
    for treated_row in treated_rows:
        pair_mask = treated_index == treated_row
        local_weights = weights[pair_mask]
        local_controls = control_index[pair_mask]
        unit_effect[treated_row] = data.outcome[treated_row] - np.sum(
            data.outcome[local_controls] * local_weights
        )
        unit_bias_gap[treated_row] = bias_covariates[treated_row] - np.sum(
            bias_covariates[local_controls] * local_weights[:, None], axis=0
        )

    bias_coefficients = np.zeros(bias_covariates.shape[1], dtype=np.float64)
    if config.bias_adjust:
        control_regression_weights = (~data.treated).astype(np.float64) * reuse
        bias_coefficients = _bias_regression(
            data.outcome,
            bias_covariates,
            control_regression_weights,
            config.numerical_tolerance,
        )
    adjusted_control = data.outcome[control_index] + (
        bias_covariates[treated_index] - bias_covariates[control_index]
    ) @ bias_coefficients
    adjusted_pair_effect = data.outcome[treated_index] - adjusted_control
    estimate = float(np.sum(adjusted_pair_effect * weights) / np.sum(weights))
    raw_pair_effect = data.outcome[treated_index] - data.outcome[control_index]
    unadjusted = float(np.sum(raw_pair_effect * weights) / np.sum(weights))
    pair_variance = float(
        np.sum(np.square(raw_pair_effect - unadjusted) * weights)
        / np.square(np.sum(weights))
    )
    pair_standard_error = float(np.sqrt(max(pair_variance, 0.0)))

    conditional = _conditional_variances(
        data.outcome,
        data.treated,
        standardized,
        metric,
        config=config,
        runtime=runtime,
    )
    treated_weight = data.treated.astype(np.float64)
    control_weight = (~data.treated).astype(np.float64)
    treated_count = float(treated_rows.size)
    population_component = np.sum(
        conditional
        * control_weight
        * (np.square(reuse) - reuse_squared)
    ) / np.square(treated_count)
    adjusted_unit_effect = unit_effect - unit_bias_gap @ bias_coefficients
    heterogeneity_component = np.sum(
        treated_weight * np.square(adjusted_unit_effect - estimate)
    ) / np.square(treated_count)
    analytic_variance = float(population_component + heterogeneity_component)
    if analytic_variance < -config.numerical_tolerance:
        raise RuntimeError("Abadie--Imbens analytic variance is negative")
    analytic_standard_error = float(np.sqrt(max(analytic_variance, 0.0)))
    inference = inference_from_standard_error(
        np.array([estimate]),
        np.array([analytic_standard_error]),
        method="abadie_imbens_bias_adjusted_analytic",
        requested_repetitions=0,
        valid_repetitions=np.array([0], dtype=np.int64),
    )
    provenance = EstimatorProvenance(
        estimator="abadie_imbens_matching",
        backend="pytorch_numpy",
        formal_eligible=False,
        numerical_policy=runtime.metadata()
        | {
            "estimand": "ATT",
            "matches": config.matches,
            "replace": True,
            "ties": True,
            "weight": 2,
            "bias_adjust": config.bias_adjust,
            "variance_neighbors": config.variance_neighbors,
            "distance_tolerance": config.distance_tolerance,
            "cpp_tolerance": config.cpp_tolerance,
            "query_batch_size": config.query_batch_size,
        },
        notes=(
            "pair and same-treatment neighbour searches execute on the torch device",
            "formal eligibility remains gated on fixture and task-wise R parity",
        ),
    )
    return AbadieImbensResult(
        estimate=estimate,
        analytic_standard_error=analytic_standard_error,
        pair_standard_error=pair_standard_error,
        unadjusted_estimate=unadjusted,
        inference=inference,
        treated_indices=treated_index,
        control_indices=control_index,
        pair_weights=weights,
        bias_coefficients=bias_coefficients,
        reuse_count=reuse,
        reuse_squared_count=reuse_squared,
        conditional_variances=conditional,
        active_matching_features=active,
        provenance=provenance,
    )
