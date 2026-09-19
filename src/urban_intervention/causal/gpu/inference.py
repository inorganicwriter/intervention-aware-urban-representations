"""Shared, estimator-agnostic frequentist inference contracts.

The numerical estimators return one effect path and a matrix of replicate
effect paths.  This module is deliberately small: it standardises the normal
approximation used by the locked R reference, records the number of usable
replicates at every period, and prevents downstream code from confusing a
requested bootstrap count with the count that actually converged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

NORMAL_975 = 1.959963984540054


def _two_sided_normal_pvalue(z_score: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return two-sided standard-normal p values without a SciPy dependency."""
    flat = np.asarray(z_score, dtype=np.float64).reshape(-1)
    values = np.fromiter(
        (math.erfc(abs(float(value)) / math.sqrt(2.0)) for value in flat),
        dtype=np.float64,
        count=flat.size,
    )
    return values.reshape(z_score.shape)


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """A complete uncertainty contract for one period-indexed effect path."""

    method: str
    estimate: npt.NDArray[np.float64]
    standard_error: npt.NDArray[np.float64]
    confidence_lower: npt.NDArray[np.float64]
    confidence_upper: npt.NDArray[np.float64]
    p_value: npt.NDArray[np.float64]
    valid_repetitions: npt.NDArray[np.int64]
    requested_repetitions: int
    replicate_estimates: npt.NDArray[np.float64] | None = None
    formal_validated: bool = False

    def __post_init__(self) -> None:
        estimate = np.asarray(self.estimate, dtype=np.float64)
        if estimate.ndim != 1:
            raise ValueError("inference estimate must be one-dimensional")
        for name in (
            "standard_error",
            "confidence_lower",
            "confidence_upper",
            "p_value",
            "valid_repetitions",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != estimate.shape:
                raise ValueError(f"{name} must match the estimate path")
        if self.requested_repetitions < 0:
            raise ValueError("requested_repetitions cannot be negative")
        valid = np.asarray(self.valid_repetitions, dtype=np.int64)
        if np.any(valid < 0) or np.any(valid > self.requested_repetitions):
            raise ValueError("valid_repetitions must be within the requested count")
        draws = self.replicate_estimates
        if draws is not None:
            draws = np.asarray(draws, dtype=np.float64)
            if draws.ndim != 2 or draws.shape[1] != estimate.size:
                raise ValueError("replicate_estimates must be repetitions by periods")
            if draws.shape[0] != self.requested_repetitions:
                raise ValueError("replicate row count must equal requested_repetitions")
            object.__setattr__(self, "replicate_estimates", draws)
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "standard_error", np.asarray(self.standard_error, dtype=np.float64))
        object.__setattr__(self, "confidence_lower", np.asarray(self.confidence_lower, dtype=np.float64))
        object.__setattr__(self, "confidence_upper", np.asarray(self.confidence_upper, dtype=np.float64))
        object.__setattr__(self, "p_value", np.asarray(self.p_value, dtype=np.float64))
        object.__setattr__(self, "valid_repetitions", valid)

    def to_frame(self, periods: list[Any] | tuple[Any, ...]) -> Any:
        """Return the canonical formal-inference table without importing pandas globally."""
        if len(periods) != self.estimate.size:
            raise ValueError("period labels must match the inference path")
        import pandas as pd

        return pd.DataFrame(
            {
                "period": periods,
                "effect": self.estimate,
                "standard_error": self.standard_error,
                "confidence_lower": self.confidence_lower,
                "confidence_upper": self.confidence_upper,
                "p_value": self.p_value,
                "inference_method": self.method,
                "requested_repetitions": self.requested_repetitions,
                "valid_repetitions": self.valid_repetitions,
                "formal_validated": self.formal_validated,
            }
        )


def inference_from_standard_error(
    estimate: npt.ArrayLike,
    standard_error: npt.ArrayLike,
    *,
    method: str,
    requested_repetitions: int,
    valid_repetitions: npt.ArrayLike,
    replicate_estimates: npt.ArrayLike | None = None,
    formal_validated: bool = False,
) -> InferenceResult:
    """Build normal CIs and p values using a precomputed estimator-specific SE."""
    point = np.asarray(estimate, dtype=np.float64)
    se = np.asarray(standard_error, dtype=np.float64)
    valid = np.asarray(valid_repetitions, dtype=np.int64)
    if point.ndim != 1 or se.shape != point.shape or valid.shape != point.shape:
        raise ValueError("estimate, standard_error and valid_repetitions must align")
    # ``requested_repetitions == 0`` denotes closed-form/analytic inference.
    # Replicate-based methods still require at least two usable draws.
    repetition_ok = (
        np.ones(point.shape, dtype=bool)
        if requested_repetitions == 0
        else valid >= 2
    )
    usable = np.isfinite(point) & np.isfinite(se) & (se >= 0) & repetition_ok
    lower = np.full_like(point, np.nan)
    upper = np.full_like(point, np.nan)
    p_value = np.full_like(point, np.nan)
    lower[usable] = point[usable] - NORMAL_975 * se[usable]
    upper[usable] = point[usable] + NORMAL_975 * se[usable]
    positive = usable & (se > 0)
    p_value[positive] = _two_sided_normal_pvalue(point[positive] / se[positive])
    zero = usable & (se == 0)
    p_value[zero & (point == 0)] = 1.0
    p_value[zero & (point != 0)] = 0.0
    draws = None if replicate_estimates is None else np.asarray(replicate_estimates, dtype=np.float64)
    return InferenceResult(
        method=method,
        estimate=point,
        standard_error=se,
        confidence_lower=lower,
        confidence_upper=upper,
        p_value=p_value,
        valid_repetitions=valid,
        requested_repetitions=requested_repetitions,
        replicate_estimates=draws,
        formal_validated=formal_validated,
    )


def jackknife_standard_error(
    estimate: npt.ArrayLike,
    leave_one_out_estimates: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Return fect-equivalent unit-jackknife SEs and valid refit counts.

    ``fect`` removes an invalid leave-one-out column before constructing
    pseudo-values.  Consequently, each period must use the number of finite
    refits for that period as ``N``.  Counting an omitted treated unit whose
    refit is undefined inflates the standard error.
    """
    point = np.asarray(estimate, dtype=np.float64)
    draws = np.asarray(leave_one_out_estimates, dtype=np.float64)
    if point.ndim != 1 or draws.ndim != 2 or draws.shape[1] != point.size:
        raise ValueError("leave-one-out estimates must be refits by periods")
    valid = np.sum(np.isfinite(draws), axis=0).astype(np.int64)
    standard_error = np.full(point.shape, np.nan, dtype=np.float64)
    for period in np.flatnonzero(np.isfinite(point) & (valid >= 2)):
        local = draws[np.isfinite(draws[:, period]), period]
        count = local.size
        pseudo_values = point[period] * count - local * (count - 1)
        standard_error[period] = float(
            np.sqrt(np.var(pseudo_values, ddof=1) / count)
        )
    return standard_error, valid


def normal_inference_from_replicates(
    estimate: npt.ArrayLike,
    replicate_estimates: npt.ArrayLike,
    *,
    method: str,
    requested_repetitions: int | None = None,
    formal_validated: bool = False,
) -> InferenceResult:
    """Compute the locked normal-approximation inference from replicate paths."""
    point = np.asarray(estimate, dtype=np.float64)
    draws = np.asarray(replicate_estimates, dtype=np.float64)
    if point.ndim != 1 or draws.ndim != 2 or draws.shape[1] != point.size:
        raise ValueError("replicate estimates must be repetitions by periods")
    requested = draws.shape[0] if requested_repetitions is None else requested_repetitions
    if requested != draws.shape[0]:
        raise ValueError("requested_repetitions must equal the stored replicate rows")
    valid = np.sum(np.isfinite(draws), axis=0).astype(np.int64)
    se = np.full(point.shape, np.nan, dtype=np.float64)
    enough = valid >= 2
    if np.any(enough):
        se[enough] = np.nanstd(draws[:, enough], axis=0, ddof=1)
    return inference_from_standard_error(
        point,
        se,
        method=method,
        requested_repetitions=requested,
        valid_repetitions=valid,
        replicate_estimates=draws,
        formal_validated=formal_validated,
    )
