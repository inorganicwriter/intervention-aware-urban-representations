"""Validated, NumPy-based contracts at the R/Python/GPU boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating[Any]]
BoolArray = npt.NDArray[np.bool_]
GPU_IMPLEMENTATION_VERSION = "gpu"
SHADOW_SCHEMA = "causal_gpu_shadow_formal_contract"
FORMAL_IMPLEMENTATION_VERSION = "python-causal"
FORMAL_RESULT_SCHEMA = "causal_python_formal_result_qualified"
CONTROL_DESIGN_SCHEMA = "grid_control_design_exact_stable_ties"
CONTROL_DESIGN_VIIRS_CACHE_CONTRACT = "complete_44_city_2012_2024_monthly_v1"
CONTROL_DESIGN_PROVENANCE = {
    "python_gpu": {
        "backend": "python_pytorch",
        "implementation_version": FORMAL_IMPLEMENTATION_VERSION,
        "selected_method": "python_gpu_M5_static_refine",
    },
    "r_reference": {
        "backend": "r_matching",
        "implementation_version": "r-reference-grid",
        "selected_method": "Matching::Match_M5_static_refine",
    },
}


def _as_float64(value: Any, *, ndim: int, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {result.shape}")
    return result


@dataclass(frozen=True, slots=True)
class EstimatorProvenance:
    """Information needed to distinguish reference and accelerated results."""

    estimator: str
    backend: str
    implementation_version: str = GPU_IMPLEMENTATION_VERSION
    reference_backend: str = "R"
    formal_eligible: bool = False
    numerical_policy: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MatchingInput:
    """A target and donor matrix for one matching task."""

    target: FloatArray
    donors: FloatArray
    donor_ids: Sequence[str] | None = None
    support_feature_indices: tuple[int, ...] | None = None
    target_static: FloatArray | None = None
    donor_static: FloatArray | None = None
    target_holdout: FloatArray | None = None
    donor_holdout: FloatArray | None = None

    def __post_init__(self) -> None:
        target = _as_float64(self.target, ndim=1, name="target")
        donors = _as_float64(self.donors, ndim=2, name="donors")
        if donors.shape[0] == 0 or donors.shape[1] == 0:
            raise ValueError("donors must be non-empty")
        if target.shape[0] != donors.shape[1]:
            raise ValueError("target and donors must have the same feature count")
        if not np.isfinite(target).all() or not np.isfinite(donors).all():
            raise ValueError("matching features must all be finite")
        if self.donor_ids is not None and len(self.donor_ids) != donors.shape[0]:
            raise ValueError("donor_ids length must equal the donor row count")
        if self.support_feature_indices is not None:
            if not self.support_feature_indices:
                raise ValueError("support_feature_indices cannot be empty")
            if min(self.support_feature_indices) < 0 or max(self.support_feature_indices) >= donors.shape[1]:
                raise ValueError("support feature index is out of bounds")
        if (self.target_static is None) != (self.donor_static is None):
            raise ValueError("target_static and donor_static must be supplied together")
        if self.target_static is not None and self.donor_static is not None:
            target_static = _as_float64(self.target_static, ndim=1, name="target_static")
            donor_static = _as_float64(self.donor_static, ndim=2, name="donor_static")
            if donor_static.shape[0] != donors.shape[0]:
                raise ValueError("donor_static row count must equal donor row count")
            if target_static.shape[0] != donor_static.shape[1]:
                raise ValueError("target_static and donor_static must have the same feature count")
            if not np.isfinite(target_static).all() or not np.isfinite(donor_static).all():
                raise ValueError("static matching features must all be finite")
            object.__setattr__(self, "target_static", target_static)
            object.__setattr__(self, "donor_static", donor_static)
        if (self.target_holdout is None) != (self.donor_holdout is None):
            raise ValueError("target_holdout and donor_holdout must be supplied together")
        if self.target_holdout is not None and self.donor_holdout is not None:
            target_holdout = _as_float64(self.target_holdout, ndim=1, name="target_holdout")
            donor_holdout = _as_float64(self.donor_holdout, ndim=2, name="donor_holdout")
            if donor_holdout.shape[0] != donors.shape[0]:
                raise ValueError("donor_holdout row count must equal donor row count")
            if target_holdout.shape[0] != donor_holdout.shape[1]:
                raise ValueError("target_holdout and donor_holdout must have the same feature count")
            if not np.isfinite(target_holdout).all() or not np.isfinite(donor_holdout).all():
                raise ValueError("holdout matching features must all be finite")
            object.__setattr__(self, "target_holdout", target_holdout)
            object.__setattr__(self, "donor_holdout", donor_holdout)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "donors", donors)


@dataclass(frozen=True, slots=True)
class MatchingResult:
    donor_indices: npt.NDArray[np.int64]
    distances: FloatArray
    support_count: int
    selected_index: int
    selected_distance: float
    training_distance: float | None
    holdout_rms_standardized_gap: float | None
    holdout_max_abs_standardized_gap: float | None
    placebo_thresholds: dict[str, float] | None
    quality_passed: bool | None
    provenance: EstimatorProvenance


@dataclass(frozen=True, slots=True)
class PanelData:
    """A time-by-unit outcome panel with explicit masks.

    ``treated`` marks cells exposed to treatment, while ``observed`` describes
    data availability.  A treated cell may be unobserved.  Pre-treatment
    fitting always uses ``observed & ~treated`` and therefore cannot leak
    post-treatment outcomes.
    """

    y: FloatArray
    observed: BoolArray | None = None
    treated: BoolArray | None = None
    unit_ids: Sequence[str] | None = None
    time_ids: Sequence[Any] | None = None

    def __post_init__(self) -> None:
        y = _as_float64(self.y, ndim=2, name="y")
        if y.shape[0] < 2 or y.shape[1] < 2:
            raise ValueError("panel must contain at least two periods and two units")
        observed = np.isfinite(y) if self.observed is None else np.asarray(self.observed, dtype=bool)
        treated = np.zeros_like(observed) if self.treated is None else np.asarray(self.treated, dtype=bool)
        if observed.shape != y.shape or treated.shape != y.shape:
            raise ValueError("observed and treated masks must match y")
        if np.isinf(y[observed]).any() or np.isnan(y[observed]).any():
            raise ValueError("observed panel cells must be finite")
        if self.unit_ids is not None and len(self.unit_ids) != y.shape[1]:
            raise ValueError("unit_ids length must equal panel unit count")
        if self.time_ids is not None and len(self.time_ids) != y.shape[0]:
            raise ValueError("time_ids length must equal panel period count")
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "treated", treated)

    @property
    def untreated_observed(self) -> BoolArray:
        assert self.observed is not None and self.treated is not None
        return self.observed & ~self.treated

    def single_treated_unit(self) -> int:
        assert self.treated is not None
        units = np.flatnonzero(self.treated.any(axis=0))
        if units.size != 1:
            raise ValueError(f"expected exactly one treated unit, found {units.size}")
        return int(units[0])

    def treatment_start(self) -> int:
        unit = self.single_treated_unit()
        assert self.treated is not None
        periods = np.flatnonzero(self.treated[:, unit])
        if periods.size == 0:
            raise ValueError("treated unit has no treated periods")
        if not np.array_equal(periods, np.arange(periods[0], self.y.shape[0])):
            raise ValueError("treatment must be absorbing through the end of the panel")
        return int(periods[0])
