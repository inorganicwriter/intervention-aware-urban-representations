"""Artifact adapter and parity checks for the R matching design."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import MatchingInput, MatchingResult

MATCHING_SCHEMAS = {
    "causal_gpu_matching_input_exact_stable_ties",
    "causal_gpu_matching_reference_exact_stable_ties",
    "causal_gpu_matching_reference_final_labels",
    "causal_gpu_matching_input_v2_exact_stable_ties",
    "causal_gpu_matching_reference_v2_exact_stable_ties",
    "causal_gpu_matching_reference_v3_final_labels",
}


def _supported_schema(schema: str) -> bool:
    return schema in MATCHING_SCHEMAS


@dataclass(frozen=True, slots=True)
class MatchingArtifacts:
    data: MatchingInput
    metadata: dict[str, Any]
    donor_ids: tuple[str, ...]
    reference_candidates: pd.DataFrame | None
    reference_selection: pd.Series | None
    reference_labels: pd.DataFrame | None


def _feature_list(value: Any) -> list[str]:
    if pd.isna(value) or not str(value):
        return []
    return str(value).split("|")


def load_matching_artifacts(directory: Path) -> MatchingArtifacts:
    """Load the compact bridge artifacts produced by the R reference exporter."""
    directory = Path(directory)
    frame = pd.read_parquet(directory / "matching_input.parquet")
    metadata_row = pd.read_csv(directory / "metadata.csv", encoding="utf-8-sig").iloc[0]
    schema = str(metadata_row.get("schema", ""))
    if not _supported_schema(schema):
        raise ValueError(f"unsupported matching GPU contract schema: {schema!r}")
    distance_tolerance = pd.to_numeric(
        pd.Series([metadata_row.get("distance_tolerance")]), errors="coerce"
    ).iloc[0]
    if not np.isfinite(distance_tolerance) or float(distance_tolerance) != 0:
        raise ValueError("matching GPU contract requires distance_tolerance=0")
    if str(metadata_row.get("tie_policy", "")) != "distance_then_original_donor_index":
        raise ValueError("matching GPU contract requires the stable donor-order tie policy")
    training = _feature_list(metadata_row["training_features"])
    static = _feature_list(metadata_row["static_features"])
    holdout = _feature_list(metadata_row["holdout_features"])
    required = {"role", "unit_key", *training, *static, *holdout}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"matching input lacks columns: {missing}")
    treated = frame.loc[frame["role"].eq("treated")]
    donors = frame.loc[frame["role"].eq("donor")]
    if not frame["role"].isin({"treated", "donor"}).all():
        raise ValueError("matching input contains an unsupported role")
    if len(treated) != 1 or len(donors) < 3:
        raise ValueError("matching input must contain one target and at least three donors")
    if frame["unit_key"].astype(str).duplicated().any():
        raise ValueError("matching unit keys must be unique")
    target = treated.iloc[0]
    donor_ids = tuple(donors["unit_key"].astype(str))
    data = MatchingInput(
        target=target[training].to_numpy(dtype=np.float64),
        donors=donors[training].to_numpy(dtype=np.float64),
        donor_ids=donor_ids,
        support_feature_indices=tuple(range(len(training))),
        target_static=(
            target[static].to_numpy(dtype=np.float64) if static else None
        ),
        donor_static=(
            donors[static].to_numpy(dtype=np.float64) if static else None
        ),
        target_holdout=target[holdout].to_numpy(dtype=np.float64),
        donor_holdout=donors[holdout].to_numpy(dtype=np.float64),
    )
    candidates_path = directory / "reference_candidates.csv"
    selection_path = directory / "reference_selection.csv"
    labels_path = directory / "reference_labels.parquet"
    if candidates_path.is_file() != selection_path.is_file():
        raise ValueError("matching reference artifacts must be both present or both absent")
    return MatchingArtifacts(
        data=data,
        metadata=metadata_row.to_dict(),
        donor_ids=donor_ids,
        reference_candidates=(
            pd.read_csv(candidates_path, encoding="utf-8-sig")
            if candidates_path.is_file()
            else None
        ),
        reference_selection=(
            pd.read_csv(selection_path, encoding="utf-8-sig").iloc[0]
            if selection_path.is_file()
            else None
        ),
        reference_labels=(pd.read_parquet(labels_path) if labels_path.is_file() else None),
    )


def matching_result_frames(
    artifacts: MatchingArtifacts,
    result: MatchingResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create durable candidate and selected-control artifacts."""
    candidates = pd.DataFrame(
        {
            "candidate_rank": np.arange(1, len(result.donor_indices) + 1),
            "control_unit_key": [
                artifacts.donor_ids[index] for index in result.donor_indices
            ],
            "matching_distance": result.distances,
        }
    )
    thresholds = result.placebo_thresholds or {}
    selection = pd.DataFrame(
        {
            "control_unit_key": [artifacts.donor_ids[result.selected_index]],
            "selected_distance": [result.selected_distance],
            "training_distance": [result.training_distance],
            "holdout_rms_standardized_gap": [result.holdout_rms_standardized_gap],
            "holdout_max_abs_standardized_gap": [result.holdout_max_abs_standardized_gap],
            "training_distance_threshold": [thresholds.get("training_distance")],
            "holdout_rms_threshold": [thresholds.get("holdout_rms_standardized_gap")],
            "holdout_max_abs_threshold": [thresholds.get("holdout_max_abs_standardized_gap")],
            "accepted": [result.quality_passed],
        }
    )
    return candidates, selection


def compare_matching_result(
    artifacts: MatchingArtifacts,
    result: MatchingResult,
    *,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compare candidates, selected control, metrics, and q95 thresholds."""
    if artifacts.reference_candidates is None or artifacts.reference_selection is None:
        raise ValueError("matching reference artifacts are unavailable")
    gpu_candidates = [artifacts.donor_ids[index] for index in result.donor_indices]
    reference_candidates = artifacts.reference_candidates["control_unit_key"].astype(str).tolist()
    gpu_selected = artifacts.donor_ids[result.selected_index]
    reference = artifacts.reference_selection
    metric_pairs = {
        "training_distance": result.training_distance,
        "holdout_rms_standardized_gap": result.holdout_rms_standardized_gap,
        "holdout_max_abs_standardized_gap": result.holdout_max_abs_standardized_gap,
        "training_distance_threshold": (
            None if result.placebo_thresholds is None else result.placebo_thresholds["training_distance"]
        ),
        "holdout_rms_threshold": (
            None
            if result.placebo_thresholds is None
            else result.placebo_thresholds["holdout_rms_standardized_gap"]
        ),
        "holdout_max_abs_threshold": (
            None
            if result.placebo_thresholds is None
            else result.placebo_thresholds["holdout_max_abs_standardized_gap"]
        ),
    }
    errors: dict[str, float] = {}
    metrics_passed = True
    for name, value in metric_pairs.items():
        reference_value = float(reference[name])
        if value is None:
            errors[name] = float("inf")
            metrics_passed = False
            continue
        error = abs(float(value) - reference_value)
        errors[name] = error
        tolerance = absolute_tolerance + relative_tolerance * abs(reference_value)
        metrics_passed = metrics_passed and error <= tolerance
    candidate_set_equal = set(gpu_candidates) == set(reference_candidates)
    selected_equal = gpu_selected == str(reference["control_unit_key"])
    quality_equal = bool(result.quality_passed) == bool(reference["accepted"])
    return {
        "candidate_set_equal": candidate_set_equal,
        "candidate_order_equal": gpu_candidates == reference_candidates,
        "selected_equal": selected_equal,
        "quality_gate_equal": quality_equal,
        "metric_absolute_errors": errors,
        "max_metric_absolute_error": max(errors.values()),
        "passed": candidate_set_equal and selected_equal and quality_equal and metrics_passed,
    }


def compare_matching_labels(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-9,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare final fixed-control label paths, not only the selected donor."""
    keys = ["outcome_family", "outcome", "event_time"]
    numeric = [
        "observed",
        "counterfactual",
        "causal_response_label",
        "treated_baseline",
        "control_baseline",
        "treated_change",
        "control_change",
    ]
    required = {*keys, "label_available", *numeric}
    for name, frame in (("reference", reference), ("candidate", candidate)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"matching {name} labels lack columns: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"matching {name} labels contain duplicate path keys")
    left = reference[[*keys, "label_available", *numeric]].copy()
    right = candidate[[*keys, "label_available", *numeric]].copy()
    merged = left.merge(
        right,
        on=keys,
        how="outer",
        suffixes=("_reference", "_python"),
        indicator=True,
        validate="one_to_one",
    )
    merged["key_present_both"] = merged["_merge"].eq("both")
    merged["availability_equal"] = (
        merged["label_available_reference"].astype("boolean")
        == merged["label_available_python"].astype("boolean")
    ).fillna(False)
    passed = bool(merged["key_present_both"].all() and merged["availability_equal"].all())
    max_error = 0.0
    for column in numeric:
        reference_values = pd.to_numeric(merged[f"{column}_reference"], errors="coerce")
        python_values = pd.to_numeric(merged[f"{column}_python"], errors="coerce")
        both_missing = reference_values.isna() & python_values.isna()
        both_finite = np.isfinite(reference_values) & np.isfinite(python_values)
        errors = (reference_values - python_values).abs()
        tolerance = absolute_tolerance + relative_tolerance * reference_values.abs()
        equal = both_missing | (both_finite & errors.le(tolerance))
        merged[f"{column}_absolute_error"] = errors
        merged[f"{column}_equal"] = equal
        finite_errors = errors[np.isfinite(errors)]
        if not finite_errors.empty:
            max_error = max(max_error, float(finite_errors.max()))
        passed = passed and bool(equal.all())
    merged = merged.drop(columns="_merge")
    return merged, {
        "available": True,
        "passed": passed,
        "reference_rows": len(reference),
        "python_rows": len(candidate),
        "keys_equal": bool(merged["key_present_both"].all()),
        "availability_equal": bool(merged["availability_equal"].all()),
        "max_absolute_error": max_error,
    }
