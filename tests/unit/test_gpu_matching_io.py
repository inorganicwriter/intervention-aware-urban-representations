from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from urban_intervention.causal.gpu.contracts import EstimatorProvenance, MatchingResult
from urban_intervention.causal.gpu.matching_io import (
    compare_matching_labels,
    compare_matching_result,
    load_matching_artifacts,
    matching_result_frames,
)


def test_matching_artifact_adapter_and_parity(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "role": ["donor", "donor", "donor", "treated"],
            "unit_key": ["d0", "d1", "d2", "target"],
            "x__lag2": [0.0, 1.0, 2.0, 1.1],
            "s": [0.0, 1.0, 2.0, 1.0],
            "x__lag1": [0.0, 1.0, 2.0, 1.2],
        }
    )
    frame.to_parquet(tmp_path / "matching_input.parquet", index=False)
    pd.DataFrame(
        {
            "schema": ["causal_gpu_matching_reference_exact_stable_ties"],
            "training_features": ["x__lag2"],
            "static_features": ["s"],
            "holdout_features": ["x__lag1"],
            "matching_candidates": [2],
            "placebo_sample": [3],
            "placebo_quantile": [0.95],
            "distance_tolerance": [0],
            "tie_policy": ["distance_then_original_donor_index"],
        }
    ).to_csv(tmp_path / "metadata.csv", index=False)
    pd.DataFrame({"control_unit_key": ["d1", "d2"]}).to_csv(
        tmp_path / "reference_candidates.csv", index=False
    )
    pd.DataFrame(
        {
            "control_unit_key": ["d1"],
            "training_distance": [0.1],
            "holdout_rms_standardized_gap": [0.2],
            "holdout_max_abs_standardized_gap": [0.2],
            "training_distance_threshold": [0.3],
            "holdout_rms_threshold": [0.4],
            "holdout_max_abs_threshold": [0.4],
            "accepted": [True],
        }
    ).to_csv(tmp_path / "reference_selection.csv", index=False)
    artifacts = load_matching_artifacts(tmp_path)
    result = MatchingResult(
        donor_indices=np.array([1, 2]),
        distances=np.array([0.1, 0.2]),
        support_count=1,
        selected_index=1,
        selected_distance=0.1,
        training_distance=0.1,
        holdout_rms_standardized_gap=0.2,
        holdout_max_abs_standardized_gap=0.2,
        placebo_thresholds={
            "training_distance": 0.3,
            "holdout_rms_standardized_gap": 0.4,
            "holdout_max_abs_standardized_gap": 0.4,
        },
        quality_passed=True,
        provenance=EstimatorProvenance(estimator="matching", backend="pytorch"),
    )
    assert compare_matching_result(artifacts, result)["passed"] is True
    candidates, selection = matching_result_frames(artifacts, result)
    assert candidates["control_unit_key"].tolist() == ["d1", "d2"]
    assert selection.loc[0, "control_unit_key"] == "d1"
    assert bool(selection.loc[0, "accepted"])


def test_matching_artifacts_allow_input_only_mode(tmp_path) -> None:
    pd.DataFrame(
        {
            "role": ["donor", "donor", "donor", "treated"],
            "unit_key": ["d0", "d1", "d2", "target"],
            "x": [0.0, 1.0, 2.0, 1.1],
            "h": [0.0, 1.0, 2.0, 1.2],
        }
    ).to_parquet(tmp_path / "matching_input.parquet", index=False)
    pd.DataFrame(
        {
            "schema": ["causal_gpu_matching_input_exact_stable_ties"],
            "training_features": ["x"],
            "static_features": [""],
            "holdout_features": ["h"],
            "matching_candidates": [2],
            "placebo_sample": [3],
            "placebo_quantile": [0.95],
            "distance_tolerance": [0],
            "tie_policy": ["distance_then_original_donor_index"],
        }
    ).to_csv(tmp_path / "metadata.csv", index=False)

    artifacts = load_matching_artifacts(tmp_path)

    assert artifacts.reference_candidates is None
    assert artifacts.reference_selection is None


def test_matching_artifacts_reject_unknown_future_schema(tmp_path) -> None:
    pd.DataFrame(
        {
            "role": ["donor", "donor", "donor", "treated"],
            "unit_key": ["d0", "d1", "d2", "target"],
            "x": [0.0, 1.0, 2.0, 1.1],
        }
    ).to_parquet(tmp_path / "matching_input.parquet", index=False)
    pd.DataFrame(
        {
            "schema": ["causal_gpu_matching_v999_incompatible"],
            "training_features": ["x"],
            "static_features": [""],
            "holdout_features": [""],
            "distance_tolerance": [0],
            "tie_policy": ["distance_then_original_donor_index"],
        }
    ).to_csv(tmp_path / "metadata.csv", index=False)

    with pytest.raises(ValueError, match="unsupported matching GPU contract schema"):
        load_matching_artifacts(tmp_path)


def test_matching_final_label_parity_compares_complete_paths() -> None:
    reference = pd.DataFrame(
        {
            "outcome_family": ["population", "population"],
            "outcome": ["population_log", "population_log"],
            "event_time": [1, 2],
            "label_available": [True, False],
            "observed": [2.0, np.nan],
            "counterfactual": [1.5, np.nan],
            "causal_response_label": [0.5, np.nan],
            "treated_baseline": [1.0, 1.0],
            "control_baseline": [1.0, 1.0],
            "treated_change": [1.0, np.nan],
            "control_change": [0.5, np.nan],
        }
    )
    comparison, parity = compare_matching_labels(reference, reference.copy())
    assert parity["passed"] is True
    assert comparison["key_present_both"].all()

    changed = reference.copy()
    changed.loc[0, "causal_response_label"] = 0.6
    _, failed = compare_matching_labels(reference, changed)
    assert failed["passed"] is False
