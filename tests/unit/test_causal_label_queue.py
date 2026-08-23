from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "causal_r" / "run_causal_label_queue.py"
SPEC = importlib.util.spec_from_file_location("run_causal_label_queue", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_mc_runner_filters_on_the_requested_treatment_order() -> None:
    source = (ROOT / "scripts" / "causal_r" / "run_complete_mc.R").read_text(encoding="utf-8")
    assert "treatment_order == treatment_order" not in source
    assert source.count("treatment_order == requested_treatment_order") == 2
    assert source.count("must resolve to exactly one treatment") == 2


def test_signature_uses_only_explicit_complete_families() -> None:
    row = pd.Series({"treatment_order": 7})
    support = pd.DataFrame(
        {
            "treatment_order": [7],
            "housing_complete": [True],
            "viirs_complete": [False],
            "population_complete": [pd.NA],
            "poi_complete": [True],
        }
    )
    assert MODULE.family_signature(row, support) == "housing+poi"


def test_atomic_resume_finalizes_completed_task(tmp_path: Path, monkeypatch) -> None:
    family_queue = tmp_path / "family.csv"
    task_root = tmp_path / "tasks"
    monkeypatch.setattr(MODULE, "FAMILY_QUEUE", family_queue)
    monkeypatch.setattr(MODULE, "TASK_ROOT", task_root)
    queue = pd.DataFrame(
        {
            "treatment_order": [1],
            "outcome_family": ["housing"],
            "city_key": ["alpha"],
            "grid_id": ["g1"],
            "station_event_id": ["s1"],
            "opening_month": ["2020-01"],
            "status": ["matching_running"],
            "selected_method": [pd.NA],
            "failure_reason": [pd.NA],
        }
    )
    MODULE.atomic_csv(queue, family_queue)
    row = queue.iloc[0]
    directory = MODULE.task_directory(1, "housing")
    directory.mkdir(parents=True)
    labels_path = directory / "labels.parquet"
    pd.DataFrame(
        {
                "treatment_order": [1] * 6,
                "city_key": ["alpha"] * 6,
                "grid_id": ["g1"] * 6,
                "opening_month": ["2020-01"] * 6,
                "outcome_family": ["housing"] * 6,
                "outcome": ["housing_log_price"] * 6,
                "event_time": [1, 3, 6, 12, 18, 24],
                "specification_id": ["main_a6_r1km"] * 6,
                "specification_fingerprint": [MODULE.specification_fingerprint(row)] * 6,
        }
    ).to_parquet(labels_path)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "causal_response_labels_v1",
                "status": "matched_labelled",
                "method": "matched_change",
                "treatment_order": 1,
                "outcome_family": "housing",
                "city_key": "alpha",
                "grid_id": "g1",
                "opening_month": "2020-01",
                "station_event_id": "s1",
                    "label_rows": 6,
                "labels_sha256": MODULE.file_sha256(labels_path),
                "production_eligible": True,
                "run_mode": "production",
                "specification_fingerprint": MODULE.specification_fingerprint(row),
                "details": {"run_id": "resume-run"},
            }
        ),
        encoding="utf-8",
    )

    assert MODULE.recover_completed_task(queue, 0)
    restored = pd.read_csv(family_queue)
    assert restored.loc[0, "status"] == "matched_labelled"
    assert restored.loc[0, "selected_method"] == "matched_change"


def test_empty_reason_columns_are_loaded_as_writable_strings(tmp_path: Path) -> None:
    path = tmp_path / "queue.csv"
    pd.DataFrame(
        {
            "treatment_order": [1],
            "outcome_family": ["housing"],
            "status": ["pending"],
            "selected_method": [pd.NA],
            "failure_reason": [pd.NA],
        }
    ).to_csv(path, index=False)
    queue = MODULE.read_family_queue(path)
    queue.loc[0, "failure_reason"] = "no_pre_data"
    assert queue.loc[0, "failure_reason"] == "no_pre_data"


def test_viirs_structural_boundary_matches_five_clean_preperiods() -> None:
    # With a six-month anticipation window, 2012-12 has clean pre months
    # 2012-01 through 2012-05; earlier openings cannot satisfy gsynth min.T0=5.
    assert not MODULE.viirs_has_min_preperiods("2012-11")
    assert MODULE.viirs_has_min_preperiods("2012-12")
    assert not MODULE.viirs_has_full_matching_window("2015-06")
    assert MODULE.viirs_has_full_matching_window("2015-07")


def test_unit_queue_aggregates_only_after_four_terminal_families(tmp_path: Path) -> None:
    unit_path = tmp_path / "units.csv"
    pd.DataFrame(
        {
            "treatment_order": [1],
            "status": ["pending"],
            "selected_method": [pd.NA],
            "selected_control_grid_id": [pd.NA],
            "failure_reason": [pd.NA],
        }
    ).to_csv(unit_path, index=False)
    families = pd.DataFrame(
        {
            "treatment_order": [1] * 4,
            "outcome_family": ["housing", "poi", "population", "viirs"],
            "status": ["skipped"] * 4,
            "selected_method": [pd.NA] * 4,
            "failure_reason": ["no_data"] * 4,
        }
    )
    controls = pd.DataFrame(
        {
            "treatment_order": [1],
            "status": ["gsc_pending"],
            "control_unit_key": [pd.NA],
        }
    )
    MODULE.sync_unit_queue(1, families, controls, unit_path)
    result = pd.read_csv(unit_path)
    assert result.loc[0, "status"] == "skipped"
    assert result.loc[0, "failure_reason"] == "all_outcome_families_unavailable"


def test_unit_queue_copies_frozen_physical_control(tmp_path: Path) -> None:
    unit_path = tmp_path / "units.csv"
    pd.DataFrame(
        {
            "treatment_order": [3],
            "status": ["pending"],
            "selected_method": [pd.NA],
            "selected_control_grid_id": [pd.NA],
            "failure_reason": [pd.NA],
        }
    ).to_csv(unit_path, index=False)
    families = pd.DataFrame(
        {
            "treatment_order": [3] * 4,
            "outcome_family": ["housing", "poi", "population", "viirs"],
            "status": ["matched_labelled"] * 4,
            "selected_method": ["frozen_matched_change_12m_baseline"] * 4,
            "failure_reason": [pd.NA] * 4,
        }
    )
    controls = pd.DataFrame(
        {
            "treatment_order": [3],
            "status": ["matched"],
            "control_unit_key": ["suzhou::g00001x00002"],
        }
    )
    MODULE.sync_unit_queue(3, families, controls, unit_path)
    result = pd.read_csv(unit_path)
    assert result.loc[0, "status"] == "labelled"
    assert result.loc[0, "selected_control_grid_id"] == "suzhou::g00001x00002"


def test_phase_selection_separates_matching_and_gsc() -> None:
    queue = pd.DataFrame(
        {
            "treatment_order": [824, 824, 825, 826, 827, 828],
            "outcome_family": pd.Series(
                ["housing", "poi", "housing", "viirs", "population", "poi"],
                dtype="string",
            ),
            "status": pd.Series(
                [
                    "pending",
                    "gsc_pending",
                    "matching_running",
                    "gsc_running",
                    "mc_pending",
                    "mc_running",
                ],
                dtype="string",
            ),
        }
    )
    matching = MODULE.eligible_indices(queue, 824, 825, None, "matching", 10)
    matching_retry = MODULE.eligible_indices(
        queue, 824, 825, None, "matching", 10, retry_matching=True
    )
    gsc = MODULE.eligible_indices(queue, 824, 826, None, "gsc", 10)
    mc = MODULE.eligible_indices(queue, 824, 828, None, "mc", 10)
    all_phases = MODULE.eligible_indices(queue, 824, 828, None, "all", 10)
    assert matching.tolist() == [0, 2]
    assert matching_retry.tolist() == [0, 1, 2]
    assert gsc.tolist() == [1, 3]
    assert mc.tolist() == [4, 5]
    assert all_phases.tolist() == [0, 1, 2, 3, 4, 5]


def test_retry_skipped_is_explicit_and_bounded_by_existing_filters() -> None:
    queue = pd.DataFrame(
        {
            "treatment_order": [10, 11, 12],
            "outcome_family": pd.Series(["poi", "poi", "population"], dtype="string"),
            "status": pd.Series(["skipped", "pending", "skipped"], dtype="string"),
        }
    )
    normal = MODULE.eligible_indices(queue, 10, 11, "poi", "all", 5)
    retry = MODULE.eligible_indices(queue, 10, 11, "poi", "all", 1, retry_skipped=True)
    assert normal.tolist() == [1]
    assert retry.tolist() == [0]


def test_write_task_rejects_swapped_treatment_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MODULE, "TASK_ROOT", tmp_path / "tasks")
    row = pd.Series(
        {
            "treatment_order": 1,
            "city_key": "alpha",
            "grid_id": "g1",
            "station_event_id": "s1",
            "opening_month": "2020-01",
            "outcome_family": "population",
        }
    )
    swapped = pd.DataFrame(
        {
            "treatment_order": [2],
            "city_key": ["beta"],
            "grid_id": ["g2"],
            "opening_month": ["2020-01"],
            "outcome_family": ["population"],
            "outcome": ["population_log"],
            "event_time": [1],
            "specification_id": ["main_a6_r1km"],
        }
    )
    with pytest.raises(ValueError, match="treatment_order"):
        MODULE.write_task(row, [swapped], "matched_labelled", "matched_change", {})
    assert not (tmp_path / "tasks" / "00001" / "population" / "labels.parquet").exists()


def test_mc_scope_keeps_successful_poi_outcomes_when_another_outcome_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "STAGING", tmp_path / "outputs" / "complete_estimators" / "staging")
    monkeypatch.setattr(
        MODULE,
        "run",
        lambda _command, _environment=None: subprocess.CompletedProcess(
            _command, 0, stdout="mc complete"
        ),
    )
    monkeypatch.setattr(MODULE, "new_run_id", lambda: "test-mc-run")
    row = pd.Series(
        {
            "treatment_order": 9,
            "city_key": "alpha",
            "grid_id": "g9",
            "opening_month": "2020-01",
            "outcome_family": "poi",
        }
    )
    status_path = MODULE.mc_family_run_output(row) / "outcome_status.csv"
    status_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "outcome": MODULE.OUTCOMES["poi"],
            "status": ["success", "failed", "failed", "failed"],
            "failure_reason": [pd.NA, "sparse", "sparse", "sparse"],
            "run_id": ["test-mc-run"] * 4,
        }
    ).to_csv(status_path, index=False)
    outcome = MODULE.OUTCOMES["poi"][0]
    output = MODULE.mc_output(row, outcome)
    output.mkdir(parents=True)
    pd.DataFrame(
        {
                "treatment_order": [9] * 3,
                "city_key": ["alpha"] * 3,
                "grid_id": ["g9"] * 3,
                "outcome": [outcome] * 3,
            "event_time": [1, 2, 3],
            "observed": [2.0, 2.1, 2.2],
            "counterfactual": [1.5, 1.6, 1.7],
            "causal_response_label": [0.5, 0.5, 0.5],
            "label_available": [True, True, True],
            "standard_error": [0.1, 0.1, 0.1],
            "confidence_lower": [0.3, 0.3, 0.3],
            "confidence_upper": [0.7, 0.7, 0.7],
            "p_value": [0.01, 0.01, 0.01],
            "bootstrap_repetitions": [0, 0, 0],
            "uncertainty_source": ["mc_jackknife"] * 3,
            "pre_observed_periods": [1, 1, 1],
            "pre_rmspe": [0.2, 0.2, 0.2],
            "mc_lambda": [0.25, 0.25, 0.25],
            "mc_cv_mspe": [0.04, 0.04, 0.04],
        }
    ).to_parquet(output / "causal_response_labels.parquet", index=False)
    pd.DataFrame(
        {
            "field": [
                "estimator",
                "fitted_method",
                "backend",
                "force",
                "criterion",
                "nlambda",
                "min_T0",
                "se",
                "run_id",
                "CV",
                "cv_method",
                "cv_nobs",
                "cv_donut",
                "cv_buffer",
                "selected_lambda",
                "two_stage_cv_inference",
                "inference_fit_CV",
                "cv_min_mspe",
                "run_mode",
                "production_eligible",
                "inference",
                "nboots",
                "specification_fingerprint",
                "price_measure",
                "observation_window",
            ],
            "value": [
                "mc",
                "mc",
                "fect",
                "two-way",
                "mspe",
                "20",
                "1",
                "TRUE",
                "test-mc-run",
                "TRUE",
                "rolling",
                "1",
                "0",
                "0",
                "0.25",
                "TRUE",
                "FALSE",
                "0.04",
                "production",
                "TRUE",
                "jackknife",
                "200",
                MODULE.specification_fingerprint(row),
                "median",
                "1",
            ],
        }
    ).to_csv(output / "manifest.csv", index=False)

    ok, labels, details = MODULE.run_mc_scope(row, "same_city")

    assert ok
    assert len(labels) == 1
    assert labels[0]["outcome"].tolist() == [outcome] * 3
    assert set(details["outcome_failures"]) == set(MODULE.OUTCOMES["poi"][1:])


def test_mc_scope_rejects_status_from_a_previous_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "STAGING", tmp_path / "outputs" / "complete_estimators" / "staging")
    monkeypatch.setattr(MODULE, "new_run_id", lambda: "current-run")
    monkeypatch.setattr(
        MODULE,
        "run",
        lambda command, _environment=None: subprocess.CompletedProcess(
            command, 0, stdout="process returned without publishing new output"
        ),
    )
    row = pd.Series(
        {
            "treatment_order": 4,
            "city_key": "alpha",
            "grid_id": "g4",
            "opening_month": "2020-01",
            "outcome_family": "population",
        }
    )
    status_path = MODULE.mc_family_run_output(row) / "outcome_status.csv"
    status_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "outcome": ["population_log"],
            "status": ["success"],
            "failure_reason": [pd.NA],
            "run_id": ["previous-run"],
        }
    ).to_csv(status_path, index=False)

    ok, labels, details = MODULE.run_mc_scope(row, "same_city")

    assert not ok
    assert labels == []
    assert details["reason"] == "mc_family_status_is_not_from_current_run"


def test_gsc_scope_rejects_smoke_or_stale_estimator_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "STAGING", tmp_path / "outputs" / "complete_estimators" / "staging")
    monkeypatch.setattr(MODULE, "new_run_id", lambda: "current-gsc-run")
    monkeypatch.setattr(
        MODULE,
        "run",
        lambda command, _environment=None: subprocess.CompletedProcess(
            command, 0, stdout="stale output left in place"
        ),
    )
    row = pd.Series(
        {
            "treatment_order": 5,
            "city_key": "alpha",
            "grid_id": "g5",
            "opening_month": "2020-01",
            "outcome_family": "population",
        }
    )
    output = MODULE.gsc_output(row, "population_log")
    output.mkdir(parents=True)
    pd.DataFrame({"placeholder": [1]}).to_parquet(
        output / "causal_response_labels.parquet", index=False
    )
    pd.DataFrame(
        {
            "field": [
                "run_id",
                "estimator",
                "CV",
                "run_mode",
                "production_eligible",
                "inference",
                "nboots",
            ],
            "value": [
                "previous-gsc-run",
                "gsynth",
                "TRUE",
                "smoke_test",
                "FALSE",
                "parametric",
                "20",
            ],
        }
    ).to_csv(output / "manifest.csv", index=False)

    ok, labels, details = MODULE.run_gsc_scope(row, "same_city")

    assert not ok
    assert labels == []
    assert details["reason"] == "xu_gsc_manifest_does_not_prove_current_production_run"
