from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import scripts.causal_r.run_parallel_production as production
from scripts.causal_r.run_parallel_production import (
    initialize_isolated_queues,
    isolated_queue_paths,
    phase2_round_specs,
    shard_queue_path,
    validate_resume_checkpoint,
)
from urban_intervention.causal.label_queue import rounds

ROOT = Path(__file__).resolve().parents[2]
ROUND_RUNNER = ROOT / "scripts" / "causal_python" / "run_causal_label_round_queue.py"


def _queue(status: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "treatment_order": [7],
            "outcome_family": pd.Series(["population"], dtype="string"),
            "city_key": ["alpha"],
            "grid_id": ["g7"],
            "station_event_id": ["station-7"],
            "opening_month": ["2020-01"],
            "status": pd.Series([status], dtype="string"),
            "selected_method": pd.Series([pd.NA], dtype="string"),
            "failure_reason": pd.Series([pd.NA], dtype="string"),
        }
    )


@pytest.mark.parametrize(
    ("round_name", "selected_status"),
    [
        ("same_city_matching", "pending"),
        ("same_city_gsc", "gsc_pending"),
        ("same_city_mc", "mc_pending"),
        ("cross_city_matching", "cross_matching_running"),
        ("cross_city_gsc", "cross_gsc_running"),
        ("cross_city_mc", "cross_mc_running"),
    ],
)
def test_global_round_selects_only_its_checkpoint(
    round_name: rounds.GlobalRound,
    selected_status: str,
) -> None:
    queue = pd.DataFrame(
        {
            "treatment_order": list(range(1, 8)),
            "outcome_family": ["population"] * 7,
            "status": [
                "pending",
                "gsc_pending",
                "mc_pending",
                "cross_matching_running",
                "cross_gsc_running",
                "cross_mc_running",
                "matched_labelled",
            ],
        }
    )
    selected = rounds.eligible_round_indices(
        queue,
        1,
        None,
        None,
        "mc",
        99,
        round_name=round_name,
    )
    assert queue.loc[selected, "status"].tolist() == [selected_status]


def test_same_city_mc_failure_advances_one_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = _queue("mc_pending")
    directory = tmp_path / "task"
    directory.mkdir()
    (directory / "gsc_attempt.json").write_text(
        json.dumps({"reason": "rank_cv_failed"}), encoding="utf-8"
    )
    monkeypatch.setattr(rounds.settings, "family_queue", tmp_path / "family.csv")
    monkeypatch.setattr(rounds, "task_directory", lambda *_args: directory)
    monkeypatch.setattr(
        rounds,
        "run_mc_scope",
        lambda *_args: (False, [], {"reason": "same_mc_failed"}),
    )

    rounds.run_global_round(
        queue,
        0,
        queue.loc[0].copy(),
        round_name="same_city_mc",
    )

    assert queue.loc[0, "status"] == "cross_matching_running"
    assert queue.loc[0, "failure_reason"] == "same_mc_failed"
    assert (directory / "same_city_mc_attempt.json").exists()


def test_cross_city_matching_failure_advances_one_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = _queue("cross_matching_running")
    directory = tmp_path / "task"
    control_queue = pd.DataFrame({"treatment_order": [7], "status": ["gsc_pending"]})
    monkeypatch.setattr(rounds.settings, "family_queue", tmp_path / "family.csv")
    monkeypatch.setattr(rounds, "task_directory", lambda *_args: directory)
    monkeypatch.setattr(
        rounds,
        "run_cross_city_matching",
        lambda *_args: (
            False,
            pd.Series(dtype=object),
            {"reason": "cross_match_failed"},
        ),
    )

    rounds.run_global_round(
        queue,
        0,
        queue.loc[0].copy(),
        control_queue,
        round_name="cross_city_matching",
    )

    assert queue.loc[0, "status"] == "cross_gsc_running"
    assert (directory / "cross_city_matching_attempt.json").exists()


def test_cross_city_gsc_failure_advances_one_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = _queue("cross_gsc_running")
    directory = tmp_path / "task"
    monkeypatch.setattr(rounds.settings, "family_queue", tmp_path / "family.csv")
    monkeypatch.setattr(rounds, "task_directory", lambda *_args: directory)
    monkeypatch.setattr(
        rounds,
        "run_gsc_scope",
        lambda *_args: (False, [], {"reason": "cross_gsc_failed"}),
    )

    rounds.run_global_round(
        queue,
        0,
        queue.loc[0].copy(),
        round_name="cross_city_gsc",
    )

    assert queue.loc[0, "status"] == "cross_mc_running"
    assert (directory / "cross_city_gsc_attempt.json").exists()


def test_cross_city_mc_final_failure_is_terminal_and_keeps_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = _queue("cross_mc_running")
    directory = tmp_path / "task"
    directory.mkdir()
    for name in (
        "matching_attempt.json",
        "gsc_attempt.json",
        "same_city_mc_attempt.json",
        "cross_city_matching_attempt.json",
        "cross_city_gsc_attempt.json",
    ):
        (directory / name).write_text(json.dumps({"reason": name}), encoding="utf-8")
    monkeypatch.setattr(rounds.settings, "family_queue", tmp_path / "family.csv")
    monkeypatch.setattr(rounds, "task_directory", lambda *_args: directory)
    monkeypatch.setattr(
        rounds,
        "run_mc_scope",
        lambda *_args: (False, [], {"reason": "cross_mc_failed"}),
    )

    rounds.run_global_round(
        queue,
        0,
        queue.loc[0].copy(),
        round_name="cross_city_mc",
    )

    assert queue.loc[0, "status"] == "skipped"
    manifest = json.loads((directory / "failure_manifest.json").read_text(encoding="utf-8"))
    assert manifest["same_city_mc_failure"]["reason"] == ("same_city_mc_attempt.json")
    assert manifest["cross_city_mc_failure"]["reason"] == "cross_mc_failed"


def test_stale_qualification_cache_is_deleted_and_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = _queue("mc_pending")
    directory = tmp_path / "task"
    directory.mkdir()
    manifest = directory / "manifest.json"
    labels = directory / "labels.parquet"
    manifest.write_text("{}", encoding="utf-8")
    labels.write_bytes(b"labels")
    monkeypatch.setattr(rounds, "task_directory", lambda *_args: directory)
    monkeypatch.setattr(
        rounds,
        "frozen_recover_completed_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError(
                "Completed Python task lacks the current formal qualification receipt"
            )
        ),
    )
    monkeypatch.setattr(
        rounds.settings,
        "qualification_proof",
        {"formal_qualification_receipt_sha256": "new-receipt"},
    )

    recovered = rounds.recover_completed_task_or_recompute_stale_qualification(
        queue, 0
    )

    assert recovered is False
    assert queue.loc[0, "status"] == "mc_pending"
    assert not manifest.exists()
    assert not labels.exists()


def test_recovery_keeps_non_receipt_validation_failures_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = _queue("mc_pending")
    directory = tmp_path / "task"
    directory.mkdir()
    manifest = directory / "manifest.json"
    labels = directory / "labels.parquet"
    manifest.write_text("{}", encoding="utf-8")
    labels.write_bytes(b"labels")
    monkeypatch.setattr(rounds, "task_directory", lambda *_args: directory)
    monkeypatch.setattr(
        rounds,
        "frozen_recover_completed_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("Completed task label hash disagrees with manifest")
        ),
    )
    monkeypatch.setattr(
        rounds.settings,
        "qualification_proof",
        {"formal_qualification_receipt_sha256": "new-receipt"},
    )

    with pytest.raises(ValueError, match="label hash disagrees"):
        rounds.recover_completed_task_or_recompute_stale_qualification(queue, 0)

    assert manifest.exists()
    assert labels.exists()


def test_phase2_global_round_specs_are_complete_and_ordered() -> None:
    specs = phase2_round_specs("all", "global_rounds", False)
    assert [label for label, _runner, _args in specs] == [
        "same-city Matching",
        "same-city GSC",
        "same-city MC",
        "cross-city Matching",
        "cross-city GSC",
        "cross-city MC",
    ]
    assert specs[0][2] == ["--global-round", "same_city_matching"]
    assert specs[1][2] == ["--global-round", "same_city_gsc"]
    assert specs[-1][2] == ["--global-round", "cross_city_mc"]


@pytest.mark.parametrize(
    ("queue_phase", "task_scheduling", "dry_run"),
    [
        ("all", "task_cascade", False),
        ("gsc", "global_rounds", False),
        ("all", "global_rounds", True),
    ],
)
def test_phase2_non_global_modes_remain_single_pass(
    queue_phase: str,
    task_scheduling: str,
    dry_run: bool,
) -> None:
    specs = phase2_round_specs(queue_phase, task_scheduling, dry_run)
    assert len(specs) == 1
    assert specs[0][0] == "task cascade"
    assert specs[0][1] == ROUND_RUNNER
    assert specs[0][2] == ["--phase", queue_phase]


def test_global_round_runner_exposes_legacy_queue_options() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROUND_RUNNER),
            "--global-round",
            "same_city_mc",
            "--help",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout
    assert "--tasks-file" in completed.stdout
    assert "--shard-id" in completed.stdout


def test_prevalidated_resume_requires_existing_nonempty_progress(tmp_path: Path) -> None:
    master = tmp_path / "outcome_family_work_queue.csv"
    for shard_id in range(2):
        pd.DataFrame({"status": ["pending"]}).to_csv(
            shard_queue_path(master, shard_id), index=False
        )
    with pytest.raises(ValueError, match="prior task progress"):
        validate_resume_checkpoint(master, 2)

    pd.DataFrame({"status": ["gsc_labelled"]}).to_csv(
        shard_queue_path(master, 1), index=False
    )
    validate_resume_checkpoint(master, 2)


def test_isolated_robustness_queues_reset_analysis_state_but_keep_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = tmp_path / "outcome_family_work_queue.csv"
    control = tmp_path / "control_design_queue.csv"
    unit = tmp_path / "counterfactual_work_queue.csv"
    pd.DataFrame(
        {
            "treatment_order": [1],
            "outcome_family": ["viirs"],
            "status": ["gsc_labelled"],
            "selected_method": ["gsc"],
            "failure_reason": [pd.NA],
        }
    ).to_csv(family, index=False)
    pd.DataFrame(
        {"treatment_order": [1], "status": ["matched"], "control_grid_id": ["g2"]}
    ).to_csv(control, index=False)
    pd.DataFrame(
        {
            "treatment_order": [1],
            "status": ["labelled"],
            "selected_method": ["gsc"],
            "selected_control_grid_id": ["g2"],
            "failure_reason": [pd.NA],
        }
    ).to_csv(unit, index=False)
    monkeypatch.setattr(production, "OUTCOME_FAMILY_QUEUE", family)
    monkeypatch.setattr(production, "CONTROL_DESIGN_QUEUE", control)
    monkeypatch.setattr(production, "COUNTERFACTUAL_QUEUE", unit)

    isolated = isolated_queue_paths("robust_m7")
    initialize_isolated_queues(*isolated)

    assert pd.read_csv(isolated[0]).loc[0, "status"] == "pending"
    assert pd.read_csv(isolated[1]).loc[0, "status"] == "matched"
    assert pd.read_csv(isolated[2]).loc[0, "status"] == "pending"
    assert isolated[0].name == "outcome_family_work_queue_robust_m7.csv"
