from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.skip_mc_insufficient_pre import (
    INVALID_PROOF,
    PREFIX,
    apply_skips,
    classify_mc_demotion,
    scan_task,
)
from urban_intervention.causal.label_queue.mc import has_sufficient_pre_observation


def _mc_frame(
    pre: object,
    *,
    available: bool = True,
    lam: float = 0.1,
    cv: float = 0.5,
    method: str = "athey_2021_mc_same_city",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": [method],
            "label_available": [available],
            "pre_observed_periods": [pre],
            "mc_lambda": [lam],
            "mc_cv_mspe": [cv],
        }
    )


def test_has_sufficient_pre_observation() -> None:
    assert has_sufficient_pre_observation(_mc_frame(3)) is True
    assert has_sufficient_pre_observation(_mc_frame(1)) is True
    assert has_sufficient_pre_observation(_mc_frame(0)) is False
    assert has_sufficient_pre_observation(_mc_frame(1, available=False)) is False
    without_column = _mc_frame(1).drop(columns="pre_observed_periods")
    assert has_sufficient_pre_observation(without_column) is False
    assert has_sufficient_pre_observation(pd.DataFrame({"label_available": []})) is False


def test_classify_mc_demotion() -> None:
    assert classify_mc_demotion(_mc_frame(0)) == PREFIX
    assert classify_mc_demotion(_mc_frame(1)) is None
    assert classify_mc_demotion(_mc_frame(2)) is None
    assert classify_mc_demotion(_mc_frame(0, lam=float("nan"))) == INVALID_PROOF
    assert classify_mc_demotion(_mc_frame(0, cv=float("nan"))) == INVALID_PROOF
    assert (
        classify_mc_demotion(
            _mc_frame(0, method="frozen_matched_change_12m_baseline")
        )
        == INVALID_PROOF
    )
    assert classify_mc_demotion(pd.DataFrame({"label_available": [False]})) is None


def test_scan_task_flags_only_mc_labels_without_anchor(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    for order, pre, _status in (
        (1, 0, "mc_labelled"),
        (2, 3, "mc_labelled"),
        (3, 0, "gsc_labelled"),
    ):
        directory = root / f"{order:05d}" / "poi"
        directory.mkdir(parents=True)
        _mc_frame(pre).to_parquet(directory / "labels.parquet", index=False)

    demoted = scan_task(1, "poi", "mc_labelled", root)
    assert demoted is not None
    assert demoted["reason"] == PREFIX
    assert demoted["pre_min"] == 0.0
    assert scan_task(2, "poi", "mc_labelled", root) is None
    assert scan_task(3, "poi", "gsc_labelled", root) is None
    missing = scan_task(4, "poi", "mc_labelled", root)
    assert missing is not None
    assert missing["reason"] == "mc_labelled_task_missing_labels"


def _queue_fixture(tmp_path: Path) -> Path:
    queue_path = tmp_path / "queue.csv"
    pd.DataFrame(
        {
            "treatment_order": [1, 2],
            "outcome_family": ["poi", "poi"],
            "city_key": ["alpha", "beta"],
            "opening_month": ["2013-01", "2013-02"],
            "status": ["mc_labelled", "gsc_labelled"],
            "selected_method": [
                "athey_2021_mc_same_city",
                "xu_2017_gsynth_same_city",
            ],
            "failure_reason": [pd.NA, pd.NA],
        }
    ).to_csv(queue_path, index=False)
    return queue_path


def _database_fixture(tmp_path: Path) -> Path:
    db_path = tmp_path / "pool.sqlite"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE tasks ("
        "run_id TEXT, treatment_order INTEGER, outcome_family TEXT, queue_status TEXT, "
        "selected_method TEXT, failure_reason TEXT, updated_at REAL)"
    )
    connection.execute(
        "INSERT INTO tasks VALUES ('run', 1, 'poi', 'completed', "
        "'athey_2021_mc_same_city', NULL, 0)"
    )
    connection.execute(
        "INSERT INTO tasks VALUES ('run', 2, 'poi', 'completed', "
        "'xu_2017_gsynth_same_city', NULL, 0)"
    )
    connection.commit()
    connection.close()
    return db_path


def test_apply_skips_updates_queue_and_sqlite(tmp_path: Path) -> None:
    queue_path = _queue_fixture(tmp_path)
    db_path = _database_fixture(tmp_path)
    records = [{"treatment_order": 1, "outcome_family": "poi", "reason": PREFIX}]

    summary = apply_skips(queue_path, records, db_path=db_path, run_id="run")
    assert summary == {"queue_rows_updated": 1, "db_rows_updated": 1}

    queue = pd.read_csv(queue_path)
    demoted = queue.loc[queue["treatment_order"].eq(1)].iloc[0]
    assert demoted["status"] == "skipped"
    assert demoted["failure_reason"] == PREFIX
    assert pd.isna(demoted["selected_method"])
    assert queue.loc[queue["treatment_order"].eq(2), "status"].item() == "gsc_labelled"

    connection = sqlite3.connect(db_path)
    row = connection.execute(
        "SELECT queue_status, selected_method, failure_reason FROM tasks "
        "WHERE treatment_order = 1"
    ).fetchone()
    untouched = connection.execute(
        "SELECT queue_status FROM tasks WHERE treatment_order = 2"
    ).fetchone()
    connection.close()
    assert row == ("skipped", None, PREFIX)
    assert untouched == ("completed",)


def test_apply_skips_requires_run_id_for_database_updates(tmp_path: Path) -> None:
    queue_path = _queue_fixture(tmp_path)
    with pytest.raises(ValueError, match="requires --run-id"):
        apply_skips(
            queue_path,
            [{"treatment_order": 1, "outcome_family": "poi", "reason": PREFIX}],
            db_path=tmp_path / "pool.sqlite",
            run_id=None,
        )
