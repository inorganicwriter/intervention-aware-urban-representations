from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "causal_r" / "run_grid_control_design_queue.py"
SPEC = importlib.util.spec_from_file_location("run_grid_control_design_queue", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_initial_queue_has_one_preonly_row_per_treated_grid(tmp_path: Path) -> None:
    path = tmp_path / "controls.csv"
    queue = MODULE.initialize_queue(path)
    assert len(queue) == 5_048
    assert not queue[["city_key", "grid_id"]].duplicated().any()
    assert queue["status"].eq("pending").all()
    assert not queue["control_selection_uses_post_outcome"].any()
    restored = pd.read_csv(path)
    assert len(restored) == 5_048


def test_batch_recovery_reads_each_durable_control_record(tmp_path: Path, monkeypatch) -> None:
    queue_path = tmp_path / "queue.csv"
    task_root = tmp_path / "tasks"
    monkeypatch.setattr(MODULE, "QUEUE", queue_path)
    monkeypatch.setattr(MODULE, "TASK_ROOT", task_root)
    queue = pd.DataFrame(
        {
            "treatment_order": [1],
            "status": ["pending"],
            "control_grid_id": [pd.NA],
            "failure_reason": [pd.NA],
        }
    )
    MODULE.atomic_csv(queue, queue_path)
    directory = task_root / "00001"
    directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "treatment_order": [1],
            "status": ["matched"],
            "control_grid_id": ["g00001x00001"],
            "failure_reason": [pd.NA],
                "viirs_cache_contract": [MODULE.VIIRS_CACHE_CONTRACT],
                "schema": [MODULE.CONTROL_DESIGN_SCHEMA],
                "implementation_version": ["r-reference-grid"],
                "backend": ["r_matching"],
        }
    ).to_csv(directory / "control_record.csv", index=False)

    class Completed:
        returncode = 0
        stdout = "ok"

    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: Completed())
    MODULE.run_batch(queue, [0], backend="r_reference")
    restored = pd.read_csv(queue_path)
    assert restored.loc[0, "status"] == "matched"
    assert restored.loc[0, "control_grid_id"] == "g00001x00001"


def test_monthly_viirs_cache_contract_requires_parquet_and_audit(tmp_path: Path) -> None:
    monthly = tmp_path / "monthly"
    audits = tmp_path / "audits"
    period = pd.Period("2012-01", freq="M")
    parquet = monthly / "city_key=a" / "year=2012" / "month=01" / "part.parquet"
    audit = audits / "a" / "2012-01.json"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"fixture")
    assert MODULE.missing_viirs_cache(["a"], [period], monthly, audits) == ["a:2012-01"]
    audit.parent.mkdir(parents=True)
    audit.write_text("{}", encoding="utf-8")
    assert MODULE.missing_viirs_cache(["a"], [period], monthly, audits) == []


def test_legacy_durable_record_without_viirs_contract_is_rejected() -> None:
    record = pd.Series({"treatment_order": 7, "status": "matched"})
    with pytest.raises(ValueError, match="complete monthly VIIRS cache contract"):
        MODULE.validate_durable_record(record, 7, "r_reference")


def test_stale_matching_schema_is_rejected() -> None:
    record = pd.Series(
        {
            "treatment_order": 7,
            "status": "matched",
            "viirs_cache_contract": MODULE.VIIRS_CACHE_CONTRACT,
            "schema": "grid_control_design_legacy",
        }
    )
    with pytest.raises(ValueError, match="stale matching schema"):
        MODULE.validate_durable_record(record, 7, "r_reference")


def test_durable_record_from_other_backend_is_rejected() -> None:
    record = pd.Series(
        {
            "treatment_order": 7,
            "status": "matched",
            "viirs_cache_contract": MODULE.VIIRS_CACHE_CONTRACT,
            "schema": MODULE.CONTROL_DESIGN_SCHEMA,
            "implementation_version": "r-reference-grid",
            "backend": "r_matching",
        }
    )
    with pytest.raises(ValueError, match="does not match requested"):
        MODULE.validate_durable_record(record, 7, "python_gpu")


def test_python_durable_record_with_stale_code_fingerprint_is_rejected() -> None:
    record = pd.Series(
        {
            "treatment_order": 7,
            "status": "matched",
            "viirs_cache_contract": MODULE.VIIRS_CACHE_CONTRACT,
            "schema": MODULE.CONTROL_DESIGN_SCHEMA,
            "implementation_version": "python-causal",
            "backend": "python_pytorch",
            "code_fingerprint": "sha256:" + "0" * 64,
        }
    )
    with pytest.raises(ValueError, match="stale matching source code"):
        MODULE.validate_durable_record(record, 7, "python_gpu")
