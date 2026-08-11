from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
COLLECTION = ROOT / "scripts" / "collection"
sys.path.insert(0, str(COLLECTION))
SCRIPT = COLLECTION / "ensure_viirs_monthly_cache.py"
SPEC = importlib.util.spec_from_file_location("ensure_viirs_monthly_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_requested_periods_is_closed_month_range() -> None:
    assert MODULE.requested_periods("2019-11", "2020-02") == [
        "2019-11",
        "2019-12",
        "2020-01",
        "2020-02",
    ]


def test_existing_partition_and_audit_are_not_reprocessed(tmp_path: Path) -> None:
    output = tmp_path / "curated"
    audits = tmp_path / "audits"
    part = output / "city_key=beijing" / "year=2012" / "month=01" / "part.parquet"
    audit = audits / "beijing" / "2012-01.json"
    part.parent.mkdir(parents=True)
    audit.parent.mkdir(parents=True)
    pd.DataFrame({"grid_id": ["g1"], "avg_rad": [-0.5]}).to_parquet(part)
    audit.write_text("{}", encoding="utf-8")

    manifest = MODULE.ensure_partitions(tmp_path / "raw", "beijing", ["2012-01"], output, audits)

    assert manifest["already_cached"] == ["2012-01"]
    assert manifest["processed"] == []
    assert manifest["raw_input_dir"] == str((tmp_path / "raw").resolve())


def test_cached_partition_does_not_require_raw_input_directory(tmp_path: Path) -> None:
    output = tmp_path / "curated"
    audits = tmp_path / "audits"
    part = output / "city_key=beijing" / "year=2012" / "month=01" / "part.parquet"
    audit = audits / "beijing" / "2012-01.json"
    part.parent.mkdir(parents=True)
    audit.parent.mkdir(parents=True)
    pd.DataFrame({"grid_id": ["g1"], "avg_rad": [-0.5]}).to_parquet(part)
    audit.write_text("{}", encoding="utf-8")

    manifest = MODULE.ensure_partitions(None, "beijing", ["2012-01"], output, audits)

    assert manifest["processed"] == []
    assert manifest["raw_input_dir"] is None


def test_uncached_partition_requires_configured_raw_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="MIT_VIIRS_RAW"):
        MODULE.ensure_partitions(
            None, "beijing", ["2012-01"], tmp_path / "curated", tmp_path / "audits"
        )
