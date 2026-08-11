from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analysis" / "audit_housing_acquisition.py"
SPEC = importlib.util.spec_from_file_location("audit_housing_acquisition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _target(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        source="lianjia",
        city="beijing",
        page_type="chengjiao",
        inventory_path=tmp_path / "inventory.json",
        manifest_path=tmp_path / "manifest.jsonl",
        output_path=tmp_path / "parsed.csv",
    )


def test_target_complete_requires_terminal_outcomes_and_traceable_rows(tmp_path: Path) -> None:
    target = _target(tmp_path)
    capture = {
        "timestamp": "20200102030405",
        "original": "https://bj.lianjia.com/chengjiao/",
    }
    target.inventory_path.write_text(json.dumps({"captures": [capture]}), encoding="utf-8")
    key = "20200102030405\thttps://bj.lianjia.com/chengjiao/"
    target.manifest_path.write_text(
        json.dumps({"capture_key": key, "status": "ok", "rows": 1}) + "\n",
        encoding="utf-8",
    )
    target.output_path.write_text(
        "snapshot_date,original_url,community,unit_price\n"
        "20200102030405,https://bj.lianjia.com/chengjiao/,测试小区,10000\n",
        encoding="utf-8",
    )

    audit, unfinished = MODULE.audit_target(target)

    assert audit.target_complete
    assert audit.parsed_rows == 1
    assert audit.orphan_parsed_rows == 0
    assert unfinished == []


def test_retryable_capture_is_reported(tmp_path: Path) -> None:
    target = _target(tmp_path)
    capture = {"timestamp": "20200102030405", "original": "https://example.test/"}
    target.inventory_path.write_text(json.dumps({"captures": [capture]}), encoding="utf-8")
    key = "20200102030405\thttps://example.test/"
    target.manifest_path.write_text(
        json.dumps({"capture_key": key, "status": "request_error", "error": "timeout"}) + "\n",
        encoding="utf-8",
    )

    audit, unfinished = MODULE.audit_target(target)

    assert not audit.target_complete
    assert audit.retryable_captures == 1
    assert unfinished[0]["latest_status"] == "request_error"
