from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "causal_gpu" / "export_matching_qualification_set.py"
SPEC = importlib.util.spec_from_file_location("export_matching_qualification_set", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_reference_quality_gate_requires_one_accepted_selection(tmp_path: Path) -> None:
    selection = tmp_path / "reference_selection.csv"
    selection.write_text("accepted,metric\nTRUE,1\n", encoding="utf-8")
    assert MODULE.reference_quality_passed(selection)

    selection.write_text("accepted,metric\nFALSE,1\n", encoding="utf-8")
    assert not MODULE.reference_quality_passed(selection)

    selection.write_text("accepted,metric\nTRUE,1\nTRUE,2\n", encoding="utf-8")
    assert not MODULE.reference_quality_passed(selection)
