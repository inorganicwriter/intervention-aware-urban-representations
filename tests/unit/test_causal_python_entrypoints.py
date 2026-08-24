from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = (
    "scripts/causal_python/prepare_causal_inputs.py",
    "scripts/causal_python/run_all_method_event_study.py",
    "scripts/causal_python/run_control_design_batch.py",
    "scripts/causal_python/run_formal_estimator.py",
    "scripts/causal_python/run_matching_event_study.py",
    "scripts/causal_python/run_causal_label_queue.py",
)


@pytest.mark.parametrize("relative_path", ENTRYPOINTS)
def test_causal_python_entrypoint_help(relative_path: str) -> None:
    completed = subprocess.run(
        [sys.executable, relative_path, "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout
    assert "usage:" in completed.stdout


def test_legacy_causal_r_queue_wrapper_remains_executable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/causal_r/run_causal_label_queue.py", "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout
    assert "--estimator-backend" in completed.stdout


def test_formal_entrypoint_separates_transaction_sensitivity_outputs() -> None:
    path = ROOT / "scripts" / "causal_python" / "run_formal_estimator.py"
    spec = importlib.util.spec_from_file_location("run_formal_estimator_entrypoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = {
        "frequency": "monthly",
        "opening_month": "2020-07",
        "outcome": "housing_log_price",
        "outcome_family": "housing",
        "treatment_order": 7,
        "donor_scope": "same_city",
        "city_key": "alpha",
    }
    default = module.output_directory(metadata, "gsc", "preview", 1)
    sensitivity = module.output_directory(metadata, "gsc", "preview", 3)
    assert default != sensitivity
    assert sensitivity.name.endswith("_tx3")
