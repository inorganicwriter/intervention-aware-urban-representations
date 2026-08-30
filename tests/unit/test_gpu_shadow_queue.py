from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from urban_intervention.causal.gpu.contracts import (
    GPU_IMPLEMENTATION_VERSION,
    SHADOW_SCHEMA,
)
from urban_intervention.causal.gpu.provenance import (
    estimator_code_fingerprint,
    fingerprint_files,
)
from urban_intervention.causal.gpu.scheduler import TaskResult

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "causal_gpu" / "run_shadow_queue.py"
SPEC = importlib.util.spec_from_file_location("run_shadow_queue", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        dtype="float64",
        chunk_size=65_536,
        memory_fraction=0.85,
        tuning="gpu",
        max_iter=5000,
        tol=1e-5,
        formal_qualification=False,
    )


def test_completed_manifest_must_match_code_config_and_source_age(tmp_path) -> None:
    panel = tmp_path / "estimation_panel.parquet"
    panel.write_bytes(b"panel")
    output = tmp_path / "output"
    output.mkdir()
    manifest = {
        "schema": SHADOW_SCHEMA,
        "implementation_version": GPU_IMPLEMENTATION_VERSION,
        "code_fingerprint": estimator_code_fingerprint("gsc"),
        "formal_eligible": False,
        "estimator": "gsc",
        "panel": str(panel),
        "tuning_source": "gpu",
        "converged": True,
        "estimator_config": {"max_iter": 5000, "tol": 1e-5},
        "runtime": {
            "dtype": "float64",
            "chunk_size": 65_536,
            "memory_fraction": 0.85,
        },
        "parity": {"available": False, "passed": None},
        "contract_backend": "python_native",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "gsc_cv_folds.parquet").write_bytes(b"folds")
    (tmp_path / "manifest.csv").write_bytes(b"contract")
    manifest["source_fingerprints"] = fingerprint_files(
        [panel, tmp_path / "gsc_cv_folds.parquet", tmp_path / "manifest.csv"]
    )
    # Rewrite the manifest last so both source artifacts are older.
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert MODULE._already_passed(output, _args())

    changed = _args()
    changed.dtype = "float32"
    assert not MODULE._already_passed(output, changed)

    panel.write_bytes(b"other")
    assert not MODULE._already_passed(output, _args())

    panel.write_bytes(b"panel")
    manifest["source_fingerprints"] = fingerprint_files(
        [panel, tmp_path / "gsc_cv_folds.parquet", tmp_path / "manifest.csv"]
    )
    manifest["code_fingerprint"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not MODULE._already_passed(output, _args())


def test_nonconverged_result_is_a_queue_error() -> None:
    row = MODULE._queue_row(
        TaskResult(
            task_id="gsc:test",
            gpu_id=0,
            value={
                "estimator": "gsc",
                "converged": False,
                "parity": {"available": False, "passed": None},
            },
        )
    )
    assert row["status"] == "error"


def test_formal_cache_reuse_requires_the_qualification_gate(tmp_path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    output = tmp_path / "output"
    output.mkdir()
    manifest = {
        "schema": SHADOW_SCHEMA,
        "implementation_version": GPU_IMPLEMENTATION_VERSION,
        "formal_eligible": False,
        "estimator": "matching",
        "qualification_passed": False,
        "runtime": {
            "dtype": "float64",
            "chunk_size": 65_536,
            "memory_fraction": 0.85,
        },
        "parity": {"available": True, "passed": True},
        "source_fingerprints": fingerprint_files([source]),
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    args = _args()
    args.formal_qualification = True
    assert not MODULE._already_passed(output, args)


def test_formal_reference_rejects_legacy_windowed_inference(tmp_path) -> None:
    labels = tmp_path / "causal_response_labels.parquet"
    import pandas as pd

    pd.DataFrame(
        {
            "minimum_window_n": [1, 3],
            "uncertainty_source": ["gsc", "gsc_window3"],
        }
    ).to_parquet(labels, index=False)
    try:
        MODULE._assert_raw_qualification_reference(labels)
    except ValueError as error:
        assert "observation_window=1" in str(error)
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("windowed R inference was accepted for qualification")


def test_task_limit_is_stratified_by_estimator(tmp_path) -> None:
    import pandas as pd

    staging = tmp_path / "staging"
    matching = tmp_path / "matching"
    for directory in ("xu_gsc", "matrix_completion"):
        for index in range(5):
            task = staging / directory / f"task-{index}"
            task.mkdir(parents=True)
            pd.DataFrame({"x": [index]}).to_parquet(
                task / "estimation_panel.parquet", index=False
            )
            pd.DataFrame({"x": [index]}).to_parquet(
                task / "causal_response_labels.parquet", index=False
            )
    for index in range(5):
        task = matching / f"task-{index}"
        task.mkdir(parents=True)
        pd.DataFrame({"x": [index]}).to_parquet(
            task / "matching_input.parquet", index=False
        )
    args = argparse.Namespace(
        estimators="matching,gsc,mc",
        staging_root=staging,
        matching_input_root=matching,
        output_root=tmp_path / "output",
        tuning="reference",
        formal_qualification=False,
        retry=False,
        dtype="float64",
        chunk_size=65_536,
        memory_fraction=0.85,
        max_iter=5000,
        tol=1e-5,
        contract_backend="any",
        gsc_bootstrap_mode="none",
        gsc_n_bootstrap=0,
        mc_inference="none",
        inference_batch_size=16,
        gsc_inference_relative_rmse_tolerance=0.35,
        mc_inference_relative_rmse_tolerance=0.02,
        minimum_ci_zero_agreement=0.9,
        max_tasks=None,
        max_tasks_per_estimator=2,
    )
    tasks = MODULE.discover_tasks(args)
    counts: dict[str, int] = {}
    for task in tasks:
        estimator = str(task.payload["estimator"])
        counts[estimator] = counts.get(estimator, 0) + 1
    assert counts == {"gsc": 2, "mc": 2, "matching": 2}


def test_formal_mode_preserves_explicit_per_estimator_limit(monkeypatch) -> None:
    args = argparse.Namespace(
        max_tasks=None,
        max_tasks_per_estimator=5,
        formal_qualification=True,
        tuning="reference",
        contract_backend="any",
        dtype="float32",
        gsc_bootstrap_mode="none",
        gsc_n_bootstrap=0,
        mc_inference="none",
        prepare_python_contracts=False,
        rebuild_python_contracts=False,
        max_iter=5000,
        tol=1e-5,
        gpu_ids="0",
        dry_run=True,
    )
    observed: dict[str, int] = {}

    def prepare(received: argparse.Namespace) -> int:
        observed["prepared_limit"] = received.max_tasks_per_estimator
        return 0

    def discover(received: argparse.Namespace):
        observed["discovered_limit"] = received.max_tasks_per_estimator
        return []

    monkeypatch.setattr(MODULE, "parse_args", lambda: args)
    monkeypatch.setattr(MODULE, "prepare_python_contracts", prepare)
    monkeypatch.setattr(MODULE, "discover_tasks", discover)
    assert MODULE.main() == 0
    assert observed == {"prepared_limit": 5, "discovered_limit": 5}
