from __future__ import annotations

import json

from urban_intervention.causal.gpu.contracts import (
    GPU_IMPLEMENTATION_VERSION,
    SHADOW_SCHEMA,
)
from urban_intervention.causal.gpu.provenance import (
    estimator_code_fingerprint,
    estimator_source_files,
    fingerprint_files,
)
from urban_intervention.causal.gpu.qualification import (
    audit_shadow_manifests,
    validate_formal_qualification_receipt,
)


def _manifest(tmp_path, estimator: str):
    source = tmp_path / f"{estimator}.bin"
    source.write_bytes(estimator.encode())
    required = {
        "matching": ("matching.py", "control_design.py", "fixed_control.py"),
        "gsc": ("gsc.py", "linalg.py", "inference.py"),
        "mc": ("matrix_completion.py", "linalg.py", "inference.py"),
    }[estimator]
    source_directory = tmp_path / f"{estimator}_sources"
    source_directory.mkdir()
    source_paths = []
    for name in required:
        path = source_directory / name
        path.write_text(f"{estimator}:{name}", encoding="utf-8")
        source_paths.append(path)
    path = tmp_path / f"{estimator}.json"
    payload = {
        "schema": SHADOW_SCHEMA,
        "implementation_version": GPU_IMPLEMENTATION_VERSION,
        "code_fingerprint": estimator_code_fingerprint(estimator),
        "estimator": estimator,
        "formal_eligible": False,
        "converged": True,
        "tuning_source": "gpu",
        "quality_passed": True,
        "qualification_requested": True,
        "qualification_passed": True,
        "estimator_config": {
            "max_iter": 5000,
            "tol": 1e-5,
            "gsc_bootstrap_mode": "auto" if estimator == "gsc" else "none",
            "gsc_n_bootstrap": 200 if estimator == "gsc" else 0,
            "mc_inference": "jackknife" if estimator == "mc" else "none",
        },
        "selected_tuning": 0.0 if estimator == "mc" else 1,
        "runtime": {
            "dtype": "float64",
            "deterministic": True,
            "allow_tf32": False,
            "resolved_device": "cuda:0",
            "cuda_version": "test-cuda",
            "cudnn_version": 90000,
        },
        "environment": {
            "python_version": "3.11.test",
            "python_implementation": "CPython",
            "packages": {
                "numpy": "test",
                "pandas": "test",
                "pyarrow": "test",
                "torch": "test",
            },
        },
        "parity": {"available": True, "passed": True},
        "label_parity": {"available": True, "passed": True},
        "source_fingerprints": fingerprint_files(
            [source, *source_paths, *estimator_source_files(estimator)]
        ),
        "inference": {"formal_validated": True},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_formal_qualification_requires_all_estimators_and_inference(tmp_path) -> None:
    paths = [_manifest(tmp_path, estimator) for estimator in ("matching", "gsc", "mc")]
    passed = audit_shadow_manifests(paths, minimum_parity_tasks=1)
    assert passed["eligible"]

    payload = json.loads(paths[1].read_text(encoding="utf-8"))
    payload["inference"]["formal_validated"] = False
    paths[1].write_text(json.dumps(payload), encoding="utf-8")
    blocked = audit_shadow_manifests(paths, minimum_parity_tasks=1)
    assert not blocked["eligible"]
    assert any("formal inference" in reason for reason in blocked["reasons"])


def test_formal_qualification_rejects_unknown_code_fingerprint(tmp_path) -> None:
    paths = [_manifest(tmp_path, estimator) for estimator in ("matching", "gsc", "mc")]
    payload = json.loads(paths[1].read_text(encoding="utf-8"))
    payload["code_fingerprint"] = "sha256:" + "0" * 64
    paths[1].write_text(json.dumps(payload), encoding="utf-8")

    blocked = audit_shadow_manifests(paths, minimum_parity_tasks=1)

    assert not blocked["eligible"]
    assert any("source fingerprint differs" in reason for reason in blocked["reasons"])


def test_qualification_receipt_is_bound_to_audited_manifests(tmp_path) -> None:
    paths = [_manifest(tmp_path, estimator) for estimator in ("matching", "gsc", "mc")]
    report = audit_shadow_manifests(paths, minimum_parity_tasks=1)
    receipt = tmp_path / "qualification.json"
    receipt.write_text(json.dumps(report), encoding="utf-8")
    proof = validate_formal_qualification_receipt(receipt)
    assert proof["formal_qualification_eligible"] is True
    assert proof["formal_qualification_environment"]["cuda_version"] == "test-cuda"

    paths[0].write_text("{}", encoding="utf-8")
    try:
        validate_formal_qualification_receipt(receipt)
    except ValueError as error:
        assert "changed after audit" in str(error)
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("tampered qualification source was accepted")


def test_qualification_receipt_rejects_stale_code_map(tmp_path) -> None:
    paths = [_manifest(tmp_path, estimator) for estimator in ("matching", "gsc", "mc")]
    report = audit_shadow_manifests(paths, minimum_parity_tasks=1)
    report["code_fingerprints"]["gsc"] = "sha256:" + "0" * 64
    receipt = tmp_path / "qualification.json"
    receipt.write_text(json.dumps(report), encoding="utf-8")

    try:
        validate_formal_qualification_receipt(receipt, verify_bound_sources=False)
    except ValueError as error:
        assert "different estimator source code" in str(error)
    else:  # pragma: no cover
        raise AssertionError("stale qualification code map was accepted")


def test_qualification_receipt_rechecks_bound_source_content(tmp_path) -> None:
    paths = [_manifest(tmp_path, estimator) for estimator in ("matching", "gsc", "mc")]
    report = audit_shadow_manifests(paths, minimum_parity_tasks=1)
    receipt = tmp_path / "qualification.json"
    receipt.write_text(json.dumps(report), encoding="utf-8")
    manifest = json.loads(paths[0].read_text(encoding="utf-8"))
    bound_source = manifest["source_fingerprints"][0]["path"]
    with open(bound_source, "ab") as handle:
        handle.write(b"changed")
    try:
        validate_formal_qualification_receipt(receipt)
    except ValueError as error:
        assert "source bound" in str(error)
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("changed qualified source was accepted")


def test_prevalidated_receipt_checks_digest_without_rehashing_bound_sources(tmp_path) -> None:
    paths = [_manifest(tmp_path, estimator) for estimator in ("matching", "gsc", "mc")]
    report = audit_shadow_manifests(paths, minimum_parity_tasks=1)
    receipt = tmp_path / "qualification.json"
    receipt.write_text(json.dumps(report), encoding="utf-8")
    proof = validate_formal_qualification_receipt(receipt)

    paths[0].write_text("{}", encoding="utf-8")
    summary = validate_formal_qualification_receipt(
        receipt,
        verify_bound_sources=False,
        expected_sha256=proof["formal_qualification_receipt_sha256"],
    )
    assert summary["formal_qualification_source_audit"] == "receipt_digest_prevalidated"


def test_handcrafted_receipt_cannot_bypass_current_manifest_audit(tmp_path) -> None:
    paths = [_manifest(tmp_path, estimator) for estimator in ("matching", "gsc", "mc")]
    report = audit_shadow_manifests(paths, minimum_parity_tasks=1)
    matching = json.loads(paths[0].read_text(encoding="utf-8"))
    matching["qualification_requested"] = False
    paths[0].write_text(json.dumps(matching), encoding="utf-8")
    report["audited"][0]["manifest_sha256"] = fingerprint_files([paths[0]])[0][
        "sha256"
    ]
    receipt = tmp_path / "qualification.json"
    receipt.write_text(json.dumps(report), encoding="utf-8")
    try:
        validate_formal_qualification_receipt(receipt)
    except ValueError as error:
        assert "no longer satisfy" in str(error)
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("handcrafted eligible receipt bypassed the current audit")
