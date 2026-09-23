from __future__ import annotations

import json

import pytest

from scripts.causal_gpu.issue_qualification_waiver import build_waiver
from tests.unit.test_gpu_qualification import _manifest
from urban_intervention.causal.gpu import qualification
from urban_intervention.causal.gpu.provenance import (
    estimator_source_files,
    file_sha256,
)
from urban_intervention.causal.gpu.qualification import (
    WAIVER_ATTESTATION,
    WAIVER_SCHEMA,
    audit_shadow_manifests,
    load_qualification_waiver,
    validate_formal_qualification_receipt,
)


def _waiver_payload(receipt, drifted_fingerprints: dict[str, str]) -> dict[str, object]:
    bound = {
        str(path.resolve()): file_sha256(path)
        for estimator in ("matching", "gsc", "mc")
        for path in estimator_source_files(estimator)
    }
    return {
        "schema": WAIVER_SCHEMA,
        "created_utc": "2026-09-19T00:00:00+00:00",
        "receipt_path": str(receipt.resolve()),
        "receipt_sha256": file_sha256(receipt),
        "code_fingerprints": drifted_fingerprints,
        "source_sha256": bound,
        "changed_sources": [
            {
                "path": path,
                "recorded_sha256": "sha256:" + "0" * 64,
                "current_sha256": digest,
            }
            for path, digest in bound.items()
        ],
        "reason": "comments and typing only; reviewed diff",
        "attestation": WAIVER_ATTESTATION,
        "operator": "test",
    }


def _simulate_drift(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    drifted = {estimator: f"sha256:{estimator}-drift" for estimator in ("matching", "gsc", "mc")}
    monkeypatch.setattr(
        qualification,
        "estimator_code_fingerprint",
        lambda estimator: drifted[estimator],
    )
    monkeypatch.setattr(
        qualification,
        "all_estimator_code_fingerprints",
        lambda: dict(drifted),
    )
    monkeypatch.setattr(
        qualification,
        "fingerprints_match",
        lambda records, verify_content=False: False,
    )
    return drifted


def test_waiver_enables_reuse_after_reviewed_source_drift(tmp_path, monkeypatch) -> None:
    paths = [_manifest(tmp_path, estimator) for estimator in ("matching", "gsc", "mc")]
    report = audit_shadow_manifests(paths, minimum_parity_tasks=1)
    receipt = tmp_path / "qualification.json"
    receipt.write_text(json.dumps(report), encoding="utf-8")

    drifted = _simulate_drift(monkeypatch)
    with pytest.raises(ValueError, match="different estimator source code"):
        validate_formal_qualification_receipt(receipt)

    waiver = tmp_path / "qualification_drift_waiver.json"
    waiver.write_text(json.dumps(_waiver_payload(receipt, drifted)), encoding="utf-8")
    proof = validate_formal_qualification_receipt(receipt, waiver_path=waiver)

    assert proof["formal_qualification_eligible"] is True
    assert proof["formal_qualification_code_fingerprints"] == drifted
    assert proof["formal_qualification_waiver_sha256"] == file_sha256(waiver)
    assert proof["formal_qualification_waiver_reason"]


def test_waiver_requires_attestation_binding_and_reason(tmp_path, monkeypatch) -> None:
    paths = [_manifest(tmp_path, estimator) for estimator in ("matching", "gsc", "mc")]
    report = audit_shadow_manifests(paths, minimum_parity_tasks=1)
    receipt = tmp_path / "qualification.json"
    receipt.write_text(json.dumps(report), encoding="utf-8")
    drifted = _simulate_drift(monkeypatch)
    payload = _waiver_payload(receipt, drifted)

    payload["attestation"] = "self_asserted"
    waiver = tmp_path / "bad_attestation.json"
    waiver.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="operator attestation"):
        load_qualification_waiver(
            waiver, receipt_path=receipt, receipt_sha256=file_sha256(receipt)
        )

    payload["attestation"] = WAIVER_ATTESTATION
    payload["reason"] = "   "
    waiver.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reason"):
        load_qualification_waiver(
            waiver, receipt_path=receipt, receipt_sha256=file_sha256(receipt)
        )

    payload["reason"] = "reviewed"
    payload["receipt_sha256"] = "sha256:" + "0" * 64
    waiver.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="different receipt revision"):
        load_qualification_waiver(
            waiver, receipt_path=receipt, receipt_sha256=file_sha256(receipt)
        )


def test_waiver_does_not_override_parity_failure(tmp_path, monkeypatch) -> None:
    paths = [_manifest(tmp_path, estimator) for estimator in ("matching", "gsc", "mc")]
    report = audit_shadow_manifests(paths, minimum_parity_tasks=1)
    receipt = tmp_path / "qualification.json"
    receipt.write_text(json.dumps(report), encoding="utf-8")
    drifted = _simulate_drift(monkeypatch)

    payload = json.loads(paths[2].read_text(encoding="utf-8"))
    payload["qualification_passed"] = False
    paths[2].write_text(json.dumps(payload), encoding="utf-8")

    waiver_payload = _waiver_payload(receipt, drifted)
    provisional = dict(waiver_payload, _path="", _sha256="")
    blocked = audit_shadow_manifests(paths, minimum_parity_tasks=1, waiver=provisional)

    assert not blocked["eligible"]
    assert any("qualification gate" in reason for reason in blocked["reasons"])
    assert blocked["source_waived_tasks"] >= 2


def test_issuer_refuses_bound_input_drift(tmp_path) -> None:
    receipt = tmp_path / "qualification.json"
    receipt.write_text("{}", encoding="utf-8")
    code = tmp_path / "code.py"
    code.write_text("value = 1\n", encoding="utf-8")
    input_path = tmp_path / "panel.parquet"
    input_path.write_text("input\n", encoding="utf-8")

    with pytest.raises(ValueError, match="input artifacts"):
        build_waiver(
            receipt,
            reason="test",
            recorded_sources={
                str(code.resolve()): "sha256:" + "0" * 64,
                str(input_path.resolve()): "sha256:" + "1" * 64,
            },
            current_sources={str(code.resolve()): file_sha256(code)},
        )

    waiver, changed = build_waiver(
        receipt,
        reason="test",
        recorded_sources={str(code.resolve()): "sha256:" + "0" * 64},
        current_sources={str(code.resolve()): file_sha256(code)},
    )
    assert changed == [str(code.resolve())]
    assert waiver["source_sha256"][str(code.resolve())] == file_sha256(code)


def test_waiver_loads_unbound_for_release_consumers(tmp_path, monkeypatch) -> None:
    paths = [_manifest(tmp_path, estimator) for estimator in ("matching", "gsc", "mc")]
    report = audit_shadow_manifests(paths, minimum_parity_tasks=1)
    receipt = tmp_path / "qualification.json"
    receipt.write_text(json.dumps(report), encoding="utf-8")
    drifted = _simulate_drift(monkeypatch)
    waiver = tmp_path / "qualification_drift_waiver.json"
    waiver.write_text(json.dumps(_waiver_payload(receipt, drifted)), encoding="utf-8")

    unbound = load_qualification_waiver(waiver)
    assert unbound["_sha256"] == file_sha256(waiver)
    assert unbound["receipt_sha256"] == file_sha256(receipt)

    with pytest.raises(ValueError, match="different receipt"):
        load_qualification_waiver(
            waiver, receipt_path=tmp_path / "other.json", receipt_sha256=file_sha256(receipt)
        )
    with pytest.raises(ValueError, match="different receipt revision"):
        load_qualification_waiver(
            waiver, receipt_path=receipt, receipt_sha256="sha256:" + "0" * 64
        )
