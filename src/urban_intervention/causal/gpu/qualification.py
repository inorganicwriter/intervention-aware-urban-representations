"""Fail-closed audit for promoting shadow estimators to formal use."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import (
    FORMAL_IMPLEMENTATION_VERSION,
    GPU_IMPLEMENTATION_VERSION,
    SHADOW_SCHEMA,
)
from .provenance import (
    all_estimator_code_fingerprints,
    estimator_code_fingerprint,
    estimator_source_files,
    file_sha256,
    fingerprints_match,
    numerical_environment_contract,
)

QUALIFICATION_SCHEMA = "causal_gpu_formal_qualification_environment_bound"
WAIVER_SCHEMA = "causal_gpu_qualification_drift_waiver"
WAIVER_ATTESTATION = "operator_reviewed_no_numerical_change"
WAIVER_ENV_VAR = "MIT_CAUSAL_QUALIFICATION_WAIVER"


def _waiver_source_hashes(waiver: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(waiver, Mapping):
        return {}
    mapping = waiver.get("source_sha256")
    return mapping if isinstance(mapping, Mapping) else {}


def _waiver_covers_estimator_code(
    waiver: Mapping[str, Any] | None, estimator: str
) -> bool:
    if not isinstance(waiver, Mapping):
        return False
    fingerprints = waiver.get("code_fingerprints")
    return isinstance(fingerprints, Mapping) and fingerprints.get(
        estimator
    ) == estimator_code_fingerprint(estimator)


def _waiver_covers_source_records(
    waiver: Mapping[str, Any] | None, records: object
) -> bool:
    """Return True only if every drifted record is covered by the waiver.

    Unchanged records do not need an entry.  Changed or missing records must
    match the waiver's recorded current hash exactly.
    """
    if not isinstance(waiver, Mapping) or not isinstance(records, list) or not records:
        return False
    mapping = _waiver_source_hashes(waiver)
    if not mapping:
        return False
    for record in records:
        if not isinstance(record, Mapping):
            return False
        path = Path(str(record.get("path", "")))
        try:
            resolved = str(path.resolve())
        except OSError:
            return False
        expected = mapping.get(resolved)
        if expected is None:
            if path.is_file() and str(record.get("sha256", "")) == file_sha256(path):
                continue
            return False
        if not path.is_file() or file_sha256(path) != str(expected):
            return False
    return True


def _waiver_covers_estimator_sources(
    waiver: Mapping[str, Any] | None, estimator: str
) -> bool:
    mapping = _waiver_source_hashes(waiver)
    if not mapping:
        return False
    for path in estimator_source_files(estimator):
        if mapping.get(str(path.resolve())) != file_sha256(path):
            return False
    return True


def load_qualification_waiver(
    path: str | Path,
    *,
    receipt_path: str | Path | None = None,
    receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and validate an operator-attested source-drift waiver.

    A waiver never overrides parity, quality, convergence or environment
    gates.  It only lets an otherwise eligible receipt survive changes to the
    qualification-bound source files that the operator has reviewed as
    non-numerical.  Pass ``receipt_path``/``receipt_sha256`` to bind the waiver
    to one receipt at load time; release-side consumers may load it unbound and
    verify the binding against each task manifest instead.
    """
    waiver_path = Path(path).resolve()
    if not waiver_path.is_file():
        raise ValueError(f"qualification waiver is unavailable: {waiver_path}")
    try:
        payload = json.loads(waiver_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable qualification waiver {waiver_path}: {exc}") from exc
    if payload.get("schema") != WAIVER_SCHEMA:
        raise ValueError("qualification waiver has an unknown schema")
    if receipt_path is not None and Path(
        str(payload.get("receipt_path", ""))
    ).resolve() != Path(receipt_path).resolve():
        raise ValueError("qualification waiver is bound to a different receipt")
    if receipt_sha256 is not None and str(payload.get("receipt_sha256", "")) != receipt_sha256:
        raise ValueError("qualification waiver was issued for a different receipt revision")
    if payload.get("attestation") != WAIVER_ATTESTATION:
        raise ValueError("qualification waiver lacks the required operator attestation")
    if not str(payload.get("reason", "")).strip():
        raise ValueError("qualification waiver lacks a reason")
    if not isinstance(payload.get("code_fingerprints"), dict):
        raise ValueError("qualification waiver lacks estimator code fingerprints")
    if not _waiver_source_hashes(payload):
        raise ValueError("qualification waiver lacks bound source hashes")
    resolved = dict(payload)
    resolved["_path"] = str(waiver_path)
    resolved["_sha256"] = file_sha256(waiver_path)
    return resolved


def audit_shadow_manifests(
    paths: Sequence[str | Path],
    *,
    minimum_parity_tasks: int = 3,
    waiver: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit artifacts without mutating or promoting any result."""
    reasons: list[str] = []
    counts: Counter[str] = Counter()
    audited: list[dict[str, Any]] = []
    qualified_environment: dict[str, Any] | None = None
    required_sources = {
        "matching": {"matching.py", "control_design.py", "fixed_control.py"},
        "gsc": {"gsc.py", "linalg.py", "inference.py"},
        "mc": {"matrix_completion.py", "linalg.py", "inference.py"},
    }
    for raw_path in paths:
        path = Path(raw_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append(f"unreadable manifest {path}: {exc}")
            continue
        estimator = str(payload.get("estimator", "unknown"))
        counts[estimator] += 1
        item_reasons: list[str] = []
        if payload.get("schema") != SHADOW_SCHEMA:
            item_reasons.append("stale schema")
        if payload.get("implementation_version") != GPU_IMPLEMENTATION_VERSION:
            item_reasons.append("stale implementation")
        if (
            estimator in required_sources
            and payload.get("code_fingerprint") != estimator_code_fingerprint(estimator)
            and not _waiver_covers_estimator_code(waiver, estimator)
        ):
            item_reasons.append("estimator source fingerprint differs from current code")
        if payload.get("formal_eligible") is not False:
            item_reasons.append("shadow manifest makes an invalid formal claim")
        if payload.get("qualification_requested") is not True:
            item_reasons.append("manifest was not produced by the formal qualification workflow")
        if payload.get("qualification_passed") is not True:
            item_reasons.append("manifest did not pass its formal qualification gate")
        runtime = payload.get("runtime", {})
        if runtime.get("dtype") != "float64":
            item_reasons.append("non-float64 runtime")
        if runtime.get("deterministic") is not True or runtime.get("allow_tf32") is not False:
            item_reasons.append("non-deterministic numerical policy")
        environment_contract, environment_reasons = numerical_environment_contract(
            payload.get("environment"), runtime
        )
        item_reasons.extend(environment_reasons)
        if environment_contract is not None and not environment_reasons:
            if qualified_environment is None:
                qualified_environment = environment_contract
            elif environment_contract != qualified_environment:
                item_reasons.append(
                    "numerical environment differs from other qualification tasks"
                )
        if estimator in {"gsc", "mc"} and payload.get("converged") is not True:
            item_reasons.append("final fit did not converge")
        if estimator in {"gsc", "mc"} and payload.get("tuning_source") != "gpu":
            item_reasons.append("formal qualification requires native GPU tuning")
        config = payload.get("estimator_config", {})
        try:
            convergence_contract_ok = (
                isinstance(config, dict)
                and int(config.get("max_iter", 0)) == 5000
                and abs(float(config.get("tol", float("nan"))) - 1e-5) < 1e-15
            )
        except (TypeError, ValueError):
            convergence_contract_ok = False
        if estimator in {"gsc", "mc"} and not convergence_contract_ok:
            item_reasons.append("formal qualification numerical convergence contract differs")
        try:
            gsc_contract_ok = (
                isinstance(config, dict)
                and config.get("gsc_bootstrap_mode") == "auto"
                and int(config.get("gsc_n_bootstrap", 0)) >= 200
            )
        except (TypeError, ValueError):
            gsc_contract_ok = False
        if estimator == "gsc" and not gsc_contract_ok:
            item_reasons.append("formal GSC qualification requires auto bootstrap with 200 draws")
        if estimator == "mc" and (
            not isinstance(config, dict)
            or config.get("mc_inference") != "jackknife"
        ):
            item_reasons.append("formal MC qualification requires unit jackknife inference")
        if estimator == "mc":
            try:
                selected_lambda = float(payload.get("selected_tuning", float("nan")))
            except (TypeError, ValueError):
                selected_lambda = float("nan")
            if not math.isfinite(selected_lambda) or selected_lambda < 0:
                item_reasons.append(
                    "formal MC qualification requires a finite non-negative selected lambda"
                )
        if estimator == "matching" and payload.get("quality_passed") is not True:
            item_reasons.append("matching quality gate did not pass")
        parity = payload.get("parity", {})
        if parity.get("available") is False or parity.get("passed") is not True:
            item_reasons.append("R/Python point-path parity is unavailable or failed")
        if estimator == "matching":
            label_parity = payload.get("label_parity", {})
            if (
                not isinstance(label_parity, dict)
                or label_parity.get("available") is not True
                or label_parity.get("passed") is not True
            ):
                item_reasons.append("R/Python final matching-label parity is unavailable or failed")
        source_fingerprints = payload.get("source_fingerprints")
        source_waived = False
        if not fingerprints_match(source_fingerprints, verify_content=True):
            if _waiver_covers_source_records(waiver, source_fingerprints):
                source_waived = True
            else:
                item_reasons.append("source fingerprints are missing or stale")
        elif estimator in required_sources:
            source_records = {
                str(Path(str(record.get("path", ""))).resolve()): str(
                    record.get("sha256", "")
                )
                for record in source_fingerprints
                if isinstance(record, dict)
            }
            source_names = {Path(path).name for path in source_records}
            missing_sources = required_sources[estimator] - source_names
            if missing_sources:
                item_reasons.append(
                    "numerical source fingerprints are incomplete: "
                    + ", ".join(sorted(missing_sources))
                )
            current_sources = {
                str(path.resolve()): file_sha256(path)
                for path in estimator_source_files(estimator)
            }
            if any(source_records.get(path) != digest for path, digest in current_sources.items()):
                if _waiver_covers_estimator_sources(waiver, estimator):
                    source_waived = True
                else:
                    item_reasons.append(
                        "qualified estimator sources do not match the current source tree"
                    )
        inference = payload.get("inference", {})
        if estimator in {"gsc", "mc"} and (
            not isinstance(inference, dict)
            or inference.get("formal_validated") is not True
        ):
            item_reasons.append("formal inference parity/coverage is not validated")
        if item_reasons:
            reasons.extend(f"{path}: {reason}" for reason in item_reasons)
        audited.append(
            {
                "path": str(path.resolve()),
                "manifest_sha256": file_sha256(path),
                "estimator": estimator,
                "code_fingerprint": payload.get("code_fingerprint"),
                "passed": not item_reasons,
                "reasons": item_reasons,
                "source_waived": bool(source_waived),
            }
        )
    for estimator in ("matching", "gsc", "mc"):
        if counts[estimator] < minimum_parity_tasks:
            reasons.append(
                f"{estimator} has {counts[estimator]} audited task(s); "
                f"minimum is {minimum_parity_tasks}"
            )
    return {
        "schema": QUALIFICATION_SCHEMA,
        "formal_implementation_version": FORMAL_IMPLEMENTATION_VERSION,
        "shadow_implementation_version": GPU_IMPLEMENTATION_VERSION,
        "code_fingerprints": all_estimator_code_fingerprints(),
        "eligible": not reasons,
        "minimum_parity_tasks": minimum_parity_tasks,
        "counts": dict(sorted(counts.items())),
        "qualified_environment": qualified_environment,
        "audited": audited,
        "reasons": reasons,
        "source_waiver": (
            str(waiver.get("_path", "")) if isinstance(waiver, Mapping) else ""
        ),
        "source_waived_tasks": sum(1 for item in audited if item["source_waived"]),
    }


def validate_formal_qualification_receipt(
    path: str | Path,
    *,
    verify_bound_sources: bool = True,
    expected_sha256: str | None = None,
    waiver_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a receipt, optionally re-auditing its bound large artifacts.

    A shard performs the full source audit once.  Its estimator subprocesses
    pass ``expected_sha256`` and use the receipt-only path, which preserves the
    exact qualification identity without repeatedly hashing every panel.
    """
    receipt = Path(path).resolve()
    receipt_sha256 = file_sha256(receipt)
    if expected_sha256 is not None and receipt_sha256 != expected_sha256:
        raise ValueError("formal qualification receipt digest differs from shard proof")
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable formal qualification receipt {receipt}: {exc}") from exc
    if payload.get("schema") != QUALIFICATION_SCHEMA:
        raise ValueError("formal qualification receipt has an unknown schema")
    if payload.get("formal_implementation_version") != FORMAL_IMPLEMENTATION_VERSION:
        raise ValueError("formal qualification receipt targets a stale implementation")
    if payload.get("shadow_implementation_version") != GPU_IMPLEMENTATION_VERSION:
        raise ValueError("formal qualification receipt targets a stale shadow contract")
    resolved_waiver_path = waiver_path
    if resolved_waiver_path is None:
        environment_waiver = os.environ.get(WAIVER_ENV_VAR, "").strip()
        if environment_waiver:
            resolved_waiver_path = environment_waiver
    waiver: dict[str, Any] | None = None
    if resolved_waiver_path is not None:
        waiver = load_qualification_waiver(
            resolved_waiver_path,
            receipt_path=receipt,
            receipt_sha256=receipt_sha256,
        )
    current_code_fingerprints = all_estimator_code_fingerprints()
    if payload.get("code_fingerprints") != current_code_fingerprints:
        waiver_fingerprints = (
            dict(waiver.get("code_fingerprints", {})) if waiver is not None else {}
        )
        if waiver is None or waiver_fingerprints != current_code_fingerprints:
            raise ValueError(
                "formal qualification receipt targets different estimator source code"
            )
    if payload.get("eligible") is not True or payload.get("reasons"):
        raise ValueError("formal qualification receipt is not eligible")
    counts = payload.get("counts")
    minimum = int(payload.get("minimum_parity_tasks", 0))
    if not isinstance(counts, dict) or minimum < 1:
        raise ValueError("formal qualification receipt lacks valid estimator counts")
    for estimator in ("matching", "gsc", "mc"):
        if int(counts.get(estimator, 0)) < minimum:
            raise ValueError(f"formal qualification receipt lacks {estimator} parity tasks")
    audited = payload.get("audited")
    if not isinstance(audited, list) or not audited:
        raise ValueError("formal qualification receipt lacks audited manifests")
    for item in audited:
        if not isinstance(item, dict) or item.get("passed") is not True:
            raise ValueError("formal qualification receipt contains a failed audit item")
        if not str(item.get("path", "")) or not str(item.get("manifest_sha256", "")):
            raise ValueError("formal qualification receipt contains an incomplete audit item")
    qualified_environment = payload.get("qualified_environment")
    if not isinstance(qualified_environment, dict):
        raise ValueError("formal qualification receipt lacks its numerical environment")
    if verify_bound_sources:
        audited_sources: list[str | Path] = []
        for item in audited:
            assert isinstance(item, dict)
            source = Path(str(item.get("path", "")))
            expected_hash = str(item.get("manifest_sha256", ""))
            if not source.is_file():
                raise ValueError(f"qualified source manifest is unavailable: {source}")
            if file_sha256(source) != expected_hash:
                raise ValueError(f"qualified source manifest changed after audit: {source}")
            try:
                json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"qualified source manifest is unreadable: {source}") from exc
            audited_sources.append(source)
        current_audit = audit_shadow_manifests(
            audited_sources, minimum_parity_tasks=minimum, waiver=waiver
        )
        if current_audit.get("eligible") is not True:
            raise ValueError(
                "a source bound by the formal qualification manifests changed; "
                "the manifests no longer satisfy the audit: "
                + "; ".join(map(str, current_audit.get("reasons", [])))
            )
        if current_audit.get("qualified_environment") != qualified_environment:
            raise ValueError("formal qualification environment changed after audit")
    return {
        "formal_qualification_eligible": True,
        "formal_qualification_receipt": str(receipt),
        "formal_qualification_receipt_sha256": receipt_sha256,
        "formal_qualification_minimum_parity_tasks": minimum,
        "formal_qualification_environment": qualified_environment,
        "formal_qualification_code_fingerprints": current_code_fingerprints,
        "formal_qualification_source_audit": (
            "full_at_worker_start" if verify_bound_sources else "receipt_digest_prevalidated"
        ),
        "formal_qualification_waiver": str(waiver["_path"]) if waiver else "",
        "formal_qualification_waiver_sha256": str(waiver["_sha256"]) if waiver else "",
        "formal_qualification_waiver_reason": (
            str(waiver.get("reason", "")) if waiver else ""
        ),
    }
