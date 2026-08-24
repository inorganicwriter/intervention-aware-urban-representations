"""Fail-closed audit for promoting shadow estimators to formal use."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .contracts import (
    FORMAL_IMPLEMENTATION_VERSION,
    GPU_IMPLEMENTATION_VERSION,
    SHADOW_SCHEMA,
)
from .provenance import (
    file_sha256,
    fingerprints_match,
    numerical_environment_contract,
)

QUALIFICATION_SCHEMA = "causal_gpu_formal_qualification_v2_environment_bound"


def audit_shadow_manifests(
    paths: list[str | Path],
    *,
    minimum_parity_tasks: int = 3,
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
        if not fingerprints_match(source_fingerprints, verify_content=True):
            item_reasons.append("source fingerprints are missing or stale")
        elif estimator in required_sources:
            source_names = {
                Path(str(record.get("path", ""))).name
                for record in source_fingerprints
                if isinstance(record, dict)
            }
            missing_sources = required_sources[estimator] - source_names
            if missing_sources:
                item_reasons.append(
                    "numerical source fingerprints are incomplete: "
                    + ", ".join(sorted(missing_sources))
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
                "passed": not item_reasons,
                "reasons": item_reasons,
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
        "eligible": not reasons,
        "minimum_parity_tasks": minimum_parity_tasks,
        "counts": dict(sorted(counts.items())),
        "qualified_environment": qualified_environment,
        "audited": audited,
        "reasons": reasons,
    }


def validate_formal_qualification_receipt(
    path: str | Path,
    *,
    verify_bound_sources: bool = True,
    expected_sha256: str | None = None,
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
        audited_sources: list[Path] = []
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
            audited_sources, minimum_parity_tasks=minimum
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
        "formal_qualification_source_audit": (
            "full_at_worker_start" if verify_bound_sources else "receipt_digest_prevalidated"
        ),
    }
