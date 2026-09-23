"""Operator-attested spot-check waivers for control-design source drift.

Control records bind the matching source fingerprint that produced them.  A
production tree assembled from reviewed local patches can carry a stored
fingerprint that no longer matches the current tree even though the numerical
control design is unchanged.  This module accepts such records only when an
operator-attested waiver documents spot-check evidence: a minimum sample of
orders recomputed under the current tree with field-by-field identical output.
The waiver never overrides schema, backend, leakage, queue-status or method
checks; those still fail the release independently.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .provenance import (
    estimator_code_fingerprint,
    estimator_source_files,
    file_sha256,
)

CONTROL_DESIGN_WAIVER_SCHEMA = "causal_control_design_drift_waiver"
CONTROL_DESIGN_WAIVER_ATTESTATION = "operator_attested_spot_check_identical"
CONTROL_DESIGN_WAIVER_ENV_VAR = "MIT_CAUSAL_CONTROL_DESIGN_WAIVER"
MINIMUM_SPOT_CHECK_ORDERS = 12
SPOT_CHECK_FIELDS = (
    "status",
    "selected_method",
    "donor_scope",
    "control_city_key",
    "control_grid_id",
    "control_unit_key",
    "training_distance",
    "holdout_rms_standardized_gap",
    "holdout_max_abs_standardized_gap",
)


def matching_source_hashes() -> dict[str, str]:
    """Hash every matching-bound source file at its resolved path."""
    return {
        str(path.resolve()): file_sha256(path)
        for path in estimator_source_files("matching")
    }


def _spot_check_evidence(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    evidence = payload.get("spot_check")
    return evidence if isinstance(evidence, Mapping) else None


def load_control_design_waiver(path: str | Path) -> dict[str, Any]:
    """Load and validate an operator-attested control-design drift waiver."""
    waiver_path = Path(path).resolve()
    if not waiver_path.is_file():
        raise ValueError(f"control-design waiver is unavailable: {waiver_path}")
    try:
        payload = json.loads(waiver_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable control-design waiver {waiver_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("control-design waiver must be a JSON object")
    if payload.get("schema") != CONTROL_DESIGN_WAIVER_SCHEMA:
        raise ValueError("control-design waiver has an unknown schema")
    if payload.get("attestation") != CONTROL_DESIGN_WAIVER_ATTESTATION:
        raise ValueError("control-design waiver lacks the required operator attestation")
    if not str(payload.get("reason", "")).strip():
        raise ValueError("control-design waiver lacks a reason")
    fingerprints = payload.get("code_fingerprints")
    if not isinstance(fingerprints, Mapping) or fingerprints.get(
        "matching"
    ) != estimator_code_fingerprint("matching"):
        raise ValueError(
            "control-design waiver was issued for a different matching source revision"
        )
    bound = payload.get("source_sha256")
    if not isinstance(bound, Mapping) or not bound:
        raise ValueError("control-design waiver lacks bound source hashes")
    for source in estimator_source_files("matching"):
        source_path = str(source.resolve())
        if not source.is_file() or str(bound.get(source_path, "")) != file_sha256(source):
            raise ValueError(
                "control-design waiver no longer matches the matching source tree"
            )
    accepted = payload.get("accepted_code_fingerprints")
    if (
        not isinstance(accepted, list)
        or not accepted
        or not all(isinstance(value, str) and value for value in accepted)
    ):
        raise ValueError("control-design waiver lacks accepted record fingerprints")
    evidence = _spot_check_evidence(payload)
    if evidence is None:
        raise ValueError("control-design waiver lacks spot-check evidence")
    if evidence.get("all_identical") is not True:
        raise ValueError("control-design waiver spot check is not fully identical")
    raw_minimum = evidence.get("min_orders", MINIMUM_SPOT_CHECK_ORDERS)
    if isinstance(raw_minimum, bool) or not isinstance(raw_minimum, int):
        raise ValueError("control-design waiver spot check lacks a minimum order count")
    minimum = raw_minimum
    if minimum < MINIMUM_SPOT_CHECK_ORDERS:
        raise ValueError(
            "control-design waiver spot check must cover at least "
            f"{MINIMUM_SPOT_CHECK_ORDERS} orders"
        )
    orders = evidence.get("orders")
    if not isinstance(orders, list) or len(orders) < minimum:
        raise ValueError("control-design waiver spot-check sample is too small")
    fields = evidence.get("fields")
    if not isinstance(fields, list) or not set(SPOT_CHECK_FIELDS).issubset(
        {str(field) for field in fields}
    ):
        raise ValueError("control-design waiver spot check lacks compared fields")
    resolved = dict(payload)
    resolved["_path"] = str(waiver_path)
    resolved["_sha256"] = file_sha256(waiver_path)
    return resolved


def waiver_accepts_record_fingerprint(
    waiver: Mapping[str, Any] | None, stored_fingerprint: object
) -> bool:
    """Return True only when the waiver explicitly covers this fingerprint."""
    if not isinstance(waiver, Mapping):
        return False
    accepted = waiver.get("accepted_code_fingerprints")
    if not isinstance(accepted, list):
        return False
    return str(stored_fingerprint) in {str(value) for value in accepted}
