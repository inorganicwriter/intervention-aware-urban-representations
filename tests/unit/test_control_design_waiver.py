from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.causal_gpu.issue_control_design_waiver import compare_spot_check_records
from urban_intervention.causal.gpu.contracts import (
    CONTROL_DESIGN_PROVENANCE,
    CONTROL_DESIGN_SCHEMA,
    CONTROL_DESIGN_VIIRS_CACHE_CONTRACT,
)
from urban_intervention.causal.gpu.control_design_waiver import (
    CONTROL_DESIGN_WAIVER_ATTESTATION,
    CONTROL_DESIGN_WAIVER_SCHEMA,
    MINIMUM_SPOT_CHECK_ORDERS,
    SPOT_CHECK_FIELDS,
    load_control_design_waiver,
    matching_source_hashes,
    waiver_accepts_record_fingerprint,
)
from urban_intervention.causal.gpu.provenance import estimator_code_fingerprint
from urban_intervention.causal.response_artifact import (
    validate_control_design_provenance,
)

LEGACY_FINGERPRINT = "sha256:legacy-tree"


def _record(order: int, *, fingerprint: str, **overrides: object) -> dict[str, object]:
    provenance = CONTROL_DESIGN_PROVENANCE["python_gpu"]
    record: dict[str, object] = {
        "treatment_order": order,
        "schema": CONTROL_DESIGN_SCHEMA,
        "implementation_version": provenance["implementation_version"],
        "backend": provenance["backend"],
        "viirs_cache_contract": CONTROL_DESIGN_VIIRS_CACHE_CONTRACT,
        "status": "gsc_pending",
        "selected_method": "",
        "donor_scope": "",
        "control_city_key": "",
        "control_grid_id": "",
        "control_unit_key": "",
        "training_distance": "",
        "holdout_rms_standardized_gap": "",
        "holdout_max_abs_standardized_gap": "",
        "control_selection_uses_post_outcome": False,
        "code_fingerprint": fingerprint,
    }
    record.update(overrides)
    return record


def _control_inputs(
    tmp_path: Path, *, fingerprint: str = LEGACY_FINGERPRINT
) -> tuple[pd.DataFrame, Path]:
    root = tmp_path / "control_tasks"
    order_dir = root / "00001"
    order_dir.mkdir(parents=True)
    pd.DataFrame([_record(1, fingerprint=fingerprint)]).to_csv(
        order_dir / "control_record.csv", index=False
    )
    queue = pd.DataFrame({"treatment_order": [1], "status": ["gsc_pending"]})
    return queue, root


def _waiver_payload(
    accepted: list[str], *, orders: list[int] | None = None
) -> dict[str, object]:
    if orders is None:
        orders = list(range(1, MINIMUM_SPOT_CHECK_ORDERS + 1))
    return {
        "schema": CONTROL_DESIGN_WAIVER_SCHEMA,
        "created_utc": "2026-09-23T00:00:00+00:00",
        "code_fingerprints": {"matching": estimator_code_fingerprint("matching")},
        "source_sha256": matching_source_hashes(),
        "accepted_code_fingerprints": accepted,
        "spot_check": {
            "root": "outputs/control_design/spot_check",
            "orders": orders,
            "fields": list(SPOT_CHECK_FIELDS),
            "all_identical": True,
            "min_orders": MINIMUM_SPOT_CHECK_ORDERS,
        },
        "reason": "server tree assembled from reviewed local patches; spot checks identical",
        "attestation": CONTROL_DESIGN_WAIVER_ATTESTATION,
        "operator": "test",
    }


def _load_payload(tmp_path: Path, payload: dict[str, object]) -> dict[str, object]:
    path = tmp_path / "control_design_drift_waiver.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_control_design_waiver(path)


def test_waiver_accepts_covered_stale_fingerprint(tmp_path: Path) -> None:
    queue, root = _control_inputs(tmp_path)
    waiver = _load_payload(tmp_path, _waiver_payload([LEGACY_FINGERPRINT]))
    record_paths = validate_control_design_provenance(
        queue, control_task_root=root, drift_waiver=waiver
    )
    assert record_paths == [root / "00001" / "control_record.csv"]


def test_stale_fingerprint_is_rejected_without_waiver(tmp_path: Path) -> None:
    queue, root = _control_inputs(tmp_path)
    with pytest.raises(ValueError, match="stale matching source code"):
        validate_control_design_provenance(queue, control_task_root=root)


def test_waiver_must_cover_the_record_fingerprint(tmp_path: Path) -> None:
    queue, root = _control_inputs(tmp_path)
    waiver = _load_payload(tmp_path, _waiver_payload(["sha256:some-other-tree"]))
    with pytest.raises(ValueError, match="stale matching source code"):
        validate_control_design_provenance(
            queue, control_task_root=root, drift_waiver=waiver
        )


def test_waiver_accepts_current_fingerprint_through_validation(tmp_path: Path) -> None:
    queue, root = _control_inputs(
        tmp_path, fingerprint=estimator_code_fingerprint("matching")
    )
    assert validate_control_design_provenance(queue, control_task_root=root)
    assert waiver_accepts_record_fingerprint(None, "anything") is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.pop("attestation"), "required operator attestation"),
        (lambda payload: payload.update(reason=""), "lacks a reason"),
        (
            lambda payload: payload.update(code_fingerprints={"matching": "sha256:old"}),
            "different matching source revision",
        ),
        (
            lambda payload: payload.update(accepted_code_fingerprints=[]),
            "lacks accepted record fingerprints",
        ),
        (lambda payload: payload.pop("spot_check"), "lacks spot-check evidence"),
        (
            lambda payload: payload["spot_check"].update(all_identical=False),
            "not fully identical",
        ),
        (
            lambda payload: payload["spot_check"].update(orders=[1, 2, 3]),
            "sample is too small",
        ),
        (
            lambda payload: payload["spot_check"].update(min_orders=3),
            "at least 12 orders",
        ),
        (
            lambda payload: payload["spot_check"].update(fields=["status"]),
            "lacks compared fields",
        ),
    ],
)
def test_loader_rejects_tampered_waivers(tmp_path: Path, mutate, message) -> None:
    payload = _waiver_payload([LEGACY_FINGERPRINT])
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        _load_payload(tmp_path, payload)


def test_loader_rejects_bound_source_drift(tmp_path: Path) -> None:
    payload = _waiver_payload([LEGACY_FINGERPRINT])
    bound = dict(payload["source_sha256"])
    bound[next(iter(bound))] = "sha256:" + "0" * 64
    payload["source_sha256"] = bound
    with pytest.raises(ValueError, match="no longer matches the matching source tree"):
        _load_payload(tmp_path, payload)


def test_compare_spot_check_records_flags_field_mismatch(tmp_path: Path) -> None:
    production = tmp_path / "production"
    spot_check = tmp_path / "spot_check"
    for root, unit_key in ((production, "shanghai:g00001"), (spot_check, "shanghai:g00002")):
        order_dir = root / "00001"
        order_dir.mkdir(parents=True)
        pd.DataFrame(
            [_record(1, fingerprint=LEGACY_FINGERPRINT, control_unit_key=unit_key)]
        ).to_csv(order_dir / "control_record.csv", index=False)

    mismatches = compare_spot_check_records(production, spot_check, [1])
    assert mismatches == [
        {
            "treatment_order": "1",
            "field": "control_unit_key",
            "recorded": "shanghai:g00001",
            "spot_check": "shanghai:g00002",
        }
    ]
    assert compare_spot_check_records(production, production, [1]) == []
