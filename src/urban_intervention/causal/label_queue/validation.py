"""Behavior-preserving component of the modular causal label queue."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pandas as pd

from urban_intervention.utils import atomic_write_csv as atomic_csv
from urban_intervention.utils import atomic_write_json
from urban_intervention.utils import sha256_file as file_sha256

from .runtime import (
    HORIZONS,
    OUTCOMES,
    ROOT,
    settings,
)
from .state import (
    _month_key,
    control_for_order,
    specification_fingerprint,
    task_directory,
)

atomic_json = partial(atomic_write_json, default=str)


def _validate_control_provenance(
    row: pd.Series,
    payload: dict[str, object],
    control_queue: pd.DataFrame,
) -> None:
    """Validate the control record actually used by a completed label."""
    details = payload.get("details")
    if not isinstance(details, dict):
        raise ValueError("Completed matched task lacks control provenance details")
    scope = str(details.get("donor_scope") or "")
    if scope not in {"same_city", "all_city_standardized"}:
        raise ValueError("Completed matched task has an unknown donor scope")
    control_key = str(details.get("control_unit_key") or "")
    if not control_key:
        raise ValueError("Completed matched task lacks control_unit_key provenance")
    record_value = str(details.get("control_record") or "")
    record_hash = str(details.get("control_record_sha256") or "")
    if not control_key or not record_value or len(record_hash) != 64:
        raise ValueError("Completed matched task lacks hashed control-record provenance")
    record_path = Path(record_value)
    if not record_path.is_absolute():
        record_path = ROOT / record_path
    if not record_path.is_file() or file_sha256(record_path) != record_hash:
        raise ValueError("Completed matched task control record is missing or has changed")
    record = pd.read_csv(record_path)
    if len(record) != 1:
        raise ValueError("Completed matched task control record is not unique")
    record_row = record.iloc[0]
    if int(record_row.get("treatment_order", -1)) != int(row.treatment_order):
        raise ValueError("Completed matched task control record has the wrong treatment order")
    if str(record_row.get("status", "")) != "matched":
        raise ValueError("Completed matched task control record is not matched")
    if str(record_row.get("control_unit_key", "")) != control_key:
        raise ValueError("Completed matched task control_unit_key disagrees with record")
    if str(record_row.get("donor_scope", scope)) not in {scope, "", "nan"}:
        raise ValueError("Completed matched task donor scope disagrees with record")
    queue_row = control_for_order(int(row.treatment_order), control_queue)
    if scope == "same_city":
        if str(queue_row.get("status", "")) != "matched":
            raise ValueError(
                f"Recovery refused: same-city control design for treatment "
                f"{int(row.treatment_order)} is {queue_row.get('status')}, not matched"
            )
        if str(queue_row.get("control_unit_key", "")) != control_key:
            raise ValueError("Completed matched task control differs from same-city queue")
    elif str(queue_row.get("status", "")) not in {"gsc_pending", "matched"}:
        raise ValueError("Completed cross-city task has an invalid source queue status")


def validate_task_labels(row: pd.Series, labels: pd.DataFrame) -> None:
    keys = [
        "treatment_order",
        "outcome_family",
        "outcome",
        "event_time",
        "specification_id",
        "specification_fingerprint",
    ]
    identity = ["city_key", "grid_id", "opening_month"]
    missing_identity = set(identity) - set(labels.columns)
    if missing_identity:
        raise ValueError(
            f"Normalized task labels lack identity columns: {sorted(missing_identity)}"
        )
    if labels.empty:
        raise ValueError("Successful task cannot contain zero label rows")
    if "treatment_order" not in labels.columns:
        raise ValueError("Normalized task labels lack required columns: ['treatment_order']")
    expected_order = int(row.treatment_order)
    orders = set(pd.to_numeric(labels["treatment_order"], errors="raise").astype(int))
    if orders != {expected_order}:
        raise ValueError(
            f"Task labels treatment_order {sorted(orders)} disagree with {expected_order}"
        )
    missing = set(keys) - set(labels.columns)
    if missing:
        raise ValueError(f"Normalized task labels lack required columns: {sorted(missing)}")
    expected_family = str(row.outcome_family)
    families = set(labels["outcome_family"].astype(str))
    if families != {expected_family}:
        raise ValueError(
            f"Task labels outcome_family {sorted(families)} disagree with {expected_family}"
        )
    for column in ("city_key", "grid_id"):
        values = set(labels[column].astype(str))
        expected = str(row[column])
        if values != {expected}:
            raise ValueError(f"Task labels {column} {sorted(values)} disagree with {expected!r}")
    months = {_month_key(value) for value in labels["opening_month"]}
    expected_month = _month_key(row.opening_month)
    if months != {expected_month}:
        raise ValueError(
            f"Task labels opening_month {sorted(months)} disagree with {expected_month!r}"
        )
    specifications = set(labels["specification_id"].astype(str))
    if specifications != {"main_a6_r1km"}:
        raise ValueError(f"Unexpected task specification_id: {sorted(specifications)}")
    fingerprints = set(labels["specification_fingerprint"].astype(str))
    expected_fingerprint = specification_fingerprint(row)
    if fingerprints != {expected_fingerprint}:
        raise ValueError(
            "Unexpected task specification_fingerprint: "
            f"{sorted(fingerprints)}; expected {expected_fingerprint}"
        )
    allowed_outcomes = set(OUTCOMES[expected_family])
    outcomes = set(labels["outcome"].astype(str))
    if not outcomes.issubset(allowed_outcomes):
        raise ValueError(
            f"Task labels contain outcomes outside {expected_family}: {sorted(outcomes)}"
        )
    allowed_horizons = set(HORIZONS[expected_family])
    horizons = set(pd.to_numeric(labels["event_time"], errors="raise").astype(int))
    if not horizons.issubset(allowed_horizons):
        raise ValueError(f"Task labels contain invalid event times: {sorted(horizons)}")
    # A multi-outcome MC family may publish the outcomes that converged while
    # recording structured failures for the remaining outcomes.  Therefore
    # completeness is enforced within each published outcome, not across the
    # entire family.  This still rejects a truncated outcome path.
    actual_event_time = pd.to_numeric(labels["event_time"], errors="raise").astype(int)
    for outcome in sorted(outcomes):
        actual_horizons = set(actual_event_time[labels["outcome"].astype(str) == outcome])
        if actual_horizons != allowed_horizons:
            missing_horizons = sorted(allowed_horizons - actual_horizons)
            extra_horizons = sorted(actual_horizons - allowed_horizons)
            raise ValueError(
                f"Task labels for outcome {outcome!r} do not contain the complete "
                f"horizon grid; missing={missing_horizons}, extra={extra_horizons}"
            )
    if labels.duplicated(keys).any():
        raise ValueError("Normalized task labels violate their primary key")


def validate_task_manifest(row: pd.Series, payload: dict[str, object], labels_path: Path) -> None:
    if payload.get("schema") != "causal_response_labels_v1":
        raise ValueError("Completed task has an unknown manifest schema")
    if int(payload.get("treatment_order", -1)) != int(row.treatment_order):  # type: ignore[call-overload]
        raise ValueError("Completed task manifest treatment_order disagrees with queue")
    if str(payload.get("outcome_family")) != str(row.outcome_family):
        raise ValueError("Completed task manifest outcome_family disagrees with queue")
    for column in ("city_key", "grid_id", "station_event_id"):
        if str(payload.get(column)) != str(row[column]):
            raise ValueError(f"Completed task manifest {column} disagrees with queue")
    if _month_key(payload.get("opening_month")) != _month_key(row.opening_month):
        raise ValueError("Completed task manifest opening_month disagrees with queue")
    expected_hash = str(payload.get("labels_sha256") or "")
    if not expected_hash:
        raise ValueError("Completed task manifest lacks labels_sha256")
    if expected_hash != file_sha256(labels_path):
        raise ValueError("Completed task label hash disagrees with manifest")
    expected_production = settings.run_mode == "production"
    if payload.get("run_mode") != settings.run_mode:
        raise ValueError("Completed task manifest run_mode disagrees with current queue mode")
    if payload.get("production_eligible") is not expected_production:
        raise ValueError("Completed task manifest production eligibility disagrees with queue mode")
    expected_fingerprint = specification_fingerprint(row)
    if payload.get("specification_fingerprint") != expected_fingerprint:
        raise ValueError(
            "Completed task manifest specification_fingerprint disagrees with current run"
        )
    details = payload.get("details")
    if not isinstance(details, dict) or not str(details.get("run_id") or ""):
        raise ValueError("Completed task manifest lacks current estimator run_id")
    if settings.estimator_backend == "python_gpu" and expected_production:
        expected_receipt = str(
            settings.qualification_proof.get("formal_qualification_receipt_sha256", "")
        )
        if (
            not expected_receipt
            or details.get("formal_qualification_receipt_sha256") != expected_receipt
        ):
            raise ValueError("Completed Python task lacks the current formal qualification receipt")


def recover_completed_task(
    queue: pd.DataFrame, index: int, control_queue: pd.DataFrame | None = None
) -> bool:
    row = queue.loc[index]
    manifest = task_directory(int(row.treatment_order), str(row.outcome_family)) / "manifest.json"
    labels = manifest.parent / "labels.parquet"
    if (
        row.status
        in {
            "matching_running",
            "gsc_pending",
            "gsc_running",
            "mc_pending",
            "mc_running",
            "cross_matching_running",
            "cross_gsc_running",
            "cross_mc_running",
        }
        and manifest.exists()
        and labels.exists()
    ):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("status") in {"matched_labelled", "gsc_labelled", "mc_labelled"}:
            # A valid task from a previous specification must be recomputed,
            # not adopted merely because its identity/hash is internally
            # consistent.  This is especially important when resuming after
            # changing the monthly window or housing price measure.
            if payload.get("specification_fingerprint") != specification_fingerprint(row):
                return False
            try:
                validate_task_manifest(row, payload, labels)
            except ValueError as exc:
                expected_receipt = str(
                    settings.qualification_proof.get(
                        "formal_qualification_receipt_sha256", ""
                    )
                )
                if (
                    str(exc)
                    != "Completed Python task lacks the current formal qualification receipt"
                    or not expected_receipt
                ):
                    raise
                manifest.unlink(missing_ok=True)
                labels.unlink(missing_ok=True)
                print(
                    "Invalidated completed task bound to a stale qualification receipt: "
                    f"{int(row.treatment_order)}/{row.outcome_family}"
                )
                return False
            if payload.get("status") == "matched_labelled" and control_queue is not None:
                _validate_control_provenance(row, payload, control_queue)
            stored_labels = pd.read_parquet(labels)
            validate_task_labels(row, stored_labels)
            if int(payload.get("label_rows", -1)) != len(stored_labels):
                raise ValueError("Completed task manifest row count disagrees with labels")
            queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
                payload["status"],
                payload["method"],
                pd.NA,
            ]
            atomic_csv(queue, settings.family_queue)
            return True
    return False


def invalidate_stale_terminal_tasks(queue: pd.DataFrame, orders: set[int]) -> int:
    """Return successful queue rows to pending when their spec is stale."""
    terminal = {"matched_labelled", "gsc_labelled", "mc_labelled"}
    changed = 0
    for index in queue.index[
        queue["treatment_order"].isin(orders) & queue["status"].isin(terminal)
    ]:
        row = queue.loc[index]
        manifest = (
            task_directory(int(row.treatment_order), str(row.outcome_family)) / "manifest.json"
        )
        payload = None
        if manifest.exists():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
        expected_receipt = ""
        if settings.estimator_backend == "python_gpu" and settings.run_mode == "production":
            expected_receipt = str(
                settings.qualification_proof.get(
                    "formal_qualification_receipt_sha256", ""
                )
            )
        receipt_is_stale = bool(expected_receipt) and (
            not isinstance(payload, dict)
            or not isinstance(payload.get("details"), dict)
            or payload["details"].get("formal_qualification_receipt_sha256")
            != expected_receipt
        )
        if (
            isinstance(payload, dict)
            and payload.get("specification_fingerprint") == specification_fingerprint(row)
            and not receipt_is_stale
        ):
            continue
        queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
            "pending",
            pd.NA,
            "stale_specification_invalidated",
        ]
        changed += 1
    if changed:
        atomic_csv(queue, settings.family_queue)
    return changed
