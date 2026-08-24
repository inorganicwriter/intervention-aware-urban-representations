"""Behavior-preserving component of the modular causal label queue."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from urban_intervention.causal.gpu.contracts import (
    FORMAL_IMPLEMENTATION_VERSION,
)
from urban_intervention.causal.gpu.control_design import (
    design_grid_control,
    write_control_design,
)
from urban_intervention.causal.gpu.fixed_control import fixed_control_labels
from urban_intervention.utils import atomic_write_csv as atomic_csv
from urban_intervention.utils import atomic_write_json
from urban_intervention.utils import atomic_write_parquet as atomic_parquet
from urban_intervention.utils import sha256_file as file_sha256

from .runtime import (
    CONTROL_QUEUE,
    OUTPUT_CONTROL_TASKS_DIR,
    OUTPUT_FIXED_CONTROL_DIR,
    R_SCRIPT,
    ROOT,
    r_script,
    settings,
)
from .state import (
    control_for_order,
    effective_price_measure,
    new_run_id,
    read_estimator_manifest,
    run,
    specification_fingerprint,
)
from .support import ensure_viirs

atomic_json = partial(atomic_write_json, default=str)


def fixed_control_output(row: pd.Series) -> Path:
    root = (
        OUTPUT_FIXED_CONTROL_DIR
        if settings.run_mode == "production"
        else OUTPUT_FIXED_CONTROL_DIR.with_name(
            f"{OUTPUT_FIXED_CONTROL_DIR.name}_{settings.run_mode}"
        )
    )
    family = str(row.outcome_family)
    if family == "housing" and settings.transaction_count_threshold != 1:
        family = f"{family}_tx{settings.transaction_count_threshold}"
    return root / f"{int(row.treatment_order):05d}" / family


def run_cross_city_matching(
    row: pd.Series, control_queue: pd.DataFrame, control_queue_path: Path = CONTROL_QUEUE
) -> tuple[bool, pd.Series, dict[str, object]]:
    """Round 4: cross-city matching, invoked after same-city GSC/MC failed.

    The Phase-1 control queue is an immutable same-city design record.  A
    cross-city control is persisted beside the treatment task and returned
    directly to the requesting outcome-family task; it must never promote the
    shared grid-level row to ``matched`` because that would make later outcome
    families skip their same-city GSC/MC rounds.
    """
    del control_queue_path  # retained for backward-compatible callers
    order = int(row.treatment_order)
    original = control_for_order(order, control_queue)
    if str(original.status) != "gsc_pending":
        return (
            False,
            pd.Series(dtype=object),
            {
                "reason": "cross_city_matching_requires_immutable_same_city_failure",
                "record_status": str(original.status),
            },
        )
    cached = settings.cross_city_design_cache.get(order)
    if cached is not None:
        cached_record, cached_details = cached
        return True, cached_record.copy(), dict(cached_details)
    task_root = OUTPUT_CONTROL_TASKS_DIR
    cross_signature = (
        "cross_city" if settings.run_mode == "production" else f"cross_city_{settings.run_mode}"
    )
    cross_city_output = task_root / f"{order:05d}" / cross_signature
    environment = os.environ.copy()
    mit_r_lib = os.environ.get("MIT_R_LIB")
    if mit_r_lib and Path(mit_r_lib).is_dir():
        environment["R_LIBS_USER"] = mit_r_lib
    if settings.estimator_backend == "python_gpu":
        try:
            design = design_grid_control(
                order,
                scope="all_city_standardized",
                root=ROOT,
                device=os.environ.get("MIT_CAUSAL_DEVICE", "auto"),
            )
            write_control_design(design, cross_city_output)
            completed_log = "Python cross-city control design completed"
        except Exception as error:
            return (
                False,
                pd.Series(dtype=object),
                {
                    "reason": "python_cross_city_control_design_runtime_error",
                    "log": str(error),
                },
            )
    else:
        completed = run(
            [
                str(R_SCRIPT),
                str(r_script("run_cross_city_control_design.R")),
                str(order),
                str(task_root),
            ],
            {"MIT_CAUSAL_RUN_ID": new_run_id()},
        )
        if completed.returncode != 0:
            return (
                False,
                pd.Series(dtype=object),
                {
                    "reason": "cross_city_control_design_runtime_error",
                    "log": completed.stdout,
                },
            )
        completed_log = completed.stdout
    attempt_path = cross_city_output / "cross_city_attempt.csv"
    if attempt_path.exists():
        attempt = pd.read_csv(attempt_path).iloc[0]
        return (
            False,
            pd.Series(dtype=object),
            {
                "reason": str(attempt.get("failure_reason", "cross_city_not_matched")),
                "record_status": str(attempt.get("status")),
            },
        )
    record_path = cross_city_output / "control_record.csv"
    if not record_path.exists():
        return (
            False,
            pd.Series(dtype=object),
            {
                "reason": "cross_city_control_record_missing",
                "log": completed_log,
            },
        )
    record = pd.read_csv(record_path).iloc[0]
    if str(record.get("status")) != "matched":
        return (
            False,
            pd.Series(dtype=object),
            {
                "reason": str(record.get("failure_reason", "cross_city_not_matched")),
                "record_status": str(record.get("status")),
            },
        )
    details = {
        "control_unit_key": str(record.get("control_unit_key")),
        "donor_scope": "all_city_standardized",
        "control_record": str(record_path.relative_to(ROOT)),
    }
    settings.cross_city_design_cache[order] = (record.copy(), details.copy())
    return True, record, details


def run_frozen_control(
    row: pd.Series, control: pd.Series
) -> tuple[bool, list[pd.DataFrame], dict[str, object]]:
    family = str(row.outcome_family)
    if str(control.status) != "matched":
        return False, [], {"reason": "grid_control_design_not_matched"}
    if family == "viirs":
        control_city = control.control_city_key
        if pd.isna(control_city):
            return False, [], {"reason": "frozen_control_city_key_is_na"}
        # The frozen-control baseline window is opening-18..opening-7; the
        # monthly VIIRS cache floor is 2012-01, so earlier openings can never
        # satisfy the 12-month baseline.  Gate before spawning R instead of
        # burning a doomed matching run.
        opening = pd.Period(str(row.opening_month), freq="M")
        if opening - 18 < pd.Period("2012-01", freq="M"):
            return False, [], {"reason": "viirs_insufficient_baseline_for_frozen_control"}
        for city in sorted({str(row.city_key), str(control_city)}):
            cached = ensure_viirs(row, require_full_matching_window=False, city_key=city)
            if cached.returncode != 0:
                return (
                    False,
                    [],
                    {"reason": "monthly_viirs_cache_unavailable", "log": cached.stdout},
                )
    output = fixed_control_output(row)
    run_id = new_run_id()
    if settings.estimator_backend == "python_gpu":
        try:
            raw = fixed_control_labels(
                int(row.treatment_order),
                str(control.control_city_key),
                str(control.control_grid_id),
                family,
                root=ROOT,
                window=settings.label_window,
                price_measure=effective_price_measure(row),
                transaction_count_threshold=settings.transaction_count_threshold,
            )
            labels_path = output / "causal_response_labels.parquet"
            atomic_parquet(raw, labels_path)
            availability = raw[
                ["outcome", "event_time", "label_available", "treated_baseline", "control_baseline"]
            ]
            atomic_csv(availability, output / "label_availability.csv")
            manifest_values: dict[str, object] = {
                "schema": "fixed_control_labels_v2_python",
                "implementation_version": FORMAL_IMPLEMENTATION_VERSION,
                "run_id": run_id,
                "estimator": "frozen_matched_change",
                "backend": "python_numpy",
                "treatment_order": int(row.treatment_order),
                "specification_fingerprint": specification_fingerprint(row),
                "outcome_family": family,
                "control_city_key": str(control.control_city_key),
                "control_grid_id": str(control.control_grid_id),
                "run_mode": settings.run_mode,
                "production_eligible": settings.run_mode == "production",
                "window": settings.label_window,
                "price_measure": effective_price_measure(row),
                "transaction_count_threshold": settings.transaction_count_threshold,
                "labels_sha256": file_sha256(labels_path),
                **settings.qualification_proof,
            }
            estimator_manifest = output / "manifest.csv"
            atomic_csv(
                pd.DataFrame({"field": manifest_values.keys(), "value": manifest_values.values()}),
                estimator_manifest,
            )
            completed_log = "Python fixed-control labels completed"
        except Exception as error:
            return (
                False,
                [],
                {
                    "reason": "python_fixed_control_label_runtime_error",
                    "log": str(error),
                },
            )
    else:
        completed = run(
            [
                str(R_SCRIPT),
                str(r_script("run_fixed_control_labels.R")),
                str(int(row.treatment_order)),
                str(control.control_city_key),
                str(control.control_grid_id),
                family,
                str(output),
                str(settings.label_window),
                effective_price_measure(row),
            ],
            {
                "MIT_CAUSAL_RUN_ID": run_id,
                "MIT_SPECIFICATION_FINGERPRINT": specification_fingerprint(row),
                "MIT_CAUSAL_RUN_MODE": settings.run_mode,
            },
        )
        if completed.returncode != 0:
            return (
                False,
                [],
                {"reason": "fixed_control_label_runtime_error", "log": completed.stdout},
            )
        labels_path = output / "causal_response_labels.parquet"
        estimator_manifest = output / "manifest.csv"
        completed_log = completed.stdout
    if not labels_path.exists() or not estimator_manifest.exists():
        return (
            False,
            [],
            {"reason": "fixed_control_manifest_or_labels_missing", "log": completed_log},
        )
    manifest_values = read_estimator_manifest(estimator_manifest)
    if (
        manifest_values.get("run_id") != run_id
        or manifest_values.get("estimator") != "frozen_matched_change"
        or manifest_values.get("run_mode") != settings.run_mode
        or manifest_values.get("production_eligible", "").upper()
        != ("TRUE" if settings.run_mode == "production" else "FALSE")
        or manifest_values.get("treatment_order") != str(int(row.treatment_order))
        or manifest_values.get("outcome_family") != family
        or manifest_values.get("specification_fingerprint") != specification_fingerprint(row)
        or manifest_values.get("transaction_count_threshold")
        != str(settings.transaction_count_threshold)
        or (
            settings.estimator_backend == "python_gpu"
            and settings.run_mode == "production"
            and manifest_values.get("formal_qualification_receipt_sha256")
            != str(settings.qualification_proof.get("formal_qualification_receipt_sha256", ""))
        )
    ):
        return (
            False,
            [],
            {
                "reason": "fixed_control_manifest_does_not_prove_current_production_run",
                "run_id": run_id,
                "log": completed_log,
            },
        )
    raw = pq.read_table(labels_path).to_pandas()
    raw["transformed_scale"] = True
    raw["specification_id"] = "main_a6_r1km"
    raw["specification_fingerprint"] = specification_fingerprint(row)
    raw["outcome_family"] = family
    raw["method"] = "frozen_matched_change_12m_baseline"
    if "control_unit_key" not in raw.columns:
        raw["control_unit_key"] = str(control.control_unit_key)
    if not bool(raw["label_available"].any()):
        return (
            False,
            [],
            {
                "reason": "frozen_control_has_no_available_outcome_label",
                "control_unit_key": str(control.control_unit_key),
                "log": completed_log,
            },
        )
    control_record = (
        OUTPUT_CONTROL_TASKS_DIR / f"{int(row.treatment_order):05d}" / "control_record.csv"
    )
    return (
        True,
        [raw],
        {
            "control_unit_key": str(control.control_unit_key),
            "donor_scope": str(control.donor_scope),
            "available_labels": int(raw["label_available"].sum()),
            "requested_labels": len(raw),
            "fixed_control_labels_sha256": file_sha256(labels_path),
            "fixed_control_manifest_sha256": file_sha256(estimator_manifest),
            "run_id": run_id,
            "control_record": str(control_record.relative_to(ROOT))
            if control_record.exists()
            else None,
            "control_record_sha256": file_sha256(control_record)
            if control_record.exists()
            else None,
            "log": completed_log,
        },
    )
