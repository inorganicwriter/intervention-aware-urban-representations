"""Behavior-preserving component of the modular causal label queue."""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import partial
from pathlib import Path

import pandas as pd

from urban_intervention.causal.gpu.contracts import (
    FORMAL_IMPLEMENTATION_VERSION,
)
from urban_intervention.utils import atomic_write_csv as atomic_csv
from urban_intervention.utils import atomic_write_json
from urban_intervention.utils import atomic_write_parquet as atomic_parquet
from urban_intervention.utils import sha256_file as file_sha256

from .gsc import (
    classify_gsc_failure,
    run_gsc_scope,
    skip_after_structural_gsc_failure,
)
from .matching import run_cross_city_matching, run_frozen_control
from .mc import run_mc_scope
from .runtime import (
    CONTROL_QUEUE,
    settings,
)
from .state import (
    _month_key,
    control_for_order,
    effective_price_measure,
    read_control_queue,
    specification_fingerprint,
    task_directory,
)
from .support import (
    ensure_cross_city_viirs,
    ensure_viirs,
    family_has_observed_support,
    family_signature,
    viirs_has_min_preperiods,
)
from .validation import recover_completed_task, validate_task_labels

atomic_json = partial(atomic_write_json, default=str)


def write_task(
    row: pd.Series,
    labels: Iterable[pd.DataFrame],
    status: str,
    method: str,
    details: dict[str, object],
) -> None:
    directory = task_directory(int(row.treatment_order), str(row.outcome_family))
    frames = list(labels)
    if not frames:
        raise ValueError("Successful task did not return any label frames")
    task_details = dict(details)
    if settings.estimator_backend == "python_gpu" and settings.run_mode == "production":
        if not settings.qualification_proof:
            raise ValueError("Production Python task lacks formal qualification proof")
        task_details.update(settings.qualification_proof)
    result = pd.concat(frames, ignore_index=True)
    result["donor_scope"] = str(task_details.get("donor_scope") or "") or pd.NA
    result["estimator_backend"] = str(
        task_details.get("backend")
        or ("python_gpu" if settings.estimator_backend == "python_gpu" else "r_reference")
    )
    result["implementation_version"] = (
        FORMAL_IMPLEMENTATION_VERSION
        if settings.estimator_backend == "python_gpu"
        else "r_reference"
    )
    if "price_measure" not in result:
        result["price_measure"] = effective_price_measure(row)
    validate_task_labels(row, result)
    if not str(task_details.get("run_id") or ""):
        raise ValueError("Production task details lack estimator run_id")
    labels_path = directory / "labels.parquet"
    atomic_parquet(result, labels_path)
    atomic_json(
        {
            "schema": "causal_response_labels_v1",
            "status": status,
            "method": method,
            "treatment_order": int(row.treatment_order),
            "outcome_family": str(row.outcome_family),
            "city_key": str(row.city_key),
            "grid_id": str(row.grid_id),
            "station_event_id": str(row.station_event_id),
            "opening_month": _month_key(row.opening_month),
            "label_rows": len(result),
            "labels_sha256": file_sha256(labels_path),
            "specification_fingerprint": specification_fingerprint(row),
            "anticipation_months": settings.anticipation_months,
            "observation_window": settings.label_window,
            "price_measure": effective_price_measure(row),
            "production_eligible": settings.run_mode == "production",
            "run_mode": settings.run_mode,
            "details": task_details,
        },
        directory / "manifest.json",
    )
    # A bounded retry may turn a previously skipped task into a valid label.
    # Remove the obsolete failure marker only after both labels and the success
    # manifest have been published atomically.
    (directory / "failure_manifest.json").unlink(missing_ok=True)


def begin_mc_stage(
    queue: pd.DataFrame, index: int, row: pd.Series, gsc_details: dict[str, object]
) -> None:
    directory = task_directory(int(row.treatment_order), str(row.outcome_family))
    atomic_json(
        {"schema": "gsc_failure_before_mc_v1", **gsc_details},
        directory / "gsc_attempt.json",
    )
    queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
        "mc_pending",
        pd.NA,
        str(gsc_details.get("reason")),
    ]
    atomic_csv(queue, settings.family_queue)


def run_mc_stage(
    queue: pd.DataFrame,
    index: int,
    row: pd.Series,
    control_queue: pd.DataFrame | None = None,
    control_queue_path: Path = CONTROL_QUEUE,
) -> None:
    """Six-round routing, rounds 3-6.

    Round 1 (same-city matching) and round 2 (same-city GSC) run in
    process_one; this stage continues from round 3:
      round 3: same-city MC
      round 4: cross-city matching (frozen control on an all-city donor)
      round 5: cross-city GSC
      round 6: cross-city MC
      all failed -> skipped
    """
    directory = task_directory(int(row.treatment_order), str(row.outcome_family))
    stage = str(row.status)
    gsc_attempt_path = directory / "gsc_attempt.json"
    cross_gsc_attempt_path = directory / "cross_city_gsc_attempt.json"
    gsc_details: dict[str, object] = (
        json.loads(gsc_attempt_path.read_text(encoding="utf-8"))
        if gsc_attempt_path.exists()
        else {"reason": "gsc_failure_details_unavailable"}
    )

    if classify_gsc_failure(gsc_details):
        skip_after_structural_gsc_failure(queue, index, row, gsc_details)
        return

    same_mc_ok = False
    same_mc_details: dict[str, object] = {"reason": "same_city_mc_not_run"}
    cross_gsc_ok = False
    cross_gsc_labels: list[pd.DataFrame] = []
    cross_gsc_details: dict[str, object] = {"reason": "cross_city_gsc_not_run"}
    cross_mc_details: dict[str, object] = {"reason": "cross_city_mc_not_run"}
    match_details: dict[str, object] = {"reason": "cross_city_matching_not_run"}

    # Round 3: same-city MC. A resumed cross-city stage must not restart
    # this expensive round.
    if stage not in {"cross_matching_running", "cross_gsc_running", "cross_mc_running"}:
        queue.loc[index, ["status", "selected_method"]] = ["mc_running", pd.NA]
        atomic_csv(queue, settings.family_queue)
        same_mc_ok, same_mc_labels, same_mc_details = run_mc_scope(row, "same_city")
        if same_mc_ok:
            selected_method = str(same_mc_details["selected_method"])
            write_task(row, same_mc_labels, "mc_labelled", selected_method, same_mc_details)
            queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
                "mc_labelled",
                selected_method,
                pd.NA,
            ]
            atomic_csv(queue, settings.family_queue)
            return

    # Round 4: cross-city matching
    if stage not in {"cross_gsc_running", "cross_mc_running"}:
        if control_queue is None:
            control_queue = read_control_queue()
        queue.loc[index, "status"] = "cross_matching_running"
        atomic_csv(queue, settings.family_queue)
        cross_match_ok = False
        if (
            str(
                control_queue.loc[
                    control_queue["treatment_order"].astype(int) == int(row.treatment_order),
                    "status",
                ].iloc[0]
            )
            == "gsc_pending"
        ):
            cross_match_ok, cross_control, cross_match_details = run_cross_city_matching(
                row, control_queue, control_queue_path
            )
            match_details = cross_match_details
            if cross_match_ok:
                cross_ok, cross_labels, cross_details = run_frozen_control(row, cross_control)
                if cross_ok:
                    selected_method = "frozen_matched_change_12m_baseline_cross_city"
                    write_task(
                        row, cross_labels, "matched_labelled", selected_method, cross_details
                    )
                    queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
                        "matched_labelled",
                        selected_method,
                        pd.NA,
                    ]
                    atomic_csv(queue, settings.family_queue)
                    return

    # Round 5: cross-city GSC. The cross-city estimator reads VIIRS
    # partitions for every donor city, so materialise the window across all
    # donor cities first (the same-city rounds only need the target city).
    if stage != "cross_mc_running":
        queue.loc[index, "status"] = "cross_gsc_running"
        atomic_csv(queue, settings.family_queue)
        if str(row.outcome_family) == "viirs":
            cached, cache_logs = ensure_cross_city_viirs(row)
            if not cached:
                cross_gsc_ok, cross_gsc_labels, cross_gsc_details = (
                    False,
                    [],
                    {"reason": "cross_city_viirs_cache_unavailable", "log": "\n".join(cache_logs)},
                )
            else:
                cross_gsc_ok, cross_gsc_labels, cross_gsc_details = run_gsc_scope(
                    row, "all_city_standardized"
                )
        else:
            cross_gsc_ok, cross_gsc_labels, cross_gsc_details = run_gsc_scope(
                row, "all_city_standardized"
            )
        if not cross_gsc_ok:
            atomic_json(
                {"schema": "cross_city_gsc_failure_v1", **cross_gsc_details},
                cross_gsc_attempt_path,
            )
    if cross_gsc_ok:
        selected_method = str(cross_gsc_details["selected_method"])
        write_task(row, cross_gsc_labels, "gsc_labelled", selected_method, cross_gsc_details)
        queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
            "gsc_labelled",
            selected_method,
            pd.NA,
        ]
        atomic_csv(queue, settings.family_queue)
        return

    # Round 6: cross-city MC (VIIRS cache already materialised in round 5
    # for the same donor scope, so no repeated materialisation here).
    if stage == "cross_mc_running" and cross_gsc_attempt_path.exists():
        cross_gsc_details = json.loads(cross_gsc_attempt_path.read_text(encoding="utf-8"))
    queue.loc[index, "status"] = "cross_mc_running"
    atomic_csv(queue, settings.family_queue)
    cross_mc_ok, cross_mc_labels, cross_mc_details = run_mc_scope(row, "all_city_standardized")
    if cross_mc_ok:
        selected_method = str(cross_mc_details["selected_method"])
        write_task(row, cross_mc_labels, "mc_labelled", selected_method, cross_mc_details)
        queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
            "mc_labelled",
            selected_method,
            pd.NA,
        ]
        atomic_csv(queue, settings.family_queue)
        return

    # All six rounds failed -> skipped
    matching_attempt_path = directory / "matching_attempt.json"
    matching_details = (
        json.loads(matching_attempt_path.read_text(encoding="utf-8"))
        if matching_attempt_path.exists()
        else {"reason": None}
    )
    atomic_json(
        {
            "schema": "causal_response_labels_v1",
            "status": "skipped",
            "method": None,
            "treatment_order": int(row.treatment_order),
            "outcome_family": str(row.outcome_family),
            "matching_failure": matching_details,
            "gsc_failure": gsc_details,
            "same_city_mc_failure": same_mc_details,
            "cross_city_matching_failure": match_details,
            "cross_city_gsc_failure": cross_gsc_details,
            "cross_city_mc_failure": cross_mc_details,
        },
        directory / "failure_manifest.json",
    )
    queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
        "skipped",
        pd.NA,
        str(cross_mc_details.get("reason")),
    ]
    atomic_csv(queue, settings.family_queue)


def process_one(
    queue: pd.DataFrame,
    index: int,
    support: pd.DataFrame,
    control_queue: pd.DataFrame,
    dry_run: bool,
    phase: str = "all",
    retry_matching: bool = False,
    control_queue_path: Path = CONTROL_QUEUE,
) -> None:
    row = queue.loc[index].copy()
    if recover_completed_task(queue, index, control_queue=control_queue):
        return
    signature = family_signature(row, support)
    control = control_for_order(int(row.treatment_order), control_queue)
    if dry_run:
        print(
            {
                "order": int(row.treatment_order),
                "family": row.outcome_family,
                "signature": signature,
                "control_status": str(control.status),
                "control_unit_key": control.get("control_unit_key", pd.NA),
            }
        )
        return
    if str(control.status) not in {"matched", "gsc_pending"}:
        raise RuntimeError(
            f"Control design for treatment {int(row.treatment_order)} is {control.status}; "
            "run the grid-control queue first"
        )
    if not family_has_observed_support(row):
        # The grid has no observation anywhere in this family's panel: GSC
        # and MC cannot estimate anything and would only fail after minutes
        # of R work.  Skip immediately with a dedicated reason.
        queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
            "skipped",
            pd.NA,
            "family_no_observed_support",
        ]
        atomic_csv(queue, settings.family_queue)
        return
    if str(row.status) in {
        "mc_pending",
        "mc_running",
        "cross_matching_running",
        "cross_gsc_running",
        "cross_mc_running",
    } and phase in {"all", "mc"}:
        run_mc_stage(queue, index, row, control_queue, control_queue_path)
        return
    if (
        str(control.status) == "gsc_pending" or str(row.status) in {"gsc_pending", "gsc_running"}
    ) and phase != "matching":
        queue.loc[index, "status"] = "gsc_running"
        atomic_csv(queue, settings.family_queue)
        if row.outcome_family == "viirs":
            if not viirs_has_min_preperiods(str(row.opening_month)):
                gsc_ok, gsc_labels, gsc_details = (
                    False,
                    [],
                    {"reason": "viirs_insufficient_clean_pre_periods_for_gsc"},
                )
            else:
                cached = ensure_viirs(row, require_full_matching_window=False)
                if cached.returncode != 0:
                    gsc_ok, gsc_labels, gsc_details = (
                        False,
                        [],
                        {"reason": "monthly_viirs_cache_unavailable", "log": cached.stdout},
                    )
                else:
                    # Round 2: same-city GSC only (cross-city GSC is round 5).
                    gsc_ok, gsc_labels, gsc_details = run_gsc_scope(row, "same_city")
        else:
            gsc_ok, gsc_labels, gsc_details = run_gsc_scope(row, "same_city")
        if gsc_ok:
            selected_method = str(gsc_details["selected_method"])
            write_task(row, gsc_labels, "gsc_labelled", selected_method, gsc_details)
            queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
                "gsc_labelled",
                selected_method,
                pd.NA,
            ]
        else:
            if classify_gsc_failure(gsc_details):
                skip_after_structural_gsc_failure(queue, index, row, gsc_details)
            else:
                begin_mc_stage(queue, index, row, gsc_details)
            if not classify_gsc_failure(gsc_details) and phase != "gsc":
                run_mc_stage(queue, index, row, control_queue, control_queue_path)
        if gsc_ok:
            atomic_csv(queue, settings.family_queue)
        return
    queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
        "matching_running",
        pd.NA,
        pd.NA,
    ]
    atomic_csv(queue, settings.family_queue)

    matching_ok, matched_labels, match_details = run_frozen_control(row, control)
    if matching_ok:
        write_task(
            row,
            matched_labels,
            "matched_labelled",
            "frozen_matched_change_12m_baseline",
            match_details,
        )
        queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
            "matched_labelled",
            "frozen_matched_change_12m_baseline",
            pd.NA,
        ]
        atomic_csv(queue, settings.family_queue)
        return

    queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
        "gsc_pending",
        pd.NA,
        str(match_details.get("reason")),
    ]
    atomic_json(
        {"schema": "preonly_matching_attempt_v1", **match_details},
        task_directory(int(row.treatment_order), str(row.outcome_family)) / "matching_attempt.json",
    )
    atomic_csv(queue, settings.family_queue)
    if phase == "matching":
        return
    queue.loc[index, "status"] = "gsc_running"
    atomic_csv(queue, settings.family_queue)
    if row.outcome_family == "viirs":
        if not viirs_has_min_preperiods(str(row.opening_month)):
            gsc_ok, gsc_labels, gsc_details = (
                False,
                [],
                {"reason": "viirs_insufficient_clean_pre_periods_for_gsc"},
            )
        else:
            cached = ensure_viirs(row, require_full_matching_window=False)
            if cached.returncode != 0:
                gsc_ok, gsc_labels, gsc_details = (
                    False,
                    [],
                    {"reason": "monthly_viirs_cache_unavailable", "log": cached.stdout},
                )
            else:
                # Round 2: same-city GSC only.
                gsc_ok, gsc_labels, gsc_details = run_gsc_scope(row, "same_city")
    else:
        gsc_ok, gsc_labels, gsc_details = run_gsc_scope(row, "same_city")
    if gsc_ok:
        selected_method = str(gsc_details["selected_method"])
        write_task(row, gsc_labels, "gsc_labelled", selected_method, gsc_details)
        queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
            "gsc_labelled",
            selected_method,
            pd.NA,
        ]
    else:
        if classify_gsc_failure(gsc_details):
            skip_after_structural_gsc_failure(queue, index, row, gsc_details)
        else:
            begin_mc_stage(queue, index, row, gsc_details)
        if not classify_gsc_failure(gsc_details) and phase != "gsc":
            run_mc_stage(queue, index, row, control_queue, control_queue_path)
    if gsc_ok:
        atomic_csv(queue, settings.family_queue)
