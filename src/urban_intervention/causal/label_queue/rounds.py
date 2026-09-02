"""Crash-safe global rounds for the causal-label fallback cascade."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Literal

import pandas as pd

from urban_intervention.utils import atomic_write_csv as atomic_csv
from urban_intervention.utils import atomic_write_json

from .gsc import classify_gsc_failure, run_gsc_scope, skip_after_structural_gsc_failure
from .matching import run_cross_city_matching, run_frozen_control
from .mc import run_mc_scope
from .orchestrator import write_task
from .runtime import CONTROL_QUEUE, settings
from .state import read_control_queue, task_directory
from .support import ensure_cross_city_viirs
from .validation import recover_completed_task as frozen_recover_completed_task

atomic_json = partial(atomic_write_json, default=str)

GlobalRound = Literal[
    "same_city_matching",
    "same_city_gsc",
    "same_city_mc",
    "cross_city_matching",
    "cross_city_gsc",
    "cross_city_mc",
]

ROUND_STATUSES: dict[GlobalRound, frozenset[str]] = {
    "same_city_matching": frozenset({"pending", "matching_running"}),
    "same_city_gsc": frozenset({"gsc_pending", "gsc_running"}),
    "same_city_mc": frozenset({"mc_pending", "mc_running"}),
    "cross_city_matching": frozenset({"cross_matching_running"}),
    "cross_city_gsc": frozenset({"cross_gsc_running"}),
    "cross_city_mc": frozenset({"cross_mc_running"}),
}

_STALE_QUALIFICATION_MESSAGE = (
    "Completed Python task lacks the current formal qualification receipt"
)


def recover_completed_task_or_recompute_stale_qualification(
    queue: pd.DataFrame,
    index: int,
    control_queue: pd.DataFrame | None = None,
) -> bool:
    """Discard only an otherwise-valid task bound to an old receipt.

    The frozen recovery validator checks identity, specification, label hash,
    run metadata, and finally the qualification receipt.  Reaching its final
    receipt error therefore proves the cache is internally valid but stale.
    Every earlier validation failure remains fail-closed.
    """
    try:
        return frozen_recover_completed_task(
            queue,
            index,
            control_queue=control_queue,
        )
    except ValueError as exc:
        expected_receipt = str(
            settings.qualification_proof.get(
                "formal_qualification_receipt_sha256", ""
            )
        )
        if str(exc) != _STALE_QUALIFICATION_MESSAGE or not expected_receipt:
            raise
        row = queue.loc[index]
        directory = task_directory(
            int(row.treatment_order), str(row.outcome_family)
        )
        manifest = directory / "manifest.json"
        labels = directory / "labels.parquet"
        manifest.unlink(missing_ok=True)
        labels.unlink(missing_ok=True)
        print(
            "Invalidated completed task bound to a stale qualification receipt: "
            f"{int(row.treatment_order)}/{row.outcome_family}"
        )
        return False


def eligible_round_indices(
    queue: pd.DataFrame,
    start_order: int,
    end_order: int | None,
    family: str | None,
    phase: str,
    max_tasks: int,
    retry_matching: bool = False,
    retry_skipped: bool = False,
    orders: set[int] | None = None,
    tasks: set[tuple[int, str]] | None = None,
    *,
    round_name: GlobalRound,
) -> pd.Index:
    """Select only tasks owned by one global fallback round."""
    del phase, retry_matching, retry_skipped
    mask = queue["status"].astype(str).isin(ROUND_STATUSES[round_name])
    if orders is not None:
        mask &= queue["treatment_order"].isin(orders)
    else:
        mask &= queue["treatment_order"].ge(start_order)
        if end_order is not None:
            mask &= queue["treatment_order"].le(end_order)
    if family is not None:
        mask &= queue["outcome_family"].eq(family)
    if tasks is not None:
        keys = pd.MultiIndex.from_arrays(
            [
                queue["treatment_order"].astype(int),
                queue["outcome_family"].astype(str),
            ]
        )
        mask &= keys.isin(tasks)
    return queue.index[mask][:max_tasks]


def _read_attempt(path: Path, fallback: str) -> dict[str, object]:
    if not path.exists():
        return {"reason": fallback}
    return json.loads(path.read_text(encoding="utf-8"))


def _publish_success(
    queue: pd.DataFrame,
    index: int,
    row: pd.Series,
    labels: list[pd.DataFrame],
    status: str,
    method: str,
    details: dict[str, object],
) -> None:
    write_task(row, labels, status, method, details)
    queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
        status,
        method,
        pd.NA,
    ]
    atomic_csv(queue, settings.family_queue)


def _same_city_mc(queue: pd.DataFrame, index: int, row: pd.Series) -> None:
    directory = task_directory(int(row.treatment_order), str(row.outcome_family))
    directory.mkdir(parents=True, exist_ok=True)
    gsc_details = _read_attempt(directory / "gsc_attempt.json", "gsc_failure_details_unavailable")
    if classify_gsc_failure(gsc_details):
        skip_after_structural_gsc_failure(queue, index, row, gsc_details)
        return
    queue.loc[index, ["status", "selected_method"]] = ["mc_running", pd.NA]
    atomic_csv(queue, settings.family_queue)
    ok, labels, details = run_mc_scope(row, "same_city")
    if ok:
        method = str(details["selected_method"])
        _publish_success(queue, index, row, labels, "mc_labelled", method, details)
        return
    atomic_json(
        {"schema": "same_city_mc_failure", **details},
        directory / "same_city_mc_attempt.json",
    )
    queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
        "cross_matching_running",
        pd.NA,
        str(details.get("reason")),
    ]
    atomic_csv(queue, settings.family_queue)


def _cross_city_matching(
    queue: pd.DataFrame,
    index: int,
    row: pd.Series,
    control_queue: pd.DataFrame | None,
    control_queue_path: Path,
) -> None:
    directory = task_directory(int(row.treatment_order), str(row.outcome_family))
    directory.mkdir(parents=True, exist_ok=True)
    if control_queue is None:
        control_queue = read_control_queue(control_queue_path)
    queue.loc[index, "status"] = "cross_matching_running"
    atomic_csv(queue, settings.family_queue)
    ok, control, details = run_cross_city_matching(row, control_queue, control_queue_path)
    if ok:
        labels_ok, labels, label_details = run_frozen_control(row, control)
        if labels_ok:
            method = "frozen_matched_change_12m_baseline_cross_city"
            _publish_success(
                queue,
                index,
                row,
                labels,
                "matched_labelled",
                method,
                label_details,
            )
            return
        details = {
            **details,
            "reason": "cross_city_fixed_control_label_failure",
            "label_failure": label_details,
        }
    atomic_json(
        {"schema": "cross_city_matching_failure", **details},
        directory / "cross_city_matching_attempt.json",
    )
    queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
        "cross_gsc_running",
        pd.NA,
        str(details.get("reason")),
    ]
    atomic_csv(queue, settings.family_queue)


def _cross_city_gsc(queue: pd.DataFrame, index: int, row: pd.Series) -> None:
    directory = task_directory(int(row.treatment_order), str(row.outcome_family))
    directory.mkdir(parents=True, exist_ok=True)
    queue.loc[index, "status"] = "cross_gsc_running"
    atomic_csv(queue, settings.family_queue)
    if str(row.outcome_family) == "viirs":
        cached, logs = ensure_cross_city_viirs(row)
        if cached:
            ok, labels, details = run_gsc_scope(row, "all_city_standardized")
        else:
            ok, labels, details = (
                False,
                [],
                {
                    "reason": "cross_city_viirs_cache_unavailable",
                    "log": "\n".join(logs),
                },
            )
    else:
        ok, labels, details = run_gsc_scope(row, "all_city_standardized")
    if ok:
        method = str(details["selected_method"])
        _publish_success(queue, index, row, labels, "gsc_labelled", method, details)
        return
    atomic_json(
        {"schema": "cross_city_gsc_failure", **details},
        directory / "cross_city_gsc_attempt.json",
    )
    queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
        "cross_mc_running",
        pd.NA,
        str(details.get("reason")),
    ]
    atomic_csv(queue, settings.family_queue)


def _cross_city_mc(queue: pd.DataFrame, index: int, row: pd.Series) -> None:
    directory = task_directory(int(row.treatment_order), str(row.outcome_family))
    directory.mkdir(parents=True, exist_ok=True)
    queue.loc[index, "status"] = "cross_mc_running"
    atomic_csv(queue, settings.family_queue)
    ok, labels, details = run_mc_scope(row, "all_city_standardized")
    if ok:
        method = str(details["selected_method"])
        _publish_success(queue, index, row, labels, "mc_labelled", method, details)
        return
    atomic_json(
        {
            "schema": "causal_response_labels_v1",
            "status": "skipped",
            "method": None,
            "treatment_order": int(row.treatment_order),
            "outcome_family": str(row.outcome_family),
            "matching_failure": _read_attempt(
                directory / "matching_attempt.json", "same_city_matching_not_run"
            ),
            "gsc_failure": _read_attempt(directory / "gsc_attempt.json", "same_city_gsc_not_run"),
            "same_city_mc_failure": _read_attempt(
                directory / "same_city_mc_attempt.json", "same_city_mc_not_run"
            ),
            "cross_city_matching_failure": _read_attempt(
                directory / "cross_city_matching_attempt.json",
                "cross_city_matching_not_run",
            ),
            "cross_city_gsc_failure": _read_attempt(
                directory / "cross_city_gsc_attempt.json", "cross_city_gsc_not_run"
            ),
            "cross_city_mc_failure": details,
        },
        directory / "failure_manifest.json",
    )
    queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
        "skipped",
        pd.NA,
        str(details.get("reason")),
    ]
    atomic_csv(queue, settings.family_queue)


def run_global_round(
    queue: pd.DataFrame,
    index: int,
    row: pd.Series,
    control_queue: pd.DataFrame | None = None,
    control_queue_path: Path = CONTROL_QUEUE,
    *,
    round_name: GlobalRound,
) -> None:
    """Run exactly one fallback round and persist its next-stage checkpoint."""
    if round_name == "same_city_mc":
        _same_city_mc(queue, index, row)
    elif round_name == "cross_city_matching":
        _cross_city_matching(queue, index, row, control_queue, control_queue_path)
    elif round_name == "cross_city_gsc":
        _cross_city_gsc(queue, index, row)
    else:
        _cross_city_mc(queue, index, row)


__all__ = [
    "GlobalRound",
    "ROUND_STATUSES",
    "eligible_round_indices",
    "recover_completed_task_or_recompute_stale_qualification",
    "run_global_round",
]
