"""Production orchestrator for one-grid-at-a-time causal response labels.

Estimation remains in the published R implementations. This script supplies
transactional queue transitions, VIIRS just-in-time caching, method routing,
normalized label files, and crash-safe resume behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.data.paths import (  # noqa: E402
    CAUSAL_DIR,
    OUTPUT_CONTROL_TASKS_DIR,
    OUTPUT_FIXED_CONTROL_DIR,
    PANEL_HOUSING_MONTHLY_DIR,
    POI_DIR,
    POPULATION_DIR,
    R_LIB_DIR,
    TREATMENT_UNIT_LIST,
    collection_script,
    r_script,
)
from urban_intervention.data.paths import (
    CONTROL_DESIGN_QUEUE as CONTROL_QUEUE,
)
from urban_intervention.data.paths import (
    COUNTERFACTUAL_QUEUE as UNIT_QUEUE,
)
from urban_intervention.data.paths import (
    ELIGIBLE_DONORS as DONOR_UNIVERSE,
)
from urban_intervention.data.paths import (
    FORMAL_TARGET_SUPPORT as SUPPORT,
)
from urban_intervention.data.paths import (
    OUTCOME_FAMILY_QUEUE as FAMILY_QUEUE,
)
from urban_intervention.data.paths import (
    OUTPUT_CAUSAL_TASKS_DIR as TASK_ROOT,
)
from urban_intervention.data.paths import (
    OUTPUT_COMPLETE_STAGING_DIR as STAGING,
)

R_SCRIPT = os.environ.get("MIT_RSCRIPT", "Rscript")
R_LIB = Path(os.environ.get("MIT_R_LIB", str(R_LIB_DIR)))
VIIRS_RAW = os.environ.get("MIT_VIIRS_RAW")

# Main anticipation window in months (complete_estimator_spec()$timing:
# main = 6; sensitivity = 0 / 12).  Set via --anticipation-months.
_ANTICIPATION_MONTHS = 6

OUTCOMES = {
    "housing": ["housing_log_price"],
    "viirs": ["viirs_avg_asinh"],
    "population": ["population_log"],
    "poi": [
        "poi_count_log",
        "poi_category_entropy",
        "poi_commercial_share",
        "poi_transport_access_log",
    ],
}
HORIZONS = {
    "housing": [1, 3, 6, 12, 18, 24],
    "viirs": [1, 3, 6, 12, 18, 24],
    "population": [1, 2, 3],
    "poi": [1, 2, 3],
}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_family_queue(path: Path = FAMILY_QUEUE) -> pd.DataFrame:
    queue = pd.read_csv(path)
    queue["treatment_order"] = queue["treatment_order"].astype("int64")
    for column in ("status", "selected_method", "failure_reason", "outcome_family"):
        queue[column] = queue[column].astype("string")
    return queue


def read_control_queue(path: Path = CONTROL_QUEUE) -> pd.DataFrame:
    queue = pd.read_csv(path)
    queue["treatment_order"] = queue["treatment_order"].astype("int64")
    return queue


def control_for_order(order: int, control_queue: pd.DataFrame) -> pd.Series:
    rows = control_queue.loc[control_queue["treatment_order"] == int(order)]
    if len(rows) != 1:
        raise ValueError("Frozen control-design row is not unique")
    return rows.iloc[0]


def sync_unit_queue(
    order: int,
    family_queue: pd.DataFrame,
    control_queue: pd.DataFrame,
    unit_queue_path: Path = UNIT_QUEUE,
) -> None:
    rows = family_queue.loc[family_queue["treatment_order"] == int(order)]
    terminal = {"matched_labelled", "gsc_labelled", "mc_labelled", "skipped"}
    if len(rows) != 4 or not rows["status"].isin(terminal).all():
        return
    unit_queue = pd.read_csv(unit_queue_path)
    for column in ("status", "selected_method", "selected_control_grid_id", "failure_reason"):
        unit_queue[column] = unit_queue[column].astype("string")
    labelled = rows["status"].isin({"matched_labelled", "gsc_labelled", "mc_labelled"})
    if labelled.all():
        status, failure = "labelled", pd.NA
    elif labelled.any():
        status, failure = "partially_labelled", "one_or_more_outcome_families_unavailable"
    else:
        status, failure = "skipped", "all_outcome_families_unavailable"
    methods = sorted(set(rows.loc[labelled, "selected_method"].dropna().astype(str)))
    control = control_for_order(order, control_queue)
    selected_control = (
        str(control.control_unit_key)
        if str(control.status) == "matched" and pd.notna(control.control_unit_key)
        else pd.NA
    )
    index = unit_queue.index[unit_queue["treatment_order"] == int(order)]
    if len(index) != 1:
        raise ValueError("Unit queue row is not unique")
    unit_queue.loc[
        index[0], ["status", "selected_method", "selected_control_grid_id", "failure_reason"]
    ] = [status, "+".join(methods) if methods else pd.NA, selected_control, failure]
    atomic_csv(unit_queue, unit_queue_path)


def new_run_id() -> str:
    return uuid.uuid4().hex


def run(
    command: list[str], environment_overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if R_LIB.is_dir():
        environment["R_LIBS_USER"] = str(R_LIB)
    # The queue's Python subprocesses (e.g. ensure_viirs_monthly_cache.py)
    # import project modules from src/; without this, an uninstalled checkout
    # fails every VIIRS cache materialisation with ModuleNotFoundError, which
    # used to masquerade as "monthly_viirs_cache_unavailable" failures.
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT / "src"), str(ROOT / "scripts"), environment.get("PYTHONPATH", "")])
    )
    if environment_overrides:
        environment.update(environment_overrides)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def family_signature(row: pd.Series, support: pd.DataFrame) -> str:
    target = support.loc[support["treatment_order"] == int(row["treatment_order"])]
    if len(target) != 1:
        raise ValueError("Treatment support row is not unique")
    names = [
        family
        for family in OUTCOMES
        if pd.notna(target.iloc[0][f"{family}_complete"])
        and bool(target.iloc[0][f"{family}_complete"])
    ]
    return "+".join(sorted(names))


# Panels are read once per (city, family) into a set of grids with any
# observed value; the R estimators are only worth invoking when the grid
# actually appears in the family's panel.
_FAMILY_SUPPORT_CACHE: dict[tuple[str, str], set[str]] = {}


def _family_observed_grids(city: str, family: str) -> set[str]:
    key = (city, family)
    if key in _FAMILY_SUPPORT_CACHE:
        return _FAMILY_SUPPORT_CACHE[key]
    grids: set[str] = set()
    if family == "housing":
        path = PANEL_HOUSING_MONTHLY_DIR / f"{city}.parquet"
        if path.is_file():
            frame = pq.read_table(
                path, columns=["grid_id", "log_price_raw_median"]
            ).to_pandas()
            grids = set(
                frame.loc[frame["log_price_raw_median"].notna(), "grid_id"].astype(str)
            )
    elif family in {"poi", "population"}:
        directory = POI_DIR if family == "poi" else POPULATION_DIR
        candidates = sorted(directory.glob(f"{city}*"))
        if candidates:
            frame = pd.read_parquet(candidates[0])
            value_columns = [
                column
                for column in frame.columns
                if column not in {"city_key", "grid_id", "year"}
            ]
            if value_columns:
                mask = frame[value_columns].notna().any(axis=1)
                grids = set(frame.loc[mask, "grid_id"].astype(str))
    _FAMILY_SUPPORT_CACHE[key] = grids
    return grids


def family_has_observed_support(row: pd.Series) -> bool:
    """True when the grid appears in the family panel with any observation.

    VIIRS is not pre-screened: the monthly partitions cover every grid, so
    the check would never fire there.  For the other families this turns
    no-data tasks into an instant skip instead of a ~3-minute GSC/MC run
    that must fail.
    """
    family = str(row.outcome_family)
    if family == "viirs":
        return True
    return str(row.grid_id) in _family_observed_grids(str(row.city_key), family)


def task_directory(order: int, family: str) -> Path:
    return TASK_ROOT / f"{order:05d}" / family


def _month_key(value: object) -> str:
    try:
        return str(pd.Period(str(value)[:7], freq="M"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid opening month: {value!r}") from exc


def _cohort_year(value: object) -> str:
    try:
        return str(pd.Period(str(value)[:7], freq="M").year)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid opening month: {value!r}") from exc


def validate_task_labels(row: pd.Series, labels: pd.DataFrame) -> None:
    keys = [
        "treatment_order",
        "outcome_family",
        "outcome",
        "event_time",
        "specification_id",
    ]
    identity = ["city_key", "grid_id", "opening_month"]
    missing = set(keys + identity) - set(labels.columns)
    if missing:
        raise ValueError(f"Normalized task labels lack identity columns: {sorted(missing)}")
    if labels.empty:
        raise ValueError("Successful task cannot contain zero label rows")
    expected_order = int(row.treatment_order)
    orders = set(pd.to_numeric(labels["treatment_order"], errors="raise").astype(int))
    if orders != {expected_order}:
        raise ValueError(
            f"Task labels treatment_order {sorted(orders)} disagree with {expected_order}"
        )
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
    if labels.duplicated(keys).any():
        raise ValueError("Normalized task labels violate their primary key")


def validate_task_manifest(row: pd.Series, payload: dict[str, object], labels_path: Path) -> None:
    if payload.get("schema") != "causal_response_labels_v1":
        raise ValueError("Completed task has an unknown manifest schema")
    if int(payload.get("treatment_order", -1)) != int(row.treatment_order):
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
    if payload.get("production_eligible") is not True:
        raise ValueError("Completed task manifest is not production eligible")
    details = payload.get("details")
    if not isinstance(details, dict) or not str(details.get("run_id") or ""):
        raise ValueError("Completed task manifest lacks current estimator run_id")


def recover_completed_task(
    queue: pd.DataFrame, index: int, control_queue: pd.DataFrame | None = None
) -> bool:
    row = queue.loc[index]
    manifest = task_directory(int(row.treatment_order), str(row.outcome_family)) / "manifest.json"
    labels = manifest.parent / "labels.parquet"
    if (
        row.status in {"matching_running", "gsc_pending", "gsc_running", "mc_pending", "mc_running"}
        and manifest.exists()
        and labels.exists()
    ):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("status") in {"matched_labelled", "gsc_labelled", "mc_labelled"}:
            validate_task_manifest(row, payload, labels)
            if payload.get("status") == "matched_labelled" and control_queue is not None:
                control = control_for_order(int(row.treatment_order), control_queue)
                if str(control.status) != "matched":
                    raise ValueError(
                        f"Recovery refused: control design for treatment "
                        f"{int(row.treatment_order)} is {control.status}, not matched"
                    )
                manifest_control = str(payload.get("details", {}).get("control_unit_key", ""))
                if (
                    manifest_control
                    and str(control.get("control_unit_key", "")) != manifest_control
                ):
                    raise ValueError(
                        f"Recovery refused: control_unit_key in manifest "
                        f"({manifest_control}) disagrees with control queue "
                        f"({control.get('control_unit_key', '')})"
                    )
            stored_labels = pd.read_parquet(labels)
            validate_task_labels(row, stored_labels)
            if int(payload.get("label_rows", -1)) != len(stored_labels):
                raise ValueError("Completed task manifest row count disagrees with labels")
            queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
                payload["status"],
                payload["method"],
                pd.NA,
            ]
            atomic_csv(queue, FAMILY_QUEUE)
            return True
    return False


def ensure_viirs(
    row: pd.Series, require_full_matching_window: bool = True, city_key: str | None = None
) -> subprocess.CompletedProcess[str]:
    requested_city = city_key or str(row.city_key)
    opening = pd.Period(str(row.opening_month), freq="M")
    requested_start = opening - 42
    start = (
        requested_start
        if require_full_matching_window
        else max(requested_start, pd.Period("2012-01", freq="M"))
    )
    end = opening + 24
    command = [
        sys.executable,
        str(collection_script("ensure_viirs_monthly_cache.py")),
        "--city",
        requested_city,
        "--start",
        str(start),
        "--end",
        str(end),
        "--manifest",
        str(
            task_directory(int(row.treatment_order), "viirs") / f"viirs_cache_{requested_city}.json"
        ),
    ]
    if VIIRS_RAW:
        command[2:2] = ["--input-dir", VIIRS_RAW]
    return run(command)


def viirs_has_min_preperiods(opening_month: str) -> bool:
    return pd.Period(str(opening_month), freq="M") >= pd.Period("2012-12", freq="M")


def viirs_has_full_matching_window(opening_month: str) -> bool:
    return pd.Period(str(opening_month), freq="M") >= pd.Period("2015-07", freq="M")


def fixed_control_output(row: pd.Series) -> Path:
    return OUTPUT_FIXED_CONTROL_DIR / f"{int(row.treatment_order):05d}" / str(row.outcome_family)


def run_cross_city_matching(
    row: pd.Series, control_queue: pd.DataFrame, control_queue_path: Path = CONTROL_QUEUE
) -> tuple[bool, pd.Series, dict[str, object]]:
    """Round 4: cross-city matching, invoked after same-city GSC/MC failed.

    Runs R design_cross_city_control for this treatment order and, on
    success, records the cross-city control into the control queue so the
    frozen-control label path can consume it.  In sharded runs the record
    lands in the shard-specific control queue (``control_queue_path``), never
    the master file, so concurrent shards cannot clobber each other.
    """
    order = int(row.treatment_order)
    task_root = OUTPUT_CONTROL_TASKS_DIR
    environment = os.environ.copy()
    if Path(os.environ.get("MIT_R_LIB", "")).exists():
        environment["R_LIBS_USER"] = os.environ["MIT_R_LIB"]
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
    attempt_path = task_root / f"{order:05d}" / "cross_city_attempt.csv"
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
    record_path = task_root / f"{order:05d}" / "control_record.csv"
    if not record_path.exists():
        return (
            False,
            pd.Series(dtype=object),
            {
                "reason": "cross_city_control_record_missing",
                "log": completed.stdout,
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
    # Apply the cross-city control into the control queue row (status stays
    # matched; donor_scope marks all_city_standardized).
    idx = control_queue.index[control_queue["treatment_order"].astype(int) == order]
    if len(idx) != 1:
        raise RuntimeError(f"Control queue row for treatment {order} not unique")
    control_queue.loc[idx[0], "status"] = "matched"
    control_queue.loc[idx[0], "selected_method"] = record.get("selected_method", pd.NA)
    control_queue.loc[idx[0], "donor_scope"] = "all_city_standardized"
    control_queue.loc[idx[0], "control_city_key"] = record.get("control_city_key", pd.NA)
    control_queue.loc[idx[0], "control_grid_id"] = record.get("control_grid_id", pd.NA)
    control_queue.loc[idx[0], "control_unit_key"] = record.get("control_unit_key", pd.NA)
    control_queue.loc[idx[0], "failure_reason"] = pd.NA
    atomic_csv(control_queue, control_queue_path)
    return (
        True,
        control_queue.loc[idx[0]],
        {
            "control_unit_key": str(record.get("control_unit_key")),
            "donor_scope": "all_city_standardized",
            "control_record": str(record_path.relative_to(ROOT)),
        },
    )


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
    completed = run(
        [
            str(R_SCRIPT),
            str(r_script("run_fixed_control_labels.R")),
            str(int(row.treatment_order)),
            str(control.control_city_key),
            str(control.control_grid_id),
            family,
            str(output),
            str(_LABEL_WINDOW),
            _PRICE_MEASURE,
        ],
        {"MIT_CAUSAL_RUN_ID": run_id},
    )
    if completed.returncode != 0:
        return False, [], {"reason": "fixed_control_label_runtime_error", "log": completed.stdout}
    labels_path = output / "causal_response_labels.parquet"
    estimator_manifest = output / "manifest.csv"
    if not labels_path.exists() or not estimator_manifest.exists():
        return (
            False,
            [],
            {"reason": "fixed_control_manifest_or_labels_missing", "log": completed.stdout},
        )
    manifest_values = read_estimator_manifest(estimator_manifest)
    if (
        manifest_values.get("run_id") != run_id
        or manifest_values.get("estimator") != "frozen_matched_change"
        or manifest_values.get("run_mode") != "production"
        or manifest_values.get("production_eligible", "").upper() != "TRUE"
        or manifest_values.get("treatment_order") != str(int(row.treatment_order))
        or manifest_values.get("outcome_family") != family
    ):
        return (
            False,
            [],
            {
                "reason": "fixed_control_manifest_does_not_prove_current_production_run",
                "run_id": run_id,
                "log": completed.stdout,
            },
        )
    raw = pq.read_table(labels_path).to_pandas()
    raw["transformed_scale"] = True
    raw["specification_id"] = "main_a6_r1km"
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
                "log": completed.stdout,
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
            "log": completed.stdout,
        },
    )


def gsc_output(row: pd.Series, outcome: str, donor_scope: str = "same_city") -> Path:
    cohort = (
        str(row.opening_month)
        if row.outcome_family in {"housing", "viirs"}
        else _cohort_year(row.opening_month)
    )
    tag = f"{outcome}_t{int(row.treatment_order):05d}"
    signature = (
        "outcome_only_prepath"
        if donor_scope == "same_city"
        else "outcome_only_prepath_all_city_standardized"
    )
    return STAGING / "xu_gsc" / str(row.city_key) / cohort / tag / signature


def run_gsc_scope(
    row: pd.Series, donor_scope: str
) -> tuple[bool, list[pd.DataFrame], dict[str, object]]:
    family = str(row.outcome_family)
    frequency = "monthly" if family in {"housing", "viirs"} else "annual"
    cohort = str(row.opening_month) if frequency == "monthly" else _cohort_year(row.opening_month)
    run_id = new_run_id()
    completed = run(
        [
            str(R_SCRIPT),
            str(r_script("run_complete_xu_gsc.R")),
            str(row.city_key),
            cohort,
            family,
            "auto",
            frequency,
            str(_ANTICIPATION_MONTHS),
            str(int(row.treatment_order)),
            donor_scope,
        ],
        {"MIT_CAUSAL_RUN_ID": run_id},
    )
    if completed.returncode != 0:
        return False, [], {"reason": "xu_gsc_runtime_or_support_failure", "log": completed.stdout}
    labels: list[pd.DataFrame] = []
    estimator_manifests: list[dict[str, object]] = []
    selected_method = (
        "xu_2017_gsynth_same_city"
        if donor_scope == "same_city"
        else "xu_2017_gsynth_all_city_standardized"
    )
    for outcome in OUTCOMES[family]:
        path = gsc_output(row, outcome, donor_scope) / "causal_response_labels.parquet"
        estimator_manifest = path.parent / "manifest.csv"
        if not estimator_manifest.exists() or not path.exists():
            return (
                False,
                [],
                {"reason": "xu_gsc_manifest_or_labels_missing", "path": str(path.parent)},
            )
        manifest_values = read_estimator_manifest(estimator_manifest)
        if (
            manifest_values.get("run_id") != run_id
            or manifest_values.get("estimator") != "gsynth"
            or manifest_values.get("CV", "").upper() != "TRUE"
            or manifest_values.get("run_mode") != "production"
            or manifest_values.get("production_eligible", "").upper() != "TRUE"
            or manifest_values.get("inference") != "parametric"
            or manifest_values.get("nboots") != "200"
        ):
            return (
                False,
                [],
                {
                    "reason": "xu_gsc_manifest_does_not_prove_current_production_run",
                    "path": str(estimator_manifest),
                    "run_id": run_id,
                },
            )
        estimator_manifests.append(
            {
                "path": str(estimator_manifest.relative_to(ROOT)),
                "sha256": file_sha256(estimator_manifest),
                "labels_sha256": file_sha256(path),
                "run_id": run_id,
            }
        )
        raw = pq.read_table(path).to_pandas()
        raw = raw.loc[raw["event_time"].isin(HORIZONS[family])].copy()
        normalized = pd.DataFrame(
            {
                "treatment_order": raw["treatment_order"].astype(int),
                "city_key": raw["city_key"],
                "grid_id": raw["grid_id"],
                "opening_month": str(row.opening_month),
                "outcome_family": family,
                "outcome": raw["outcome"],
                "event_time": raw["event_time"].astype(int),
                "specification_id": "main_a6_r1km",
                "observed": raw["observed"],
                "counterfactual": raw["counterfactual"],
                "causal_response_label": raw["causal_response_label"],
                "transformed_scale": True,
                "method": selected_method,
                "label_available": raw["label_available"],
                "control_unit_key": pd.NA,
                "standard_error": raw.get("standard_error", pd.NA),
                "confidence_lower": raw.get("confidence_lower", pd.NA),
                "confidence_upper": raw.get("confidence_upper", pd.NA),
                "p_value": raw.get("p_value", pd.NA),
                "bootstrap_repetitions": raw.get("bootstrap_repetitions", 0),
                "uncertainty_source": raw.get("uncertainty_source", pd.NA),
            }
        )
        labels.append(normalized)
    return (
        True,
        labels,
        {
            "selected_method": selected_method,
            "donor_scope": donor_scope,
            "run_id": run_id,
            "estimator_manifests": estimator_manifests,
            "log": completed.stdout,
        },
    )


def ensure_cross_city_viirs(row: pd.Series) -> tuple[bool, list[str]]:
    cities = sorted(
        set(pq.read_table(DONOR_UNIVERSE, columns=["city_key"]).column("city_key").to_pylist())
    )
    logs: list[str] = []
    for city in cities:
        completed = ensure_viirs(row, require_full_matching_window=False, city_key=city)
        logs.append(completed.stdout)
        if completed.returncode != 0:
            return False, logs
    return True, logs


def read_estimator_manifest(path: Path) -> dict[str, str]:
    manifest = pd.read_csv(path)
    if set(manifest.columns) != {"field", "value"} or manifest["field"].duplicated().any():
        raise ValueError(f"Malformed estimator manifest: {path}")
    return dict(zip(manifest["field"].astype(str), manifest["value"].astype(str), strict=False))


def mc_output(row: pd.Series, outcome: str, donor_scope: str = "same_city") -> Path:
    cohort = (
        str(row.opening_month)
        if row.outcome_family in {"housing", "viirs"}
        else _cohort_year(row.opening_month)
    )
    tag = f"{outcome}_t{int(row.treatment_order):05d}"
    signature = (
        "outcome_only_prepath_mc"
        if donor_scope == "same_city"
        else "outcome_only_prepath_mc_all_city"
    )
    return STAGING / "matrix_completion" / str(row.city_key) / cohort / tag / signature


def mc_family_run_output(row: pd.Series, donor_scope: str = "same_city") -> Path:
    cohort = (
        str(row.opening_month)
        if row.outcome_family in {"housing", "viirs"}
        else _cohort_year(row.opening_month)
    )
    tag = f"{row.outcome_family}_t{int(row.treatment_order):05d}"
    signature = (
        "outcome_only_prepath_mc"
        if donor_scope == "same_city"
        else "outcome_only_prepath_mc_all_city"
    )
    return STAGING / "matrix_completion_runs" / str(row.city_key) / cohort / tag / signature


def run_mc_scope(
    row: pd.Series, donor_scope: str
) -> tuple[bool, list[pd.DataFrame], dict[str, object]]:
    family = str(row.outcome_family)
    frequency = "monthly" if family in {"housing", "viirs"} else "annual"
    cohort = str(row.opening_month) if frequency == "monthly" else _cohort_year(row.opening_month)
    run_id = new_run_id()
    completed = run(
        [
            str(R_SCRIPT),
            str(r_script("run_complete_mc.R")),
            str(row.city_key),
            cohort,
            family,
            "auto",
            frequency,
            str(_ANTICIPATION_MONTHS),
            str(int(row.treatment_order)),
            donor_scope,
        ],
        {"MIT_CAUSAL_RUN_ID": run_id},
    )
    status_path = mc_family_run_output(row, donor_scope) / "outcome_status.csv"
    if completed.returncode != 0:
        return (
            False,
            [],
            {
                "reason": "mc_runtime_or_support_failure",
                "log": completed.stdout[-4000:],
                "outcome_status": str(status_path.relative_to(ROOT))
                if status_path.exists()
                else None,
            },
        )
    if not status_path.exists():
        return False, [], {"reason": "mc_family_status_missing", "log": completed.stdout[-4000:]}
    outcome_status = pd.read_csv(status_path)
    required_status = {"outcome", "status", "failure_reason", "run_id"}
    if not required_status.issubset(outcome_status.columns):
        return False, [], {"reason": "mc_family_status_malformed", "path": str(status_path)}
    if (
        len(outcome_status) != len(OUTCOMES[family])
        or outcome_status["outcome"].astype(str).duplicated().any()
        or set(outcome_status["outcome"].astype(str)) != set(OUTCOMES[family])
    ):
        return False, [], {"reason": "mc_family_status_outcome_mismatch", "path": str(status_path)}
    if set(outcome_status["run_id"].astype(str)) != {run_id}:
        return (
            False,
            [],
            {
                "reason": "mc_family_status_is_not_from_current_run",
                "path": str(status_path),
                "run_id": run_id,
            },
        )
    labels: list[pd.DataFrame] = []
    estimator_manifests: list[dict[str, object]] = []
    outcome_failures: dict[str, str] = {}
    selected_method = (
        "athey_2021_mc_same_city"
        if donor_scope == "same_city"
        else "athey_2021_mc_all_city_standardized"
    )
    for outcome in OUTCOMES[family]:
        outcome_record = outcome_status.loc[outcome_status["outcome"].astype(str) == outcome].iloc[
            0
        ]
        if str(outcome_record["status"]) != "success":
            reason = outcome_record.get("failure_reason")
            outcome_failures[outcome] = (
                "mc_outcome_failed" if pd.isna(reason) or not str(reason).strip() else str(reason)
            )
            continue
        path = mc_output(row, outcome, donor_scope) / "causal_response_labels.parquet"
        estimator_manifest = path.parent / "manifest.csv"
        if not estimator_manifest.exists() or not path.exists():
            outcome_failures[outcome] = "mc_success_record_lacks_manifest_or_labels"
            continue
        manifest_values = read_estimator_manifest(estimator_manifest)
        try:
            selected_lambda = float(manifest_values.get("selected_lambda", "nan"))
            cv_min_mspe = float(manifest_values.get("cv_min_mspe", "nan"))
        except ValueError:
            selected_lambda = cv_min_mspe = float("nan")
        if (
            manifest_values.get("estimator") != "mc"
            or manifest_values.get("fitted_method") != "mc"
            or manifest_values.get("backend") != "fect"
            or manifest_values.get("run_id") != run_id
            or manifest_values.get("CV", "").upper() != "TRUE"
            or manifest_values.get("cv_method") != "rolling"
            or manifest_values.get("cv_nobs") != "1"
            or manifest_values.get("cv_donut") != "0"
            or manifest_values.get("cv_buffer") != "0"
            or manifest_values.get("two_stage_cv_inference", "").upper() != "TRUE"
            or manifest_values.get("inference_fit_CV", "").upper() != "FALSE"
            or manifest_values.get("run_mode") != "production"
            or manifest_values.get("production_eligible", "").upper() != "TRUE"
            or manifest_values.get("inference") != "bootstrap"
            or manifest_values.get("nboots") != "200"
            or not math.isfinite(selected_lambda)
            or selected_lambda <= 0
            or not math.isfinite(cv_min_mspe)
        ):
            outcome_failures[outcome] = "mc_manifest_does_not_prove_cross_validated_mc"
            continue
        estimator_manifests.append(
            {
                "path": str(estimator_manifest.relative_to(ROOT)),
                "sha256": file_sha256(estimator_manifest),
                "labels_sha256": file_sha256(path),
                "selected_lambda": selected_lambda,
                "cv_min_mspe": cv_min_mspe,
                "run_id": run_id,
            }
        )
        raw = pq.read_table(path).to_pandas()
        raw = raw.loc[raw["event_time"].isin(HORIZONS[family])].copy()
        if raw.empty or not bool(raw["label_available"].fillna(False).any()):
            outcome_failures[outcome] = "mc_has_no_available_target_horizon"
            continue
        normalized = pd.DataFrame(
            {
                "treatment_order": raw["treatment_order"].astype(int),
                "city_key": raw["city_key"],
                "grid_id": raw["grid_id"],
                "opening_month": str(row.opening_month),
                "outcome_family": family,
                "outcome": raw["outcome"],
                "event_time": raw["event_time"].astype(int),
                "specification_id": "main_a6_r1km",
                "observed": raw["observed"],
                "counterfactual": raw["counterfactual"],
                "causal_response_label": raw["causal_response_label"],
                "transformed_scale": True,
                "method": selected_method,
                "label_available": raw["label_available"],
                "control_unit_key": pd.NA,
                "standard_error": raw.get("standard_error", pd.NA),
                "confidence_lower": raw.get("confidence_lower", pd.NA),
                "confidence_upper": raw.get("confidence_upper", pd.NA),
                "p_value": raw.get("p_value", pd.NA),
                "bootstrap_repetitions": raw.get("bootstrap_repetitions", 0),
                "uncertainty_source": raw.get("uncertainty_source", pd.NA),
                "pre_observed_periods": raw.get("pre_observed_periods", pd.NA),
                "pre_rmspe": raw.get("pre_rmspe", pd.NA),
                "mc_lambda": raw.get("mc_lambda", selected_lambda),
                "mc_cv_mspe": raw.get("mc_cv_mspe", cv_min_mspe),
            }
        )
        labels.append(normalized)
    if not labels:
        return (
            False,
            [],
            {
                "reason": "mc_no_outcome_produced_available_labels",
                "donor_scope": donor_scope,
                "outcome_failures": outcome_failures,
                "outcome_status": str(status_path.relative_to(ROOT)),
                "log": completed.stdout[-4000:],
            },
        )
    return (
        True,
        labels,
        {
            "selected_method": selected_method,
            "donor_scope": donor_scope,
            "run_id": run_id,
            "estimator_manifests": estimator_manifests,
            "outcome_failures": outcome_failures,
            "outcome_status": str(status_path.relative_to(ROOT)),
            "outcome_status_sha256": file_sha256(status_path),
            "log": completed.stdout,
        },
    )


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
    result = pd.concat(frames, ignore_index=True)
    validate_task_labels(row, result)
    if not str(details.get("run_id") or ""):
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
            "production_eligible": True,
            "details": details,
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
    atomic_csv(queue, FAMILY_QUEUE)


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
    gsc_attempt_path = directory / "gsc_attempt.json"
    gsc_details: dict[str, object] = (
        json.loads(gsc_attempt_path.read_text(encoding="utf-8"))
        if gsc_attempt_path.exists()
        else {"reason": "gsc_failure_details_unavailable"}
    )

    # Round 3: same-city MC
    queue.loc[index, ["status", "selected_method"]] = ["mc_running", pd.NA]
    atomic_csv(queue, FAMILY_QUEUE)
    same_mc_ok, same_mc_labels, same_mc_details = run_mc_scope(row, "same_city")
    if same_mc_ok:
        selected_method = str(same_mc_details["selected_method"])
        write_task(row, same_mc_labels, "mc_labelled", selected_method, same_mc_details)
        queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
            "mc_labelled",
            selected_method,
            pd.NA,
        ]
        atomic_csv(queue, FAMILY_QUEUE)
        return

    # Round 4: cross-city matching
    if control_queue is None:
        control_queue = read_control_queue()
    cross_match_ok = False
    match_details: dict[str, object] = {}
    if (
        str(
            control_queue.loc[
                control_queue["treatment_order"].astype(int) == int(row.treatment_order), "status"
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
                write_task(row, cross_labels, "matched_labelled", selected_method, cross_details)
                queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
                    "matched_labelled",
                    selected_method,
                    pd.NA,
                ]
                atomic_csv(queue, FAMILY_QUEUE)
                return

    # Round 5: cross-city GSC.  The cross-city estimator reads VIIRS
    # partitions for every donor city, so materialise the window across all
    # donor cities first (the same-city rounds only need the target city).
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
    if cross_gsc_ok:
        selected_method = str(cross_gsc_details["selected_method"])
        write_task(row, cross_gsc_labels, "gsc_labelled", selected_method, cross_gsc_details)
        queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
            "gsc_labelled",
            selected_method,
            pd.NA,
        ]
        atomic_csv(queue, FAMILY_QUEUE)
        return

    # Round 6: cross-city MC (VIIRS cache already materialised in round 5
    # for the same donor scope, so no repeated materialisation here).
    cross_mc_ok, cross_mc_labels, cross_mc_details = run_mc_scope(row, "all_city_standardized")
    if cross_mc_ok:
        selected_method = str(cross_mc_details["selected_method"])
        write_task(row, cross_mc_labels, "mc_labelled", selected_method, cross_mc_details)
        queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
            "mc_labelled",
            selected_method,
            pd.NA,
        ]
        atomic_csv(queue, FAMILY_QUEUE)
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
    atomic_csv(queue, FAMILY_QUEUE)


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
    if not signature and str(control.status) == "gsc_pending":
        queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
            "skipped",
            pd.NA,
            "no_complete_pre_treatment_families",
        ]
        atomic_csv(queue, FAMILY_QUEUE)
        return
    if not family_has_observed_support(row):
        # The grid has no observation anywhere in this family's panel: GSC
        # and MC cannot estimate anything and would only fail after minutes
        # of R work.  Skip immediately with a dedicated reason.
        queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
            "skipped",
            pd.NA,
            "family_no_observed_support",
        ]
        atomic_csv(queue, FAMILY_QUEUE)
        return
    if str(row.status) in {"mc_pending", "mc_running"} and phase in {"all", "mc"}:
        run_mc_stage(queue, index, row, control_queue, control_queue_path)
        return
    if (
        str(control.status) == "gsc_pending" or str(row.status) in {"gsc_pending", "gsc_running"}
    ) and phase != "matching":
        queue.loc[index, "status"] = "gsc_running"
        atomic_csv(queue, FAMILY_QUEUE)
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
            begin_mc_stage(queue, index, row, gsc_details)
            if phase != "gsc":
                run_mc_stage(queue, index, row, control_queue, control_queue_path)
        if gsc_ok:
            atomic_csv(queue, FAMILY_QUEUE)
        return
    queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
        "matching_running",
        pd.NA,
        pd.NA,
    ]
    atomic_csv(queue, FAMILY_QUEUE)

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
        atomic_csv(queue, FAMILY_QUEUE)
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
    atomic_csv(queue, FAMILY_QUEUE)
    if phase == "matching":
        return
    queue.loc[index, "status"] = "gsc_running"
    atomic_csv(queue, FAMILY_QUEUE)
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
        begin_mc_stage(queue, index, row, gsc_details)
        if phase != "gsc":
            run_mc_stage(queue, index, row, control_queue, control_queue_path)
    if gsc_ok:
        atomic_csv(queue, FAMILY_QUEUE)


def eligible_indices(
    queue: pd.DataFrame,
    start_order: int,
    end_order: int | None,
    family: str | None,
    phase: str,
    max_tasks: int,
    retry_matching: bool = False,
    retry_skipped: bool = False,
    orders: set[int] | None = None,
) -> pd.Index:
    statuses = {
        "matching": {"pending", "matching_running"},
        "gsc": {"gsc_pending", "gsc_running"},
        "mc": {"mc_pending", "mc_running"},
        "all": {
            "pending",
            "matching_running",
            "gsc_pending",
            "gsc_running",
            "mc_pending",
            "mc_running",
        },
    }[phase]
    if phase == "matching" and retry_matching:
        statuses.add("gsc_pending")
    if retry_skipped:
        statuses.add("skipped")
    if orders is not None:
        mask = queue["treatment_order"].isin(orders) & queue["status"].isin(statuses)
    else:
        mask = (queue["treatment_order"] >= start_order) & queue["status"].isin(statuses)
        if end_order is not None:
            mask &= queue["treatment_order"] <= end_order
    if family is not None:
        mask &= queue["outcome_family"].eq(family)
    return queue.index[mask][:max_tasks]


def main() -> int:
    global FAMILY_QUEUE, _ANTICIPATION_MONTHS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-order", type=int, default=1)
    parser.add_argument("--end-order", type=int)
    parser.add_argument(
        "--orders",
        help="Comma-separated treatment orders to process (mutually exclusive "
        "with --start-order/--end-order ranges)",
    )
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--family", choices=sorted(OUTCOMES))
    parser.add_argument("--phase", choices=("matching", "gsc", "mc", "all"), default="all")
    parser.add_argument(
        "--anticipation-months",
        type=int,
        default=6,
        help="Anticipation window in months (main=6; sensitivity 0/12)",
    )
    parser.add_argument(
        "--price-measure",
        choices=("median", "hedonic"),
        default="median",
        help="Housing price measure for monthly families (hedonic = Lianjia "
        "quality-adjusted panel; defaults to the unified median measure)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Observation-window width in months for monthly fixed-control "
        "labels (1 = single-month specification; 3/6 = robustness views)",
    )
    parser.add_argument("--retry-matching", action="store_true")
    parser.add_argument(
        "--retry-skipped",
        action="store_true",
        help="Retry explicitly bounded skipped tasks after a code/data correction",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sync-all-units", action="store_true")
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Shard index (0-based) for parallel execution. Requires --shard-count.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=None,
        help="Total number of shards for parallel execution.",
    )
    args = parser.parse_args()
    _ANTICIPATION_MONTHS = args.anticipation_months
    global _PRICE_MEASURE, _LABEL_WINDOW
    _PRICE_MEASURE = args.price_measure
    _LABEL_WINDOW = args.window

    has_shard = args.shard_id is not None and args.shard_count is not None
    if bool(args.shard_id is not None) != bool(args.shard_count is not None):
        parser.error("--shard-id and --shard-count must be used together")

    family_queue = FAMILY_QUEUE
    unit_queue = UNIT_QUEUE
    control_queue_path = CONTROL_QUEUE

    if has_shard:
        shard_tag = f"_shard_{args.shard_id:02d}"
        family_queue = CAUSAL_DIR / f"outcome_family_work_queue{shard_tag}.csv"
        unit_queue = CAUSAL_DIR / f"counterfactual_work_queue{shard_tag}.csv"
        control_queue_path = CAUSAL_DIR / f"control_design_queue{shard_tag}.csv"
        # All queue writes inside process_one / recover_completed_task /
        # run_mc_stage / begin_mc_stage target the module-level FAMILY_QUEUE;
        # rebind it to the shard file so concurrent shards never clobber the
        # master queue and each shard's progress lands in its own file.
        master_family_queue = FAMILY_QUEUE
        FAMILY_QUEUE = family_queue

        # On first run, copy master queues to shard-specific files.  Use the
        # master path captured before the rebind: copying FAMILY_QUEUE onto
        # itself raises SameFileError.
        if not family_queue.exists():
            import shutil as _shutil

            _shutil.copy2(master_family_queue, family_queue)
            _shutil.copy2(CONTROL_QUEUE, control_queue_path)
            if UNIT_QUEUE.exists():
                _shutil.copy2(UNIT_QUEUE, unit_queue)
            print(f"Initialized shard {args.shard_id + 1}/{args.shard_count}: {family_queue.name}")

        # Compute this shard's portion of the 5,048 treatment orders
        treatments = pq.read_table(
            TREATMENT_UNIT_LIST,
            columns=["treatment_order"],
        ).to_pandas()
        all_orders = sorted(treatments["treatment_order"].astype(int).tolist())
        chunk_size = len(all_orders) // args.shard_count
        remainder = len(all_orders) % args.shard_count
        start_idx = args.shard_id * chunk_size + min(args.shard_id, remainder)
        if args.shard_id < remainder:
            chunk_size += 1
        shard_orders = set(all_orders[start_idx : start_idx + chunk_size])
        # Override CLI range to match shard
        args.start_order = min(shard_orders)
        args.end_order = max(shard_orders)
        print(
            f"Shard {args.shard_id + 1}/{args.shard_count}: "
            f"orders {args.start_order}-{args.end_order} ({len(shard_orders)} treatments)"
        )

    queue = read_family_queue(family_queue)
    support = pq.read_table(SUPPORT).to_pandas()
    control_queue = read_control_queue(control_queue_path)
    if args.sync_all_units:
        terminal = {"matched_labelled", "gsc_labelled", "mc_labelled", "skipped"}
        counts = queue.loc[queue["status"].isin(terminal)].groupby("treatment_order").size()
        for order in sorted(counts.index[counts.eq(4)]):
            sync_unit_queue(int(order), queue, control_queue, unit_queue_path=unit_queue)
        print("Synchronized terminal family rows into the treatment-unit queue")
        return 0
    orders_set: set[int] | None = None
    if args.orders is not None:
        orders_set = {int(value) for value in args.orders.split(",") if value.strip()}
        if not orders_set:
            parser.error("--orders must contain at least one treatment order")
        if args.start_order != 1 or args.end_order is not None:
            parser.error("--orders is mutually exclusive with --start-order/--end-order")
    eligible = eligible_indices(
        queue,
        args.start_order,
        args.end_order,
        args.family,
        args.phase,
        args.max_tasks if orders_set is None else len(orders_set) * 4,
        retry_matching=args.retry_matching,
        retry_skipped=args.retry_skipped,
        orders=orders_set,
    )
    for index in eligible:
        process_one(
            queue,
            int(index),
            support,
            control_queue,
            args.dry_run,
            phase=args.phase,
            retry_matching=args.retry_matching,
            control_queue_path=control_queue_path,
        )
        if not args.dry_run:
            sync_unit_queue(
                int(queue.loc[index, "treatment_order"]),
                queue,
                control_queue,
                unit_queue_path=unit_queue,
            )
    print(f"Processed {len(eligible)} task(s); phase={args.phase}; dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
