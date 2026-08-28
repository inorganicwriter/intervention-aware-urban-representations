"""Python/GPU production orchestrator for one-grid-at-a-time causal labels.

The production default is the contract-tested Python/PyTorch implementation;
the audited R implementation remains available as an explicit reference
backend.  This script supplies transactional queue transitions, method
routing, normalized label files, and crash-safe resume behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import uuid
from collections.abc import Iterable
from contextlib import suppress
from functools import partial
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from urban_intervention.causal.gpu.contracts import (  # noqa: E402
    FORMAL_IMPLEMENTATION_VERSION,
    FORMAL_RESULT_SCHEMA,
)
from urban_intervention.causal.gpu.control_design import (  # noqa: E402
    design_grid_control,
    write_control_design,
)
from urban_intervention.causal.gpu.fixed_control import fixed_control_labels  # noqa: E402
from urban_intervention.causal.gpu.qualification import (  # noqa: E402
    validate_formal_qualification_receipt,
)
from urban_intervention.causal.setup_inputs import (  # noqa: E402
    validate_frozen_formal_matching_spec,
)
from urban_intervention.data.paths import (  # noqa: E402
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
from urban_intervention.data.paths import (
    OUTPUT_CONTROL_TASKS_DIR,
    OUTPUT_FIXED_CONTROL_DIR,
    PANEL_HOUSING_MONTHLY_DIR,
    POI_DIR,
    POPULATION_DIR,
    PROJECT_ROOT,
    R_LIB_DIR,
    TREATMENT_UNIT_LIST,
    collection_script,
    r_script,
)
from urban_intervention.utils import (  # noqa: E402
    atomic_write_csv as atomic_csv,
)
from urban_intervention.utils import (
    atomic_write_json,
)
from urban_intervention.utils import (
    atomic_write_parquet as atomic_parquet,
)
from urban_intervention.utils import (
    sha256_file as file_sha256,
)

atomic_json = partial(atomic_write_json, default=str)

R_SCRIPT = os.environ.get("MIT_RSCRIPT", "Rscript")
R_LIB = Path(os.environ.get("MIT_R_LIB", str(R_LIB_DIR)))
VIIRS_RAW = os.environ.get("MIT_VIIRS_RAW")
ROOT = PROJECT_ROOT

# Main anticipation window in months (complete_estimator_spec()$timing:
# main = 6; sensitivity = 0 / 12).  Set via --anticipation-months.
_ANTICIPATION_MONTHS = 6
_PRICE_MEASURE = "median"
_LABEL_WINDOW = 1
_TRANSACTION_COUNT_THRESHOLD = 1
_RUN_MODE = "production"
DEFAULT_ESTIMATOR_BACKEND = "python_gpu"
_ESTIMATOR_BACKEND = DEFAULT_ESTIMATOR_BACKEND
_MAX_GSC_CROSS_CITY_DONORS = 50_000
_GSC_DONOR_SAMPLING_SEED = 20260823
_QUALIFICATION_RECEIPT: Path | None = None
_QUALIFICATION_PROOF: dict[str, object] = {}
_CROSS_CITY_DESIGN_CACHE: dict[int, tuple[pd.Series, dict[str, object]]] = {}
_R_TIMEOUT_SECONDS = int(os.environ.get("MIT_R_TIMEOUT_SECONDS", "7200"))

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


def effective_price_measure(row: pd.Series) -> str:
    """Resolve the housing measure for the approved mixed main specification."""
    if _PRICE_MEASURE != "main" or str(row.outcome_family) != "housing":
        return "median" if _PRICE_MEASURE == "main" else _PRICE_MEASURE
    hedonic_path = ROOT / "outputs" / "causal_labels" / "housing_hedonic" / (
        f"{row.city_key}_monthly.parquet"
    )
    return "hedonic" if hedonic_path.exists() else "median"


def specification_fingerprint(row: pd.Series) -> str:
    """Identify the exact label specification used by this queue process."""
    backend = (
        FORMAL_IMPLEMENTATION_VERSION
        if _ESTIMATOR_BACKEND == "python_gpu"
        else "r-reference"
    )
    return (
        f"main_a6_r1km__a{_ANTICIPATION_MONTHS}__w{_LABEL_WINDOW}"
        f"__price_{effective_price_measure(row)}__tx{_TRANSACTION_COUNT_THRESHOLD}"
        f"__backend_{backend}"
        f"__xgsc{_MAX_GSC_CROSS_CITY_DONORS}s{_GSC_DONOR_SAMPLING_SEED}"
    )


def read_family_queue(path: Path = FAMILY_QUEUE) -> pd.DataFrame:
    queue = pd.read_csv(path)
    queue["treatment_order"] = queue["treatment_order"].astype("int64")
    for column in ("status", "selected_method", "failure_reason", "outcome_family"):
        queue[column] = queue[column].astype("string")
    return queue


def read_orders_file(path: Path) -> set[int]:
    """Read a treatment-order sample manifest with strict uniqueness checks."""
    frame = pd.read_csv(path)
    if "treatment_order" not in frame.columns:
        raise ValueError(f"Orders file lacks treatment_order: {path}")
    values = pd.to_numeric(frame["treatment_order"], errors="raise").astype(int).tolist()
    orders = set(values)
    if len(orders) != len(values):
        raise ValueError(f"Orders file contains duplicate treatment_order values: {path}")
    if not orders:
        raise ValueError(f"Orders file is empty: {path}")
    return orders


def shard_order_slice(orders: list[int], shard_id: int, shard_count: int) -> list[int]:
    """Split an explicit processing order list into balanced shard slices."""
    chunk_size = len(orders) // shard_count
    remainder = len(orders) % shard_count
    start = shard_id * chunk_size + min(shard_id, remainder)
    size = chunk_size + (1 if shard_id < remainder else 0)
    return orders[start : start + size]


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
    environment["MIT_PROJECT_ROOT"] = str(ROOT)
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
    # Keep BLAS/data.table from creating a second hidden parallel layer unless
    # the server explicitly opts into it.  The estimator-level cores are
    # controlled in complete_estimator_spec().
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment.setdefault(variable, "1")
    environment.setdefault("R_DATATABLE_NUM_THREADS", "1")
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=os.name != "nt",
        creationflags=creationflags,
    )
    try:
        stdout, _ = process.communicate(timeout=_R_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        else:
            with suppress(ProcessLookupError):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        stdout, _ = process.communicate()
        output = (stdout or "") + f"\nR subprocess timed out after {_R_TIMEOUT_SECONDS}s\n"
        if exc.stdout:
            output = str(exc.stdout) + output
        return subprocess.CompletedProcess(command, 124, output)
    return subprocess.CompletedProcess(command, process.returncode, stdout or "")


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
    root = TASK_ROOT if _RUN_MODE == "production" else TASK_ROOT.parent / "tasks_preview"
    return root / f"{order:05d}" / family


def queue_variant_path(path: Path, run_mode: str) -> Path:
    """Return an isolated queue path for non-production preview work."""
    if run_mode == "production":
        return path
    return path.with_name(f"{path.stem}_{run_mode}{path.suffix}")


def shard_queue_path(path: Path, shard_id: int) -> Path:
    return path.with_name(f"{path.stem}_shard_{shard_id:02d}{path.suffix}")


def read_tasks_file(path: Path) -> set[tuple[int, str]]:
    """Read successful preview task keys for a formal uncertainty rerun."""
    frame = pd.read_csv(path)
    required = {"treatment_order", "outcome_family"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Tasks file lacks columns: {sorted(missing)}")
    keys = {
        (int(order), str(family))
        for order, family in zip(frame["treatment_order"], frame["outcome_family"], strict=False)
    }
    if len(keys) != len(frame):
        raise ValueError(f"Tasks file contains duplicate task keys: {path}")
    if not keys:
        raise ValueError(f"Tasks file is empty: {path}")
    unknown = {family for _, family in keys} - set(OUTCOMES)
    if unknown:
        raise ValueError(f"Tasks file contains unknown outcome families: {sorted(unknown)}")
    return keys


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
        "specification_fingerprint",
    ]
    identity = ["city_key", "grid_id", "opening_month"]
    missing_identity = set(identity) - set(labels.columns)
    if missing_identity:
        raise ValueError(
            "Normalized task labels lack identity columns: "
            f"{sorted(missing_identity)}"
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
    expected_production = _RUN_MODE == "production"
    if payload.get("run_mode") != _RUN_MODE:
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
    if _ESTIMATOR_BACKEND == "python_gpu" and expected_production:
        expected_receipt = str(
            _QUALIFICATION_PROOF.get("formal_qualification_receipt_sha256", "")
        )
        if not expected_receipt or details.get(
            "formal_qualification_receipt_sha256"
        ) != expected_receipt:
            raise ValueError(
                "Completed Python task lacks the current formal qualification receipt"
            )


def recover_completed_task(
    queue: pd.DataFrame, index: int, control_queue: pd.DataFrame | None = None
) -> bool:
    row = queue.loc[index]
    manifest = task_directory(int(row.treatment_order), str(row.outcome_family)) / "manifest.json"
    labels = manifest.parent / "labels.parquet"
    if (
        row.status in {
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


def invalidate_stale_terminal_tasks(queue: pd.DataFrame, orders: set[int]) -> int:
    """Return successful queue rows to pending when their spec is stale."""
    terminal = {"matched_labelled", "gsc_labelled", "mc_labelled"}
    changed = 0
    for index in queue.index[
        queue["treatment_order"].isin(orders) & queue["status"].isin(terminal)
    ]:
        row = queue.loc[index]
        manifest = task_directory(int(row.treatment_order), str(row.outcome_family)) / "manifest.json"
        payload = None
        if manifest.exists():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
        if isinstance(payload, dict) and payload.get("specification_fingerprint") == specification_fingerprint(row):
            continue
        queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
            "pending",
            pd.NA,
            "stale_specification_invalidated",
        ]
        changed += 1
    if changed:
        atomic_csv(queue, FAMILY_QUEUE)
    return changed


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
    root = (
        OUTPUT_FIXED_CONTROL_DIR
        if _RUN_MODE == "production"
        else OUTPUT_FIXED_CONTROL_DIR.with_name(
            f"{OUTPUT_FIXED_CONTROL_DIR.name}_{_RUN_MODE}"
        )
    )
    family = str(row.outcome_family)
    if family == "housing" and _TRANSACTION_COUNT_THRESHOLD != 1:
        family = f"{family}_tx{_TRANSACTION_COUNT_THRESHOLD}"
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
        return False, pd.Series(dtype=object), {
            "reason": "cross_city_matching_requires_immutable_same_city_failure",
            "record_status": str(original.status),
        }
    cached = _CROSS_CITY_DESIGN_CACHE.get(order)
    if cached is not None:
        cached_record, cached_details = cached
        return True, cached_record.copy(), dict(cached_details)
    task_root = OUTPUT_CONTROL_TASKS_DIR
    cross_signature = (
        "cross_city" if _RUN_MODE == "production" else f"cross_city_{_RUN_MODE}"
    )
    cross_city_output = task_root / f"{order:05d}" / cross_signature
    environment = os.environ.copy()
    mit_r_lib = os.environ.get("MIT_R_LIB")
    if mit_r_lib and Path(mit_r_lib).is_dir():
        environment["R_LIBS_USER"] = mit_r_lib
    if _ESTIMATOR_BACKEND == "python_gpu":
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
            return False, pd.Series(dtype=object), {
                "reason": "python_cross_city_control_design_runtime_error",
                "log": str(error),
            }
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
    _CROSS_CITY_DESIGN_CACHE[order] = (record.copy(), details.copy())
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
    if _ESTIMATOR_BACKEND == "python_gpu":
        try:
            raw = fixed_control_labels(
                int(row.treatment_order),
                str(control.control_city_key),
                str(control.control_grid_id),
                family,
                root=ROOT,
                window=_LABEL_WINDOW,
                price_measure=effective_price_measure(row),
                transaction_count_threshold=_TRANSACTION_COUNT_THRESHOLD,
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
                "run_mode": _RUN_MODE,
                "production_eligible": _RUN_MODE == "production",
                "window": _LABEL_WINDOW,
                "price_measure": effective_price_measure(row),
                "transaction_count_threshold": _TRANSACTION_COUNT_THRESHOLD,
                "labels_sha256": file_sha256(labels_path),
                **_QUALIFICATION_PROOF,
            }
            estimator_manifest = output / "manifest.csv"
            atomic_csv(
                pd.DataFrame(
                    {"field": manifest_values.keys(), "value": manifest_values.values()}
                ),
                estimator_manifest,
            )
            completed_log = "Python fixed-control labels completed"
        except Exception as error:
            return False, [], {
                "reason": "python_fixed_control_label_runtime_error",
                "log": str(error),
            }
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
                str(_LABEL_WINDOW),
                effective_price_measure(row),
            ],
            {
                "MIT_CAUSAL_RUN_ID": run_id,
                "MIT_SPECIFICATION_FINGERPRINT": specification_fingerprint(row),
                "MIT_CAUSAL_RUN_MODE": _RUN_MODE,
            },
        )
        if completed.returncode != 0:
            return False, [], {"reason": "fixed_control_label_runtime_error", "log": completed.stdout}
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
        or manifest_values.get("run_mode") != _RUN_MODE
        or manifest_values.get("production_eligible", "").upper()
        != ("TRUE" if _RUN_MODE == "production" else "FALSE")
        or manifest_values.get("treatment_order") != str(int(row.treatment_order))
        or manifest_values.get("outcome_family") != family
        or manifest_values.get("specification_fingerprint") != specification_fingerprint(row)
        or manifest_values.get("transaction_count_threshold")
        != str(_TRANSACTION_COUNT_THRESHOLD)
        or (
            _ESTIMATOR_BACKEND == "python_gpu"
            and _RUN_MODE == "production"
            and manifest_values.get("formal_qualification_receipt_sha256")
            != str(
                _QUALIFICATION_PROOF.get(
                    "formal_qualification_receipt_sha256", ""
                )
            )
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
    if _RUN_MODE != "production":
        signature = f"{signature}_{_RUN_MODE}"
    if str(row.outcome_family) == "housing" and _TRANSACTION_COUNT_THRESHOLD != 1:
        signature = f"{signature}_tx{_TRANSACTION_COUNT_THRESHOLD}"
    return STAGING / "xu_gsc" / str(row.city_key) / cohort / tag / signature


def python_estimator_command(
    row: pd.Series, estimator: str, donor_scope: str, run_id: str
) -> tuple[list[str], dict[str, str]]:
    """Build the isolated Python formal-estimator subprocess contract."""
    command = [
        sys.executable,
        str(ROOT / "scripts" / "causal_python" / "run_formal_estimator.py"),
        "--treatment-order",
        str(int(row.treatment_order)),
        "--outcome-family",
        str(row.outcome_family),
        "--estimator",
        estimator,
        "--donor-scope",
        donor_scope,
        "--anticipation-months",
        str(_ANTICIPATION_MONTHS),
        "--price-measure",
        effective_price_measure(row),
        "--observation-window",
        str(_LABEL_WINDOW if str(row.outcome_family) in {"housing", "viirs"} else 1),
        "--transaction-count-threshold",
        str(_TRANSACTION_COUNT_THRESHOLD),
        "--max-gsc-cross-city-donors",
        str(_MAX_GSC_CROSS_CITY_DONORS),
        "--gsc-donor-sampling-seed",
        str(_GSC_DONOR_SAMPLING_SEED),
        "--run-mode",
        _RUN_MODE,
        "--device",
        os.environ.get("MIT_CAUSAL_DEVICE", "auto"),
        "--run-id",
        run_id,
        "--specification-fingerprint",
        specification_fingerprint(row),
    ]
    if _QUALIFICATION_RECEIPT is not None:
        command.extend(["--qualification-receipt", str(_QUALIFICATION_RECEIPT)])
    environment = {
        "MIT_CAUSAL_RUN_ID": run_id,
        "MIT_SPECIFICATION_FINGERPRINT": specification_fingerprint(row),
    }
    prevalidated_sha256 = str(
        _QUALIFICATION_PROOF.get("formal_qualification_receipt_sha256", "")
    )
    if prevalidated_sha256:
        environment["MIT_CAUSAL_QUALIFICATION_PREVALIDATED_SHA256"] = prevalidated_sha256
    return command, environment


def validate_python_estimator_manifest(
    values: dict[str, str],
    row: pd.Series,
    *,
    estimator: str,
    outcome: str,
    donor_scope: str,
    run_id: str,
) -> tuple[bool, float, float]:
    """Fail closed unless a manifest proves the exact current Python run."""
    try:
        selected = float(values.get("selected_tuning", "nan"))
        cv_min_mspe = float(values.get("cv_min_mspe", "nan"))
    except ValueError:
        selected = cv_min_mspe = float("nan")
    production = _RUN_MODE == "production"
    inference = values.get("inference", "")
    expected_inference = (
        inference.startswith("gsc_parametric_")
        if estimator == "gsc" and production
        else inference == "mc_unit_jackknife"
        if estimator == "mc" and production
        else inference == "preview_point_estimate"
    )
    valid = (
        values.get("schema") == FORMAL_RESULT_SCHEMA
        and values.get("implementation_version") == FORMAL_IMPLEMENTATION_VERSION
        and values.get("run_id") == run_id
        and values.get("estimator") == estimator
        and values.get("backend") == "python_pytorch"
        and values.get("run_mode") == _RUN_MODE
        and values.get("production_eligible", "").upper()
        == ("TRUE" if production else "FALSE")
        and (
            not production
            or (
                values.get("formal_qualification_eligible", "").upper() == "TRUE"
                and values.get("formal_qualification_receipt_sha256")
                == str(
                    _QUALIFICATION_PROOF.get(
                        "formal_qualification_receipt_sha256", ""
                    )
                )
            )
        )
        and values.get("treatment_order") == str(int(row.treatment_order))
        and values.get("city_key") == str(row.city_key)
        and values.get("grid_id") == str(row.grid_id)
        and values.get("opening_month") == str(row.opening_month)
        and values.get("outcome_family") == str(row.outcome_family)
        and values.get("outcome") == outcome
        and values.get("donor_scope") == donor_scope
        and values.get("specification_fingerprint") == specification_fingerprint(row)
        and len(values.get("input_panel_signature", "")) == 64
        and values.get("price_measure") == effective_price_measure(row)
        and values.get("observation_window")
        == str(_LABEL_WINDOW if str(row.outcome_family) in {"housing", "viirs"} else 1)
        and values.get("transaction_count_threshold")
        == str(_TRANSACTION_COUNT_THRESHOLD)
        and expected_inference
        and math.isfinite(selected)
        and (math.isfinite(cv_min_mspe) or _RUN_MODE == "preview")
    )
    return valid, selected, cv_min_mspe


def normalized_python_labels(
    raw: pd.DataFrame, row: pd.Series, selected_method: str
) -> pd.DataFrame:
    family = str(row.outcome_family)
    return pd.DataFrame(
        {
            "treatment_order": raw["treatment_order"].astype(int),
            "city_key": raw["city_key"],
            "grid_id": raw["grid_id"],
            "opening_month": str(row.opening_month),
            "outcome_family": family,
            "outcome": raw["outcome"],
            "event_time": raw["event_time"].astype(int),
            "specification_id": "main_a6_r1km",
            "specification_fingerprint": specification_fingerprint(row),
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
            "valid_inference_repetitions": raw.get(
                "valid_inference_repetitions", 0
            ),
            "uncertainty_source": raw.get("uncertainty_source", pd.NA),
            "pre_observed_periods": raw.get("pre_observed_periods", pd.NA),
            "pre_rmspe": raw.get("pre_rmspe", pd.NA),
            "pre_mean_effect": raw.get("pre_mean_effect", pd.NA),
            "pretrend_slope": raw.get("pretrend_slope", pd.NA),
            "pretrend_slope_p_value": raw.get("pretrend_slope_p_value", pd.NA),
            "pretrend_task_flag": raw.get("pretrend_task_flag", pd.NA),
            "selected_factors": raw.get("selected_factors", pd.NA),
            "mc_lambda": raw.get("mc_lambda", pd.NA),
            "mc_regularized": raw.get("mc_regularized", pd.NA),
            "mc_cv_mspe": raw.get("mc_cv_mspe", pd.NA),
            "minimum_window_n": raw.get("minimum_window_n", pd.NA),
            "effective_n_observed": raw.get("effective_n_observed", pd.NA),
            "effective_n_counterfactual": raw.get(
                "effective_n_counterfactual", pd.NA
            ),
            "window_supported": raw.get("window_supported", pd.NA),
            "transaction_count": raw.get("transaction_count", pd.NA),
            "transaction_count_min": raw.get("transaction_count_min", pd.NA),
            "control_transaction_count": raw.get(
                "control_transaction_count", pd.NA
            ),
            "transaction_count_threshold": raw.get(
                "transaction_count_threshold", pd.NA
            ),
            "transaction_count_supported": raw.get(
                "transaction_count_supported", pd.NA
            ),
            "control_transaction_count_supported": raw.get(
                "control_transaction_count_supported", pd.NA
            ),
            "price_measure": raw.get("price_measure", effective_price_measure(row)),
            "donor_scope": raw.get("donor_scope", pd.NA),
            "estimator_backend": raw.get("estimator_backend", "python_pytorch"),
            "implementation_version": raw.get(
                "implementation_version", FORMAL_IMPLEMENTATION_VERSION
            ),
        }
    )


def run_python_gsc_scope(
    row: pd.Series, donor_scope: str
) -> tuple[bool, list[pd.DataFrame], dict[str, object]]:
    family = str(row.outcome_family)
    run_id = new_run_id()
    command, environment = python_estimator_command(row, "gsc", donor_scope, run_id)
    completed = run(command, environment)
    if completed.returncode != 0:
        return False, [], {
            "reason": "python_gsc_runtime_or_support_failure",
            "backend": "python_gpu",
            "log": completed.stdout[-4000:],
        }
    selected_method = (
        "xu_2017_gsynth_same_city"
        if donor_scope == "same_city"
        else "xu_2017_gsynth_all_city_standardized"
    )
    labels: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []
    for outcome in OUTCOMES[family]:
        path = gsc_output(row, outcome, donor_scope) / "causal_response_labels.parquet"
        manifest_path = path.parent / "manifest.csv"
        if not path.exists() or not manifest_path.exists():
            return False, [], {
                "reason": "python_gsc_manifest_or_labels_missing",
                "path": str(path.parent),
            }
        values = read_estimator_manifest(manifest_path)
        valid, selected_rank, cv_min_mspe = validate_python_estimator_manifest(
            values,
            row,
            estimator="gsc",
            outcome=outcome,
            donor_scope=donor_scope,
            run_id=run_id,
        )
        if (
            not valid
            or not float(selected_rank).is_integer()
            or values.get("labels_sha256") != file_sha256(path)
        ):
            return False, [], {
                "reason": "python_gsc_manifest_does_not_prove_current_run",
                "path": str(manifest_path),
            }
        raw = pq.read_table(path).to_pandas()
        raw = raw.loc[raw["event_time"].isin(HORIZONS[family])].copy()
        actual = set(pd.to_numeric(raw["event_time"], errors="coerce").dropna().astype(int))
        if actual != set(HORIZONS[family]):
            return False, [], {
                "reason": "python_gsc_outcome_horizon_grid_incomplete",
                "outcome": outcome,
                "actual_horizons": sorted(actual),
            }
        labels.append(normalized_python_labels(raw, row, selected_method))
        manifests.append(
            {
                "path": str(manifest_path.relative_to(ROOT)),
                "sha256": file_sha256(manifest_path),
                "labels_sha256": file_sha256(path),
                "selected_rank": int(selected_rank),
                "cv_min_mspe": cv_min_mspe,
                "run_id": run_id,
            }
        )
    return True, labels, {
        "selected_method": selected_method,
        "donor_scope": donor_scope,
        "backend": "python_gpu",
        "run_id": run_id,
        "estimator_manifests": manifests,
        "log": completed.stdout,
    }


def run_gsc_scope(
    row: pd.Series, donor_scope: str
) -> tuple[bool, list[pd.DataFrame], dict[str, object]]:
    if _ESTIMATOR_BACKEND == "python_gpu":
        return run_python_gsc_scope(row, donor_scope)
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
            _RUN_MODE,
            effective_price_measure(row),
            str(_LABEL_WINDOW),
        ],
        {
            "MIT_CAUSAL_RUN_ID": run_id,
            "MIT_SPECIFICATION_FINGERPRINT": specification_fingerprint(row),
        },
    )
    if completed.returncode != 0:
        log = completed.stdout
        if "non-finite treated-target counterfactual" in log:
            reason = "xu_gsc_target_counterfactual_nonfinite"
        elif "uncertainty event times do not align" in log:
            reason = "xu_gsc_uncertainty_event_time_mismatch"
        else:
            reason = "xu_gsc_runtime_or_support_failure"
        return False, [], {"reason": reason, "log": log}
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
        expected_production = _RUN_MODE == "production"
        expected_nboots = "200" if expected_production else "0"
        if (
            manifest_values.get("schema")
            != "complete_published_estimators_v3_explicit_deterministic_contracts"
            or manifest_values.get("run_id") != run_id
            or manifest_values.get("estimator") != "gsynth"
            or manifest_values.get("CV", "").upper() != "TRUE"
            or manifest_values.get("cv_method") != "rolling"
            or manifest_values.get("cv_folds") != "5"
            or manifest_values.get("cv_prop") != "0.1"
            or manifest_values.get("cv_nobs") != "3"
            or manifest_values.get("cv_buffer") != "1"
            or manifest_values.get("cv_rule") != "1se"
            or manifest_values.get("tol") != "1e-05"
            or manifest_values.get("max_iteration") != "5000"
            or manifest_values.get("run_mode") != _RUN_MODE
            or manifest_values.get("production_eligible", "").upper()
            != ("TRUE" if expected_production else "FALSE")
            or manifest_values.get("specification_fingerprint") != specification_fingerprint(row)
            or manifest_values.get("price_measure") != effective_price_measure(row)
            or manifest_values.get("observation_window")
            != str(_LABEL_WINDOW if frequency == "monthly" else 1)
            or manifest_values.get("inference") != "parametric"
            or manifest_values.get("nboots") != expected_nboots
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
        expected_horizons = set(HORIZONS[family])
        actual_horizons = set(pd.to_numeric(raw["event_time"], errors="coerce").dropna().astype(int))
        if actual_horizons != expected_horizons:
            return (
                False,
                [],
                {
                    "reason": "xu_gsc_outcome_horizon_grid_incomplete",
                    "outcome": outcome,
                    "expected_horizons": sorted(expected_horizons),
                    "actual_horizons": sorted(actual_horizons),
                },
            )
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
                "specification_fingerprint": specification_fingerprint(row),
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


_GSC_STRUCTURAL_FAILURE_REASONS = {
    "monthly_viirs_cache_unavailable",
}
_GSC_STRUCTURAL_FAILURE_PATTERNS = (
    "no post-treatment full-year outcome",
    "insufficient clean post-treatment annual periods",
    "insufficient post-treatment monthly periods",
)


def classify_gsc_failure(details: dict[str, object]) -> str | None:
    """Classify failures that cannot be rescued by MC or cross-city donors.

    Only failures that also make an MC panel impossible are structural.  GSC's
    complete-path and five-pre-period requirements are deliberately excluded:
    MC accepts partial target paths and one finite pre-treatment observation.
    """
    reason = str(details.get("reason") or "")
    if reason in _GSC_STRUCTURAL_FAILURE_REASONS:
        return "structural_support_failure"
    text = (reason + "\n" + str(details.get("log") or "")).lower()
    if any(pattern in text for pattern in _GSC_STRUCTURAL_FAILURE_PATTERNS):
        return "structural_support_failure"
    return None


def skip_after_structural_gsc_failure(
    queue: pd.DataFrame,
    index: int,
    row: pd.Series,
    gsc_details: dict[str, object],
) -> None:
    """Persist a bounded GSC support failure without launching fallback R jobs."""
    directory = task_directory(int(row.treatment_order), str(row.outcome_family))
    atomic_json(
        {"schema": "gsc_failure_before_skip_v1", **gsc_details},
        directory / "gsc_attempt.json",
    )
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
            "failure_class": classify_gsc_failure(gsc_details),
            "fallback_suppressed": True,
        },
        directory / "failure_manifest.json",
    )
    queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
        "skipped",
        pd.NA,
        str(gsc_details.get("reason")),
    ]
    atomic_csv(queue, FAMILY_QUEUE)


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
    if _RUN_MODE != "production":
        signature = f"{signature}_{_RUN_MODE}"
    if str(row.outcome_family) == "housing" and _TRANSACTION_COUNT_THRESHOLD != 1:
        signature = f"{signature}_tx{_TRANSACTION_COUNT_THRESHOLD}"
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
    if _RUN_MODE != "production":
        signature = f"{signature}_{_RUN_MODE}"
    if str(row.outcome_family) == "housing" and _TRANSACTION_COUNT_THRESHOLD != 1:
        signature = f"{signature}_tx{_TRANSACTION_COUNT_THRESHOLD}"
    return STAGING / "matrix_completion_runs" / str(row.city_key) / cohort / tag / signature


def run_python_mc_scope(
    row: pd.Series, donor_scope: str
) -> tuple[bool, list[pd.DataFrame], dict[str, object]]:
    family = str(row.outcome_family)
    run_id = new_run_id()
    command, environment = python_estimator_command(row, "mc", donor_scope, run_id)
    completed = run(command, environment)
    status_path = mc_family_run_output(row, donor_scope) / "outcome_status.csv"
    if completed.returncode != 0 and not status_path.exists():
        return False, [], {
            "reason": "python_mc_runtime_or_support_failure",
            "backend": "python_gpu",
            "log": completed.stdout[-4000:],
        }
    if not status_path.exists():
        return False, [], {
            "reason": "python_mc_family_status_missing",
            "log": completed.stdout[-4000:],
        }
    status = pd.read_csv(status_path)
    required = {"outcome", "status", "failure_reason", "run_id"}
    if not required.issubset(status.columns) or set(status["run_id"].astype(str)) != {run_id}:
        return False, [], {
            "reason": "python_mc_family_status_malformed_or_stale",
            "path": str(status_path),
        }
    selected_method = (
        "athey_2021_mc_same_city"
        if donor_scope == "same_city"
        else "athey_2021_mc_all_city_standardized"
    )
    labels: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    manifests: list[dict[str, object]] = []
    for outcome in OUTCOMES[family]:
        outcome_rows = status.loc[status["outcome"].astype(str).eq(outcome)]
        if len(outcome_rows) != 1 or str(outcome_rows.iloc[0]["status"]) != "success":
            reason = outcome_rows.iloc[0].get("failure_reason") if len(outcome_rows) else None
            failures[outcome] = str(reason or "python_mc_outcome_failed")
            continue
        path = mc_output(row, outcome, donor_scope) / "causal_response_labels.parquet"
        manifest_path = path.parent / "manifest.csv"
        if not path.exists() or not manifest_path.exists():
            failures[outcome] = "python_mc_success_record_lacks_manifest_or_labels"
            continue
        values = read_estimator_manifest(manifest_path)
        valid, selected_lambda, cv_min_mspe = validate_python_estimator_manifest(
            values,
            row,
            estimator="mc",
            outcome=outcome,
            donor_scope=donor_scope,
            run_id=run_id,
        )
        if (
            not valid
            or selected_lambda < 0
            or values.get("labels_sha256") != file_sha256(path)
        ):
            failures[outcome] = "python_mc_manifest_does_not_prove_current_run"
            continue
        raw = pq.read_table(path).to_pandas()
        raw = raw.loc[raw["event_time"].isin(HORIZONS[family])].copy()
        actual = set(pd.to_numeric(raw["event_time"], errors="coerce").dropna().astype(int))
        if actual != set(HORIZONS[family]):
            failures[outcome] = "python_mc_outcome_horizon_grid_incomplete"
            continue
        if raw.empty or not bool(raw["label_available"].fillna(False).any()):
            failures[outcome] = "python_mc_has_no_available_target_horizon"
            continue
        labels.append(normalized_python_labels(raw, row, selected_method))
        manifests.append(
            {
                "path": str(manifest_path.relative_to(ROOT)),
                "sha256": file_sha256(manifest_path),
                "labels_sha256": file_sha256(path),
                "selected_lambda": selected_lambda,
                "cv_min_mspe": cv_min_mspe,
                "run_id": run_id,
            }
        )
    if not labels:
        return False, [], {
            "reason": "python_mc_no_outcome_produced_available_labels",
            "backend": "python_gpu",
            "outcome_failures": failures,
            "log": completed.stdout[-4000:],
        }
    return True, labels, {
        "selected_method": selected_method,
        "donor_scope": donor_scope,
        "backend": "python_gpu",
        "run_id": run_id,
        "estimator_manifests": manifests,
        "outcome_failures": failures,
        "outcome_status": str(status_path.relative_to(ROOT)),
        "outcome_status_sha256": file_sha256(status_path),
        "log": completed.stdout,
    }


def run_mc_scope(
    row: pd.Series, donor_scope: str
) -> tuple[bool, list[pd.DataFrame], dict[str, object]]:
    if _ESTIMATOR_BACKEND == "python_gpu":
        return run_python_mc_scope(row, donor_scope)
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
            _RUN_MODE,
            effective_price_measure(row),
            str(_LABEL_WINDOW),
        ],
        {
            "MIT_CAUSAL_RUN_ID": run_id,
            "MIT_SPECIFICATION_FINGERPRINT": specification_fingerprint(row),
        },
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
        expected_production = _RUN_MODE == "production"
        expected_nboots = "200" if expected_production else "0"
        if (
            manifest_values.get("schema")
            != "complete_published_estimators_v3_explicit_deterministic_contracts"
            or manifest_values.get("estimator") != "mc"
            or manifest_values.get("fitted_method") != "mc"
            or manifest_values.get("backend") != "fect"
            or manifest_values.get("force") != "two-way"
            or manifest_values.get("criterion") != "mspe"
            or manifest_values.get("nlambda") != "20"
            or manifest_values.get("min_T0") != "1"
            or manifest_values.get("se", "").upper() != "TRUE"
            or manifest_values.get("run_id") != run_id
            or manifest_values.get("CV", "").upper() != "TRUE"
            or manifest_values.get("cv_method") != "rolling"
            or manifest_values.get("cv_folds") != "20"
            or manifest_values.get("cv_prop") != "0.1"
            or manifest_values.get("cv_rule") != "1se"
            or manifest_values.get("cv_nobs") != "1"
            or manifest_values.get("cv_donut") != "0"
            or manifest_values.get("cv_buffer") != "0"
            or manifest_values.get("tol") != "1e-05"
            or manifest_values.get("max_iteration") != "5000"
            or manifest_values.get("two_stage_cv_inference", "").upper() != "TRUE"
            or manifest_values.get("inference_fit_CV", "").upper() != "FALSE"
            or manifest_values.get("run_mode") != _RUN_MODE
            or manifest_values.get("production_eligible", "").upper()
            != ("TRUE" if expected_production else "FALSE")
            or manifest_values.get("specification_fingerprint") != specification_fingerprint(row)
            or manifest_values.get("price_measure") != effective_price_measure(row)
            or manifest_values.get("observation_window")
            != str(_LABEL_WINDOW if frequency == "monthly" else 1)
            or manifest_values.get("inference") != "jackknife"
            or manifest_values.get("nboots") != expected_nboots
            or not math.isfinite(selected_lambda)
            or selected_lambda < 0
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
        expected_horizons = set(HORIZONS[family])
        actual_horizons = set(pd.to_numeric(raw["event_time"], errors="coerce").dropna().astype(int))
        if actual_horizons != expected_horizons:
            outcome_failures[outcome] = "mc_outcome_horizon_grid_incomplete"
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
                "specification_fingerprint": specification_fingerprint(row),
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
    task_details = dict(details)
    if _ESTIMATOR_BACKEND == "python_gpu" and _RUN_MODE == "production":
        if not _QUALIFICATION_PROOF:
            raise ValueError("Production Python task lacks formal qualification proof")
        task_details.update(_QUALIFICATION_PROOF)
    result = pd.concat(frames, ignore_index=True)
    result["donor_scope"] = str(task_details.get("donor_scope") or "") or pd.NA
    result["estimator_backend"] = str(
        task_details.get("backend")
        or ("python_gpu" if _ESTIMATOR_BACKEND == "python_gpu" else "r_reference")
    )
    result["implementation_version"] = (
        FORMAL_IMPLEMENTATION_VERSION
        if _ESTIMATOR_BACKEND == "python_gpu"
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
            "anticipation_months": _ANTICIPATION_MONTHS,
            "observation_window": _LABEL_WINDOW,
            "price_measure": effective_price_measure(row),
            "production_eligible": _RUN_MODE == "production",
            "run_mode": _RUN_MODE,
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
    if stage not in {"cross_gsc_running", "cross_mc_running"}:
        if control_queue is None:
            control_queue = read_control_queue()
        queue.loc[index, "status"] = "cross_matching_running"
        atomic_csv(queue, FAMILY_QUEUE)
        cross_match_ok = False
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

    # Round 5: cross-city GSC. The cross-city estimator reads VIIRS
    # partitions for every donor city, so materialise the window across all
    # donor cities first (the same-city rounds only need the target city).
    if stage != "cross_mc_running":
        queue.loc[index, "status"] = "cross_gsc_running"
        atomic_csv(queue, FAMILY_QUEUE)
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
        atomic_csv(queue, FAMILY_QUEUE)
        return

    # Round 6: cross-city MC (VIIRS cache already materialised in round 5
    # for the same donor scope, so no repeated materialisation here).
    if stage == "cross_mc_running" and cross_gsc_attempt_path.exists():
        cross_gsc_details = json.loads(cross_gsc_attempt_path.read_text(encoding="utf-8"))
    queue.loc[index, "status"] = "cross_mc_running"
    atomic_csv(queue, FAMILY_QUEUE)
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
            if classify_gsc_failure(gsc_details):
                skip_after_structural_gsc_failure(queue, index, row, gsc_details)
            else:
                begin_mc_stage(queue, index, row, gsc_details)
            if not classify_gsc_failure(gsc_details) and phase != "gsc":
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
        if classify_gsc_failure(gsc_details):
            skip_after_structural_gsc_failure(queue, index, row, gsc_details)
        else:
            begin_mc_stage(queue, index, row, gsc_details)
        if not classify_gsc_failure(gsc_details) and phase != "gsc":
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
    tasks: set[tuple[int, str]] | None = None,
) -> pd.Index:
    statuses = {
        "matching": {"pending", "matching_running"},
        "gsc": {"gsc_pending", "gsc_running"},
        "mc": {
            "mc_pending",
            "mc_running",
            "cross_matching_running",
            "cross_gsc_running",
            "cross_mc_running",
        },
        "all": {
            "pending",
            "matching_running",
            "gsc_pending",
            "gsc_running",
            "mc_pending",
            "mc_running",
            "cross_matching_running",
            "cross_gsc_running",
            "cross_mc_running",
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
    if tasks is not None:
        task_index = pd.MultiIndex.from_arrays(
            [queue["treatment_order"].astype(int), queue["outcome_family"].astype(str)]
        )
        mask &= task_index.isin(tasks)
    return queue.index[mask][:max_tasks]


def main() -> int:
    global FAMILY_QUEUE, _ANTICIPATION_MONTHS, _RUN_MODE, _ESTIMATOR_BACKEND
    global _QUALIFICATION_RECEIPT, _QUALIFICATION_PROOF
    global _MAX_GSC_CROSS_CITY_DONORS, _GSC_DONOR_SAMPLING_SEED
    global _TRANSACTION_COUNT_THRESHOLD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-order", type=int, default=1)
    parser.add_argument("--end-order", type=int)
    parser.add_argument(
        "--orders",
        help="Comma-separated treatment orders to process (mutually exclusive "
        "with --start-order/--end-order ranges)",
    )
    parser.add_argument(
        "--orders-file",
        type=Path,
        help="CSV containing treatment_order values; mutually exclusive with --orders",
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
        choices=("main", "median", "hedonic"),
        default="median",
        help="Housing price measure: main = hedonic where the city panel exists, "
        "otherwise median; median/hedonic force one measure.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Observation-window width in months for monthly labels "
        "(main housing specification uses 3; 1/6 are sensitivity views)",
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
        "--run-mode",
        choices=("production", "preview"),
        default="production",
        help="Use isolated preview artifacts with point estimates only, or formal production inference.",
    )
    parser.add_argument(
        "--transaction-count-threshold",
        type=int,
        default=1,
        help="Minimum transactions for every housing grid-month used by Matching, GSC, or MC.",
    )
    parser.add_argument(
        "--estimator-backend",
        choices=("python_gpu", "r_reference"),
        default=DEFAULT_ESTIMATOR_BACKEND,
        help="Run formal GSC/MC with Python/PyTorch (default) or the audited R reference.",
    )
    parser.add_argument(
        "--qualification-receipt",
        type=Path,
        default=(
            Path(os.environ["MIT_CAUSAL_QUALIFICATION_RECEIPT"])
            if os.environ.get("MIT_CAUSAL_QUALIFICATION_RECEIPT")
            else None
        ),
        help="Eligible R/Python parity audit receipt required by production Python tasks.",
    )
    parser.add_argument(
        "--max-gsc-cross-city-donors",
        type=int,
        default=50_000,
        help="Pre-outcome deterministic donor cap for cross-city GSC.",
    )
    parser.add_argument(
        "--gsc-donor-sampling-seed",
        type=int,
        default=20260823,
        help="Fixed seed embedded in the cross-city GSC donor-sampling contract.",
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        help="CSV with treatment_order and outcome_family; restrict formal reruns to these task keys.",
    )
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
    _RUN_MODE = args.run_mode
    _ESTIMATOR_BACKEND = args.estimator_backend
    _QUALIFICATION_RECEIPT = args.qualification_receipt
    _MAX_GSC_CROSS_CITY_DONORS = args.max_gsc_cross_city_donors
    _GSC_DONOR_SAMPLING_SEED = args.gsc_donor_sampling_seed
    _ANTICIPATION_MONTHS = args.anticipation_months
    global _PRICE_MEASURE, _LABEL_WINDOW
    _PRICE_MEASURE = args.price_measure
    _LABEL_WINDOW = args.window
    _TRANSACTION_COUNT_THRESHOLD = args.transaction_count_threshold
    print(
        f"Configured causal-label backend={_ESTIMATOR_BACKEND}, "
        f"run_mode={_RUN_MODE}, gpu_ids={os.environ.get('MIT_CAUSAL_GPU_IDS', 'auto')}"
    )
    if _MAX_GSC_CROSS_CITY_DONORS < 20:
        parser.error("--max-gsc-cross-city-donors must be at least 20")
    if _TRANSACTION_COUNT_THRESHOLD < 1:
        parser.error("--transaction-count-threshold must be positive")
    if _RUN_MODE == "production" and not args.dry_run:
        try:
            validate_frozen_formal_matching_spec(ROOT)
        except ValueError as exc:
            parser.error(str(exc))

    has_shard = args.shard_id is not None and args.shard_count is not None
    if bool(args.shard_id is not None) != bool(args.shard_count is not None):
        parser.error("--shard-id and --shard-count must be used together")

    family_master_queue = queue_variant_path(FAMILY_QUEUE, _RUN_MODE)
    unit_master_queue = queue_variant_path(UNIT_QUEUE, _RUN_MODE)
    control_master_queue = queue_variant_path(CONTROL_QUEUE, _RUN_MODE)
    if _RUN_MODE != "production":
        import shutil as _shutil

        for source, target in (
            (FAMILY_QUEUE, family_master_queue),
            (UNIT_QUEUE, unit_master_queue),
            (CONTROL_QUEUE, control_master_queue),
        ):
            if not target.exists():
                _shutil.copy2(source, target)

    family_queue = family_master_queue
    unit_queue = unit_master_queue
    control_queue_path = control_master_queue

    if has_shard:
        family_queue = shard_queue_path(family_master_queue, args.shard_id)
        unit_queue = shard_queue_path(unit_master_queue, args.shard_id)
        control_queue_path = shard_queue_path(control_master_queue, args.shard_id)
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
            _shutil.copy2(control_master_queue, control_queue_path)
            if unit_master_queue.exists():
                _shutil.copy2(unit_master_queue, unit_queue)
            print(f"Initialized shard {args.shard_id + 1}/{args.shard_count}: {family_queue.name}")

        # Compute this shard's portion of the selected treatment orders.
        treatments = pq.read_table(
            TREATMENT_UNIT_LIST,
            columns=["treatment_order"],
        ).to_pandas()
        all_orders = sorted(treatments["treatment_order"].astype(int).tolist())
        # For a representative sample, balance the selected orders themselves
        # rather than the full 5,048-order universe.  Otherwise a sample
        # concentrated in later opening cohorts can leave most shards idle.
        shard_pool = all_orders
        if args.tasks_file is not None:
            task_path = args.tasks_file if args.tasks_file.is_absolute() else ROOT / args.tasks_file
            shard_pool = sorted({order for order, _ in read_tasks_file(task_path)})
        elif args.orders_file is not None:
            sample_path = args.orders_file if args.orders_file.is_absolute() else ROOT / args.orders_file
            shard_pool = sorted(read_orders_file(sample_path))
        elif args.orders is not None:
            shard_pool = sorted({int(value) for value in args.orders.split(",") if value.strip()})
        shard_orders = set(shard_order_slice(shard_pool, args.shard_id, args.shard_count))
        if not shard_orders:
            print(f"Shard {args.shard_id + 1}/{args.shard_count}: no assigned orders")
            return 0
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
    if (
        _ESTIMATOR_BACKEND == "python_gpu"
        and _RUN_MODE == "production"
        and not args.dry_run
    ):
        if _QUALIFICATION_RECEIPT is None:
            parser.error(
                "production Python tasks require --qualification-receipt "
                "(or MIT_CAUSAL_QUALIFICATION_RECEIPT)"
            )
        _QUALIFICATION_PROOF = validate_formal_qualification_receipt(
            _QUALIFICATION_RECEIPT
        )
    else:
        _QUALIFICATION_PROOF = {}
    orders_set: set[int] | None = None
    task_keys: set[tuple[int, str]] | None = None
    if args.tasks_file is not None:
        if args.orders is not None or args.orders_file is not None:
            parser.error("--tasks-file is mutually exclusive with --orders/--orders-file")
        task_path = args.tasks_file if args.tasks_file.is_absolute() else ROOT / args.tasks_file
        task_keys = read_tasks_file(task_path)
        orders_set = {order for order, _ in task_keys}
        if has_shard:
            # A task-file run must be partitioned by the same selected-order
            # pool used to initialize the shard.  Without this restriction,
            # every shard processes the full task file and concurrent workers
            # overwrite the same fixed-control staging artifacts.
            orders_set &= shard_orders
            task_keys = {
                (order, family) for order, family in task_keys if order in orders_set
            }
            if not orders_set:
                print(f"Shard {args.shard_id + 1}/{args.shard_count}: no selected task orders")
                return 0
    elif args.orders is not None or args.orders_file is not None:
        if args.orders is not None and args.orders_file is not None:
            parser.error("--orders and --orders-file are mutually exclusive")
        orders_set = (
            {int(value) for value in args.orders.split(",") if value.strip()}
            if args.orders is not None
            else read_orders_file(
                args.orders_file if args.orders_file.is_absolute() else ROOT / args.orders_file
            )
        )
        if not orders_set:
            parser.error("selected orders must contain at least one treatment order")
        if not has_shard and (args.start_order != 1 or args.end_order is not None):
            parser.error("selected orders are mutually exclusive with --start-order/--end-order")
        available_orders = set(queue["treatment_order"].astype(int))
        missing_orders = sorted(orders_set - available_orders)
        if missing_orders:
            parser.error(f"orders file contains unknown treatment orders: {missing_orders[:10]}")
        if has_shard:
            orders_set &= shard_orders
            if not orders_set:
                print(f"Shard {args.shard_id + 1}/{args.shard_count}: no selected sample orders")
                return 0
        invalidated = invalidate_stale_terminal_tasks(queue, orders_set)
        if invalidated:
            print(f"Invalidated {invalidated} terminal tasks from a different specification")
    eligible = eligible_indices(
        queue,
        args.start_order,
        args.end_order,
        args.family,
        args.phase,
        args.max_tasks
        if task_keys is not None
        else (len(orders_set) * 4 if orders_set is not None else args.max_tasks),
        retry_matching=args.retry_matching,
        retry_skipped=args.retry_skipped,
        orders=orders_set,
        tasks=task_keys,
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
        if not args.dry_run and _RUN_MODE == "production":
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
