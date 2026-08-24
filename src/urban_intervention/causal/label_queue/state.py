"""Behavior-preserving component of the modular causal label queue."""

from __future__ import annotations

import os
import signal
import subprocess
import uuid
from contextlib import suppress
from functools import partial
from pathlib import Path

import pandas as pd

from urban_intervention.causal.gpu.contracts import (
    FORMAL_IMPLEMENTATION_VERSION,
)
from urban_intervention.utils import atomic_write_csv as atomic_csv
from urban_intervention.utils import atomic_write_json

from .runtime import (
    CONTROL_QUEUE,
    OUTCOMES,
    R_LIB,
    ROOT,
    TASK_ROOT,
    UNIT_QUEUE,
    settings,
)

atomic_json = partial(atomic_write_json, default=str)


def effective_price_measure(row: pd.Series) -> str:
    """Resolve the housing measure for the approved mixed main specification."""
    if settings.price_measure != "main" or str(row.outcome_family) != "housing":
        return "median" if settings.price_measure == "main" else settings.price_measure
    hedonic_path = (
        ROOT / "outputs" / "causal_labels" / "housing_hedonic" / (f"{row.city_key}_monthly.parquet")
    )
    return "hedonic" if hedonic_path.exists() else "median"


def specification_fingerprint(row: pd.Series) -> str:
    """Identify the exact label specification used by this queue process."""
    backend = (
        FORMAL_IMPLEMENTATION_VERSION
        if settings.estimator_backend == "python_gpu"
        else "r-reference"
    )
    return (
        f"main_a6_r1km__a{settings.anticipation_months}__w{settings.label_window}"
        f"__price_{effective_price_measure(row)}__tx{settings.transaction_count_threshold}"
        f"__backend_{backend}"
        f"__xgsc{settings.max_gsc_cross_city_donors}s{settings.gsc_donor_sampling_seed}"
    )


def read_family_queue(path: Path = settings.family_queue) -> pd.DataFrame:
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
        stdout, _ = process.communicate(timeout=settings.r_timeout_seconds)
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
        output = (stdout or "") + f"\nR subprocess timed out after {settings.r_timeout_seconds}s\n"
        if exc.stdout:
            output = str(exc.stdout) + output
        return subprocess.CompletedProcess(command, 124, output)
    return subprocess.CompletedProcess(command, process.returncode, stdout or "")


def task_directory(order: int, family: str) -> Path:
    root = TASK_ROOT if settings.run_mode == "production" else TASK_ROOT.parent / "tasks_preview"
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


def read_estimator_manifest(path: Path) -> dict[str, str]:
    manifest = pd.read_csv(path)
    if set(manifest.columns) != {"field", "value"} or manifest["field"].duplicated().any():
        raise ValueError(f"Malformed estimator manifest: {path}")
    return dict(zip(manifest["field"].astype(str), manifest["value"].astype(str), strict=False))
