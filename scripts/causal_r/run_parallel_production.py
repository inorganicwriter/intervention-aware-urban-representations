"""Parallel production launcher for the causal-label pipeline.

Phase 1: Control design (parallel Python/GPU workers by default)
Phase 2: Causal labels (one Python/GPU shard per GPU by default)
Phase 3: Merge shard queues back to master queue

Usage:
  For the approved 300-grid sample:

      python scripts/causal_r/run_parallel_production.py --phase 2 \
          --orders-file outputs/causal_labels/representative_sample_300.csv

  Or run one phase at a time:

      python scripts/causal_r/run_parallel_production.py --phase 1  # control design
      python scripts/causal_r/run_parallel_production.py --phase 2  # causal labels
      python scripts/causal_r/run_parallel_production.py --phase 3  # merge shards

  Dry-run first:

      python scripts/causal_r/run_parallel_production.py --phase 1 --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from urban_intervention.causal.gpu.qualification import (  # noqa: E402
    validate_formal_qualification_receipt,
)
from urban_intervention.causal.setup_inputs import (  # noqa: E402
    validate_frozen_formal_matching_spec,
)
from urban_intervention.data.paths import (  # noqa: E402
    CONTROL_DESIGN_QUEUE,
    COUNTERFACTUAL_QUEUE,
    OUTCOME_FAMILY_QUEUE,
    OUTPUT_PYTHON_ESTIMATORS_DIR,
    PROJECT_ROOT,
    R_LIB_DIR,
    TREATMENT_UNIT_LIST,
    r_script,
)
from urban_intervention.utils import atomic_write_csv  # noqa: E402

R_SCRIPT = os.environ.get("MIT_RSCRIPT", "Rscript")
R_LIB = Path(os.environ.get("MIT_R_LIB", str(R_LIB_DIR)))
VIIRS_RAW = os.environ.get("MIT_VIIRS_RAW")
ROOT = PROJECT_ROOT


# The deployment server has four GPUs.  One formal estimator process per GPU
# avoids bootstrap/jackknife jobs competing for the same device memory.
DEFAULT_GPU_IDS = "0,1,2,3"
DEFAULT_SHARD_COUNT = 4
DEFAULT_CONTROL_WORKERS = 4


def parse_gpu_ids(value: str) -> list[str]:
    """Parse a comma-separated list of non-negative CUDA device indices."""
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    if any(not item.isdigit() for item in gpu_ids):
        raise ValueError("GPU ids must be comma-separated non-negative integers")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU ids must not contain duplicates")
    return gpu_ids


def read_orders_file(path: Path) -> list[int]:
    frame = pd.read_csv(path)
    if "treatment_order" not in frame.columns:
        raise ValueError(f"Orders file lacks treatment_order: {path}")
    values = pd.to_numeric(frame["treatment_order"], errors="raise").astype(int).tolist()
    orders = sorted(set(values))
    if len(orders) != len(values):
        raise ValueError(f"Orders file contains duplicate treatment_order values: {path}")
    if not orders:
        raise ValueError(f"Orders file is empty: {path}")
    return orders


def read_tasks_file(path: Path) -> set[tuple[int, str]]:
    frame = pd.read_csv(path)
    required = {"treatment_order", "outcome_family"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Tasks file lacks columns: {sorted(missing)}")
    keys = {
        (int(order), str(family))
        for order, family in zip(frame["treatment_order"], frame["outcome_family"], strict=False)
    }
    if len(keys) != len(frame) or not keys:
        raise ValueError(f"Tasks file is empty or contains duplicate keys: {path}")
    return keys


def queue_variant_path(path: Path, run_mode: str) -> Path:
    if run_mode == "production":
        return path
    return path.with_name(f"{path.stem}_{run_mode}{path.suffix}")


def shard_queue_path(path: Path, shard_id: int) -> Path:
    return path.with_name(f"{path.stem}_shard_{shard_id:02d}{path.suffix}")


def isolated_queue_paths(namespace: str | None) -> tuple[Path, Path, Path]:
    """Return family/control/unit queue paths for one isolated specification."""
    if namespace is None:
        return OUTCOME_FAMILY_QUEUE, CONTROL_DESIGN_QUEUE, COUNTERFACTUAL_QUEUE
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", namespace):
        raise ValueError(
            "queue namespace must contain only lowercase letters, digits, '_' or '-'"
        )
    return tuple(
        path.with_name(f"{path.stem}_{namespace}{path.suffix}")
        for path in (OUTCOME_FAMILY_QUEUE, CONTROL_DESIGN_QUEUE, COUNTERFACTUAL_QUEUE)
    )  # type: ignore[return-value]


def initialize_isolated_queues(
    family_queue: Path, control_queue: Path, unit_queue: Path, *, force: bool = False
) -> None:
    """Create pending analysis queues while preserving frozen Phase 1 controls.

    ``force`` implements an explicit ``--reset-queues`` for an isolated queue
    namespace; without it a namespace rebuild silently resumes the old files
    because the canonical reset never touches namespace queues.
    """
    targets = (family_queue, control_queue, unit_queue)
    canonical = (OUTCOME_FAMILY_QUEUE, CONTROL_DESIGN_QUEUE, COUNTERFACTUAL_QUEUE)
    if targets == canonical:
        return
    if force:
        for path in targets:
            path.unlink(missing_ok=True)
    existing = [path.exists() for path in targets]
    if any(existing) and not all(existing):
        raise RuntimeError(
            "isolated queue namespace is incomplete; remove all three namespace "
            "queues or restore the missing files before retrying"
        )
    if all(existing):
        return
    family = pd.read_csv(OUTCOME_FAMILY_QUEUE)
    family["status"] = "pending"
    for column in ("selected_method", "failure_reason"):
        if column in family:
            family[column] = pd.NA
    unit = pd.read_csv(COUNTERFACTUAL_QUEUE)
    unit["status"] = "pending"
    for column in ("selected_method", "selected_control_grid_id", "failure_reason"):
        if column in unit:
            unit[column] = pd.NA
    atomic_write_csv(family, family_queue)
    atomic_write_csv(pd.read_csv(CONTROL_DESIGN_QUEUE), control_queue)
    atomic_write_csv(unit, unit_queue)
    print(f"Initialized isolated queues: {family_queue.stem}")


def validate_resume_checkpoint(master_family: Path, shard_count: int) -> None:
    """Require durable, non-empty progress before allowing receipt-only resume."""
    paths = [shard_queue_path(master_family, shard_id) for shard_id in range(shard_count)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(
            "prevalidated resume requires every existing family shard; missing "
            + ", ".join(str(path) for path in missing)
        )
    statuses = pd.concat(
        [pd.read_csv(path, usecols=["status"])["status"].astype(str) for path in paths],
        ignore_index=True,
    )
    if statuses.empty or statuses.eq("pending").all():
        raise ValueError(
            "prevalidated resume requires a shard checkpoint with prior task progress"
        )


def resource_environment(
    base: dict[str, str], gsc_cv_cores: int, gsc_bootstrap_cores: int, mc_cores: int
) -> dict[str, str]:
    env = base.copy()
    env["MIT_PROJECT_ROOT"] = str(ROOT)
    env["MIT_R_LIB"] = str(R_LIB)
    env["MIT_GSC_CV_CORES"] = str(gsc_cv_cores)
    env["MIT_GSC_BOOTSTRAP_CORES"] = str(gsc_bootstrap_cores)
    env["MIT_MC_CORES"] = str(mc_cores)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(variable, "1")
    env.setdefault("R_DATATABLE_NUM_THREADS", "1")
    return env


def phase1_control_design(
    dry_run: bool,
    workers: int = DEFAULT_CONTROL_WORKERS,
    orders_file: Path | None = None,
    estimator_backend: str = "python_gpu",
    gpu_ids: list[str] | None = None,
    env: dict[str, str] | None = None,
    force_recompute: bool = False,
    repair_stale_controls: bool = False,
) -> None:
    """Run control design for the selected treatment orders."""
    print("=" * 60)
    print(f"PHASE 1: Grid-level control design (workers={workers})")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(r_script("run_grid_control_design_queue.py")),
        "--start-order",
        "1",
        "--workers",
        str(workers),
        "--estimator-backend",
        estimator_backend,
    ]
    if orders_file is not None:
        cmd.extend(["--orders-file", str(orders_file)])
    else:
        cmd.extend(["--max-units", "5048"])
    if dry_run:
        cmd.append("--dry-run")
    if force_recompute:
        cmd.append("--retry")
    elif repair_stale_controls:
        cmd.append("--repair-stale")

    env = env or os.environ.copy()
    env["R_LIBS_USER"] = str(R_LIB)
    if estimator_backend == "python_gpu" and gpu_ids:
        env["MIT_CAUSAL_GPU_IDS"] = ",".join(gpu_ids)
    start = time.time()
    result = subprocess.run(
        cmd, cwd=str(ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    elapsed = time.time() - start
    print(result.stdout[-2000:] if result.stdout else "(no output)")
    if result.returncode != 0:
        raise RuntimeError(f"Phase 1 failed with code {result.returncode}")
    print(f"Phase 1 completed in {timedelta(seconds=int(elapsed))}")


def phase2_round_specs(
    queue_phase: str,
    task_scheduling: str,
    dry_run: bool,
) -> list[tuple[str, Path, list[str]]]:
    """Return the ordered Phase 2 worker rounds.

    Dry runs and explicitly selected legacy phases remain single-pass.  A
    production ``all`` run uses global barriers so cheap same-city work is
    exhausted before any task starts an expensive cross-city fallback.
    """
    round_runner = ROOT / "scripts" / "causal_python" / "run_causal_label_round_queue.py"
    if task_scheduling != "global_rounds" or queue_phase != "all" or dry_run:
        return [("task cascade", round_runner, ["--phase", queue_phase])]
    return [
        (
            "same-city Matching",
            round_runner,
            ["--global-round", "same_city_matching"],
        ),
        ("same-city GSC", round_runner, ["--global-round", "same_city_gsc"]),
        ("same-city MC", round_runner, ["--global-round", "same_city_mc"]),
        (
            "cross-city Matching",
            round_runner,
            ["--global-round", "cross_city_matching"],
        ),
        ("cross-city GSC", round_runner, ["--global-round", "cross_city_gsc"]),
        ("cross-city MC", round_runner, ["--global-round", "cross_city_mc"]),
    ]


def phase2_causal_labels(
    dry_run: bool,
    shard_count: int = DEFAULT_SHARD_COUNT,
    orders_file: Path | None = None,
    tasks_file: Path | None = None,
    run_mode: str = "production",
    price_measure: str = "main",
    window: int = 1,
    anticipation_months: int = 0,
    queue_phase: str = "all",
    retry_matching: bool = False,
    reset_shards: bool = False,
    estimator_backend: str = "python_gpu",
    gpu_ids: list[str] | None = None,
    env: dict[str, str] | None = None,
    qualification_receipt: Path | None = None,
    max_gsc_cross_city_donors: int = 5_000,
    gsc_donor_sampling_seed: int = 20260823,
    transaction_count_threshold: int = 1,
    task_scheduling: str = "global_rounds",
    queue_paths: tuple[Path, Path, Path] | None = None,
    round_pool_db: Path | None = None,
    round_pool_run_root: Path | None = None,
    round_pool_run_id: str | None = None,
    queue_namespace: str | None = None,
) -> None:
    """Launch shard workers for the selected treatment-order sample."""
    print("=" * 60)
    print(f"PHASE 2: Causal labels ({shard_count} shards)")
    print("=" * 60)

    master_family_base, master_control_base, master_unit_base = (
        queue_paths
        if queue_paths is not None
        else (OUTCOME_FAMILY_QUEUE, CONTROL_DESIGN_QUEUE, COUNTERFACTUAL_QUEUE)
    )
    initialize_isolated_queues(
        master_family_base, master_control_base, master_unit_base, force=reset_shards
    )

    # Validate control design is complete
    cq = pd.read_csv(master_control_base)
    statuses = cq["status"].astype(str)
    task_keys = read_tasks_file(tasks_file) if tasks_file is not None else None
    selected_orders = (
        {order for order, _ in task_keys}
        if task_keys is not None
        else (
            set(read_orders_file(orders_file))
            if orders_file is not None
            else set(cq["treatment_order"].astype(int))
        )
    )
    queue_orders = set(cq["treatment_order"].astype(int))
    missing_orders = sorted(selected_orders - queue_orders)
    if missing_orders:
        raise RuntimeError(f"Orders file contains unknown treatment orders: {missing_orders[:10]}")
    scoped = cq[cq["treatment_order"].astype(int).isin(selected_orders)]
    statuses = scoped["status"].astype(str)
    terminal_control = statuses.isin({"matched", "gsc_pending"})
    n_valid = terminal_control.sum()
    n_total = len(scoped)
    if n_valid < n_total:
        raise RuntimeError(
            f"Phase 1 incomplete: only {n_valid}/{n_total} control designs "
            f"in terminal state (matched or gsc_pending). "
            f"Run Phase 1 first with: --phase 1"
        )
    n_matched = (statuses == "matched").sum()
    print(f"  Control designs: {n_matched} matched, {n_total - n_matched} gsc_pending, 0 error")

    # Prepare master queues for each shard.  Existing shard files are the
    # durable checkpoint for an interrupted Phase 2 and must be preserved.
    # They are archived and rebuilt only for an explicit queue reset.
    master_family = queue_variant_path(master_family_base, run_mode)
    master_control = queue_variant_path(master_control_base, run_mode)
    master_unit = queue_variant_path(master_unit_base, run_mode)
    if run_mode != "production":
        for source, target in (
            (master_family_base, master_family),
            (master_control_base, master_control),
            (master_unit_base, master_unit),
        ):
            if not target.exists():
                shutil.copy2(source, target)
    master_paths = [master_family, master_control, master_unit]
    if task_scheduling == "round_pool":
        if queue_phase != "all" or dry_run:
            raise ValueError("round_pool requires --queue-phase all and cannot be a dry run")
        phase2_round_pool(
            shard_count=shard_count,
            tasks_file=tasks_file,
            orders_file=orders_file,
            run_mode=run_mode,
            price_measure=price_measure,
            window=window,
            anticipation_months=anticipation_months,
            estimator_backend=estimator_backend,
            gpu_ids=gpu_ids or [],
            env=env,
            qualification_receipt=qualification_receipt,
            max_gsc_cross_city_donors=max_gsc_cross_city_donors,
            gsc_donor_sampling_seed=gsc_donor_sampling_seed,
            transaction_count_threshold=transaction_count_threshold,
            master_paths=(master_family, master_control, master_unit),
            round_pool_db=round_pool_db,
            round_pool_run_root=round_pool_run_root,
            round_pool_run_id=round_pool_run_id,
            round_pool_queue_namespace=queue_namespace or "production",
        )
        return
    shard_paths = []
    for shard_id in range(shard_count):
        shard_paths.extend(shard_queue_path(master_path, shard_id) for master_path in master_paths)

    if reset_shards:
        existing = [path for path in shard_paths if path.exists()]
        if existing:
            archive_dir = (
                ROOT
                / "outputs"
                / "archive"
                / f"causal_shards_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )
            archive_dir.mkdir(parents=True, exist_ok=False)
            for path in existing:
                shutil.move(str(path), str(archive_dir / path.name))
            print(f"Archived {len(existing)} existing shard queues to {archive_dir}")

    copied = 0
    resumed = 0
    for shard_id in range(shard_count):
        for master_path in master_paths:
            shard_path = shard_queue_path(master_path, shard_id)
            if shard_path.exists():
                master_frame = pd.read_csv(master_path)
                shard_frame = pd.read_csv(shard_path)
                if list(master_frame.columns) != list(shard_frame.columns):
                    raise RuntimeError(
                        f"Shard queue schema differs from master: {shard_path}. "
                        "Use --reset-queues to rebuild shard queues."
                    )
                key_columns = ["treatment_order"]
                if "outcome_family" in master_frame.columns:
                    key_columns.append("outcome_family")
                if len(shard_frame) != len(master_frame) or set(
                    map(tuple, shard_frame[key_columns].to_numpy())
                ) != set(map(tuple, master_frame[key_columns].to_numpy())):
                    raise RuntimeError(
                        f"Shard queue keys differ from master: {shard_path}. "
                        "Use --reset-queues to rebuild shard queues."
                    )
                resumed += 1
                continue
            shutil.copy2(master_path, shard_path)
            copied += 1
    print(f"Shard queues ready: {copied} copied, {resumed} resumed")

    # Launch shard workers.  Each global round is a durable barrier: all
    # shards finish the current method/scope before any shard advances to the
    # next fallback.  Queue CSVs remain the restart checkpoints between rounds.
    env = env or os.environ.copy()
    env["R_LIBS_USER"] = str(R_LIB)
    if VIIRS_RAW:
        env["MIT_VIIRS_RAW"] = VIIRS_RAW

    start = time.time()
    round_specs = phase2_round_specs(queue_phase, task_scheduling, dry_run)
    print(f"  Task scheduling: {task_scheduling}; worker rounds: {len(round_specs)}")
    for round_number, (round_label, runner, round_args) in enumerate(round_specs, start=1):
        print(f"\n  ROUND {round_number}/{len(round_specs)}: {round_label}")
        round_start = time.time()
        processes: list[tuple[int, subprocess.Popen[str]]] = []
        for shard_id in range(shard_count):
            cmd = [
                sys.executable,
                str(runner),
                "--shard-id",
                str(shard_id),
                "--shard-count",
                str(shard_count),
                "--max-tasks",
                "9999",
                "--price-measure",
                price_measure,
                "--window",
                str(window),
                "--anticipation-months",
                str(anticipation_months),
                "--run-mode",
                run_mode,
                "--estimator-backend",
                estimator_backend,
                "--max-gsc-cross-city-donors",
                str(max_gsc_cross_city_donors),
                "--gsc-donor-sampling-seed",
                str(gsc_donor_sampling_seed),
                "--transaction-count-threshold",
                str(transaction_count_threshold),
                *round_args,
            ]
            if orders_file is not None:
                cmd.extend(["--orders-file", str(orders_file)])
            if tasks_file is not None:
                cmd.extend(["--tasks-file", str(tasks_file)])
            if retry_matching:
                cmd.append("--retry-matching")
            if qualification_receipt is not None:
                cmd.extend(["--qualification-receipt", str(qualification_receipt)])
            if dry_run:
                cmd.append("--dry-run")

            shard_env = env.copy()
            if estimator_backend == "python_gpu" and gpu_ids:
                gpu_id = gpu_ids[shard_id % len(gpu_ids)]
                shard_env["MIT_CAUSAL_DEVICE"] = f"cuda:{gpu_id}"
                device_note = f", cuda:{gpu_id}"
            else:
                device_note = ""

            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=shard_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            processes.append((shard_id, proc))
            print(f"    Launched shard {shard_id + 1}/{shard_count} (PID {proc.pid}{device_note})")

        print(f"    Waiting for {shard_count} shards...")
        failures = []
        for shard_id, proc in processes:
            stdout, _ = proc.communicate()
            if proc.returncode != 0:
                failures.append(shard_id)
                print(f"    SHARD {shard_id + 1} FAILED (rc={proc.returncode})")
                print(f"    {stdout[-1000:] if stdout else '(no output)'}")
            else:
                print(f"    Shard {shard_id + 1}/{shard_count} complete")
                for line in stdout.splitlines()[-5:] if stdout else []:
                    print(f"      {line.strip()}")
        if failures:
            raise RuntimeError(
                f"Phase 2 round {round_label!r}: "
                f"{len(failures)}/{shard_count} shards failed: {failures}"
            )
        round_elapsed = time.time() - round_start
        print(
            f"  Round {round_number}/{len(round_specs)} completed in "
            f"{timedelta(seconds=int(round_elapsed))}"
        )

    elapsed = time.time() - start
    print(f"Phase 2 completed in {timedelta(seconds=int(elapsed))}")


def phase2_round_pool(
    *,
    shard_count: int,
    tasks_file: Path | None,
    orders_file: Path | None,
    run_mode: str,
    price_measure: str,
    window: int,
    anticipation_months: int,
    estimator_backend: str,
    gpu_ids: list[str],
    env: dict[str, str] | None,
    qualification_receipt: Path | None,
    max_gsc_cross_city_donors: int,
    gsc_donor_sampling_seed: int,
    transaction_count_threshold: int,
    master_paths: tuple[Path, Path, Path],
    round_pool_db: Path | None,
    round_pool_run_root: Path | None,
    round_pool_run_id: str | None,
    round_pool_queue_namespace: str,
) -> None:
    """Run the SQLite-backed dynamic pool with strict Round barriers."""

    family_queue, control_queue, unit_queue = master_paths
    db_path = round_pool_db or (ROOT / "outputs" / "causal_labels" / "round_pool.sqlite")
    run_root = round_pool_run_root or (ROOT / "outputs" / "causal_labels" / "round_pool_runs")
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    command = [
        sys.executable,
        str(ROOT / "scripts" / "causal_python" / "run_round_pool.py"),
        "--db",
        str(db_path.resolve()),
        "--run-root",
        str(run_root.resolve()),
        "--family-queue",
        str(family_queue.resolve()),
        "--control-queue",
        str(control_queue.resolve()),
        "--unit-queue",
        str(unit_queue.resolve()),
        "--worker-count",
        str(shard_count),
        "--gpu-ids",
        ",".join(gpu_ids),
        "--run-mode",
        run_mode,
        "--queue-namespace",
        round_pool_queue_namespace,
        "--estimator-backend",
        estimator_backend,
        "--price-measure",
        price_measure,
        "--window",
        str(window),
        "--anticipation-months",
        str(anticipation_months),
        "--transaction-count-threshold",
        str(transaction_count_threshold),
        "--max-gsc-cross-city-donors",
        str(max_gsc_cross_city_donors),
        "--gsc-donor-sampling-seed",
        str(gsc_donor_sampling_seed),
    ]
    if tasks_file is not None:
        command.extend(["--tasks-file", str(tasks_file.resolve())])
    if orders_file is not None:
        command.extend(["--orders-file", str(orders_file.resolve())])
    if qualification_receipt is not None:
        command.extend(["--qualification-receipt", str(qualification_receipt.resolve())])
    if round_pool_run_id is not None:
        command.extend(["--run-id", round_pool_run_id])
    child_env = (env or os.environ.copy()).copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["MIT_PROJECT_ROOT"] = str(ROOT)
    if VIIRS_RAW:
        child_env["MIT_VIIRS_RAW"] = VIIRS_RAW
    print(f"  Dynamic Round Pool database: {db_path.resolve()}")
    result = subprocess.run(command, cwd=str(ROOT), env=child_env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Dynamic Round Pool failed with code {result.returncode}")


def shard_order_range(all_orders: list[int], shard_id: int, shard_count: int) -> list[int]:
    """Orders owned by one shard, using the same split as run_causal_label_queue.py."""
    chunk_size = len(all_orders) // shard_count
    remainder = len(all_orders) % shard_count
    start = shard_id * chunk_size + min(shard_id, remainder)
    size = chunk_size + (1 if shard_id < remainder else 0)
    return all_orders[start : start + size]


def merge_queue_parts(
    master_path: Path,
    parts: list[pd.DataFrame],
    key_columns: list[str],
    selected_orders: set[int] | None,
    selected_task_keys: set[tuple[int, str]] | None = None,
) -> pd.DataFrame:
    """Merge shard updates, preserving non-sample rows in the master queue."""
    if not parts:
        raise RuntimeError(f"No shard parts found for {master_path}")
    merged = pd.concat(parts, ignore_index=True)
    if merged.duplicated(key_columns).any():
        raise RuntimeError(f"Merged shard queue contains duplicate keys: {master_path}")
    master = pd.read_csv(master_path)
    if selected_orders is None:
        expected_keys = set(map(tuple, master[key_columns].to_numpy()))
        actual_keys = set(map(tuple, merged[key_columns].to_numpy()))
        if actual_keys != expected_keys:
            raise RuntimeError(f"Merged shard keys differ from master queue: {master_path}")
        return merged.sort_values(key_columns).reset_index(drop=True)

    if selected_task_keys is not None:
        master_keys = pd.MultiIndex.from_arrays(
            [master["treatment_order"].astype(int), master["outcome_family"].astype(str)]
        )
        selected_mask = master_keys.isin(selected_task_keys)
    else:
        selected_mask = master["treatment_order"].astype(int).isin(selected_orders)
    expected_keys = set(map(tuple, master.loc[selected_mask, key_columns].to_numpy()))
    actual_keys = set(map(tuple, merged[key_columns].to_numpy()))
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"Selected shard keys differ from master queue: {master_path}; "
            f"expected={len(expected_keys)}, actual={len(actual_keys)}"
        )
    combined = pd.concat([master.loc[~selected_mask], merged], ignore_index=True)
    if len(combined) != len(master) or combined.duplicated(key_columns).any():
        raise RuntimeError(f"Selected shard merge changed queue cardinality: {master_path}")
    return combined.sort_values(key_columns).reset_index(drop=True)


def phase3_merge_shards(
    shard_count: int = DEFAULT_SHARD_COUNT,
    orders_file: Path | None = None,
    tasks_file: Path | None = None,
    run_mode: str = "production",
    queue_paths: tuple[Path, Path, Path] | None = None,
    dry_run: bool = False,
) -> None:
    """Merge per-shard queue CSVs back to master and run sync_unit_queue."""
    print("=" * 60)
    print(f"PHASE 3: Merge {shard_count} shard queues → master")
    print("=" * 60)

    family_base, control_base, unit_base = (
        queue_paths
        if queue_paths is not None
        else (OUTCOME_FAMILY_QUEUE, CONTROL_DESIGN_QUEUE, COUNTERFACTUAL_QUEUE)
    )
    master_family = queue_variant_path(family_base, run_mode)
    master_unit = queue_variant_path(unit_base, run_mode)
    master_control = queue_variant_path(control_base, run_mode)

    # Each shard file is a full copy of the master queue at launch; only the
    # rows inside the shard's order range carry that shard's progress, so trim
    # every shard file to its own range before concatenating.  Otherwise every
    # (treatment_order, outcome_family) key would be duplicated shard_count
    # times and the release build would fail.
    treatments = pd.read_parquet(TREATMENT_UNIT_LIST, columns=["treatment_order"])
    all_orders = sorted(treatments["treatment_order"].astype(int).tolist())
    task_keys = read_tasks_file(tasks_file) if tasks_file is not None else None
    selected_orders = (
        {order for order, _ in task_keys}
        if task_keys is not None
        else (set(read_orders_file(orders_file)) if orders_file is not None else None)
    )
    # Use the same selected-order pool as the workers when merging a sample
    # run; non-sample rows remain unchanged in their master queues.
    shard_pool = sorted(selected_orders) if selected_orders is not None else all_orders

    family_parts = []
    unit_parts = []
    control_parts = []
    missing_family_shards = []

    for shard_id in range(shard_count):
        shard_family = shard_queue_path(master_family, shard_id)
        shard_unit = shard_queue_path(master_unit, shard_id)
        shard_control = shard_queue_path(master_control, shard_id)
        shard_orders = set(shard_order_range(shard_pool, shard_id, shard_count))

        if not shard_family.exists():
            missing_family_shards.append(shard_id)
            continue

        fq = pd.read_csv(shard_family)
        fq = fq.loc[fq["treatment_order"].astype(int).isin(shard_orders)]
        if task_keys is not None:
            fq_keys = pd.MultiIndex.from_arrays(
                [fq["treatment_order"].astype(int), fq["outcome_family"].astype(str)]
            )
            fq = fq.loc[fq_keys.isin(task_keys)]
        family_parts.append(fq)

        if not shard_unit.exists():
            raise RuntimeError(f"Missing unit-queue shard file for shard {shard_id}: {shard_unit}")
        uq = pd.read_csv(shard_unit)
        uq = uq.loc[uq["treatment_order"].astype(int).isin(shard_orders)]
        unit_parts.append(uq)

        if not shard_control.exists():
            raise RuntimeError(
                f"Missing control-queue shard file for shard {shard_id}: {shard_control}"
            )
        cq = pd.read_csv(shard_control)
        cq = cq.loc[cq["treatment_order"].astype(int).isin(shard_orders)]
        control_parts.append(cq)

    if missing_family_shards:
        raise RuntimeError(
            f"Missing family-queue shard files for shard(s): {missing_family_shards}"
        )
    if not family_parts:
        raise RuntimeError("No shard family queues found to merge")

    merged_family = merge_queue_parts(
        master_family,
        family_parts,
        ["treatment_order", "outcome_family"],
        selected_orders,
        selected_task_keys=task_keys,
    )

    terminal = {"matched_labelled", "gsc_labelled", "mc_labelled", "skipped"}
    if selected_orders is None:
        report_family = merged_family
    elif task_keys is not None:
        report_keys = pd.MultiIndex.from_arrays(
            [
                merged_family["treatment_order"].astype(int),
                merged_family["outcome_family"].astype(str),
            ]
        )
        report_family = merged_family.loc[report_keys.isin(task_keys)]
    else:
        report_family = merged_family.loc[
            merged_family["treatment_order"].astype(int).isin(selected_orders)
        ]
    n_terminal = report_family["status"].astype(str).isin(terminal).sum()
    n_total = len(report_family)
    print(f"  Terminal tasks: {n_terminal} / {n_total} ({100 * n_terminal / n_total:.1f}%)")

    if dry_run:
        print(
            "  DRY RUN: would merge shard queues into "
            f"{master_family}, {master_unit}, {master_control}; no files were written "
            "and no shard queues were deleted."
        )
        return

    # Write merged master queues atomically so an interrupted merge never
    # leaves a truncated master file behind.
    atomic_write_csv(merged_family, master_family)
    print(f"  Written: {master_family}")

    if unit_parts:
        merged_unit = merge_queue_parts(
            master_unit, unit_parts, ["treatment_order"], selected_orders
        )
        atomic_write_csv(merged_unit, master_unit)
        print(f"  Written: {master_unit}")

    if control_parts:
        merged_control = merge_queue_parts(
            master_control, control_parts, ["treatment_order"], selected_orders
        )
        atomic_write_csv(merged_control, master_control)
        print(f"  Written: {master_control}")

    # Optionally clean up shard files
    for shard_id in range(shard_count):
        for base in [master_family, master_control, master_unit]:
            path = shard_queue_path(base, shard_id)
            if path.exists():
                path.unlink()
    print(f"  Cleaned up {shard_count} shard queue files")

    print("Phase 3 complete.  Master queues restored.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3], help="Run a single phase, or omit for all three"
    )
    parser.add_argument("--run-all", action="store_true", help="Run all three phases sequentially")
    parser.add_argument(
        "--shard-count",
        type=int,
        default=DEFAULT_SHARD_COUNT,
        help=f"Shards for phase 2 (default: {DEFAULT_SHARD_COUNT})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_CONTROL_WORKERS,
        help=f"Control-design workers (default: {DEFAULT_CONTROL_WORKERS})",
    )
    parser.add_argument(
        "--estimator-backend",
        choices=("python_gpu", "r_reference"),
        default="python_gpu",
        help="Python/GPU production backend or explicit R reference fallback.",
    )
    parser.add_argument(
        "--qualification-receipt",
        type=Path,
        default=(
            Path(os.environ["MIT_CAUSAL_QUALIFICATION_RECEIPT"])
            if os.environ.get("MIT_CAUSAL_QUALIFICATION_RECEIPT")
            else None
        ),
        help="Eligible R/Python parity audit receipt required for production Python labels.",
    )
    parser.add_argument(
        "--resume-prevalidated-qualification",
        action="store_true",
        help=(
            "Resume existing global-round shard checkpoints after this exact receipt "
            "was fully audited at the original launch. This skips only the mutable "
            "bound-source re-audit; receipt digest, code, counts, and environment "
            "remain validated."
        ),
    )
    parser.add_argument(
        "--max-gsc-cross-city-donors",
        type=int,
        default=5_000,
        help="Pre-outcome deterministic cap for cross-city Matching, GSC, and MC donors.",
    )
    parser.add_argument(
        "--gsc-donor-sampling-seed",
        type=int,
        default=20260823,
        help="Fixed donor-sampling seed recorded in the formal specification.",
    )
    parser.add_argument(
        "--transaction-count-threshold",
        type=int,
        default=1,
        help="Minimum transactions per housing grid-month; use sensitivity runs before changing it.",
    )
    parser.add_argument(
        "--gpu-ids",
        default=DEFAULT_GPU_IDS,
        help=f"Comma-separated CUDA device indices (default: {DEFAULT_GPU_IDS}).",
    )
    parser.add_argument(
        "--orders-file",
        type=Path,
        help="CSV containing the treatment_order sample to process",
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        help="CSV containing treatment_order and outcome_family keys for a formal rerun",
    )
    parser.add_argument(
        "--run-mode",
        choices=("production", "preview"),
        default="production",
        help="Preview uses isolated point-estimate artifacts; production uses formal uncertainty.",
    )
    parser.add_argument(
        "--price-measure",
        choices=("main", "median", "hedonic"),
        default="main",
        help="Housing price measure (default: main)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Monthly observation-window width (default: 1)",
    )
    parser.add_argument(
        "--anticipation-months",
        type=int,
        default=0,
        help=(
            "Monthly anticipation window (default: 0, giving the main t=-1 "
            "reference); use 6 for the monthly t=-7 robustness specification"
        ),
    )
    parser.add_argument(
        "--estimator-output-root",
        type=Path,
        default=OUTPUT_PYTHON_ESTIMATORS_DIR,
        help=(
            "Isolated Python estimator artifact root. Use a different root for "
            "the monthly t=-7 robustness run so it cannot overwrite main results."
        ),
    )
    parser.add_argument(
        "--queue-namespace",
        help=(
            "Isolate Phase 2/3 queues for a separate specification (for example "
            "robust_m7). The isolated family/unit queues start pending while the "
            "completed Phase 1 control design is copied read-only."
        ),
    )
    parser.add_argument(
        "--task-output-root",
        type=Path,
        help=(
            "Isolated task artifact root. Required with --queue-namespace so a "
            "robustness run cannot overwrite main causal-label task manifests."
        ),
    )
    parser.add_argument(
        "--queue-phase",
        choices=("all", "matching", "gsc", "mc"),
        default="all",
        help="Phase passed to each causal-label shard (default: all)",
    )
    parser.add_argument(
        "--task-scheduling",
        choices=("global_rounds", "task_cascade", "round_pool"),
        default="global_rounds",
        help=(
            "Run all selected tasks through six global method/scope rounds "
            "(default), use the legacy one-task-at-a-time cascade, or use "
            "a SQLite-backed dynamically balanced pool with the same six barriers."
        ),
    )
    parser.add_argument(
        "--round-pool-db",
        type=Path,
        help="SQLite state database for --task-scheduling round_pool; put this on local SSD when available.",
    )
    parser.add_argument(
        "--round-pool-run-root",
        type=Path,
        help="Directory for private worker queue snapshots created by round_pool.",
    )
    parser.add_argument(
        "--round-pool-run-id",
        help="Resume an existing Round Pool run after verifying its identity.",
    )
    parser.add_argument(
        "--retry-matching",
        action="store_true",
        help="Re-run matching for tasks currently marked gsc_pending",
    )
    parser.add_argument(
        "--repair-stale-controls",
        action="store_true",
        help="Recompute only control records rejected by current schema/backend provenance checks.",
    )
    parser.add_argument(
        "--gsc-cv-cores",
        type=int,
        default=1,
        help="R-reference cores per GSC factor-selection fit (default: 1)",
    )
    parser.add_argument(
        "--gsc-bootstrap-cores",
        type=int,
        default=1,
        help="R-reference cores per GSC bootstrap fit (default: 1)",
    )
    parser.add_argument(
        "--mc-cores",
        type=int,
        default=1,
        help="R-reference cores per MC fit (default: 1)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reset-queues",
        action="store_true",
        help="Rebuild formal inputs/support and reset queues with the selected backend.",
    )
    args = parser.parse_args()

    if args.queue_namespace is not None and args.task_output_root is None:
        parser.error("--queue-namespace requires --task-output-root")
    if args.queue_namespace is not None and (args.run_all or args.phase == 1):
        parser.error("--queue-namespace is only supported for isolated Phase 2/3 runs")
    if args.task_scheduling == "round_pool" and (args.phase != 2 or args.run_all):
        parser.error("--task-scheduling round_pool is currently only supported with --phase 2")
    if args.task_scheduling == "round_pool" and args.queue_phase != "all":
        parser.error("--task-scheduling round_pool requires --queue-phase all")
    if args.task_scheduling == "round_pool" and args.dry_run:
        parser.error("--task-scheduling round_pool cannot be combined with --dry-run")
    try:
        queue_paths = isolated_queue_paths(args.queue_namespace)
    except ValueError as exc:
        parser.error(str(exc))

    if args.shard_count < 1 or args.workers < 1:
        parser.error("--shard-count and --workers must be positive")
    if min(args.gsc_cv_cores, args.gsc_bootstrap_cores, args.mc_cores) < 1:
        parser.error("estimator core counts must be positive")
    if not 1 <= args.window <= 6:
        parser.error("--window must be in 1..6")
    if args.anticipation_months < 0:
        parser.error("--anticipation-months must be non-negative")
    if args.max_gsc_cross_city_donors < 20:
        parser.error("--max-gsc-cross-city-donors must be at least 20")
    if args.transaction_count_threshold < 1:
        parser.error("--transaction-count-threshold must be positive")
    try:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
    except ValueError as exc:
        parser.error(str(exc))
    if args.estimator_backend == "python_gpu" and args.shard_count > len(gpu_ids):
        parser.error(
            "Python/GPU production allows at most one shard per GPU; "
            "reduce --shard-count or add --gpu-ids."
        )
    qualification_receipt = args.qualification_receipt
    if qualification_receipt is not None:
        qualification_receipt = qualification_receipt.resolve()
    qualification_proof: dict[str, object] = {}
    if args.resume_prevalidated_qualification:
        if args.reset_queues or args.task_scheduling != "global_rounds":
            parser.error(
                "--resume-prevalidated-qualification requires existing global-round "
                "shards and cannot be combined with --reset-queues"
            )
        try:
            validate_resume_checkpoint(
                queue_variant_path(queue_paths[0], args.run_mode),
                args.shard_count,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    if (
        args.estimator_backend == "python_gpu"
        and args.run_mode == "production"
        and not args.dry_run
        and (args.run_all or args.phase == 2)
    ):
        if qualification_receipt is None:
            parser.error("production Python labels require --qualification-receipt")
        qualification_proof = validate_formal_qualification_receipt(
            qualification_receipt,
            verify_bound_sources=not args.resume_prevalidated_qualification,
        )
    elif args.resume_prevalidated_qualification:
        parser.error(
            "--resume-prevalidated-qualification is only valid for a non-dry-run "
            "production Python Phase 2 run"
        )
    if args.run_mode == "production" and not args.dry_run:
        try:
            validate_frozen_formal_matching_spec(ROOT)
        except ValueError as exc:
            parser.error(str(exc))

    if args.orders_file is not None and args.tasks_file is not None:
        parser.error("--orders-file and --tasks-file are mutually exclusive")

    orders_file = None
    if args.orders_file is not None:
        orders_file = (
            args.orders_file if args.orders_file.is_absolute() else ROOT / args.orders_file
        )
        orders_file = orders_file.resolve()
        orders = read_orders_file(orders_file)
        if len(orders) != 300:
            parser.error(
                f"--orders-file must contain exactly 300 unique orders; found {len(orders)}"
            )
        print(f"Selected treatment-order sample: {len(orders)} grids from {orders_file}")

    tasks_file = None
    if args.tasks_file is not None:
        tasks_file = args.tasks_file if args.tasks_file.is_absolute() else ROOT / args.tasks_file
        tasks_file = tasks_file.resolve()
        task_keys = read_tasks_file(tasks_file)
        print(f"Selected task subset: {len(task_keys)} treatment-family tasks from {tasks_file}")

    estimator_env = resource_environment(
        os.environ,
        args.gsc_cv_cores,
        args.gsc_bootstrap_cores,
        args.mc_cores,
    )
    if qualification_proof:
        estimator_env["MIT_CAUSAL_QUALIFICATION_PREVALIDATED_SHA256"] = str(
            qualification_proof["formal_qualification_receipt_sha256"]
        )
    if args.estimator_backend == "python_gpu":
        estimator_output_root = args.estimator_output_root
        if not estimator_output_root.is_absolute():
            estimator_output_root = ROOT / estimator_output_root
        estimator_env["MIT_CAUSAL_ESTIMATOR_OUTPUT_ROOT"] = str(
            estimator_output_root.resolve()
        )
    if args.queue_namespace is not None:
        task_output_root = args.task_output_root
        assert task_output_root is not None
        if not task_output_root.is_absolute():
            task_output_root = ROOT / task_output_root
        estimator_env["MIT_CAUSAL_FAMILY_QUEUE"] = str(queue_paths[0].resolve())
        estimator_env["MIT_CAUSAL_CONTROL_QUEUE"] = str(queue_paths[1].resolve())
        estimator_env["MIT_CAUSAL_UNIT_QUEUE"] = str(queue_paths[2].resolve())
        estimator_env["MIT_CAUSAL_TASK_ROOT"] = str(task_output_root.resolve())

    if not args.phase and not args.run_all:
        parser.error("Specify --phase, --run-all, or both phases separately")

    if args.reset_queues:
        if args.estimator_backend == "python_gpu":
            setup_commands = [
                [
                    sys.executable,
                    str(ROOT / "scripts" / "causal_python" / "prepare_causal_inputs.py"),
                    "--all",
                ]
            ]
        else:
            setup_commands = [
                [R_SCRIPT, str(r_script("reset_counterfactual_queues.R"))],
                [R_SCRIPT, str(r_script("build_formal_matching_inputs.R"))],
                [R_SCRIPT, str(r_script("audit_formal_target_support.R"))],
            ]
        for command in setup_commands:
            print(f"  Running {' '.join(command)} ...")
            result = subprocess.run(
                command,
                cwd=str(ROOT),
                env=estimator_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            print(result.stdout[-500:] if result.stdout else "(no output)")
            if result.returncode != 0:
                raise RuntimeError(f"Causal input setup failed: {' '.join(command)}")

    phases = [args.phase] if args.phase else [1, 2, 3]
    start = time.time()

    for phase in phases:
        if phase == 1:
            phase1_control_design(
                args.dry_run,
                workers=args.workers,
                orders_file=orders_file,
                estimator_backend=args.estimator_backend,
                gpu_ids=gpu_ids,
                env=estimator_env,
                force_recompute=args.reset_queues,
                repair_stale_controls=args.repair_stale_controls,
            )
        elif phase == 2:
            phase2_causal_labels(
                args.dry_run,
                shard_count=args.shard_count,
                orders_file=orders_file,
                tasks_file=tasks_file,
                run_mode=args.run_mode,
                price_measure=args.price_measure,
                window=args.window,
                anticipation_months=args.anticipation_months,
                queue_phase=args.queue_phase,
                retry_matching=args.retry_matching,
                reset_shards=args.reset_queues,
                estimator_backend=args.estimator_backend,
                gpu_ids=gpu_ids,
                env=estimator_env,
                qualification_receipt=qualification_receipt,
                max_gsc_cross_city_donors=args.max_gsc_cross_city_donors,
                gsc_donor_sampling_seed=args.gsc_donor_sampling_seed,
                transaction_count_threshold=args.transaction_count_threshold,
                task_scheduling=args.task_scheduling,
                queue_paths=queue_paths,
                round_pool_db=args.round_pool_db,
                round_pool_run_root=args.round_pool_run_root,
                round_pool_run_id=args.round_pool_run_id,
                queue_namespace=args.queue_namespace,
            )
        elif phase == 3:
            phase3_merge_shards(
                shard_count=args.shard_count,
                orders_file=orders_file,
                tasks_file=tasks_file,
                run_mode=args.run_mode,
                queue_paths=queue_paths,
                dry_run=args.dry_run,
            )

    elapsed = time.time() - start
    print(f"\nTotal wall time: {timedelta(seconds=int(elapsed))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
