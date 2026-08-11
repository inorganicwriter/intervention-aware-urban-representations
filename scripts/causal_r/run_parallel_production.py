"""Parallel production launcher for the complete 5,048-grid pipeline.

Phase 1: Control design (parallel via R processes, --workers 48)
Phase 2: Causal labels (sharded, 12 parallel orchestrator instances)
Phase 3: Merge shard queues back to master queue

Usage:
  For a 48-core server, start all 12 shards in parallel:

      python scripts/causal_r/run_parallel_production.py --run-all

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
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.data.paths import (  # noqa: E402
    CAUSAL_DIR,
    CONTROL_DESIGN_QUEUE,
    COUNTERFACTUAL_QUEUE,
    OUTCOME_FAMILY_QUEUE,
    R_LIB_DIR,
    TREATMENT_UNIT_LIST,
    r_script,
)

R_SCRIPT = os.environ.get("MIT_RSCRIPT", "Rscript")
R_LIB = Path(os.environ.get("MIT_R_LIB", str(R_LIB_DIR)))
VIIRS_RAW = os.environ.get("MIT_VIIRS_RAW")


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV via temp file + os.replace so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as fh:
            frame.to_csv(fh, index=False)
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

# Default: 12 shards on a 48-core machine (4 cores idle for OS + I/O)
DEFAULT_SHARD_COUNT = 12


def phase1_control_design(dry_run: bool, workers: int = 48) -> None:
    """Run control design for all 5,048 grids in parallel (R processes)."""
    print("=" * 60)
    print(f"PHASE 1: Grid-level control design (workers={workers})")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(r_script("run_grid_control_design_queue.py")),
        "--start-order",
        "1",
        "--max-units",
        "5048",
        "--workers",
        str(workers),
    ]
    if dry_run:
        cmd.append("--dry-run")

    env = os.environ.copy()
    env["R_LIBS_USER"] = str(R_LIB)
    start = time.time()
    result = subprocess.run(
        cmd, cwd=str(ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    elapsed = time.time() - start
    print(result.stdout[-2000:] if result.stdout else "(no output)")
    if result.returncode != 0:
        raise RuntimeError(f"Phase 1 failed with code {result.returncode}")
    print(f"Phase 1 completed in {timedelta(seconds=int(elapsed))}")


def phase2_causal_labels(dry_run: bool, shard_count: int = DEFAULT_SHARD_COUNT) -> None:
    """Launch N parallel causal-label orchestrators, each handling a shard."""
    print("=" * 60)
    print(f"PHASE 2: Causal labels ({shard_count} shards)")
    print("=" * 60)

    # Validate control design is complete
    cq = pd.read_csv(CONTROL_DESIGN_QUEUE)
    statuses = cq["status"].astype(str)
    terminal_control = statuses.isin({"matched", "gsc_pending"})
    n_valid = terminal_control.sum()
    n_total = len(cq)
    if n_valid < n_total:
        raise RuntimeError(
            f"Phase 1 incomplete: only {n_valid}/{n_total} control designs "
            f"in terminal state (matched or gsc_pending). "
            f"Run Phase 1 first with: --phase 1"
        )
    n_matched = (statuses == "matched").sum()
    print(f"  Control designs: {n_matched} matched, {n_total - n_matched} gsc_pending, 0 error")

    # Copy master queues for each shard.  A stale shard file (left over from a
    # previous run whose master queue has since been reset/rebuilt) must never
    # be reused: compare content hashes so --reset-queues + rerun of phase 2
    # cannot resurrect old shard progress at merge time.
    master_family = OUTCOME_FAMILY_QUEUE
    master_control = CONTROL_DESIGN_QUEUE
    master_unit = COUNTERFACTUAL_QUEUE
    for shard_id in range(shard_count):
        tag = f"_shard_{shard_id:02d}"
        for master_path in [master_family, master_control, master_unit]:
            shard_path = CAUSAL_DIR / master_path.name.replace(".csv", f"{tag}.csv")
            if shard_path.exists() and _file_sha256(shard_path) == _file_sha256(master_path):
                continue
            shutil.copy2(master_path, shard_path)
    print(f"Copied master queues → {shard_count} shard copies")

    # Launch shard workers
    env = os.environ.copy()
    env["R_LIBS_USER"] = str(R_LIB)
    if VIIRS_RAW:
        env["MIT_VIIRS_RAW"] = VIIRS_RAW

    processes: list[tuple[int, subprocess.Popen]] = []
    start = time.time()

    for shard_id in range(shard_count):
        cmd = [
            sys.executable,
            str(r_script("run_causal_label_queue.py")),
            "--shard-id",
            str(shard_id),
            "--shard-count",
            str(shard_count),
            "--max-tasks",
            "9999",
        ]
        if dry_run:
            cmd.append("--dry-run")

        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        processes.append((shard_id, proc))
        print(f"  Launched shard {shard_id + 1}/{shard_count} (PID {proc.pid})")

    # Wait for all shards
    print(f"\n  Waiting for {shard_count} shards to complete...")
    failures = []
    for shard_id, proc in processes:
        stdout, _ = proc.communicate()
        if proc.returncode != 0:
            failures.append(shard_id)
            print(f"  SHARD {shard_id + 1} FAILED (rc={proc.returncode})")
            print(f"  {stdout[-1000:] if stdout else '(no output)'}")
        else:
            print(f"  Shard {shard_id + 1}/{shard_count} complete")
            # Print summary from stdout's last few lines
            for line in stdout.splitlines()[-5:] if stdout else []:
                print(f"    {line.strip()}")

    elapsed = time.time() - start
    if failures:
        raise RuntimeError(f"{len(failures)}/{shard_count} shards failed: {failures}")
    print(f"Phase 2 completed in {timedelta(seconds=int(elapsed))}")


def shard_order_range(all_orders: list[int], shard_id: int, shard_count: int) -> list[int]:
    """Orders owned by one shard, using the same split as run_causal_label_queue.py."""
    chunk_size = len(all_orders) // shard_count
    remainder = len(all_orders) % shard_count
    start = shard_id * chunk_size + min(shard_id, remainder)
    size = chunk_size + (1 if shard_id < remainder else 0)
    return all_orders[start : start + size]


def phase3_merge_shards(shard_count: int = DEFAULT_SHARD_COUNT) -> None:
    """Merge per-shard queue CSVs back to master and run sync_unit_queue."""
    print("=" * 60)
    print(f"PHASE 3: Merge {shard_count} shard queues → master")
    print("=" * 60)

    master_family = OUTCOME_FAMILY_QUEUE
    master_unit = COUNTERFACTUAL_QUEUE

    # Each shard file is a full copy of the master queue at launch; only the
    # rows inside the shard's order range carry that shard's progress, so trim
    # every shard file to its own range before concatenating.  Otherwise every
    # (treatment_order, outcome_family) key would be duplicated shard_count
    # times and the release build would fail.
    treatments = pd.read_parquet(TREATMENT_UNIT_LIST, columns=["treatment_order"])
    all_orders = sorted(treatments["treatment_order"].astype(int).tolist())

    family_parts = []
    unit_parts = []
    missing_family_shards = []

    for shard_id in range(shard_count):
        tag = f"_shard_{shard_id:02d}"
        shard_family = CAUSAL_DIR / f"outcome_family_work_queue{tag}.csv"
        shard_unit = CAUSAL_DIR / f"counterfactual_work_queue{tag}.csv"
        shard_orders = set(shard_order_range(all_orders, shard_id, shard_count))

        if not shard_family.exists():
            missing_family_shards.append(shard_id)
            continue

        fq = pd.read_csv(shard_family)
        fq = fq.loc[fq["treatment_order"].astype(int).isin(shard_orders)]
        family_parts.append(fq)

        if not shard_unit.exists():
            raise RuntimeError(
                f"Missing unit-queue shard file for shard {shard_id}: {shard_unit}"
            )
        uq = pd.read_csv(shard_unit)
        uq = uq.loc[uq["treatment_order"].astype(int).isin(shard_orders)]
        unit_parts.append(uq)

    if missing_family_shards:
        raise RuntimeError(
            f"Missing family-queue shard files for shard(s): {missing_family_shards}"
        )
    if not family_parts:
        raise RuntimeError("No shard family queues found to merge")

    merged_family = pd.concat(family_parts, ignore_index=True)
    if merged_family.duplicated(["treatment_order", "outcome_family"]).any():
        raise RuntimeError(
            "Merged family queue contains duplicate (treatment_order, outcome_family) keys"
        )
    expected_family_rows = 5048 * len(set(merged_family["outcome_family"]))
    if len(merged_family) != expected_family_rows:
        raise RuntimeError(
            f"Merged family queue has {len(merged_family)} rows, expected "
            f"{expected_family_rows}; a shard queue is truncated, refusing to "
            "overwrite the master queue"
        )
    merged_family = merged_family.sort_values(["treatment_order", "outcome_family"]).reset_index(
        drop=True
    )

    terminal = {"matched_labelled", "gsc_labelled", "mc_labelled", "skipped"}
    n_terminal = merged_family["status"].astype(str).isin(terminal).sum()
    n_total = len(merged_family)
    print(f"  Terminal tasks: {n_terminal} / {n_total} ({100 * n_terminal / n_total:.1f}%)")

    # Write merged master queues atomically so an interrupted merge never
    # leaves a truncated master file behind.
    atomic_write_csv(merged_family, master_family)
    print(f"  Written: {master_family}")

    if unit_parts:
        merged_unit = pd.concat(unit_parts, ignore_index=True)
        if merged_unit["treatment_order"].duplicated().any():
            raise RuntimeError("Merged unit queue contains duplicate treatment_order keys")
        if len(merged_unit) != 5048:
            raise RuntimeError(
                f"Merged unit queue has {len(merged_unit)} rows, expected 5048; "
                "refusing to overwrite the master queue"
            )
        merged_unit = merged_unit.sort_values("treatment_order").reset_index(drop=True)
        atomic_write_csv(merged_unit, master_unit)
        print(f"  Written: {master_unit}")

    # Optionally clean up shard files
    for shard_id in range(shard_count):
        tag = f"_shard_{shard_id:02d}"
        for base in [
            "outcome_family_work_queue",
            "control_design_queue",
            "counterfactual_work_queue",
        ]:
            path = CAUSAL_DIR / f"{base}{tag}.csv"
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
        "--workers", type=int, default=48, help="Control-design R workers (default: 48)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reset-queues", action="store_true", help="Re-run R queue reset scripts before starting"
    )
    args = parser.parse_args()

    if not args.phase and not args.run_all:
        parser.error("Specify --phase, --run-all, or both phases separately")

    if args.reset_queues:
        env = os.environ.copy()
        env["R_LIBS_USER"] = str(R_LIB)
        for script in [
            r_script("reset_counterfactual_queues.R"),
            r_script("build_formal_matching_inputs.R"),
            r_script("audit_formal_target_support.R"),
        ]:
            print(f"  Running {script} ...")
            result = subprocess.run(
                [R_SCRIPT, str(script)],
                cwd=str(ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            print(result.stdout[-500:] if result.stdout else "(no output)")
            if result.returncode != 0:
                raise RuntimeError(f"Reset script failed: {script}")

    phases = [args.phase] if args.phase else [1, 2, 3]
    start = time.time()

    for phase in phases:
        if phase == 1:
            phase1_control_design(args.dry_run, workers=args.workers)
        elif phase == 2:
            phase2_causal_labels(args.dry_run, shard_count=args.shard_count)
        elif phase == 3:
            phase3_merge_shards(shard_count=args.shard_count)

    elapsed = time.time() - start
    print(f"\nTotal wall time: {timedelta(seconds=int(elapsed))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
