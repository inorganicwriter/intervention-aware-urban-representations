"""Coordinate a strict-barrier, dynamically balanced causal-label run.

The coordinator is the only process that publishes queue snapshots. Workers
claim individual tasks from SQLite and write only private queue copies.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ROUND_NAMES = (
    "same_city_matching",
    "same_city_gsc",
    "same_city_mc",
    "cross_city_matching",
    "cross_city_gsc",
    "cross_city_mc",
)
TERMINAL = {"matched_labelled", "gsc_labelled", "mc_labelled", "skipped"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--family-queue", type=Path, required=True)
    parser.add_argument("--control-queue", type=Path, required=True)
    parser.add_argument("--unit-queue", type=Path, required=True)
    parser.add_argument("--tasks-file", type=Path)
    parser.add_argument("--orders-file", type=Path)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--run-mode", choices=("production", "preview"), default="production")
    parser.add_argument("--queue-namespace", default="production")
    parser.add_argument("--estimator-backend", choices=("python_gpu", "r_reference"), default="python_gpu")
    parser.add_argument("--price-measure", choices=("main", "median", "hedonic"), default="main")
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--anticipation-months", type=int, default=0)
    parser.add_argument("--transaction-count-threshold", type=int, default=1)
    parser.add_argument("--max-gsc-cross-city-donors", type=int, default=5_000)
    parser.add_argument("--gsc-donor-sampling-seed", type=int, default=20260823)
    parser.add_argument("--qualification-receipt", type=Path)
    parser.add_argument("--receipt-digest", default="")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--requeue-failed",
        action="store_true",
        help="Return hard-failed tasks of this run to pending before starting.",
    )
    return parser.parse_args()


def file_digest(path: Path | None) -> str:
    if path is None:
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_task_keys(path: Path) -> set[tuple[int, str]]:
    import pandas as pd

    frame = pd.read_csv(path)
    required = {"treatment_order", "outcome_family"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"tasks file lacks columns: {sorted(missing)}")
    keys = {
        (int(order), str(family))
        for order, family in zip(frame["treatment_order"], frame["outcome_family"], strict=False)
    }
    if len(keys) != len(frame) or not keys:
        raise ValueError(f"tasks file is empty or contains duplicate keys: {path}")
    return keys


def read_order_keys(path: Path) -> set[int]:
    import pandas as pd

    frame = pd.read_csv(path)
    if "treatment_order" not in frame.columns:
        raise ValueError(f"orders file lacks treatment_order: {path}")
    values = [int(value) for value in frame["treatment_order"]]
    if len(set(values)) != len(values) or not values:
        raise ValueError(f"orders file is empty or contains duplicate orders: {path}")
    return set(values)


def snapshot_queue(pool, run_id: str, family_path: Path, unit_path: Path, control_path: Path, run_mode: str) -> None:
    import pandas as pd

    from urban_intervention.causal.label_queue.runtime import settings
    from urban_intervention.causal.label_queue.state import sync_unit_queue
    from urban_intervention.utils import atomic_write_csv

    family = pd.read_csv(family_path)
    for column in ("status", "selected_method", "failure_reason"):
        if column in family.columns:
            family[column] = family[column].astype("object")
    records = pool.task_records(run_id)
    index = {
        (int(order), str(family_name)): row_index
        for row_index, (order, family_name) in enumerate(
            zip(family["treatment_order"], family["outcome_family"], strict=False)
        )
    }
    for record in records:
        key = (int(record["treatment_order"]), str(record["outcome_family"]))
        row_index = index.get(key)
        if row_index is None:
            raise RuntimeError(f"SQLite task is absent from family queue: {key}")
        if record["pool_status"] == "failed":
            continue
        family.loc[row_index, "status"] = record["queue_status"]
        if "selected_method" in family.columns:
            value = record["selected_method"]
            family.loc[row_index, "selected_method"] = pd.NA if value is None else value
        if "failure_reason" in family.columns:
            value = record["failure_reason"]
            family.loc[row_index, "failure_reason"] = pd.NA if value is None else value
    atomic_write_csv(family, family_path)

    control = pd.read_csv(control_path)
    settings.run_mode = run_mode
    affected_orders = {
        int(record["treatment_order"])
        for record in records
        if record["pool_status"] != "failed"
    }
    for order in affected_orders:
        rows = family.loc[family["treatment_order"].astype(int).eq(int(order))]
        if len(rows) == 4 and rows["status"].astype(str).isin(TERMINAL).all():
            sync_unit_queue(int(order), family, control, unit_queue_path=unit_path)


def ensure_private_queues(run_dir: Path, worker_count: int, family: Path, control: Path, unit: Path) -> list[tuple[Path, Path, Path]]:
    queues: list[tuple[Path, Path, Path]] = []
    for worker_number in range(worker_count):
        directory = run_dir / "workers" / f"worker_{worker_number:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        paths = (
            directory / family.name,
            directory / control.name,
            directory / unit.name,
        )
        for source, target in zip((family, control, unit), paths, strict=True):
            shutil.copy2(source, target)
        queues.append(paths)
    return queues


def create_or_load_run(pool, args, task_rows):
    from urban_intervention.causal.label_queue.round_pool import (
        RunIdentity,
        digest_json,
        new_run_id,
    )

    receipt_digest = args.receipt_digest or file_digest(args.qualification_receipt)
    identity = RunIdentity(
        run_id=args.run_id or new_run_id(),
        specification_fingerprint=digest_json(
            {
                "estimator_backend": args.estimator_backend,
                "price_measure": args.price_measure,
                "window": args.window,
                "anticipation_months": args.anticipation_months,
                "transaction_count_threshold": args.transaction_count_threshold,
                "max_gsc_cross_city_donors": args.max_gsc_cross_city_donors,
                "gsc_donor_sampling_seed": args.gsc_donor_sampling_seed,
            }
        ),
        receipt_digest=receipt_digest,
        sample_digest=digest_json(sorted((int(row["treatment_order"]), str(row["outcome_family"])) for row in task_rows)),
        queue_namespace=args.queue_namespace,
    )
    existing = pool.get_run(identity.run_id)
    if existing is None:
        pool.create_run(identity, task_rows)
    else:
        actual = (
            existing.specification_fingerprint,
            existing.receipt_digest,
            existing.sample_digest,
            existing.queue_namespace,
        )
        expected = (
            identity.specification_fingerprint,
            identity.receipt_digest,
            identity.sample_digest,
            identity.queue_namespace,
        )
        if actual != expected:
            raise RuntimeError("existing Round Pool run identity does not match the requested specification")
    return identity


def main() -> int:
    args = parse_args()
    if args.worker_count < 1 or args.max_attempts < 1:
        raise ValueError("worker count and max attempts must be positive")
    if args.tasks_file is not None and args.orders_file is not None:
        raise ValueError("--tasks-file and --orders-file are mutually exclusive")
    args.family_queue = args.family_queue.resolve()
    args.control_queue = args.control_queue.resolve()
    args.unit_queue = args.unit_queue.resolve()
    args.run_root = args.run_root.resolve()
    args.db = args.db.resolve()
    if args.qualification_receipt is not None:
        args.qualification_receipt = args.qualification_receipt.resolve()

    import pandas as pd

    from urban_intervention.causal.label_queue.round_pool import ROUND_NAMES, RoundPool

    family = pd.read_csv(args.family_queue)
    keys: set[tuple[int, str]] | None = None
    if args.tasks_file is not None:
        keys = read_task_keys(args.tasks_file.resolve())
    elif args.orders_file is not None:
        orders = read_order_keys(args.orders_file.resolve())
        keys = {
            (int(order), str(family_name))
            for order, family_name in zip(family["treatment_order"], family["outcome_family"], strict=False)
            if int(order) in orders
        }
    all_rows = family.to_dict(orient="records")
    task_rows = [
        row
        for row in all_rows
        if keys is None or (int(row["treatment_order"]), str(row["outcome_family"])) in keys
    ]
    if not task_rows:
        raise ValueError("selected Round Pool task set is empty")

    pool = RoundPool(args.db)
    identity = create_or_load_run(pool, args, task_rows)
    if args.requeue_failed:
        recovered = pool.requeue_failed(identity.run_id)
        print(f"Requeued {recovered} hard-failed task(s) for run {identity.run_id}", flush=True)
    run_dir = args.run_root / identity.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if len(gpu_ids) < args.worker_count:
        raise ValueError("--gpu-ids must provide at least one device per worker")

    os.environ["MIT_CAUSAL_FAMILY_QUEUE"] = str(args.family_queue)
    os.environ["MIT_CAUSAL_CONTROL_QUEUE"] = str(args.control_queue)
    os.environ["MIT_CAUSAL_UNIT_QUEUE"] = str(args.unit_queue)
    # On resume, SQLite is authoritative.  Publish its snapshot before
    # refreshing private worker queues so stale CSV checkpoints cannot win.
    snapshot_queue(pool, identity.run_id, args.family_queue, args.unit_queue, args.control_queue, args.run_mode)
    private_queues = ensure_private_queues(
        run_dir,
        args.worker_count,
        args.family_queue,
        args.control_queue,
        args.unit_queue,
    )
    print(f"Round Pool run_id={identity.run_id}", flush=True)
    print(f"SQLite state={args.db}", flush=True)
    print(f"Selected tasks={len(task_rows)}", flush=True)

    worker_script = Path(__file__).with_name("run_round_pool_worker.py")
    for round_name in ROUND_NAMES:
        state = {str(row["name"]): str(row["state"]) for row in pool.round_states(identity.run_id)}
        if state.get(round_name) == "closed":
            continue
        print(f"ROUND {round_name}: launching {args.worker_count} dynamic workers", flush=True)
        processes: list[tuple[str, subprocess.Popen[bytes]]] = []
        for worker_number, (family_private, control_private, unit_private) in enumerate(private_queues):
            worker_id = f"{identity.run_id}:worker:{worker_number}"
            command = [
                sys.executable,
                str(worker_script),
                "--db",
                str(args.db),
                "--run-id",
                identity.run_id,
                "--round-name",
                round_name,
                "--worker-id",
                worker_id,
                "--family-queue",
                str(family_private),
                "--control-queue",
                str(control_private),
                "--unit-queue",
                str(unit_private),
                "--run-mode",
                args.run_mode,
                "--estimator-backend",
                args.estimator_backend,
                "--price-measure",
                args.price_measure,
                "--window",
                str(args.window),
                "--anticipation-months",
                str(args.anticipation_months),
                "--transaction-count-threshold",
                str(args.transaction_count_threshold),
                "--max-gsc-cross-city-donors",
                str(args.max_gsc_cross_city_donors),
                "--gsc-donor-sampling-seed",
                str(args.gsc_donor_sampling_seed),
                "--max-attempts",
                str(args.max_attempts),
            ]
            if args.qualification_receipt is not None:
                command.extend(["--qualification-receipt", str(args.qualification_receipt), "--receipt-digest", identity.receipt_digest])
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["MIT_CAUSAL_DEVICE"] = f"cuda:{gpu_ids[worker_number]}"
            child_env["MIT_CAUSAL_GPU_IDS"] = gpu_ids[worker_number]
            child_env["MIT_PROJECT_ROOT"] = str(ROOT)
            process = subprocess.Popen(command, cwd=ROOT, env=child_env)
            processes.append((worker_id, process))
            print(f"  worker {worker_number + 1}/{args.worker_count} pid={process.pid} device=cuda:{gpu_ids[worker_number]}", flush=True)
        return_codes: list[tuple[str, int]] = []
        for worker_id, process in processes:
            return_codes.append((worker_id, process.wait()))
        for worker_id, return_code in return_codes:
            if return_code != 0:
                recovered = pool.requeue_worker(identity.run_id, worker_id, f"worker exited with code {return_code}")
                print(f"  {worker_id} exited={return_code}; requeued={recovered}", flush=True)
        pool.requeue_expired(identity.run_id)
        snapshot_queue(pool, identity.run_id, args.family_queue, args.unit_queue, args.control_queue, args.run_mode)
        counts = pool.counts(identity.run_id)
        print(f"  Round snapshot: {counts}", flush=True)
        if counts.get("failed", 0):
            failures = [
                record
                for record in pool.task_records(identity.run_id)
                if record["pool_status"] == "failed"
            ]
            raise RuntimeError(f"Round {round_name} has hard task failures: {failures[:5]}")
        if not pool.close_round_if_drained(identity.run_id, round_name):
            raise RuntimeError(f"Round {round_name} did not drain after all workers exited")

    snapshot_queue(pool, identity.run_id, args.family_queue, args.unit_queue, args.control_queue, args.run_mode)
    counts = pool.counts(identity.run_id)
    print(f"Round Pool complete: {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
