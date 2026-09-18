"""Execute one dynamic Round Pool using a private queue snapshot."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--round-name",
        choices=(
            "same_city_matching",
            "same_city_gsc",
            "same_city_mc",
            "cross_city_matching",
            "cross_city_gsc",
            "cross_city_mc",
        ),
        required=True,
    )
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--family-queue", type=Path, required=True)
    parser.add_argument("--control-queue", type=Path, required=True)
    parser.add_argument("--unit-queue", type=Path, required=True)
    parser.add_argument("--run-mode", choices=("production", "preview"), default="production")
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
    return parser.parse_args()


def nullable(value: object) -> str | None:
    if value is None:
        return None
    try:
        if bool(value != value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    return None if text in {"", "<NA>", "nan", "None"} else text


def start_heartbeat(pool, run_id: str, lease, worker_id: str):
    stop = threading.Event()

    def heartbeat_loop(stop_event=stop, task_lease=lease) -> None:
        interval = max(min(pool.lease_seconds / 3.0, 300.0), 1.0)
        while not stop_event.wait(interval):
            if not pool.heartbeat(run_id, task_lease, worker_id):
                return

    thread = threading.Thread(
        target=heartbeat_loop,
        name=f"heartbeat-{worker_id}",
        daemon=True,
    )
    thread.start()
    return stop, thread


def main() -> int:
    args = parse_args()
    for key, value in {
        "MIT_PROJECT_ROOT": str(ROOT),
        "MIT_CAUSAL_FAMILY_QUEUE": str(args.family_queue.resolve()),
        "MIT_CAUSAL_CONTROL_QUEUE": str(args.control_queue.resolve()),
        "MIT_CAUSAL_UNIT_QUEUE": str(args.unit_queue.resolve()),
    }.items():
        os.environ[key] = value

    import pyarrow.parquet as pq

    from urban_intervention.causal.gpu.qualification import (
        validate_formal_qualification_receipt,
    )
    from urban_intervention.causal.label_queue.orchestrator import process_one
    from urban_intervention.causal.label_queue.round_pool import RoundPool
    from urban_intervention.causal.label_queue.rounds import run_global_round
    from urban_intervention.causal.label_queue.runtime import (
        CONTROL_QUEUE,
        SUPPORT,
        settings,
    )
    from urban_intervention.causal.label_queue.state import (
        read_control_queue,
        read_family_queue,
    )

    settings.run_mode = args.run_mode
    settings.estimator_backend = args.estimator_backend
    settings.qualification_receipt = args.qualification_receipt.resolve() if args.qualification_receipt else None
    settings.max_gsc_cross_city_donors = args.max_gsc_cross_city_donors
    settings.gsc_donor_sampling_seed = args.gsc_donor_sampling_seed
    settings.anticipation_months = args.anticipation_months
    settings.price_measure = args.price_measure
    settings.label_window = args.window
    settings.transaction_count_threshold = args.transaction_count_threshold
    settings.family_queue = args.family_queue.resolve()

    if args.estimator_backend == "python_gpu" and args.run_mode == "production":
        if settings.qualification_receipt is None:
            raise ValueError("production Python Round Pool workers require a qualification receipt")
        settings.qualification_proof = validate_formal_qualification_receipt(
            settings.qualification_receipt,
            verify_bound_sources=False,
            expected_sha256=args.receipt_digest or None,
        )
    else:
        settings.qualification_proof = {}

    pool = RoundPool(args.db)
    queue = read_family_queue(args.family_queue.resolve())
    support = pq.read_table(SUPPORT).to_pandas()
    control_queue = read_control_queue(args.control_queue.resolve())
    claimed = 0
    completed = 0
    failed = 0
    print(f"[{args.worker_id}] started Round {args.round_name}", flush=True)
    while True:
        pool.requeue_expired(args.run_id)
        lease = pool.claim_next(args.run_id, args.round_name, args.worker_id)
        if lease is None:
            break
        claimed += 1
        mask = (
            queue["treatment_order"].astype(int).eq(lease.treatment_order)
            & queue["outcome_family"].astype(str).eq(lease.outcome_family)
        )
        indices = queue.index[mask]
        if len(indices) != 1:
            pool.fail(
                args.run_id,
                lease,
                args.worker_id,
                f"private queue does not contain unique claimed task: {lease.treatment_order}/{lease.outcome_family}",
            )
            failed += 1
            continue
        index = int(indices[0])
        # SQLite is authoritative after a restart.  Normalize stale running
        # markers in this private snapshot before invoking the legacy task
        # implementation for exactly this Round.
        queue.loc[index, "status"] = lease.queue_status
        row = queue.loc[index].copy()
        heartbeat_stop, heartbeat_thread = start_heartbeat(
            pool, args.run_id, lease, args.worker_id
        )
        try:
            if args.round_name == "same_city_matching":
                process_one(
                    queue,
                    index,
                    support,
                    control_queue,
                    False,
                    phase="matching",
                    control_queue_path=CONTROL_QUEUE,
                )
            elif args.round_name == "same_city_gsc":
                process_one(
                    queue,
                    index,
                    support,
                    control_queue,
                    False,
                    phase="gsc",
                    control_queue_path=CONTROL_QUEUE,
                )
            else:
                run_global_round(
                    queue,
                    index,
                    row,
                    control_queue,
                    CONTROL_QUEUE,
                    round_name=args.round_name,
                )
            after = queue.loc[index]
            status = str(after["status"])
            if status not in {
                "matched_labelled",
                "gsc_labelled",
                "mc_labelled",
                "skipped",
                "gsc_pending",
                "mc_pending",
                "cross_matching_running",
                "cross_gsc_running",
                "cross_mc_running",
            }:
                raise RuntimeError(f"task returned unsupported queue status {status!r}")
            if not pool.finish(
                args.run_id,
                lease,
                args.worker_id,
                queue_status=status,
                selected_method=nullable(after.get("selected_method")),
                failure_reason=nullable(after.get("failure_reason")),
            ):
                raise RuntimeError("lease was fenced before result commit")
            completed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            error = f"{type(exc).__name__}: {exc}"
            pool.fail(
                args.run_id,
                lease,
                args.worker_id,
                error,
                retry=lease.attempt_count < args.max_attempts,
            )
            print(
                f"[{args.worker_id}] task {lease.treatment_order}/{lease.outcome_family} failed ({'requeued' if lease.attempt_count < args.max_attempts else 'marked failed'}): {exc}",
                flush=True,
            )
            traceback.print_exc()
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2.0)
    print(
        f"[{args.worker_id}] finished Round {args.round_name}: claimed={claimed}, completed={completed}, failed={failed}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
