"""Python/GPU production orchestrator for one-grid-at-a-time causal labels.

The production default is the contract-tested Python/PyTorch implementation;
the audited R implementation remains available as an explicit reference
backend. This module supplies transactional queue transitions, method routing,
normalized label files, and crash-safe resume behavior.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from urban_intervention.causal.gpu.qualification import (
    validate_formal_qualification_receipt,
)
from urban_intervention.utils import atomic_write_json

from .orchestrator import process_one
from .runtime import (
    CONTROL_QUEUE,
    OUTCOMES,
    ROOT,
    SUPPORT,
    TREATMENT_UNIT_LIST,
    UNIT_QUEUE,
    settings,
)
from .state import (
    queue_variant_path,
    read_control_queue,
    read_family_queue,
    read_orders_file,
    read_tasks_file,
    shard_order_slice,
    shard_queue_path,
    sync_unit_queue,
)
from .validation import invalidate_stale_terminal_tasks

atomic_json = partial(atomic_write_json, default=str)


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
        default="python_gpu",
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
    settings.run_mode = args.run_mode
    settings.estimator_backend = args.estimator_backend
    settings.qualification_receipt = args.qualification_receipt
    settings.max_gsc_cross_city_donors = args.max_gsc_cross_city_donors
    settings.gsc_donor_sampling_seed = args.gsc_donor_sampling_seed
    settings.anticipation_months = args.anticipation_months
    settings.price_measure = args.price_measure
    settings.label_window = args.window
    settings.transaction_count_threshold = args.transaction_count_threshold
    if settings.max_gsc_cross_city_donors < 20:
        parser.error("--max-gsc-cross-city-donors must be at least 20")
    if settings.transaction_count_threshold < 1:
        parser.error("--transaction-count-threshold must be positive")

    has_shard = args.shard_id is not None and args.shard_count is not None
    if bool(args.shard_id is not None) != bool(args.shard_count is not None):
        parser.error("--shard-id and --shard-count must be used together")

    family_master_queue = queue_variant_path(settings.family_queue, settings.run_mode)
    unit_master_queue = queue_variant_path(UNIT_QUEUE, settings.run_mode)
    control_master_queue = queue_variant_path(CONTROL_QUEUE, settings.run_mode)
    if settings.run_mode != "production":
        import shutil as _shutil

        for source, target in (
            (settings.family_queue, family_master_queue),
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
        master_family_queue = settings.family_queue
        settings.family_queue = family_queue

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
            sample_path = (
                args.orders_file if args.orders_file.is_absolute() else ROOT / args.orders_file
            )
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
        settings.estimator_backend == "python_gpu"
        and settings.run_mode == "production"
        and not args.dry_run
    ):
        if settings.qualification_receipt is None:
            parser.error(
                "production Python tasks require --qualification-receipt "
                "(or MIT_CAUSAL_QUALIFICATION_RECEIPT)"
            )
        settings.qualification_proof = validate_formal_qualification_receipt(
            settings.qualification_receipt
        )
    else:
        settings.qualification_proof = {}
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
            task_keys = {(order, family) for order, family in task_keys if order in orders_set}
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
        if not args.dry_run and settings.run_mode == "production":
            sync_unit_queue(
                int(queue.loc[index, "treatment_order"]),
                queue,
                control_queue,
                unit_queue_path=unit_queue,
            )
    print(f"Processed {len(eligible)} task(s); phase={args.phase}; dry_run={args.dry_run}")
    return 0
