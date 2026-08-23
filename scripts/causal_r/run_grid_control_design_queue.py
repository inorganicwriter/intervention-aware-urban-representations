"""Transactional queue for one frozen physical control per treated grid."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from urban_intervention.data.paths import (  # noqa: E402
    CONTROL_DESIGN_QUEUE as QUEUE,
)
from urban_intervention.data.paths import (
    OUTPUT_CONTROL_TASKS_DIR as TASK_ROOT,
)
from urban_intervention.data.paths import (
    OUTPUT_VIIRS_PARTITION_AUDITS_DIR as VIIRS_AUDITS,
)
from urban_intervention.data.paths import (
    PROJECT_ROOT,
    R_LIB_DIR,
    collection_script,
    r_script,
)
from urban_intervention.data.paths import (
    TREATMENT_UNIT_LIST as TREATMENTS,
)
from urban_intervention.data.paths import (
    VIIRS_MONTHLY_DIR as VIIRS_MONTHLY,
)

R_SCRIPT = os.environ.get("MIT_RSCRIPT", "Rscript")
R_LIB = Path(os.environ.get("MIT_R_LIB", str(R_LIB_DIR)))
ROOT = PROJECT_ROOT
VIIRS_START = pd.Period("2012-01", freq="M")
VIIRS_END = pd.Period("2024-12", freq="M")
VIIRS_CACHE_CONTRACT = "complete_44_city_2012_2024_monthly_v1"
STRING_COLUMNS = (
    "status",
    "active_families",
    "selected_method",
    "donor_scope",
    "control_city_key",
    "control_grid_id",
    "control_unit_key",
    "failure_reason",
)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def initialize_queue(path: Path = QUEUE) -> pd.DataFrame:
    treatments = pq.read_table(TREATMENTS).to_pandas()
    if len(treatments) != 5_048 or treatments[["city_key", "grid_id"]].duplicated().any():
        raise ValueError("Treatment list is not the immutable 5,048-grid list")
    queue = treatments[
        ["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"]
    ].copy()
    queue["status"] = "pending"
    for column in (
        "active_families",
        "selected_method",
        "donor_scope",
        "control_city_key",
        "control_grid_id",
        "control_unit_key",
        "failure_reason",
    ):
        queue[column] = pd.NA
    for column in (
        "candidate_count",
        "candidate_city_count",
        "training_feature_count",
        "holdout_feature_count",
        "training_distance",
        "holdout_rms_standardized_gap",
        "holdout_max_abs_standardized_gap",
        "training_distance_threshold",
        "holdout_rms_threshold",
        "holdout_max_abs_threshold",
    ):
        queue[column] = pd.NA
    queue["control_selection_uses_post_outcome"] = False
    atomic_csv(queue, path)
    return queue


def read_queue(path: Path = QUEUE) -> pd.DataFrame:
    queue = pd.read_csv(path)
    queue["treatment_order"] = queue["treatment_order"].astype("int64")
    for column in STRING_COLUMNS:
        queue[column] = queue[column].astype("string")
    return queue


def read_orders_file(path: Path) -> list[int]:
    """Read and validate a treatment-order sample manifest."""
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


def treatment_cities() -> list[str]:
    treatments = pq.read_table(TREATMENTS, columns=["city_key"]).to_pandas()
    cities = sorted(treatments["city_key"].astype(str).unique())
    if len(cities) != 44:
        raise ValueError(f"Expected 44 treatment cities, found {len(cities)}")
    return cities


def expected_viirs_periods() -> list[pd.Period]:
    return list(pd.period_range(VIIRS_START, VIIRS_END, freq="M"))


def missing_viirs_cache(
    cities: list[str] | None = None,
    periods: list[pd.Period] | None = None,
    monthly_root: Path = VIIRS_MONTHLY,
    audit_root: Path = VIIRS_AUDITS,
) -> list[str]:
    cities = treatment_cities() if cities is None else cities
    periods = expected_viirs_periods() if periods is None else periods
    missing: list[str] = []
    for city in cities:
        for period in periods:
            parquet = (
                monthly_root
                / f"city_key={city}"
                / f"year={period.year}"
                / f"month={period.month:02d}"
                / "part.parquet"
            )
            audit = audit_root / city / f"{period}.json"
            if not parquet.is_file() or not audit.is_file():
                missing.append(f"{city}:{period}")
    return missing


def assert_complete_viirs_cache() -> None:
    missing = missing_viirs_cache()
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            "Control-design production requires all 6,864 monthly VIIRS "
            "Parquet+audit partitions before matching; missing "
            f"{len(missing)} ({preview}). Run this command with "
            "--prepare-viirs-cache first."
        )


def prepare_complete_viirs_cache() -> None:
    raw = os.environ.get("MIT_VIIRS_RAW")
    if not raw:
        raise RuntimeError(
            "--prepare-viirs-cache requires MIT_VIIRS_RAW to point to the "
            "complete raw monthly VIIRS directory"
        )
    ensure_script = collection_script("ensure_viirs_monthly_cache.py")
    for city in treatment_cities():
        completed = subprocess.run(
            [
                sys.executable,
                str(ensure_script),
                "--input-dir",
                raw,
                "--city",
                city,
                "--start",
                str(VIIRS_START),
                "--end",
                str(VIIRS_END),
            ],
            cwd=ROOT,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"VIIRS cache preparation failed for {city}")
    assert_complete_viirs_cache()


def validate_durable_record(record: pd.Series, expected_order: int) -> None:
    if int(record["treatment_order"]) != expected_order:
        raise ValueError(
            f"Durable control record identity mismatch: expected {expected_order}, "
            f"found {record['treatment_order']}"
        )
    if str(record.get("viirs_cache_contract", "")) != VIIRS_CACHE_CONTRACT:
        raise ValueError(
            f"Durable control record {expected_order} lacks the complete monthly "
            "VIIRS cache contract and cannot be reused"
        )


def apply_durable_record(queue: pd.DataFrame, index: int, record_path: Path) -> None:
    record = pd.read_csv(record_path).iloc[0]
    validate_durable_record(record, int(queue.loc[index, "treatment_order"]))
    for column in record.index:
        if column in queue.columns:
            queue.loc[index, column] = record[column]


def run_batch(
    queue: pd.DataFrame, indices: list[int], reuse_durable: bool = True, workers: int = 1
) -> None:
    missing: list[int] = []
    for index in indices:
        order = int(queue.loc[index, "treatment_order"])
        record_path = TASK_ROOT / f"{order:05d}" / "control_record.csv"
        if reuse_durable and record_path.exists():
            apply_durable_record(queue, index, record_path)
        else:
            missing.append(index)
    if not missing:
        atomic_csv(queue, QUEUE)
        return

    missing_orders = [int(queue.loc[index, "treatment_order"]) for index in missing]
    queue.loc[missing, "status"] = "running"
    atomic_csv(queue, QUEUE)

    environment = os.environ.copy()
    if R_LIB.exists():
        environment["R_LIBS_USER"] = str(R_LIB)

    if workers <= 1:
        _run_single_batch(missing_orders, environment)
    else:
        _run_parallel_batches(missing_orders, environment, workers)

    for index, order in zip(missing, missing_orders, strict=False):
        record_path = TASK_ROOT / f"{order:05d}" / "control_record.csv"
        if not record_path.exists():
            queue.loc[index, ["status", "failure_reason"]] = [
                "error",
                "R process produced no output",
            ]
            continue
        apply_durable_record(queue, index, record_path)
    atomic_csv(queue, QUEUE)


def _run_single_batch(orders: list[int], environment: dict) -> None:
    completed = subprocess.run(
        [
            str(R_SCRIPT),
            str(r_script("run_grid_control_design_batch.R")),
            ",".join(map(str, orders)),
            str(TASK_ROOT),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        print(f"R batch exited {completed.returncode}: {completed.stdout[-500:]}")


def _run_parallel_batches(orders: list[int], environment: dict, workers: int) -> None:
    treatments = pq.read_table(TREATMENTS, columns=["treatment_order", "city_key"]).to_pandas()
    order_to_city = dict(
        zip(
            treatments["treatment_order"].astype(int),
            treatments["city_key"].astype(str),
            strict=False,
        )
    )

    orders_by_city: dict[str, list[int]] = {}
    for order in orders:
        city = order_to_city.get(order, "")
        orders_by_city.setdefault(city, []).append(order)

    sorted_cities = sorted(orders_by_city, key=lambda c: -len(orders_by_city[c]))
    batches: list[list[int]] = [[] for _ in range(workers)]
    for i, city in enumerate(sorted_cities):
        batches[i % workers].extend(orders_by_city[city])

    batches = [b for b in batches if b]
    if len(batches) == 1:
        _run_single_batch(batches[0], environment)
        return

    processes: list[tuple[list[int], subprocess.Popen]] = []
    for batch_orders in batches:
        orders_str = ",".join(map(str, batch_orders))
        proc = subprocess.Popen(
            [
                str(R_SCRIPT),
                str(r_script("run_grid_control_design_batch.R")),
                orders_str,
                str(TASK_ROOT),
            ],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((batch_orders, proc))
        print(f"  Worker {len(processes)}: {len(batch_orders)} tasks ({orders_str[:80]}...)")

    for batch_orders, proc in processes:
        stdout, _ = proc.communicate()
        if proc.returncode != 0:
            print(f"  Worker failed (rc={proc.returncode}): {stdout[-500:]}")
        else:
            done = sum(
                1 for o in batch_orders if (TASK_ROOT / f"{o:05d}" / "control_record.csv").exists()
            )
            print(f"  Worker completed: {done}/{len(batch_orders)} tasks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--start-order", type=int, default=1)
    parser.add_argument("--end-order", type=int)
    parser.add_argument("--max-units", type=int, default=1)
    parser.add_argument(
        "--orders",
        help="Comma-separated treatment orders to process (mutually exclusive "
        "with --start-order/--end-order ranges)",
    )
    parser.add_argument(
        "--orders-file",
        type=Path,
        help="CSV containing a treatment_order column; mutually exclusive with --orders",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel R processes (default: 1)"
    )
    parser.add_argument(
        "--prepare-viirs-cache",
        action="store_true",
        help="Materialize and verify all 44-city monthly VIIRS partitions before matching",
    )
    parser.add_argument(
        "--prepare-viirs-cache-only",
        action="store_true",
        help="Materialize and verify the VIIRS cache, then exit without changing the queue",
    )
    args = parser.parse_args()
    if args.prepare_viirs_cache or args.prepare_viirs_cache_only:
        prepare_complete_viirs_cache()
        print("Verified complete 6,864-partition monthly VIIRS cache")
        if args.prepare_viirs_cache_only:
            return 0
    if args.initialize:
        if QUEUE.exists():
            raise FileExistsError(f"Refusing to overwrite existing queue: {QUEUE}")
        queue = initialize_queue()
        print(f"Initialized {len(queue)} frozen-control rows at {QUEUE}")
        return 0
    queue = read_queue(QUEUE)
    statuses = queue["status"].astype("string")
    eligible_statuses = ["pending", "running"]
    if args.retry:
        eligible_statuses.extend(["matched", "gsc_pending", "not_matched", "error"])
    if args.orders is not None or args.orders_file is not None:
        if args.orders is not None and args.orders_file is not None:
            parser.error("--orders and --orders-file are mutually exclusive")
        orders = (
            sorted({int(value) for value in args.orders.split(",") if value.strip()})
            if args.orders is not None
            else read_orders_file(args.orders_file.resolve())
        )
        if not orders:
            parser.error("selected orders must contain at least one treatment order")
        if args.start_order != 1 or args.end_order is not None:
            parser.error("--orders is mutually exclusive with --start-order/--end-order")
        mask = (queue["treatment_order"].isin(orders)) & statuses.isin(eligible_statuses)
        indices = queue.index[mask]
    else:
        mask = (queue["treatment_order"] >= args.start_order) & statuses.isin(eligible_statuses)
        if args.end_order is not None:
            mask &= queue["treatment_order"] <= args.end_order
        indices = queue.index[mask][: args.max_units]
    selected = [int(index) for index in indices]
    if args.dry_run:
        for index in selected:
            print({"treatment_order": int(queue.loc[index, "treatment_order"])})
    elif selected:
        assert_complete_viirs_cache()
        run_batch(queue, selected, reuse_durable=not args.retry, workers=args.workers)
    print(f"Processed {len(indices)} control-design row(s); dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
