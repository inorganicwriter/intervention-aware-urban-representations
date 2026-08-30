"""Reset one representative sample for a fresh causal-label production run.

The production queues are intentionally reusable, but their terminal states can
come from an earlier specification.  This utility archives the sample's prior
task artifacts and resets only the selected treatment orders; it never deletes
the archived files.  Control-design rows are left untouched by default, but
can be reset explicitly when the prior matching specification is obsolete.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CAUSAL_DIR = ROOT / "data" / "active" / "causal"
TASK_DIR = ROOT / "outputs" / "causal_labels" / "tasks"
CONTROL_TASK_DIR = ROOT / "outputs" / "control_design" / "tasks"
FIXED_CONTROL_DIR = ROOT / "outputs" / "causal_labels" / "fixed_control_staging"
COMPLETE_STAGING_DIR = ROOT / "outputs" / "complete_estimators" / "staging"
RESET_COLUMNS = ("status", "selected_method", "failure_reason")
CONTROL_RESET_COLUMNS = (
    "status",
    "active_families",
    "selected_method",
    "donor_scope",
    "control_city_key",
    "control_grid_id",
    "control_unit_key",
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
    "failure_reason",
)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def reset_queue(path: Path, orders: set[int], columns: tuple[str, ...]) -> int:
    if not path.exists():
        return 0
    frame = pd.read_csv(path)
    mask = frame["treatment_order"].astype(int).isin(orders)
    if not mask.any():
        return 0
    for column in columns:
        frame.loc[mask, column] = pd.NA
    frame.loc[mask, "status"] = "pending"
    atomic_csv(frame, path)
    return int(mask.sum())


def archive_task_artifacts(orders: set[int], archive_root: Path) -> int:
    moved = 0
    for order in sorted(orders):
        source_dir = TASK_DIR / f"{order:05d}"
        if not source_dir.exists():
            continue
        target_dir = archive_root / source_dir.relative_to(ROOT)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_dir), str(target_dir))
        moved += 1
    return moved


def archive_control_artifacts(
    orders: set[int], archive_root: Path, reset_control_design: bool
) -> int:
    """Archive control artifacts at the same scope as the requested reset."""
    moved = 0
    for order in sorted(orders):
        order_dir = CONTROL_TASK_DIR / f"{order:05d}"
        if reset_control_design and order_dir.exists():
            target = archive_root / order_dir.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(order_dir), str(target))
            moved += 1
            continue
        cross_city_dir = order_dir / "cross_city"
        if cross_city_dir.exists():
            target = archive_root / cross_city_dir.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(cross_city_dir), str(target))
            moved += 1
        legacy_attempt = order_dir / "cross_city_attempt.csv"
        if legacy_attempt.exists():
            target = archive_root / legacy_attempt.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(legacy_attempt), str(target))
            moved += 1
    return moved


def archive_order_staging(orders: set[int], archive_root: Path) -> int:
    """Move order-specific estimator staging so a fresh run cannot reuse it."""
    moved = 0
    for order in sorted(orders):
        fixed_source = FIXED_CONTROL_DIR / f"{order:05d}"
        if fixed_source.exists():
            target = archive_root / fixed_source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(fixed_source), str(target))
            moved += 1

    if COMPLETE_STAGING_DIR.exists():
        pattern = re.compile(r"(?:^|_)t(\d{5})$")
        estimator_dirs = []
        for candidate in COMPLETE_STAGING_DIR.rglob("*"):
            if not candidate.is_dir():
                continue
            match = pattern.search(candidate.name)
            if match and int(match.group(1)) in orders:
                estimator_dirs.append(candidate)
        for source in sorted(estimator_dirs, key=lambda path: len(path.parts)):
            if not source.exists():
                continue
            target = archive_root / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved += 1
    return moved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders-file", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Archive directory (default: outputs/archive/causal_reset_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--reset-control-design",
        action="store_true",
        help="Also reset matching/control-design fields for the selected orders",
    )
    args = parser.parse_args()
    if args.archive_dir is None:
        args.archive_dir = (
            ROOT
            / "outputs"
            / "archive"
            / f"causal_reset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    sample = pd.read_csv(args.orders_file)
    if "treatment_order" not in sample.columns:
        raise ValueError("orders file must contain treatment_order")
    orders = set(sample["treatment_order"].astype(int))
    if len(orders) != len(sample):
        raise ValueError("orders file contains duplicate treatment_order values")

    if args.archive_dir.exists():
        raise FileExistsError(f"archive directory already exists: {args.archive_dir}")
    args.archive_dir.mkdir(parents=True)
    moved = archive_task_artifacts(orders, args.archive_dir)
    control_moved = archive_control_artifacts(
        orders, args.archive_dir, reset_control_design=args.reset_control_design
    )
    staging_moved = archive_order_staging(orders, args.archive_dir)

    queue_paths = [CAUSAL_DIR / "outcome_family_work_queue.csv"]
    queue_paths += [
        CAUSAL_DIR / f"outcome_family_work_queue_shard_{shard:02d}.csv"
        for shard in range(args.shard_count)
    ]
    unit_paths = [CAUSAL_DIR / "counterfactual_work_queue.csv"]
    unit_paths += [
        CAUSAL_DIR / f"counterfactual_work_queue_shard_{shard:02d}.csv"
        for shard in range(args.shard_count)
    ]
    family_rows = sum(reset_queue(path, orders, RESET_COLUMNS) for path in queue_paths)
    unit_rows = sum(reset_queue(path, orders, RESET_COLUMNS) for path in unit_paths)
    control_rows = 0
    if args.reset_control_design:
        control_rows = reset_queue(
            CAUSAL_DIR / "control_design_queue.csv",
            orders,
            CONTROL_RESET_COLUMNS,
        )

    print(f"sample_orders={len(orders)}")
    print(f"archived_artifacts={moved}")
    print(f"archived_control_artifacts={control_moved}")
    print(f"archived_order_staging={staging_moved}")
    print(f"reset_family_queue_rows={family_rows}")
    print(f"reset_unit_queue_rows={unit_rows}")
    print(f"reset_control_design_rows={control_rows}")
    print(f"archive_dir={args.archive_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
