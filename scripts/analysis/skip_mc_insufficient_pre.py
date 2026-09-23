"""One-off maintenance: demote MC labels without a clean pre-period anchor.

The Response Artifact contract requires at least one observed pretreatment
period for MC labels (``pre_observed_periods >= 1``).  Early cohorts that
predate the outcome panels were nevertheless recorded as ``mc_labelled``.
This command applies the release rule to the terminal queues and, with
``--apply``, records the affected tasks as ``skipped`` in the family queue and
the SQLite round-pool state.  Label and manifest files are never edited.

Usage:
  python scripts/analysis/skip_mc_insufficient_pre.py            # dry run
  python scripts/analysis/skip_mc_insufficient_pre.py --apply \
    --db /home/data_ssd/mit_phase2_round_pool_20260913.sqlite \
    --run-id <run-id>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.data.paths import (  # noqa: E402
    OUTCOME_FAMILY_QUEUE,
    OUTPUT_CAUSAL_TASKS_DIR,
)
from urban_intervention.utils import atomic_write_csv  # noqa: E402

PREFIX = "mc_insufficient_pre_observation"
INVALID_PROOF = "mc_invalid_estimator_proof"


def _to_numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def classify_mc_demotion(frame: pd.DataFrame) -> str | None:
    """Return the demotion reason for an MC task, or None when it is valid."""
    if "label_available" not in frame.columns:
        return None
    available = frame["label_available"].astype("boolean").fillna(False).astype(bool)
    selected = frame.loc[available]
    if selected.empty:
        return None
    method_ok = (
        selected["method"].astype("string").str.startswith("athey_2021_mc_", na=False).all()
        if "method" in selected.columns
        else False
    )
    lam = _to_numeric(selected, "mc_lambda")
    cv = _to_numeric(selected, "mc_cv_mspe")
    if (
        not method_ok
        or not np.isfinite(lam).all()
        or (lam.fillna(-1) < 0).any()
        or not np.isfinite(cv).all()
    ):
        return INVALID_PROOF
    if "pre_observed_periods" not in selected.columns:
        return PREFIX
    if _to_numeric(selected, "pre_observed_periods").fillna(-1).lt(1).any():
        return PREFIX
    return None


def scan_task(
    order: int, family: str, status: str, task_root: Path
) -> dict[str, object] | None:
    """Return a demotion record for one MC task, or None when it stays valid."""
    if str(status) != "mc_labelled":
        return None
    path = task_root / f"{order:05d}" / str(family) / "labels.parquet"
    if not path.is_file():
        return {
            "treatment_order": order,
            "outcome_family": str(family),
            "reason": "mc_labelled_task_missing_labels",
            "pre_min": np.nan,
            "available_rows": 0,
        }
    try:
        frame = pd.read_parquet(path)
    except Exception as error:  # noqa: BLE001
        return {
            "treatment_order": order,
            "outcome_family": str(family),
            "reason": "mc_labelled_task_unreadable_labels",
            "pre_min": np.nan,
            "available_rows": 0,
            "note": str(error)[:120],
        }
    reason = classify_mc_demotion(frame)
    if reason is None:
        return None
    available = frame["label_available"].astype("boolean").fillna(False).astype(bool)
    pre = _to_numeric(frame.loc[available], "pre_observed_periods")
    return {
        "treatment_order": order,
        "outcome_family": str(family),
        "reason": reason,
        "pre_min": float(pre.min()) if len(pre) else np.nan,
        "available_rows": int(available.sum()),
    }


def apply_skips(
    queue_path: Path,
    records: list[dict[str, object]],
    *,
    db_path: Path | None,
    run_id: str | None,
) -> dict[str, int]:
    """Record the demoted tasks as skipped in the queue and the run database."""
    if db_path is not None and not run_id:
        raise ValueError("--db requires --run-id so SQLite stays authoritative")
    reasons = {
        (int(str(record["treatment_order"])), str(record["outcome_family"])): str(
            record["reason"]
        )
        for record in records
    }
    if not reasons:
        raise ValueError("no demotion records supplied")
    queue = pd.read_csv(queue_path)
    for column in ("status", "selected_method", "failure_reason"):
        if column in queue.columns:
            queue[column] = queue[column].astype("object")
    mask = pd.Series(
        [
            (int(order), str(family)) in reasons
            for order, family in zip(
                queue["treatment_order"], queue["outcome_family"], strict=False
            )
        ],
        index=queue.index,
    )
    if not mask.any():
        raise ValueError("no queue rows matched the demotion records")
    queue.loc[mask, "status"] = "skipped"
    queue.loc[mask, "selected_method"] = pd.NA
    queue.loc[mask, "failure_reason"] = [
        reasons[(int(order), str(family))]
        for order, family in zip(
            queue.loc[mask, "treatment_order"],
            queue.loc[mask, "outcome_family"],
            strict=False,
        )
    ]
    atomic_write_csv(queue, queue_path)
    db_updated = 0
    if db_path is not None:
        now = time.time()
        connection = sqlite3.connect(db_path)
        try:
            for (order, family), reason in reasons.items():
                cursor = connection.execute(
                    "UPDATE tasks SET queue_status='skipped', selected_method=NULL, "
                    "failure_reason=?, updated_at=? "
                    "WHERE run_id=? AND treatment_order=? AND outcome_family=?",
                    (reason, now, run_id, order, family),
                )
                db_updated += max(cursor.rowcount, 0)
            connection.commit()
        finally:
            connection.close()
    return {"queue_rows_updated": int(mask.sum()), "db_rows_updated": db_updated}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-queue", type=Path, default=OUTCOME_FAMILY_QUEUE)
    parser.add_argument("--task-root", type=Path, default=OUTPUT_CAUSAL_TASKS_DIR)
    parser.add_argument(
        "--report-dir", type=Path, default=ROOT / "outputs" / "causal_labels" / "qc"
    )
    parser.add_argument("--db", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.apply and (args.db is None or not args.run_id):
        print("refusing --apply: pass both --db and --run-id so SQLite stays authoritative")
        return 1
    queue = pd.read_csv(args.family_queue)
    targets = queue.loc[
        queue["status"].astype(str).eq("mc_labelled"),
        ["treatment_order", "outcome_family", "status"],
    ]
    rows = list(targets.itertuples(index=False))
    print(f"mc_labelled tasks: {len(rows)}", flush=True)

    def scan_row(row: object) -> dict[str, object] | None:
        return scan_task(
            int(row.treatment_order),  # type: ignore[attr-defined]
            str(row.outcome_family),  # type: ignore[attr-defined]
            str(row.status),  # type: ignore[attr-defined]
            args.task_root,
        )

    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for start in range(0, len(rows), 200):
            chunk = rows[start : start + 200]
            for result in pool.map(scan_row, chunk):
                if result is not None:
                    records.append(result)
            print(
                f"scanned {min(start + 200, len(rows))}/{len(rows)} | demotions: {len(records)}",
                flush=True,
            )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        print("no demotions required")
        return 0
    report = pd.DataFrame(records)
    report_path = args.report_dir / "mc_insufficient_pre_demotions.csv"
    report.to_csv(report_path, index=False)
    merged = report.merge(
        queue[
            [
                "treatment_order",
                "outcome_family",
                "city_key",
                "opening_month",
                "selected_method",
            ]
        ],
        on=["treatment_order", "outcome_family"],
        how="left",
    )
    print("\n== demotions by family x reason ==")
    print(merged.groupby(["outcome_family", "reason"]).size().to_string())
    print(f"\ndemotion rows: {len(merged)} | report: {report_path}")
    if not args.apply:
        print("dry run: rerun with --apply --db <sqlite> --run-id <run-id>")
        return 0
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = args.report_dir / f"outcome_family_work_queue.before_{stamp}.csv"
    shutil.copy2(args.family_queue, backup)
    summary = apply_skips(
        args.family_queue, records, db_path=args.db, run_id=args.run_id
    )
    payload: dict[str, object] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "family_queue": str(args.family_queue),
        "queue_backup": str(backup),
        "db": str(args.db),
        "run_id": args.run_id,
        "demotions": len(records),
        "by_family": merged.groupby("outcome_family").size().to_dict(),
        "by_reason": merged.groupby("reason").size().to_dict(),
        "orders": sorted(int(value) for value in merged["treatment_order"].unique()),
        **summary,
        "operator": os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
    }
    summary_path = args.report_dir / f"mc_insufficient_pre_apply_{stamp}.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("\n== apply summary ==")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print("summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
