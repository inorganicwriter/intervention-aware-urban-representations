"""Keep the monthly VIIRS Drive export queue full until every job is submitted.

Earth Engine limits a project to roughly 3,000 queued export tasks.  This
supervisor therefore refills freed slots instead of trying to create all 6,864
tasks at once.  A batch start timestamp separates this clean run from older
tasks whose Drive outputs may have been deleted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import ee

from urban_intervention.data.paths import CATALOG_DIR

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "collection"))

from run_gee_viirs_bbox_export import (  # noqa: E402
    ACTIVE_CITIES,
    PROJECT,
    SUPPORTED_YEARS,
    _description,
    queue,
)

DEFAULT_LOG = CATALOG_DIR / "snapshots" / "viirs_gee_submission.jsonl"


def log_event(path: Path, event: str, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        **values,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_tasks() -> list[dict]:
    for attempt in range(1, 11):
        try:
            ee.Initialize(project=PROJECT)
            return ee.data.getTaskList()
        except Exception as exc:
            print(f"[STATUS RETRY {attempt}/10] {exc}", flush=True)
            time.sleep(min(30, attempt * 3))
    raise RuntimeError("Unable to read the Earth Engine task list")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-start-ms",
        type=int,
        required=True,
        help="Creation timestamp marking this clean Drive-export batch",
    )
    parser.add_argument("--queue-limit", type=int, default=3000)
    parser.add_argument("--reserve-slots", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    jobs = [
        (city, year, month)
        for city in ACTIVE_CITIES
        for year in SUPPORTED_YEARS
        for month in range(1, 13)
    ]
    desired = {_description(*job): job for job in jobs}
    print(
        f"supervising {len(jobs)} monthly Drive exports; batch_start_ms={args.batch_start_ms}",
        flush=True,
    )
    log_event(
        args.log, "supervisor_started", universe=len(jobs), batch_start_ms=args.batch_start_ms
    )

    while True:
        tasks = get_tasks()
        active = [task for task in tasks if task.get("state") in {"READY", "RUNNING"}]
        current = [
            task
            for task in tasks
            if task.get("description") in desired
            and int(task.get("creation_timestamp_ms", 0)) >= args.batch_start_ms
        ]
        accepted = {
            task["description"]
            for task in current
            if task.get("state") in {"READY", "RUNNING", "COMPLETED"}
        }
        missing = [job for job in jobs if _description(*job) not in accepted]
        states = dict(Counter(task.get("state") for task in current))
        print(
            f"coverage={len(accepted)}/{len(jobs)} active_all={len(active)} "
            f"batch_states={states} missing={len(missing)}",
            flush=True,
        )
        log_event(
            args.log,
            "status",
            coverage=len(accepted),
            universe=len(jobs),
            active_all=len(active),
            batch_states=states,
            missing=len(missing),
        )

        if not missing:
            log_event(args.log, "all_jobs_submitted", universe=len(jobs), batch_states=states)
            print("all 6864 monthly Drive exports have been submitted", flush=True)
            return

        capacity = max(0, args.queue_limit - args.reserve_slots - len(active))
        if capacity == 0:
            time.sleep(args.poll_seconds)
            continue

        submitted = 0
        for job in missing[:capacity]:
            description = _description(*job)
            try:
                task_id = queue(*job)
                submitted += 1
                log_event(args.log, "submitted", description=description, task_id=task_id)
            except Exception as exc:
                message = str(exc)
                print(f"[SUBMIT PAUSE] {description}: {message}", flush=True)
                log_event(args.log, "submit_error", description=description, error=message)
                break
        print(f"refill_submitted={submitted}", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
