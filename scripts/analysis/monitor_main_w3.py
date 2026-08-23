"""Lightweight read-only monitor for the representative main-spec run."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = ROOT / "outputs" / "causal_labels" / "tasks"
QUEUE_ROOT = ROOT / "data" / "active" / "causal"
CUTOFF = datetime(2026, 8, 19, 2, 55).timestamp()
SAMPLE_FILE = ROOT / "outputs" / "causal_labels" / "representative_sample_400.csv"
SHARD_COUNT = 8


def main() -> None:
    with SAMPLE_FILE.open(newline="", encoding="utf-8-sig") as handle:
        sample_orders = {int(row["treatment_order"]) for row in csv.DictReader(handle)}

    base_queue = QUEUE_ROOT / "outcome_family_work_queue_shard_00.csv"
    with base_queue.open(newline="", encoding="utf-8-sig") as handle:
        all_orders = sorted({int(row["treatment_order"]) for row in csv.DictReader(handle)})
    quotient, remainder = divmod(len(all_orders), SHARD_COUNT)
    owner = {}
    start = 0
    for shard in range(SHARD_COUNT):
        length = quotient + int(shard < remainder)
        for order in all_orders[start : start + length]:
            owner[order] = shard
        start += length

    manifests = []
    gsc_attempts = 0
    for path in TASK_ROOT.rglob("manifest.json"):
        order_part = path.parent.parent.name
        if path.stat().st_mtime >= CUTOFF and order_part.isdigit() and int(order_part) in sample_orders:
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
    for path in TASK_ROOT.rglob("gsc_attempt.json"):
        order_part = path.parent.parent.name
        if path.stat().st_mtime >= CUTOFF and order_part.isdigit() and int(order_part) in sample_orders:
            gsc_attempts += 1

    print(f"manifests={len(manifests)}")
    print(f"gsc_attempts={gsc_attempts}")
    print(f"manifest_status={dict(Counter(item.get('status') for item in manifests))}")

    queue_counts = Counter()
    for path in sorted(QUEUE_ROOT.glob("outcome_family_work_queue_shard_*.csv")):
        shard_part = path.stem.rsplit("_", 1)[-1]
        if not shard_part.isdigit():
            continue
        shard = int(shard_part)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                order = int(row["treatment_order"])
                if order in sample_orders and owner.get(order) == shard:
                    queue_counts[row["status"]] += 1
    print(f"sample_queue_status={dict(queue_counts)}")


if __name__ == "__main__":
    main()
