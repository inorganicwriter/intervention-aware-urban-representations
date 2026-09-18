"""Read-only status report for a SQLite Round Pool run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    from urban_intervention.causal.label_queue.round_pool import RoundPool

    pool = RoundPool(args.db.resolve())
    run = pool.get_run(args.run_id)
    if run is None:
        print(f"run not found: {args.run_id}", file=sys.stderr)
        return 1
    records = pool.task_records(args.run_id)
    payload = {
        "run": {
            "run_id": run.run_id,
            "specification_fingerprint": run.specification_fingerprint,
            "receipt_digest": run.receipt_digest,
            "sample_digest": run.sample_digest,
            "queue_namespace": run.queue_namespace,
            "status": run.status,
        },
        "rounds": pool.round_states(args.run_id),
        "counts": dict(Counter(str(record["pool_status"]) for record in records)),
        "by_round": dict(
            Counter(
                str(record["round_name"] or "terminal")
                for record in records
                if record["pool_status"] != "failed"
            )
        ),
        "hard_failures": [
            {
                "treatment_order": record["treatment_order"],
                "outcome_family": record["outcome_family"],
                "round_name": record["round_name"],
                "last_error": record["last_error"],
                "attempt_count": record["attempt_count"],
            }
            for record in records
            if record["pool_status"] == "failed"
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
