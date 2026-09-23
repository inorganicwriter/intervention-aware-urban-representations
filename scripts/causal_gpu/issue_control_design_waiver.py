"""Issue an operator-attested spot-check waiver for control-design records.

A waiver lets a strict Response Artifact release accept control records whose
stored matching fingerprint differs from the current source tree, when the
operator attests that the records came from reviewed production patches and
spot checks recompute field-identical controls under the current tree.  It
never overrides schema, backend, leakage, queue-status or method checks.

Usage:
  python scripts/causal_gpu/issue_control_design_waiver.py \
    --reason "server tree assembled from reviewed local patches; spot checks identical" \
    --confirm-spot-check
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.causal.gpu.control_design_waiver import (  # noqa: E402
    CONTROL_DESIGN_WAIVER_ATTESTATION,
    CONTROL_DESIGN_WAIVER_SCHEMA,
    MINIMUM_SPOT_CHECK_ORDERS,
    SPOT_CHECK_FIELDS,
    load_control_design_waiver,
    matching_source_hashes,
)
from urban_intervention.causal.gpu.provenance import (  # noqa: E402
    estimator_code_fingerprint,
)
from urban_intervention.data.paths import (  # noqa: E402
    CONTROL_DESIGN_QUEUE,
    OUTPUT_CONTROL_TASKS_DIR,
)
from urban_intervention.utils import atomic_write_json  # noqa: E402

ACTIVE_STATUSES = {"matched", "gsc_pending"}


def read_control_record(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"control record is not unique: {path}")
    return {str(key): value for key, value in frame.iloc[0].to_dict().items()}


def compare_spot_check_records(
    production_root: Path,
    spot_check_root: Path,
    orders: list[int],
) -> list[dict[str, str]]:
    """Return every field mismatch between recorded and recomputed controls."""
    mismatches: list[dict[str, str]] = []
    for order in orders:
        recorded = read_control_record(
            production_root / f"{order:05d}" / "control_record.csv"
        )
        recomputed = read_control_record(
            spot_check_root / f"{order:05d}" / "control_record.csv"
        )
        for field in SPOT_CHECK_FIELDS:
            recorded_value = "" if pd.isna(recorded.get(field)) else str(recorded.get(field))
            recomputed_value = (
                "" if pd.isna(recomputed.get(field)) else str(recomputed.get(field))
            )
            if recorded_value != recomputed_value:
                mismatches.append(
                    {
                        "treatment_order": str(order),
                        "field": field,
                        "recorded": recorded_value,
                        "spot_check": recomputed_value,
                    }
                )
    return mismatches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-queue", type=Path, default=CONTROL_DESIGN_QUEUE)
    parser.add_argument(
        "--control-task-root", type=Path, default=OUTPUT_CONTROL_TASKS_DIR
    )
    parser.add_argument(
        "--spot-check-root",
        type=Path,
        default=ROOT / "outputs" / "control_design" / "spot_check",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "causal_gpu" / "control_design_drift_waiver.json",
    )
    parser.add_argument("--min-orders", type=int, default=MINIMUM_SPOT_CHECK_ORDERS)
    parser.add_argument(
        "--reason",
        required=True,
        help="Audit-trail reason for accepting the fingerprint drift.",
    )
    parser.add_argument(
        "--confirm-spot-check",
        action="store_true",
        help="Required acknowledgement that the spot check is complete and identical.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_orders < MINIMUM_SPOT_CHECK_ORDERS:
        print(
            f"refusing waiver: --min-orders must be at least {MINIMUM_SPOT_CHECK_ORDERS}"
        )
        return 1
    queue = pd.read_csv(args.control_queue)
    active = queue[queue["status"].astype(str).isin(ACTIVE_STATUSES)]
    current = estimator_code_fingerprint("matching")
    stale: dict[int, str] = {}
    for _, row in active.iterrows():
        order = int(row["treatment_order"])
        path = args.control_task_root / f"{order:05d}" / "control_record.csv"
        if not path.is_file():
            print(f"refusing waiver: control record is missing: {path}")
            return 1
        record = read_control_record(path)
        fingerprint = str(record.get("code_fingerprint") or "")
        if fingerprint != current:
            stale[order] = fingerprint
    if not stale:
        print("no stale control-record fingerprints; no waiver is needed")
        return 0
    spot_orders = sorted(
        int(path.parent.name)
        for path in args.spot_check_root.glob("*/control_record.csv")
        if path.parent.name.isdigit()
    )
    print(f"active control records: {len(active)}")
    print(f"stale records: {len(stale)}")
    print(f"stale fingerprints: {sorted(set(stale.values()))}")
    print(f"spot-check orders: {len(spot_orders)}")
    if len(spot_orders) < args.min_orders:
        print(
            f"refusing waiver: spot check must cover at least {args.min_orders} orders"
        )
        return 1
    unknown = sorted(set(spot_orders) - set(stale))
    if unknown:
        print(f"refusing waiver: spot-check orders are not stale records: {unknown[:10]}")
        return 1
    try:
        mismatches = compare_spot_check_records(
            args.control_task_root.resolve(), args.spot_check_root.resolve(), spot_orders
        )
    except (OSError, ValueError) as error:
        print(f"refusing waiver: {error}")
        return 1
    if mismatches:
        print("refusing waiver: spot check found non-identical control records:")
        for item in mismatches[:20]:
            print("  -", item)
        return 1
    if not args.confirm_spot_check:
        print(f"{len(spot_orders)}/{len(spot_orders)} spot-checked records are identical")
        print("rerun with --confirm-spot-check and --reason ... to issue the waiver")
        return 2
    waiver: dict[str, Any] = {
        "schema": CONTROL_DESIGN_WAIVER_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "code_fingerprints": {"matching": current},
        "source_sha256": matching_source_hashes(),
        "accepted_code_fingerprints": sorted(set(stale.values())),
        "spot_check": {
            "root": str(args.spot_check_root.resolve()),
            "orders": spot_orders,
            "fields": list(SPOT_CHECK_FIELDS),
            "all_identical": True,
            "min_orders": int(args.min_orders),
        },
        "reason": args.reason,
        "attestation": CONTROL_DESIGN_WAIVER_ATTESTATION,
        "operator": os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(waiver, output, default=str)
    try:
        load_control_design_waiver(output)
    except ValueError as error:
        output.unlink(missing_ok=True)
        print(f"refusing waiver: issued payload failed validation: {error}")
        return 1
    print(f"waiver written: {output}")
    print(f"stale records covered: {len(stale)}")
    print(f"spot-check orders: {len(spot_orders)} (all identical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
