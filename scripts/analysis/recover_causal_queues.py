"""Recover the canonical family and control queues from authoritative sources.

The frozen baseline queue CSVs in ``data/active/causal/`` are tracked in git.
Restoring a code package over the server working copy therefore reverts the
terminal production queues to that baseline.  The label products themselves are
untouched by this; only the two ledgers must be rebuilt:

- family queue: the pre-demotion QC backup plus the recorded MC demotions;
- control queue: the published Response Artifact, whose matched labels carry
  ``control_unit_key``.

The command writes recovered queues to a new directory and never edits the
release, the task artifacts, or the frozen baseline files.

Usage:
  python scripts/analysis/recover_causal_queues.py \
    --family-queue-backup outputs/causal_labels/qc/outcome_family_work_queue.before_<stamp>.csv \
    --demotions outputs/causal_labels/qc/mc_insufficient_pre_demotions.csv \
    --response-release data/active/causal/releases/production \
    --output-directory outputs/causal_labels/recovered
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.utils import atomic_write_csv  # noqa: E402

MATCHED_PREFIX = "frozen_matched_change"
QUEUE_MUTABLE_COLUMNS = ("status", "selected_method", "failure_reason")
EXPECTED_STATUS_COUNTS = {
    "skipped": 9162,
    "matched_labelled": 7553,
    "gsc_labelled": 2322,
    "mc_labelled": 1155,
}


def apply_demotions(
    family_queue: pd.DataFrame, demotions: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Apply the recorded MC demotions to the pre-demotion queue backup."""
    queue = family_queue.copy()
    for column in QUEUE_MUTABLE_COLUMNS:
        if column in queue.columns:
            queue[column] = queue[column].astype("object")
    reasons = {
        (int(order), str(family)): str(reason)
        for order, family, reason in zip(
            demotions["treatment_order"],
            demotions["outcome_family"],
            demotions["reason"],
            strict=False,
        )
    }
    mask = pd.Series(
        [
            (int(order), str(family)) in reasons
            for order, family in zip(
                queue["treatment_order"], queue["outcome_family"], strict=False
            )
        ],
        index=queue.index,
    )
    if mask.any():
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
    return queue, int(mask.sum())


def build_control_queue(artifact: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the frozen-control ledger from the published Response Artifact."""
    for column in ("method", "control_unit_key", "donor_scope", "treatment_order"):
        if column not in artifact.columns:
            raise ValueError(f"response artifact lacks required column: {column}")
    method = artifact["method"].astype("string")
    scope = artifact["donor_scope"].astype("string")
    keys = artifact["control_unit_key"].astype("string")
    selected = artifact.loc[
        method.str.startswith(MATCHED_PREFIX, na=False)
        & scope.eq("same_city")
        & keys.notna()
        & keys.ne(""),
        ["treatment_order", "control_unit_key"],
    ].copy()
    selected = selected.drop_duplicates("treatment_order")
    parts = selected["control_unit_key"].astype(str).str.partition("::")
    selected["control_city_key"] = parts[0]
    selected["control_grid_id"] = parts[2]
    selected["status"] = "matched"
    selected["donor_scope"] = "same_city"
    return selected[
        [
            "treatment_order",
            "status",
            "control_unit_key",
            "control_city_key",
            "control_grid_id",
            "donor_scope",
        ]
    ].reset_index(drop=True)


def _resolve_backup(pattern: str) -> Path:
    matches = sorted(Path().glob(pattern))
    if not matches:
        raise SystemExit(f"no family-queue backup matches: {pattern}")
    return matches[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family-queue-backup",
        type=Path,
        help="Pre-demotion QC backup; defaults to the newest matching file.",
    )
    parser.add_argument(
        "--demotions",
        type=Path,
        default=Path("outputs/causal_labels/qc/mc_insufficient_pre_demotions.csv"),
    )
    parser.add_argument(
        "--response-release",
        type=Path,
        default=Path("data/active/causal/releases/production"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/causal_labels/recovered"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backup = args.family_queue_backup or _resolve_backup(
        "outputs/causal_labels/qc/outcome_family_work_queue.before_*.csv"
    )
    if not backup.is_file():
        raise SystemExit(f"family-queue backup not found: {backup}")
    if not args.demotions.is_file():
        raise SystemExit(f"demotions file not found: {args.demotions}")
    artifact_path = args.response_release / "response_artifact.parquet"
    if not artifact_path.is_file():
        raise SystemExit(f"response artifact not found: {artifact_path}")

    before = pd.read_csv(backup)
    demotions = pd.read_csv(args.demotions)
    queue, demoted = apply_demotions(before, demotions)

    artifact = pd.read_parquet(
        artifact_path,
        columns=["treatment_order", "method", "donor_scope", "control_unit_key"],
    )
    control = build_control_queue(artifact)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    family_path = args.output_directory / "outcome_family_work_queue.csv"
    control_path = args.output_directory / "control_design_queue.csv"
    atomic_write_csv(queue, family_path)
    atomic_write_csv(control, control_path)

    counts = queue["status"].value_counts().to_dict()
    matched_orders = set(
        queue.loc[queue["status"].eq("matched_labelled"), "treatment_order"].astype(int)
    )
    control_orders = set(control["treatment_order"].astype(int))
    missing = sorted(matched_orders - control_orders)
    payload = {
        "family_queue_backup": str(backup),
        "demotions_applied": demoted,
        "family_status_counts": {str(key): int(value) for key, value in counts.items()},
        "family_status_matches_expected": {
            str(key): int(counts.get(key, 0)) == expected
            for key, expected in EXPECTED_STATUS_COUNTS.items()
        },
        "control_rows": int(len(control)),
        "control_missing_matched_orders": len(missing),
        "control_missing_orders_sample": missing[:10],
        "family_queue": str(family_path),
        "control_queue": str(control_path),
    }
    (args.output_directory / "recovery_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if missing:
        print(
            f"WARNING: {len(missing)} matched_labelled orders lack a control key "
            "in the release; inspect before rerunning the event study",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
