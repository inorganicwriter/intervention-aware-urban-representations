"""Issue an operator-attested drift waiver for a qualification receipt.

A waiver lets an otherwise eligible, environment-bound qualification receipt
survive changes to the qualification-bound estimator sources that the operator
reviewed as non-numerical (comments, typing, error paths).  It never overrides
parity, quality, convergence, environment or input-artifact gates: if any of
those fail, this command refuses to issue the waiver.

Usage:
  python scripts/causal_gpu/issue_qualification_waiver.py \
    --receipt outputs/causal_gpu/formal_qualification.json \
    --reason "comments and typing only; reviewed diff" \
    --confirm-no-numerical-change
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.causal.gpu.provenance import (  # noqa: E402
    all_estimator_code_fingerprints,
    estimator_source_files,
    file_sha256,
)
from urban_intervention.causal.gpu.qualification import (  # noqa: E402
    WAIVER_ATTESTATION,
    WAIVER_SCHEMA,
    audit_shadow_manifests,
)
from urban_intervention.utils import atomic_write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/causal_gpu/qualification_drift_waiver.json"),
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="Audit-trail reason for treating the source drift as non-numerical.",
    )
    parser.add_argument(
        "--confirm-no-numerical-change",
        action="store_true",
        help="Required acknowledgement that the reviewed diff changes no estimator numerics.",
    )
    return parser.parse_args()


def bound_source_records(manifest_paths: list[Path]) -> dict[str, str]:
    records: dict[str, str] = {}
    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in payload.get("source_fingerprints", []) or []:
            if isinstance(record, dict) and record.get("path"):
                resolved = str(Path(str(record["path"])).resolve())
                records[resolved] = str(record.get("sha256", ""))
    return records


def current_estimator_sources() -> dict[str, str]:
    package = Path(__file__).resolve().parents[2] / "src" / "urban_intervention" / "causal" / "gpu"
    current = {
        str(path.resolve()): file_sha256(path)
        for estimator in ("matching", "gsc", "mc")
        for path in estimator_source_files(estimator)
    }
    workers = package / "workers.py"
    if workers.is_file():
        current[str(workers.resolve())] = file_sha256(workers)
    return current


def build_waiver(
    receipt_path: Path,
    *,
    reason: str,
    recorded_sources: dict[str, str],
    current_sources: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    """Build the waiver payload, refusing any drift outside bound code sources."""
    missing_baseline = sorted(
        path for path in current_sources if path not in recorded_sources
    )
    if missing_baseline:
        raise ValueError(
            "bound sources lack a recorded qualification baseline: "
            + ", ".join(missing_baseline)
        )
    input_drift = sorted(
        path
        for path, digest in recorded_sources.items()
        if path not in current_sources
        and Path(path).is_file()
        and file_sha256(Path(path)) != digest
    )
    missing_inputs = sorted(
        path
        for path in recorded_sources
        if path not in current_sources and not Path(path).is_file()
    )
    if input_drift or missing_inputs:
        raise ValueError(
            "bound input artifacts changed or disappeared; a source-drift waiver "
            "is not applicable: "
            + ", ".join([*input_drift, *missing_inputs])
        )
    changed = sorted(
        path
        for path, digest in current_sources.items()
        if recorded_sources.get(path) != digest
    )
    waiver: dict[str, Any] = {
        "schema": WAIVER_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "receipt_path": str(receipt_path),
        "receipt_sha256": file_sha256(receipt_path),
        "code_fingerprints": all_estimator_code_fingerprints(),
        "source_sha256": current_sources,
        "changed_sources": [
            {
                "path": path,
                "recorded_sha256": recorded_sources.get(path, ""),
                "current_sha256": current_sources[path],
            }
            for path in changed
        ],
        "reason": reason,
        "attestation": WAIVER_ATTESTATION,
        "operator": os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
    }
    return waiver, changed


def main() -> int:
    args = parse_args()
    receipt = args.receipt.resolve()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if payload.get("schema") != "causal_gpu_formal_qualification_environment_bound":
        raise ValueError("input is not a formal qualification receipt")
    if payload.get("eligible") is not True or payload.get("reasons"):
        print("receipt is not eligible; a waiver cannot repair parity or quality failures")
        return 1
    minimum = int(payload.get("minimum_parity_tasks", 0)) or 1
    audited = payload.get("audited", [])
    manifests = [
        Path(str(item.get("path", "")))
        for item in audited
        if isinstance(item, dict) and item.get("path")
    ]
    missing = [path for path in manifests if not path.is_file()]
    if missing:
        print("audited manifests are unavailable; run the audit where they exist:")
        for path in missing:
            print("  -", path)
        return 1
    recorded = bound_source_records(manifests)
    current = current_estimator_sources()
    try:
        waiver, changed = build_waiver(
            receipt,
            reason=args.reason,
            recorded_sources=recorded,
            current_sources=current,
        )
    except ValueError as error:
        print(f"refusing waiver: {error}")
        return 1
    if not changed:
        print("no qualification-bound source drift; no waiver is needed")
        return 0
    provisional = dict(waiver, _path="", _sha256="")
    report = audit_shadow_manifests(
        manifests, minimum_parity_tasks=minimum, waiver=provisional
    )
    if report.get("eligible") is not True:
        print("refusing waiver: the receipt still fails audit for non-source reasons:")
        for reason in report.get("reasons", []):
            print("  -", reason)
        return 1
    if not args.confirm_no_numerical_change:
        print("changed qualification-bound sources:")
        for item in waiver["changed_sources"]:
            print("  -", item["path"])
        print(
            "review the diff, then rerun with --confirm-no-numerical-change "
            "and --reason ..."
        )
        return 2
    output = args.output.resolve()
    atomic_write_json(waiver, output, default=str)
    print(f"waiver written: {output}")
    print(f"changed bound sources: {len(changed)}")
    print(f"reason: {args.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
