"""Audit causal GPU shadows against the fail-closed formal promotion gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.causal.gpu.provenance import file_sha256  # noqa: E402
from urban_intervention.causal.gpu.qualification import (  # noqa: E402
    audit_shadow_manifests,
    load_qualification_waiver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-parity-tasks", type=int, default=3)
    parser.add_argument(
        "--waiver",
        type=Path,
        help="Optional qualification drift waiver covering bound source changes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifests = sorted(args.root.rglob("manifest.json"))
    if not manifests:
        raise ValueError(f"no shadow manifests found under {args.root}")
    waiver = None
    if args.waiver is not None:
        waiver_path = args.waiver.resolve()
        try:
            waiver_payload = json.loads(waiver_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"unreadable qualification waiver {waiver_path}: {exc}"
            ) from exc
        receipt_path = Path(str(waiver_payload.get("receipt_path", "")))
        receipt_sha256 = str(waiver_payload.get("receipt_sha256", ""))
        if not receipt_path.is_file() or file_sha256(receipt_path) != receipt_sha256:
            raise ValueError(
                "qualification waiver is not bound to an existing receipt revision"
            )
        waiver = load_qualification_waiver(
            waiver_path, receipt_path=receipt_path, receipt_sha256=receipt_sha256
        )
    report = audit_shadow_manifests(
        manifests,
        minimum_parity_tasks=args.minimum_parity_tasks,
        waiver=waiver,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, args.output)
    print(rendered)
    return int(not report["eligible"])


if __name__ == "__main__":
    raise SystemExit(main())
