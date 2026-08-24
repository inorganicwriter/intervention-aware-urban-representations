"""Audit causal GPU shadows against the fail-closed formal promotion gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.causal.gpu.qualification import audit_shadow_manifests  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-parity-tasks", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifests = sorted(args.root.rglob("manifest.json"))
    if not manifests:
        raise ValueError(f"no shadow manifests found under {args.root}")
    report = audit_shadow_manifests(
        manifests,
        minimum_parity_tasks=args.minimum_parity_tasks,
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
