#!/usr/bin/env python3
"""Reset causal queues and rebuild formal inputs without invoking R."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from urban_intervention.causal.setup_inputs import (  # noqa: E402
    audit_formal_target_support,
    rebuild_formal_inputs,
    reset_queues,
)
from urban_intervention.data.paths import PROJECT_ROOT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset-queues", action="store_true")
    parser.add_argument("--rebuild-formal-inputs", action="store_true")
    parser.add_argument("--audit-support", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run all three stages in order.")
    args = parser.parse_args()
    if not any((args.reset_queues, args.rebuild_formal_inputs, args.audit_support, args.all)):
        parser.error("select at least one stage or --all")
    if args.reset_queues or args.all:
        counts = reset_queues(PROJECT_ROOT)
        print(f"Reset queues: {counts}")
    if args.rebuild_formal_inputs or args.all:
        counts = rebuild_formal_inputs(PROJECT_ROOT)
        print(f"Rebuilt formal inputs: {counts}")
    if args.audit_support or args.all:
        audit = audit_formal_target_support(PROJECT_ROOT)
        print(
            "Audited formal support: "
            f"{len(audit)} treatments; "
            f"{int(audit['complete_families'].ge(1).sum())} with matching support"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
