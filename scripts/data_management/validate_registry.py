"""Validate concrete paths declared by the canonical dataset registry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from urban_intervention.data.registry import iter_datasets, missing_concrete_paths  # noqa: E402


def main() -> int:
    datasets = list(iter_datasets())
    missing = missing_concrete_paths()
    report = {
        "datasets": len(datasets),
        "concrete_paths": sum(dataset.concrete_path is not None for dataset in datasets),
        "missing": [
            {"dataset": dataset.name, "path": str(dataset.concrete_path)} for dataset in missing
        ],
        "ok": not missing,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
