#!/usr/bin/env python
"""Compatibility wrapper for the packaged ablation runner.

Preferred invocations:
    urban-run-ablation DATA_DIR --specs specs.json --output outputs/ablation
    python -m urban_intervention.representation.ablation_cli DATA_DIR --specs specs.json
"""

from urban_intervention.representation.ablation_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
