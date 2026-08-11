#!/usr/bin/env python
"""Compatibility wrapper for the packaged model-card builder.

Preferred invocations:
    urban-build-model-card OUTPUT_DIR [--out model_card.md]
    python -m urban_intervention.representation.model_card_cli OUTPUT_DIR
"""

from urban_intervention.representation.model_card_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
