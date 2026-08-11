#!/usr/bin/env python
"""Compatibility wrapper for the packaged training entry point.

Preferred invocations:
    urban-train-representation DATA_DIR --output OUTPUT_DIR [options]
    python -m urban_intervention.representation.cli DATA_DIR --output OUTPUT_DIR [options]
"""

from urban_intervention.representation.cli import main

if __name__ == "__main__":
    main()
