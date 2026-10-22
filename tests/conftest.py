"""Pytest configuration shared across all test modules.

Ensures the packaged source tree and compatibility script directories are
available without requiring an editable install during local tests.
"""

import os
import sys
from pathlib import Path

# Windows: torch and numpy both load libiomp5md.dll; whichever is imported first
# can abort the process when the other is loaded later.  Allow the duplicate
# OpenMP runtime before any test module imports torch (same workaround as the
# representation CLI entry points).
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.data.paths import (  # noqa: E402
    SCRIPTS_ANALYSIS_DIR,
    SCRIPTS_COLLECTION_DIR,
    SCRIPTS_DIR,
    SCRIPTS_LABELS_DIR,
    SRC_DIR,
)

SCRIPTS = SCRIPTS_DIR
SRC = SRC_DIR
COLLECTION = SCRIPTS_COLLECTION_DIR
LABELS = SCRIPTS_LABELS_DIR
ANALYSIS = SCRIPTS_ANALYSIS_DIR

for p in [str(SRC), str(SCRIPTS), str(COLLECTION), str(LABELS), str(ANALYSIS)]:
    if p not in sys.path:
        sys.path.insert(0, p)
