#!/usr/bin/env python3
"""Compatibility entry point for the relocated Python causal-label queue.

The production implementation moved to
``scripts/causal_python/run_causal_label_queue.py`` during the GPU migration.
This wrapper preserves older deployment commands; new documentation and code
must use the canonical ``causal_python`` path.
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from scripts.causal_python.run_causal_label_queue import *  # noqa: F403,E402
from scripts.causal_python.run_causal_label_queue import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
