"""Modular Python/GPU production orchestrator for causal response labels.

This is an independent replacement candidate for ``run_causal_label_queue.py``.
The original entry remains frozen while both implementations are compared.
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from urban_intervention.causal.label_queue import *  # noqa: E402,F403
from urban_intervention.causal.label_queue import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
