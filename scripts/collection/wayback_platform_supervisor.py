"""Start Lianjia only after a complete Beike Wayback run has finished.

This small supervisor is intentionally conservative: it waits for the active
collector lock to disappear, then verifies that inventories exist for every
Beike city/page target before launching the next platform.  A crashed or
manually interrupted Beike run therefore cannot silently trigger Lianjia.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(BASE / "scripts" / "collection"))
from wayback_research_scraper import BEIKE_SUB, INVENTORY_DIR, LOCK_PATH  # noqa: E402

from urban_intervention.config.project import ACTIVE_CITIES  # noqa: E402


def beike_inventory_complete() -> bool:
    expected = {
        INVENTORY_DIR / f"beike_{city}_{page}.json"
        for city in ACTIVE_CITIES
        if city in BEIKE_SUB
        for page in ("xiaoqu", "chengjiao")
    }
    missing = sorted(path.name for path in expected if not path.exists())
    if missing:
        logging.error(
            "Beike inventory incomplete (%d missing); not starting Lianjia: %s",
            len(missing),
            ", ".join(missing),
        )
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely chain Beike then Lianjia Wayback collection"
    )
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-hours", type=float, default=48)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    deadline = time.monotonic() + args.max_wait_hours * 3600
    logging.info("Waiting for Beike collector lock to clear: %s", LOCK_PATH)
    while LOCK_PATH.exists():
        if time.monotonic() >= deadline:
            logging.error("Timed out waiting for Beike collector; not starting Lianjia.")
            return 2
        time.sleep(args.poll_seconds)

    if not beike_inventory_complete():
        return 3

    command = [
        sys.executable,
        str(BASE / "scripts" / "collection" / "wayback_research_scraper.py"),
        "--platform",
        "lianjia",
        "--city",
        "all",
        "--workers",
        "1",
        "--min-interval",
        "3",
        "--proxy",
        args.proxy,
        "--wayback-scheme",
        "http",
        "--log-level",
        "INFO",
    ]
    logging.info("Beike inventory complete. Starting Lianjia: %s", " ".join(command))
    return subprocess.run(command, cwd=BASE).returncode


if __name__ == "__main__":
    raise SystemExit(main())
