"""Print a compact state count for all VIIRS Earth Engine tasks."""

import time
from collections import Counter

import ee


def main() -> None:
    for attempt in range(20):
        try:
            ee.Initialize(project="macro-city-engine")
            tasks = [
                task
                for task in ee.data.getTaskList()
                if task.get("description", "").lower().startswith("viirs")
            ]
            print(dict(Counter(task.get("state") for task in tasks)), flush=True)
            return
        except Exception as exc:
            print(f"retry={attempt + 1}: {exc}", flush=True)
            time.sleep(3)
    raise RuntimeError("Unable to read VIIRS task status")


if __name__ == "__main__":
    main()
