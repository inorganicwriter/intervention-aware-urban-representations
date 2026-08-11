"""Cancel every active VIIRS export submitted by this project."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee


def _cancel(task_id: str) -> tuple[str, str]:
    for attempt in range(5):
        try:
            ee.data.cancelTask(task_id)
            return task_id, "cancelled"
        except Exception as exc:
            if attempt == 4:
                return task_id, f"failed: {exc}"
            time.sleep(1 + attempt)
    return task_id, "failed"


def main() -> None:
    for attempt in range(20):
        try:
            ee.Initialize(project="macro-city-engine")
            tasks = [
                task
                for task in ee.data.getTaskList()
                if task.get("description", "").lower().startswith("viirs")
            ]
            break
        except Exception as exc:
            if attempt == 19:
                raise
            print(f"initialization retry={attempt + 1}: {exc}", flush=True)
            time.sleep(3)
    active = [task for task in tasks if task.get("state") in {"READY", "RUNNING"}]
    failed = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_cancel, task["id"]) for task in active]
        for index, future in enumerate(as_completed(futures), 1):
            task_id, state = future.result()
            if state != "cancelled":
                failed.append((task_id, state))
            if index % 50 == 0:
                print(f"cancel progress={index}/{len(active)}", flush=True)
    completed = sum(task.get("state") == "COMPLETED" for task in tasks)
    print(
        f"viirs_tasks={len(tasks)} cancel_requested={len(active)} "
        f"completed={completed} failed={len(failed)}",
        flush=True,
    )
    if failed:
        raise RuntimeError(f"Failed cancellations: {failed[:5]}")


if __name__ == "__main__":
    main()
