"""One-process-per-GPU scheduling with sticky panel-cache affinity."""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import queue
import traceback
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GpuTask:
    task_id: str
    cache_key: str
    payload: dict[str, Any]
    cost: float = 1.0


@dataclass(frozen=True, slots=True)
class TaskResult:
    task_id: str
    gpu_id: int
    value: Any = None
    error: str | None = None


def sticky_partitions(tasks: Iterable[GpuTask], gpu_ids: Sequence[int]) -> dict[int, list[GpuTask]]:
    """Assign all tasks sharing a panel cache key to the same GPU worker."""
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    partitions: dict[int, list[GpuTask]] = {int(gpu_id): [] for gpu_id in gpu_ids}
    loads: dict[int, float] = {int(gpu_id): 0.0 for gpu_id in gpu_ids}
    grouped: dict[str, list[GpuTask]] = defaultdict(list)
    for task in tasks:
        if task.cost <= 0:
            raise ValueError("task cost must be positive")
        grouped[task.cache_key].append(task)
    # Longest-processing-time assignment avoids the path-order bias of the
    # former greedy loop while preserving cache affinity within each group.
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (-sum(task.cost for task in item[1]), item[0]),
    )
    for _, group in ordered_groups:
        owner = min(loads, key=lambda gpu_id: (loads[gpu_id], gpu_id))
        partitions[owner].extend(group)
        loads[owner] += sum(task.cost for task in group)
    return partitions


def _resolve_callable(path: str) -> Any:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("worker path must have the form 'module:function'")
    return getattr(importlib.import_module(module_name), attribute)


def _gpu_worker(
    gpu_id: int,
    tasks: list[GpuTask],
    callable_path: str,
    output: Any,
) -> None:
    # Must be set before importing the estimator module/PyTorch.  Each worker
    # then sees its assigned physical GPU as local cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        function = _resolve_callable(callable_path)
        cache: dict[object, Any] = {}
        for task in tasks:
            try:
                value = function(task.payload, cache)
                output.put(TaskResult(task.task_id, gpu_id, value=value))
            except Exception:
                output.put(TaskResult(task.task_id, gpu_id, error=traceback.format_exc()))
    except Exception:
        error = traceback.format_exc()
        for task in tasks:
            output.put(TaskResult(task.task_id, gpu_id, error=error))


def run_gpu_tasks(
    tasks: Sequence[GpuTask],
    *,
    callable_path: str,
    gpu_ids: Sequence[int],
) -> list[TaskResult]:
    """Run serially within each GPU and concurrently across GPUs.

    The worker callable receives ``(payload, cache)``.  Keeping the cache
    process-local allows outcomes for the same city/cohort/family/scope to
    reuse tensors and CV masks without copying CUDA tensors between cards.
    """
    if not tasks:
        return []
    partitions = sticky_partitions(tasks, gpu_ids)
    context = mp.get_context("spawn")
    output = context.Queue()
    processes = [
        context.Process(
            target=_gpu_worker,
            args=(gpu_id, assigned, callable_path, output),
            daemon=False,
        )
        for gpu_id, assigned in partitions.items()
        if assigned
    ]
    for process in processes:
        process.start()
    results: list[TaskResult] = []
    while len(results) < len(tasks):
        try:
            results.append(output.get(timeout=1))
        except queue.Empty:
            if all(not process.is_alive() for process in processes):
                break
    for process in processes:
        process.join()
        if process.exitcode not in {0, None}:
            raise RuntimeError(f"GPU worker {process.pid} exited with code {process.exitcode}")
    if len(results) != len(tasks):
        raise RuntimeError(
            f"GPU workers returned {len(results)} of {len(tasks)} expected task results"
        )
    by_id = {result.task_id: result for result in results}
    return [by_id[task.task_id] for task in tasks]
