"""Discover R artifacts and distribute Matching/GSC/MC shadow fits over GPUs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.causal.gpu.contract_builder import build_python_contract  # noqa: E402
from urban_intervention.causal.gpu.contracts import (  # noqa: E402
    GPU_IMPLEMENTATION_VERSION,
    SHADOW_SCHEMA,
)
from urban_intervention.causal.gpu.io import cv_contract_artifact_paths  # noqa: E402
from urban_intervention.causal.gpu.provenance import fingerprints_match  # noqa: E402
from urban_intervention.causal.gpu.runtime import RuntimeConfig, TorchRuntime  # noqa: E402
from urban_intervention.causal.gpu.scheduler import (  # noqa: E402
    GpuTask,
    TaskResult,
    run_gpu_tasks,
)

ESTIMATOR_DIRS = {"gsc": "xu_gsc", "mc": "matrix_completion"}


def _estimated_cost(path: Path, estimator: str) -> float:
    """Approximate repeated SVD/search work, not merely serialized row count."""
    metadata = pq.ParquetFile(path).metadata
    rows = max(1, metadata.num_rows)
    columns = max(1, metadata.num_columns)
    repetitions = {"matching": 1, "gsc": 5 * 6, "mc": 20 * 20}[estimator]
    return max(1.0, rows * columns * repetitions / 10_000)


def _already_passed(output: Path, args: argparse.Namespace) -> bool:
    manifest = output / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != SHADOW_SCHEMA
            or payload.get("implementation_version") != GPU_IMPLEMENTATION_VERSION
            or payload.get("formal_eligible") is not False
        ):
            return False
        runtime = payload.get("runtime", {})
        if (
            runtime.get("dtype") != args.dtype
            or runtime.get("chunk_size") != args.chunk_size
            or not abs(float(runtime.get("memory_fraction")) - args.memory_fraction) < 1e-12
        ):
            return False
        estimator = payload.get("estimator")
        if estimator in {"gsc", "mc"}:
            config = payload.get("estimator_config", {})
            expected_contract = getattr(args, "contract_backend", "any")
            contract_mismatch = (
                payload.get("contract_backend") != "not_used"
                if args.tuning == "reference"
                else expected_contract != "any"
                and payload.get("contract_backend") != expected_contract
            )
            if (
                payload.get("converged") is not True
                or payload.get("tuning_source") != args.tuning
                or contract_mismatch
                or config.get("max_iter") != args.max_iter
                or not abs(float(config.get("tol")) - args.tol) < 1e-15
                or config.get("gsc_bootstrap_mode", "none")
                != getattr(args, "gsc_bootstrap_mode", "none")
                or config.get("gsc_n_bootstrap", 0)
                != getattr(args, "gsc_n_bootstrap", 0)
                or config.get("mc_inference", "none")
                != getattr(args, "mc_inference", "none")
                or config.get("inference_batch_size", 16)
                != getattr(args, "inference_batch_size", 16)
            ):
                return False
            if getattr(args, "formal_qualification", False) and payload.get(
                "inference", {}
            ).get(
                "formal_validated"
            ) is not True:
                return False
        if getattr(args, "formal_qualification", False) and payload.get(
            "qualification_passed"
        ) is not True:
            return False
        if not fingerprints_match(payload.get("source_fingerprints")):
            return False
        parity = payload["parity"]
        return parity.get("passed") is True or parity.get("available") is False
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _assert_raw_qualification_reference(labels: Path) -> None:
    """Require unwindowed R labels for estimator-level inference parity.

    Moving-window standard errors in the legacy R artifact combine marginal
    errors and therefore are not a valid reference for the Python joint-path
    aggregation.  Qualification is performed at window=1; the formal Python
    runner validates moving-window inference separately from joint replicates.
    """
    schema = pq.ParquetFile(labels).schema_arrow
    columns = set(schema.names)
    selected = [
        column
        for column in ("minimum_window_n", "uncertainty_source")
        if column in columns
    ]
    if not selected:
        return
    frame = pd.read_parquet(labels, columns=selected)
    if "minimum_window_n" in frame:
        windows = pd.to_numeric(frame["minimum_window_n"], errors="coerce")
        if windows.gt(1).any():
            raise ValueError(
                "formal qualification requires R reference labels produced with "
                "observation_window=1"
            )
    if "uncertainty_source" in frame:
        sources = frame["uncertainty_source"].astype("string")
        if sources.str.contains(r"_window(?:[2-9]|[1-9][0-9]+)$", na=False).any():
            raise ValueError(
                "formal qualification requires unwindowed R inference labels"
            )


def discover_tasks(args: argparse.Namespace) -> list[GpuTask]:
    selected = set(args.estimators.split(","))
    unknown = selected - {"matching", "gsc", "mc"}
    if unknown:
        raise ValueError(f"unknown estimators: {sorted(unknown)}")
    tasks: list[GpuTask] = []
    for estimator, directory in ESTIMATOR_DIRS.items():
        if estimator not in selected:
            continue
        root = args.staging_root / directory
        if not root.exists():
            continue
        for panel in sorted(root.rglob("estimation_panel.parquet")):
            labels = panel.with_name("causal_response_labels.parquet")
            if (args.tuning == "reference" or args.formal_qualification) and not labels.is_file():
                continue
            if args.formal_qualification:
                _assert_raw_qualification_reference(labels)
            if args.tuning == "gpu" and any(
                not path.is_file()
                for path in cv_contract_artifact_paths(panel.parent, estimator)
            ):
                continue
            relative = panel.parent.relative_to(root)
            output = args.output_root / estimator / relative
            if not args.retry and _already_passed(output, args):
                continue
            parts = relative.parts
            cache_key = "/".join(parts[:4]) if len(parts) >= 4 else str(relative)
            tasks.append(
                GpuTask(
                    task_id=f"{estimator}:{relative.as_posix()}",
                    cache_key=f"{estimator}:{cache_key}",
                    cost=_estimated_cost(panel, estimator),
                    payload={
                        "estimator": estimator,
                        "panel": str(panel.resolve()),
                        "reference_labels": str(labels.resolve()) if labels.is_file() else None,
                        "tuning_source": args.tuning,
                        "output": str(output.resolve()),
                        "dtype": args.dtype,
                        "chunk_size": args.chunk_size,
                        "memory_fraction": args.memory_fraction,
                        "max_iter": args.max_iter,
                        "tol": args.tol,
                        "contract_backend": args.contract_backend,
                        "gsc_bootstrap_mode": args.gsc_bootstrap_mode,
                        "gsc_n_bootstrap": args.gsc_n_bootstrap,
                        "mc_inference": args.mc_inference,
                        "inference_batch_size": args.inference_batch_size,
                        "inference_relative_rmse_tolerance": (
                            args.gsc_inference_relative_rmse_tolerance
                            if estimator == "gsc"
                            else args.mc_inference_relative_rmse_tolerance
                        ),
                        "minimum_ci_zero_agreement": args.minimum_ci_zero_agreement,
                        "formal_qualification": args.formal_qualification,
                    },
                )
            )
    if "matching" in selected and args.matching_input_root.exists():
        for artifact in sorted(args.matching_input_root.rglob("matching_input.parquet")):
            input_path = artifact.parent
            relative = input_path.relative_to(args.matching_input_root)
            output = args.output_root / "matching" / relative
            if not args.retry and _already_passed(output, args):
                continue
            tasks.append(
                GpuTask(
                    task_id=f"matching:{relative.as_posix()}",
                    cache_key=f"matching:{relative.as_posix()}",
                    cost=_estimated_cost(artifact, "matching"),
                    payload={
                        "estimator": "matching",
                        "input": str(input_path.resolve()),
                        "output": str(output.resolve()),
                        "dtype": args.dtype,
                        "chunk_size": args.chunk_size,
                        "memory_fraction": args.memory_fraction,
                        "formal_qualification": args.formal_qualification,
                    },
                )
            )
    per_estimator = getattr(args, "max_tasks_per_estimator", None)
    if per_estimator is not None:
        selected_tasks: list[GpuTask] = []
        counts: dict[str, int] = {}
        for task in tasks:
            estimator = str(task.payload["estimator"])
            if counts.get(estimator, 0) >= per_estimator:
                continue
            selected_tasks.append(task)
            counts[estimator] = counts.get(estimator, 0) + 1
        tasks = selected_tasks
    return tasks[: args.max_tasks] if args.max_tasks is not None else tasks


def prepare_python_contracts(args: argparse.Namespace) -> int:
    """Stage native contracts in-process before GPU task discovery."""
    selected = set(args.estimators.split(",")) & {"gsc", "mc"}
    candidates: list[tuple[str, Path]] = []
    for estimator, directory in ESTIMATOR_DIRS.items():
        if estimator not in selected:
            continue
        root = args.staging_root / directory
        if root.exists():
            candidates.extend(
                (estimator, panel)
                for panel in sorted(root.rglob("estimation_panel.parquet"))
                if not args.formal_qualification
                or panel.with_name("causal_response_labels.parquet").is_file()
            )
    per_estimator = getattr(args, "max_tasks_per_estimator", None)
    if per_estimator is not None:
        limited: list[tuple[str, Path]] = []
        counts: dict[str, int] = {}
        for estimator, panel in candidates:
            if counts.get(estimator, 0) >= per_estimator:
                continue
            limited.append((estimator, panel))
            counts[estimator] = counts.get(estimator, 0) + 1
        candidates = limited
    if args.max_tasks is not None:
        candidates = candidates[: args.max_tasks]
    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value.strip()]
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu"
    runtime: TorchRuntime | None = None
    completed = 0
    for estimator, panel in candidates:
        native_manifest = panel.with_name("gpu_contract_manifest.csv")
        native_artifacts = (
            (panel.with_name("gsc_cv_folds.python.parquet"),)
            if estimator == "gsc"
            else (
                panel.with_name("mc_cv_folds.python.parquet"),
                panel.with_name("mc_lambda_grid.python.csv"),
            )
        )
        if (
            not args.rebuild_python_contracts
            and native_manifest.is_file()
            and all(path.is_file() for path in native_artifacts)
        ):
            continue
        if estimator == "mc" and runtime is None:
            runtime = TorchRuntime(RuntimeConfig(device=device, dtype="float64", seed=20260725))
        build_python_contract(
            panel,
            estimator,  # type: ignore[arg-type]
            runtime=runtime,
            force=args.rebuild_python_contracts,
        )
        completed += 1
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimators", default="matching,gsc,mc")
    parser.add_argument(
        "--staging-root", type=Path, default=ROOT / "outputs" / "complete_estimators" / "staging"
    )
    parser.add_argument(
        "--matching-input-root",
        type=Path,
        default=ROOT / "outputs" / "complete_estimators" / "gpu_matching_inputs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "complete_estimators" / "gpu_shadow_queue",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--tuning", choices=("reference", "gpu"), default="reference")
    parser.add_argument(
        "--contract-backend",
        choices=("any", "r_fect_2.4.5", "python_native"),
        default="any",
        help="Fail closed if a GPU tuning contract was built by another backend.",
    )
    parser.add_argument("--chunk-size", type=int, default=65_536)
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument(
        "--gsc-bootstrap-mode",
        choices=("none", "auto", "reference_empirical", "reference_ar"),
        default="none",
    )
    parser.add_argument("--gsc-n-bootstrap", type=int, default=0)
    parser.add_argument("--mc-inference", choices=("none", "jackknife"), default="none")
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--gsc-inference-relative-rmse-tolerance", type=float, default=0.35)
    parser.add_argument("--mc-inference-relative-rmse-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-ci-zero-agreement", type=float, default=0.9)
    parser.add_argument(
        "--formal-qualification",
        action="store_true",
        help="Run GPU tuning plus formal GSC bootstrap/MC jackknife parity checks.",
    )
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument(
        "--max-tasks-per-estimator",
        type=int,
        help="Stratified limit applied independently to Matching, GSC and MC.",
    )
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-python-contracts",
        action="store_true",
        help="Build Python-native folds/lambda grids beside staged panels before discovery.",
    )
    parser.add_argument("--rebuild-python-contracts", action="store_true")
    return parser.parse_args()


def _queue_row(result: TaskResult) -> dict[str, object]:
    error = result.error
    value = result.value or {}
    if (
        error is None
        and value.get("estimator") in {"gsc", "mc"}
        and value.get("converged") is not True
    ):
        error = "GPU final refit did not converge"
    parity_value = None if error else value.get("parity", {}).get("passed")
    if (
        error is None
        and value.get("qualification_requested") is True
        and value.get("qualification_passed") is not True
    ):
        error = "formal point/inference/quality qualification gate failed"
    return {
        "task_id": result.task_id,
        "gpu_id": result.gpu_id,
        "status": "error" if error else "completed",
        "parity_passed": None if parity_value is None else bool(parity_value),
        "error": error,
    }


def main() -> int:
    args = parse_args()
    if args.max_tasks is not None and args.max_tasks_per_estimator is not None:
        raise ValueError("use only one of --max-tasks and --max-tasks-per-estimator")
    if args.formal_qualification:
        # Qualification requires the same minimum count for every estimator.
        # Treat the legacy global limit as a per-estimator limit in this mode.
        args.max_tasks_per_estimator = (
            args.max_tasks
            if args.max_tasks is not None
            else args.max_tasks_per_estimator or 3
        )
        args.max_tasks = None
        args.tuning = "gpu"
        args.contract_backend = "python_native"
        args.dtype = "float64"
        args.gsc_bootstrap_mode = "auto"
        args.gsc_n_bootstrap = 200
        args.mc_inference = "jackknife"
        args.prepare_python_contracts = True
    if args.tuning == "gpu" and (args.max_iter != 5000 or args.tol != 1e-5):
        raise ValueError("strict GPU tuning requires the frozen fect max_iter=5000 and tol=1e-5")
    if args.gsc_bootstrap_mode == "none" and args.gsc_n_bootstrap != 0:
        raise ValueError("--gsc-n-bootstrap must be zero when bootstrap mode is none")
    if args.gsc_bootstrap_mode != "none" and args.gsc_n_bootstrap < 2:
        raise ValueError("GSC bootstrap requires at least two replicates")
    if args.rebuild_python_contracts:
        args.prepare_python_contracts = True
    if args.prepare_python_contracts:
        prepared = prepare_python_contracts(args)
        print(f"Built {prepared} Python-native contract(s)")
    gpu_ids = tuple(int(value) for value in args.gpu_ids.split(",") if value.strip())
    tasks = discover_tasks(args)
    if args.dry_run:
        for task in tasks:
            print({"task_id": task.task_id, "cache_key": task.cache_key, "cost": task.cost})
        print(f"Discovered {len(tasks)} task(s)")
        return 0
    if not tasks:
        print("No runnable GPU tasks were discovered; export R contracts first.")
        return 1
    results = run_gpu_tasks(
        tasks,
        callable_path="urban_intervention.causal.gpu.workers:run_shadow_task",
        gpu_ids=gpu_ids,
    )
    rows = [
        _queue_row(result)
        for result in results
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_root / "queue_results.csv", index=False)
    failed = sum(row["status"] == "error" or row["parity_passed"] is False for row in rows)
    print(f"Completed {len(rows) - failed}/{len(rows)} task(s); failed={failed}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
