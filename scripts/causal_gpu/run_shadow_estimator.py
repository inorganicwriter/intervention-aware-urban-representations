"""Run the PyTorch causal backend against an R estimation-panel artifact."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.causal.gpu.contracts import (  # noqa: E402
    GPU_IMPLEMENTATION_VERSION,
    SHADOW_SCHEMA,
)
from urban_intervention.causal.gpu.gsc import GSCConfig, fit_gsc  # noqa: E402
from urban_intervention.causal.gpu.io import (  # noqa: E402
    compare_counterfactuals,
    compare_inference_paths,
    cv_contract_artifact_paths,
    cv_contract_manifest_path,
    load_cv_contract_manifest,
    load_estimation_panel,
    load_gsc_cv_folds,
    load_mc_cv_contract,
)
from urban_intervention.causal.gpu.matrix_completion import (  # noqa: E402
    MatrixCompletionConfig,
    fit_matrix_completion,
)
from urban_intervention.causal.gpu.provenance import (  # noqa: E402
    estimator_code_fingerprint,
    estimator_source_files,
    fingerprint_files,
    python_environment,
)
from urban_intervention.causal.gpu.runtime import RuntimeConfig, TorchRuntime  # noqa: E402


def reference_tuning(labels_path: Path, estimator: str) -> float | int:
    labels = pd.read_parquet(labels_path)
    column = "selected_factors" if estimator == "gsc" else "mc_lambda"
    if column not in labels:
        raise ValueError(f"reference labels lack tuning column {column}")
    values = pd.to_numeric(labels[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(f"reference tuning column {column} is not one finite value")
    return int(values[0]) if estimator == "gsc" else float(values[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", choices=("gsc", "mc"), required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--reference-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--tuning", choices=("reference", "gpu"), default="reference")
    parser.add_argument(
        "--contract-backend",
        choices=("any", "r_fect_2.4.5", "python_native"),
        default="any",
    )
    parser.add_argument("--inference", choices=("none", "jackknife"), default="none")
    parser.add_argument(
        "--gsc-bootstrap-mode",
        choices=("none", "auto", "reference_empirical", "reference_ar"),
        default="none",
    )
    parser.add_argument("--gsc-n-bootstrap", type=int, default=0)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-5)
    parser.add_argument("--relative-tolerance", type=float, default=1e-5)
    parser.add_argument("--inference-relative-rmse-tolerance", type=float)
    parser.add_argument("--minimum-ci-zero-agreement", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tuning == "gpu" and (args.max_iter != 5000 or args.tol != 1e-5):
        raise ValueError("strict GPU tuning requires the frozen fect max_iter=5000 and tol=1e-5")
    loaded = load_estimation_panel(args.panel, args.estimator)
    runtime = TorchRuntime(
        RuntimeConfig(device=args.device, dtype=args.dtype, deterministic=True, allow_tf32=False)
    )
    tuning = (
        reference_tuning(args.reference_labels, args.estimator)
        if args.tuning == "reference"
        else None
    )
    contract_metadata = (
        load_cv_contract_manifest(args.panel.parent, args.estimator)
        if args.tuning == "gpu"
        else {"schema": "not_used", "contract_backend": "not_used", "cv_seed": "not_used"}
    )
    if args.tuning == "gpu" and args.contract_backend != "any" and (
        contract_metadata["contract_backend"] != args.contract_backend
    ):
        raise ValueError(
            "GPU tuning contract backend mismatch: required "
            f"{args.contract_backend}, received {contract_metadata['contract_backend']}"
        )
    started = time.perf_counter()
    if args.estimator == "gsc":
        (folds_path,) = cv_contract_artifact_paths(args.panel.parent, "gsc")
        if args.tuning == "gpu" and not folds_path.is_file():
            raise ValueError("GPU GSC tuning requires an R-exported rolling-CV contract")
        gsc_folds = load_gsc_cv_folds(folds_path, loaded) if args.tuning == "gpu" else None
        gsc_result = fit_gsc(
            loaded.panel,
            config=GSCConfig(
                fixed_rank=tuning if isinstance(tuning, int) else None,
                max_iter=args.max_iter,
                tol=args.tol,
                bootstrap_mode=args.gsc_bootstrap_mode,
                n_bootstrap=args.gsc_n_bootstrap,
                inference_batch_size=args.inference_batch_size,
            ),
            cv_folds=gsc_folds,
            runtime=runtime,
        )
        if not gsc_result.converged:
            raise RuntimeError("GPU GSC final refit did not converge")
        counterfactual = gsc_result.counterfactual
        selected_tuning: float | int = gsc_result.selected_rank
        converged = gsc_result.converged
        iterations = gsc_result.iterations
        cv_mean_mspe: dict[str, float] = {
            str(key): value for key, value in gsc_result.cv_mean_mspe.items()
        }
        cv_se_mspe: dict[str, float] = {
            str(key): value for key, value in gsc_result.cv_se_mspe.items()
        }
        inference_method = gsc_result.provenance.numerical_policy["bootstrap_mode"]
        inference_draws = gsc_result.bootstrap_draws
        inference_result = gsc_result.inference
        estimator_numerical_policy = gsc_result.provenance.numerical_policy
    else:
        folds_path, lambda_path = cv_contract_artifact_paths(args.panel.parent, "mc")
        if args.tuning == "gpu" and (not folds_path.is_file() or not lambda_path.is_file()):
            raise ValueError("GPU MC tuning requires R-exported CV folds and lambda grid")
        if args.tuning == "gpu":
            mc_folds, lambda_grid = load_mc_cv_contract(args.panel.parent, loaded)
        else:
            mc_folds, lambda_grid = None, None
        mc_result = fit_matrix_completion(
            loaded.panel,
            config=MatrixCompletionConfig(
                fixed_lambda=float(tuning) if tuning is not None else None,
                max_iter=args.max_iter,
                tol=args.tol,
                inference=args.inference,
                inference_batch_size=args.inference_batch_size,
            ),
            cv_folds=mc_folds,
            lambda_grid=lambda_grid,
            runtime=runtime,
        )
        if not mc_result.converged:
            raise RuntimeError("GPU MC final refit did not converge")
        counterfactual = mc_result.counterfactual
        selected_tuning = mc_result.selected_lambda
        converged = mc_result.converged
        iterations = mc_result.iterations
        cv_mean_mspe = {str(key): value for key, value in mc_result.cv_mean_mspe.items()}
        cv_se_mspe = {str(key): value for key, value in mc_result.cv_se_mspe.items()}
        inference_method = mc_result.provenance.numerical_policy["inference"]
        inference_draws = mc_result.jackknife_effect_draws
        inference_result = mc_result.inference
        estimator_numerical_policy = mc_result.provenance.numerical_policy
    elapsed = time.perf_counter() - started
    comparison, parity = compare_counterfactuals(
        loaded.periods,
        counterfactual,
        args.reference_labels,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(args.output / "counterfactual_comparison.parquet", index=False)
    if inference_result is not None:
        inference_frame = inference_result.to_frame(loaded.periods)
        inference_frame["gpu_effect"] = inference_frame["effect"]
        inference_frame["gpu_standard_error"] = inference_frame["standard_error"]
        inference_frame.to_parquet(args.output / "gpu_inference.parquet", index=False)
    if inference_draws is not None:
        draw_frame = pd.DataFrame(inference_draws, columns=loaded.periods)
        draw_frame.insert(0, "replicate", range(len(draw_frame)))
        draw_frame.to_parquet(args.output / "gpu_inference_draws.parquet", index=False)
    if inference_result is not None:
        inference_tolerance = (
            args.inference_relative_rmse_tolerance
            if args.inference_relative_rmse_tolerance is not None
            else 0.35
            if args.estimator == "gsc"
            else 0.02
        )
        inference_comparison, inference_parity = compare_inference_paths(
            loaded.periods,
            inference_result.estimate,
            inference_result.standard_error,
            args.reference_labels,
            relative_rmse_tolerance=inference_tolerance,
            minimum_ci_zero_agreement=args.minimum_ci_zero_agreement,
        )
        inference_comparison.to_parquet(
            args.output / "inference_comparison.parquet", index=False
        )
    else:
        inference_parity = {
            "available": False,
            "passed": False,
            "reason": "formal inference was not requested",
        }
    manifest = {
        "schema": SHADOW_SCHEMA,
        "implementation_version": GPU_IMPLEMENTATION_VERSION,
        "code_fingerprint": estimator_code_fingerprint(args.estimator),
        "estimator": args.estimator,
        "backend": "pytorch",
        "mode": "shadow",
        "formal_eligible": False,
        "panel": str(args.panel.resolve()),
        "reference_labels": str(args.reference_labels.resolve()),
        "tuning_source": args.tuning,
        "contract_schema": contract_metadata["schema"],
        "contract_backend": contract_metadata["contract_backend"],
        "cv_seed": contract_metadata["cv_seed"],
        "selected_tuning": selected_tuning,
        "elapsed_seconds": elapsed,
        "converged": converged,
        "iterations": iterations,
        "estimator_config": {
            "max_iter": args.max_iter,
            "tol": args.tol,
            "gsc_bootstrap_mode": args.gsc_bootstrap_mode,
            "gsc_n_bootstrap": args.gsc_n_bootstrap,
            "mc_inference": args.inference,
            "inference_batch_size": args.inference_batch_size,
        },
        "estimator_numerical_policy": estimator_numerical_policy,
        "inference": {
            "method": (
                inference_result.method
                if inference_result is not None
                else inference_method
            ),
            "available": inference_result is not None,
            "formal_validated": bool(
                parity.get("passed") is True
                and inference_parity.get("passed") is True
            ),
            "parity": inference_parity,
        },
        "cv_mean_mspe": cv_mean_mspe,
        "cv_se_mspe": cv_se_mspe,
        "runtime": runtime.metadata(),
        "parity": parity,
        "source_fingerprints": fingerprint_files(
            [
                path
                for path in (
                    args.panel,
                    cv_contract_manifest_path(args.panel.parent),
                    folds_path,
                    lambda_path if args.estimator == "mc" else None,
                    args.reference_labels,
                    Path(__file__).resolve(),
                    *estimator_source_files(args.estimator),
                )
                if path is not None and path.is_file()
            ]
        ),
        "environment": python_environment(),
        "promotion_status": (
            "eligible_for_stratified_qualification_audit"
            if parity.get("passed") is True
            and inference_parity.get("passed") is True
            else "blocked_by_point_or_inference_parity"
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
