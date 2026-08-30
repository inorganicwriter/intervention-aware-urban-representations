"""Spawn-safe worker entry points for the multi-GPU shadow queue."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from urban_intervention.utils import atomic_write_json

from .contracts import GPU_IMPLEMENTATION_VERSION, SHADOW_SCHEMA
from .fixed_control import fixed_control_labels
from .gsc import GSCConfig, fit_gsc
from .io import (
    compare_counterfactuals,
    compare_inference_paths,
    cv_contract_artifact_paths,
    cv_contract_manifest_path,
    load_cv_contract_manifest,
    load_estimation_panel,
    load_gsc_cv_folds,
    load_mc_cv_contract,
)
from .matching import MatchingConfig, fit_matching
from .matching_io import (
    compare_matching_labels,
    compare_matching_result,
    load_matching_artifacts,
    matching_result_frames,
)
from .matrix_completion import MatrixCompletionConfig, fit_matrix_completion
from .provenance import (
    estimator_code_fingerprint,
    estimator_source_files,
    fingerprint_files,
    python_environment,
)
from .runtime import RuntimeConfig, TorchRuntime


def _runtime(payload: dict[str, Any], cache: dict[object, Any]) -> TorchRuntime:
    key = (
        "runtime",
        payload.get("dtype", "float64"),
        int(payload.get("chunk_size", 65_536)),
        float(payload.get("memory_fraction", 0.85)),
        int(payload.get("seed", 20260723)),
    )
    if key not in cache:
        cache[key] = TorchRuntime(
            RuntimeConfig(
                device="auto",
                dtype=str(key[1]),
                deterministic=True,
                allow_tf32=False,
                chunk_size=int(key[2]),
                memory_fraction=float(key[3]),
                seed=int(key[4]),
            )
        )
    return cache[key]


def _reference_tuning(labels_path: Path, estimator: str) -> float | int:
    import pandas as pd

    labels = pd.read_parquet(labels_path)
    column = "selected_factors" if estimator == "gsc" else "mc_lambda"
    values = pd.to_numeric(labels[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(f"reference tuning column {column} is not one finite value")
    return int(values[0]) if estimator == "gsc" else float(values[0])


def run_shadow_task(payload: dict[str, Any], cache: dict[object, Any]) -> dict[str, Any]:
    """Execute one Matching/GSC/MC task on the worker's visible GPU."""
    estimator = str(payload["estimator"])
    output = Path(payload["output"])
    runtime = _runtime(payload, cache)
    started = time.perf_counter()
    if estimator == "matching":
        input_path = Path(payload["input"])
        cache_key = ("matching", str(input_path.resolve()))
        artifacts = cache.get(cache_key)
        if artifacts is None:
            artifacts = load_matching_artifacts(input_path)
            cache[cache_key] = artifacts
        matching_result = fit_matching(
            artifacts.data,
            config=MatchingConfig(
                candidates=int(artifacts.metadata["matching_candidates"]),
                placebo_sample=int(artifacts.metadata["placebo_sample"]),
                placebo_quantile=float(artifacts.metadata["placebo_quantile"]),
                chunk_size=runtime.config.chunk_size,
            ),
            runtime=runtime,
        )
        if artifacts.reference_candidates is not None:
            parity = compare_matching_result(
                artifacts,
                matching_result,
                absolute_tolerance=float(payload.get("absolute_tolerance", 1e-9)),
                relative_tolerance=float(payload.get("relative_tolerance", 1e-9)),
            )
        else:
            parity = {"available": False, "passed": None}
        label_parity: dict[str, Any] = {
            "available": False,
            "passed": False,
            "reason": "R final matching labels were not supplied",
        }
        output.mkdir(parents=True, exist_ok=True)
        if artifacts.reference_labels is not None:
            selected_control = artifacts.donor_ids[matching_result.selected_index]
            control_city, separator, control_grid = selected_control.partition("::")
            if not separator:
                raise ValueError("selected matching control lacks city::grid identity")
            families = [
                value
                for value in str(artifacts.metadata.get("active_families", "")).split("+")
                if value
            ]
            if not families:
                raise ValueError("matching reference lacks active_families metadata")
            python_labels = pd.concat(
                [
                    fixed_control_labels(
                        int(artifacts.metadata["treatment_order"]),
                        control_city,
                        control_grid,
                        family,
                        window=int(artifacts.metadata.get("reference_label_window", 1)),
                        price_measure=str(
                            artifacts.metadata.get("reference_label_price_measure", "median")
                        ),
                    )
                    for family in families
                ],
                ignore_index=True,
            )
            label_comparison, label_parity = compare_matching_labels(
                artifacts.reference_labels,
                python_labels,
                absolute_tolerance=float(payload.get("absolute_tolerance", 1e-9)),
                relative_tolerance=float(payload.get("relative_tolerance", 1e-9)),
            )
            label_comparison.to_parquet(
                output / "matching_label_comparison.parquet", index=False
            )
        candidates, selection = matching_result_frames(artifacts, matching_result)
        candidates.to_csv(output / "gpu_candidates.csv", index=False, encoding="utf-8-sig")
        selection.to_csv(output / "gpu_selection.csv", index=False, encoding="utf-8-sig")
        manifest = {
            "schema": SHADOW_SCHEMA,
            "implementation_version": GPU_IMPLEMENTATION_VERSION,
            "code_fingerprint": estimator_code_fingerprint("matching"),
            "estimator": estimator,
            "backend": "pytorch",
            "mode": "shadow",
            "formal_eligible": False,
            "input": str(input_path.resolve()),
            "elapsed_seconds": time.perf_counter() - started,
            "candidate_count": len(artifacts.donor_ids),
            "selected_control": artifacts.donor_ids[matching_result.selected_index],
            "quality_passed": matching_result.quality_passed,
            "qualification_requested": bool(payload.get("formal_qualification", False)),
            "qualification_passed": bool(
                parity.get("passed") is True
                and label_parity.get("passed") is True
                and matching_result.quality_passed is True
            ),
            "runtime": runtime.metadata(),
            "estimator_numerical_policy": matching_result.provenance.numerical_policy,
            "parity": parity,
            "label_parity": label_parity,
            "source_fingerprints": fingerprint_files(
                [
                    path
                    for path in (
                        input_path / "matching_input.parquet",
                        input_path / "metadata.csv",
                        input_path / "reference_candidates.csv",
                        input_path / "reference_selection.csv",
                        input_path / "reference_labels.parquet",
                        Path(__file__).resolve(),
                        *estimator_source_files("matching"),
                    )
                    if path.is_file()
                ]
            ),
            "environment": python_environment(),
            "promotion_status": (
                "eligible_for_stratified_qualification_audit"
                if parity.get("passed") is True
                and label_parity.get("passed") is True
                and matching_result.quality_passed is True
                else "blocked_by_matching_design_label_parity_or_quality"
            ),
        }
    elif estimator in {"gsc", "mc"}:
        panel_path = Path(payload["panel"])
        tuning_source = str(payload.get("tuning_source", "reference"))
        if tuning_source == "gpu" and (
            int(payload.get("max_iter", 5000)) != 5000
            or float(payload.get("tol", 1e-5)) != 1e-5
        ):
            raise ValueError(
                "strict GPU tuning requires the frozen fect max_iter=5000 and tol=1e-5"
            )
        labels_value = payload.get("reference_labels")
        labels_path = Path(labels_value) if labels_value else None
        cache_key = (estimator, str(panel_path.resolve()))
        loaded = cache.get(cache_key)
        if loaded is None:
            loaded = load_estimation_panel(panel_path, estimator)
            cache[cache_key] = loaded
        if tuning_source == "reference":
            if labels_path is None:
                raise ValueError("reference tuning requires reference labels")
            tuning: float | int | None = _reference_tuning(labels_path, estimator)
            contract_metadata: dict[str, str] = {
                "schema": "not_used",
                "contract_backend": "not_used",
                "cv_seed": "not_used",
            }
        elif tuning_source == "gpu":
            tuning = None
            contract_metadata = load_cv_contract_manifest(panel_path.parent, estimator)
            required_contract_backend = str(payload.get("contract_backend", "any"))
            if required_contract_backend != "any" and (
                contract_metadata["contract_backend"] != required_contract_backend
            ):
                raise ValueError(
                    "GPU tuning contract backend mismatch: required "
                    f"{required_contract_backend}, received "
                    f"{contract_metadata['contract_backend']}"
                )
        else:
            raise ValueError(f"unsupported tuning source: {tuning_source}")
        if estimator == "gsc":
            (folds_path,) = cv_contract_artifact_paths(panel_path.parent, "gsc")
            if tuning_source == "gpu" and not folds_path.is_file():
                raise ValueError("GPU GSC tuning requires an R-exported rolling-CV contract")
            gsc_folds = (
                load_gsc_cv_folds(folds_path, loaded) if tuning_source == "gpu" else None
            )
            gsc_result = fit_gsc(
                loaded.panel,
                config=GSCConfig(
                    fixed_rank=int(tuning) if tuning is not None else None,
                    max_iter=int(payload.get("max_iter", 5000)),
                    tol=float(payload.get("tol", 1e-5)),
                    bootstrap_mode=str(payload.get("gsc_bootstrap_mode", "none")),  # type: ignore[arg-type]
                    n_bootstrap=int(payload.get("gsc_n_bootstrap", 0)),
                    seed=int(payload.get("seed", 20260723)),
                    inference_batch_size=int(payload.get("inference_batch_size", 16)),
                ),
                cv_folds=gsc_folds,
                runtime=runtime,
            )
            if not gsc_result.converged:
                raise RuntimeError("GPU GSC final refit did not converge")
            selected_tuning: float | int = gsc_result.selected_rank
            counterfactual = gsc_result.counterfactual
            effect = gsc_result.effect
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
            folds_path, lambda_path = cv_contract_artifact_paths(panel_path.parent, "mc")
            if tuning_source == "gpu" and (not folds_path.is_file() or not lambda_path.is_file()):
                raise ValueError("GPU MC tuning requires R-exported CV folds and lambda grid")
            if tuning_source == "gpu":
                mc_folds, lambda_grid = load_mc_cv_contract(panel_path.parent, loaded)
            else:
                mc_folds, lambda_grid = None, None
            mc_result = fit_matrix_completion(
                loaded.panel,
                config=MatrixCompletionConfig(
                    fixed_lambda=float(tuning) if tuning is not None else None,
                    max_iter=int(payload.get("max_iter", 5000)),
                    tol=float(payload.get("tol", 1e-5)),
                    inference=str(payload.get("mc_inference", "none")),  # type: ignore[arg-type]
                    batch_inference=True,
                    seed=int(payload.get("seed", 20260725)),
                    inference_batch_size=int(payload.get("inference_batch_size", 16)),
                ),
                cv_folds=mc_folds,
                lambda_grid=lambda_grid,
                runtime=runtime,
            )
            if not mc_result.converged:
                raise RuntimeError("GPU MC final refit did not converge")
            selected_tuning = mc_result.selected_lambda
            counterfactual = mc_result.counterfactual
            effect = mc_result.effect
            converged = mc_result.converged
            iterations = mc_result.iterations
            cv_mean_mspe = {
                str(key): value for key, value in mc_result.cv_mean_mspe.items()
            }
            cv_se_mspe = {
                str(key): value for key, value in mc_result.cv_se_mspe.items()
            }
            inference_method = mc_result.provenance.numerical_policy["inference"]
            inference_draws = mc_result.jackknife_effect_draws
            inference_result = mc_result.inference
            estimator_numerical_policy = mc_result.provenance.numerical_policy
        output.mkdir(parents=True, exist_ok=True)
        if labels_path is not None:
            comparison, parity = compare_counterfactuals(
                loaded.periods,
                counterfactual,
                labels_path,
                absolute_tolerance=float(payload.get("absolute_tolerance", 1e-5)),
                relative_tolerance=float(payload.get("relative_tolerance", 1e-5)),
            )
            comparison.to_parquet(output / "counterfactual_comparison.parquet", index=False)
        else:
            parity = {"available": False, "passed": None}
        pd.DataFrame(
            {
                "period": loaded.periods,
                "gpu_counterfactual": counterfactual,
                "gpu_effect": effect,
            }
        ).to_parquet(output / "gpu_estimates.parquet", index=False)
        if inference_result is not None:
            inference_frame = inference_result.to_frame(loaded.periods)
            inference_frame["gpu_effect"] = inference_frame["effect"]
            inference_frame["gpu_standard_error"] = inference_frame["standard_error"]
            inference_frame.to_parquet(output / "gpu_inference.parquet", index=False)
        if inference_draws is not None:
            draw_frame = pd.DataFrame(inference_draws, columns=loaded.periods)
            draw_frame.insert(0, "replicate", range(len(draw_frame)))
            draw_frame.to_parquet(output / "gpu_inference_draws.parquet", index=False)
        if inference_result is not None and labels_path is not None:
            inference_tolerance = float(
                payload.get(
                    "inference_relative_rmse_tolerance",
                    0.35 if estimator == "gsc" else 0.02,
                )
            )
            inference_comparison, inference_parity = compare_inference_paths(
                loaded.periods,
                inference_result.estimate,
                inference_result.standard_error,
                labels_path,
                relative_rmse_tolerance=inference_tolerance,
                minimum_ci_zero_agreement=float(
                    payload.get("minimum_ci_zero_agreement", 0.9)
                ),
            )
            inference_comparison.to_parquet(
                output / "inference_comparison.parquet", index=False
            )
        else:
            inference_parity = {
                "available": False,
                "passed": False,
                "reason": "formal inference or R reference labels were not supplied",
            }
        manifest = {
            "schema": SHADOW_SCHEMA,
            "implementation_version": GPU_IMPLEMENTATION_VERSION,
            "code_fingerprint": estimator_code_fingerprint(estimator),
            "estimator": estimator,
            "backend": "pytorch",
            "mode": "shadow",
            "formal_eligible": False,
            "panel": str(panel_path.resolve()),
            "reference_labels": str(labels_path.resolve()) if labels_path else None,
            "tuning_source": tuning_source,
            "contract_schema": contract_metadata["schema"],
            "contract_backend": contract_metadata["contract_backend"],
            "cv_seed": contract_metadata["cv_seed"],
            "selected_tuning": selected_tuning,
            "elapsed_seconds": time.perf_counter() - started,
            "converged": converged,
            "iterations": iterations,
            "estimator_config": {
                "max_iter": int(payload.get("max_iter", 5000)),
                "tol": float(payload.get("tol", 1e-5)),
                "gsc_bootstrap_mode": str(payload.get("gsc_bootstrap_mode", "none")),
                "gsc_n_bootstrap": int(payload.get("gsc_n_bootstrap", 0)),
                "mc_inference": str(payload.get("mc_inference", "none")),
                "inference_batch_size": int(payload.get("inference_batch_size", 16)),
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
            "qualification_requested": bool(payload.get("formal_qualification", False)),
            "qualification_passed": bool(
                parity.get("passed") is True
                and inference_parity.get("passed") is True
            ),
            "cv_mean_mspe": cv_mean_mspe,
            "cv_se_mspe": cv_se_mspe,
            "runtime": runtime.metadata(),
            "parity": parity,
            "source_fingerprints": fingerprint_files(
                [
                    path
                    for path in (
                        panel_path,
                        cv_contract_manifest_path(panel_path.parent),
                        folds_path,
                        lambda_path if estimator == "mc" else None,
                        labels_path,
                        Path(__file__).resolve(),
                        *estimator_source_files(estimator),
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
    else:
        raise ValueError(f"unsupported estimator: {estimator}")
    atomic_write_json(manifest, output / "manifest.json")
    return manifest
