#!/usr/bin/env python3
"""Build and run one R-free formal GSC/MC family task."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from urban_intervention.causal.gpu.contracts import FORMAL_IMPLEMENTATION_VERSION  # noqa: E402
from urban_intervention.causal.gpu.formal_runner import (  # noqa: E402
    FormalRunRequest,
    run_formal_panel,
)
from urban_intervention.causal.gpu.gsc import GSCConfig, fit_gsc  # noqa: E402
from urban_intervention.causal.gpu.io import load_estimation_panel  # noqa: E402
from urban_intervention.causal.gpu.matrix_completion import (  # noqa: E402
    MatrixCompletionConfig,
    fit_matrix_completion,
)
from urban_intervention.causal.gpu.panel_builder import (  # noqa: E402
    OUTCOMES,
    PanelBuildRequest,
    build_estimation_panel,
)
from urban_intervention.causal.gpu.runtime import RuntimeConfig, TorchRuntime  # noqa: E402
from urban_intervention.causal.gpu.tuning_cache import (  # noqa: E402
    TUNING_CACHE_SCHEMA,
    load_tuning_cache,
    panel_tuning_signature,
    tuning_cache_path,
    write_tuning_cache,
)
from urban_intervention.data.paths import OUTPUT_COMPLETE_STAGING_DIR  # noqa: E402
from urban_intervention.utils import atomic_write_csv  # noqa: E402


def output_directory(
    metadata: dict[str, object],
    estimator: str,
    run_mode: str,
    transaction_count_threshold: int = 1,
) -> Path:
    monthly = str(metadata["frequency"]) == "monthly"
    cohort = str(metadata["opening_month"])[:7] if monthly else str(metadata["opening_month"])[:4]
    outcome = str(metadata["outcome"])
    order = int(metadata["treatment_order"])
    scope = str(metadata["donor_scope"])
    if estimator == "gsc":
        method_dir = "xu_gsc"
        signature = (
            "outcome_only_prepath"
            if scope == "same_city"
            else "outcome_only_prepath_all_city_standardized"
        )
    else:
        method_dir = "matrix_completion"
        signature = (
            "outcome_only_prepath_mc"
            if scope == "same_city"
            else "outcome_only_prepath_mc_all_city"
        )
    if run_mode != "production":
        signature = f"{signature}_{run_mode}"
    if (
        str(metadata["outcome_family"]) == "housing"
        and transaction_count_threshold != 1
    ):
        signature = f"{signature}_tx{transaction_count_threshold}"
    return (
        OUTPUT_COMPLETE_STAGING_DIR
        / method_dir
        / str(metadata["city_key"])
        / cohort
        / f"{outcome}_t{order:05d}"
        / signature
    )


def family_status_directory(
    metadata: dict[str, object],
    run_mode: str,
    transaction_count_threshold: int = 1,
) -> Path:
    monthly = str(metadata["frequency"]) == "monthly"
    cohort = str(metadata["opening_month"])[:7] if monthly else str(metadata["opening_month"])[:4]
    scope = str(metadata["donor_scope"])
    signature = (
        "outcome_only_prepath_mc"
        if scope == "same_city"
        else "outcome_only_prepath_mc_all_city"
    )
    if run_mode != "production":
        signature = f"{signature}_{run_mode}"
    if (
        str(metadata["outcome_family"]) == "housing"
        and transaction_count_threshold != 1
    ):
        signature = f"{signature}_tx{transaction_count_threshold}"
    return (
        OUTPUT_COMPLETE_STAGING_DIR
        / "matrix_completion_runs"
        / str(metadata["city_key"])
        / cohort
        / f"{metadata['outcome_family']}_t{int(metadata['treatment_order']):05d}"
        / signature
    )


def _resolve_tuning(
    panel: pd.DataFrame,
    estimator: str,
    *,
    device: str,
    run_mode: str,
) -> tuple[GSCConfig | None, MatrixCompletionConfig | None, dict[str, object]]:
    """Load or compute an estimator tuning choice under an exact panel signature."""
    if estimator == "gsc":
        selection_config = GSCConfig(bootstrap_mode="none", n_bootstrap=0, seed=20260723)
        tuning_contract = asdict(selection_config) | {"fixed_rank": None}
    else:
        selection_config = MatrixCompletionConfig(inference="none", seed=20260725)
        tuning_contract = asdict(selection_config) | {"fixed_lambda": None}
    signature = panel_tuning_signature(
        panel, estimator, tuning_contract=tuning_contract
    )
    cache_path = tuning_cache_path(OUTPUT_COMPLETE_STAGING_DIR, estimator, signature)
    cached = load_tuning_cache(cache_path, estimator=estimator, signature=signature)
    cache_hit = cached is not None
    if cached is None:
        runtime = TorchRuntime(
            RuntimeConfig(
                device=device,
                seed=20260723 if estimator == "gsc" else 20260725,
            )
        )
        loaded = load_estimation_panel(panel, estimator)
        if estimator == "gsc":
            selected = fit_gsc(loaded.panel, config=selection_config, runtime=runtime)
            selected_tuning: float | int = int(selected.selected_rank)
            cv_mean = selected.cv_mean_mspe
        else:
            selected = fit_matrix_completion(
                loaded.panel, config=selection_config, runtime=runtime
            )
            selected_tuning = float(selected.selected_lambda)
            cv_mean = selected.cv_mean_mspe
        if not selected.converged or not cv_mean:
            raise RuntimeError(f"{estimator.upper()} tuning fit did not converge completely")
        cached = {
            "schema": TUNING_CACHE_SCHEMA,
            "implementation_version": FORMAL_IMPLEMENTATION_VERSION,
            "estimator": estimator,
            "panel_signature": signature,
            "selected_tuning": selected_tuning,
            "cv_min_mspe": float(min(cv_mean.values())),
            "cv_mean_mspe": {str(key): float(value) for key, value in cv_mean.items()},
            "tuning_contract": tuning_contract,
        }
        write_tuning_cache(cache_path, cached)
    selected_value = float(cached["selected_tuning"])
    if estimator == "mc" and run_mode == "production" and (
        not math.isfinite(selected_value) or selected_value < 0
    ):
        raise RuntimeError(
            "production matrix completion requires a finite non-negative lambda"
        )
    if estimator == "gsc":
        formal_config = GSCConfig(
            fixed_rank=int(selected_value),
            bootstrap_mode="auto" if run_mode == "production" else "none",
            n_bootstrap=200 if run_mode == "production" else 0,
            seed=20260723,
        )
        gsc_config: GSCConfig | None = formal_config
        mc_config: MatrixCompletionConfig | None = None
    else:
        formal_mc = MatrixCompletionConfig(
            fixed_lambda=selected_value,
            inference="jackknife" if run_mode == "production" else "none",
            seed=20260725,
        )
        gsc_config = None
        mc_config = formal_mc
    metadata: dict[str, object] = {
        "cached_cv_min_mspe": float(cached["cv_min_mspe"]),
        "tuning_source": "content_addressed_cache" if cache_hit else "fresh_gpu_cv_then_cached",
        "tuning_cache_hit": cache_hit,
        "tuning_panel_signature": signature,
        "tuning_cache_path": str(cache_path),
    }
    return gsc_config, mc_config, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment-order", type=int, required=True)
    parser.add_argument("--outcome-family", choices=sorted(OUTCOMES), required=True)
    parser.add_argument("--outcome", choices=sum((list(value) for value in OUTCOMES.values()), []))
    parser.add_argument("--estimator", choices=("gsc", "mc"), required=True)
    parser.add_argument(
        "--donor-scope", choices=("same_city", "all_city_standardized"), default="same_city"
    )
    parser.add_argument("--anticipation-months", type=int, default=6)
    parser.add_argument("--price-measure", choices=("median", "hedonic"), default="median")
    parser.add_argument("--observation-window", type=int, default=1)
    parser.add_argument(
        "--transaction-count-threshold",
        type=int,
        default=1,
        help="Minimum transactions for each housing grid-month admitted to the panel.",
    )
    parser.add_argument("--max-gsc-cross-city-donors", type=int, default=50_000)
    parser.add_argument("--gsc-donor-sampling-seed", type=int, default=20260823)
    parser.add_argument("--run-mode", choices=("production", "preview"), default="production")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id", default=os.environ.get("MIT_CAUSAL_RUN_ID", ""))
    parser.add_argument(
        "--qualification-receipt",
        type=Path,
        default=(
            Path(os.environ["MIT_CAUSAL_QUALIFICATION_RECEIPT"])
            if os.environ.get("MIT_CAUSAL_QUALIFICATION_RECEIPT")
            else None
        ),
        help="Eligible R/Python parity audit receipt required for production runs.",
    )
    parser.add_argument(
        "--specification-fingerprint",
        default=os.environ.get("MIT_SPECIFICATION_FINGERPRINT", ""),
    )
    args = parser.parse_args()
    outcomes = (
        [args.outcome] if args.outcome else list(OUTCOMES[args.outcome_family])
    )
    if any(value not in OUTCOMES[args.outcome_family] for value in outcomes):
        parser.error("--outcome does not belong to --outcome-family")
    run_id = args.run_id or uuid.uuid4().hex
    statuses: list[dict[str, object]] = []
    last_metadata: dict[str, object] | None = None
    for outcome in outcomes:
        try:
            build_request = PanelBuildRequest(
                treatment_order=args.treatment_order,
                outcome_family=args.outcome_family,
                outcome=outcome,
                estimator=args.estimator,
                donor_scope=args.donor_scope,
                anticipation_months=args.anticipation_months,
                price_measure=args.price_measure,
                max_gsc_cross_city_donors=args.max_gsc_cross_city_donors,
                gsc_donor_sampling_seed=args.gsc_donor_sampling_seed,
                transaction_count_threshold=args.transaction_count_threshold,
            )
            built = build_estimation_panel(build_request)
            gsc_config, mc_config, tuning_metadata = _resolve_tuning(
                built.panel,
                args.estimator,
                device=args.device,
                run_mode=args.run_mode,
            )
            metadata = built.metadata | tuning_metadata
            last_metadata = metadata
            output = output_directory(
                metadata,
                args.estimator,
                args.run_mode,
                args.transaction_count_threshold,
            )
            output.mkdir(parents=True, exist_ok=True)
            panel_path = output / "estimation_panel.parquet"
            built.panel.to_parquet(panel_path, index=False, compression="zstd")
            (output / "panel_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            run_request = FormalRunRequest(
                estimator=args.estimator,
                output_directory=output,
                treatment_order=args.treatment_order,
                city_key=str(metadata["city_key"]),
                grid_id=str(metadata["grid_id"]),
                opening_month=str(metadata["opening_month"]),
                outcome_family=args.outcome_family,
                outcome=outcome,
                donor_scope=args.donor_scope,
                run_mode=args.run_mode,
                run_id=run_id,
                specification_fingerprint=args.specification_fingerprint,
                price_measure=args.price_measure,
                observation_window=args.observation_window,
                transaction_count_threshold=args.transaction_count_threshold,
                device=args.device,
                qualification_receipt=args.qualification_receipt,
                qualification_receipt_sha256=os.environ.get(
                    "MIT_CAUSAL_QUALIFICATION_PREVALIDATED_SHA256", ""
                ),
            )
            result = run_formal_panel(
                built.panel,
                run_request,
                panel_metadata=metadata,
                gsc_config=gsc_config,
                mc_config=mc_config,
            )
            statuses.append(
                {
                    "outcome": outcome,
                    "status": "success",
                    "failure_reason": None,
                    "run_id": run_id,
                    "labels_path": str(result.labels_path),
                }
            )
            print(f"Completed Python {args.estimator.upper()} for {outcome}: {output}")
        except Exception as error:  # family runner records bounded outcome failures
            statuses.append(
                {
                    "outcome": outcome,
                    "status": "failed",
                    "failure_reason": str(error),
                    "run_id": run_id,
                    "labels_path": None,
                }
            )
            print(f"Python {args.estimator.upper()} failed for {outcome}: {error}", file=sys.stderr)
            if args.estimator == "gsc":
                break
    status = pd.DataFrame(statuses)
    if args.estimator == "mc" and last_metadata is not None:
        status_dir = family_status_directory(
            last_metadata, args.run_mode, args.transaction_count_threshold
        )
        status_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(status, status_dir / "outcome_status.csv")
    success = int(status["status"].eq("success").sum()) if not status.empty else 0
    if args.estimator == "gsc":
        return 0 if success == len(outcomes) else 1
    return 0 if success > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
