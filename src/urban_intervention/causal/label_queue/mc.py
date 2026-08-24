"""Behavior-preserving component of the modular causal label queue."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from urban_intervention.utils import atomic_write_json
from urban_intervention.utils import sha256_file as file_sha256

from .estimators import (
    normalized_python_labels,
    python_estimator_command,
    validate_python_estimator_manifest,
)
from .runtime import (
    HORIZONS,
    OUTCOMES,
    R_SCRIPT,
    ROOT,
    STAGING,
    r_script,
    settings,
)
from .state import (
    _cohort_year,
    effective_price_measure,
    new_run_id,
    read_estimator_manifest,
    run,
    specification_fingerprint,
)

atomic_json = partial(atomic_write_json, default=str)


def mc_output(row: pd.Series, outcome: str, donor_scope: str = "same_city") -> Path:
    cohort = (
        str(row.opening_month)
        if row.outcome_family in {"housing", "viirs"}
        else _cohort_year(row.opening_month)
    )
    tag = f"{outcome}_t{int(row.treatment_order):05d}"
    signature = (
        "outcome_only_prepath_mc"
        if donor_scope == "same_city"
        else "outcome_only_prepath_mc_all_city"
    )
    if settings.run_mode != "production":
        signature = f"{signature}_{settings.run_mode}"
    if str(row.outcome_family) == "housing" and settings.transaction_count_threshold != 1:
        signature = f"{signature}_tx{settings.transaction_count_threshold}"
    return STAGING / "matrix_completion" / str(row.city_key) / cohort / tag / signature


def mc_family_run_output(row: pd.Series, donor_scope: str = "same_city") -> Path:
    cohort = (
        str(row.opening_month)
        if row.outcome_family in {"housing", "viirs"}
        else _cohort_year(row.opening_month)
    )
    tag = f"{row.outcome_family}_t{int(row.treatment_order):05d}"
    signature = (
        "outcome_only_prepath_mc"
        if donor_scope == "same_city"
        else "outcome_only_prepath_mc_all_city"
    )
    if settings.run_mode != "production":
        signature = f"{signature}_{settings.run_mode}"
    if str(row.outcome_family) == "housing" and settings.transaction_count_threshold != 1:
        signature = f"{signature}_tx{settings.transaction_count_threshold}"
    return STAGING / "matrix_completion_runs" / str(row.city_key) / cohort / tag / signature


def run_python_mc_scope(
    row: pd.Series, donor_scope: str
) -> tuple[bool, list[pd.DataFrame], dict[str, object]]:
    family = str(row.outcome_family)
    run_id = new_run_id()
    command, environment = python_estimator_command(row, "mc", donor_scope, run_id)
    completed = run(command, environment)
    status_path = mc_family_run_output(row, donor_scope) / "outcome_status.csv"
    if completed.returncode != 0 and not status_path.exists():
        return (
            False,
            [],
            {
                "reason": "python_mc_runtime_or_support_failure",
                "backend": "python_gpu",
                "log": completed.stdout[-4000:],
            },
        )
    if not status_path.exists():
        return (
            False,
            [],
            {
                "reason": "python_mc_family_status_missing",
                "log": completed.stdout[-4000:],
            },
        )
    status = pd.read_csv(status_path)
    required = {"outcome", "status", "failure_reason", "run_id"}
    if not required.issubset(status.columns) or set(status["run_id"].astype(str)) != {run_id}:
        return (
            False,
            [],
            {
                "reason": "python_mc_family_status_malformed_or_stale",
                "path": str(status_path),
            },
        )
    selected_method = (
        "athey_2021_mc_same_city"
        if donor_scope == "same_city"
        else "athey_2021_mc_all_city_standardized"
    )
    labels: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    manifests: list[dict[str, object]] = []
    for outcome in OUTCOMES[family]:
        outcome_rows = status.loc[status["outcome"].astype(str).eq(outcome)]
        if len(outcome_rows) != 1 or str(outcome_rows.iloc[0]["status"]) != "success":
            reason = outcome_rows.iloc[0].get("failure_reason") if len(outcome_rows) else None
            failures[outcome] = str(reason or "python_mc_outcome_failed")
            continue
        path = mc_output(row, outcome, donor_scope) / "causal_response_labels.parquet"
        manifest_path = path.parent / "manifest.csv"
        if not path.exists() or not manifest_path.exists():
            failures[outcome] = "python_mc_success_record_lacks_manifest_or_labels"
            continue
        values = read_estimator_manifest(manifest_path)
        valid, selected_lambda, cv_min_mspe = validate_python_estimator_manifest(
            values,
            row,
            estimator="mc",
            outcome=outcome,
            donor_scope=donor_scope,
            run_id=run_id,
        )
        if not valid or selected_lambda < 0 or values.get("labels_sha256") != file_sha256(path):
            failures[outcome] = "python_mc_manifest_does_not_prove_current_run"
            continue
        raw = pq.read_table(path).to_pandas()
        raw = raw.loc[raw["event_time"].isin(HORIZONS[family])].copy()
        actual = set(pd.to_numeric(raw["event_time"], errors="coerce").dropna().astype(int))
        if actual != set(HORIZONS[family]):
            failures[outcome] = "python_mc_outcome_horizon_grid_incomplete"
            continue
        if raw.empty or not bool(raw["label_available"].fillna(False).any()):
            failures[outcome] = "python_mc_has_no_available_target_horizon"
            continue
        labels.append(normalized_python_labels(raw, row, selected_method))
        manifests.append(
            {
                "path": str(manifest_path.relative_to(ROOT)),
                "sha256": file_sha256(manifest_path),
                "labels_sha256": file_sha256(path),
                "selected_lambda": selected_lambda,
                "cv_min_mspe": cv_min_mspe,
                "run_id": run_id,
            }
        )
    if not labels:
        return (
            False,
            [],
            {
                "reason": "python_mc_no_outcome_produced_available_labels",
                "backend": "python_gpu",
                "outcome_failures": failures,
                "log": completed.stdout[-4000:],
            },
        )
    return (
        True,
        labels,
        {
            "selected_method": selected_method,
            "donor_scope": donor_scope,
            "backend": "python_gpu",
            "run_id": run_id,
            "estimator_manifests": manifests,
            "outcome_failures": failures,
            "outcome_status": str(status_path.relative_to(ROOT)),
            "outcome_status_sha256": file_sha256(status_path),
            "log": completed.stdout,
        },
    )


def run_mc_scope(
    row: pd.Series, donor_scope: str
) -> tuple[bool, list[pd.DataFrame], dict[str, object]]:
    if settings.estimator_backend == "python_gpu":
        return run_python_mc_scope(row, donor_scope)
    family = str(row.outcome_family)
    frequency = "monthly" if family in {"housing", "viirs"} else "annual"
    cohort = str(row.opening_month) if frequency == "monthly" else _cohort_year(row.opening_month)
    run_id = new_run_id()
    completed = run(
        [
            str(R_SCRIPT),
            str(r_script("run_complete_mc.R")),
            str(row.city_key),
            cohort,
            family,
            "auto",
            frequency,
            str(settings.anticipation_months),
            str(int(row.treatment_order)),
            donor_scope,
            settings.run_mode,
            effective_price_measure(row),
            str(settings.label_window),
        ],
        {
            "MIT_CAUSAL_RUN_ID": run_id,
            "MIT_SPECIFICATION_FINGERPRINT": specification_fingerprint(row),
        },
    )
    status_path = mc_family_run_output(row, donor_scope) / "outcome_status.csv"
    if completed.returncode != 0:
        return (
            False,
            [],
            {
                "reason": "mc_runtime_or_support_failure",
                "log": completed.stdout[-4000:],
                "outcome_status": str(status_path.relative_to(ROOT))
                if status_path.exists()
                else None,
            },
        )
    if not status_path.exists():
        return False, [], {"reason": "mc_family_status_missing", "log": completed.stdout[-4000:]}
    outcome_status = pd.read_csv(status_path)
    required_status = {"outcome", "status", "failure_reason", "run_id"}
    if not required_status.issubset(outcome_status.columns):
        return False, [], {"reason": "mc_family_status_malformed", "path": str(status_path)}
    if (
        len(outcome_status) != len(OUTCOMES[family])
        or outcome_status["outcome"].astype(str).duplicated().any()
        or set(outcome_status["outcome"].astype(str)) != set(OUTCOMES[family])
    ):
        return False, [], {"reason": "mc_family_status_outcome_mismatch", "path": str(status_path)}
    if set(outcome_status["run_id"].astype(str)) != {run_id}:
        return (
            False,
            [],
            {
                "reason": "mc_family_status_is_not_from_current_run",
                "path": str(status_path),
                "run_id": run_id,
            },
        )
    labels: list[pd.DataFrame] = []
    estimator_manifests: list[dict[str, object]] = []
    outcome_failures: dict[str, str] = {}
    selected_method = (
        "athey_2021_mc_same_city"
        if donor_scope == "same_city"
        else "athey_2021_mc_all_city_standardized"
    )
    for outcome in OUTCOMES[family]:
        outcome_record = outcome_status.loc[outcome_status["outcome"].astype(str) == outcome].iloc[
            0
        ]
        if str(outcome_record["status"]) != "success":
            reason = outcome_record.get("failure_reason")
            outcome_failures[outcome] = (
                "mc_outcome_failed" if pd.isna(reason) or not str(reason).strip() else str(reason)
            )
            continue
        path = mc_output(row, outcome, donor_scope) / "causal_response_labels.parquet"
        estimator_manifest = path.parent / "manifest.csv"
        if not estimator_manifest.exists() or not path.exists():
            outcome_failures[outcome] = "mc_success_record_lacks_manifest_or_labels"
            continue
        manifest_values = read_estimator_manifest(estimator_manifest)
        try:
            selected_lambda = float(manifest_values.get("selected_lambda", "nan"))
            cv_min_mspe = float(manifest_values.get("cv_min_mspe", "nan"))
        except ValueError:
            selected_lambda = cv_min_mspe = float("nan")
        expected_production = settings.run_mode == "production"
        expected_nboots = "200" if expected_production else "0"
        if (
            manifest_values.get("schema")
            != "complete_published_estimators_v3_explicit_deterministic_contracts"
            or manifest_values.get("estimator") != "mc"
            or manifest_values.get("fitted_method") != "mc"
            or manifest_values.get("backend") != "fect"
            or manifest_values.get("force") != "two-way"
            or manifest_values.get("criterion") != "mspe"
            or manifest_values.get("nlambda") != "20"
            or manifest_values.get("min_T0") != "1"
            or manifest_values.get("se", "").upper() != "TRUE"
            or manifest_values.get("run_id") != run_id
            or manifest_values.get("CV", "").upper() != "TRUE"
            or manifest_values.get("cv_method") != "rolling"
            or manifest_values.get("cv_folds") != "20"
            or manifest_values.get("cv_prop") != "0.1"
            or manifest_values.get("cv_rule") != "1se"
            or manifest_values.get("cv_nobs") != "1"
            or manifest_values.get("cv_donut") != "0"
            or manifest_values.get("cv_buffer") != "0"
            or manifest_values.get("tol") != "1e-05"
            or manifest_values.get("max_iteration") != "5000"
            or manifest_values.get("two_stage_cv_inference", "").upper() != "TRUE"
            or manifest_values.get("inference_fit_CV", "").upper() != "FALSE"
            or manifest_values.get("run_mode") != settings.run_mode
            or manifest_values.get("production_eligible", "").upper()
            != ("TRUE" if expected_production else "FALSE")
            or manifest_values.get("specification_fingerprint") != specification_fingerprint(row)
            or manifest_values.get("price_measure") != effective_price_measure(row)
            or manifest_values.get("observation_window")
            != str(settings.label_window if frequency == "monthly" else 1)
            or manifest_values.get("inference") != "jackknife"
            or manifest_values.get("nboots") != expected_nboots
            or not math.isfinite(selected_lambda)
            or selected_lambda < 0
            or not math.isfinite(cv_min_mspe)
        ):
            outcome_failures[outcome] = "mc_manifest_does_not_prove_cross_validated_mc"
            continue
        estimator_manifests.append(
            {
                "path": str(estimator_manifest.relative_to(ROOT)),
                "sha256": file_sha256(estimator_manifest),
                "labels_sha256": file_sha256(path),
                "selected_lambda": selected_lambda,
                "cv_min_mspe": cv_min_mspe,
                "run_id": run_id,
            }
        )
        raw = pq.read_table(path).to_pandas()
        raw = raw.loc[raw["event_time"].isin(HORIZONS[family])].copy()
        if raw.empty or not bool(raw["label_available"].fillna(False).any()):
            outcome_failures[outcome] = "mc_has_no_available_target_horizon"
            continue
        expected_horizons = set(HORIZONS[family])
        actual_horizons = set(
            pd.to_numeric(raw["event_time"], errors="coerce").dropna().astype(int)
        )
        if actual_horizons != expected_horizons:
            outcome_failures[outcome] = "mc_outcome_horizon_grid_incomplete"
            continue
        normalized = pd.DataFrame(
            {
                "treatment_order": raw["treatment_order"].astype(int),
                "city_key": raw["city_key"],
                "grid_id": raw["grid_id"],
                "opening_month": str(row.opening_month),
                "outcome_family": family,
                "outcome": raw["outcome"],
                "event_time": raw["event_time"].astype(int),
                "specification_id": "main_a6_r1km",
                "specification_fingerprint": specification_fingerprint(row),
                "observed": raw["observed"],
                "counterfactual": raw["counterfactual"],
                "causal_response_label": raw["causal_response_label"],
                "transformed_scale": True,
                "method": selected_method,
                "label_available": raw["label_available"],
                "control_unit_key": pd.NA,
                "standard_error": raw.get("standard_error", pd.NA),
                "confidence_lower": raw.get("confidence_lower", pd.NA),
                "confidence_upper": raw.get("confidence_upper", pd.NA),
                "p_value": raw.get("p_value", pd.NA),
                "bootstrap_repetitions": raw.get("bootstrap_repetitions", 0),
                "uncertainty_source": raw.get("uncertainty_source", pd.NA),
                "pre_observed_periods": raw.get("pre_observed_periods", pd.NA),
                "pre_rmspe": raw.get("pre_rmspe", pd.NA),
                "mc_lambda": raw.get("mc_lambda", selected_lambda),
                "mc_cv_mspe": raw.get("mc_cv_mspe", cv_min_mspe),
            }
        )
        labels.append(normalized)
    if not labels:
        return (
            False,
            [],
            {
                "reason": "mc_no_outcome_produced_available_labels",
                "donor_scope": donor_scope,
                "outcome_failures": outcome_failures,
                "outcome_status": str(status_path.relative_to(ROOT)),
                "log": completed.stdout[-4000:],
            },
        )
    return (
        True,
        labels,
        {
            "selected_method": selected_method,
            "donor_scope": donor_scope,
            "run_id": run_id,
            "estimator_manifests": estimator_manifests,
            "outcome_failures": outcome_failures,
            "outcome_status": str(status_path.relative_to(ROOT)),
            "outcome_status_sha256": file_sha256(status_path),
            "log": completed.stdout,
        },
    )
