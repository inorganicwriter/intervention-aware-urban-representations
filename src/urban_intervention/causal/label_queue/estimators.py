"""Behavior-preserving component of the modular causal label queue."""

from __future__ import annotations

import math
import os
import sys
from functools import partial
from pathlib import Path

import pandas as pd

from urban_intervention.causal.gpu.contracts import (
    FORMAL_IMPLEMENTATION_VERSION,
    FORMAL_RESULT_SCHEMA,
)
from urban_intervention.utils import atomic_write_json

from .runtime import (
    ROOT,
    STAGING,
    settings,
)
from .state import (
    _cohort_year,
    effective_price_measure,
    specification_fingerprint,
)

atomic_json = partial(atomic_write_json, default=str)


def gsc_output(row: pd.Series, outcome: str, donor_scope: str = "same_city") -> Path:
    cohort = (
        str(row.opening_month)
        if row.outcome_family in {"housing", "viirs"}
        else _cohort_year(row.opening_month)
    )
    tag = f"{outcome}_t{int(row.treatment_order):05d}"
    signature = (
        "outcome_only_prepath"
        if donor_scope == "same_city"
        else "outcome_only_prepath_all_city_standardized"
    )
    if settings.run_mode != "production":
        signature = f"{signature}_{settings.run_mode}"
    if str(row.outcome_family) == "housing" and settings.transaction_count_threshold != 1:
        signature = f"{signature}_tx{settings.transaction_count_threshold}"
    return STAGING / "xu_gsc" / str(row.city_key) / cohort / tag / signature


def python_estimator_command(
    row: pd.Series, estimator: str, donor_scope: str, run_id: str
) -> tuple[list[str], dict[str, str]]:
    """Build the isolated Python formal-estimator subprocess contract."""
    command = [
        sys.executable,
        str(ROOT / "scripts" / "causal_python" / "run_formal_estimator.py"),
        "--treatment-order",
        str(int(row.treatment_order)),
        "--outcome-family",
        str(row.outcome_family),
        "--estimator",
        estimator,
        "--donor-scope",
        donor_scope,
        "--anticipation-months",
        str(settings.anticipation_months),
        "--price-measure",
        effective_price_measure(row),
        "--observation-window",
        str(settings.label_window if str(row.outcome_family) in {"housing", "viirs"} else 1),
        "--transaction-count-threshold",
        str(settings.transaction_count_threshold),
        "--max-gsc-cross-city-donors",
        str(settings.max_gsc_cross_city_donors),
        "--gsc-donor-sampling-seed",
        str(settings.gsc_donor_sampling_seed),
        "--run-mode",
        settings.run_mode,
        "--device",
        os.environ.get("MIT_CAUSAL_DEVICE", "auto"),
        "--run-id",
        run_id,
        "--specification-fingerprint",
        specification_fingerprint(row),
    ]
    if settings.qualification_receipt is not None:
        command.extend(["--qualification-receipt", str(settings.qualification_receipt)])
    environment = {
        "MIT_CAUSAL_RUN_ID": run_id,
        "MIT_SPECIFICATION_FINGERPRINT": specification_fingerprint(row),
    }
    prevalidated_sha256 = str(
        settings.qualification_proof.get("formal_qualification_receipt_sha256", "")
    )
    if prevalidated_sha256:
        environment["MIT_CAUSAL_QUALIFICATION_PREVALIDATED_SHA256"] = prevalidated_sha256
    return command, environment


def validate_python_estimator_manifest(
    values: dict[str, str],
    row: pd.Series,
    *,
    estimator: str,
    outcome: str,
    donor_scope: str,
    run_id: str,
) -> tuple[bool, float, float]:
    """Fail closed unless a manifest proves the exact current Python run."""
    try:
        selected = float(values.get("selected_tuning", "nan"))
        cv_min_mspe = float(values.get("cv_min_mspe", "nan"))
    except ValueError:
        selected = cv_min_mspe = float("nan")
    production = settings.run_mode == "production"
    inference = values.get("inference", "")
    expected_inference = (
        inference.startswith("gsc_parametric_")
        if estimator == "gsc" and production
        else inference == "mc_unit_jackknife"
        if estimator == "mc" and production
        else inference == "preview_point_estimate"
    )
    valid = (
        values.get("schema") == FORMAL_RESULT_SCHEMA
        and values.get("implementation_version") == FORMAL_IMPLEMENTATION_VERSION
        and values.get("run_id") == run_id
        and values.get("estimator") == estimator
        and values.get("backend") == "python_pytorch"
        and values.get("run_mode") == settings.run_mode
        and values.get("production_eligible", "").upper() == ("TRUE" if production else "FALSE")
        and (
            not production
            or (
                values.get("formal_qualification_eligible", "").upper() == "TRUE"
                and values.get("formal_qualification_receipt_sha256")
                == str(settings.qualification_proof.get("formal_qualification_receipt_sha256", ""))
            )
        )
        and values.get("treatment_order") == str(int(row.treatment_order))
        and values.get("city_key") == str(row.city_key)
        and values.get("grid_id") == str(row.grid_id)
        and values.get("opening_month") == str(row.opening_month)
        and values.get("outcome_family") == str(row.outcome_family)
        and values.get("outcome") == outcome
        and values.get("donor_scope") == donor_scope
        and values.get("specification_fingerprint") == specification_fingerprint(row)
        and len(values.get("input_panel_signature", "")) == 64
        and values.get("price_measure") == effective_price_measure(row)
        and values.get("observation_window")
        == str(settings.label_window if str(row.outcome_family) in {"housing", "viirs"} else 1)
        and values.get("transaction_count_threshold") == str(settings.transaction_count_threshold)
        and expected_inference
        and math.isfinite(selected)
        and (math.isfinite(cv_min_mspe) or settings.run_mode == "preview")
    )
    return valid, selected, cv_min_mspe


def normalized_python_labels(
    raw: pd.DataFrame, row: pd.Series, selected_method: str
) -> pd.DataFrame:
    family = str(row.outcome_family)
    return pd.DataFrame(
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
            "valid_inference_repetitions": raw.get("valid_inference_repetitions", 0),
            "uncertainty_source": raw.get("uncertainty_source", pd.NA),
            "pre_observed_periods": raw.get("pre_observed_periods", pd.NA),
            "pre_rmspe": raw.get("pre_rmspe", pd.NA),
            "pre_mean_effect": raw.get("pre_mean_effect", pd.NA),
            "pretrend_slope": raw.get("pretrend_slope", pd.NA),
            "pretrend_slope_p_value": raw.get("pretrend_slope_p_value", pd.NA),
            "pretrend_task_flag": raw.get("pretrend_task_flag", pd.NA),
            "selected_factors": raw.get("selected_factors", pd.NA),
            "mc_lambda": raw.get("mc_lambda", pd.NA),
            "mc_regularized": raw.get("mc_regularized", pd.NA),
            "mc_cv_mspe": raw.get("mc_cv_mspe", pd.NA),
            "minimum_window_n": raw.get("minimum_window_n", pd.NA),
            "effective_n_observed": raw.get("effective_n_observed", pd.NA),
            "effective_n_counterfactual": raw.get("effective_n_counterfactual", pd.NA),
            "window_supported": raw.get("window_supported", pd.NA),
            "transaction_count": raw.get("transaction_count", pd.NA),
            "transaction_count_min": raw.get("transaction_count_min", pd.NA),
            "control_transaction_count": raw.get("control_transaction_count", pd.NA),
            "transaction_count_threshold": raw.get("transaction_count_threshold", pd.NA),
            "transaction_count_supported": raw.get("transaction_count_supported", pd.NA),
            "control_transaction_count_supported": raw.get(
                "control_transaction_count_supported", pd.NA
            ),
            "price_measure": raw.get("price_measure", effective_price_measure(row)),
            "donor_scope": raw.get("donor_scope", pd.NA),
            "estimator_backend": raw.get("estimator_backend", "python_pytorch"),
            "implementation_version": raw.get(
                "implementation_version", FORMAL_IMPLEMENTATION_VERSION
            ),
        }
    )
