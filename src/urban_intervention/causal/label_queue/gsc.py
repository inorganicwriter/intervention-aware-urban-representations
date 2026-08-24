"""Behavior-preserving component of the modular causal label queue."""

from __future__ import annotations

import json
from functools import partial

import pandas as pd
import pyarrow.parquet as pq

from urban_intervention.utils import atomic_write_csv as atomic_csv
from urban_intervention.utils import atomic_write_json
from urban_intervention.utils import sha256_file as file_sha256

from .estimators import (
    gsc_output,
    normalized_python_labels,
    python_estimator_command,
    validate_python_estimator_manifest,
)
from .runtime import (
    HORIZONS,
    OUTCOMES,
    R_SCRIPT,
    ROOT,
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
    task_directory,
)

atomic_json = partial(atomic_write_json, default=str)

_GSC_STRUCTURAL_FAILURE_REASONS = {
    "monthly_viirs_cache_unavailable",
}
_GSC_STRUCTURAL_FAILURE_PATTERNS = (
    "no post-treatment full-year outcome",
    "insufficient clean post-treatment annual periods",
    "insufficient post-treatment monthly periods",
)


def run_python_gsc_scope(
    row: pd.Series, donor_scope: str
) -> tuple[bool, list[pd.DataFrame], dict[str, object]]:
    family = str(row.outcome_family)
    run_id = new_run_id()
    command, environment = python_estimator_command(row, "gsc", donor_scope, run_id)
    completed = run(command, environment)
    if completed.returncode != 0:
        return (
            False,
            [],
            {
                "reason": "python_gsc_runtime_or_support_failure",
                "backend": "python_gpu",
                "log": completed.stdout[-4000:],
            },
        )
    selected_method = (
        "xu_2017_gsynth_same_city"
        if donor_scope == "same_city"
        else "xu_2017_gsynth_all_city_standardized"
    )
    labels: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []
    for outcome in OUTCOMES[family]:
        path = gsc_output(row, outcome, donor_scope) / "causal_response_labels.parquet"
        manifest_path = path.parent / "manifest.csv"
        if not path.exists() or not manifest_path.exists():
            return (
                False,
                [],
                {
                    "reason": "python_gsc_manifest_or_labels_missing",
                    "path": str(path.parent),
                },
            )
        values = read_estimator_manifest(manifest_path)
        valid, selected_rank, cv_min_mspe = validate_python_estimator_manifest(
            values,
            row,
            estimator="gsc",
            outcome=outcome,
            donor_scope=donor_scope,
            run_id=run_id,
        )
        if (
            not valid
            or not float(selected_rank).is_integer()
            or values.get("labels_sha256") != file_sha256(path)
        ):
            return (
                False,
                [],
                {
                    "reason": "python_gsc_manifest_does_not_prove_current_run",
                    "path": str(manifest_path),
                },
            )
        raw = pq.read_table(path).to_pandas()
        raw = raw.loc[raw["event_time"].isin(HORIZONS[family])].copy()
        actual = set(pd.to_numeric(raw["event_time"], errors="coerce").dropna().astype(int))
        if actual != set(HORIZONS[family]):
            return (
                False,
                [],
                {
                    "reason": "python_gsc_outcome_horizon_grid_incomplete",
                    "outcome": outcome,
                    "actual_horizons": sorted(actual),
                },
            )
        labels.append(normalized_python_labels(raw, row, selected_method))
        manifests.append(
            {
                "path": str(manifest_path.relative_to(ROOT)),
                "sha256": file_sha256(manifest_path),
                "labels_sha256": file_sha256(path),
                "selected_rank": int(selected_rank),
                "cv_min_mspe": cv_min_mspe,
                "run_id": run_id,
            }
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
            "log": completed.stdout,
        },
    )


def run_gsc_scope(
    row: pd.Series, donor_scope: str
) -> tuple[bool, list[pd.DataFrame], dict[str, object]]:
    if settings.estimator_backend == "python_gpu":
        return run_python_gsc_scope(row, donor_scope)
    family = str(row.outcome_family)
    frequency = "monthly" if family in {"housing", "viirs"} else "annual"
    cohort = str(row.opening_month) if frequency == "monthly" else _cohort_year(row.opening_month)
    run_id = new_run_id()
    completed = run(
        [
            str(R_SCRIPT),
            str(r_script("run_complete_xu_gsc.R")),
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
    if completed.returncode != 0:
        log = completed.stdout
        if "non-finite treated-target counterfactual" in log:
            reason = "xu_gsc_target_counterfactual_nonfinite"
        elif "uncertainty event times do not align" in log:
            reason = "xu_gsc_uncertainty_event_time_mismatch"
        else:
            reason = "xu_gsc_runtime_or_support_failure"
        return False, [], {"reason": reason, "log": log}
    labels: list[pd.DataFrame] = []
    estimator_manifests: list[dict[str, object]] = []
    selected_method = (
        "xu_2017_gsynth_same_city"
        if donor_scope == "same_city"
        else "xu_2017_gsynth_all_city_standardized"
    )
    for outcome in OUTCOMES[family]:
        path = gsc_output(row, outcome, donor_scope) / "causal_response_labels.parquet"
        estimator_manifest = path.parent / "manifest.csv"
        if not estimator_manifest.exists() or not path.exists():
            return (
                False,
                [],
                {"reason": "xu_gsc_manifest_or_labels_missing", "path": str(path.parent)},
            )
        manifest_values = read_estimator_manifest(estimator_manifest)
        expected_production = settings.run_mode == "production"
        expected_nboots = "200" if expected_production else "0"
        if (
            manifest_values.get("schema")
            != "complete_published_estimators_v3_explicit_deterministic_contracts"
            or manifest_values.get("run_id") != run_id
            or manifest_values.get("estimator") != "gsynth"
            or manifest_values.get("CV", "").upper() != "TRUE"
            or manifest_values.get("cv_method") != "rolling"
            or manifest_values.get("cv_folds") != "5"
            or manifest_values.get("cv_prop") != "0.1"
            or manifest_values.get("cv_nobs") != "3"
            or manifest_values.get("cv_buffer") != "1"
            or manifest_values.get("cv_rule") != "1se"
            or manifest_values.get("tol") != "1e-05"
            or manifest_values.get("max_iteration") != "5000"
            or manifest_values.get("run_mode") != settings.run_mode
            or manifest_values.get("production_eligible", "").upper()
            != ("TRUE" if expected_production else "FALSE")
            or manifest_values.get("specification_fingerprint") != specification_fingerprint(row)
            or manifest_values.get("price_measure") != effective_price_measure(row)
            or manifest_values.get("observation_window")
            != str(settings.label_window if frequency == "monthly" else 1)
            or manifest_values.get("inference") != "parametric"
            or manifest_values.get("nboots") != expected_nboots
        ):
            return (
                False,
                [],
                {
                    "reason": "xu_gsc_manifest_does_not_prove_current_production_run",
                    "path": str(estimator_manifest),
                    "run_id": run_id,
                },
            )
        estimator_manifests.append(
            {
                "path": str(estimator_manifest.relative_to(ROOT)),
                "sha256": file_sha256(estimator_manifest),
                "labels_sha256": file_sha256(path),
                "run_id": run_id,
            }
        )
        raw = pq.read_table(path).to_pandas()
        raw = raw.loc[raw["event_time"].isin(HORIZONS[family])].copy()
        expected_horizons = set(HORIZONS[family])
        actual_horizons = set(
            pd.to_numeric(raw["event_time"], errors="coerce").dropna().astype(int)
        )
        if actual_horizons != expected_horizons:
            return (
                False,
                [],
                {
                    "reason": "xu_gsc_outcome_horizon_grid_incomplete",
                    "outcome": outcome,
                    "expected_horizons": sorted(expected_horizons),
                    "actual_horizons": sorted(actual_horizons),
                },
            )
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
            }
        )
        labels.append(normalized)
    return (
        True,
        labels,
        {
            "selected_method": selected_method,
            "donor_scope": donor_scope,
            "run_id": run_id,
            "estimator_manifests": estimator_manifests,
            "log": completed.stdout,
        },
    )


def classify_gsc_failure(details: dict[str, object]) -> str | None:
    """Classify failures that cannot be rescued by MC or cross-city donors.

    Only failures that also make an MC panel impossible are structural.  GSC's
    complete-path and five-pre-period requirements are deliberately excluded:
    MC accepts partial target paths and one finite pre-treatment observation.
    """
    reason = str(details.get("reason") or "")
    if reason in _GSC_STRUCTURAL_FAILURE_REASONS:
        return "structural_support_failure"
    text = (reason + "\n" + str(details.get("log") or "")).lower()
    if any(pattern in text for pattern in _GSC_STRUCTURAL_FAILURE_PATTERNS):
        return "structural_support_failure"
    return None


def skip_after_structural_gsc_failure(
    queue: pd.DataFrame,
    index: int,
    row: pd.Series,
    gsc_details: dict[str, object],
) -> None:
    """Persist a bounded GSC support failure without launching fallback R jobs."""
    directory = task_directory(int(row.treatment_order), str(row.outcome_family))
    atomic_json(
        {"schema": "gsc_failure_before_skip_v1", **gsc_details},
        directory / "gsc_attempt.json",
    )
    matching_attempt_path = directory / "matching_attempt.json"
    matching_details = (
        json.loads(matching_attempt_path.read_text(encoding="utf-8"))
        if matching_attempt_path.exists()
        else {"reason": None}
    )
    atomic_json(
        {
            "schema": "causal_response_labels_v1",
            "status": "skipped",
            "method": None,
            "treatment_order": int(row.treatment_order),
            "outcome_family": str(row.outcome_family),
            "matching_failure": matching_details,
            "gsc_failure": gsc_details,
            "failure_class": classify_gsc_failure(gsc_details),
            "fallback_suppressed": True,
        },
        directory / "failure_manifest.json",
    )
    queue.loc[index, ["status", "selected_method", "failure_reason"]] = [
        "skipped",
        pd.NA,
        str(gsc_details.get("reason")),
    ]
    atomic_csv(queue, settings.family_queue)
