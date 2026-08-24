from __future__ import annotations

import numpy as np
import pandas as pd

from urban_intervention.causal.gpu.io import (
    compare_counterfactuals,
    compare_inference_paths,
    load_estimation_panel,
    load_gsc_cv_folds,
    load_mc_cv_contract,
)


def _write_cv_manifest(path, estimator: str) -> None:
    settings = (
        {"cv_folds": 5, "cv_nobs": 3, "cv_buffer": 1, "cv_seed": 20260723}
        if estimator == "gsc"
        else {"cv_folds": 20, "cv_nobs": 1, "cv_buffer": 0, "cv_seed": 20260725}
    )
    fields = {
        "schema": "causal_gpu_input_v1",
        "run_mode": "gpu_export",
        "production_eligible": "FALSE",
        "fect_version": "2.4.5",
        "cv_method": "rolling",
        "cv_prop": 0.1,
        "cv_rule": "1se",
        "tol": 1e-5,
        "max_iteration": 5000,
    } | settings
    pd.DataFrame({"field": fields.keys(), "value": fields.values()}).to_csv(
        path / "manifest.csv", index=False
    )


def test_load_estimation_panel_preserves_nonzero_target_column(tmp_path) -> None:
    rows = []
    for unit in (10, 20, 30):
        for period in range(4):
            rows.append(
                {
                    "gsc_unit_id": unit,
                    "time_id": period + 1,
                    "period": 2010 + period,
                    "model_value": float(unit + period),
                    "D": int(unit == 20 and period >= 2),
                }
            )
    path = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    loaded = load_estimation_panel(path, "gsc")
    assert loaded.panel.single_treated_unit() == 1
    assert loaded.panel.treatment_start() == 2
    assert loaded.periods == (2010, 2011, 2012, 2013)


def test_counterfactual_comparison_applies_absolute_and_relative_gate(tmp_path) -> None:
    reference = tmp_path / "labels.parquet"
    pd.DataFrame(
        {"period": [1, 2, 3], "counterfactual": [10.0, 20.0, 30.0]}
    ).to_parquet(reference, index=False)
    comparison, metrics = compare_counterfactuals(
        (1, 2, 3),
        np.array([10.0, 20.0, 30.0001]),
        reference,
        absolute_tolerance=1e-4,
        relative_tolerance=0,
    )
    assert len(comparison) == 3
    assert metrics["passed"]


def test_inference_comparison_checks_se_scale_and_ci_decisions(tmp_path) -> None:
    reference = tmp_path / "labels.parquet"
    pd.DataFrame(
        {
            "period": [1, 2, 3],
            "event_time": [1, 2, 3],
            "causal_response_label": [0.1, 2.0, -0.2],
            "standard_error": [0.5, 0.5, 0.5],
        }
    ).to_parquet(reference, index=False)
    comparison, metrics = compare_inference_paths(
        (1, 2, 3),
        np.asarray([0.1, 2.0, -0.2]),
        np.asarray([0.51, 0.49, 0.5]),
        reference,
        relative_rmse_tolerance=0.05,
    )
    assert metrics["passed"]
    assert metrics["ci_zero_agreement"] == 1.0
    assert comparison["inference_comparable"].all()


def test_load_gsc_cv_folds_aligns_exported_numeric_ids(tmp_path) -> None:
    rows = []
    for unit in (10, 20, 30):
        for period in range(4):
            rows.append(
                {
                    "gsc_unit_id": unit,
                    "time_id": period + 1,
                    "model_value": float(unit + period),
                    "D": int(unit == 20 and period >= 2),
                }
            )
    panel_path = tmp_path / "gsc_panel.parquet"
    pd.DataFrame(rows).to_parquet(panel_path, index=False)
    loaded = load_estimation_panel(panel_path, "gsc")
    fold_path = tmp_path / "gsc_cv_folds.parquet"
    _write_cv_manifest(tmp_path, "gsc")
    fold_rows = []
    for fold_id in range(1, 6):
        for unit_id in (10, 30):
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "gsc_unit_id": unit_id,
                    "time_id": ((fold_id - 1) % 4) + 1,
                    "scored": True,
                }
            )
    pd.DataFrame(fold_rows).to_parquet(fold_path, index=False)

    folds = load_gsc_cv_folds(fold_path, loaded)

    assert len(folds) == 5
    assert folds[0].removed[0].tolist() == [True, True]
    assert folds[1].scored[1].tolist() == [True, True]


def test_load_mc_cv_contract_aligns_folds_and_lambda_grid(tmp_path) -> None:
    rows = []
    for unit in (10, 20, 30):
        for period in range(4):
            rows.append(
                {
                    "mc_unit_id": unit,
                    "time_id": period + 1,
                    "model_value": float(unit + period),
                    "D": int(unit == 20 and period >= 2),
                }
            )
    panel_path = tmp_path / "mc_panel.parquet"
    pd.DataFrame(rows).to_parquet(panel_path, index=False)
    loaded = load_estimation_panel(panel_path, "mc")
    _write_cv_manifest(tmp_path, "mc")
    fold_rows = []
    for fold_id in range(1, 21):
        for unit_id in (10, 30):
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "mc_unit_id": unit_id,
                    "time_id": ((fold_id - 1) % 4) + 1,
                    "scored": True,
                }
            )
    pd.DataFrame(fold_rows).to_parquet(tmp_path / "mc_cv_folds.parquet", index=False)
    lambdas = np.concatenate((np.geomspace(1.0, 0.001, 19), [0.0]))
    pd.DataFrame(
        {"sequence": np.arange(1, 21), "lambda": lambdas}
    ).to_csv(tmp_path / "mc_lambda_grid.csv", index=False)

    folds, lambdas = load_mc_cv_contract(tmp_path, loaded)

    assert len(folds) == 20
    assert folds[0].score[0].tolist() == [True, False, True]
    assert lambdas[0] == 1.0
    assert lambdas[-1] == 0.0
