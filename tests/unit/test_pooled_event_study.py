from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from summarize_causal_labels import classify_failure

from urban_intervention.causal.diagnostic_figures import (
    _matching_frame,
    write_pooled_outcome_event_studies,
)
from urban_intervention.causal.event_study import EventStudyResult
from urban_intervention.causal.gpu.provenance import file_sha256
from urban_intervention.causal.pooled_event_study import (
    PooledPathEventStudyResult,
    aggregate_effect_paths,
    attach_pretrend_metadata,
    pretrend_tests,
    read_production_effect_paths,
    restrict_to_common_event_support,
    run_pooled_path_event_study,
    select_pretrend_rows,
)


def _paths() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for order in range(1, 13):
        for event_time in (-9, -8, -7, 1, 2):
            rows.append(
                {
                    "treatment_order": order,
                    "city_key": f"city_{order % 4}",
                    "frequency": "monthly",
                    "outcome_family": "viirs",
                    "outcome": "viirs_avg_asinh",
                    "method": "xu_2017_gsynth",
                    "donor_scope": "same_city",
                    "observation_window": 3,
                    "event_time": event_time,
                    "causal_response_label": (
                        order * 0.001 if event_time < 0 else 0.2 * event_time + order * 0.001
                    ),
                    "label_available": True,
                    "standard_error": 0.05,
                    "specification_fingerprint": "main",
                    "run_id": f"run-{order}",
                }
            )
    return pd.DataFrame(rows)


def test_pooled_series_combines_within_and_between_variance() -> None:
    series = aggregate_effect_paths(_paths())
    row = series.loc[series["event_time"].eq(1)].iloc[0]
    assert row["n_grids"] == 12
    assert row["se_available"] == 12
    assert row["within_var"] > 0
    assert row["between_var"] > 0
    assert row["n_cities"] == 4
    assert np.isfinite(row["se_label_city_cluster"])
    assert row["ci_lower_city_cluster"] < row["mean_label"] < row["ci_upper_city_cluster"]
    assert row["ci_lower"] < row["mean_label"] < row["ci_upper"]


def test_pretrend_metadata_is_diagnostic_not_an_admission_gate() -> None:
    paths = _paths()
    selected = select_pretrend_rows(paths, latest_n=2)
    assert set(selected["event_time"]) == {-8, -7}
    grid = pretrend_tests(paths, cluster="grid", latest_n=2)
    city = pretrend_tests(paths, cluster="city", latest_n=2)
    enriched = attach_pretrend_metadata(paths, grid, city)
    assert np.isfinite(enriched["grid_pretrend_p_value"]).all()
    assert enriched["pretrend_flag"].str.contains("cluster_").all()
    assert enriched["label_available"].all()


def test_pooled_path_runner_reads_manifest_and_writes_metadata(tmp_path, caplog) -> None:
    task = tmp_path / "staging" / "task"
    task.mkdir(parents=True)
    labels = _paths()
    labels.to_parquet(task / "causal_response_labels.parquet", index=False)
    manifest = {
        "run_mode": "production",
        "production_eligible": "TRUE",
        "estimator": "gsc",
        "frequency": "monthly",
        "donor_scope": "same_city",
        "observation_window": "3",
        "specification_fingerprint": "main",
        "run_id": "qualification-run",
        "labels_sha256": file_sha256(task / "causal_response_labels.parquet"),
    }
    pd.DataFrame({"field": manifest.keys(), "value": manifest.values()}).to_csv(
        task / "manifest.csv", index=False
    )
    rejected = tmp_path / "staging" / "missing_manifest"
    rejected.mkdir()
    labels.to_parquet(rejected / "causal_response_labels.parquet", index=False)
    output = tmp_path / "result"
    result = run_pooled_path_event_study(tmp_path / "staging", output, figures=False)
    assert result.diagnostics["treatment_orders"] == 12
    assert result.diagnostics["admission_candidate_files"] == 2
    assert result.diagnostics["admission_admitted_files"] == 1
    assert result.diagnostics["admission_missing_manifest"] == 1
    persisted = pd.read_csv(output / "diagnostics.csv").iloc[0]
    assert int(persisted["admission_candidate_files"]) == 2
    assert int(persisted["admission_admitted_files"]) == 1
    assert int(persisted["admission_missing_manifest"]) == 1
    assert "missing_manifest=1" in caplog.text
    assert (output / "effect_paths_with_pretrend.parquet").is_file()
    assert (output / "pretrend_city_cluster.csv").is_file()


def test_pooled_path_runner_filters_exact_sample_task_keys(tmp_path) -> None:
    task = tmp_path / "staging" / "task"
    task.mkdir(parents=True)
    labels = _paths()
    labels.to_parquet(task / "causal_response_labels.parquet", index=False)
    manifest = {
        "run_mode": "production",
        "production_eligible": "TRUE",
        "estimator": "gsc",
        "frequency": "monthly",
        "donor_scope": "same_city",
        "observation_window": "3",
        "specification_fingerprint": "main",
        "run_id": "qualification-run",
        "labels_sha256": file_sha256(task / "causal_response_labels.parquet"),
    }
    pd.DataFrame({"field": manifest.keys(), "value": manifest.values()}).to_csv(
        task / "manifest.csv", index=False
    )

    result = run_pooled_path_event_study(
        tmp_path / "staging",
        tmp_path / "result",
        figures=False,
        task_keys={(1, "viirs"), (2, "housing")},
    )

    assert set(result.paths["treatment_order"]) == {1}
    assert result.diagnostics["requested_task_keys"] == 2


def test_pooled_path_runner_explains_zero_admission(tmp_path) -> None:
    task = tmp_path / "staging" / "missing_manifest"
    task.mkdir(parents=True)
    _paths().to_parquet(task / "causal_response_labels.parquet", index=False)

    with pytest.raises(ValueError, match="missing_manifest=1"):
        run_pooled_path_event_study(tmp_path / "staging", tmp_path / "result", figures=False)


def test_expected_frequency_filter_is_not_logged_as_warning(tmp_path, caplog) -> None:
    task = tmp_path / "staging" / "task"
    task.mkdir(parents=True)
    labels_path = task / "causal_response_labels.parquet"
    _paths().to_parquet(labels_path, index=False)
    manifest = {
        "run_mode": "production",
        "production_eligible": "TRUE",
        "estimator": "gsc",
        "frequency": "monthly",
        "observation_window": "3",
        "specification_fingerprint": "main",
        "labels_sha256": file_sha256(labels_path),
    }
    pd.DataFrame({"field": manifest.keys(), "value": manifest.values()}).to_csv(
        task / "manifest.csv", index=False
    )

    caplog.set_level(logging.INFO, logger="urban_intervention.causal.pooled_event_study")
    paths = read_production_effect_paths(
        tmp_path / "staging",
        frequency="annual",
    )

    assert paths.empty
    assert paths.attrs["admission_counts"]["frequency_filter_mismatch"] == 1
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_effect_path_reader_rejects_a_different_monthly_window(tmp_path) -> None:
    task = tmp_path / "staging" / "task"
    task.mkdir(parents=True)
    labels_path = task / "causal_response_labels.parquet"
    _paths().to_parquet(labels_path, index=False)
    manifest = {
        "run_mode": "production",
        "production_eligible": "TRUE",
        "estimator": "gsc",
        "frequency": "monthly",
        "observation_window": "1",
        "specification_fingerprint": "main",
        "labels_sha256": file_sha256(labels_path),
    }
    pd.DataFrame({"field": manifest.keys(), "value": manifest.values()}).to_csv(
        task / "manifest.csv", index=False
    )

    paths = read_production_effect_paths(
        tmp_path / "staging", monthly_observation_window=3
    )

    assert paths.empty
    assert paths.attrs["admission_counts"]["observation_window_filter_mismatch"] == 1


def test_common_event_support_aligns_monthly_and_annual_methods() -> None:
    frame = pd.DataFrame(
        {
            "frequency": ["monthly"] * 7 + ["annual"] * 6,
            "event_time": [-43, -42, -7, -6, 1, 2, 24, -5, -4, -1, 0, 1, 4],
        }
    )

    supported = restrict_to_common_event_support(frame)

    monthly = supported.loc[supported["frequency"].eq("monthly"), "event_time"].tolist()
    annual = supported.loc[supported["frequency"].eq("annual"), "event_time"].tolist()
    assert monthly == [-42, -7, 1, 24]
    assert annual == [-4, -1, 1]


def test_task_not_terminal_is_an_execution_state_failure() -> None:
    assert classify_failure("task_not_terminal") == "execution_state"
    assert classify_failure("preonly_placebo_quality_gate_failed") == "research"


def test_four_primary_outcome_event_study_bundles_are_written(tmp_path) -> None:
    paths = _paths()
    series = aggregate_effect_paths(paths)
    grid = pretrend_tests(paths, cluster="grid", latest_n=2)
    city = pretrend_tests(paths, cluster="city", latest_n=2)
    pooled = PooledPathEventStudyResult(paths, series, grid, city, {})

    written = write_pooled_outcome_event_studies({}, pooled, tmp_path)

    assert len(written) == 16
    assert all(path.is_file() for path in written)


def test_monthly_matching_figure_uses_formal_label_horizons_and_keeps_t0() -> None:
    event_times = [-9, -8, -7, 0, 1, 2, 3, 6, 12, 18, 24]
    coefficients = pd.DataFrame(
        {
            "event_time": event_times,
            "estimate": np.arange(len(event_times), dtype=float),
            "confidence_lower_city": np.arange(len(event_times), dtype=float) - 0.1,
            "confidence_upper_city": np.arange(len(event_times), dtype=float) + 0.1,
        }
    )
    empty_test = pd.DataFrame([{"p_value": np.nan}])
    result = EventStudyResult(
        coefficients,
        empty_test,
        empty_test,
        {"reference_event_time": -7, "reference_treated": 1},
    )

    plotted = _matching_frame(result, family="viirs")

    assert set(plotted.loc[plotted["event_time"].gt(0), "event_time"]) == {
        1,
        3,
        6,
        12,
        18,
        24,
    }
    assert 0 in set(plotted["event_time"])
    assert 2 not in set(plotted["event_time"])
