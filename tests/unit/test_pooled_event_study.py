from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from summarize_causal_labels import classify_failure

from urban_intervention.causal.diagnostic_figures import (
    _matching_frame,
    _segments,
    event_study_coverage_table,
    event_study_support_table,
    write_pooled_outcome_event_studies,
)
from urban_intervention.causal.event_study import EventStudyResult
from urban_intervention.causal.gpu.provenance import file_sha256
from urban_intervention.causal.pooled_event_study import (
    PooledPathEventStudyResult,
    _reference_contrast_standard_errors,
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
        for event_time in (-4, -3, -2, -1, 0, 1, 2):
            rows.append(
                {
                    "treatment_order": order,
                    "city_key": f"city_{order % 4}",
                    "frequency": "monthly",
                    "outcome_family": "viirs",
                    "outcome": "viirs_avg_asinh",
                    "method": "xu_2017_gsynth",
                    "donor_scope": "same_city",
                    "observation_window": 1,
                    "reference_event_time": -1,
                    "event_time": event_time,
                    "causal_response_label": (
                        0.0
                        if event_time == -1
                        else order * 0.001
                        if event_time < 0
                        else 0.2 * event_time + order * 0.001
                    ),
                    "label_available": True,
                    "standard_error": 0.05,
                    "specification_fingerprint": "main_a6_r1km__a0__w1__price_median",
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


def test_pooled_series_does_not_impute_zero_uncertainty_for_missing_se() -> None:
    paths = _paths()
    paths = paths.loc[
        (paths["treatment_order"] == 1) & (paths["event_time"] == 1)
    ].copy()
    paths["causal_response_label"] = 5.0
    paths["standard_error"] = np.nan

    series = aggregate_effect_paths(paths)
    row = series.iloc[0]
    assert row["mean_label"] == pytest.approx(5.0)
    assert pd.isna(row["se_label"])
    assert pd.isna(row["ci_lower"])
    assert pd.isna(row["ci_upper"])
    assert row["uncertainty_status"] == "missing_task_standard_errors"


def test_pooled_series_does_not_treat_partial_se_as_complete_inference() -> None:
    all_paths = _paths()
    paths = all_paths.loc[
        (all_paths["event_time"] == 1) & all_paths["treatment_order"].isin([1, 2])
    ].copy()
    paths.loc[paths["treatment_order"] == 2, "standard_error"] = np.nan

    row = aggregate_effect_paths(paths).iloc[0]

    assert row["se_available"] == 1
    assert pd.isna(row["within_var"])
    assert pd.isna(row["se_label"])
    assert pd.isna(row["ci_lower"])
    assert pd.isna(row["ci_upper"])
    assert row["uncertainty_status"] == "partially_missing_task_standard_errors"


def test_pretrend_metadata_is_diagnostic_not_an_admission_gate() -> None:
    paths = _paths()
    selected = select_pretrend_rows(paths, latest_n=2)
    assert set(selected["event_time"]) == {-3, -2}
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
        "observation_window": "1",
        "specification_fingerprint": "main_a6_r1km__a0__w1__price_median",
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
        "observation_window": "1",
        "specification_fingerprint": "main_a6_r1km__a0__w1__price_median",
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
        "observation_window": "1",
        "specification_fingerprint": "main_a6_r1km__a0__w1__price_median",
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
        "specification_fingerprint": "main_a6_r1km__a0__w1__price_median",
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


def test_effect_path_reader_accepts_family_specific_monthly_windows(tmp_path) -> None:
    task = tmp_path / "staging" / "viirs_task"
    task.mkdir(parents=True)
    labels_path = task / "causal_response_labels.parquet"
    _paths().to_parquet(labels_path, index=False)
    manifest = {
        "run_mode": "production",
        "production_eligible": "TRUE",
        "estimator": "gsc",
        "frequency": "monthly",
        "observation_window": "1",
        "specification_fingerprint": "main_a6_r1km__a0__w1__price_median",
        "labels_sha256": file_sha256(labels_path),
    }
    pd.DataFrame({"field": manifest.keys(), "value": manifest.values()}).to_csv(
        task / "manifest.csv", index=False
    )

    paths = read_production_effect_paths(
        tmp_path / "staging",
        monthly_observation_windows={"housing": 3, "viirs": 1},
    )

    assert len(paths) == len(_paths())
    assert set(paths["outcome_family"]) == {"viirs"}
    assert paths.attrs["admission_counts"]["admitted_files"] == 1


def test_effect_path_reader_normalizes_each_task_to_minus_one(tmp_path) -> None:
    task = tmp_path / "staging" / "task"
    task.mkdir(parents=True)
    labels = _paths()
    labels.loc[labels["event_time"].eq(-1), "causal_response_label"] = (
        labels.loc[labels["event_time"].eq(-1), "treatment_order"] * 0.01
    )
    labels_path = task / "causal_response_labels.parquet"
    labels.to_parquet(labels_path, index=False)
    manifest = {
        "run_mode": "production",
        "production_eligible": "TRUE",
        "estimator": "gsc",
        "frequency": "monthly",
        "observation_window": "1",
        "specification_fingerprint": "main_a6_r1km__a0__w1__price_median",
        "labels_sha256": file_sha256(labels_path),
    }
    pd.DataFrame({"field": manifest.keys(), "value": manifest.values()}).to_csv(
        task / "manifest.csv", index=False
    )

    paths = read_production_effect_paths(tmp_path / "staging")

    assert paths.loc[paths["event_time"].eq(-1), "causal_response_label"].eq(0).all()
    order_one_t1 = paths.loc[
        paths["treatment_order"].eq(1) & paths["event_time"].eq(1),
        "causal_response_label",
    ].iloc[0]
    assert order_one_t1 == pytest.approx(0.191)
    assert paths.loc[paths["event_time"].ne(-1), "standard_error"].isna().all()


@pytest.mark.parametrize(
    ("estimator", "expected"),
    [
        ("gsc", 1.0),
        ("mc", 2.0 / np.sqrt(3.0)),
    ],
)
def test_reference_contrast_standard_errors_use_joint_draws(
    estimator: str, expected: float
) -> None:
    part = pd.DataFrame(
        {
            "treatment_order": [1, 1, 1],
            "outcome": ["y", "y", "y"],
            "event_time": [-1, 0, 1],
            "causal_response_label": [1.0, 3.0, 5.0],
        }
    )
    draws = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 3.0, 5.0],
            [2.0, 5.0, 8.0],
        ]
    )

    standard_error = _reference_contrast_standard_errors(
        part,
        draws,
        estimator=estimator,
        reference_event_time=-1,
    )

    assert standard_error[0] == pytest.approx(0.0)
    assert standard_error[1] == pytest.approx(expected)
    assert standard_error[2] == pytest.approx(2 * expected)


def test_effect_path_reader_uses_hashed_joint_draws_for_contrast_se(tmp_path) -> None:
    task = tmp_path / "staging" / "task"
    task.mkdir(parents=True)
    labels = pd.DataFrame(
        {
            "treatment_order": [1, 1, 1],
            "city_key": ["city"] * 3,
            "outcome_family": ["viirs"] * 3,
            "outcome": ["viirs_avg_asinh"] * 3,
            "period": ["2019-12-01", "2020-01-01", "2020-02-01"],
            "event_time": [-1, 0, 1],
            "causal_response_label": [1.0, 3.0, 5.0],
            "label_available": [True] * 3,
        }
    )
    labels_path = task / "causal_response_labels.parquet"
    labels.to_parquet(labels_path, index=False)
    draws = pd.DataFrame(
        {
            "replicate": [1, 2, 3],
            "2019-12-01": [0.0, 1.0, 2.0],
            "2020-01-01": [1.0, 3.0, 5.0],
            "2020-02-01": [2.0, 5.0, 8.0],
        }
    )
    draws_path = task / "inference_draws.parquet"
    draws.to_parquet(draws_path, index=False)
    manifest = {
        "run_mode": "production",
        "production_eligible": "TRUE",
        "estimator": "gsc",
        "frequency": "monthly",
        "donor_scope": "same_city",
        "observation_window": "1",
        "specification_fingerprint": "main_a6_r1km__a0__w1__price_median",
        "labels_sha256": file_sha256(labels_path),
        "inference_draws_sha256": file_sha256(draws_path),
    }
    pd.DataFrame({"field": manifest.keys(), "value": manifest.values()}).to_csv(
        task / "manifest.csv", index=False
    )

    paths = read_production_effect_paths(tmp_path / "staging")

    assert paths["causal_response_label"].tolist() == pytest.approx([0.0, 2.0, 4.0])
    assert paths["standard_error"].tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert set(paths["contrast_inference_source"]) == {
        "joint_bootstrap_reference_contrast"
    }


def test_main_reader_rejects_t_minus_seven_phase2_artifacts(tmp_path) -> None:
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
        "specification_fingerprint": "main_a6_r1km__a6__w1__price_median",
        "labels_sha256": file_sha256(labels_path),
    }
    pd.DataFrame({"field": manifest.keys(), "value": manifest.values()}).to_csv(
        task / "manifest.csv", index=False
    )

    paths = read_production_effect_paths(tmp_path / "staging", reference_event_time=-1)

    assert paths.empty
    assert paths.attrs["admission_counts"]["reference_specification_mismatch"] == 1


def test_annual_reader_uses_prior_year_reference_even_with_shared_monthly_a6_metadata(
    tmp_path,
) -> None:
    task = tmp_path / "staging" / "task"
    task.mkdir(parents=True)
    labels = pd.DataFrame(
        {
            "treatment_order": [1, 1, 1],
            "city_key": ["city"] * 3,
            "outcome_family": ["population"] * 3,
            "outcome": ["population_log"] * 3,
            "period": [2019, 2020, 2021],
            "event_time": [-1, 0, 1],
            "causal_response_label": [1.0, 2.0, 3.0],
            "label_available": [True] * 3,
        }
    )
    labels_path = task / "causal_response_labels.parquet"
    labels.to_parquet(labels_path, index=False)
    manifest = {
        "run_mode": "production",
        "production_eligible": "TRUE",
        "estimator": "gsc",
        "frequency": "annual",
        "observation_window": "1",
        "anticipation_months": "6",
        "specification_fingerprint": "main_a6_r1km__a6__w1__price_median",
        "labels_sha256": file_sha256(labels_path),
    }
    pd.DataFrame({"field": manifest.keys(), "value": manifest.values()}).to_csv(
        task / "manifest.csv", index=False
    )

    paths = read_production_effect_paths(tmp_path / "staging", reference_event_time=-1)

    assert len(paths) == 3
    assert paths["causal_response_label"].tolist() == pytest.approx([0.0, 1.0, 2.0])


def test_event_support_keeps_full_history_including_visual_diagnostic_periods() -> None:
    frame = pd.DataFrame(
        {
            "frequency": ["monthly"] * 7 + ["annual"] * 6,
            "event_time": [-43, -42, -7, -6, 1, 2, 24, -5, -4, -1, 0, 1, 4],
        }
    )

    supported = restrict_to_common_event_support(frame)

    monthly = supported.loc[supported["frequency"].eq("monthly"), "event_time"].tolist()
    annual = supported.loc[supported["frequency"].eq("annual"), "event_time"].tolist()
    assert monthly == [-43, -42, -7, -6, 1, 2, 24]
    assert annual == [-5, -4, -1, 0, 1, 4]


def test_monthly_pretrend_excludes_anticipation_periods() -> None:
    paths = _paths().copy()
    paths["reference_event_time"] = -7
    paths["event_time"] = paths["event_time"] - 6
    paths.loc[paths["event_time"].eq(-6), "causal_response_label"] = 100.0

    selected = select_pretrend_rows(paths, latest_n=5)

    assert -6 not in set(selected["event_time"])
    assert selected["causal_response_label"].abs().max() < 100.0


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


def test_monthly_matching_figure_keeps_all_available_periods_and_t0() -> None:
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
        2,
        3,
        6,
        12,
        18,
        24,
    }
    assert 0 in set(plotted["event_time"])


def test_plot_segments_never_bridge_a_missing_t0() -> None:
    frame = pd.DataFrame({"event_time": [-1, 1], "estimate": [0.0, 0.2]})
    assert len(_segments(frame)) == 2


def test_support_table_uses_longest_overlap_across_available_methods() -> None:
    matching_times = [-7, -6, -5, -4, -3, -2, 0, 1, 2, 3, 4]
    coefficients = pd.DataFrame(
        {
            "event_time": matching_times,
            "estimate": np.zeros(len(matching_times)),
            "confidence_lower_city": np.full(len(matching_times), -0.1),
            "confidence_upper_city": np.full(len(matching_times), 0.1),
        }
    )
    empty_test = pd.DataFrame([{"p_value": np.nan}])
    matching = EventStudyResult(
        coefficients,
        empty_test,
        empty_test,
        {"reference_event_time": -1, "reference_treated": 1},
    )
    paths = _paths()
    pooled = PooledPathEventStudyResult(
        paths,
        aggregate_effect_paths(paths),
        pretrend_tests(paths, cluster="grid", latest_n=2),
        pretrend_tests(paths, cluster="city", latest_n=2),
        {},
    )

    support = event_study_support_table({"viirs": matching}, pooled)
    row = support.loc[
        support["outcome_family"].eq("viirs") & support["method"].eq("matching")
    ].iloc[0]
    assert row["method_pre_min"] == -7
    assert row["common_pre_min"] == -4
    assert row["common_pre_max"] == -1
    assert row["common_post_min"] == 1
    assert row["common_post_max"] == 2

    coverage = event_study_coverage_table({"viirs": matching}, pooled)
    matching_coverage = coverage.loc[
        coverage["outcome_family"].eq("viirs") & coverage["method"].eq("matching")
    ].set_index("event_time")
    assert not bool(matching_coverage.loc[-7, "included_in_comparison_figure"])
    assert bool(matching_coverage.loc[-4, "included_in_comparison_figure"])
    assert bool(matching_coverage.loc[0, "included_in_comparison_figure"])
    assert not bool(matching_coverage.loc[4, "included_in_comparison_figure"])
