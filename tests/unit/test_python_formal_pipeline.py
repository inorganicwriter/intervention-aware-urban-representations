from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from urban_intervention.causal.gpu.formal_runner import (
    FormalRunRequest,
    _apply_observation_window,
    _apply_window_replicate_inference,
    _cross_city_masked_placebo,
    _validate_production_estimator_config,
    run_formal_panel,
)
from urban_intervention.causal.gpu.gsc import GSCConfig
from urban_intervention.causal.gpu.io import load_estimation_panel
from urban_intervention.causal.gpu.matrix_completion import MatrixCompletionConfig
from urban_intervention.causal.gpu.panel_builder import (
    PanelBuildRequest,
    build_estimation_panel_from_frames,
    deterministic_cross_city_gsc_sample,
    monthly_event_calendar,
)
from urban_intervention.causal.gpu.runtime import RuntimeConfig, TorchRuntime


def _long_outcomes() -> pd.DataFrame:
    periods = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
    rows: list[dict[str, object]] = []
    for city, grid, offset in (
        ("a", "target", 10.0),
        ("a", "d1", 0.0),
        ("a", "d2", 2.0),
        ("b", "d3", 100.0),
        ("b", "d4", 104.0),
    ):
        for position, period in enumerate(periods):
            value = offset + position
            if grid == "d2" and period == 2022:
                value = np.nan
            rows.append(
                {
                    "city_key": city,
                    "grid_id": grid,
                    "period": period,
                    "population_log": value,
                }
            )
    return pd.DataFrame(rows)


def test_monthly_calendar_excludes_anticipation_and_opening_month() -> None:
    calendar = monthly_event_calendar("2020-07", anticipation_months=6)
    assert calendar["pre"][0] == pd.Timestamp("2017-01-01")
    assert calendar["pre"][-1] == pd.Timestamp("2019-12-01")
    assert calendar["post"][0] == pd.Timestamp("2020-08-01")
    assert list(calendar["excluded"]) == list(pd.date_range("2020-01-01", "2020-07-01", freq="MS"))


def test_cross_city_gsc_donor_sample_is_deterministic_and_outcome_free() -> None:
    donors = pd.DataFrame(
        {
            "city_key": [f"c{index % 4}" for index in range(100)],
            "grid_id": [f"g{index:04d}" for index in range(100)],
        }
    )
    donors["unit_id"] = donors["city_key"] + "::" + donors["grid_id"]
    first = deterministic_cross_city_gsc_sample(donors, 20, 17)
    shuffled = deterministic_cross_city_gsc_sample(
        donors.sample(frac=1, random_state=3), 20, 17
    )
    assert first["unit_id"].tolist() == shuffled["unit_id"].tolist()
    assert len(first) == 20
    assert set(deterministic_cross_city_gsc_sample(donors, 20, 18)["unit_id"]) != set(
        first["unit_id"]
    )


def test_window_inference_uses_joint_gsc_draw_paths() -> None:
    raw = pd.DataFrame(
        {
            "period": ["2020-08", "2020-09"],
            "event_time": [1, 2],
            "observed": [2.0, 2.0],
            "counterfactual": [1.0, 1.0],
            "causal_response_label": [1.0, 1.0],
            "label_available": [True, True],
            "standard_error": [1.0, 1.0],
            "confidence_lower": [-0.96, -0.96],
            "confidence_upper": [2.96, 2.96],
            "p_value": [0.3, 0.3],
            "valid_inference_repetitions": [3, 3],
            "uncertainty_source": ["gsc_parametric_reference_ar"] * 2,
        }
    )
    windowed = _apply_observation_window(raw, 2)
    draws = np.asarray([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    result, aggregated = _apply_window_replicate_inference(
        windowed,
        raw_event_time=np.asarray([1, 2]),
        replicate_estimates=draws,
        estimator="gsc",
        window=2,
        requested_repetitions=3,
    )
    assert result.loc[result["event_time"].eq(2), "standard_error"].iloc[0] == 1.0
    np.testing.assert_allclose(aggregated[:, 1], [-1.0, 0.0, 1.0])


def test_window_inference_rebuilds_mc_jackknife_pseudo_values() -> None:
    raw = pd.DataFrame(
        {
            "period": ["2020-08", "2020-09"],
            "event_time": [1, 2],
            "observed": [2.0, 2.0],
            "counterfactual": [1.0, 1.0],
            "causal_response_label": [1.0, 1.0],
            "label_available": [True, True],
            "standard_error": [0.1, 0.1],
            "confidence_lower": [0.8, 0.8],
            "confidence_upper": [1.2, 1.2],
            "p_value": [0.01, 0.01],
            "valid_inference_repetitions": [2, 2],
            "uncertainty_source": ["mc_unit_jackknife"] * 2,
        }
    )
    windowed = _apply_observation_window(raw, 2)
    draws = np.asarray([[0.5, 0.5], [1.5, 1.5], [np.nan, np.nan]])
    result, _ = _apply_window_replicate_inference(
        windowed,
        raw_event_time=np.asarray([1, 2]),
        replicate_estimates=draws,
        estimator="mc",
        window=2,
        requested_repetitions=3,
    )
    # pseudo-values are 2 and 0, whose jackknife SE is sqrt(var / 2) = 1
    assert result.loc[result["event_time"].eq(2), "standard_error"].iloc[0] == 1.0


def test_gsc_builder_uses_pre_only_admission_and_city_pre_scaling() -> None:
    request = PanelBuildRequest(
        treatment_order=7,
        outcome_family="population",
        outcome="population_log",
        estimator="gsc",
        donor_scope="all_city_standardized",
    )
    target = pd.Series(
        {
            "treatment_order": 7,
            "city_key": "a",
            "grid_id": "target",
            "opening_month": "2020-01",
        }
    )
    donors = pd.DataFrame(
        {
            "city_key": ["a", "a", "b", "b"],
            "grid_id": ["d1", "d2", "d3", "d4"],
        }
    )
    built = build_estimation_panel_from_frames(
        target=target,
        donors=donors,
        outcomes=_long_outcomes(),
        request=request,
        pre=pd.Index([2015, 2016, 2017, 2018, 2019]),
        post=pd.Index([2021, 2022, 2023]),
        opening_period_excluded=2020,
    )
    assert built.metadata["donors_used"] == 4
    assert built.metadata["donor_admission_uses_post_outcome"] is False
    assert built.panel.loc[built.panel["grid_id"].eq("d2"), "value"].isna().sum() == 1
    a_pre = built.panel.loc[
        built.panel["role"].eq("donor")
        & built.panel["city_key"].eq("a")
        & built.panel["time_id"].le(5),
        "value",
    ]
    target_rows = built.panel.loc[built.panel["role"].eq("treated")]
    expected_center = a_pre.mean()
    expected_scale = a_pre.std(ddof=1)
    np.testing.assert_allclose(
        target_rows["model_value"],
        (target_rows["value"] - expected_center) / expected_scale,
    )


def test_mc_builder_caps_donors_by_pre_support_without_post_leakage() -> None:
    outcomes = _long_outcomes()
    outcomes.loc[
        outcomes["grid_id"].eq("d3") & outcomes["period"].isin([2015, 2016]),
        "population_log",
    ] = np.nan
    request = PanelBuildRequest(
        treatment_order=7,
        outcome_family="population",
        outcome="population_log",
        estimator="mc",
        max_mc_donors=2,
    )
    target = pd.Series(
        {
            "treatment_order": 7,
            "city_key": "a",
            "grid_id": "target",
            "opening_month": "2020-01",
        }
    )
    donors = pd.DataFrame(
        {"city_key": ["a", "a", "b"], "grid_id": ["d1", "d2", "d3"]}
    )
    built = build_estimation_panel_from_frames(
        target=target,
        donors=donors,
        outcomes=outcomes,
        request=request,
        pre=pd.Index([2015, 2016, 2017, 2018, 2019]),
        post=pd.Index([2021, 2022, 2023]),
        opening_period_excluded=2020,
    )
    assert set(built.panel.loc[built.panel["role"].eq("donor"), "grid_id"]) == {"d1", "d2"}
    assert built.metadata["donor_cap"] == "top_2_by_pre_finite_count"


def test_housing_panel_masks_low_transaction_observations_before_fitting() -> None:
    periods = pd.date_range("2019-01-01", periods=3, freq="MS")
    outcomes = pd.DataFrame(
        {
            "city_key": ["a"] * 6,
            "grid_id": ["target"] * 3 + ["donor"] * 3,
            "period": list(periods) * 2,
            "housing_log_price": [1.0, 1.1, 1.2, 0.9, 1.0, 1.1],
            "transaction_count": [2, 2, 2, 2, 1, 2],
        }
    )
    request = PanelBuildRequest(
        treatment_order=7,
        outcome_family="housing",
        outcome="housing_log_price",
        estimator="mc",
        transaction_count_threshold=2,
    )
    built = build_estimation_panel_from_frames(
        target=pd.Series(
            {
                "treatment_order": 7,
                "city_key": "a",
                "grid_id": "target",
                "opening_month": "2019-02",
            }
        ),
        donors=pd.DataFrame({"city_key": ["a"], "grid_id": ["donor"]}),
        outcomes=outcomes,
        request=request,
        pre=pd.Index(periods[:2]),
        post=pd.Index(periods[2:]),
        opening_period_excluded=pd.Timestamp("2019-02-01"),
    )
    low_support = built.panel.loc[
        built.panel["grid_id"].eq("donor")
        & built.panel["period"].eq(periods[1])
    ]
    assert low_support["value"].isna().all()
    assert low_support["model_value"].isna().all()
    assert built.metadata["transaction_count_threshold"] == 2
    assert built.metadata["transaction_count_threshold_unit"] == "grid_month"


def test_formal_preview_runner_publishes_complete_python_contract(tmp_path: Path) -> None:
    periods = np.asarray([2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023])
    controls = np.stack(
        [
            1.0 + 0.2 * np.arange(len(periods)),
            2.0 + 0.1 * np.arange(len(periods)),
            0.5 + 0.3 * np.arange(len(periods)),
        ],
        axis=1,
    )
    target = 0.5 * controls[:, 0] + 0.5 * controls[:, 1]
    target[5:] += 2.0
    y = np.column_stack([controls, target])
    rows: list[dict[str, object]] = []
    for unit in range(y.shape[1]):
        for time in range(y.shape[0]):
            rows.append(
                {
                    "gsc_unit_id": unit + 1,
                    "time_id": time + 1,
                    "period": int(periods[time]),
                    "role": "treated" if unit == 3 else "donor",
                    "city_key": "a",
                    "grid_id": "target" if unit == 3 else f"d{unit}",
                    "treatment_order": 7 if unit == 3 else pd.NA,
                    "value": y[time, unit],
                    "model_value": y[time, unit],
                    "D": int(unit == 3 and time >= 5),
                }
            )
    frame = pd.DataFrame(rows)
    request = FormalRunRequest(
        estimator="gsc",
        output_directory=tmp_path,
        treatment_order=7,
        city_key="a",
        grid_id="target",
        opening_month="2020-01",
        outcome_family="population",
        outcome="population_log",
        donor_scope="same_city",
        run_mode="preview",
        run_id="test-run",
        specification_fingerprint="test-spec",
        device="cpu",
    )
    result = run_formal_panel(
        frame,
        request,
        panel_metadata={
            "frequency": "annual",
            "opening_period_excluded": 2020,
            "clean_pre_periods": 5,
            "post_periods": 3,
            "donors_used": 3,
            "target_effect_scale_to_original_units": 1.0,
            "target_center_to_original_units": 0.0,
        },
        gsc_config=GSCConfig(fixed_rank=1, bootstrap_mode="none", n_bootstrap=0),
    )
    assert result.labels_path.exists()
    assert result.manifest_path.exists()
    assert result.manifest["schema"] == "causal_python_formal_result_v3_qualified"
    assert result.manifest["backend"] == "python_pytorch"
    assert result.manifest["production_eligible"] is False
    assert result.labels["event_time"].tolist() == [-5, -4, -3, -2, -1, 1, 2, 3]
    json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))


def test_formal_production_refuses_missing_qualification_receipt(tmp_path: Path) -> None:
    request = FormalRunRequest(
        estimator="gsc",
        output_directory=tmp_path,
        treatment_order=1,
        city_key="a",
        grid_id="g1",
        opening_month="2020-01",
        outcome_family="population",
        outcome="population_log",
        run_mode="production",
        device="cpu",
    )
    with np.testing.assert_raises_regex(ValueError, "qualification receipt"):
        run_formal_panel(pd.DataFrame(), request)


def test_production_config_cannot_weaken_qualified_inference() -> None:
    with np.testing.assert_raises_regex(ValueError, "production GSC"):
        _validate_production_estimator_config(
            "gsc", GSCConfig(bootstrap_mode="auto", n_bootstrap=20), None
        )
    with np.testing.assert_raises_regex(ValueError, "production MC"):
        _validate_production_estimator_config(
            "mc", None, MatrixCompletionConfig(inference="none")
        )


def test_cross_city_gsc_masked_placebo_compares_target_to_twenty_donors(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    periods = np.asarray([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022])
    base = np.arange(len(periods), dtype=float)[:, None] * 0.2
    y = base + np.arange(31, dtype=float)[None, :] * 0.03
    y += rng.normal(scale=0.002, size=y.shape)
    rows: list[dict[str, object]] = []
    for unit in range(y.shape[1]):
        for time in range(y.shape[0]):
            rows.append(
                {
                    "gsc_unit_id": unit + 1,
                    "time_id": time + 1,
                    "period": int(periods[time]),
                    "model_value": y[time, unit],
                    "D": int(unit == 30 and time >= 8),
                }
            )
    loaded = load_estimation_panel(pd.DataFrame(rows), "gsc")
    request = FormalRunRequest(
        estimator="gsc",
        output_directory=tmp_path,
        treatment_order=7,
        city_key="a",
        grid_id="target",
        opening_month="2020-01",
        outcome_family="population",
        outcome="population_log",
        donor_scope="all_city_standardized",
    )
    placebo = _cross_city_masked_placebo(
        loaded,
        0,
        GSCConfig(fixed_rank=0),
        TorchRuntime(RuntimeConfig(device="cpu")),
        request,
    )
    assert len(placebo) == 21
    assert placebo["placebo_role"].eq("target").sum() == 1
    assert np.isfinite(placebo["masked_rmspe"]).all()
