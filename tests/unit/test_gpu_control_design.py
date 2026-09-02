from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urban_intervention.causal.gpu import control_design, panel_builder


def test_structurally_missing_nearest_station_distance_does_not_drop_task(
    tmp_path: Path, monkeypatch
) -> None:
    target = pd.Series(
        {
            "city_key": "test_city",
            "grid_id": "treated",
            "opening_month": "2010-09",
        }
    )
    grid_ids = ["treated", "donor_1", "donor_2", "donor_3"]
    feature_data = pd.DataFrame(
        {
            "city_key": ["test_city"] * 4,
            "grid_id": grid_ids,
            "housing_log_price__lag1": [1.0, 1.1, 1.2, 1.3],
            "housing_log_price__lag2": [2.0, 2.1, 2.2, 2.3],
            "housing_log_price__lag3": [3.0, 3.1, 3.2, 3.3],
            "loc_dist_main_km": [1.0, 2.0, 3.0, 4.0],
            "loc_dist_nearest_subcentre_km": [1.0, 2.0, 3.0, 4.0],
            "loc_dist_nearest_centre_km": [1.0, 2.0, 3.0, 4.0],
            "transit_dist_nearest_station_m": [np.nan] * 4,
            "transit_stations_500m": [0.0] * 4,
            "transit_stations_800m": [0.0] * 4,
            "transit_stations_1500m": [0.0] * 4,
            "transit_lines_in_1500m": [0.0] * 4,
            "transit_network_closeness": [0.0] * 4,
        }
    )
    donor_universe = pd.DataFrame(
        {
            "city_key": ["test_city"] * 3,
            "grid_id": grid_ids[1:],
            "unit_id": [f"test_city::{grid_id}" for grid_id in grid_ids[1:]],
        }
    )
    monkeypatch.setattr(
        control_design.pd,
        "read_parquet",
        lambda *_args, **_kwargs: donor_universe,
    )

    frame, features = control_design.build_scope_frame(
        target,
        ("housing",),
        "same_city",
        root=tmp_path,
        target_feature_data=feature_data,
    )

    assert len(frame) == 4
    treated = frame.loc[frame["role"].eq("treated")].iloc[0]
    donors = frame.loc[frame["role"].eq("donor")]
    usable = control_design.usable_static_features(treated, donors, control_design.STATIC_FEATURES)
    assert "transit_dist_nearest_station_m" not in usable
    assert "loc_dist_main_km" in usable
    assert set(control_design.family_feature_names("housing")).issubset(features)


def test_monthly_blocks_vectorization_preserves_finite_count_contract() -> None:
    pre = pd.date_range("2020-01-01", periods=36, freq="MS")
    first = pd.DataFrame(
        {
            "city_key": "alpha",
            "grid_id": "g1",
            "period": pre,
            "value": np.arange(1.0, 37.0),
        }
    )
    second = first.assign(grid_id="g2", value=2.0)
    second.loc[second.index[[0, 12]], "value"] = [np.nan, np.inf]
    result = control_design._monthly_blocks(
        pd.concat([first, second], ignore_index=True),
        ("value",),
        pre,
        12,
    ).set_index("grid_id")

    assert result.loc["g1", "value__lag1"] == pytest.approx(30.5)
    assert result.loc["g1", "value__lag2"] == pytest.approx(18.5)
    assert result.loc["g1", "value__lag3"] == pytest.approx(6.5)
    assert np.isnan(result.loc["g2", "value__lag2"])
    assert np.isnan(result.loc["g2", "value__lag3"])
    assert result.loc["g2", "value__lag1"] == pytest.approx(2.0)


def test_viirs_monthly_path_features_preserve_all_clean_pre_months() -> None:
    pre = pd.date_range("2020-01-01", periods=36, freq="MS")
    frame = pd.DataFrame(
        {
            "city_key": "alpha",
            "grid_id": "g1",
            "period": pre,
            "viirs_avg_asinh": np.arange(1.0, 37.0),
        }
    )
    result = control_design._monthly_path_features(frame, ("viirs_avg_asinh",), pre).iloc[0]

    assert result["viirs_avg_asinh__pre_m01"] == 1.0
    assert result["viirs_avg_asinh__pre_m24"] == 24.0
    assert result["viirs_avg_asinh__pre_m36"] == 36.0
    assert len(control_design.family_feature_names("viirs")) == 36


def test_viirs_preonly_split_uses_24_training_and_12_holdout_months() -> None:
    training, holdout = control_design.split_preonly_features(
        control_design.family_feature_names("viirs")
    )

    assert len(training) == 24
    assert len(holdout) == 12
    assert training[-1].endswith("__pre_m24")
    assert holdout[0].endswith("__pre_m25")


def test_vectorized_city_standardization_matches_frozen_loop() -> None:
    frame = pd.DataFrame(
        {
            "city_key": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "role": ["donor", "donor", "donor", "treated"] * 2,
            "x": [1.0, 2.0, 5.0, 3.0, 4.0, 4.0, 4.0, 7.0],
            "y": [2.0, 4.0, 8.0, 5.0, 1.0, 2.0, 3.0, 2.5],
        }
    )
    expected = frame.copy()
    donor_rows = expected.loc[expected["role"].eq("donor")]
    for city, indices in expected.groupby("city_key", sort=False).groups.items():
        city_donors = donor_rows.loc[donor_rows["city_key"].eq(city)]
        for feature in ("x", "y"):
            values = pd.to_numeric(city_donors[feature], errors="coerce")
            center = float(values[np.isfinite(values)].median())
            scale = control_design._r_mad(values)
            if not np.isfinite(scale) or scale <= np.sqrt(np.finfo(float).eps):
                scale = float(values.std(ddof=1))
            if not np.isfinite(scale) or scale <= np.sqrt(np.finfo(float).eps):
                scale = 1.0
            expected.loc[indices, feature] = (
                pd.to_numeric(expected.loc[indices, feature], errors="coerce") - center
            ) / scale

    actual = control_design.robust_city_standardize(frame, ("x", "y"))

    np.testing.assert_allclose(actual[["x", "y"]], expected[["x", "y"]])


def test_cross_city_matching_caps_donors_before_feature_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = pd.DataFrame(
        [
            {
                "treatment_order": 7,
                "city_key": "target",
                "grid_id": "treated",
                "opening_month": "2020-01",
            }
        ]
    )
    donors = pd.DataFrame(
        {
            "city_key": [f"city_{index % 4}" for index in range(100)],
            "grid_id": [f"g{index:03d}" for index in range(100)],
            "unit_id": [f"u{index:03d}" for index in range(100)],
        }
    )
    observed: dict[str, int] = {}

    monkeypatch.setattr(control_design, "_cached_treatment_units", lambda _root: target)
    monkeypatch.setattr(control_design, "_cached_donor_universe", lambda _root: donors)
    monkeypatch.setattr(
        control_design,
        "read_city_control_features",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(control_design, "active_families", lambda *_args: ("housing",))

    def stop_after_cap(*_args, donor_universe: pd.DataFrame, **_kwargs):
        observed["donors"] = len(donor_universe)
        raise RuntimeError("stop after observing donor cap")

    monkeypatch.setattr(control_design, "build_scope_frame", stop_after_cap)

    result = control_design.design_grid_control(
        7,
        scope="all_city_standardized",
        root=tmp_path,
        device="cpu",
        progress=lambda _details: None,
        max_cross_city_donors=20,
        donor_sampling_seed=17,
    )
    record = result.record.iloc[0]

    assert observed["donors"] == 20
    assert int(record["eligible_scope_donors"]) == 100
    assert int(record["donors_after_pre_outcome_cap"]) == 20
    assert record["pre_outcome_donor_cap"] == "uniform_stable_hash_20_seed_17_before_outcomes"


@pytest.mark.parametrize("estimator", ["gsc", "mc"])
def test_cross_city_estimators_cap_donors_before_outcome_reads(
    estimator: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = pd.DataFrame(
        [
            {
                "treatment_order": 9,
                "city_key": "target",
                "grid_id": "treated",
                "opening_month": "2020-01",
            }
        ]
    )
    donors = pd.DataFrame(
        {
            "city_key": [f"city_{index % 4}" for index in range(100)],
            "grid_id": [f"g{index:03d}" for index in range(100)],
            "unit_id": [f"u{index:03d}" for index in range(100)],
        }
    )
    observed: dict[str, int] = {}
    monkeypatch.setattr(panel_builder, "_cached_panel_treatments", lambda _root: target)
    monkeypatch.setattr(panel_builder, "_cached_panel_donors", lambda _root: donors)

    def stop_before_reading(_request, _cities, _times=None, *, units=None):
        observed["units"] = len(units)
        raise RuntimeError("stop after observing requested units")

    monkeypatch.setattr(panel_builder, "read_outcomes_for_request", stop_before_reading)
    request = panel_builder.PanelBuildRequest(
        treatment_order=9,
        outcome_family="housing",
        outcome="housing_log_price",
        estimator=estimator,
        donor_scope="all_city_standardized",
        max_gsc_cross_city_donors=20,
        gsc_donor_sampling_seed=17,
        root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="requested units"):
        panel_builder.build_estimation_panel(request)

    assert observed["units"] == 21
