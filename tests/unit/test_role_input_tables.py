import numpy as np
import pandas as pd
import pytest

from scripts.analysis import build_role_input_tables as roles
from scripts.analysis.build_role_input_tables import attach_annual_lags
from scripts.collection.download_satellite_event_sequences import build_queue


def test_exact_pretreatment_year_is_used_and_future_never_fills_missing() -> None:
    identity = pd.DataFrame({"grid_id": ["g", "h"], "opening_month": ["2020-01", "2021-12"]})
    source = pd.DataFrame({"grid_id": ["g", "g", "h"], "year": [2018, 2020, 2020],
                           "value": [1., 999., 3.]})
    result = attach_annual_lags(identity, source, ["value"])
    assert np.isnan(result.loc[0, "value_lag1"])
    assert not result.loc[0, "value_lag1_available"]
    assert result.loc[0, "value_lag2"] == 1
    assert result.loc[1, "value_lag1"] == 3
    assert result.lag1_year.tolist() == [2019, 2020]


def test_duplicate_grid_year_is_not_silently_averaged() -> None:
    identity = pd.DataFrame({"grid_id": ["g"], "opening_month": ["2020-01"]})
    source = pd.DataFrame({"grid_id": ["g", "g"], "year": [2019, 2019], "value": [1, 2]})
    with pytest.raises(ValueError, match="Duplicate"):
        attach_annual_lags(identity, source, ["value"])


def test_city_background_uses_all_source_grids_not_only_treatment_sample(tmp_path, monkeypatch) -> None:
    reference = tmp_path / "grids" / "city"
    reference.mkdir(parents=True)
    pd.DataFrame({"grid_id": ["a", "b"]}).to_parquet(reference / "city_grids.parquet", index=False)
    source = tmp_path / "poi.parquet"
    pd.DataFrame({"city": ["city", "city"], "grid_id": ["a", "b"], "year": [2019, 2019],
                  "poi_count": [1, 9], "poi_category_entropy": [0, 0],
                  "poi_commercial_share": [0.5, 0.5], "poi_transport_access_count": [0, 1]}).to_parquet(source, index=False)
    monkeypatch.setattr(roles, "REFERENCE_GRID_DIR", tmp_path / "grids")
    monkeypatch.setattr(roles, "ROOT", tmp_path)
    monkeypatch.setitem(roles.SPECS["poi"], "path", lambda city: source)
    local, summary, _ = roles.source_city_features("city", "poi", {"a"})
    assert local.grid_id.tolist() == ["a"]
    assert local.poi_count_log.iloc[0] == pytest.approx(np.log1p(1))
    assert summary.city_poi_log_total.iloc[0] == pytest.approx(np.log1p(10))
    assert summary.city_poi_grid_coverage.iloc[0] == 1


def test_sequence_binding_excludes_opening_and_post_years_and_keeps_missing_pre_year() -> None:
    state = pd.DataFrame({"treatment_order": [1], "city_key": ["city"], "grid_id": ["g"],
                          "station_event_id": ["event"], "opening_month": ["2025-12"]})
    context = pd.DataFrame({"treatment_order": [1], "alphaearth_grid_geometry_sha256": ["grid_hash"]})
    images = build_queue(state, 2025)
    images["grid_geometry_sha256"] = "grid_hash"
    images["satellite_image_available"] = images.calendar_year_complete & images.event_time.ne(-6)
    images["state_image_available"] = images.state_temporal_allowed & images.satellite_image_available
    images["is_temporal_proxy"] = False
    images["is_dry_run"] = False
    pre = roles.validate_sequence_binding(state, context, images, 2025)
    assert pre.event_time.tolist() == list(range(-6, 0))
    assert pre.state_image_available.tolist() == [False] + [True] * 5
    leaked = images.copy()
    leaked.loc[leaked.event_time.eq(0), "state_temporal_allowed"] = True
    with pytest.raises(ValueError, match="predictor/time masks"):
        roles.validate_sequence_binding(state, context, leaked, 2025)
    duplicate = pd.concat([images.iloc[:-1], images.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="distinct annual rows"):
        roles.validate_sequence_binding(state, context, duplicate, 2025)
