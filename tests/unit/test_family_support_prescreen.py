import pandas as pd
import pytest

from scripts.causal_python.run_causal_label_queue import (
    _FAMILY_SUPPORT_CACHE,
    family_has_observed_support,
)


@pytest.fixture(autouse=True)
def clear_cache():
    _FAMILY_SUPPORT_CACHE.clear()
    yield
    _FAMILY_SUPPORT_CACHE.clear()


def test_housing_no_observation_grid_skipped(tmp_path, monkeypatch):
    import scripts.causal_python.run_causal_label_queue as queue_module

    panel = tmp_path / "beijing.parquet"
    pd.DataFrame(
        {
            "grid_id": ["g1", "g2"],
            "observed_month": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "log_price_raw_median": [50000.0, None],
        }
    ).to_parquet(panel)
    monkeypatch.setattr(queue_module, "PANEL_HOUSING_MONTHLY_DIR", tmp_path)

    assert family_has_observed_support(pd.Series({"city_key": "beijing", "grid_id": "g1", "outcome_family": "housing"}))
    assert not family_has_observed_support(pd.Series({"city_key": "beijing", "grid_id": "g2", "outcome_family": "housing"}))
    assert not family_has_observed_support(pd.Series({"city_key": "beijing", "grid_id": "g_missing", "outcome_family": "housing"}))


def test_viirs_never_prescreened():
    assert family_has_observed_support(pd.Series({"city_key": "beijing", "grid_id": "anything", "outcome_family": "viirs"}))


def test_poi_observed_grids(tmp_path, monkeypatch):
    import scripts.causal_python.run_causal_label_queue as queue_module

    poi_dir = tmp_path / "poi"
    poi_dir.mkdir()
    pd.DataFrame(
        {
            "city_key": ["beijing"] * 2,
            "grid_id": ["g1", "g2"],
            "year": [2019, 2019],
            "poi_count_log": [5.0, None],
            "poi_category_entropy": [1.2, None],
        }
    ).to_parquet(poi_dir / "beijing_poi_grid_yearly.parquet")
    monkeypatch.setattr(queue_module, "POI_DIR", poi_dir)

    assert family_has_observed_support(pd.Series({"city_key": "beijing", "grid_id": "g1", "outcome_family": "poi"}))
    assert not family_has_observed_support(pd.Series({"city_key": "beijing", "grid_id": "g2", "outcome_family": "poi"}))
    assert not family_has_observed_support(pd.Series({"city_key": "beijing", "grid_id": "g_missing", "outcome_family": "poi"}))


def test_population_missing_panel_means_no_support(tmp_path, monkeypatch):
    import scripts.causal_python.run_causal_label_queue as queue_module

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(queue_module, "POPULATION_DIR", empty_dir)
    assert not family_has_observed_support(pd.Series({"city_key": "beijing", "grid_id": "g1", "outcome_family": "population"}))
