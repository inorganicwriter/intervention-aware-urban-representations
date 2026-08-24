from __future__ import annotations

import numpy as np
import pandas as pd

from urban_intervention.causal.setup_inputs import (
    audit_family_support,
    build_eligible_donors,
    build_housing_annual,
    build_pending_queues,
)


def test_pending_queue_shapes_and_order_are_frozen() -> None:
    count = 5_048
    treatments = pd.DataFrame(
        {
            "treatment_order": np.arange(count, 0, -1),
            "city_key": [f"c{index // 200}" for index in range(count)],
            "grid_id": [f"g{index}" for index in range(count)],
            "station_event_id": [f"s{index}" for index in range(count)],
            "opening_month": "2020-01",
        }
    )
    queues = build_pending_queues(treatments)
    assert len(queues["unit"]) == count
    assert len(queues["control"]) == count
    assert len(queues["family"]) == count * 4
    assert queues["unit"]["treatment_order"].is_monotonic_increasing
    assert set(queues["family"]["status"]) == {"pending"}


def test_donor_filter_and_housing_annual_aggregation() -> None:
    universe = pd.DataFrame(
        {
            "city_key": ["a"] * 4,
            "grid_id": ["g1", "g2", "g3", "g4"],
            "is_nonexperimental_grid": [True, True, True, False],
            "known_station_contamination": [False, True, False, False],
            "primary_spatial_exclusion_reason": [
                "eligible_spatial_donor",
                "eligible_spatial_donor",
                "other",
                "eligible_spatial_donor",
            ],
        }
    )
    assert build_eligible_donors([universe])["unit_id"].tolist() == ["a::g1"]
    housing = pd.DataFrame(
        {
            "city_key": ["a"] * 3,
            "grid_id": ["g1"] * 3,
            "observed_month": pd.to_datetime(["2020-01-01", "2020-02-01", "2021-01-01"]),
            "log_price_raw_median": [1.0, 3.0, 5.0],
            "n_observations": [2, 3, 4],
        }
    )
    annual = build_housing_annual(housing)
    assert annual["housing_log_price"].tolist() == [2.0, 5.0]
    assert annual["housing_observations"].tolist() == [5, 4]


def test_target_support_requires_three_matching_lags_and_five_clean_pre_years() -> None:
    treatments = pd.DataFrame(
        {
            "treatment_order": [1, 2],
            "city_key": ["a", "a"],
            "grid_id": ["g1", "g2"],
            "opening_year": [2020, 2020],
        }
    )
    rows: list[dict[str, object]] = []
    for grid in ("g1", "g2"):
        for year in range(2015, 2024):
            value = float(year)
            if grid == "g2" and year == 2019:
                value = np.nan
            rows.append(
                {
                    "city_key": "a",
                    "grid_id": grid,
                    "year": year,
                    "population_log": value,
                }
            )
    support = audit_family_support(treatments, pd.DataFrame(rows), "population")
    first = support.loc[support["treatment_order"].eq(1)].iloc[0]
    second = support.loc[support["treatment_order"].eq(2)].iloc[0]
    assert bool(first["population_complete"])
    assert bool(first["population_gsc_ready"])
    assert not bool(second["population_complete"])
    assert not bool(second["population_gsc_ready"])
