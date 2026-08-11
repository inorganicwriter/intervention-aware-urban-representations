from __future__ import annotations

import numpy as np
import pandas as pd

from urban_intervention.pipelines.housing.panel import (
    attach_time_fields,
    finalize_observations,
    join_flags,
    source_balanced_panel,
)


def test_annual_observation_does_not_invent_month() -> None:
    result = attach_time_fields(
        pd.DataFrame({"date": [pd.Timestamp("2025-01-01")]}), "date", "year"
    )
    assert result.loc[0, "observed_year"] == 2025
    assert pd.isna(result.loc[0, "observed_month"])
    assert pd.isna(result.loc[0, "observed_quarter"])


def test_source_balancing_prevents_large_source_from_dominating() -> None:
    frame = pd.DataFrame(
        {
            "city_key": ["beijing"] * 101,
            "grid_id": ["g1"] * 101,
            "observed_month": [pd.Timestamp("2020-01-01")] * 101,
            "source_id": ["large_listing_source"] * 100 + ["small_transaction_source"],
            "price_cny_m2": [10_000.0] * 100 + [40_000.0],
            "observation_id": [str(index) for index in range(101)],
            "price_stage": ["listing"] * 100 + ["transaction"],
            "analysis_eligible_month": [True] * 101,
        }
    )
    _, panel = source_balanced_panel(frame, ["observed_month"], "analysis_eligible_month")
    assert panel.loc[0, "price_raw_median_cny_m2"] == 10_000
    assert np.isclose(panel.loc[0, "price_source_balanced_cny_m2"], 20_000)


def test_finalize_observations_keeps_noncanonical_row_but_excludes_analysis() -> None:
    frame = pd.DataFrame(
        {
            "observation_id": ["a", "b"],
            "source_record_id": ["1", "2"],
            "source_id": ["source", "source"],
            "city_key": ["beijing", "beijing"],
            "price_cny_m2": [50_000.0, 60_000.0],
            "grid_id": ["g1", "g1"],
            "observed_month": [pd.Timestamp("2020-01-01")] * 2,
            "observed_quarter": ["2020Q1"] * 2,
            "observed_year": [2020, 2020],
            "canonical_for_aggregation": [True, False],
        }
    )
    result = finalize_observations(frame)
    assert len(result) == 2
    assert result["analysis_eligible_month"].tolist() == [True, False]


def test_join_flags_is_sorted_and_unique() -> None:
    assert join_flags("b;a", "a;c", "") == "a;b;c"
