"""Tests for reviewed transit station resolution."""

from __future__ import annotations

import pandas as pd

from urban_intervention.config.project import norm_station_name
from urban_intervention.interventions.transit.station_names import normalize_station_name
from urban_intervention.interventions.transit.station_resolution import (
    compile_station_resolution,
)


def _event(city: str, event_id: str, name: str, year: int, line: str) -> dict:
    return {
        "city_key": city,
        "station_event_id": event_id,
        "canonical_station_name": name,
        "normalized_name": name,
        "lines": line,
        "wgs84_lon": 120.0,
        "wgs84_lat": 30.0,
        "opening_year": year,
        "opening_date": f"{year}-01-01",
        "station_sources": "test",
        "raw_record_count": 1,
    }


def test_station_name_normalization_is_shared() -> None:
    variants = ["建国路站", "西二旗（地铁）", "海淀黄庄·换乘", None]
    assert [normalize_station_name(value) for value in variants] == [
        norm_station_name(value) for value in variants
    ]


def test_missing_competing_opening_year_keeps_nullable_censor() -> None:
    source = pd.DataFrame(
        [
            _event("a", "metro", "主站", 2019, "Metro"),
            {
                **_event("a", "other", "接驳站", 2021, "Other"),
                "opening_year": pd.NA,
                "opening_date": pd.NaT,
            },
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "issue_type": "multiple_station_events_in_grid",
                "original_city_key": "a",
                "grid_id": "g1",
                "station_names": "主站;接驳站",
                "review_decision": "distinct_modes_same_interchange",
                "canonical_station_name": "主站",
                "primary_station_name": "主站",
                "verified_city_key": "a",
                "study_disposition": "keep_primary_and_censor_when_known",
                "review_basis": "test",
            }
        ]
    )

    resolved, _, _, _ = compile_station_resolution(source, decisions, active_cities=["a"])
    primary = resolved.loc[resolved.station_event_id == "metro"].iloc[0]
    assert pd.isna(primary.post_treatment_censor_year)
    assert primary.resolution_status == ("primary_mode_with_competing_intervention_date_unknown")


def test_compile_station_resolution_applies_all_review_actions() -> None:
    source = pd.DataFrame(
        [
            _event("a", "m1", "旧名站", 2015, "L1"),
            _event("a", "m2", "新名站", 2015, "L2"),
            _event("a", "d1", "相邻甲站", 2018, "L3"),
            _event("a", "d2", "相邻乙站", 2020, "L4"),
            _event("a", "p1", "地铁站", 2019, "Metro"),
            _event("a", "p2", "云巴站", 2021, "Yunba"),
            _event("a", "r1", "改归站", 2020, "L5"),
            _event("a", "x1", "样本外站", 2020, "Tram"),
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "issue_type": "multiple_station_events_in_grid",
                "original_city_key": "a",
                "grid_id": "g1",
                "station_names": "旧名站;新名站",
                "review_decision": "same_physical_station",
                "canonical_station_name": "正式站",
                "primary_station_name": "旧名站",
                "verified_city_key": "a",
                "study_disposition": "merge_to_primary_metro_event",
                "review_basis": "test",
            },
            {
                "issue_type": "multiple_station_events_in_grid",
                "original_city_key": "a",
                "grid_id": "g2",
                "station_names": "相邻甲站;相邻乙站",
                "review_decision": "distinct_physical_stations",
                "canonical_station_name": "",
                "primary_station_name": "",
                "verified_city_key": "a",
                "study_disposition": "exclude_grid_from_primary_design",
                "review_basis": "test",
            },
            {
                "issue_type": "multiple_station_events_in_grid",
                "original_city_key": "a",
                "grid_id": "g3",
                "station_names": "地铁站;云巴站",
                "review_decision": "distinct_modes_same_interchange",
                "canonical_station_name": "地铁站",
                "primary_station_name": "地铁站",
                "verified_city_key": "a",
                "study_disposition": "keep_metro_primary_censor_at_yunba_opening",
                "review_basis": "test",
            },
            {
                "issue_type": "outside_reference_grid",
                "original_city_key": "a",
                "grid_id": "",
                "station_names": "改归站",
                "review_decision": "wrong_city_assignment",
                "canonical_station_name": "",
                "primary_station_name": "",
                "verified_city_key": "b",
                "study_disposition": "reassign_to_active_city",
                "review_basis": "test",
            },
            {
                "issue_type": "outside_reference_grid",
                "original_city_key": "a",
                "grid_id": "",
                "station_names": "样本外站",
                "review_decision": "wrong_city_assignment",
                "canonical_station_name": "",
                "primary_station_name": "",
                "verified_city_key": "outside",
                "study_disposition": "exclude_outside_study_universe",
                "review_basis": "test",
            },
        ]
    )

    resolved, competing, excluded, manifest = compile_station_resolution(
        source, decisions, active_cities=["a", "b"]
    )

    assert len(resolved) == 5
    merged = resolved[resolved.station_event_id == "m1"].iloc[0]
    assert merged.canonical_station_name == "正式站"
    assert merged.lines == "L1;L2"
    assert merged.original_station_event_ids == "m1;m2"
    distinct = resolved[resolved.station_event_id.isin(["d1", "d2"])]
    assert distinct.primary_design_excluded.all()
    assert set(distinct.primary_design_exclusion_reason) == {"multiple_distinct_stations_in_grid"}
    primary = resolved[resolved.station_event_id == "p1"].iloc[0]
    assert primary.competing_event_ids == "p2"
    assert primary.post_treatment_censor_year == 2021
    assert resolved.loc[resolved.station_event_id == "r1", "city_key"].item() == "b"
    assert competing.station_event_id.tolist() == ["p2"]
    assert set(excluded.event_disposition) == {
        "merged_into_primary_event",
        "primary_design_excluded_retained_for_spatial_exposure",
        "moved_to_competing_events",
        "outside_active_city_universe",
    }
    assert manifest["resolution_rows"] == 5
    assert manifest["source_events_touched"] == 8
    assert manifest["primary_design_excluded_events"] == 2
