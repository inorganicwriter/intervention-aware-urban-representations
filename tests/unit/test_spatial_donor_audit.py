"""Tests for the outcome-free DDR-001 spatial donor audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from urban_intervention.causal.spatial_donors import (
    SpatialDonorSpec,
    build_data_quality_issues,
    build_treated_grid_registry,
    compute_spatial_exposure,
)


def _stations(points_and_years: list[tuple[float, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city_key": ["test_city"] * len(points_and_years),
            "station_event_id": [f"s{i}" for i in range(len(points_and_years))],
            "canonical_station_name": [f"Station {i}" for i in range(len(points_and_years))],
            "wgs84_lon": [value[0] for value in points_and_years],
            "wgs84_lat": [value[1] for value in points_and_years],
            "opening_year": [value[2] for value in points_and_years],
            "opening_date": [f"{value[2]}-01-01" for value in points_and_years],
            "date_precision": ["day"] * len(points_and_years),
        }
    )


def test_point_to_polygon_distance_controls_primary_donor_eligibility() -> None:
    grids = pd.DataFrame(
        {
            "grid_id": ["treated", "near", "eligible"],
            "geometry_wkt": ["unused"] * 3,
        }
    )
    polygons = np.array(
        [
            box(0, 0, 500, 500),
            box(1_150, 0, 1_650, 500),
            box(1_500, 600, 2_000, 1_100),
        ],
        dtype=object,
    )
    station_points = np.array([Point(250, 250)], dtype=object)
    stations = _stations([(250, 250, 2020)])
    exposure, mapping = compute_spatial_exposure(
        "test_city", grids, stations, polygons, station_points, SpatialDonorSpec()
    )

    distance = exposure.set_index("grid_id")["nearest_station_polygon_distance_m"]
    assert distance["treated"] == 0.0
    assert distance["near"] == 900.0
    assert distance["eligible"] > 1_000.0
    eligible = exposure.set_index("grid_id")["spatial_donor_eligible_1000m"]
    assert not bool(eligible["treated"])
    assert not bool(eligible["near"])
    assert bool(eligible["eligible"])
    assert mapping.iloc[0]["mapping_status"] == "unique"
    assert mapping.iloc[0]["candidate_grid_ids"] == "treated"
    assert mapping.iloc[0]["wgs84_lon"] == 250.0
    assert mapping.iloc[0]["wgs84_lat"] == 250.0


def test_multiple_station_events_in_one_grid_fail_unique_treatment_flag() -> None:
    grids = pd.DataFrame({"grid_id": ["g1"], "geometry_wkt": ["unused"]})
    polygons = np.array([box(0, 0, 500, 500)], dtype=object)
    station_points = np.array([Point(100, 100), Point(400, 400)], dtype=object)
    stations = _stations([(100, 100, 2018), (400, 400, 2020)])
    exposure, mapping = compute_spatial_exposure(
        "test_city", grids, stations, polygons, station_points, SpatialDonorSpec()
    )
    treated = build_treated_grid_registry(exposure, mapping, SpatialDonorSpec())

    assert exposure.iloc[0]["station_event_count_in_grid"] == 2
    assert len(treated) == 2
    assert not treated["unique_treatment_event"].any()
    assert not treated["analysis_treated_grid"].any()
    assert set(treated["treatment_exclusion_reason"]) == {"multiple_station_events_in_grid"}
    issues = build_data_quality_issues(mapping, treated)
    assert len(issues) == 1
    assert issues.iloc[0]["issue_type"] == "multiple_station_events_in_grid"
    assert set(issues.iloc[0]["station_event_ids"].split(";")) == {"s0", "s1"}
    assert issues.iloc[0]["station_wgs84_lons"].startswith("100")
    assert issues.iloc[0]["station_wgs84_lats"].startswith("100")


def test_station_on_grid_boundary_is_reported_as_ambiguous() -> None:
    grids = pd.DataFrame(
        {
            "grid_id": ["left", "right"],
            "geometry_wkt": ["unused", "unused"],
        }
    )
    polygons = np.array([box(0, 0, 500, 500), box(500, 0, 1_000, 500)], dtype=object)
    station_points = np.array([Point(500, 250)], dtype=object)
    stations = _stations([(500, 250, 2020)])
    _, mapping = compute_spatial_exposure(
        "test_city", grids, stations, polygons, station_points, SpatialDonorSpec()
    )

    assert mapping.iloc[0]["mapping_status"] == "boundary_multiple"
    assert mapping.iloc[0]["candidate_grid_count"] == 2
    assert set(mapping.iloc[0]["candidate_grid_ids"].split(";")) == {"left", "right"}


def test_reviewed_distinct_station_grid_is_excluded_not_reopened_as_issue() -> None:
    grids = pd.DataFrame({"grid_id": ["g1"], "geometry_wkt": ["unused"]})
    polygons = np.array([box(0, 0, 500, 500)], dtype=object)
    station_points = np.array([Point(100, 100), Point(400, 400)], dtype=object)
    stations = _stations([(100, 100, 2018), (400, 400, 2020)])
    stations["resolution_status"] = "resolved_distinct_station_grid_exclusion"
    stations["resolution_grid_id"] = "g1"
    stations["primary_design_excluded"] = True
    stations["primary_design_exclusion_reason"] = "multiple_distinct_stations_in_grid"
    stations["competing_event_ids"] = ""
    stations["post_treatment_censor_year"] = pd.NA
    exposure, mapping = compute_spatial_exposure(
        "test_city", grids, stations, polygons, station_points, SpatialDonorSpec()
    )
    treated = build_treated_grid_registry(exposure, mapping, SpatialDonorSpec())

    assert not treated["analysis_treated_grid"].any()
    assert set(treated["treatment_exclusion_reason"]) == {"multiple_distinct_stations_in_grid"}
    assert build_data_quality_issues(mapping, treated).empty


def test_spatial_spec_rejects_non_ddr_distance_metric() -> None:
    spec = SpatialDonorSpec(distance_metric="centroid")
    try:
        spec.validate()
    except ValueError as exc:
        assert "point_to_polygon_minimum" in str(exc)
    else:
        raise AssertionError("invalid distance metric should fail")
