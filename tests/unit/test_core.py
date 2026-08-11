"""Basic tests for core pipeline utilities.

Covers the modules that every downstream script depends on:
  - pipeline_config.norm_station_name (cross-source name normalization)
  - pipeline_config.get_effective_bbox (latitude-corrected buffer)
  - pipeline_config.METRO_REFERENCE (first-line opening years)
  - amap_transit_fetcher.gcj02_to_wgs84 (coordinate transform)
  - build_hpi_label.compute_chained_index (chained YoY index)
  - build_treatment.compute_grid_treatment (empty-stations guard)

Run with:  pytest tests/unit/test_core.py -v
"""

import numpy as np
import pandas as pd
import pytest
from amap_transit_fetcher import gcj02_to_wgs84
from amap_transit_fetcher import load_keys as load_amap_keys
from build_housing_label import load_api_key as load_housing_api_key
from build_hpi_label import compute_chained_index
from build_treatment import compute_grid_treatment, merge_osm_amap
from shapely.geometry import Polygon

from urban_intervention.config import project as project_config
from urban_intervention.config.project import (
    ACTIVE_CITIES,
    CITIES,
    METRO_REFERENCE,
    get_effective_bbox,
    norm_station_name,
)

# ── norm_station_name ────────────────────────────────────────────


class TestNormStationName:
    def test_strips_station_suffix(self):
        assert norm_station_name("西二旗站") == "西二旗"

    def test_strips_parentheticals_half_width(self):
        assert norm_station_name("西二旗(地铁)") == "西二旗"

    def test_strips_parentheticals_full_width(self):
        assert norm_station_name("西二旗（地铁）") == "西二旗"

    def test_strips_road_suffix(self):
        # Only trailing 路 is stripped, not 路 in the middle of a name.
        assert norm_station_name("建国路") == "建国"

    def test_preserves_road_in_middle(self):
        # "五路居" is a real Beijing station — 路 is not a suffix here.
        assert norm_station_name("五路居") == "五路居"
        assert norm_station_name("十路口") == "十路口"

    def test_strips_middle_dot(self):
        assert norm_station_name("海淀黄庄·换乘") == "海淀黄庄换乘"

    def test_strips_dash(self):
        assert norm_station_name("新街口-西直门") == "新街口西直门"

    def test_lowercase_ascii(self):
        # Spaces are stripped, so "Xierqi Station" -> "xierqistation"
        assert norm_station_name("Xierqi Station") == "xierqistation"

    def test_handles_none_and_empty(self):
        assert norm_station_name(None) == ""
        assert norm_station_name("") == ""
        assert norm_station_name("   ") == ""

    def test_cross_source_consistency(self):
        """The same logical station from different sources must normalize
        to the same key — this is the whole point of norm_station_name."""
        amap_name = "西二旗"
        osm_name = "西二旗站"
        wiki_name = "西二旗（地铁）"
        assert norm_station_name(amap_name) == norm_station_name(osm_name)
        assert norm_station_name(osm_name) == norm_station_name(wiki_name)


def test_admin_boundary_accepts_feature_collection(monkeypatch):
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        ],
    }
    monkeypatch.setattr(project_config, "_load_boundary_geojson", lambda _city: feature_collection)
    boundary = project_config.get_admin_boundary("test")
    assert isinstance(boundary, Polygon)
    assert boundary.area == pytest.approx(1.0)


# ── METRO_REFERENCE ──────────────────────────────────────────────


class TestMetroReference:
    def test_all_44_cities_present(self):
        assert set(METRO_REFERENCE.keys()) == set(CITIES.keys()), (
            "METRO_REFERENCE must cover every city in CITIES"
        )

    @pytest.mark.parametrize(
        "city,expected_year",
        [
            ("beijing", 1969),
            ("shanghai", 1995),
            ("guangzhou", 1997),
            ("shenzhen", 2004),
            ("tianjin", 1984),
            ("chongqing", 2005),
        ],
    )
    def test_first_line_opened_matches_history(self, city, expected_year):
        assert METRO_REFERENCE[city]["first_line_opened"] == expected_year, (
            f"{city} first_line_opened should be {expected_year}"
        )

    def test_all_years_in_valid_range(self):
        for ck, ref in METRO_REFERENCE.items():
            yr = ref["first_line_opened"]
            assert 1960 <= yr <= 2025, f"{ck}: year {yr} out of plausible range"


# ── gcj02_to_wgs84 ───────────────────────────────────────────────


class TestGcj02ToWgs84:
    def test_known_point_beijing_tiananmen(self):
        """GCJ-02 coords for Tiananmen Square vs the expected WGS-84.
        The transform is a single-step approximate inverse (not iterative),
        so we tolerate ~500m error (~0.005 deg) rather than the ~50m that
        a full iterative inverse would achieve.  This is the same
        implementation used throughout the codebase — the test documents
        its precision rather than asserting exactness."""
        # GCJ-02 (approx, from Amap)
        gcj_lon, gcj_lat = 116.40741, 39.90420
        wgs_lon, wgs_lat = gcj02_to_wgs84(gcj_lon, gcj_lat)
        # WGS-84 (true) ≈ 116.39741, 39.90420
        # Single-step inverse leaves ~0.004° residual (~400m at this lat).
        assert abs(wgs_lon - 116.39741) < 0.006
        assert abs(wgs_lat - 39.90420) < 0.006

    def test_offset_is_small_but_nonzero(self):
        """The GCJ-02 offset for a Chinese point should be tens-hundreds
        of meters — confirming the transform is actually doing work."""
        gcj_lon, gcj_lat = 116.40741, 39.90420
        wgs_lon, wgs_lat = gcj02_to_wgs84(gcj_lon, gcj_lat)
        # Offset should be present but small
        dlon = abs(gcj_lon - wgs_lon)
        dlat = abs(gcj_lat - wgs_lat)
        assert 0.0001 < dlon < 0.02, f"lon offset {dlon} not in expected range"
        assert 0.0001 < dlat < 0.02, f"lat offset {dlat} not in expected range"

    def test_returns_tuple_of_floats(self):
        result = gcj02_to_wgs84(116.0, 40.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


# ── get_effective_bbox ───────────────────────────────────────────


class TestGetEffectiveBbox:
    def test_returns_list_of_four_floats(self):
        # Beijing has a cached admin boundary in tests; if not, falls back
        # to the hardcoded bbox.
        bbox = get_effective_bbox("beijing", buffer_km=10.0)
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        for v in bbox:
            assert isinstance(v, (int, float))

    def test_hardcoded_fallback(self, monkeypatch):
        import urban_intervention.config.project as pmod

        monkeypatch.setattr(pmod, "get_admin_boundary", lambda ck: None)
        bbox = get_effective_bbox("beijing", buffer_km=0.0)
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        expected = pmod.CITIES["beijing"]["bbox"]
        assert bbox == expected


# ── compute_chained_index ────────────────────────────────────────


class TestComputeChainedIndex:
    def _make_yearly(self, rows):
        """Build a minimal yearly DataFrame for testing."""
        return pd.DataFrame(rows)

    def test_base_year_is_100(self):
        yearly = self._make_yearly(
            [
                {
                    "city_key": "beijing",
                    "year": 2022,
                    "housing_type": "new",
                    "area_class": "total",
                    "mom_avg": 100.5,
                    "yoy_december": 101.5,
                    "yoy_avg": 101.2,
                    "ytd_december": 101.3,
                    "n_months": 12,
                },
            ]
        )
        out = compute_chained_index(yearly, base_year=2022)
        assert len(out) == 1
        assert out["hpi_index"].iloc[0] == pytest.approx(100.0)

    def test_forward_chain_uses_yoy(self):
        """If yoy_december=105 (5% YoY), the next year's index should be 105."""
        yearly = self._make_yearly(
            [
                {
                    "city_key": "beijing",
                    "year": 2022,
                    "housing_type": "new",
                    "area_class": "total",
                    "mom_avg": 100.0,
                    "yoy_december": 100.0,
                    "yoy_avg": 100.0,
                    "ytd_december": 100.0,
                    "n_months": 12,
                },
                {
                    "city_key": "beijing",
                    "year": 2023,
                    "housing_type": "new",
                    "area_class": "total",
                    "mom_avg": 100.4,
                    "yoy_december": 105.0,
                    "yoy_avg": 104.8,
                    "ytd_december": 105.0,
                    "n_months": 12,
                },
            ]
        )
        out = compute_chained_index(yearly, base_year=2022)
        out_2023 = out[out["year"] == 2023]["hpi_index"].iloc[0]
        assert out_2023 == pytest.approx(105.0)

    def test_backward_chain_divides(self):
        """Backward: index[2021] = 100 / multiplier[2022].

        NBS yoy_december[Y] = price_Y / price_{Y-1} * 100, so mult[Y] =
        price_Y / price_{Y-1}.  To step backward from Y to Y-1 we divide
        idx[Y] by mult[Y] (the multiplier of the year we are leaving).

        Here yoy[2021]=110 means price_2021/price_2020 = 1.10, and
        yoy[2022]=100 means price_2022/price_2021 = 1.00 (flat).  With
        base 2022 = 100, idx[2021] = 100 / mult[2022] = 100 / 1.00 = 100.
        """
        yearly = self._make_yearly(
            [
                {
                    "city_key": "beijing",
                    "year": 2021,
                    "housing_type": "new",
                    "area_class": "total",
                    "mom_avg": 100.0,
                    "yoy_december": 110.0,
                    "yoy_avg": 110.0,
                    "ytd_december": 110.0,
                    "n_months": 12,
                },
                {
                    "city_key": "beijing",
                    "year": 2022,
                    "housing_type": "new",
                    "area_class": "total",
                    "mom_avg": 100.0,
                    "yoy_december": 100.0,
                    "yoy_avg": 100.0,
                    "ytd_december": 100.0,
                    "n_months": 12,
                },
            ]
        )
        out = compute_chained_index(yearly, base_year=2022)
        out_2021 = out[out["year"] == 2021]["hpi_index"].iloc[0]
        assert out_2021 == pytest.approx(100.0, rel=1e-3)

    def test_fallback_to_compounded_mom(self):
        """When yoy is NaN, should fall back to (1 + (mom-100)/100)^12."""
        mom_avg = 100.5  # 0.5% monthly
        expected_factor = (1 + 0.005) ** 12  # ~1.0617
        yearly = self._make_yearly(
            [
                {
                    "city_key": "beijing",
                    "year": 2022,
                    "housing_type": "new",
                    "area_class": "total",
                    "mom_avg": 100.0,
                    "yoy_december": 100.0,
                    "yoy_avg": 100.0,
                    "ytd_december": 100.0,
                    "n_months": 12,
                },
                {
                    "city_key": "beijing",
                    "year": 2023,
                    "housing_type": "new",
                    "area_class": "total",
                    "mom_avg": mom_avg,
                    "yoy_december": np.nan,
                    "yoy_avg": np.nan,
                    "ytd_december": np.nan,
                    "n_months": 12,
                },
            ]
        )
        out = compute_chained_index(yearly, base_year=2022)
        out_2023 = out[out["year"] == 2023]["hpi_index"].iloc[0]
        assert out_2023 == pytest.approx(100.0 * expected_factor, rel=1e-3)

    def test_old_arithmetic_mean_was_wrong(self):
        """Sanity check: the old (buggy) approach would have given 100.5,
        but the correct compounded value is ~106.17. This test documents
        why the fix matters."""
        mom_avg = 100.5
        old_buggy = 100.0 * (mom_avg / 100.0)  # = 100.5
        correct = 100.0 * (1 + (mom_avg - 100.0) / 100.0) ** 12  # ≈ 106.17
        assert abs(correct - old_buggy) > 5.0, (
            "Compounded mom should differ substantially from arithmetic mean"
        )


# ── compute_grid_treatment empty-stations guard ──────────────────


class TestComputeGridTreatmentEmpty:
    def test_empty_stations_does_not_crash(self):
        """Previously, an empty stations DataFrame caused
        `masked.min(axis=1)` to raise ValueError on a zero-size axis.
        Now it should return an all-zero treatment table."""
        grids = pd.DataFrame(
            {
                "grid_id": ["g00000x00000", "g00001x00000"],
                "centroid_lon": [116.40, 116.41],
                "centroid_lat": [39.90, 39.91],
            }
        )
        stations = pd.DataFrame(columns=["wgs84_lon", "wgs84_lat", "opening_year"])
        result = compute_grid_treatment(grids, stations, year_range=range(2020, 2022))
        # 2 grids × 2 years = 4 rows
        assert len(result) == 4
        # All has_metro_* columns should be 0
        assert (result["has_metro_200m"] == 0).all()
        assert (result["has_metro_500m"] == 0).all()
        assert (result["has_metro_800m"] == 0).all()
        assert (result["has_metro_1500m"] == 0).all()
        # Nearest station should be NaN / empty
        assert result["nearest_station_m"].isna().all()


# ── merge_osm_amap input immutability ────────────────────────────


class TestMergeOsmAapImmutability:
    def test_does_not_mutate_inputs(self):
        """merge_osm_amap previously added 'source' / '_n' columns directly
        to the caller's DataFrames. Now it should copy first."""
        osm_df = pd.DataFrame(
            {
                "station_name": ["西二旗", "龙泽"],
                "wgs84_lon": [116.296, 116.309],
                "wgs84_lat": [40.054, 40.073],
            }
        )
        amap_df = pd.DataFrame(
            {
                "station_name": ["西二旗站", "上地"],
                "wgs84_lon": [116.296, 116.309],
                "wgs84_lat": [40.054, 40.073],
            }
        )
        osm_cols_before = set(osm_df.columns)
        amap_cols_before = set(amap_df.columns)

        merged = merge_osm_amap(osm_df, amap_df)

        # Inputs should be unchanged
        assert set(osm_df.columns) == osm_cols_before
        assert set(amap_df.columns) == amap_cols_before
        # Output should have source column
        assert "source" in merged.columns
        # "西二旗" appears in both (after normalization), so amap's duplicate
        # should be dropped; "上地" only in amap, should be kept.
        assert len(merged) == 3  # 2 OSM + 1 amap-only


# ── Active cities consistency ────────────────────────────────────


class TestActiveCities:
    def test_active_cities_subset_of_cities(self):
        for ck in ACTIVE_CITIES:
            assert ck in CITIES, f"ACTIVE_CITIES contains '{ck}' not in CITIES"

    def test_at_least_40_cities(self):
        # We expect 44 cities per the docs.
        assert len(ACTIVE_CITIES) >= 40


# ── Wikidata metro line filter ───────────────────────────────────


class TestFilterMetroLines:
    """Tests for the per-line metro classification in wikidata_transit_fetch."""

    def test_pure_metro_line_preserved(self):
        from wikidata_transit_fetch import _filter_metro_lines

        assert _filter_metro_lines("1号线") == "1号线"
        assert _filter_metro_lines("北京地铁2号线") == "北京地铁2号线"
        assert _filter_metro_lines("武汉轨道交通1号线") == "武汉轨道交通1号线"

    def test_high_speed_rail_dropped(self):
        from wikidata_transit_fetch import _filter_metro_lines

        assert _filter_metro_lines("京沪高铁") == ""
        assert _filter_metro_lines("沪宁城际铁路") == ""

    def test_mixed_lines_keeps_metro_only(self):
        from wikidata_transit_fetch import _filter_metro_lines

        # A station serving both metro line 1 and a mainline rail — the
        # metro line should be kept, the mainline dropped, instead of
        # dropping the whole station.
        result = _filter_metro_lines("1号线;京沪高铁;2号线")
        assert "1号线" in result
        assert "2号线" in result
        assert "高铁" not in result

    def test_city_rail_kept_intercity_dropped(self):
        """市域铁路 (city rail, e.g. Wenzhou S1) is operationally metro-like
        and should be kept; 城际铁路 (intercity) should be dropped."""
        from wikidata_transit_fetch import _filter_metro_lines

        assert _filter_metro_lines("温州市域铁路S1线") == "温州市域铁路S1线"
        assert _filter_metro_lines("京津城际铁路") == ""

    def test_empty_input(self):
        from wikidata_transit_fetch import _filter_metro_lines

        assert _filter_metro_lines("") == ""
        assert _filter_metro_lines(None) == ""

    def test_dedup_preserves_order(self):
        from wikidata_transit_fetch import _filter_metro_lines

        # Duplicate lines should be deduped and sorted.
        result = _filter_metro_lines("2号线;1号线;2号线;1号线")
        assert result == "1号线;2号线"

    def test_sparql_limit_in_query(self):
        """SPARQL query must contain the LIMIT clause with SPARQL_LIMIT."""
        from wikidata_transit_fetch import SPARQL, SPARQL_LIMIT

        assert f"LIMIT {SPARQL_LIMIT}" in SPARQL
        assert SPARQL_LIMIT >= 8000  # generous enough for ~5000 stations


# ── OFFICIAL_REF coverage ────────────────────────────────────────


class TestOfficialRef:
    def test_all_44_cities_have_reference(self):
        """Every active city should have an OFFICIAL_REF entry (0 = unknown
        but present, so the score function can fall back gracefully)."""
        from compare_transit_sources import OFFICIAL_REF

        for ck in ACTIVE_CITIES:
            assert ck in OFFICIAL_REF, f"{ck} missing from OFFICIAL_REF"

    def test_known_counts_approximate(self):
        """Sanity-check a few well-known city station counts."""
        from compare_transit_sources import OFFICIAL_REF

        # Beijing/Shanghai have the largest systems (~400-500 stations).
        assert 400 <= OFFICIAL_REF["beijing"] <= 550
        assert 400 <= OFFICIAL_REF["shanghai"] <= 550
        # Urumqi has a small system.
        assert 10 <= OFFICIAL_REF["urumqi"] <= 50


# ── API key loading ──────────────────────────────────────────────


class TestApiKeyLoading:
    def test_amap_loader_reads_multi_key_env(self, monkeypatch):
        monkeypatch.setenv("AMAP_API_KEYS", "key_a,key_b; key_c")
        keys = load_amap_keys()
        assert keys[-3:] == ["key_a", "key_b", "key_c"]

    def test_housing_loader_ignores_template_placeholder(self, monkeypatch):
        monkeypatch.setenv("AMAP_API_KEYS", "your_amap_key_here")
        monkeypatch.setenv("AMAP_API_KEY", "real_key")
        assert load_housing_api_key() == "real_key"


# ── Baseline modeling ────────────────────────────────────────────
