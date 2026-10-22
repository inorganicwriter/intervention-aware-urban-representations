from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "build_transit_snapshots.py"
SPEC = importlib.util.spec_from_file_location("transit_snapshot_tests", SCRIPT)
transit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transit)


def _stations(points: list[tuple[float, float]], dates: list[str] | None = None) -> pd.DataFrame:
    n = len(points)
    return pd.DataFrame(
        {
            "station_event_id": [f"s{i}" for i in range(n)],
            "_norm": [f"station{i}" for i in range(n)],
            "wgs84_lon": [p[0] for p in points],
            "wgs84_lat": [p[1] for p in points],
            "opening_date": pd.to_datetime(dates or ["2010-01-01"] * n),
            "lines": [f"line{i}" for i in range(n)],
        }
    )


def _edges(pairs: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_norm": [f"station{a}" for a, _ in pairs],
            "_norm_adj": [f"station{b}" for _, b in pairs],
        }
    )


def _grid(points: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "grid_id": [f"g{i}" for i in range(len(points))],
            "centroid_lon": [p[0] for p in points],
            "centroid_lat": [p[1] for p in points],
        }
    )


def _snapshot(stations, edges=None, grids=None, **kwargs):
    return transit.build_snapshot(
        "test",
        pd.Timestamp("2020-01-01"),
        stations,
        _edges([]) if edges is None else edges,
        _grid([(116.0, 40.0)]) if grids is None else grids,
        **kwargs,
    )


def test_spherical_distance_and_1500m_inclusion() -> None:
    result = _snapshot(_stations([(116.017, 40.0)]))
    expected = transit._haversine_km(116, 40, 116.017, 40) * 1000
    assert result.loc[0, "dist_nearest_station_m"] == pytest.approx(expected, abs=1e-6)
    assert result.loc[0, "stations_1500m"] == 1
    assert result.loc[0, "stations_800m"] == 0
    assert result.loc[0, "lines_in_1500m"] == 1


def test_nearest_station_uses_spherical_ranking() -> None:
    stations = _stations([(116.01, 40.0), (116, 40.009)])
    result = _snapshot(stations)
    assert result.loc[0, "dist_nearest_station_m"] == pytest.approx(
        transit._haversine_km(116, 40, 116.01, 40) * 1000, abs=1e-6
    )
    assert result.loc[0, "stations_800m"] == 0


def test_buffer_counts_include_more_than_25_stations() -> None:
    stations = _stations([(116 + i / 100000, 40) for i in range(35)])
    result = _snapshot(stations)
    for radius in (500, 800, 1500):
        assert result.loc[0, f"stations_{radius}m"] == 35
    assert result.loc[0, "lines_in_1500m"] == 35


def test_future_station_and_injected_full_graph_scores_cannot_change_history() -> None:
    old = _stations([(116, 40), (116.01, 40)])
    full = _stations(
        [(116, 40), (116.01, 40), (116.02, 40)],
        ["2010-01-01", "2010-01-01", "2025-01-01"],
    )
    edges = _edges([(0, 1), (1, 2)])
    cache = {}
    clean = _snapshot(old, edges, closeness_cache=cache)
    with_future = _snapshot(full, edges, closeness={"s0": 999}, closeness_cache=cache)
    pd.testing.assert_frame_equal(clean, with_future)
    assert clean.loc[0, "network_closeness"] > 0
    assert len(cache) == 1


def test_graph_cache_changes_with_coordinates() -> None:
    cache = {}
    old = _stations([(116, 40), (116.01, 40)])
    edges = _edges([(0, 1)])
    first = _snapshot(old, edges, closeness_cache=cache)
    old.loc[1, "wgs84_lon"] += 0.01
    second = _snapshot(old, edges, closeness_cache=cache)
    assert len(cache) == 2
    assert second.loc[0, "network_closeness"] == pytest.approx(
        first.loc[0, "network_closeness"] / 2, rel=1e-7
    )


@pytest.mark.parametrize("column,date", [("valid_from", "2021-01-01"), ("valid_to", "2018-01-01")])
def test_dated_edges_respect_cutoff(column: str, date: str) -> None:
    stations = _stations([(116, 40), (116.01, 40)])
    edges = _edges([(0, 1)])
    edges[column] = date
    assert _snapshot(stations, edges).loc[0, "network_closeness"] == 0


def test_cutoff_boundary_and_invalid_coordinates() -> None:
    stations = _stations([(116, 40), (116.01, 40)], ["2019-01-01", "2019-01-02"])
    result = _snapshot(stations, grids=_grid([(116, 40), (np.nan, 40)]))
    assert result.loc[0, "stations_1500m"] == 1
    assert result.loc[0, "dist_nearest_station_m"] == 0
    assert result.loc[1, "stations_1500m"] == 0
    assert np.isnan(result.loc[1, "dist_nearest_station_m"])
    result = _snapshot(_stations([(116, 40)], ["2021-01-01"]))
    assert result.loc[0, "stations_1500m"] == 0
    assert np.isnan(result.loc[0, "dist_nearest_station_m"])


def test_versioned_cli_cache_does_not_accept_legacy_or_modified_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    causal = tmp_path / "data" / "active" / "causal"
    causal.mkdir(parents=True)
    pd.DataFrame({"city_key": ["test"], "opening_month": ["2020-01-01"]}).to_csv(
        causal / "treatment_unit_list.csv", index=False
    )
    grid_root = tmp_path / "grids"
    (grid_root / "test").mkdir(parents=True)
    _grid([(116, 40)]).to_parquet(grid_root / "test" / "test_grids.parquet", index=False)
    monkeypatch.setattr(transit, "BASE_DIR", tmp_path)
    monkeypatch.setattr(transit, "GRID_DIR", grid_root)
    monkeypatch.setattr(transit, "_load_city_stations", lambda city: _stations([(116.017, 40)]))
    monkeypatch.setattr(transit, "_load_adjacency_edges", lambda city: _edges([]))
    output = tmp_path / "corrected"
    argv = [str(SCRIPT), "--city", "test", "--output-dir", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    assert transit.main() == 0
    target = output / "test" / "2020-01.parquet"
    metadata = target.with_suffix(".metadata.json")
    record = json.loads(metadata.read_text())
    assert record["identity"]["schema"] == transit.SNAPSHOT_SCHEMA
    assert record["identity"]["edge_dates_available"] is False
    original_bytes = target.read_bytes()
    assert transit.main() == 0
    assert target.read_bytes() == original_bytes
    frame = pd.read_parquet(target)
    frame.loc[0, "stations_1500m"] = 999
    frame.to_parquet(target, index=False)
    with pytest.raises(ValueError, match="Incompatible or unversioned"):
        transit.main()
    monkeypatch.setattr(sys, "argv", [*argv, "--overwrite"])
    assert transit.main() == 0
    assert pd.read_parquet(target).loc[0, "stations_1500m"] == 1
    metadata.unlink()
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="Incompatible or unversioned"):
        transit.main()
