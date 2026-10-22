"""Tests for the street-view index builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box

from scripts.collection.build_streetview_index import (
    assign_grids_bbox,
    assign_grids_polygon,
    build_city_index,
    read_metadata,
    resolve_image_path,
)

SAMPLE_ROW = (
    "Anhui,Anqing,Daguan,117.020255031731,30.511363744707783,"
    "13028104.674322821,3547978.5579769365,09009100121804201622108201J,"
    "1302811661,354797704,203.602,2018-04-01,成功\n"
)
HEADER = (
    "province,city,county,wgs_lon,wgs_lat,calc_bd_x,calc_bd_y,panoid,"
    "ret_bd_x,ret_bd_y,heading,capture_date,status\n"
)


def _lattice() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "grid_id": ["g00", "g01", "g10", "g11"],
            "lat_min": [30.0, 30.0, 30.005, 30.005],
            "lat_max": [30.005, 30.005, 30.010, 30.010],
            "lon_min": [117.0, 117.005, 117.0, 117.005],
            "lon_max": [117.005, 117.011, 117.005, 117.011],
        }
    )


def test_read_metadata_filters_status_and_invalid_rows(tmp_path: Path) -> None:
    path = tmp_path / "meta.csv"
    path.write_text(
        HEADER
        + SAMPLE_ROW
        + "Anhui,Anqing,Daguan,117.1,30.5,0,0,P2,0,0,1,2018-05-01,失败\n"
        + "Anhui,Anqing,Daguan,,30.5,0,0,P3,0,0,1,2018-06-01,成功\n",
        encoding="utf-8",
    )
    data = read_metadata([path], "成功")
    assert len(data) == 1
    assert data["panoid"].iloc[0] == "09009100121804201622108201J"
    assert data["capture_date"].iloc[0] == pd.Timestamp("2018-04-01")


def test_assign_grids_bbox_matches_lattice() -> None:
    grids = _lattice()
    longitudes = np.array([117.002, 117.002, 117.006, 117.016, 116.999])
    latitudes = np.array([30.002, 30.007, 30.007, 30.002, 30.002])
    assigned = assign_grids_bbox(longitudes, latitudes, grids)
    normalized = [value if isinstance(value, str) else None for value in assigned.tolist()]
    assert normalized == ["g00", "g10", "g11", None, None]


def test_assign_grids_bbox_shared_edge_belongs_to_next_cell() -> None:
    grids = _lattice()
    assigned = assign_grids_bbox(
        np.array([117.005, 117.005]), np.array([30.006, 30.004]), grids
    )
    assert assigned.tolist() == ["g11", "g01"]


def test_resolve_image_path(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "abc.jpg").write_bytes(b"x")
    (images / "def_1.png").write_bytes(b"x")
    assert resolve_image_path(images, "abc") == images / "abc.jpg"
    assert resolve_image_path(images, "def") == images / "def_1.png"
    assert resolve_image_path(images, "missing") is None


def test_polygon_assignment_rejects_bbox_corner_and_shared_boundary() -> None:
    # Rotated cells model the reference UTM grid after WGS84 reprojection.
    first = Polygon([(0, 0), (2, 1), (1, 3), (-1, 2)])
    second = Polygon([(2, 1), (4, 2), (3, 4), (1, 3)])
    grids = pd.DataFrame(
        {"grid_id": ["first", "second"], "geometry_wkt": [first.wkt, second.wkt]}
    )
    assigned = assign_grids_polygon(
        np.array([0.5, 2.5, 1.8, 1.5, np.nan]),
        np.array([1.5, 2.5, 2.8, 2.0, 1.5]),
        grids,
    )
    # 1.8,2.8 is in the second polygon, although inside the first one's bbox.
    assert assigned.tolist() == ["first", "second", "second", None, None]


def test_polygon_assignment_inside_bbox_but_outside_polygon() -> None:
    polygon = Polygon([(0, 0), (2, 1), (1, 3), (-1, 2)])
    grids = pd.DataFrame({"grid_id": ["g"], "geometry_wkt": [polygon.wkt]})
    assert assign_grids_polygon(np.array([1.8]), np.array([2.8]), grids).tolist() == [None]


def test_build_city_index_end_to_end(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "09009100121804201622108201J.jpg").write_bytes(b"x")
    meta = tmp_path / "meta.csv"
    meta.write_text(
        HEADER
        + SAMPLE_ROW
        + "Anhui,Anqing,Daguan,117.0205,30.5115,0,0,P_MISSING,0,0,1,2018-05-01,成功\n",
        encoding="utf-8",
    )
    grids = pd.DataFrame(
        {
            "grid_id": ["g00000x00000"],
            "lat_min": [30.511],
            "lat_max": [30.512],
            "lon_min": [117.019],
            "lon_max": [117.022],
            "geometry_wkt": [box(117.019, 30.511, 117.022, 30.512).wkt],
        }
    )
    frame, audit = build_city_index(
        "anqing", [meta], images, grids, status_value="成功"
    )
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["city_key"] == "anqing"
    assert row["grid_id"] == "g00000x00000"
    assert Path(row["asset_path"]) == (images / "09009100121804201622108201J.jpg").resolve()
    assert audit["metadata_rows"] == 2
    assert audit["matched_grid_rows"] == 2
    assert audit["rows_with_image"] == 1
    assert audit["missing_image_rows"] == 1
