"""Actual capture coordinates and month-level cutoffs must survive rebuilding."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from shapely.geometry import box

from scripts.analysis.publish_server_streetview_inputs import copy_verified_image, sha256
from scripts.analysis.publish_streetview_inputs import assert_binding
from scripts.collection.rebuild_remote_streetview_index import (
    assign_unique,
    select_pre_assets,
    select_verified_pre,
    verify_image,
)
from scripts.collection.rebuild_remote_streetview_post_archive import post_candidates, post_coverage
from scripts.collection.streetview_coordinates import (
    actual_pano_coordinates,
    gcj_to_wgs,
    wgs_to_gcj,
)


def test_panorama_scaled_mc_and_unknown_units():
    ret_x = np.array([1290016997., 12900169.97, 12900169970.])
    ret_y = np.array([489153950., 4891539.50, 4891539500.])
    calc_x, calc_y = np.full(3, 12900149.18), np.full(3, 4891512.91)
    lon, lat, scale = actual_pano_coordinates(ret_x, ret_y, calc_x, calc_y)
    assert np.allclose(lon[:2], 115.87050, atol=.0005, rtol=0)
    assert np.allclose(lat[:2], 40.36002, atol=.0005, rtol=0)
    assert scale[:2].tolist() == [100., 1.]
    assert np.isnan(lon[2]) and np.isnan(lat[2])


def test_iterative_gcj_inverse():
    lon, lat = np.array([116.4, 121.5, 104.0]), np.array([39.9, 31.2, 30.6])
    gcj_lon, gcj_lat = wgs_to_gcj(lon, lat)
    x, y = gcj_to_wgs(gcj_lon, gcj_lat)
    np.testing.assert_allclose(x, lon, atol=1e-10, rtol=0)
    np.testing.assert_allclose(y, lat, atol=1e-10, rtol=0)


def test_month_cutoff_rejects_opening_month_and_preserves_event_specificity():
    events = pd.DataFrame({"treatment_order": [1, 2], "city_key": ["c", "c"],
                           "grid_id": ["g", "g"], "station_event_id": ["a", "b"],
                           "opening_month": ["2018-04", "2018-06"]})
    index = pd.DataFrame({"city_key": ["c"] * 3, "grid_id": ["g"] * 3,
                          "capture_month": ["2018-03", "2018-04", "2018-06"],
                          "panoid": ["p1", "p2", "p3"], "asset_path": ["a", "b", "c"]})
    result = select_pre_assets(events, index, 2)
    assert result.groupby("treatment_order").panoid.agg(list).to_dict() == {1: ["p1"], 2: ["p2", "p1"]}


def test_post_archive_keeps_same_month_uncertainty_and_masks_future_years():
    events = pd.DataFrame({"treatment_order": [1, 2], "city_key": ["c", "c"],
                           "grid_id": ["g", "g"], "station_event_id": ["a", "b"],
                           "opening_month": ["2018-04", "2024-04"]})
    index = pd.DataFrame({"city_key": ["c"] * 6, "grid_id": ["g"] * 6,
                          "capture_month": ["2018-03", "2018-04", "2018-05", "2019-01", "2022-01", "2024-04"],
                          "panoid": ["a", "b", "c", "d", "e", "f"]})
    eligible, all_year = post_candidates(events, index)
    assert eligible.panoid.tolist() == ["c", "d"]
    assert eligible.event_time.tolist() == [0, 1]
    assert not eligible.state_allowed.any()
    coverage = post_coverage(events, eligible, eligible, all_year)
    assert len(coverage) == 8
    opening = coverage.loc[coverage.treatment_order.eq(2) & coverage.event_time.eq(0)].iloc[0]
    assert opening.post_streetview_status == "opening_month_temporal_relation_unknown"
    assert opening.opening_month_ambiguous_candidate_count == 1
    assert not opening.post_streetview_available
    future = coverage.loc[coverage.target_year.gt(2025)]
    assert future.post_streetview_status.eq("incomplete_calendar_year").all()
    assert not coverage.state_allowed.any()


def test_shared_boundary_and_corrupt_image(tmp_path: Path):
    grids = [{"geometry_wkt": box(0, 0, 1, 1).wkt}, {"geometry_wkt": box(1, 0, 2, 1).wkt}]
    assert assign_unique(np.array([.5, 1., 1.5]), np.array([.5, .5, .5]), grids).tolist() == [0, -1, 1]
    good, bad = tmp_path / "good.jpg", tmp_path / "bad.jpg"
    Image.new("RGB", (256, 128)).save(good)
    bad.write_bytes(b"not an image")
    assert verify_image(str(good))["image_valid"]
    assert not verify_image(str(bad))["image_valid"]


def test_server_copy_checks_bytes_and_preserves_collector_original(tmp_path: Path):
    raw = tmp_path / "collector"
    source = raw / "images/city/pano.jpg"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (256, 128)).save(source)
    original = source.read_bytes()
    record = {"city_key": "city", "panoid": "pano", "asset_path": str(source),
              "asset_bytes": len(original), "asset_sha256": sha256(source), "width": 256, "height": 128}
    root = tmp_path / "MIT_summer/data/active/streetview_grid_v2/images"
    receipt = copy_verified_image(record, root, raw)
    assert Path(receipt["asset_path"]).read_bytes() == original
    assert source.read_bytes() == original
    assert receipt == copy_verified_image(record, root, raw)
    Path(receipt["asset_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing destination differs"):
        copy_verified_image(record, root, raw)


def test_server_copy_rejects_changed_source_and_path_escape(tmp_path: Path):
    raw = tmp_path / "collector"
    source = raw / "images/city/pano.jpg"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (256, 128)).save(source)
    record = {"city_key": "city", "panoid": "pano", "asset_path": str(source),
              "asset_bytes": source.stat().st_size, "asset_sha256": "0" * 64,
              "width": 256, "height": 128}
    root = tmp_path / "MIT_summer/images"
    with pytest.raises(ValueError, match="changed during copying"):
        copy_verified_image(record, root, raw)
    assert not (root / "city/pano.jpg").exists()
    record["panoid"] = "../escape"
    with pytest.raises(ValueError, match="Unsafe image identity"):
        copy_verified_image(record, root, raw)


def test_publisher_rejects_changed_capture_or_future_date():
    contract = {"cities": {"c": {"grids": [{"grid_id": "g", "geometry_wkt": box(0, 0, 1, 1).wkt,
                                             "grid_geometry_sha256": "hash"}]}}}
    treatment = pd.DataFrame({"treatment_order": [1], "city_key": ["c"], "grid_id": ["g"],
                              "station_event_id": ["e"], "opening_month": ["2018-04"]})
    index = pd.DataFrame({"city_key": ["c"], "grid_id": ["g"], "panoid": ["p"],
                          "asset_path": ["image.jpg"],
                          "capture_month": ["2018-03"], "capture_date_precision": ["month"],
                          "wgs_lon": [.5], "wgs_lat": [.5], "grid_geometry_sha256": ["hash"]})
    selected = treatment.merge(index, on=["city_key", "grid_id"])
    selected["image_valid"], selected["asset_sha256"] = True, "a" * 64
    assert all(assert_binding(index, selected, treatment, contract).values())
    round_trip = selected.copy()
    round_trip["wgs_lon"] = np.nextafter(.5, .6)
    assert all(assert_binding(index, round_trip, treatment, contract).values())
    moved = selected.copy()
    moved["wgs_lon"] = .5 + 1e-7
    with pytest.raises(ValueError, match="selected_metadata_equal"):
        assert_binding(index, moved, treatment, contract)
    changed = selected.copy()
    changed["wgs_lon"] = .6
    with pytest.raises(ValueError, match="selected_metadata_equal"):
        assert_binding(index, changed, treatment, contract)
    future = selected.copy()
    future["capture_month"] = "2018-04"
    with pytest.raises(ValueError, match="strictly_before"):
        assert_binding(index, future, treatment, contract)


def test_selected_only_verification_falls_back_to_older_real_images(tmp_path: Path):
    events = pd.DataFrame({"treatment_order": [1], "city_key": ["c"], "grid_id": ["g"],
                           "station_event_id": ["e"], "opening_month": ["2018-06"]})
    rows = []
    for pano, month, exists in [("P0AA", "2018-05", False), ("P1AB", "2018-04", True), ("P2AC", "2018-03", True)]:
        path = tmp_path / "images/P/C/Saved" / pano[-2] / pano[-1] / f"{pano}.jpg"
        if exists:
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (256, 128)).save(path)
        rows.append({"city_key": "c", "grid_id": "g", "panoid": pano,
                     "capture_month": month, "asset_path": str(path), "asset_bytes": 0,
                     "source_metadata": str(tmp_path / "metadata/Saved.csv")})
    result, quality = select_verified_pre(events, pd.DataFrame(rows),
        {"cities": {"c": {"province": "P", "source_city": "C"}}}, tmp_path, 2, 2)
    assert result.panoid.tolist() == ["P1AB", "P2AC"]
    assert result.asset_bytes.gt(0).all()
    assert result.image_valid.all()
    assert not quality.image_valid.all()
