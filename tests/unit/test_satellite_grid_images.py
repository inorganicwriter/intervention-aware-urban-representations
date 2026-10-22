from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform

from scripts.collection import download_satellite_grid_images as images


@pytest.mark.parametrize("year,sensor", [(2010, "landsat5"), (2012, "landsat7"),
                                        (2013, "landsat8"), (2018, "sentinel2")])
def test_sensor_routes_respect_product_start_year(year: int, sensor: str) -> None:
    assert images.sensor_candidates(year)[0] == sensor


@pytest.mark.parametrize("sensor,size", [("sentinel2", 50), ("landsat8", 17)])
def test_export_affine_matches_frozen_500m_extent(sensor: str, size: int) -> None:
    polygon = transform(Transformer.from_crs("EPSG:32650", "EPSG:4326", always_xy=True).transform,
                        box(400000, 3400000, 400500, 3400500))
    spec = images.raster_spec(polygon.wkt, "EPSG:32650", sensor)
    affine = images.Affine(*spec["crs_transform"])
    assert spec["width"] == size
    assert affine * (0, 0) == pytest.approx((400000, 3400500), abs=0.001)
    assert affine * (size, size) == pytest.approx((400500, 3400000), abs=0.001)


def test_invalid_and_nonfinite_pixels_never_enter_reflectance() -> None:
    names = images.BANDS + ["valid_mask", "observation_count"]
    array = np.ones((2, 2), dtype=[(name, "f4") for name in names])
    array["valid_mask"][0, 0] = 0
    array["blue"][0, 1] = np.nan
    array["observation_count"][1, 0] = 0
    payload = io.BytesIO()
    np.save(payload, array)
    reflectance, valid, count = images.decode_pixels(payload.getvalue(), 2)
    assert valid.sum() == 1
    assert np.isnan(reflectance[:, ~valid]).all()
    assert (count[~valid] == 0).all()
    with pytest.raises(ValueError, match="shape or bands"):
        images.decode_pixels(payload.getvalue(), 3)


def test_resume_rejects_different_grid_and_corrupt_image(tmp_path: Path) -> None:
    image_path = tmp_path / "image.tif"
    image_path.write_bytes(b"real downloaded bytes")
    for name in ["rgb.png", "mask.png", "source.json"]:
        (tmp_path / name).write_bytes(b"preview")
    row = {"city_key": "city", "grid_id": "grid", "treatment_order": 1,
           "target_year": 2020, "grid_geometry_sha256": "a"}
    rec = {**row, "contract_sha256": "contract", "image_path": "image.tif",
           "image_sha256": images._checksum(image_path), "preview_path": "rgb.png",
           "mask_path": "mask.png", "source_manifest_path": "source.json"}
    record_path = tmp_path / "record.json"
    images._write_json(record_path, rec)
    assert images.resume_record(record_path, row, "contract", tmp_path) is not None
    assert images.resume_record(record_path, {**row, "grid_id": "other"}, "contract", tmp_path) is None
    images._write_json(record_path, {**rec, "satellite_image_available": False,
                                   "sensor_attempts": [{"reason": "download_failed"}]})
    assert images.resume_record(record_path, row, "contract", tmp_path) is None
    images._write_json(record_path, rec)
    image_path.write_bytes(b"corrupt")
    assert images.resume_record(record_path, row, "contract", tmp_path) is None


def test_dry_run_never_creates_or_claims_available_images(tmp_path: Path) -> None:
    rows = pd.DataFrame({"treatment_order": [1], "city_key": ["city"], "grid_id": ["grid"],
                         "opening_month": ["2020-01"], "target_year": [2019],
                         "geometry_wkt": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"],
                         "grid_geometry_sha256": ["hash"]})
    result = images.collect_group(rows, tmp_path, "contract", True)
    assert result[0]["is_dry_run"]
    assert not result[0]["satellite_image_available"]
    assert result[0]["availability_reason"] == "dry_run"
    assert not list(tmp_path.rglob("*.tif"))
