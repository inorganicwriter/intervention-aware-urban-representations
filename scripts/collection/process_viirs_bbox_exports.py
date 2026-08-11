"""Download, nearest-grid match, aggregate and validate VIIRS bbox exports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from google.cloud import storage
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from urban_intervention.config.project import CITIES, GRID_DIR  # noqa: E402
from urban_intervention.data.paths import STAGING_DIR, VIIRS_DIR  # noqa: E402

STAGING = STAGING_DIR / "gee" / "viirs"
CURATED = VIIRS_DIR
PROJECT = "macro-city-engine"
BUCKET = "macro-city-engine-mit-summer-gee-staging"
MAX_MATCH_DISTANCE_M = 500.0


def _download(city: str, label: str) -> list[Path]:
    ee.Initialize(project=PROJECT)
    client = storage.Client(project=PROJECT, credentials=ee.data.get_persistent_credentials())
    prefix = f"MIT_Summer_VIIRS/{city}/viirs_{city}_{label}"
    blobs = list(client.list_blobs(BUCKET, prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No export found: gs://{BUCKET}/{prefix}")
    out_dir = STAGING / city / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for blob in blobs:
        path = out_dir / Path(blob.name).name
        blob.download_to_filename(path)
        paths.append(path)
    return paths


def _grid(city: str) -> tuple[pd.DataFrame, np.ndarray, Transformer]:
    path = GRID_DIR / city / f"{city}_grids.parquet"
    grid = pd.read_parquet(path, columns=["grid_id", "centroid_lon", "centroid_lat"])
    if grid.grid_id.isna().any() or grid.grid_id.duplicated().any():
        raise ValueError(f"{city}: invalid reference grid keys")
    epsg = int(CITIES[city]["projected_crs"].split(":")[-1])
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    gx, gy = transformer.transform(grid.centroid_lon.to_numpy(), grid.centroid_lat.to_numpy())
    return grid, np.column_stack([gx, gy]), transformer


def process(city: str, period: str) -> Path:
    label = period.replace("-", "_")
    files = _download(city, label)
    parts = []
    for path in files:
        frame = pd.read_csv(path)
        required = {"period", "radiance", "valid_days", "latitude", "longitude"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        frame["source_file"] = path.name
        parts.append(frame)
    points = pd.concat(parts, ignore_index=True)
    if set(points.period.astype(str)) != {period}:
        raise ValueError(f"Unexpected periods: {sorted(points.period.astype(str).unique())}")

    grid, grid_xy, transformer = _grid(city)
    px, py = transformer.transform(points.longitude.to_numpy(), points.latitude.to_numpy())
    distances, indices = cKDTree(grid_xy).query(
        np.column_stack([px, py]), k=1, distance_upper_bound=MAX_MATCH_DISTANCE_M
    )
    matched = indices < len(grid)
    points = points.loc[matched].copy()
    points["grid_id"] = grid.grid_id.to_numpy()[indices[matched]]
    points["match_distance_m"] = distances[matched]

    matched_dir = STAGING / city / "matched"
    matched_dir.mkdir(parents=True, exist_ok=True)
    points.to_parquet(matched_dir / f"viirs_{city}_{label}_matched_points.parquet", index=False)

    grouped = points.groupby("grid_id", as_index=False).agg(
        radiance_mean=("radiance", "mean"),
        valid_days_mean=("valid_days", "mean"),
        source_point_count=("radiance", "size"),
        match_distance_m_mean=("match_distance_m", "mean"),
        match_distance_m_max=("match_distance_m", "max"),
        source_files=("source_file", lambda s: ";".join(sorted(set(s)))),
    )
    result = grid[["grid_id"]].merge(grouped, on="grid_id", how="left", validate="one_to_one")
    result.insert(0, "city_key", city)
    result.insert(2, "period", period)
    result["source_point_count"] = result["source_point_count"].fillna(0).astype("int32")
    result["source_product"] = "NASA/VIIRS/002/VNP46A2"
    result["aggregation"] = "nearest_grid_then_mean"

    keys = ["city_key", "grid_id", "period"]
    if result[keys].isna().any(axis=1).any() or result.duplicated(keys).any():
        raise AssertionError("VIIRS primary-key contract failed")
    if len(result) != len(grid):
        raise AssertionError("VIIRS did not preserve the reference-grid universe")

    CURATED.mkdir(parents=True, exist_ok=True)
    out = CURATED / f"{city}_viirs_{label}.parquet"
    result.to_parquet(out, index=False)
    print(
        f"saved={out} rows={len(result)} matched_grids={(result.source_point_count > 0).sum()} "
        f"duplicate_keys=0 source_points={len(points)}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--period", required=True, help="YYYY or YYYY-MM")
    args = parser.parse_args()
    process(args.city, args.period)


if __name__ == "__main__":
    main()
