"""Compute per-grid locational distance features from the centre registry.

For every grid in every city this builds:
  - ``dist_main_km``: distance to the city's main centre (UTM straight line)
  - ``dist_nearest_subcentre_km``: distance to the nearest identified
    subcentre (or the main centre if the city has no subcentres)
  - ``dist_nearest_centre_km``: distance to the nearest of all centres

Centres come from ``data/active/reference/city_centers.csv`` (McMillen 2001 LWR
significant peaks on a composite POI + VIIRS + population density surface).
Distances use the city's projected UTM CRS (the same projection as the 500m
grids), so they are planar distances in kilometres.

Output: ``data/active/curated/location_features/{city}_location.parquet`` with one
row per grid_id.

Usage:
    python scripts/analysis/build_location_features.py --city all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "scripts"))
from urban_intervention.config.project import ACTIVE_CITIES, CITIES, GRID_DIR  # noqa: E402

REFERENCE_DIR = BASE_DIR / "data" / "active" / "reference"
OUT_DIR = BASE_DIR / "data" / "active" / "curated" / "location_features"


def build_city(city: str, centres: pd.DataFrame) -> dict:
    grid_path = GRID_DIR / city / f"{city}_grids.parquet"
    grids = pd.read_parquet(
        grid_path,
        columns=[
            "grid_id",
            "centroid_lon",
            "centroid_lat",
        ],
    )
    city_centres = centres[centres["city_key"] == city].copy()
    if city_centres.empty:
        raise ValueError(f"{city}: no centres in registry")

    epsg = int(CITIES[city]["projected_crs"].split(":")[-1])
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    gx, gy = transformer.transform(
        grids["centroid_lon"].to_numpy(),
        grids["centroid_lat"].to_numpy(),
    )
    c_x, c_y = transformer.transform(
        city_centres["centroid_lon"].to_numpy(),
        city_centres["centroid_lat"].to_numpy(),
    )
    centre_coords = np.column_stack([c_x, c_y]) / 1000.0  # km

    main_row = city_centres[city_centres["role"] == "main"]
    if main_row.empty:
        raise ValueError(f"{city}: no main centre in registry")
    main_xy = (
        np.array(
            [
                transformer.transform(
                    float(main_row["centroid_lon"].iloc[0]),
                    float(main_row["centroid_lat"].iloc[0]),
                )
            ]
        )
        / 1000.0
    )

    grid_xy = np.column_stack([gx, gy]) / 1000.0
    delta = grid_xy[:, None, :] - centre_coords[None, :, :]
    all_dist = np.sqrt((delta**2).sum(axis=2))
    delta_main = grid_xy - main_xy
    dist_main = np.sqrt((delta_main**2).sum(axis=1))

    sub_mask = (city_centres["role"] != "main").to_numpy()
    if sub_mask.any():
        dist_sub = all_dist[:, sub_mask]
        dist_nearest_sub = dist_sub.min(axis=1)
    else:
        dist_nearest_sub = dist_main

    out = pd.DataFrame(
        {
            "city_key": city,
            "grid_id": grids["grid_id"],
            "dist_main_km": dist_main,
            "dist_nearest_subcentre_km": dist_nearest_sub,
            "dist_nearest_centre_km": all_dist.min(axis=1),
        }
    )
    return {
        "city": city,
        "grids": len(out),
        "centres": len(city_centres),
        "main_km_median": float(np.median(dist_main)),
        "sub_km_median": float(np.median(dist_nearest_sub)),
        "out_path": OUT_DIR / f"{city}_location.parquet",
        "frame": out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="all")
    args = parser.parse_args()

    centres = pd.read_csv(REFERENCE_DIR / "city_centers.csv")
    cities = ACTIVE_CITIES if args.city == "all" else [args.city]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for city in cities:
        result = build_city(city, centres)
        result["frame"].to_parquet(result["out_path"], index=False)
        print(
            f"  {city}: {result['grids']:,} grids, {result['centres']} centres, "
            f"median dist_main {result['main_km_median']:.2f} km, "
            f"median dist_sub {result['sub_km_median']:.2f} km"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
