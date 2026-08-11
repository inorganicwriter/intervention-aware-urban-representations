"""Spatial join: match GEE sample points to 500m grid centroids.

Strategy: project both grid centroids and GEE points to the city's UTM
zone, then use scipy KD-Tree for O(N log N) nearest-neighbour search.
For a 500m grid the offset is always < 250m (half a cell), so the 1-NN
is the correct grid assignment.

Output:  data/active/curated/{source}/{city}_{source}.parquet
         One parquet per city per source with grid_id merged in,
         lat/lon columns dropped.

Usage:
    python scripts/collection/join_gee_to_grids.py --source all --city all
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from urban_intervention.config.project import ACTIVE_CITIES, CITIES, GRID_DIR
from urban_intervention.data.paths import CURATED_DIR, STAGING_DIR

GEE_STAGING_DIR = STAGING_DIR / "gee"
MAX_MATCH_DIST_M = 500  # reject points farther than this from any grid


def _get_utm_proj(city_key: str):
    """Return the pyproj Transformer to UTM for a city."""
    from pyproj import Transformer

    epsg = int(CITIES[city_key]["projected_crs"].split(":")[-1])
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def _load_grid_utm(city_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (grid_coords_utm, grid_lons, grid_lats, grid_ids).

    grid_coords_utm: (N, 2) array of (easting, northing) in UTM metres.
    grid_ids: (N,) array of grid_id strings.
    """
    p = GRID_DIR / city_key / f"{city_key}_grids.parquet"
    df = pd.read_parquet(p)[["grid_id", "centroid_lon", "centroid_lat"]]
    grid_ids = df["grid_id"].values.astype(object)
    lons = df["centroid_lon"].values.astype(float)
    lats = df["centroid_lat"].values.astype(float)
    transformer = _get_utm_proj(city_key)
    x, y = transformer.transform(lons, lats)
    return np.column_stack([x, y]), lons, lats, grid_ids


def _load_csv_utm(csv_path: Path, transformer) -> tuple[np.ndarray, pd.Index, int]:
    """Return (utm_coords, df_index, year) for a single GEE CSV."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    year = int(csv_path.stem.rsplit("_", 1)[-1])
    valid = df["latitude"].notna() & df["longitude"].notna()
    if not valid.any():
        return None, None, year
    lons = df.loc[valid, "longitude"].values.astype(float)
    lats = df.loc[valid, "latitude"].values.astype(float)
    x, y = transformer.transform(lons, lats)
    return np.column_stack([x, y]), valid[valid].index, year


def join_city(city_key: str, source: str) -> None:
    """Spatial join one source for one city."""
    csv_dir = GEE_STAGING_DIR / source
    pattern = f"{source}_{city_key}_*.csv"
    csv_files = sorted(csv_dir.glob(pattern))
    if not csv_files:
        print(f"  [SKIP] {city_key}: no {source} files")
        return

    print(f"  {city_key}: {len(csv_files)} CSVs ...")
    t0 = time.time()

    # Load grids in UTM and build KD-Tree
    grid_utm, grid_lons, grid_lats, grid_ids = _load_grid_utm(city_key)
    tree = cKDTree(grid_utm)

    transformer = _get_utm_proj(city_key)
    all_parts = []
    n_matched = 0
    n_total = 0

    for fp in csv_files:
        pt_utm, valid_idx, year = _load_csv_utm(fp, transformer)
        if pt_utm is None:
            continue

        n_total += len(valid_idx)

        # KD-Tree query: for each GEE point, find nearest grid ≤ MAX_MATCH_DIST_M
        dists_m, nn_idx = tree.query(pt_utm, k=1, distance_upper_bound=MAX_MATCH_DIST_M)

        df = pd.read_csv(fp, encoding="utf-8")
        df_valid = df.loc[valid_idx].copy()
        # Fill matched grid_ids; unmatched stay None
        matched = nn_idx < len(grid_ids)
        result_ids = np.full(len(pt_utm), None, dtype=object)
        result_ids[matched] = grid_ids[nn_idx[matched]]
        df_valid["grid_id"] = result_ids
        df_valid = df_valid.drop(columns=["latitude", "longitude"], errors="ignore")
        # Keep only rows that matched
        df_valid = df_valid[df_valid["grid_id"].notna()]
        n_matched += len(df_valid)
        all_parts.append(df_valid)

    if not all_parts:
        return

    combined = pd.concat(all_parts, ignore_index=True)
    output_dirs = {
        "viirs": CURATED_DIR / "viirs",
        "s2": CURATED_DIR / "sentinel2",
        "pop": CURATED_DIR / "population",
    }
    output_dirs[source].mkdir(parents=True, exist_ok=True)
    out = output_dirs[source] / f"{city_key}_{source}.parquet"
    combined.to_parquet(out, index=False)

    n_grids = combined["grid_id"].nunique()
    yrs = sorted(combined["year"].unique())
    print(
        f"    -> {out.name}  "
        f"({n_matched}/{n_total} matched, {n_grids} grids, "
        f"{yrs[0]}–{yrs[-1]}, {time.time() - t0:.0f}s)"
    )


def main():
    parser = argparse.ArgumentParser(description="Spatial join GEE CSVs to 500m grids")
    parser.add_argument("--source", default="all", choices=["all", "viirs", "s2", "pop"])
    parser.add_argument("--city", default="all")
    args = parser.parse_args()

    sources = ["viirs", "s2", "pop"] if args.source == "all" else [args.source]
    cities = ACTIVE_CITIES if args.city == "all" else [args.city]

    for source in sources:
        csv_dir = GEE_STAGING_DIR / source
        if not csv_dir.exists():
            continue
        print(f"\n{'=' * 50}\n{source.upper()}\n{'=' * 50}")
        for ck in cities:
            join_city(ck, source)


if __name__ == "__main__":
    main()
