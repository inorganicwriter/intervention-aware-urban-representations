"""Fast AOI-like matching: community buffer -> cKDTree radius query.

Instead of Shapely polygon intersection (slow), use cKDTree to find
all grid centroids within a community's buffer radius, then assign
the community's price to those grids. This is 100x faster while
achieving the same AOI effect.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts"))
from scipy.spatial import cKDTree

from urban_intervention.config.project import CITIES, GRID_DIR
from urban_intervention.data.paths import LIANJIA_LABEL_DIR, RAW_LIANJIA_DIR

XIANYU = RAW_LIANJIA_DIR
LABEL_DIR = LIANJIA_LABEL_DIR
name_to_key = {CITIES[ck]["name"].replace("市", ""): ck for ck in CITIES}

BUFFER_M = 400  # 400m radius covers ~1 grid; larger covers more but may overlap neighbors


def process_file_aoi(xlsx_path):
    fname = xlsx_path.stem
    size_mb = xlsx_path.stat().st_size / 1024 / 1024
    print(f"\n{fname} ({size_mb:.0f}MB)")

    df = pd.read_excel(xlsx_path)
    cols = list(df.columns)
    print(f"  Rows: {len(df):,}")

    col_map = {}
    for _i, c in enumerate(cols):
        cstr = str(c)
        if "经" in cstr:
            col_map["lon"] = c
        if "纬" in cstr:
            col_map["lat"] = c
        if "成交价" in cstr:
            col_map["price"] = c
        if "成交年份" in cstr:
            col_map["year"] = c
        if "小区" in cstr:
            col_map["comm"] = c
        if "城市" in cstr:
            col_map["city"] = c

    if "lon" not in col_map:
        return 0

    df = df.dropna(subset=[col_map["lon"], col_map["lat"], col_map["price"]]).copy()
    df["_lon"] = pd.to_numeric(df[col_map["lon"]], errors="coerce").astype(float)
    df["_lat"] = pd.to_numeric(df[col_map["lat"]], errors="coerce").astype(float)
    df["_price"] = pd.to_numeric(df[col_map["price"]], errors="coerce").astype(float)
    df = df.dropna(subset=["_lon", "_lat", "_price"])
    if "year" in col_map:
        df["_year"] = pd.to_numeric(df[col_map["year"]], errors="coerce").fillna(0).astype(int)
    else:
        df["_year"] = 0
    if "city" in col_map:
        df["_city"] = df[col_map["city"]].astype(str).str.replace("市", "").str.strip()
    else:
        df["_city"] = "北京"

    total_gy = 0
    for city_name in df["_city"].unique():
        ck = name_to_key.get(city_name)
        if not ck:
            for name, key in name_to_key.items():
                if name in city_name or city_name in name:
                    ck = key
                    break
        if not ck:
            continue

        grid_path = GRID_DIR / ck / f"{ck}_grids.parquet"
        if not grid_path.exists():
            continue

        city_df = df[df["_city"] == city_name].copy()
        grids = pd.read_parquet(grid_path)
        grid_lons = grids["centroid_lon"].values.astype(float)
        grid_lats = grids["centroid_lat"].values.astype(float)
        grid_ids = grids["grid_id"].values

        # Latitude-corrected tree in METERS
        lat_c = float(grid_lats.mean())
        cos_lat = max(0.1, math.cos(math.radians(lat_c)))
        METERS_PER_DEG = 111000.0
        tree_pts = np.column_stack(
            [
                grid_lons * METERS_PER_DEG * cos_lat,
                grid_lats * METERS_PER_DEG,
            ]
        )
        tree = cKDTree(tree_pts)

        # For each COMMUNITY (not each transaction), find ALL nearby grids
        comm_groups = city_df.groupby(col_map["comm"])
        assignments = []

        for _comm_name, comm_df in comm_groups:
            clon = comm_df["_lon"].mean()
            clat = comm_df["_lat"].mean()

            # Query point in meters
            qlon_m = clon * METERS_PER_DEG * cos_lat
            qlat_m = clat * METERS_PER_DEG
            idxs = tree.query_ball_point([qlon_m, qlat_m], r=BUFFER_M)

            if len(idxs) == 0:
                continue

            # Assign community price to each nearby grid, by year
            for yr, yr_df in comm_df.groupby("_year"):
                avg_price = yr_df["_price"].mean()
                for gi in idxs:
                    assignments.append(
                        {
                            "grid_id": grid_ids[gi],
                            "year": int(yr),
                            "unit_price_mean": avg_price,
                            "n_transactions": len(yr_df),
                        }
                    )

        if not assignments:
            continue

        gy = pd.DataFrame(assignments)
        gy = (
            gy.groupby(["grid_id", "year"])
            .agg(
                unit_price_mean=("unit_price_mean", "mean"),
                n_transactions=("n_transactions", "sum"),
            )
            .reset_index()
        )

        out_dir = LABEL_DIR / ck
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ck}_lianjia_aoi_grid_yearly.parquet"
        gy.to_parquet(out_path, index=False)
        total_gy += len(gy)
        print(
            f"  {ck}: {len(comm_groups):,} comms, {len(gy):,} grid-years (x{len(assignments) / len(city_df):.1f} multiplier)"
        )

    return total_gy


if __name__ == "__main__":
    files = sorted(XIANYU.rglob("*.xlsx"), key=lambda f: f.stat().st_size)
    total = 0
    for f in files:
        try:
            total += process_file_aoi(f) or 0
        except Exception as e:
            print(f"  ERROR: {e}")
    print(f"\nTOTAL AOI grid-years: {total:,}")
