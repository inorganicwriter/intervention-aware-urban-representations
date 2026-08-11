"""Process one xianyu file at a time - optimized for large Excel files."""

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
MATCH_RADIUS_M = 1500
name_to_key = {CITIES[ck]["name"].replace("市", ""): ck for ck in CITIES}


def process_file(xlsx_path):
    fname = xlsx_path.stem
    print(f"\n{fname} ({xlsx_path.stat().st_size / 1024 / 1024:.0f}MB)")

    # Read with optimized settings
    df = pd.read_excel(xlsx_path, dtype={})
    cols = list(df.columns)
    print(f"  Rows: {len(df):,}, Cols: {len(cols)}")

    # Find key columns
    col_info = {}
    for _i, c in enumerate(cols):
        cstr = str(c)
        if "经" in cstr:
            col_info["lon"] = c
        if "纬" in cstr:
            col_info["lat"] = c
        if "成交价" in cstr or "单价" in cstr:
            col_info["price"] = c
        if "成交年份" in cstr:
            col_info["year"] = c
        if "小区" in cstr:
            col_info["comm"] = c
        if "城市" in cstr:
            col_info["city"] = c
        if "成交时间" in cstr:
            col_info["deal_time"] = c
        if "区域" in cstr:
            col_info["district"] = c

    print(f"  Found columns: {col_info}")

    if "lon" not in col_info or "lat" not in col_info:
        print("  SKIP: no lon/lat")
        return 0

    # Clean
    df = df.dropna(subset=[col_info["lon"], col_info["lat"], col_info["price"]]).copy()
    df["_lon"] = pd.to_numeric(df[col_info["lon"]], errors="coerce")
    df["_lat"] = pd.to_numeric(df[col_info["lat"]], errors="coerce")
    df["_price"] = pd.to_numeric(df[col_info["price"]], errors="coerce")
    df = df.dropna(subset=["_lon", "_lat", "_price"])
    df["_lon"] = df["_lon"].astype(float)
    df["_lat"] = df["_lat"].astype(float)
    df["_price"] = df["_price"].astype(float)

    # Extract year
    if "year" in col_info:
        df["_year"] = pd.to_numeric(df[col_info["year"]], errors="coerce").fillna(0).astype(int)
    elif "deal_time" in col_info:
        df["_year"] = df[col_info["deal_time"]].astype(str).str.extract(r"(\d{4})")[0]
        df["_year"] = pd.to_numeric(df["_year"], errors="coerce").fillna(0).astype(int)
    else:
        df["_year"] = 0

    # Extract city
    if "city" in col_info:
        df["_city"] = df[col_info["city"]].astype(str).str.replace("市", "").str.strip()
    else:
        df["_city"] = "北京"

    print(
        f"  After cleaning: {len(df):,} rows, price range={df['_price'].min():.0f}-{df['_price'].max():.0f}"
    )

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

        lat_c = float(grid_lats.mean())
        cos_lat = max(0.1, math.cos(math.radians(lat_c)))
        tree = cKDTree(np.column_stack([grid_lons * cos_lat, grid_lats]))
        query = np.column_stack([city_df["_lon"].values * cos_lat, city_df["_lat"].values])
        dist_deg, idx_arr = tree.query(query, k=1)
        dist_m = dist_deg * 111000.0

        city_df["grid_id"] = grid_ids[idx_arr]
        city_df["dist_m"] = dist_m
        city_df = city_df[city_df["dist_m"] <= MATCH_RADIUS_M]

        n_matched = len(city_df)
        if n_matched == 0:
            continue

        # Aggregate to grid x year
        gy = (
            city_df.groupby(["grid_id", "_year"])
            .agg(
                unit_price_mean=("_price", "mean"),
                unit_price_median=("_price", "median"),
                n_transactions=("_price", "count"),
            )
            .reset_index()
        )
        gy.rename(columns={"_year": "year"}, inplace=True)
        gy["year"] = gy["year"].astype(int)

        out_dir = LABEL_DIR / ck
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ck}_lianjia_grid_yearly.parquet"

        # Merge with existing if any
        if out_path.exists():
            old = pd.read_parquet(out_path)
            gy = pd.concat([old, gy], ignore_index=True).drop_duplicates(subset=["grid_id", "year"])

        gy.to_parquet(out_path, index=False)
        total_gy += len(gy)
        print(f"  {ck}: {n_matched:,} matched -> {len(gy):,} grid-years")

    return total_gy


# Process smallest files first
import sys

if len(sys.argv) > 1:
    target = Path(sys.argv[1])
    process_file(target)
else:
    files = sorted(XIANYU.rglob("*.xlsx"), key=lambda f: f.stat().st_size)
    total = 0
    for f in files:
        try:
            total += process_file(f) or 0
        except Exception as e:
            print(f"  ERROR: {e}")
    print(f"\nTOTAL: {total:,} grid-year rows")
