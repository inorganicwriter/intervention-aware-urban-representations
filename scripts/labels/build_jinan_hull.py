"""Build AOI for Jinan using convex hull of Lianjia transaction points.
Each community's transaction coordinates form a natural AOI."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from shapely import STRtree
from shapely.geometry import Polygon, box

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts"))
from urban_intervention.config.project import GRID_DIR
from urban_intervention.data.paths import LIANJIA_LABEL_DIR, RAW_LIANJIA_DIR

XIANYU = RAW_LIANJIA_DIR
LABEL_DIR = LIANJIA_LABEL_DIR

# Load Jinan data from Shandong province file
sd_file = XIANYU / "山东省.xlsx"
df = pd.read_excel(sd_file)
cols = list(df.columns)

# Find columns
col_map = {}
for _i, c in enumerate(cols):
    cs = str(c)
    if "经" in cs:
        col_map["lon"] = c
    if "纬" in cs:
        col_map["lat"] = c
    if "成交价" in cs:
        col_map["price"] = c
    if "成交年份" in cs:
        col_map["year"] = c
    if "小区" in cs:
        col_map["comm"] = c
    if "城市" in cs:
        col_map["city"] = c

print(f"Shandong file: {len(df)} rows")
print(f"Columns found: {col_map}")

# Filter to Jinan only
df["_city"] = df[col_map["city"]].astype(str).str.replace("市", "").str.strip()
jn = df[df["_city"] == "济南"].copy()
print(f"Jinan rows: {len(jn)}")

# Clean
jn["_lon"] = pd.to_numeric(jn[col_map["lon"]], errors="coerce").astype(float)
jn["_lat"] = pd.to_numeric(jn[col_map["lat"]], errors="coerce").astype(float)
jn["_price"] = pd.to_numeric(jn[col_map["price"]], errors="coerce").astype(float)
jn = jn.dropna(subset=["_lon", "_lat", "_price"])
jn["_year"] = pd.to_numeric(jn[col_map["year"]], errors="coerce").fillna(0).astype(int)
print(f"Jinan valid: {len(jn)}")

# Load grids
grids = pd.read_parquet(GRID_DIR / "jinan" / "jinan_grids.parquet")
grid_ids = grids["grid_id"].values
grid_lons = grids["centroid_lon"].values.astype(float)
grid_lats = grids["centroid_lat"].values.astype(float)
print(f"Jinan grids: {len(grids)}")

lat_c = float(grid_lats.mean())
cos_lat = max(0.1, math.cos(math.radians(lat_c)))
mdlon = 111320.0 * cos_lat
mdlat = 111320.0

# Build grid STRtree
grid_polys = []
for i in range(len(grid_ids)):
    dlon = 250 / mdlon
    dlat = 250 / mdlat
    grid_polys.append(
        box(grid_lons[i] - dlon, grid_lats[i] - dlat, grid_lons[i] + dlon, grid_lats[i] + dlat)
    )
tree = STRtree(grid_polys)

# Group by community, compute convex hull
comm_groups = jn.groupby(col_map["comm"])
assignments = []
n_hull = 0
n_buffer = 0
n_skip = 0

for _comm_name, comm_df in comm_groups:
    pts = np.unique(
        np.column_stack([comm_df["_lon"].values, comm_df["_lat"].values]).round(6), axis=0
    )

    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
            aoi = Polygon(pts[hull.vertices])
            n_hull += 1
        except Exception:
            # Degenerate hull, use buffer
            clon = comm_df["_lon"].mean()
            clat = comm_df["_lat"].mean()
            dlon = 250 / mdlon
            dlat = 250 / mdlat
            aoi = box(clon - dlon, clat - dlat, clon + dlon, clat + dlat)
            n_buffer += 1
    elif len(pts) == 2:
        # Two points: use bounding box
        aoi = box(min(pts[:, 0]), min(pts[:, 1]), max(pts[:, 0]), max(pts[:, 1]))
        n_hull += 1
    else:
        # Single point: use buffer
        clon = comm_df["_lon"].mean()
        clat = comm_df["_lat"].mean()
        dlon = 250 / mdlon
        dlat = 250 / mdlat
        aoi = box(clon - dlon, clat - dlat, clon + dlon, clat + dlat)
        n_buffer += 1

    # Find intersecting grids
    hits = tree.query(aoi, predicate="intersects")
    if len(hits) == 0:
        n_skip += 1
        continue

    covered = grid_ids[hits]
    for yr, yr_df in comm_df.groupby("_year"):
        avg_price = float(yr_df["_price"].mean())
        for gid in covered:
            assignments.append(
                {
                    "grid_id": gid,
                    "year": int(yr),
                    "unit_price_mean": avg_price,
                    "n_transactions": len(yr_df),
                }
            )

print("\nJinan convex hull AOI:")
print(f"  Hull: {n_hull}, Buffer: {n_buffer}, Skip: {n_skip}")
print(f"  Total communities: {n_hull + n_buffer + n_skip}")

if assignments:
    gy = pd.DataFrame(assignments)
    gy = (
        gy.groupby(["grid_id", "year"])
        .agg(
            unit_price_mean=("unit_price_mean", "mean"),
            n_transactions=("n_transactions", "sum"),
        )
        .reset_index()
    )

    out_dir = LABEL_DIR / "jinan"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "jinan_lianjia_aoi_grid_yearly.parquet"

    # Merge with existing if any
    if out_path.exists():
        old = pd.read_parquet(out_path)
        gy = pd.concat([old, gy], ignore_index=True)
        gy = gy.drop_duplicates(subset=["grid_id", "year"])

    gy.to_parquet(out_path, index=False)
    print(f"  Saved: {len(gy)} grid-years, {gy.grid_id.nunique()} grids")
    print(f"  Years: {gy.year.min()}-{gy.year.max()}")
else:
    print("  No data to save")
