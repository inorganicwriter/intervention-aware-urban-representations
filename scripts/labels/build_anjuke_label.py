"""Convert Anjuke listing snapshots to grid-level price labels.

The Anjuke data in data/raw_housing/anjuke_cross/{city}_house.csv is a listing snapshot
(no transaction dates), but it has built-in coordinates and unit prices
for 620k+ listings across 43 of our 44 project cities. This script
converts it to a grid-level price baseline that can be used as:
  - a cross-sectional control variable in DiD models
  - a baseline price level for hedonic adjustments
  - a fallback when Lianjia transaction data is unavailable

Output: data/active/labels/{city}/{city}_anjuke_grid_price.parquet
  grid_id, unit_price_mean, unit_price_median, unit_price_p25, unit_price_p75,
  total_price_mean, n_houses, n_communities, community_names

Usage:
    python scripts/labels/build_anjuke_label.py
    python scripts/labels/build_anjuke_label.py --city beijing
"""

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree as _cKDTree
except ImportError:
    _cKDTree = None

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR / "scripts" / "collection"))
from amap_transit_fetcher import gcj02_to_wgs84

from urban_intervention.config.project import ACTIVE_CITIES, CITIES, GRID_DIR
from urban_intervention.data.paths import ANJUKE_LABEL_DIR, RAW_ANJUKE_DIR

HOUSING_DIR = RAW_ANJUKE_DIR
LABEL_DIR = ANJUKE_LABEL_DIR

MATCH_RADIUS_M = 1500


def _vec_haversine_m(
    grid_lons: np.ndarray, grid_lats: np.ndarray, lon: float, lat: float
) -> np.ndarray:
    R = 6371000.0
    glat_r = np.radians(grid_lats)
    qlat_r = np.radians(lat)
    dlat = glat_r - qlat_r
    dlon = np.radians(grid_lons - lon)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(glat_r) * np.cos(qlat_r) * np.sin(dlon / 2.0) ** 2
    return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def parse_unit_price(text: str) -> float | None:
    """Parse '22631元/平' or '22631' -> 22631.0."""
    if not text or pd.isna(text):
        return None
    m = re.search(r"([\d.]+)", str(text))
    return float(m.group(1)) if m else None


def parse_total_price(text: str) -> float | None:
    """Parse '235万' -> 235.0, '235.5万' -> 235.5."""
    if not text or pd.isna(text):
        return None
    m = re.search(r"([\d.]+)", str(text))
    return float(m.group(1)) if m else None


def parse_point(text: str) -> tuple[float, float] | None:
    """Parse 'POINT(109.159582 18.341284)' -> (lon, lat) in GCJ-02."""
    if not text or pd.isna(text):
        return None
    m = re.search(r"POINT\(\s*([\d.]+)\s+([\d.]+)\s*\)", str(text))
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def load_anjuke_city(city_key: str) -> pd.DataFrame | None:
    """Load the Anjuke house.csv for a city, return DataFrame with parsed fields."""
    cfg = CITIES[city_key]
    city_name = cfg["name"]
    # Anjuke filenames use the full city name (e.g. 北京市_house.csv).
    candidates = list(HOUSING_DIR.glob(f"{city_name}*_house.csv"))
    if not candidates:
        return None
    f = candidates[0]
    df = pd.read_csv(f, encoding="utf-8")
    # Column names are Chinese; alias them for safe access.
    col_map = {
        df.columns[0]: "community_id",
        df.columns[1]: "community_name",
        df.columns[2]: "coord",
        df.columns[3]: "community_price",
        df.columns[4]: "house_title",
        df.columns[5]: "house_desc",
        df.columns[6]: "total_price_raw",
        df.columns[7]: "unit_price_raw",
    }
    df = df.rename(columns=col_map)
    # Parse coordinates (GCJ-02) and convert to WGS-84.
    coords = df["coord"].apply(parse_point)
    df["gcj_lon"] = coords.apply(lambda c: c[0] if c else None)
    df["gcj_lat"] = coords.apply(lambda c: c[1] if c else None)
    valid = df["gcj_lon"].notna() & df["gcj_lat"].notna()
    wgs = df.loc[valid, ["gcj_lon", "gcj_lat"]].apply(
        lambda r: gcj02_to_wgs84(r["gcj_lon"], r["gcj_lat"]), axis=1
    )
    df.loc[valid, "lon"] = wgs.apply(lambda t: t[0])
    df.loc[valid, "lat"] = wgs.apply(lambda t: t[1])
    # Parse prices.
    df["unit_price"] = df["unit_price_raw"].apply(parse_unit_price)
    df["total_price"] = df["total_price_raw"].apply(parse_total_price)
    return df


def build_city_label(city_key: str) -> int:
    """Build grid-level price label for one city from Anjuke data."""
    cfg = CITIES[city_key]
    city_name = cfg["name"]

    df = load_anjuke_city(city_key)
    if df is None or df.empty:
        print(f"  [SKIP] {city_key}: no Anjuke data")
        return 0

    n_raw = len(df)
    df = df.dropna(subset=["lon", "lat", "unit_price"]).copy()
    n_valid = len(df)
    if n_valid == 0:
        print(f"  [SKIP] {city_key}: no valid rows after parsing")
        return 0

    grid_path = GRID_DIR / city_key / f"{city_key}_grids.parquet"
    if not grid_path.exists():
        print(f"  [SKIP] {city_key}: no grid parquet")
        return 0
    grids = pd.read_parquet(grid_path)
    grid_ids = grids["grid_id"].values.astype(object)
    grid_lons = grids["centroid_lon"].values.astype(float)
    grid_lats = grids["centroid_lat"].values.astype(float)

    # Fast nearest-grid assignment via cKDTree on (lon, lat).
    if _cKDTree is None:
        raise ImportError(
            "scipy is required for fast grid matching. Install with: pip install scipy"
        )
    lat_c = float(grid_lats.mean())
    cos_lat = max(0.1, math.cos(math.radians(lat_c)))
    tree_pts = np.column_stack([grid_lons * cos_lat, grid_lats])
    query_pts = np.column_stack(
        [
            df["lon"].values.astype(float) * cos_lat,
            df["lat"].values.astype(float),
        ]
    )
    tree = _cKDTree(tree_pts)
    dist_deg, idx_arr = tree.query(query_pts, k=1)
    # Convert degree-distance to meters (1 deg ≈ 111 km).
    dist_m = dist_deg * 111000.0

    assigned = pd.DataFrame(
        {
            "grid_id": grid_ids[idx_arr],
            "community_name": df["community_name"].values,
            "unit_price": df["unit_price"].values.astype(float),
            "total_price": df["total_price"].values,
            "dist_m": dist_m,
        }
    )
    assigned = assigned[assigned["dist_m"] <= MATCH_RADIUS_M]
    if assigned.empty:
        print(f"  [SKIP] {city_key}: no grid assignments within {MATCH_RADIUS_M}m")
        return 0

    # Aggregate to grid level.
    grid_yearly = (
        assigned.groupby("grid_id")
        .agg(
            unit_price_mean=("unit_price", "mean"),
            unit_price_median=("unit_price", "median"),
            unit_price_p25=("unit_price", lambda x: float(x.quantile(0.25))),
            unit_price_p75=("unit_price", lambda x: float(x.quantile(0.75))),
            total_price_mean=("total_price", "mean"),
            n_houses=("unit_price", "count"),
            n_communities=("community_name", "nunique"),
            community_names=(
                "community_name",
                lambda x: ";".join(sorted({v for v in x if pd.notna(v)})[:5]),
            ),
        )
        .reset_index()
    )

    out_dir = LABEL_DIR / city_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{city_key}_anjuke_grid_price.parquet"
    grid_yearly.to_parquet(out, index=False)
    print(
        f"  [OK] {city_key} ({city_name}): {n_raw:,} raw -> {n_valid:,} valid -> "
        f"{len(grid_yearly):,} grids -> {out.name}"
    )
    return len(grid_yearly)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build grid-level price labels from Anjuke listing snapshots"
    )
    p.add_argument("--city", default="all")
    args = p.parse_args()

    cities = ACTIVE_CITIES if args.city == "all" else [args.city]
    if args.city != "all" and args.city not in CITIES:
        print(f"ERROR: unknown city '{args.city}'")
        return 1

    total = 0
    for ck in cities:
        try:
            total += build_city_label(ck)
        except Exception as e:
            print(f"  [{ck}] ERROR: {e}")

    print(f"\nDone. {total:,} grid-price rows built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
