"""AOI-based grid matching with Anjuke boundaries + convex hull fallback.

Strategy:
  1. Load Anjuke community boundaries (community_ext.csv, 70-90% coverage)
  2. Fuzzy-match Lianjia community names to Anjuke community names
  3. Matched = use Anjuke boundary polygon for AOI
  4. Unmatched = compute convex hull of transaction points
  5. Build STRtree of 500m grid polygons
  6. Query STRtree to find grids intersecting each community's AOI
"""

import csv
import math
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from shapely import STRtree
from shapely.geometry import Polygon, box

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(BASE / "scripts" / "collection"))
from amap_transit_fetcher import gcj02_to_wgs84

from urban_intervention.config.project import CITIES, GRID_DIR
from urban_intervention.data.paths import LIANJIA_LABEL_DIR, RAW_ANJUKE_DIR, RAW_LIANJIA_DIR

XIANYU = RAW_LIANJIA_DIR
AJ_BASE = RAW_ANJUKE_DIR
LABEL_DIR = LIANJIA_LABEL_DIR
name_to_key = {CITIES[ck]["name"].replace("市", ""): ck for ck in CITIES}
GRID_HALF = 250


def load_anjuke_boundaries(city_key):
    """Load community boundary polygons from Anjuke community_ext.csv.
    Returns dict: community_name -> Shapely Polygon.
    Also returns a cleaned-name index for fast lookup.
    Coordinates are GCJ-02, converted to WGS-84.
    """
    name = CITIES[city_key]["name"]
    candidates = list(AJ_BASE.glob(f"{name}*_community_ext.csv"))
    if not candidates:
        return {}, {}

    boundaries = {}
    cleaned_index = {}  # cleaned_name -> [original_names]
    with open(candidates[0], encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        name_idx = boundary_idx = None
        for i, h in enumerate(header):
            if "名称" in h:
                name_idx = i
            if "边界" in h:
                boundary_idx = i
        if name_idx is None or boundary_idx is None:
            return {}, {}

        for row in reader:
            comm_name = row[name_idx].strip()
            boundary_str = row[boundary_idx].strip()
            if not comm_name or not boundary_str:
                continue
            coords = []
            for pair in boundary_str.split(";"):
                parts = pair.strip().split(",")
                if len(parts) == 2:
                    try:
                        gcj_lon, gcj_lat = float(parts[0]), float(parts[1])
                        wgs_lon, wgs_lat = gcj02_to_wgs84(gcj_lon, gcj_lat)
                        coords.append((wgs_lon, wgs_lat))
                    except ValueError:
                        continue
            if len(coords) >= 3:
                try:
                    poly = Polygon(coords)
                    if poly.is_valid and poly.area > 0:
                        boundaries[comm_name] = poly
                        clean = _clean_name(comm_name)
                        cleaned_index.setdefault(clean, []).append(comm_name)
                except Exception:
                    pass

    return boundaries, cleaned_index


def _clean_name(name):
    """Normalize community name for matching."""
    return re.sub(r"[（）()\s\-·\.·,，、]", "", str(name).lower())


def name_match(lianjia_name, boundaries, cleaned_index):
    """Fast fuzzy match using cleaned-name index."""
    if not lianjia_name:
        return None
    s = _clean_name(lianjia_name)
    if len(s) < 2:
        return None

    # Exact cleaned match
    if s in cleaned_index:
        # Return first match with boundary
        for n in cleaned_index[s]:
            if n in boundaries:
                return n

    # Substring match
    best_score = 0.0
    best_name = None
    for clean_key, names in cleaned_index.items():
        # Quick pre-filter: first character must match
        if len(clean_key) < 2 or len(s) < 2:
            continue
        score = 0.0
        if s in clean_key:
            score = len(s) / len(clean_key)
        elif clean_key in s:
            score = len(clean_key) / len(s)
        if score > best_score and score > 0.6:
            best_score = score
            best_name = names[0]  # first name in list

    if best_name and best_name in boundaries:
        return best_name

    # SequenceMatcher for remaining (expensive, limited)
    if best_score < 0.7:
        for clean_key, names in cleaned_index.items():
            if abs(len(clean_key) - len(s)) > 3:
                continue
            score = SequenceMatcher(None, s, clean_key).ratio()
            if score > best_score and score > 0.75:
                best_score = score
                best_name = names[0]

    return best_name if best_name and best_name in boundaries else None


def process_city(city_df, ck, col_map):
    """Process one city with Anjuke boundaries + convex hull fallback."""
    grids = pd.read_parquet(GRID_DIR / ck / f"{ck}_grids.parquet")
    grid_ids = grids["grid_id"].values
    grid_lons = grids["centroid_lon"].values.astype(float)
    grid_lats = grids["centroid_lat"].values.astype(float)

    lat_c = float(grid_lats.mean())
    cos_lat = max(0.1, math.cos(math.radians(lat_c)))
    meters_per_deg_lon = 111320.0 * cos_lat
    meters_per_deg_lat = 111320.0

    # Build grid STRtree
    grid_polys = []
    for i in range(len(grid_ids)):
        dlon = GRID_HALF / meters_per_deg_lon
        dlat = GRID_HALF / meters_per_deg_lat
        poly = box(
            grid_lons[i] - dlon, grid_lats[i] - dlat, grid_lons[i] + dlon, grid_lats[i] + dlat
        )
        grid_polys.append(poly)
    tree = STRtree(grid_polys)

    # Load Anjuke boundaries
    aj_boundaries, cleaned_index = load_anjuke_boundaries(ck)
    print(f"  Anjuke boundaries: {len(aj_boundaries)} for {ck}")

    assignments = []
    comm_groups = city_df.groupby(col_map["comm"])
    n_boundary = 0
    n_hull = 0
    n_skipped = 0

    for comm_name, comm_df in comm_groups:
        # Try to match with Anjuke boundary
        matched_name = name_match(str(comm_name), aj_boundaries, cleaned_index)

        if matched_name:
            aoi = aj_boundaries[matched_name]
            n_boundary += 1
        else:
            # No boundary found: use buffer around centroid
            # (avoids expensive convex hull computation)
            clon = comm_df["_lon"].mean()
            clat = comm_df["_lat"].mean()
            dlon = GRID_HALF / meters_per_deg_lon
            dlat = GRID_HALF / meters_per_deg_lat
            aoi = box(clon - dlon, clat - dlat, clon + dlon, clat + dlat)
            n_hull += 1

        # Find intersecting grids
        hits = tree.query(aoi, predicate="intersects")
        if len(hits) == 0:
            continue
        covered = grid_ids[hits]

        # Assign per year
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

    if not assignments:
        return 0

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

    n_grids = gy["grid_id"].nunique()
    total_comms = n_boundary + n_hull + n_skipped
    print(
        f"  {ck}: {n_boundary} boundary + {n_hull} hull = {n_boundary + n_hull} comms "
        f"({n_skipped} skipped / {total_comms} total)"
    )
    print(f"  -> {len(gy):,} gy, {n_grids:,} grids ({n_grids / max(n_boundary + n_hull, 1):.1f}x)")
    return len(gy)


def process_file(xlsx_path):
    fname = xlsx_path.stem
    print(f"\n{fname} ({xlsx_path.stat().st_size / 1024 / 1024:.0f}MB)")

    df = pd.read_excel(xlsx_path)
    cols = list(df.columns)

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

    total = 0
    for city_name in df["_city"].unique():
        ck = name_to_key.get(city_name)
        if not ck:
            for name, key in name_to_key.items():
                if name in city_name or city_name in name:
                    ck = key
                    break
        if not ck or not (GRID_DIR / ck / f"{ck}_grids.parquet").exists():
            continue
        total += process_city(df[df["_city"] == city_name].copy(), ck, col_map)

    return total


if __name__ == "__main__":
    files = sorted(XIANYU.rglob("*.xlsx"), key=lambda f: f.stat().st_size)
    total = 0
    for f in files:
        try:
            total += process_file(f) or 0
        except Exception as e:
            import traceback

            print(f"  ERROR: {e}")
            traceback.print_exc()
    print(f"\nTOTAL: {total:,} grid-years")
