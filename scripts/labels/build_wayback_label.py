"""Build grid × year housing price labels from Wayback Machine snapshots.

This is the time-series complement to build_anjuke_label.py:
  - build_anjuke_label.py: cross-sectional baseline (2025 snapshot, no time)
  - build_wayback_label.py: time-series panel (2014-2024, from Wayback archives)

The Wayback snapshots provide community-level listing prices at different
points in time. We geocode community names (reusing the cache from
build_housing_label.py), match to nearest 500m grid, and aggregate to
grid × year.

Output: data/active/labels/{city}/{city}_wayback_grid_yearly.parquet
  grid_id, year, unit_price_mean, unit_price_median, n_communities,
  n_snapshots, community_names

Usage:
    python scripts/labels/build_wayback_label.py
    python scripts/labels/build_wayback_label.py --city beijing
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR / "scripts" / "collection"))
from amap_transit_fetcher import gcj02_to_wgs84

from urban_intervention.config.project import ACTIVE_CITIES, CITIES, GRID_DIR
from urban_intervention.data.paths import (
    GEOCODE_CACHE,
    RAW_ANJUKE_DIR,
    RAW_WAYBACK_PARSED_DIR,
    WAYBACK_LABEL_DIR,
)

# Canonical location written by wayback_research_scraper.py.  Older scripts
# wrote to data/archive/raw/housing, which disconnected fresh collection from labels.
TRANSACTION_DIR = RAW_WAYBACK_PARSED_DIR
LABEL_DIR = WAYBACK_LABEL_DIR
AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
MATCH_RADIUS_M = 1500

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


def _split_env_keys(value: str) -> list[str]:
    if not value:
        return []
    return [k.strip() for k in re.split(r"[,;\s]+", value) if k.strip()]


def _valid_key(value: str) -> bool:
    if not value:
        return False
    return value.strip().lower() not in {
        "your_amap_key_here",
        "your_amap_key_1",
        "your_amap_key_2",
        "your_amap_key_3",
    }


def load_api_key() -> str:
    p = BASE_DIR / "config.yaml"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            cfg = yaml.safe_load(f).get("amap", {})
        keys = [k for k in cfg.get("web_api_keys", []) if _valid_key(k)]
        if keys:
            return keys[0]
        single = cfg.get("web_api_key", "")
        if _valid_key(single):
            return single
    for env_name in ("AMAP_API_KEYS", "AMAP_API_KEY"):
        keys = _split_env_keys(os.environ.get(env_name, ""))
        for key in keys:
            if _valid_key(key):
                return key
    return ""


def _load_geocode_cache() -> dict:
    if GEOCODE_CACHE.exists():
        with open(GEOCODE_CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_geocode_cache(cache: dict):
    with open(GEOCODE_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def geocode_community(
    community: str, city_name: str, api_key: str, cache: dict
) -> tuple[float, float] | None:
    cache_key = f"{city_name}|{community}"
    if cache_key in cache:
        entry = cache[cache_key]
        if entry.get("lon") is not None:
            return entry["lon"], entry["lat"]
        if "status" in entry:
            return None
    params = {"key": api_key, "address": community, "city": city_name, "output": "json"}
    try:
        resp = requests.get(AMAP_GEOCODE_URL, params=params, timeout=10)
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None
    if data.get("status") == "1" and data.get("geocodes"):
        geo = data["geocodes"][0]
        location = geo.get("location", "")
        try:
            gcj_lon_str, gcj_lat_str = location.split(",", 1)
            gcj_lon, gcj_lat = float(gcj_lon_str), float(gcj_lat_str)
        except (ValueError, AttributeError):
            cache[cache_key] = {"lon": None, "lat": None, "status": "malformed"}
            return None
        wgs_lon, wgs_lat = gcj02_to_wgs84(gcj_lon, gcj_lat)
        cache[cache_key] = {
            "lon": wgs_lon,
            "lat": wgs_lat,
            "formatted": geo.get("formatted_address", ""),
        }
        return wgs_lon, wgs_lat
    cache[cache_key] = {"lon": None, "lat": None, "status": data.get("info", "unknown")}
    return None


def _load_anjuke_geocodes(city_key: str) -> dict[str, tuple[float, float]]:
    """Load community coordinates from the Anjuke house.csv for this city.

    The Anjuke data has built-in GCJ-02 coordinates for each listing. We
    aggregate to community-level mean coordinates and convert to WGS-84.
    This provides free geocoding without needing an Amap API key.
    """
    import csv

    anjuke_dir = RAW_ANJUKE_DIR
    city_name = CITIES[city_key]["name"]
    candidates = list(anjuke_dir.glob(f"{city_name}*_house.csv"))
    if not candidates:
        return {}
    geocodes: dict[str, list[tuple[float, float]]] = {}
    with open(candidates[0], encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        {header[1]: "community", header[2]: "coord"}
        for row in reader:
            comm = row[1]
            coord_str = row[2]
            m = re.search(r"POINT\(\s*([\d.]+)\s+([\d.]+)\s*\)", coord_str)
            if not m:
                continue
            gcj_lon, gcj_lat = float(m.group(1)), float(m.group(2))
            wgs_lon, wgs_lat = gcj02_to_wgs84(gcj_lon, gcj_lat)
            geocodes.setdefault(comm, []).append((wgs_lon, wgs_lat))
    # Aggregate to mean per community
    result = {}
    for comm, coords in geocodes.items():
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        result[comm] = (sum(lons) / len(lons), sum(lats) / len(lats))
    return result


def build_city_label(city_key: str, api_key: str) -> int:
    """Build grid-year labels from both chengjiao (transactions) and xiaoqu (listings)."""
    cfg = CITIES[city_key]
    city_name = cfg["name"]
    grid_path = GRID_DIR / city_key / f"{city_key}_grids.parquet"
    if not grid_path.exists():
        print(f"  [SKIP] {city_key}: no grid parquet")
        return 0

    print(f"\n{'=' * 50}\n{city_name} ({city_key})\n{'=' * 50}")

    # Load Anjuke geocodes (free, no API key)
    anjuke_geocodes = _load_anjuke_geocodes(city_key)

    # API geocodes for unmatched
    cache = _load_geocode_cache()
    api_key_local = api_key

    all_grid_yearly = []

    # ---- Part 1: Chengjiao (transactions with deal_year) ----
    cj_path = TRANSACTION_DIR / f"{city_key}_wayback_chengjiao.csv"
    if cj_path.exists():
        df = pd.read_csv(cj_path, encoding="utf-8-sig")
        df = df[df["unit_price"].notna() & df["deal_year"].notna()].copy()
        if not df.empty:
            print(f"  [chengjiao] {len(df)} transactions, {df['community'].nunique()} communities")
            gy = _process_df_to_grid_yearly(
                df,
                city_key,
                city_name,
                grid_path,
                anjuke_geocodes,
                cache,
                api_key_local,
                year_col="deal_year",
                source_label="lianjia_chengjiao",
            )
            if gy is not None and not gy.empty:
                all_grid_yearly.append(gy)
                print(f"  [chengjiao] -> {len(gy)} grid-year rows")

    # ---- Part 2: Xiaoqu (listings with snapshot_year) ----
    xq_path = TRANSACTION_DIR / f"{city_key}_wayback_xiaoqu.csv"
    if xq_path.exists():
        df = pd.read_csv(xq_path, encoding="utf-8-sig")
        df = df[df["unit_price"].notna()].copy()
        if not df.empty:
            print(f"  [xiaoqu] {len(df)} listings, {df['community'].nunique()} communities")
            gy = _process_df_to_grid_yearly(
                df,
                city_key,
                city_name,
                grid_path,
                anjuke_geocodes,
                cache,
                api_key_local,
                year_col="snapshot_year",
                source_label="lianjia_xiaoqu",
            )
            if gy is not None and not gy.empty:
                all_grid_yearly.append(gy)
                print(f"  [xiaoqu] -> {len(gy)} grid-year rows")

    # ---- Part 3: Anjuke Wayback (listings with snapshot_year, 2012-2018) ----
    aj_path = TRANSACTION_DIR / f"{city_key}_wayback_anjuke.csv"
    if aj_path.exists():
        df = pd.read_csv(aj_path, encoding="utf-8-sig")
        df = df[df["unit_price"].notna()].copy()
        if not df.empty:
            print(f"  [anjuke] {len(df)} listings, {df['community'].nunique()} communities")
            gy = _process_df_to_grid_yearly(
                df,
                city_key,
                city_name,
                grid_path,
                anjuke_geocodes,
                cache,
                api_key_local,
                year_col="snapshot_year",
                source_label="anjuke",
            )
            if gy is not None and not gy.empty:
                all_grid_yearly.append(gy)
                print(f"  [anjuke] -> {len(gy)} grid-year rows")

    # ---- Part 4: Beike chengjiao + xiaoqu ----
    for bk_suffix, bk_label, bk_year_col in [
        ("beike_chengjiao", "beike_cj", "deal_year"),
        ("beike_xiaoqu", "beike_xq", "snapshot_year"),
    ]:
        bk_path = TRANSACTION_DIR / f"{city_key}_wayback_{bk_suffix}.csv"
        if bk_path.exists():
            df = pd.read_csv(bk_path, encoding="utf-8-sig")
            df = df[df["unit_price"].notna()].copy()
            if not df.empty:
                print(f"  [{bk_label}] {len(df)} rows, {df['community'].nunique()} communities")
                gy = _process_df_to_grid_yearly(
                    df,
                    city_key,
                    city_name,
                    grid_path,
                    anjuke_geocodes,
                    cache,
                    api_key_local,
                    year_col=bk_year_col,
                    source_label=bk_suffix,
                )
                if gy is not None and not gy.empty:
                    all_grid_yearly.append(gy)
                    print(f"  [{bk_label}] -> {len(gy)} grid-year rows")

    if not all_grid_yearly:
        print(f"  [SKIP] {city_key}: no data")
        return 0

    # Merge chengjiao + xiaoqu grid-year, preferring chengjiao prices
    combined = pd.concat(all_grid_yearly, ignore_index=True)
    combined = combined.groupby(["grid_id", "year"], as_index=False).agg(
        {
            "unit_price_mean": "mean",
            "unit_price_median": "mean",
            "n_communities": "sum",
            "n_snapshots": "sum",
            "centroid_lon": "first",
            "centroid_lat": "first",
            "source": lambda x: ";".join(set(x)),
        }
    )

    out_dir = LABEL_DIR / city_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{city_key}_wayback_grid_yearly.parquet"
    combined.to_parquet(out, index=False)
    print(
        f"  [OK] {len(combined)} grid-year rows, "
        f"years {combined['year'].min()}-{combined['year'].max()} -> {out.name}"
    )
    return len(combined)


def _process_df_to_grid_yearly(
    df,
    city_key,
    city_name,
    grid_path,
    anjuke_geocodes,
    cache,
    api_key,
    year_col="deal_year",
    source_label="lianjia_chengjiao",
):
    """Shared logic: geocode -> grid match -> aggregate to grid-year."""
    df = df.copy()
    df[year_col] = df[year_col].astype(int)

    # Geocode: exact match first, then fuzzy match
    communities = df["community"].dropna().unique()
    all_geocodes = dict(anjuke_geocodes)
    # Fuzzy match for unmatched communities
    from difflib import SequenceMatcher

    fuzzy_matches = {}
    for comm in communities:
        s_comm = str(comm)
        if s_comm in all_geocodes:
            continue
        s_clean = re.sub(r"[（）()\s\-·\.]", "", s_comm.lower())
        if len(s_clean) < 2:
            continue
        best_score = 0.0
        best_name = None
        for name in anjuke_geocodes:
            n_clean = re.sub(r"[（）()\s\-·\.]", "", name.lower())
            if s_clean in n_clean or n_clean in s_clean:
                score = min(len(s_clean), len(n_clean)) / max(len(s_clean), len(n_clean))
                if score > best_score and score > 0.6:
                    best_score = score
                    best_name = name
            else:
                score = SequenceMatcher(None, s_clean, n_clean).ratio()
                if score > best_score and score > 0.85:
                    best_score = score
                    best_name = name
        if best_name:
            fuzzy_matches[s_comm] = best_name
    # Copy coordinates from fuzzy matches
    for comm, matched_name in fuzzy_matches.items():
        all_geocodes[comm] = anjuke_geocodes[matched_name]

    exact_n = sum(1 for c in communities if str(c) in anjuke_geocodes)
    fuzzy_n = len(fuzzy_matches)
    total_n = len(communities)
    pct = (exact_n + fuzzy_n) / max(total_n, 1) * 100
    print(f"  Geocode: {exact_n} exact + {fuzzy_n} fuzzy / {total_n} total = {pct:.0f}%")

    # Try API for remaining unmatched
    unmatched = [c for c in communities if str(c) not in all_geocodes and api_key]
    for comm in unmatched[:100]:
        result = geocode_community(str(comm), city_name, api_key, cache)
        if result:
            all_geocodes[str(comm)] = result
    _save_geocode_cache(cache)

    df["lon"] = df["community"].map(
        lambda c: all_geocodes.get(str(c), (None, None))[0] if pd.notna(c) else None
    )
    df["lat"] = df["community"].map(
        lambda c: all_geocodes.get(str(c), (None, None))[1] if pd.notna(c) else None
    )
    df = df.dropna(subset=["lon", "lat"])
    if df.empty:
        return None

    # Grid match
    if cKDTree is None:
        return None
    grids = pd.read_parquet(grid_path)
    lat_c = float(grids["centroid_lat"].mean())
    cos_lat = max(0.1, math.cos(math.radians(lat_c)))
    tree_pts = np.column_stack(
        [
            grids["centroid_lon"].values * cos_lat,
            grids["centroid_lat"].values,
        ]
    )
    query_pts = np.column_stack(
        [
            df["lon"].values.astype(float) * cos_lat,
            df["lat"].values.astype(float),
        ]
    )
    tree = cKDTree(tree_pts)
    dist_deg, idx_arr = tree.query(query_pts, k=1)
    dist_m = dist_deg * 111000.0
    df["grid_id"] = grids["grid_id"].values[idx_arr]
    df["dist_m"] = dist_m
    df = df[df["dist_m"] <= MATCH_RADIUS_M]
    if df.empty:
        return None

    # Aggregate (include coordinates)
    gy = (
        df.groupby(["grid_id", year_col])
        .agg(
            unit_price_mean=("unit_price", "mean"),
            unit_price_median=("unit_price", "median"),
            n_communities=("community", "nunique"),
            n_snapshots=("community", "count"),
            centroid_lon=("lon", "mean"),
            centroid_lat=("lat", "mean"),
        )
        .reset_index()
    )
    gy.rename(columns={year_col: "year"}, inplace=True)
    gy["source"] = source_label
    return gy


def main() -> int:
    p = argparse.ArgumentParser(description="Build grid-year housing labels from Wayback snapshots")
    p.add_argument("--city", default="all")
    args = p.parse_args()

    api_key = load_api_key()
    if api_key:
        print("Amap API key found — will geocode unmatched communities via Amap")
    else:
        print("No Amap API key — will rely on Anjuke community coordinates only")

    cities = ACTIVE_CITIES if args.city == "all" else [args.city]
    total = 0
    for ck in cities:
        try:
            total += build_city_label(ck, api_key)
        except Exception as e:
            print(f"  [{ck}] ERROR: {e}")
    print(f"\nDone. {total:,} grid-year rows built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
