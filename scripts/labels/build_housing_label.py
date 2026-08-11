"""Build grid × year housing price labels from lianjia transaction data.

Steps per city:
  1. Load {city}_lianjia_transactions.csv
  2. Geocode unique communities via Amap API → lon/lat (with local cache)
  3. Aggregate community × year → mean unit_price, transaction count
  4. Assign each community to nearest 500m grid cell via grid parquet
  5. Aggregate grid × year → mean unit_price weighted by transaction count
  6. Save {city}_housing_grid_yearly.parquet

Output schema:
  grid_id, year, unit_price_mean, total_price_mean, n_transactions,
  n_communities, community_names

Usage:
    # Build labels from scraped data
    python scripts/labels/build_housing_label.py

    # Single city
    python scripts/labels/build_housing_label.py --city beijing

    # Geocode only (dry-run for community coverage check)
    python scripts/labels/build_housing_label.py --geocode-only
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from urban_intervention.config.project import ACTIVE_CITIES, CITIES, GRID_DIR
from urban_intervention.data.paths import GEOCODE_CACHE, LIANJIA_LABEL_DIR, RAW_WAYBACK_PARSED_DIR

# Reuse the validated GCJ-02 -> WGS-84 transform implemented in the
# Amap fetcher so all sources share one coordinate pipeline.
sys.path.insert(0, str(BASE_DIR / "scripts" / "collection"))
from amap_transit_fetcher import gcj02_to_wgs84

TRANSACTION_DIR = RAW_WAYBACK_PARSED_DIR
LABEL_DIR = LIANJIA_LABEL_DIR
LABEL_DIR.mkdir(parents=True, exist_ok=True)

AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"

# Maximum distance (meters) from a community centroid to the nearest grid
# centroid for the community to be assigned to that grid.  Prevents assigning
# communities that geocoded to a wrong city / out-of-range location.
MATCH_RADIUS_M = 1500


def _split_env_keys(value: str) -> list[str]:
    if not value:
        return []
    return [k.strip() for k in re.split(r"[,;\s]+", value) if k.strip()]


def _valid_key(value: str) -> bool:
    if not value:
        return False
    lowered = value.strip().lower()
    return lowered not in {
        "your_amap_key_here",
        "your_amap_key_1",
        "your_amap_key_2",
        "your_amap_key_3",
    }


def load_api_key() -> str:
    """Load a single Amap API key (first available) for geocoding.

    Supports the new multi-key format (web_api_keys list) and the legacy
    single-key format (web_api_key).  Geocoding uses one key at a time;
    if it gets exhausted, re-run with a different key in config.yaml.
    """
    p = BASE_DIR / "config.yaml"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            cfg = yaml.safe_load(f).get("amap", {})
        # New format: list of keys — use the first
        keys = [k for k in cfg.get("web_api_keys", []) if _valid_key(k)]
        if keys:
            return keys[0]
        # Legacy format: single key
        single = cfg.get("web_api_key", "")
        if _valid_key(single):
            return single
    for env_name in ("AMAP_API_KEYS", "AMAP_API_KEY"):
        keys = _split_env_keys(os.environ.get(env_name, ""))
        for key in keys:
            if _valid_key(key):
                return key
    return ""


def _load_geocode_cache() -> dict[str, dict]:
    if GEOCODE_CACHE.exists():
        with open(GEOCODE_CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_geocode_cache(cache: dict[str, dict]):
    with open(GEOCODE_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def geocode_community(
    community: str, city_name: str, api_key: str, cache: dict[str, dict]
) -> tuple[float, float] | None:
    """Geocode a community name via Amap API. Returns (lon, lat) in WGS-84 or None.

    Amap returns GCJ-02 coordinates; we convert to WGS-84 to match the
    grid layer (EPSG:4326 / WGS-84).

    Cache key = "{city_name}|{community}" to disambiguate same-named
    communities across different cities.

    Only deterministic failures (API returned a definitive "no match" or a
    parseable non-success status) are cached.  Transient network/parse
    errors are NOT cached so a later retry can succeed.
    """
    cache_key = f"{city_name}|{community}"
    if cache_key in cache:
        entry = cache[cache_key]
        if entry.get("lon") is not None:
            return entry["lon"], entry["lat"]
        # Cached only if marked as a deterministic miss (status key present)
        if "status" in entry:
            return None
        # Otherwise: previously failed transiently; fall through and retry.

    params = {
        "key": api_key,
        "address": community,
        "city": city_name,
        "output": "json",
    }
    try:
        resp = requests.get(AMAP_GEOCODE_URL, params=params, timeout=10)
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        # Transient: do NOT cache so a subsequent run can retry.
        print(f"    [geocode retry-able] {community}: {e}")
        return None

    if data.get("status") == "1" and data.get("geocodes"):
        geo = data["geocodes"][0]
        location = geo.get("location", "")
        try:
            gcj_lon_str, gcj_lat_str = location.split(",", 1)
            gcj_lon, gcj_lat = float(gcj_lon_str), float(gcj_lat_str)
        except (ValueError, AttributeError):
            # Deterministic: API returned success but bad payload — cache to
            # avoid wasting quota re-querying the same broken record.
            cache[cache_key] = {"lon": None, "lat": None, "status": "malformed_location"}
            return None
        # GCJ-02 -> WGS-84 (critical: grid layer is WGS-84)
        wgs_lon, wgs_lat = gcj02_to_wgs84(gcj_lon, gcj_lat)
        cache[cache_key] = {
            "lon": wgs_lon,
            "lat": wgs_lat,
            "formatted": geo.get("formatted_address", ""),
            "level": geo.get("level", ""),
        }
        return wgs_lon, wgs_lat

    # Deterministic miss: cache so we don't keep re-querying.
    cache[cache_key] = {"lon": None, "lat": None, "status": data.get("info", "unknown")}
    return None


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in meters between two WGS-84 points."""
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _vec_haversine_m(
    grid_lons: np.ndarray, grid_lats: np.ndarray, lon: float, lat: float
) -> np.ndarray:
    """Vectorized haversine: (G,) grid arrays vs a single query point.

    Returns an array of shape (G,) with distances in meters.
    """
    R = 6371000.0
    glat_r = np.radians(grid_lats)
    qlat_r = np.radians(lat)
    dlat = glat_r - qlat_r
    dlon = np.radians(grid_lons - lon)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(glat_r) * np.cos(qlat_r) * np.sin(dlon / 2.0) ** 2
    return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _weighted_mean(values, weights):
    """Safe weighted mean: drop pairs where either value or weight is
    missing / non-positive. Returns NaN if no valid pairs remain."""
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def run_city(city_key: str, api_key: str, geocode_only: bool = False) -> int:
    cfg = CITIES[city_key]
    city_name = cfg["name"]

    csv_path = TRANSACTION_DIR / f"{city_key}_lianjia_transactions.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {city_key}: no transaction CSV — run housing_price_fetcher.py first")
        return 0

    grid_path = GRID_DIR / city_key / f"{city_key}_grids.parquet"
    if not grid_path.exists():
        print(f"  [SKIP] {city_key}: no grid parquet — run grid_builder.py first")
        return 0

    print(f"\n{'=' * 50}\n{city_name} ({city_key})\n{'=' * 50}")

    # 1. Load transactions
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"  Loaded {len(df)} transactions")

    # Filter to valid records
    df = df[df["unit_price"].notna() & df["deal_year"].notna()].copy()
    df["deal_year"] = df["deal_year"].astype(int)
    print(f"  Valid records: {len(df)}")

    # 2. Geocode communities
    cache = _load_geocode_cache()
    communities = df["community"].dropna().unique()
    print(f"  Unique communities: {len(communities)}")

    geocoded = {}
    new_geocodes = 0
    last_save_at = 0
    for i, comm in enumerate(communities):
        if not comm or len(str(comm)) < 2:
            continue
        result = geocode_community(str(comm), city_name, api_key, cache)
        if result:
            geocoded[str(comm)] = result
            new_geocodes += 1
        # Progress
        if (i + 1) % 50 == 0:
            print(f"    Geocoding: {i + 1}/{len(communities)} ({new_geocodes} new hits)")
        if new_geocodes > 0 and new_geocodes % 20 == 0:
            time.sleep(random.uniform(0.3, 0.8))  # Amap free-tier rate limit
        # Incremental cache save every 100 new hits — protects against
        # crashes mid-run losing all geocode progress.
        if new_geocodes - last_save_at >= 100:
            _save_geocode_cache(cache)
            last_save_at = new_geocodes

    _save_geocode_cache(cache)
    matched = len(geocoded)
    print(
        f"  Geocoded: {matched}/{len(communities)} ({matched / max(1, len(communities)) * 100:.0f}%)"
    )

    if geocode_only:
        return matched

    if matched == 0:
        print("  [SKIP] No geocoded communities")
        return 0

    # 3. Attach coordinates to transactions
    df["lon"] = df["community"].map(
        lambda c: geocoded.get(str(c), (None, None))[0] if pd.notna(c) else None
    )
    df["lat"] = df["community"].map(
        lambda c: geocoded.get(str(c), (None, None))[1] if pd.notna(c) else None
    )
    df = df.dropna(subset=["lon", "lat"])

    # 4. Community × year aggregation
    comm_yearly = (
        df.groupby(["community", "deal_year"])
        .agg(
            unit_price_mean=("unit_price", "mean"),
            total_price_mean=("total_price", "mean"),
            n_transactions=("unit_price", "count"),
            lon=("lon", "first"),
            lat=("lat", "first"),
        )
        .reset_index()
    )
    comm_yearly.rename(columns={"deal_year": "year"}, inplace=True)
    print(f"  Community×year rows: {len(comm_yearly)}")

    # 5. Assign communities to nearest grid (haversine, meters — the grid
    #    layer is WGS-84 and the community coords are now WGS-84 too after
    #    the GCJ-02 -> WGS-84 conversion above).
    grids = pd.read_parquet(grid_path)
    grid_ids = grids["grid_id"].values.astype(object)
    grid_lons = grids["centroid_lon"].values.astype(float)
    grid_lats = grids["centroid_lat"].values.astype(float)

    assignments = []
    for _, row in comm_yearly.iterrows():
        clon, clat = float(row["lon"]), float(row["lat"])
        dists_m = _vec_haversine_m(grid_lons, grid_lats, clon, clat)
        nearest_idx = int(np.argmin(dists_m))
        min_dist_m = float(dists_m[nearest_idx])
        # Reject communities whose nearest grid is too far away — these are
        # almost certainly geocoding errors or out-of-city matches.
        if min_dist_m > MATCH_RADIUS_M:
            continue
        assignments.append(
            {
                "community": row["community"],
                "year": row["year"],
                "grid_id": grid_ids[nearest_idx],
                "unit_price_mean": row["unit_price_mean"],
                "total_price_mean": row["total_price_mean"],
                "n_transactions": row["n_transactions"],
            }
        )

    assigned = pd.DataFrame(assignments)
    assigned_pct = len(assigned) / max(1, len(comm_yearly)) * 100
    print(f"  Assigned to grid: {len(assigned)}/{len(comm_yearly)} ({assigned_pct:.0f}%)")

    if assigned.empty:
        print("  [SKIP] No grid assignments")
        return 0

    # 6. Grid × year aggregation
    grid_yearly = (
        assigned.groupby(["grid_id", "year"])
        .agg(
            unit_price_mean=(
                "unit_price_mean",
                lambda x: _weighted_mean(x, assigned.loc[x.index, "n_transactions"]),
            ),
            total_price_mean=(
                "total_price_mean",
                lambda x: _weighted_mean(x, assigned.loc[x.index, "n_transactions"]),
            ),
            n_transactions=("n_transactions", "sum"),
            n_communities=("community", "nunique"),
        )
        .reset_index()
    )

    # 7. Save
    city_label_dir = LABEL_DIR / city_key
    city_label_dir.mkdir(parents=True, exist_ok=True)
    out = city_label_dir / f"{city_key}_housing_grid_yearly.parquet"
    grid_yearly.to_parquet(out, index=False)
    print(
        f"  Saved: {out} ({len(grid_yearly)} grid-year rows, "
        f"years {grid_yearly.year.min()}-{grid_yearly.year.max()})"
    )

    return len(grid_yearly)


def main():
    p = argparse.ArgumentParser(
        description="Build grid-year housing price labels from lianjia data"
    )
    p.add_argument("--city", default="all")
    p.add_argument(
        "--geocode-only", action="store_true", help="Only geocode communities, don't build labels"
    )
    args = p.parse_args()

    api_key = load_api_key()
    if not api_key:
        print("ERROR: No Amap API key. Set amap.web_api_key in config.yaml")
        return 1

    cities = ACTIVE_CITIES if args.city == "all" else [args.city]
    if args.city != "all" and args.city not in CITIES:
        print(f"ERROR: unknown city '{args.city}'. Available: {list(CITIES.keys())}")
        return 1
    total = 0
    for ck in cities:
        try:
            n = run_city(ck, api_key, geocode_only=args.geocode_only)
            total += n
        except Exception as e:
            print(f"  [{ck}] ERROR: {e}")

    print(f"\nDone. {total} grid-year rows built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
