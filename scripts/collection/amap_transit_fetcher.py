"""Amap Metro Station Fetcher — polygon search by default.

Strategy:
  1. Polygon search for all cities (no 225-result text search cap)
  2. Adaptive quad density: 10×10 to 15×15 quads per city (~9-13km² each)
  3. Robust city-name filter handles Amap name variants (北京/北京市/北京城区)
  4. Overflow detection: warns if any quad hits 500+ stations
  5. Use --use-text for legacy text search ([WARN] may truncate)
"""

import argparse
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from urban_intervention.config.project import CITIES
from urban_intervention.config.project import norm_station_name as _norm

# ── GCJ-02 -> WGS84 (local implementation, no external dependency) ──
_PI = math.pi
_A = 6378245.0
_EE = 0.00669342162296594323


def _transform_lat(x: float, y: float) -> float:
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * _PI) + 40.0 * math.sin(y / 3.0 * _PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * _PI) + 320.0 * math.sin(y * _PI / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(x: float, y: float) -> float:
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * _PI) + 40.0 * math.sin(x / 3.0 * _PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * _PI) + 300.0 * math.sin(x / 30.0 * _PI)) * 2.0 / 3.0
    return ret


def gcj02_to_wgs84(lon: float, lat: float):
    """GCJ-02 -> WGS84 coordinate transform (China-specific)."""
    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * _PI
    magic = math.sin(radlat)
    magic = 1 - _EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrtmagic) * _PI)
    dlon = (dlon * 180.0) / (_A / sqrtmagic * math.cos(radlat) * _PI)
    return lon - dlon, lat - dlat


AMAP_TEXT = "https://restapi.amap.com/v3/place/text"
AMAP_POLYGON = "https://restapi.amap.com/v3/place/polygon"
PAGE_SIZE = 25
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0

# Known Amap text-search hard cap — text search is NOT used by default
AMAP_TEXT_CAP = 225

# ── Polygon search quad density (City -> quads per side) ──
# Smaller quads = lower per-quad station count = no API cap per quad
POLYGON_QUADS = {
    # Tier 1: 500~600 stations -> 15×15=225 quads, ~9km each, avg 2.2 stations/quad
    "beijing": 15,
    "shanghai": 15,
    "guangzhou": 12,
    "shenzhen": 12,
    "chengdu": 12,
    "chongqing": 12,
    "hangzhou": 12,
    "nanjing": 12,
    "wuhan": 12,
    "changsha": 12,
    "zhengzhou": 12,
    "tianjin": 12,
    "suzhou": 12,
    "xian": 12,
    "shenyang": 12,
    "qingdao": 12,
    # Tier 2: 200~400 stations -> 10×10=100 quads, ~13km each
    "dalian": 10,
    "kunming": 10,
    "ningbo": 10,
    "hefei": 10,
    "nanning": 10,
    "fuzhou": 10,
    "xiamen": 10,
    "wuxi": 10,
    "guiyang": 10,
    "shijiazhuang": 10,
    "changchun": 10,
    "jinan": 10,
    "harbin": 10,
    "foshan": 10,
    "dongguan": 10,
    "xuzhou": 10,
    "nanchang": 10,
    "wenzhou": 10,
    "luoyang": 10,
    "shaoxing": 10,
    "nantong": 10,
    "lanzhou": 10,
    "taiyuan": 10,
    "urumqi": 10,
    "hohhot": 10,
    "taizhou": 10,
}
DEFAULT_QUADS = 10  # for cities not in the map above


def _city_match(api_cityname: str, config_name: str) -> bool:
    """Check if Amap's cityname field matches the config city name.
    Handles variants: "北京"/"北京市"/"北京城区" all match config "北京".

    Returns False (reject) when cityname is missing, to avoid cross-city
    pollution from border stations that Amap returns without city info.
    """
    if not api_cityname:
        return False
    api_clean = api_cityname.strip()
    if api_clean == config_name or api_clean == config_name + "市":
        return True
    for suffix in ["市", "城区", "市区", "市辖区"]:
        bare = api_clean.removesuffix(suffix)
        if bare == config_name:
            return True
    return False


def _api_get(url: str, params: dict, timeout: int = 30) -> dict | None:
    """GET with exponential backoff + jitter.

    Returns None on persistent failure, or the JSON dict on success (status=1)
    or error (status != 1).  Callers should check `data.get("status")`.
    Raises SystemExit on non-retryable auth errors (invalid key).
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            data = resp.json()
            if data.get("status") == "1":
                return data
            # Fatal auth errors — abort the entire script
            if data.get("infocode") in ("10001", "10003"):
                raise SystemExit(
                    f"FATAL: Amap API auth error — {data.get('info', '')} "
                    "(check config.yaml web_api_key)"
                )
            # Other non-retryable errors — return as-is for callers to decide
            if data.get("infocode") in ("20000", "20001"):
                return data
            # Rate-limited or server error -> retry
            delay = RETRY_DELAY_BASE * (2**attempt) + random.uniform(0, 1)
            print(
                f"    API retry {attempt + 1}/{MAX_RETRIES} in {delay:.1f}s: {data.get('info', '')}"
            )
            time.sleep(delay)
        except (requests.ConnectionError, requests.Timeout):
            delay = RETRY_DELAY_BASE * (2**attempt) + random.uniform(0, 1)
            print(f"    Network retry {attempt + 1}/{MAX_RETRIES} in {delay:.1f}s")
            time.sleep(delay * 2)
        except Exception:
            time.sleep(RETRY_DELAY_BASE * 3)
    return None


def fetch_text(api_key, city_name):
    """Text search. Returns (deduped_df, raw_hits, api_reported_total).
    Caller should check raw_hits < api_reported_total to detect truncation.
    """
    all_raw = []
    api_total = 0
    for page in range(1, 50):
        params = {
            "key": api_key,
            "keywords": "地铁站",
            "types": "150500",
            "city": city_name,
            "citylimit": "true",
            "offset": PAGE_SIZE,
            "page": page,
            "extensions": "all",
        }
        data = _api_get(AMAP_TEXT, params)
        if data is None:
            if page == 1:
                print("  API error: request failed after retries")
            break
        if data.get("status") != "1":
            if page == 1:
                print(f"  API error: {data.get('info', '')}")
            break
        pois = data.get("pois", [])
        if not pois:
            break
        if page == 1:
            api_total = int(data.get("count", 0))
            print(f"  Total: {api_total} (text search)")
        for p in pois:
            if str(p.get("typecode", "")) != "150500":
                continue
            loc = p.get("location", "")
            if not loc:
                continue
            try:
                gcj_lon, gcj_lat = map(float, loc.split(","))
            except (ValueError, TypeError):
                continue
            all_raw.append(_mk_row(p, gcj_lon, gcj_lat))
        if len(pois) < PAGE_SIZE:
            break
        time.sleep(0.2)
    raw_count = len(all_raw)
    if raw_count >= AMAP_TEXT_CAP:
        print(f"  [WARN] Hit text search cap ({raw_count} raw)")
    return _dedup(all_raw), raw_count, api_total


def fetch_polygon(api_key, city_name, bbox, quads=10):
    """Polygon search. Splits bbox into quads×quads sub-regions.
    quads=10 -> 100 quads (~13×13km each), quads=15 -> 225 quads (~9×9km each).
    With small quads, each sub-query returns well under any API cap (safe for 600+ stations).
    """
    # Expand bbox slightly for search — grid bbox may clip edge stations.
    # 0.05° ≈ 4.3-5.5 km depending on latitude (smaller at higher latitudes).
    lon_min, lat_min, lon_max, lat_max = bbox
    pad = 0.05
    lon_min -= pad
    lat_min -= pad
    lon_max += pad
    lat_max += pad

    n = quads
    lons = [lon_min + (lon_max - lon_min) * i / n for i in range(n + 1)]
    lats = [lat_min + (lat_max - lat_min) * i / n for i in range(n + 1)]
    all_raw = []
    quads_overflow = 0
    quads_failed = 0
    quad_size_km = round((lon_max - lon_min) / n * 111, 1)

    print(
        f"  Polygon: {n}×{n}={n * n} quads (~{quad_size_km}×{quad_size_km}km), city='{city_name}'"
    )

    # Progress bar: print every 10%
    milestone = max(1, (n * n) // 10)

    for qi in range(n):
        for qj in range(n):
            q_idx = qi * n + qj
            poly = f"{lons[qi]},{lats[qj]}|{lons[qi + 1]},{lats[qj + 1]}"
            quad_stations = 0
            for page in range(1, 30):  # high limit, but small quads rarely go past page 1-2
                params = {
                    "key": api_key,
                    "types": "150500",
                    "polygon": poly,
                    "offset": PAGE_SIZE,
                    "page": page,
                    "extensions": "all",
                }
                data = _api_get(AMAP_POLYGON, params)
                if data is None:
                    quads_failed += 1
                    break
                if data.get("status") != "1":
                    quads_failed += 1
                    break
                pois = data.get("pois", [])
                if not pois:
                    break
                for p in pois:
                    if str(p.get("typecode", "")) != "150500":
                        continue
                    api_city = p.get("cityname", "")
                    if not _city_match(api_city, city_name):
                        continue
                    loc = p.get("location", "")
                    if not loc:
                        continue
                    try:
                        gcj_lon, gcj_lat = map(float, loc.split(","))
                    except (ValueError, TypeError):
                        continue
                    all_raw.append(_mk_row(p, gcj_lon, gcj_lat))
                    quad_stations += 1
                if len(pois) < PAGE_SIZE:
                    break
                time.sleep(0.3)  # respect Amap QPS limit (~3 QPS)
            # Track quads that might overflow
            if quad_stations >= 500:
                quads_overflow += 1
                if quads_overflow <= 3:  # only log first few
                    print(f"    [WARN] Quad [{qi},{qj}] dense: {quad_stations} stations")
            if q_idx % milestone == 0:
                print(f"    Progress: {q_idx}/{n * n} quads, {len(all_raw)} stations so far")
            time.sleep(0.5)  # 2 QPS, safe for Amap free tier

    print(
        f"  Polygon done: {len(all_raw)} raw hits "
        f"(overflow quads: {quads_overflow}, failed quads: {quads_failed}, total: {n * n})"
    )
    if quads_overflow > 0:
        print(f"  [WARN] {quads_overflow} quads had 500+ stations — consider increasing quads")
    if quads_failed > 0:
        print(f"  [WARN] {quads_failed} quads FAILED (network/API error) — stations may be missing")
    return _dedup(all_raw)


def _mk_row(poi, gcj_lon, gcj_lat):
    wgs_lon, wgs_lat = gcj02_to_wgs84(gcj_lon, gcj_lat)
    return {
        "station_name": poi.get("name", ""),
        "line": poi.get("address", ""),
        "wgs84_lon": round(wgs_lon, 7),
        "wgs84_lat": round(wgs_lat, 7),
    }


def _dedup(raw):
    """Dedup by normalized name + coordinate proximity (same-name stations >500m apart kept separate)."""
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df["_n"] = df["station_name"].apply(_norm)
    # Round coords to ~100m to cluster same-station entries, different-location same-name kept separate
    df["_clat"] = df["wgs84_lat"].round(3)
    df["_clon"] = df["wgs84_lon"].round(3)
    df = df.groupby(["_n", "_clat", "_clon"], as_index=False).first()
    df = df.drop(columns=["_n", "_clat", "_clon"]).reset_index(drop=True)
    return df


def _split_env_keys(value: str) -> list[str]:
    """Parse comma/semicolon/whitespace separated API keys from an env var."""
    if not value:
        return []
    return [k.strip() for k in re.split(r"[,;\s]+", value) if k.strip()]


def _valid_key(value: str) -> bool:
    """Reject empty template placeholders so config.yaml can stay checked-safe."""
    if not value:
        return False
    lowered = value.strip().lower()
    return lowered not in {
        "your_amap_key_here",
        "your_amap_key_1",
        "your_amap_key_2",
        "your_amap_key_3",
    }


def load_keys():
    """Load Amap API keys from config.yaml.

    Returns a list of keys.  Supports both the new ``web_api_keys`` (list)
    and the legacy ``web_api_key`` (single string) for backward compat.
    Also falls back to ``AMAP_API_KEYS`` / ``AMAP_API_KEY`` env vars.
    """
    p = BASE_DIR / "config.yaml"
    keys = []
    if p.exists():
        with open(p, encoding="utf-8") as f:
            cfg = yaml.safe_load(f).get("amap", {})
        # New format: list of keys
        keys = [k for k in cfg.get("web_api_keys", []) if _valid_key(k)]
        # Legacy format: single key
        single = cfg.get("web_api_key", "")
        if _valid_key(single) and single not in keys:
            keys.append(single)
    for env_name in ("AMAP_API_KEYS", "AMAP_API_KEY"):
        for env_key in _split_env_keys(os.environ.get(env_name, "")):
            if _valid_key(env_key) and env_key not in keys:
                keys.append(env_key)
    return keys


def _get_quads(ck: str) -> int:
    """Return recommended quad count for a city."""
    return POLYGON_QUADS.get(ck, DEFAULT_QUADS)


def _fetch_city_smart(
    api_key: str,
    ck: str,
    cfg: dict,
    force_polygon: bool = False,  # noqa: ARG001
    use_text: bool = False,
):  # kept for external import use, main() calls fetch_polygon directly
    """Fetch with configurable strategy. Default: polygon search for all cities."""
    name = cfg["name"]
    bbox = cfg["bbox"]

    # ── Option A: text search (--use-text, for quick diagnostics) ──
    if use_text and not force_polygon:
        df, raw_hits, api_reported = fetch_text(api_key, name)
        is_truncated = raw_hits >= AMAP_TEXT_CAP or (api_reported > 0 and raw_hits < api_reported)
        if is_truncated:
            print(
                f"  [WARN] Truncation detected ({raw_hits}/{api_reported}), but --use-text set -> no fallback"
            )
        return df

    # ── Option B: polygon search (default) ──
    quads = _get_quads(ck)
    return fetch_polygon(api_key, name, bbox, quads=quads)


def _split_batches(cities: list[str], n_keys: int) -> list[list[str]]:
    """Split a city list into ``n_keys`` roughly-equal batches.

    Example: 44 cities / 3 keys → [15, 15, 14].
    """
    if n_keys <= 0:
        return [cities]
    batch_size = -(-len(cities) // n_keys)  # ceil division
    return [cities[i : i + batch_size] for i in range(0, len(cities), batch_size)]


def main():
    p = argparse.ArgumentParser(
        description="Amap metro station fetcher — polygon search by default"
    )
    p.add_argument("--city", default="all", help="City key or 'all' (comma-separated for multiple)")
    p.add_argument(
        "--use-text",
        action="store_true",
        help="Use text search instead of default polygon ([WARN] may truncate at 225)",
    )
    p.add_argument(
        "--quads",
        type=int,
        default=None,
        help="Override quad count (e.g. --quads 20 for ultra-dense)",
    )
    p.add_argument(
        "--key",
        default=None,
        help="Single Amap API key (overrides config.yaml). "
        "Use --batch instead for multi-key batch mode.",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch number (1-based). Splits all 44 cities into "
        "N batches where N = number of keys in config.yaml. "
        "Batch 1 uses key 1, batch 2 uses key 2, etc. "
        "Example: --batch 1 runs cities 1-15 with the first key.",
    )
    args = p.parse_args()

    # ── Resolve cities ──────────────────────────────────────────
    all_cities = list(CITIES.keys())
    if args.city == "all":
        cities = all_cities
    else:
        cities = [c.strip() for c in args.city.split(",") if c.strip() in CITIES]

    if not cities:
        print("ERROR: No valid cities specified.")
        return 1

    # ── Resolve API key ────────────────────────────────────────
    keys = load_keys()
    if args.key:
        # Single key mode — run all specified cities with this key
        key = args.key
        print(f"Using API key ...{key[-6:]} (single-key mode)")
    elif args.batch is not None:
        # Batch mode — split all 44 cities into len(keys) batches
        if not keys:
            print("ERROR: No API keys in config.yaml. Configure amap.web_api_keys.")
            return 1
        n_batches = len(keys)
        if args.batch < 1 or args.batch > n_batches:
            print(f"ERROR: --batch must be 1..{n_batches} (you have {n_batches} keys).")
            return 1
        batches = _split_batches(all_cities, n_batches)
        batch_cities = batches[args.batch - 1]
        key = keys[args.batch - 1]
        # Intersect batch cities with --city selection (default: all → full batch)
        if args.city == "all":
            cities = batch_cities
        else:
            cities = [c for c in batch_cities if c in set(cities)]
        print(f"Batch {args.batch}/{n_batches}: {len(cities)} cities with key ...{key[-6:]}")
        print(f"  Cities: {', '.join(cities)}")
    else:
        # Default: use the first key from config
        if not keys:
            print(
                "ERROR: No API key. Configure config.yaml amap.web_api_keys "
                "or pass --key, or set AMAP_API_KEY env var."
            )
            return 1
        key = keys[0]
        print(f"Using API key ...{key[-6:]} (first key from config.yaml)")
        print("  Tip: use --batch 1/2/3 to split cities across multiple keys.")

    if not cities:
        print("No cities to process after batch selection.")
        return 0

    # ── Fetch each city ────────────────────────────────────────
    total_stations = 0
    for ck in cities:
        cfg = CITIES[ck]
        name = cfg["name"]
        quads = args.quads if args.quads else _get_quads(ck)

        print(f"\n{'=' * 40}\n{name} (quads={quads}×{quads}={quads * quads})\n{'=' * 40}")

        if args.use_text:
            print("  Mode: text search ([WARN] may truncate)")
            df, raw, _ = fetch_text(key, name)
            if raw >= AMAP_TEXT_CAP:
                print(f"  [WARN] TRUNCATED at cap ({raw} raw)")
        else:
            print("  Mode: polygon search")
            df = fetch_polygon(key, name, cfg["bbox"], quads=quads)

        if df.empty:
            print("  [!] No data")
            continue

        out = BASE_DIR / "data" / "archive" / "raw" / "transit" / ck
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / f"{ck}_metro_stations_amap.csv", index=False, encoding="utf-8-sig")
        total_stations += len(df)
        print(f"  [OK] {len(df)} unique stations -> {out}")
        time.sleep(2)

    print(f"\n{'=' * 40}\nDone. {len(cities)} cities, {total_stations} total unique stations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
