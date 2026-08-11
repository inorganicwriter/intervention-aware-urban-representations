"""Fetch admin boundaries for 44 Chinese cities via OSM Overpass or GADM.

Two backends, tried in order:
  1. GADM (local .gpkg file) — fastest, works offline after single download
  2. OSM Overpass API — live query, needs network access

Outputs per city:
  data/active/reference/boundaries/{city_key}_boundary.geojson  (polygon)
  data/active/reference/boundaries/{city_key}_bbox.json          (envelope)

Also regenerates pipeline_config CITIES bbox values from fetched boundaries.

Usage:
    # Download GADM first (one-time, ~12 MB):
    python scripts/collection/admin_boundary_fetcher.py --download-gadm

    # Then fetch boundaries using GADM (fast, offline):
    python scripts/collection/admin_boundary_fetcher.py --backend gadm

    # Or use Overpass API (auto-detects Clash proxy at 127.0.0.1:7890):
    python scripts/collection/admin_boundary_fetcher.py --backend overpass

    # Default: auto (tries GADM first, falls back to Overpass):
    python scripts/collection/admin_boundary_fetcher.py

    # Single city:
    python scripts/collection/admin_boundary_fetcher.py --city beijing

    # Regenerate pipeline_config CITIES bbox from boundaries:
    python scripts/collection/admin_boundary_fetcher.py --update-config
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from urban_intervention.config.project import ACTIVE_CITIES, CITIES, get_proxies, set_proxy
from urban_intervention.data.paths import BOUNDARY_DIR

BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
GADM_GPKG = BOUNDARY_DIR / "gadm41_CHN.gpkg"
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_CHN.gpkg"

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
HEADERS = {
    "User-Agent": "MIT-Summer-Research/1.0 (admin boundary collection)",
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept": "application/json",
}

# Proxy auto-detection is handled by pipeline_config (get_proxies / get_proxy
# / set_proxy).  We just call those functions — no duplicate logic here.

# Administrative-level mapping.  Prefecture-level cities (地级市) typically
# use admin_level=5 in OSM[1].  Province-level municipalities (直辖市: 北京、
# 上海、天津、重庆) use admin_level=4, but their boundaries cover ~16,000 km2.
# We fetch the municipal boundary anyway and let the buffer parameter control
# how far beyond it we extend.
#
# [1] OSM CN tagging convention: 4=province, 5=prefecture, 6=county, 7=town.

DIRECT_ADMIN_CITIES = {"beijing", "shanghai", "tianjin", "chongqing"}

# ── GADM backend ─────────────────────────────────────────────────


def _download_gadm() -> bool:
    """Download GADM China GeoPackage (~12 MB)."""
    if GADM_GPKG.exists():
        print(f"GADM already cached: {GADM_GPKG} ({GADM_GPKG.stat().st_size / 1e6:.1f} MB)")
        return True
    print(f"Downloading {GADM_URL} ...")
    try:
        resp = requests.get(GADM_URL, stream=True, timeout=300, proxies=get_proxies())
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(GADM_GPKG, "wb") as f:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="")
            print()
        print(f"Saved: {GADM_GPKG} ({GADM_GPKG.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        if GADM_GPKG.exists():
            GADM_GPKG.unlink()
        return False


def _fetch_gadm(city_key: str) -> dict | None:
    """Extract a single city's boundary from the GADM GeoPackage.

    Matches by English name (NAME_2 for prefectures, NAME_1 for zhixiashi)
    using the CN_TO_EN mapping, with province disambiguation.
    """
    if not GADM_GPKG.exists():
        print(f"  GADM not found: {GADM_GPKG}")
        print("  Run with --download-gadm first, or manually download from:")
        print(f"  {GADM_URL}")
        return None

    try:
        import geopandas as gpd
    except ImportError:
        print("  geopandas not installed; cannot use GADM backend")
        return None

    cfg = CITIES[city_key]
    city_name_cn = cfg["name"]
    en_name = CN_TO_EN.get(city_name_cn, city_name_cn)
    province = CITY_PROVINCE.get(city_key)

    if city_key in DIRECT_ADMIN_CITIES:
        gdf = gpd.read_file(GADM_GPKG, layer="ADM_ADM_1")
        gdf = gdf[gdf["COUNTRY"] == "China"]
        sub = _gadm_match_en(gdf, en_name, "NAME_1")
    else:
        gdf = gpd.read_file(GADM_GPKG, layer="ADM_ADM_2")
        gdf = gdf[gdf["COUNTRY"] == "China"]
        sub = _gadm_match_en(gdf, en_name, "NAME_2", province=province)

    if sub.empty:
        print(f"  GADM: '{city_name_cn}' not found (en={en_name})")
        return None

    union = sub.dissolve().geometry.iloc[0]
    if union is None or union.is_empty:
        print(f"  GADM: empty geometry for '{city_name_cn}'")
        return None
    if union.geom_type == "MultiPolygon":
        polys = sorted(union.geoms, key=lambda p: p.area, reverse=True)
        union = polys[0]
        print(f"  GADM: kept largest of {len(polys)} sub-polygons (area={union.area:.3f} deg2)")

    bbox = union.bounds
    return {
        "geometry": union,
        "bbox": list(bbox),
        "city_key": city_key,
        "city_name": city_name_cn,
        "source": "gadm",
    }


def _gadm_match_en(gdf, en_name: str, name_col: str, province: str | None = None):
    """Match a GADM dataframe by English NAME_* column.

    When province is provided, also filters by NAME_1 to disambiguate
    cities with identical English names in different provinces (e.g.
    Suzhou/Jiangsu vs Suzhou/Anhui).
    """
    mask = gdf[name_col] == en_name
    sub = gdf[mask]

    # Province disambiguation
    if province is not None and not sub.empty and len(sub) > 1:
        prov_mask = sub["NAME_1"] == province
        if prov_mask.any():
            sub = sub[prov_mask]

    if not sub.empty:
        return sub

    # Fallback: handle apostrophe variants (Xi'an vs Xian)
    if "'" in en_name:
        fallback = en_name.replace("'", "")
        mask = gdf[name_col] == fallback
        sub = gdf[mask]
        if not sub.empty:
            return sub

    return gdf.iloc[:0]


# ── Chinese -> English name mapping for GADM NAME_2 lookup ──
# GADM stores prefecture names in NAME_2 as English (Beijing, Hangzhou, ...).
# We match by looking up our city's Chinese name -> English, then filtering
# by province where disambiguation is needed (e.g. Suzhou = Jiangsu not Anhui).

CN_TO_EN = {
    "北京": "Beijing",
    "上海": "Shanghai",
    "天津": "Tianjin",
    "重庆": "Chongqing",
    "广州": "Guangzhou",
    "深圳": "Shenzhen",
    "成都": "Chengdu",
    "武汉": "Wuhan",
    "杭州": "Hangzhou",
    "南京": "Nanjing",
    "苏州": "Suzhou",
    "西安": "Xi'an",
    "郑州": "Zhengzhou",
    "青岛": "Qingdao",
    "长沙": "Changsha",
    "大连": "Dalian",
    "沈阳": "Shenyang",
    "长春": "Changchun",
    "哈尔滨": "Harbin",
    "昆明": "Kunming",
    "南宁": "Nanning",
    "合肥": "Hefei",
    "福州": "Fuzhou",
    "厦门": "Xiamen",
    "南昌": "Nanchang",
    "济南": "Jinan",
    "太原": "Taiyuan",
    "贵阳": "Guiyang",
    "石家庄": "Shijiazhuang",
    "呼和浩特": "Hohhot",
    "乌鲁木齐": "Urumqi",
    "兰州": "Lanzhou",
    "温州": "Wenzhou",
    "无锡": "Wuxi",
    "宁波": "Ningbo",
    "常州": "Changzhou",
    "徐州": "Xuzhou",
    "佛山": "Foshan",
    "东莞": "Dongguan",
    "金华": "Jinhua",
    "绍兴": "Shaoxing",
    "台州": "Taizhou",
    "洛阳": "Luoyang",
    "南通": "Nantong",
}

# Province mapping for disambiguation (e.g. Suzhou/Jiangsu vs Suzhou/Anhui).
# Province names match GADM NAME_1 values (English).
CITY_PROVINCE = {
    "beijing": "Beijing",
    "shanghai": "Shanghai",
    "tianjin": "Tianjin",
    "chongqing": "Chongqing",
    "guangzhou": "Guangdong",
    "shenzhen": "Guangdong",
    "dongguan": "Guangdong",
    "foshan": "Guangdong",
    "hangzhou": "Zhejiang",
    "ningbo": "Zhejiang",
    "wenzhou": "Zhejiang",
    "shaoxing": "Zhejiang",
    "jinhua": "Zhejiang",
    "taizhou": "Zhejiang",
    "nanjing": "Jiangsu",
    "suzhou": "Jiangsu",
    "wuxi": "Jiangsu",
    "changzhou": "Jiangsu",
    "xuzhou": "Jiangsu",
    "nantong": "Jiangsu",
    "chengdu": "Sichuan",
    "wuhan": "Hubei",
    "changsha": "Hunan",
    "zhengzhou": "Henan",
    "luoyang": "Henan",
    "jinan": "Shandong",
    "qingdao": "Shandong",
    "shenyang": "Liaoning",
    "dalian": "Liaoning",
    "changchun": "Jilin",
    "harbin": "Heilongjiang",
    "fuzhou": "Fujian",
    "xiamen": "Fujian",
    "hefei": "Anhui",
    "nanchang": "Jiangxi",
    "kunming": "Yunnan",
    "guiyang": "Guizhou",
    "nanning": "Guangxi",
    "taiyuan": "Shanxi",
    "shijiazhuang": "Hebei",
    "hohhot": "Inner Mongolia",
    "lanzhou": "Gansu",
    "urumqi": "Xinjiang Uygur Zizhiqu",
    "xian": "Shaanxi",
}

# ── Overpass backend ─────────────────────────────────────────────


def _build_overpass_query(city_key: str) -> str:
    """Build an Overpass QL query for a city's admin boundary.

    Uses ``out geom`` so way coordinates are embedded directly in the
    response (no separate node fetch needed).  Searches by Chinese name
    with regex matching (``name~"XX"``) to catch both "XX" and "XX市",
    restricted to ``boundary=administrative`` to avoid parks/stations.
    """
    cfg = CITIES[city_key]
    name = cfg["name"]
    bbox = cfg["bbox"]
    # Overpass bbox format: (south, west, north, east)
    bbox_str = f"({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]})"

    level = "4" if city_key in DIRECT_ADMIN_CITIES else "5"

    return (
        f"[out:json][timeout:180];"
        f'relation["admin_level"="{level}"]["name"~"{name}"]'
        f'["boundary"="administrative"]{bbox_str};'
        f"out geom;"
    )


def _overpass_post(query: str, timeout: int = 180) -> dict | None:
    """Post a query to Overpass API, trying multiple mirrors with retries.

    Automatically routes through the Clash proxy at 127.0.0.1:7890 when
    detected (required on networks where Overpass returns 406 to direct
    Python requests).
    """
    proxies = get_proxies()
    last_err = None

    for url in OVERPASS_URLS:
        for attempt in range(3):
            try:
                resp = requests.post(
                    url,
                    data={"data": query},
                    headers=HEADERS,
                    proxies=proxies,
                    timeout=timeout,
                )
                if resp.status_code == 429:
                    wait = 10 * (attempt + 1)
                    last_err = f"HTTP 429 (rate limit) @ {url}"
                    print(f"      rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 504:
                    last_err = f"HTTP 504 (gateway timeout) @ {url}"
                    print("      gateway timeout, retrying...")
                    time.sleep(5)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.Timeout:
                last_err = f"timeout @ {url}"
                time.sleep(5)
            except Exception as e:
                last_err = f"{e} @ {url}"
                time.sleep(3)
        # All attempts for this mirror failed; try next mirror

    print(f"    All Overpass mirrors failed: {last_err}")
    return None


def _parse_overpass_boundary(data: dict) -> list[dict]:
    """Extract polygon geometry from an Overpass ``out geom`` response.

    With ``out geom``, each relation's way members include a ``geometry``
    field with coordinate arrays.  The outer ways are segments of the
    boundary that must be stitched into closed rings — we use
    ``shapely.ops.polygonize`` for this rather than treating each way
    as an independent ring (which fails when OSM splits a boundary into
    many chained segments).
    """
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union

    elements = data.get("elements", [])
    relations = [el for el in elements if el.get("type") == "relation"]
    if not relations:
        return []

    results = []
    for rel in relations:
        tags = rel.get("tags", {})
        members = rel.get("members", [])

        # Collect outer way segments as LineStrings
        lines = []
        for m in members:
            if m.get("role") == "outer" and m.get("type") == "way":
                geom_pts = m.get("geometry", [])
                if len(geom_pts) >= 2:
                    coords = [(pt["lon"], pt["lat"]) for pt in geom_pts]
                    lines.append(LineString(coords))

        if not lines:
            continue

        # polygonize automatically stitches connected segments into
        # closed polygons, handling the case where OSM splits a boundary
        # into many ways that must be chained together.
        try:
            polygons = list(polygonize(lines))
            if not polygons:
                continue
            union = unary_union(polygons)
            if union.geom_type == "MultiPolygon":
                union = sorted(union.geoms, key=lambda p: p.area, reverse=True)[0]
            b = union.bounds
            results.append(
                {
                    "geometry": union,
                    "bbox": list(b),
                    "osm_id": rel.get("id"),
                    "source": "overpass",
                    "admin_level": tags.get("admin_level", ""),
                    "name": tags.get("name", ""),
                }
            )
        except Exception as e:
            print(f"    [WARN] failed to build polygon for relation {rel.get('id')}: {e}")
            continue
    return results


def _fetch_overpass(city_key: str) -> dict | None:
    """Fetch a single city's boundary via Overpass API."""
    cfg = CITIES[city_key]
    name = cfg["name"]

    query = _build_overpass_query(city_key)
    print(f"  OSM query: admin boundaries for {name} ...")

    data = _overpass_post(query, timeout=180)
    if data is None:
        return None

    n_elements = len(data.get("elements", []))
    results = _parse_overpass_boundary(data)
    if not results:
        print(f"  OSM: no boundary parsed ({n_elements} elements returned)")
        return None

    # Pick the largest polygon by area
    best = max(results, key=lambda r: r["geometry"].area)
    best["city_key"] = city_key
    best["city_name"] = name
    geom = best["geometry"]
    print(
        f"  OSM: boundary found (OSM id={best.get('osm_id', '')}, "
        f"{len(results)} candidates, picked largest area={geom.area:.2f} deg2)"
    )
    return best


# ── Unified fetch + caching ──────────────────────────────────────


def fetch_city(city_key: str, backend: str = "auto") -> dict | None:
    """Fetch boundary for one city. Returns dict with geometry + bbox or None.

    Caches results to disk so re-runs are instant.
    """
    cache_geojson = BOUNDARY_DIR / f"{city_key}_boundary.geojson"
    cache_bbox = BOUNDARY_DIR / f"{city_key}_bbox.json"

    # Return cached if exists
    if cache_geojson.exists() and cache_bbox.exists():
        from shapely.geometry import shape

        with open(cache_bbox, encoding="utf-8") as f:
            info = json.load(f)
        with open(cache_geojson, encoding="utf-8") as f:
            geojson = json.load(f)
        info["geometry"] = shape(geojson)
        return info

    result = None

    if backend in ("gadm", "auto"):
        result = _fetch_gadm(city_key)

    if result is None and backend in ("overpass", "auto"):
        result = _fetch_overpass(city_key)

    if result is None:
        return None

    # Cache to disk
    from shapely.geometry import mapping

    geojson = mapping(result["geometry"])
    with open(cache_geojson, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

    bbox_info = {k: v for k, v in result.items() if k != "geometry"}
    with open(cache_bbox, "w", encoding="utf-8") as f:
        json.dump(bbox_info, f, ensure_ascii=False, default=str)

    return result


def get_buffered_bbox(
    city_key: str,
    buffer_km: float = 10.0,
    backend: str = "auto",
    existing_result: dict | None = None,
) -> list[float] | None:
    """Return bbox = boundary.bounds + buffer_km.

    If ``existing_result`` is provided (already fetched boundary dict),
    use it directly instead of re-fetching from cache/network.
    """
    import math

    result = existing_result or fetch_city(city_key, backend=backend)
    if result is None:
        return None

    minx, miny, maxx, maxy = result["bbox"]
    # Latitude-corrected buffer (matches pipeline_config.get_effective_bbox)
    lat_c = (miny + maxy) / 2.0
    buf_lat_deg = buffer_km / 111.0
    cos_lat = max(0.1, math.cos(math.radians(lat_c)))
    buf_lon_deg = buffer_km / (111.0 * cos_lat)
    return [
        round(minx - buf_lon_deg, 4),
        round(miny - buf_lat_deg, 4),
        round(maxx + buf_lon_deg, 4),
        round(maxy + buf_lat_deg, 4),
    ]


# ── Config regeneration ──────────────────────────────────────────


def update_pipeline_config_bbox(
    backend: str = "auto", buffer_km: float = 10.0
) -> dict[str, list[float]]:
    """Regenerate CITIES bbox values from fetched admin boundaries.

    Returns {city_key: [lon_min, lat_min, lon_max, lat_max]} for cities
    where a boundary was successfully fetched.
    """
    updated = {}
    for ck in ACTIVE_CITIES:
        result = fetch_city(ck, backend=backend)
        bbox = get_buffered_bbox(ck, buffer_km=buffer_km, backend=backend, existing_result=result)
        if bbox:
            updated[ck] = [round(v, 2) for v in bbox]
            print(f"  {ck:14s}: {updated[ck]}")
        else:
            old = CITIES[ck]["bbox"]
            updated[ck] = old
            print(f"  {ck:14s}: [fallback] {old}")

    if updated:
        out_path = BOUNDARY_DIR / "cities_bbox_from_boundaries.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(updated, f, ensure_ascii=False, indent=2)
        print(f"\nSaved bbox map -> {out_path}")
        print("To apply, copy values into pipeline_config.py CITIES bbox fields.")
    return updated


# ── CLI ──────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Fetch admin boundaries for 44 Chinese cities")
    p.add_argument("--city", default="all", help="City key or 'all'")
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "gadm", "overpass"],
        help="Boundary source backend",
    )
    p.add_argument(
        "--buffer-km", type=float, default=10.0, help="Buffer around boundary for bbox (km)"
    )
    p.add_argument(
        "--download-gadm", action="store_true", help="Download GADM China GeoPackage (~12 MB)"
    )
    p.add_argument(
        "--update-config", action="store_true", help="Regenerate bbox values from boundaries"
    )
    p.add_argument("--show", action="store_true", help="Print boundaries info for inspection")
    p.add_argument(
        "--proxy",
        default=None,
        help="Proxy URL (e.g. http://127.0.0.1:7890). Auto-detected if omitted.",
    )
    args = p.parse_args()

    # Override proxy if specified via CLI
    if args.proxy:
        set_proxy(args.proxy)
        print(f"  [proxy] Using proxy: {args.proxy}")

    if args.download_gadm:
        ok = _download_gadm()
        if not ok:
            return 1
        if not args.city or (args.city == "all" and not args.update_config):
            return 0

    if args.update_config:
        update_pipeline_config_bbox(backend=args.backend, buffer_km=args.buffer_km)
        return 0

    cities = ACTIVE_CITIES if args.city == "all" else [args.city]

    ok, fail = 0, 0
    for ck in cities:
        if ck not in CITIES:
            continue
        name = CITIES[ck]["name"]
        print(f"\n{'=' * 50}\n{name} ({ck})\n{'=' * 50}")

        result = fetch_city(ck, backend=args.backend)
        if result is None:
            print("  [FAIL] No boundary found")
            fail += 1
            continue

        bbox = result["bbox"]
        buf_bbox = get_buffered_bbox(
            ck, buffer_km=args.buffer_km, backend=args.backend, existing_result=result
        )
        src = result.get("source", "?")
        print(f"  source: {src}")
        print(f"  bbox (raw):      [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")
        if buf_bbox:
            print(
                f"  bbox (+{args.buffer_km}km): [{buf_bbox[0]:.4f}, {buf_bbox[1]:.4f}, "
                f"{buf_bbox[2]:.4f}, {buf_bbox[3]:.4f}]"
            )
        # Compare with hardcoded
        old = CITIES[ck]["bbox"]
        print(f"  old bbox:        [{old[0]}, {old[1]}, {old[2]}, {old[3]}]")

        if args.show:
            try:
                from shapely.geometry import mapping

                geo = mapping(result["geometry"])
                print(f"  geometry type: {geo['type']}")
            except Exception:
                pass
        ok += 1

        # Rate limit: pause between cities to avoid Overpass 429
        if src == "overpass":
            time.sleep(3)

    print(f"\n{'=' * 50}")
    print(f"Done: {ok} ok, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
