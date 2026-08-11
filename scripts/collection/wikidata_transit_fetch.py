"""Wikidata Metro Station Fetcher (SPARQL, no API key).

Queries Wikidata for all Chinese metro/rail stations with structured fields:
  - coordinate (P625, WGS84)
  - opening date (P1619)
  - connecting line (P81)
  - located in admin unit (P131) -> mapped to our 44 cities by Chinese label

A single SPARQL query covers all 44 cities; results are split per city and
saved alongside the OSM/Amap outputs for cross-source comparison.

Usage:
    python scripts/collection/wikidata_transit_fetch.py
    python scripts/collection/wikidata_transit_fetch.py --city beijing
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
import pandas as pd
import requests
from shapely.geometry import Point
from shapely.geometry import box as shapely_box

from urban_intervention.config.project import ACTIVE_CITIES, CITIES
from urban_intervention.config.project import norm_station_name as _norm

WD_SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "MIT-Summer-Research/1.0 (wikidata metro station collection)",
    "Accept": "application/sparql-results+json",
}

# Items in China with a coordinate + a connecting line. Broad on purpose so we
# can client-side filter to metro using the line's label / station's location.
# wdt:P17 = country, wdt:P625 = coordinate location, wdt:P81 = connecting line.
# LIMIT 8000 is generous (China has ~5000 metro stations as of 2025); if the
# count ever approaches the limit, the fetcher prints a truncation warning so
# results aren't silently lost.
# The SPARQL endpoint times out on large paginated OFFSET queries (504
# Gateway Timeout after offset=4000).  A single query with LIMIT 16000
# successfully returns all Chinese metro stations (~8k after metro-line
# filter, out of ~16k raw bindings including non-metro rail).  The
# previous truncation warning was a false alarm — 16000 is well above the
# true total.  If the raw binding count ever approaches this limit, run
# Keep pagination self-contained; do not depend on a user-local helper script.
# and pass --from-file.
PAGE_SIZE = 16000
MAX_TOTAL = 64000
SPARQL_TEMPLATE = """
SELECT ?station ?stationLabel ?geo ?opening ?line ?lineLabel ?city ?cityLabel ?adj ?adjLabel WHERE {
  ?station wdt:P17 wd:Q148 ;
           wdt:P625 ?geo ;
           wdt:P81 ?line .
  OPTIONAL { ?station wdt:P1619 ?opening . }
  OPTIONAL { ?station wdt:P131 ?city . }
  OPTIONAL { ?station wdt:P197 ?adj . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "zh,en". }
}
LIMIT %d
OFFSET %d
"""

# Kept for backwards compatibility with tests that import SPARQL / SPARQL_LIMIT.
SPARQL_LIMIT = PAGE_SIZE
SPARQL = SPARQL_TEMPLATE % (PAGE_SIZE, 0)


def _parse_point(point_str: str) -> tuple[float, float] | tuple[None, None]:
    """Wikidata Point: 'Point(116.391 39.907)' -> (lon, lat)."""
    m = re.match(r"Point\((-?\d+\.?\d*)\s+(-?\d+\.?\d*)\)", str(point_str))
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def _year_from_date(date_str: str) -> int | None:
    m = re.search(r"(\d{4})-\d{2}-\d{2}", str(date_str))
    return int(m.group(1)) if m else None


def _parse_opening_date(date_str: str) -> dict:
    """Parse a Wikidata P1619 date into year, month, day and precision.

    Wikidata dates use ISO 8601 with ``-00`` placeholders for unknown
    components:

    - Day precision:   ``"1969-10-01T00:00:00Z"``  → year=1969, month=10, day=1
    - Month precision: ``"1969-10-00T00:00:00Z"``  → year=1969, month=10, day=None
    - Year precision:  ``"1969-00-00T00:00:00Z"``  → year=1969, month=None, day=None

    Returns a dict with keys ``opening_date`` (raw string, or ``""`` if
    missing), ``opening_year``, ``opening_month``, ``opening_day`` and
    ``date_precision`` (``"day"``, ``"month"``, ``"year"`` or ``""``).
    """
    s = str(date_str).strip()
    if not s:
        return {
            "opening_date": "",
            "opening_year": None,
            "opening_month": None,
            "opening_day": None,
            "date_precision": "",
        }
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if not m:
        return {
            "opening_date": s,
            "opening_year": None,
            "opening_month": None,
            "opening_day": None,
            "date_precision": "",
        }
    year = int(m.group(1))
    month_raw = m.group(2)
    day_raw = m.group(3)
    # Normalise the raw ISO date to a clean YYYY-MM-DD (drop time component).
    clean_date = f"{m.group(1)}-{month_raw}-{day_raw}"
    if month_raw == "00":
        return {
            "opening_date": clean_date,
            "opening_year": year,
            "opening_month": None,
            "opening_day": None,
            "date_precision": "year",
        }
    if day_raw == "00":
        return {
            "opening_date": clean_date,
            "opening_year": year,
            "opening_month": int(month_raw),
            "opening_day": None,
            "date_precision": "month",
        }
    return {
        "opening_date": clean_date,
        "opening_year": year,
        "opening_month": int(month_raw),
        "opening_day": int(day_raw),
        "date_precision": "day",
    }


# Regex patterns for classifying rail lines as metro vs non-metro.
# A line is metro if it matches METRO_PATTERN and does NOT match
# EXCLUDE_PATTERN (高铁/国铁/动车/城际铁路 — note 城际铁路 is excluded but
# 市域铁路 is kept, since Chinese 市域 rail systems like Wenzhou S1 / Taizhou S1
# are operationally metro-like and included in our 44-city coverage).
METRO_PATTERN = re.compile(r"\d+号线|\d+线|号线|地铁|轨道交通|轻轨|磁浮|磁悬浮|有轨|市域")
EXCLUDE_PATTERN = re.compile(r"高铁|动车|城际铁路|普速|客专|国铁|干线")


def _filter_metro_lines(line_str: str) -> str:
    """Given a (possibly multi-) line string, return only the metro lines
    joined by ';'.  Returns '' if no metro lines remain.

    A single Wikidata row's `line` field may contain multiple lines separated
    by ';' (some Wikidata stations have one row per station with all lines
    concatenated).  We split, filter each, and rejoin so a station with both
    a metro line and a mainline rail line keeps the metro line and drops the
    mainline — instead of dropping the entire station.
    """
    parts = [p.strip() for p in str(line_str).split(";") if p.strip()]
    kept = []
    for p in parts:
        if EXCLUDE_PATTERN.search(p):
            continue
        if METRO_PATTERN.search(p):
            kept.append(p)
    return ";".join(sorted(set(kept)))


def _query_sparql_page(proxy: str | None, offset: int, limit: int) -> list[dict]:
    """Query one page of the SPARQL endpoint. Returns a list of bindings.

    Handles control-char cleaning and timeout retries.  Returns an empty
    list on persistent failure.
    """
    if proxy:
        proxies = {"http": proxy, "https": proxy}
    else:
        from urban_intervention.config.project import get_proxies

        proxies = get_proxies()
    query = SPARQL_TEMPLATE % (limit, offset)
    for attempt in range(3):
        try:
            resp = requests.get(
                WD_SPARQL, params={"query": query}, headers=HEADERS, timeout=180, proxies=proxies
            )
            resp.raise_for_status()
            # Wikidata occasionally returns JSON with unescaped control
            # characters (e.g. \x0b in station labels).  Strip them before
            # parsing so json.loads does not raise.
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", resp.text)
            data = json.loads(text, strict=False)
            bindings = data.get("results", {}).get("bindings", [])
            if not isinstance(bindings, list):
                print(
                    f"    [WARN] offset={offset}: malformed response (bindings "
                    f"not a list). Endpoint may have timed out."
                )
                return []
            return bindings
        except Exception as e:
            print(f"    offset={offset} attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    print(f"    offset={offset}: all attempts failed")
    return []


def fetch_all(proxy: str | None = None, from_file: str | None = None) -> pd.DataFrame:
    """Run SPARQL in pages; return a flat DataFrame of all Chinese metro-ish stations.

    Proxy resolution order:
      1. Explicit ``proxy`` argument (e.g. http://127.0.0.1:7890)
      2. Auto-detected Clash proxy (via pipeline_config.get_proxies)
      3. HTTPS_PROXY/HTTP_PROXY env var (via requests defaults)

    If from_file is given, load the SPARQL JSON response from that file instead
    of querying the endpoint (useful behind the GFW — see wikidata_query_url.py).
    """
    if from_file:
        from pathlib import Path

        p = Path(from_file)
        if not p.is_absolute():
            p = BASE_DIR / from_file
        if not p.exists():
            print(f"  [!] File not found: {p}")
            return pd.DataFrame()
        print(f"  Loading SPARQL response from {p} ...")
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [!] Failed to parse JSON: {e}")
            return pd.DataFrame()
        all_bindings = data.get("results", {}).get("bindings", [])
        if not isinstance(all_bindings, list):
            all_bindings = []
    else:
        from urban_intervention.config.project import get_proxy

        detected = proxy or get_proxy()
        print(
            f"  Querying Wikidata SPARQL endpoint (paginated, "
            f"page={PAGE_SIZE})..."
            f"{' via proxy ' + detected if detected else ' (direct)'}"
        )
        all_bindings = []
        offset = 0
        while offset < MAX_TOTAL:
            batch = _query_sparql_page(proxy, offset, PAGE_SIZE)
            if not batch:
                break
            all_bindings.extend(batch)
            print(f"    offset={offset}: got {len(batch)} bindings (total: {len(all_bindings)})")
            if len(batch) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
            time.sleep(2)
        if offset >= MAX_TOTAL and len(all_bindings) >= MAX_TOTAL:
            print(
                f"  [WARN] hit MAX_TOTAL={MAX_TOTAL}; results may still be "
                f"truncated. Raise MAX_TOTAL in this file and re-run."
            )

    bindings = all_bindings
    print(f"  Wikidata returned {len(bindings)} raw bindings")

    rows = []
    adj_rows = []
    for b in bindings:
        geo = b.get("geo", {}).get("value", "")
        lon, lat = _parse_point(geo)
        if lon is None:
            continue
        parsed = _parse_opening_date(b.get("opening", {}).get("value", ""))
        station_wd = b.get("station", {}).get("value", "").rsplit("/", 1)[-1]
        adj = b.get("adj", {}).get("value", "").rsplit("/", 1)[-1]
        adj_name = b.get("adjLabel", {}).get("value", "")
        rows.append(
            {
                "station_name": b.get("stationLabel", {}).get("value", ""),
                "name_en": "",  # could parse from labels but keep simple
                "wgs84_lon": round(lon, 7),
                "wgs84_lat": round(lat, 7),
                "opening_year": parsed["opening_year"],
                "opening_month": parsed["opening_month"],
                "opening_day": parsed["opening_day"],
                "opening_date": parsed["opening_date"],
                "date_precision": parsed["date_precision"],
                "line": b.get("lineLabel", {}).get("value", ""),
                "city_wd": b.get("city", {}).get("value", "").rsplit("/", 1)[-1],
                "city_label": b.get("cityLabel", {}).get("value", ""),
                "station_entity_id": station_wd,
            }
        )
        if adj:
            adj_rows.append(
                {
                    "station_entity_id": station_wd,
                    "station_name": b.get("stationLabel", {}).get("value", ""),
                    "line_label": b.get("lineLabel", {}).get("value", ""),
                    "adj_station_entity_id": adj,
                    "adj_station_name": adj_name,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep only rows whose city_label matches one of our 44 cities (by Chinese name).
    # Match on full name OR stripped-suffix variants (北京市 -> 北京).
    city_name_to_key = {v["name"]: k for k, v in CITIES.items()}

    def _map_city(label: str) -> str:
        if not label:
            return ""
        if label in city_name_to_key:
            return city_name_to_key[label]
        for suffix in ("市", "城区", "市区", "市辖区"):
            bare = label.removesuffix(suffix)
            if bare in city_name_to_key:
                return city_name_to_key[bare]
        return ""

    df["city_key"] = df["city_label"].apply(_map_city)

    # Also keep stations that fall inside a city's bbox even if P131 didn't map,
    # so we don't lose stations in suburbs / districts not labeled as the city.
    # When bboxes overlap (e.g. Dongguan/Shenzhen), assign to the city whose
    # bbox *centroid* is closest — this is more geographically meaningful than
    # the previous "first match in dict iteration order" behavior.
    city_boxes = {}
    city_centroids = {}
    for ck, cfg in CITIES.items():
        bb = cfg["bbox"]
        city_boxes[ck] = shapely_box(bb[0], bb[1], bb[2], bb[3])
        city_centroids[ck] = (
            cfg.get("center_lon", (bb[0] + bb[2]) / 2),
            cfg.get("center_lat", (bb[1] + bb[3]) / 2),
        )

    def _bbox_city(row) -> str:
        if row["city_key"]:
            return row["city_key"]
        pt = Point(row["wgs84_lon"], row["wgs84_lat"])
        candidates = [
            ck for ck, geom in city_boxes.items() if geom.contains(pt) or geom.touches(pt)
        ]
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]
        # Overlap: pick the city whose centroid is nearest.
        lon, lat = row["wgs84_lon"], row["wgs84_lat"]
        return min(
            candidates,
            key=lambda ck: (city_centroids[ck][0] - lon) ** 2 + (city_centroids[ck][1] - lat) ** 2,
        )

    df["city_key"] = df.apply(_bbox_city, axis=1)

    # Drop stations we couldn't place in any of the 44 cities.
    df = df[df["city_key"] != ""].reset_index(drop=True)

    # Filter lines to metro-only.  Each row's `line` field may contain
    # multiple lines separated by ';'.  We keep only the metro lines per
    # row (dropping 高铁/国铁/动车/城际铁路 etc.) and then drop rows that
    # have no metro lines left.  This preserves stations that have BOTH a
    # metro line and a mainline rail line — previously the whole station
    # was dropped if its single concatenated line string happened to also
    # match a non-metro keyword.
    n_before = len(df)
    df["line"] = df["line"].apply(_filter_metro_lines)
    df = df[df["line"] != ""].reset_index(drop=True)
    print(
        f"  Mapped to 44 cities: {n_before} rows ({df['city_key'].nunique()} cities); "
        f"kept {len(df)} after metro-line filter "
        f"(dropped {n_before - len(df)} non-metro)"
    )

    # Persist station adjacency (P197) per city alongside the station CSVs.
    if adj_rows:
        adj_df = pd.DataFrame(adj_rows).drop_duplicates()
        station_city = df[["station_entity_id", "city_key"]].drop_duplicates()
        adj_df = adj_df.merge(station_city, on="station_entity_id", how="inner")
        out_adj = BASE_DIR / "data" / "active" / "reference" / "transit"
        out_adj.mkdir(parents=True, exist_ok=True)
        adj_df.to_parquet(out_adj / "wikidata_adjacency.parquet", index=False)
        print(
            f"  Adjacency (P197): {len(adj_df)} edges saved -> "
            f"{out_adj / 'wikidata_adjacency.parquet'}"
        )

    return df


def split_and_save(df: pd.DataFrame, only_city: str | None = None) -> int:
    """Split the global DataFrame per city and save CSVs."""
    if df.empty:
        return 0
    cities = [only_city] if only_city else ACTIVE_CITIES
    total = 0
    for ck in cities:
        sub = df[df["city_key"] == ck].copy()
        if sub.empty:
            continue
        # Collapse multiple lines per station into one row.
        # Group by (normalised name, ~100m coord cluster) so that the same
        # station appearing in multiple line rows collapses to one record.
        sub["_n"] = sub["station_name"].apply(_norm)
        sub["_clat"] = sub["wgs84_lat"].round(3)
        sub["_clon"] = sub["wgs84_lon"].round(3)

        # Select the single best date record per group: finest precision, then
        # month/day availability, then earliest year.  All date components come
        # from the same source row.
        base = sub.groupby(["_n", "_clat", "_clon"], as_index=False).agg(
            {
                "station_name": "first",
                "name_en": "first",
                "wgs84_lon": "first",
                "wgs84_lat": "first",
                "line": lambda s: ";".join(sorted({x for y in s for x in str(y).split(";") if x})),
            }
        )

        def _pick_best_date(group: pd.DataFrame) -> pd.Series:
            rank_map = {"day": 3, "month": 2, "year": 1}
            g = group.copy()
            g["_r"] = g["date_precision"].map(rank_map).fillna(0)
            g["_m"] = g["opening_month"].notna().astype(int)
            g = g.sort_values(["_r", "_m", "opening_year"], ascending=[False, False, True])
            b = g.iloc[0]
            return pd.Series(
                {
                    "opening_year": float(b["opening_year"])
                    if pd.notna(b["opening_year"])
                    else float("nan"),
                    "opening_month": int(b["opening_month"])
                    if pd.notna(b["opening_month"])
                    else None,
                    "opening_day": int(b["opening_day"]) if pd.notna(b["opening_day"]) else None,
                    "opening_date": str(b["opening_date"])
                    if pd.notna(b.get("opening_date")) and b["opening_date"] != ""
                    else "",
                    "date_precision": str(b["date_precision"])
                    if pd.notna(b["date_precision"])
                    else "",
                }
            )

        date_cols = sub.groupby(["_n", "_clat", "_clon"]).apply(_pick_best_date).reset_index()
        date_cols = date_cols.drop(columns=["level_2"], errors="ignore")

        agg = base.merge(date_cols, on=["_n", "_clat", "_clon"], how="left")
        agg = agg.drop(columns=["_n", "_clat", "_clon"]).reset_index(drop=True)

        out = BASE_DIR / "data" / "archive" / "raw" / "transit" / "wikidata" / ck
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{ck}_metro_stations_wikidata.csv"
        agg.to_csv(path, index=False, encoding="utf-8-sig")
        n_year = agg["opening_year"].notna().sum()
        n_month = agg["opening_month"].notna().sum()
        n_day = agg["opening_day"].notna().sum()
        print(
            f"  [{ck}] {len(agg)} stations, {n_year} with year, "
            f"{n_month} with month, {n_day} with day -> {path}"
        )
        total += len(agg)
    return total


def main():
    parser = argparse.ArgumentParser(description="Wikidata metro station fetcher")
    parser.add_argument(
        "--city", default="all", help="City key or 'all' (split global query per city)"
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="Proxy URL, e.g. http://127.0.0.1:7890 (also reads HTTPS_PROXY env var)",
    )
    parser.add_argument(
        "--from-file",
        default=None,
        help="Load SPARQL JSON response from a local file instead of "
        "querying the endpoint (useful behind GFW — see "
        "wikidata_query_url.py)",
    )
    args = parser.parse_args()
    only = None if args.city == "all" else args.city

    df = fetch_all(proxy=args.proxy, from_file=args.from_file)
    if df.empty:
        print("  [!] No data")
        return 1
    total = split_and_save(df, only_city=only)
    print(f"\nDone. Total {total} stations saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
