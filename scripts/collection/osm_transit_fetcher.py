"""OSM Metro Station Fetcher — route relation based (v2).

Queries route=subway / route=light_rail relations, then recursively fetches
all member nodes. This gives station order, line membership, and transfer
detection, which the v1 node-only query missed.

Outputs per city:
  {city}_metro_stations_osm.csv   — stations with line(s) from relations
  {city}_metro_lines_osm.csv      — line metadata (name, ref, colour, operator)

Usage:
    python scripts/collection/osm_transit_fetcher.py --city beijing
    python scripts/collection/osm_transit_fetcher.py --city all
"""

import argparse
import re
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
import pandas as pd
import requests

from urban_intervention.config.project import ACTIVE_CITIES, CITIES, get_proxies
from urban_intervention.config.project import norm_station_name as _norm

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
HEADERS = {
    "User-Agent": "MIT-Summer-Research/1.0 (metro station collection)",
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept": "application/json",
}

# Query relations first, then recurse all members. `out body` on the recursed
# set keeps tags+coords on member nodes (out skel would drop tags, leaving
# stations unnamed).
QUERY_TEMPLATE = """
[out:json][timeout:180];
(
  relation["route"="subway"]({s},{w},{n},{e});
  relation["route"="light_rail"]({s},{w},{n},{e});
  relation["route"="train"]["operator"~"地铁|metro|轨道交通|Metro"]({s},{w},{n},{e});
);
out body; >; out body qt;
"""


def _overpass(query: str, timeout: int = 180) -> dict | None:
    """Try multiple Overpass mirrors with proxy auto-detection and retries."""
    proxies = get_proxies()
    last_err = None
    for url in OVERPASS_URLS:
        for attempt in range(3):
            try:
                resp = requests.post(
                    url, data={"data": query}, headers=HEADERS, proxies=proxies, timeout=timeout
                )
                if resp.status_code == 429:
                    wait = 10 * (attempt + 1)
                    last_err = f"HTTP 429 @ {url}"
                    time.sleep(wait)
                    continue
                if resp.status_code == 504:
                    last_err = f"HTTP 504 @ {url}"
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
    print(f"    All Overpass mirrors failed: {last_err}")
    return None


def _extract_year(tag_dict: dict) -> int | None:
    for key in ("start_date", "opening_date", "opened"):
        val = tag_dict.get(key, "")
        m = re.search(r"(19|20)\d{2}", str(val))
        if m:
            return int(m.group(0))
    return None


def _extract_names(tag_dict: dict) -> tuple:
    name = tag_dict.get("name", "")
    name_zh = tag_dict.get("name:zh", "") or tag_dict.get("name:zh-Hans", "") or name
    name_en = tag_dict.get("name:en", "") or ""
    return name, name_zh, name_en


def _norm_name(name: str) -> str:
    """Backward-compatible alias for the shared norm_station_name."""
    return _norm(name)


def _line_name(rel_tags: dict) -> str:
    """Pick a clean line identifier from relation tags.

    OSM subway relations often have names like '北京 14号线: 张郭庄 -> 善各庄'
    (line + endpoints) and come in forward/backward variants. We strip the
    route-endpoint suffix and the city prefix so the two variants collapse to
    the same line id (e.g. '14号线').
    """
    # Prefer the cleaned name tag (most consistent for CN subway), fall back to ref
    raw = ""
    for key in ("name", "name:zh"):
        v = rel_tags.get(key, "")
        if v:
            raw = str(v)
            break
    if not raw:
        for key in ("ref", "short_name", "name:en"):
            v = rel_tags.get(key, "")
            if v:
                return str(v).strip()
    if not raw:
        return ""
    # Strip route suffixes: ': ...', ' -> ...', ' → ...', ' - ...'
    for sep in ["：", ":", " -> ", " → ", " — ", " - "]:
        if sep in raw:
            raw = raw.split(sep, 1)[0]
    # Strip a leading city prefix like '北京 ' / '上海 ' (with or without space)
    raw = re.sub(
        r"^(北京|上海|广州|深圳|成都|重庆|杭州|南京|武汉|天津|西安|苏州|郑州|青岛|长沙|大连|沈阳|哈尔滨|长春|昆明|南宁|合肥|福州|厦门|南昌|济南|太原|贵阳|石家庄|呼和浩特|乌鲁木齐|兰州|温州|无锡|宁波|金华|南通|徐州|常州|台州|绍兴|洛阳|佛山|东莞|香港)[\s　]*",
        "",
        raw,
    )
    # Strip English airport-express style suffixes
    raw = re.sub(r"\s+\(.*$", "", raw)
    return raw.strip()


def _clean_line_id(name: str) -> str:
    """Normalized key for grouping forward/backward variants of the same line."""
    s = str(name).strip()
    # Drop everything that isn't a digit/letter so '14号线' and '14线' match loosely
    m = re.search(r"\d+", s)
    if m:
        return f"line_{m.group(0)}"
    return s


def parse_overpass_response(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split elements into stations (nodes) and lines (relations).

    Returns (stations_df, lines_df). Stations get a 'line' column built from
    the set of relations that reference them as a stop/platform member.
    """
    elements = data.get("elements", [])
    nodes = [el for el in elements if el.get("type") == "node"]
    relations = [el for el in elements if el.get("type") == "relation"]

    # node_id -> node record
    node_map = {}
    for el in nodes:
        tags = el.get("tags", {}) or {}
        _, name_zh, name_en = _extract_names(tags)
        display = name_zh or name_en or tags.get("name", "")
        if not display:
            continue
        # Defensive: use .get() for id/lon/lat — OSM elements normally have
        # these, but a malformed response shouldn't crash the whole pipeline.
        nid = el.get("id")
        lon = el.get("lon")
        lat = el.get("lat")
        if nid is None or lon is None or lat is None:
            continue
        node_map[nid] = {
            "station_name": display,
            "name_en": name_en if name_en and name_en != display else "",
            "wgs84_lon": round(float(lon), 7),
            "wgs84_lat": round(float(lat), 7),
            "opening_year": _extract_year(tags),
            "_node_id": nid,
        }

    # relation_id -> line metadata + member node ids
    # Group forward/backward variants by cleaned line id so they don't double-count
    lines_records = []
    node_to_lines: dict[int, set[str]] = {}
    grouped: dict[str, dict] = {}  # clean_id -> {display_name, tags, member_nodes}
    for rel in relations:
        tags = rel.get("tags", {}) or {}
        lname = _line_name(tags)
        cid = _clean_line_id(lname)
        members = [
            m
            for m in rel.get("members", [])
            if m.get("type") == "node"
            and m.get("role") in ("stop", "platform", "", "platform_entry_only")
        ]
        member_refs = [m.get("ref") for m in members]
        if cid not in grouped:
            grouped[cid] = {
                "display_name": lname,
                "tags": tags,
                "member_nodes": list(member_refs),
                "n_relations": 1,
            }
        else:
            grouped[cid]["member_nodes"].extend(member_refs)
            grouped[cid]["n_relations"] += 1
            # keep richer tags (one with start_date) for the grouped line
            if tags.get("start_date") and not grouped[cid]["tags"].get("start_date"):
                grouped[cid]["tags"] = tags

    for _cid, g in grouped.items():
        tags = g["tags"]
        lines_records.append(
            {
                "line_name": g["display_name"],
                "ref": tags.get("ref", ""),
                "colour": tags.get("colour", ""),
                "operator": tags.get("operator", ""),
                "network": tags.get("network", ""),
                "start_date": tags.get("start_date", ""),
                "opening_year": _extract_year(tags),
                "route": tags.get("route", ""),
                "n_variants": g["n_relations"],
                "n_stations": len(set(n for n in g["member_nodes"] if n is not None)),
            }
        )
        display = g["display_name"]
        for nid in g["member_nodes"]:
            if nid is not None and display:
                node_to_lines.setdefault(nid, set()).add(display)

    # Attach lines to stations (dedup, preserve sorted order)
    stations = []
    for nid, rec in node_map.items():
        lines = node_to_lines.get(nid, set())
        ordered = sorted(lines)
        rec["line"] = ";".join(ordered)
        rec["n_lines"] = len(ordered)
        stations.append(rec)

    stations_df = pd.DataFrame(stations)
    if not stations_df.empty:
        # Metro station names are unique within a city, so collapse by normalized
        # name. This merges OSM's separate stop/platform/entrance nodes for the
        # same station (which sit 100-400m apart and otherwise inflate counts).
        # Coordinates become the centroid of all matching nodes.
        stations_df["_n"] = stations_df["station_name"].apply(_norm_name)
        before = len(stations_df)
        # Aggregate coordinates/year via groupby + agg, and lines via a
        # SEPARATE groupby + apply then merge on _n.  Previously we relied
        # on positional alignment between two groupbys (both assumed to
        # sort identically), which is fragile and silently wrong if pandas
        # ever changes its sort default.  An explicit merge is robust.
        agg_df = stations_df.groupby("_n", as_index=False).agg(
            {
                "station_name": "first",
                "name_en": "first",
                "wgs84_lon": "mean",
                "wgs84_lat": "mean",
                "opening_year": "min",
            }
        )
        line_df = (
            stations_df.groupby("_n")["line"]
            .apply(lambda s: ";".join(sorted({x for y in s for x in str(y).split(";") if x})))
            .reset_index(name="line")
        )
        stations_df = agg_df.merge(line_df, on="_n", how="left")
        stations_df["n_lines"] = (
            stations_df["line"].fillna("").apply(lambda s: len([x for x in str(s).split(";") if x]))
        )
        stations_df["wgs84_lon"] = stations_df["wgs84_lon"].round(7)
        stations_df["wgs84_lat"] = stations_df["wgs84_lat"].round(7)
        stations_df = stations_df.drop(columns=["_n"]).reset_index(drop=True)
        if before > len(stations_df):
            print(f"    Dedup by name: {before} -> {len(stations_df)} unique")

    lines_df = pd.DataFrame(lines_records)
    return stations_df, lines_df


def fetch_city(city_key: str, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    name = cfg["name"]
    s, w, n, e = cfg["bbox"][1], cfg["bbox"][0], cfg["bbox"][3], cfg["bbox"][2]
    query = QUERY_TEMPLATE.format(s=s, w=w, n=n, e=e)
    print(
        f"  Querying OSM route relations for {name} (bbox: {w:.2f},{s:.2f} to {e:.2f},{n:.2f})..."
    )

    data = _overpass(query)
    if data is None:
        return pd.DataFrame(), pd.DataFrame()

    n_rel = sum(1 for el in data.get("elements", []) if el.get("type") == "relation")
    n_node = sum(1 for el in data.get("elements", []) if el.get("type") == "node")
    print(f"  OSM returned {n_rel} relations, {n_node} nodes")

    stations_df, lines_df = parse_overpass_response(data)
    n_year = stations_df["opening_year"].notna().sum() if not stations_df.empty else 0
    n_transfer = (stations_df["n_lines"] >= 2).sum() if not stations_df.empty else 0
    print(
        f"  Result: {len(stations_df)} stations, {n_year} with year, "
        f"{n_transfer} transfer stations, {len(lines_df)} lines"
    )
    return stations_df, lines_df


def main():
    parser = argparse.ArgumentParser(description="OSM metro station fetcher (route relations)")
    parser.add_argument("--city", default="all")
    args = parser.parse_args()

    cities = []
    for c in args.city.split(","):
        c = c.strip()
        if c == "all":
            cities = ACTIVE_CITIES
            break
        if c in CITIES:
            cities.append(c)

    total = 0
    for ck in cities:
        cfg = CITIES[ck]
        print(f"\n{'=' * 50}\n{cfg['name']}\n{'=' * 50}")
        stations_df, lines_df = fetch_city(ck, cfg)
        if stations_df.empty:
            print("  [!] No data")
            continue

        out = BASE_DIR / "data" / "archive" / "raw" / "transit" / ck
        out.mkdir(parents=True, exist_ok=True)
        spath = out / f"{ck}_metro_stations_osm.csv"
        lpath = out / f"{ck}_metro_lines_osm.csv"
        stations_df.to_csv(spath, index=False, encoding="utf-8-sig")
        if not lines_df.empty:
            lines_df.to_csv(lpath, index=False, encoding="utf-8-sig")
        total += len(stations_df)
        print(f"  [OK] {len(stations_df)} stations -> {spath}")
        if not lines_df.empty:
            print(f"       {len(lines_df)} lines -> {lpath}")
        time.sleep(2)

    print(f"\n{'=' * 50}\nDone. {len(cities)} cities, {total} total stations.")


if __name__ == "__main__":
    raise SystemExit(main())
