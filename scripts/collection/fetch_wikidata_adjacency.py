"""Fetch Wikidata metro line entities and P197 adjacency for 44 cities.

City line names in the canonical station table are short forms ("1号线")
without the city prefix; Wikidata line entities carry full names
("北京地铁1号线").  Strategy:

1. For each city, resolve its Wikidata line entities via a prefix query
   (e.g. 北京地铁*, 上海轨道交通*, ...) plus P81-reverse sanity check.
2. For each resolved line entity, fetch station -> adjacent station (P197).
3. Align with the canonical station table via the numeric line token
   (e.g. "1号线" matches 北京地铁1号线) per city, so the adjacency can be
   joined to project station_event_ids.

Output: data/active/reference/transit/wikidata_adjacency.parquet
  line_entity_id, line_label, city_key, line_token, station_entity_id,
  station_name, adj_station_entity_id, adj_station_name

Usage:
    python scripts/collection/fetch_wikidata_adjacency.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "scripts"))
from urban_intervention.config.project import ACTIVE_CITIES  # noqa: E402

OUT_PATH = BASE_DIR / "data" / "active" / "reference" / "transit" / "wikidata_adjacency.parquet"
SPARQL_URL = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "MIT-summer-urban-research/1.0 (academic; contact: research@example.org)",
    "Accept": "application/sparql-results+json",
}

# 城市 -> 线路名前缀候选（Wikidata 全称形态）
CITY_PREFIXES = {
    "beijing": ["北京地铁", "北京市郊铁路", "北京机场线"],
    "shanghai": ["上海轨道交通", "上海地铁"],
    "guangzhou": ["广州地铁"],
    "shenzhen": ["深圳地铁"],
    "chengdu": ["成都地铁"],
    "wuhan": ["武汉轨道交通", "武汉地铁"],
    "chongqing": ["重庆轨道交通", "重庆地铁"],
    "tianjin": ["天津地铁", "天津轨道交通"],
    "nanjing": ["南京地铁", "南京轨道交通"],
    "hangzhou": ["杭州地铁", "杭州轨道交通"],
    "xian": ["西安地铁", "西安轨道交通"],
    "suzhou": ["苏州地铁", "苏州轨道交通"],
    "changsha": ["长沙地铁"],
    "zhengzhou": ["郑州地铁", "郑州轨道交通"],
    "qingdao": ["青岛地铁", "青岛轨道交通"],
    "shenyang": ["沈阳地铁"],
    "ningbo": ["宁波轨道交通", "宁波地铁"],
    "kunming": ["昆明地铁", "昆明轨道交通"],
    "dalian": ["大连地铁"],
    "hefei": ["合肥轨道交通", "合肥地铁"],
    "changchun": ["长春轨道交通", "长春地铁"],
    "harbin": ["哈尔滨地铁"],
    "fuzhou": ["福州地铁", "福州轨道交通"],
    "jinan": ["济南轨道交通", "济南地铁"],
    "nanning": ["南宁轨道交通", "南宁地铁"],
    "wuxi": ["无锡地铁"],
    "xiamen": ["厦门轨道交通", "厦门地铁"],
    "nanchang": ["南昌地铁", "南昌轨道交通"],
    "foshan": ["佛山地铁", "佛山轨道交通"],
    "dongguan": ["东莞轨道交通", "东莞地铁"],
    "wenzhou": ["温州轨道交通", "温州地铁"],
    "guiyang": ["贵阳轨道交通", "贵阳地铁"],
    "shijiazhuang": ["石家庄地铁"],
    "lanzhou": ["兰州轨道交通", "兰州地铁"],
    "taiyuan": ["太原轨道交通", "太原地铁"],
    "xuzhou": ["徐州地铁"],
    "nantong": ["南通轨道交通", "南通地铁"],
    "changzhou": ["常州地铁"],
    "urumqi": ["乌鲁木齐地铁", "乌鲁木齐轨道交通"],
    "shaoxing": ["绍兴轨道交通", "绍兴地铁"],
    "luoyang": ["洛阳轨道交通", "洛阳地铁"],
    "hohhot": ["呼和浩特地铁", "呼和浩特轨道交通"],
    "taizhou": ["台州轨道交通", "台州地铁"],
    "jinhua": ["金华轨道交通", "金义东轨道交通"],
}


def _query_sparql(query: str, timeout: int = 180) -> list[dict]:
    r = requests.post(SPARQL_URL, data={"query": query}, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()["results"]["bindings"]


def resolve_city_lines(city: str) -> dict[str, str]:
    """{line_entity_id: display_label} for one city's prefixes."""
    prefixes = CITY_PREFIXES.get(city, [city])
    conds = " || ".join(f"CONTAINS(?lbl, '{p}')" for p in prefixes)
    query = f"""
    SELECT DISTINCT ?line ?lbl ?stationCount WHERE {{
      ?line wdt:P31 wd:Q25301987 .
      ?line rdfs:label ?lbl .
      FILTER(LANG(?lbl) = 'zh')
      FILTER({conds})
      OPTIONAL {{
        SELECT ?line (COUNT(?s) AS ?stationCount) WHERE {{
          ?s wdt:P81 ?line .
        }} GROUP BY ?line
      }}
    }}
    """
    try:
        bindings = _query_sparql(query)
    except requests.RequestException as exc:  # noqa: BLE001
        print(f"  [warn] {city}: {exc}")
        return {}
    result: dict[str, str] = {}
    for b in bindings:
        eid = b["line"]["value"].rsplit("/", 1)[-1]
        lbl = b.get("lbl", {}).get("value", eid)
        result[eid] = lbl
    return result


def fetch_line_adjacency(line_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for eid in line_ids:
        query = f"""
        SELECT ?station ?stationLabel ?adj ?adjLabel WHERE {{
          ?station wdt:P81 wd:{eid} .
          OPTIONAL {{ ?station wdt:P197 ?adj .
                     ?adj rdfs:label ?adjLabel .
                     FILTER(LANG(?adjLabel) = 'zh') }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'zh'. }}
        }}
        """
        try:
            bindings = _query_sparql(query)
        except requests.RequestException as exc:  # noqa: BLE001
            print(f"  [warn] line {eid}: {exc}")
            time.sleep(2)
            continue
        for b in bindings:
            sid = b["station"]["value"].rsplit("/", 1)[-1]
            sname = b.get("stationLabel", {}).get("value", sid)
            if b.get("adj"):
                aid = b["adj"]["value"].rsplit("/", 1)[-1]
                aname = b.get("adjLabel", {}).get("value", aid)
            else:
                aid, aname = None, None
            rows.append(
                {
                    "line_entity_id": eid,
                    "station_entity_id": sid,
                    "station_name": sname,
                    "adj_station_entity_id": aid,
                    "adj_station_name": aname,
                }
            )
        time.sleep(0.3)
    return pd.DataFrame(rows)


def line_token(label: str) -> str:
    m = re.search(r"(\d+)号线|(\d+)线", label)
    if not m:
        return ""
    num = m.group(1) or m.group(2)
    return f"{num}号线"


def main() -> int:
    events = pd.read_parquet(
        BASE_DIR / "data" / "active" / "reference" / "transit" / "canonical_station_events_resolved.parquet",
        columns=["city_key", "lines"],
    )
    city_tokens: dict[str, set[str]] = {}
    for _, r in events.iterrows():
        city_tokens.setdefault(r["city_key"], set())
        for ln in re.split(r"[;；]", str(r["lines"])):
            ln = ln.strip()
            if ln:
                city_tokens[r["city_key"]].add(ln)

    all_edges: list[pd.DataFrame] = []
    for city in ACTIVE_CITIES:
        lines = resolve_city_lines(city)
        if not lines:
            print(f"  {city}: 0 线路实体")
            continue
        # 对齐：仅保留与站点表 token 匹配的线路
        tokens = city_tokens.get(city, set())
        keep = {eid: lbl for eid, lbl in lines.items() if line_token(lbl) in tokens}
        print(f"  {city}: {len(lines)} 实体, 对齐 {len(keep)}")
        if not keep:
            continue
        frame = fetch_line_adjacency(list(keep.keys()))
        if frame.empty:
            continue
        frame["city_key"] = city
        frame["line_label"] = frame["line_entity_id"].map(keep)
        all_edges.append(frame)

    if not all_edges:
        print("无结果")
        return 1
    combined = pd.concat(all_edges, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
    print(
        f"\nSaved: {OUT_PATH} ({len(combined)} rows, {combined['line_entity_id'].nunique()} lines)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
