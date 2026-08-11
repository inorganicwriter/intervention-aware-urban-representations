"""Generic local-dataset loader for cross-source metro station comparison.

This does NOT hard-code any specific GitHub repo (URLs go stale and inventing
them would mislead). Instead it auto-detects common JSON/GeoJSON/CSV shapes
found in the open-source "China metro data" projects on GitHub, normalizes
them to our schema, and writes per-city CSVs alongside the OSM/Amap/Wikidata
outputs so compare_transit_sources.py can score them too.

Expected input formats (auto-detected):

  1. GeoJSON FeatureCollection of station points:
       {"type":"FeatureCollection","features":[
         {"type":"Feature","properties":{"name":"...","line":"...","open":2014},
          "geometry":{"type":"Point","coordinates":[lon, lat]}}]}
     (also handles properties keys: station, stationName, station_name, 名称)

  2. JSON array of station objects:
       [{"name":"...","lon":..,"lat":..,"line":"...","year":2014}, ...]
       (also handles keys: lng, longitude, x; latitude, y; coord:[lon,lat])

  3. JSON object keyed by city, each value a list of stations:
       {"北京":[{...},{...}], "上海":[...]}

  4. CSV with a name + lon + lat (+ optional line/year) column.

Usage:
    # Point this at a directory containing cloned repo files
    python scripts/collection/github_dataset_loader.py \
        --src D:/repos/chn-metro-data --tag github_src1

    # Or a single file
    python scripts/collection/github_dataset_loader.py \
        --src D:/repos/chn-metro-data/beijing.json --tag github_src1
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
import pandas as pd

from urban_intervention.config.project import ACTIVE_CITIES, CITIES
from urban_intervention.config.project import norm_station_name as _norm


def _year(val) -> int | None:
    if val is None or val == "":
        return None
    m = re.search(r"(19|20)\d{2}", str(val))
    return int(m.group(0)) if m else None


def _city_by_bbox(lon: float, lat: float) -> str:
    for ck, cfg in CITIES.items():
        bb = cfg["bbox"]
        if bb[0] <= lon <= bb[2] and bb[1] <= lat <= bb[3]:
            return ck
    return ""


def _city_by_label(label: str) -> str:
    name_to_key = {v["name"]: k for k, v in CITIES.items()}
    label = str(label).strip()
    if label in name_to_key:
        return name_to_key[label]
    for suffix in ("市", "城区", "市区", "市辖区"):
        bare = label.removesuffix(suffix)
        if bare in name_to_key:
            return name_to_key[bare]
    # English pinyin key match (e.g. "beijing")
    for ck in CITIES:
        if ck.lower() == label.lower():
            return ck
    return ""


def _pick(d: dict, keys) -> str:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return ""


def _parse_record(rec: dict) -> dict | None:
    """Normalize a single station record to our schema. Returns None if no coords."""
    name = _pick(
        rec, ["name", "station", "stationName", "station_name", "名称", "中文名", "cn_name"]
    )
    if not name:
        return None
    line = _pick(rec, ["line", "lines", "线路", "route", "railway_line"])
    year = _year(
        _pick(
            rec,
            ["open", "opening", "opening_year", "year", "opened", "start_date", "开通", "开通日期"],
        )
    )

    lon = lat = None
    # Direct lon/lat keys
    for lk in ["lon", "lng", "longitude", "x", "经度", "wgs84_lon"]:
        if lk in rec and rec[lk] not in (None, ""):
            try:
                lon = float(rec[lk])
                break
            except (TypeError, ValueError):
                pass
    for lk in ["lat", "latitude", "y", "纬度", "wgs84_lat"]:
        if lk in rec and rec[lk] not in (None, ""):
            try:
                lat = float(rec[lk])
                break
            except (TypeError, ValueError):
                pass
    # coord / coordinates as [lon, lat] or "lon,lat"
    if lon is None or lat is None:
        for ck_key in ["coord", "coordinates", "coords", "lnglat", "point"]:
            v = rec.get(ck_key)
            if v is None:
                continue
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    lon2, lat2 = float(v[0]), float(v[1])
                    if lon is None:
                        lon = lon2
                    if lat is None:
                        lat = lat2
                    break
                except (TypeError, ValueError):
                    pass
            elif isinstance(v, str):
                m = re.findall(r"-?\d+\.?\d*", v)
                if len(m) >= 2:
                    if lon is None:
                        lon = float(m[0])
                    if lat is None:
                        lat = float(m[1])
                    break

    if lon is None or lat is None:
        return None

    city = _pick(rec, ["city", "city_name", "cityKey", "城市", "所在城市"])
    return {
        "station_name": str(name),
        "name_en": _pick(rec, ["name_en", "en_name", "english_name", "name:en"]),
        "wgs84_lon": round(lon, 7),
        "wgs84_lat": round(lat, 7),
        "opening_year": year,
        "line": str(line) if line else "",
        "_city_hint": str(city) if city else "",
    }


def _records_from_geojson(obj) -> list[dict]:
    feats = obj.get("features", []) if isinstance(obj, dict) else []
    out = []
    for f in feats:
        if not isinstance(f, dict):
            continue
        props = f.get("properties", {}) or {}
        geom = f.get("geometry", {}) or {}
        rec = dict(props)
        if geom.get("type") == "Point":
            rec["coord"] = geom.get("coordinates", [])
        out.append(rec)
    return out


def _records_from_json(obj) -> list[dict]:
    """Handle list of records, or dict keyed by city, or GeoJSON."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if obj.get("type") == "FeatureCollection":
            return _records_from_geojson(obj)
        # dict keyed by city -> flatten with city hint
        out = []
        for k, v in obj.items():
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        it = dict(it)
                        it.setdefault("city", k)
                        out.append(it)
            elif isinstance(v, dict):
                it = dict(v)
                it.setdefault("city", k)
                out.append(it)
        return out
    return []


def _read_text_with_fallback(path: Path) -> str:
    """Read text trying UTF-8 first, then GBK (common for Chinese datasets)."""
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    # Last resort: replace bad bytes
    return path.read_text(encoding="utf-8", errors="replace")


def load_file(path: Path) -> pd.DataFrame:
    """Auto-detect format and return a normalized DataFrame."""
    suffix = path.suffix.lower()
    records = []
    if suffix == ".json" or suffix == ".geojson":
        try:
            obj = json.loads(_read_text_with_fallback(path))
        except Exception as e:
            print(f"    [skip] {path.name}: JSON parse error: {e}")
            return pd.DataFrame()
        records = _records_from_json(obj)
    elif suffix == ".csv":
        try:
            # Try UTF-8-BOM first, then GBK for Chinese datasets
            for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312"):
                try:
                    with open(path, encoding=enc) as f:
                        records = list(csv.DictReader(f))
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UnicodeDecodeError("all encodings failed", b"", 0, 1, "ill-formed")
        except Exception as e:
            print(f"    [skip] {path.name}: CSV parse error: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

    rows = []
    for r in records:
        if not isinstance(r, dict):
            continue
        parsed = _parse_record(r)
        if parsed:
            rows.append(parsed)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Map city: prefer hint, else bbox
    df["city_key"] = df["_city_hint"].apply(_city_by_label)
    df.loc[df["city_key"] == "", "city_key"] = df.apply(
        lambda r: (
            _city_by_bbox(r["wgs84_lon"], r["wgs84_lat"]) if r["city_key"] == "" else r["city_key"]
        ),
        axis=1,
    )
    df = df[df["city_key"] != ""].drop(columns=["_city_hint"]).reset_index(drop=True)
    return df


def split_and_save(df: pd.DataFrame, tag: str) -> int:
    if df.empty:
        print("  No records mapped to any of the 44 cities")
        return 0
    total = 0
    for ck in ACTIVE_CITIES:
        sub = df[df["city_key"] == ck].copy()
        if sub.empty:
            continue
        sub["_n"] = sub["station_name"].apply(_norm)
        sub["_clat"] = sub["wgs84_lat"].round(3)
        sub["_clon"] = sub["wgs84_lon"].round(3)
        agg = (
            sub.groupby(["_n", "_clat", "_clon"], as_index=False)
            .agg(
                {
                    "station_name": "first",
                    "name_en": "first",
                    "wgs84_lon": "first",
                    "wgs84_lat": "first",
                    "opening_year": "min",
                    "line": lambda s: ";".join(
                        sorted({x for y in s for x in str(y).split(";") if x})
                    ),
                }
            )
            .drop(columns=["_n", "_clat", "_clon"])
            .reset_index(drop=True)
        )

        out = BASE_DIR / "data" / "archive" / "raw" / "transit" / ck
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{ck}_metro_stations_{tag}.csv"
        agg.to_csv(path, index=False, encoding="utf-8-sig")
        n_year = agg["opening_year"].notna().sum()
        print(f"  [{ck}] {len(agg)} stations, {n_year} with year -> {path}")
        total += len(agg)
    return total


def main():
    p = argparse.ArgumentParser(description="Generic local metro dataset loader")
    p.add_argument("--src", required=True, help="Directory or single file to load")
    p.add_argument(
        "--tag", required=True, help="Source tag used in output filenames, e.g. github_src1"
    )
    args = p.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"ERROR: path does not exist: {src}")
        return 1

    files = (
        [src]
        if src.is_file()
        else (list(src.rglob("*.json")) + list(src.rglob("*.geojson")) + list(src.rglob("*.csv")))
    )
    print(f"Scanning {len(files)} file(s) under {src}")
    all_dfs = []
    for f in files:
        df = load_file(f)
        if not df.empty:
            print(f"  {f.name}: {len(df)} records mapped")
            all_dfs.append(df)
    if not all_dfs:
        print("No usable station records found.")
        return 1
    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal mapped records: {len(merged)} across {merged['city_key'].nunique()} cities")
    total = split_and_save(merged, tag=args.tag)
    print(f"\nDone. Total {total} stations saved with tag '{args.tag}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
