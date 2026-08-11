"""Audit the canonical station-event table without depending on the CWD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.data.paths import CANONICAL_STATION_EVENTS  # noqa: E402

DEFAULT_EVENTS = CANONICAL_STATION_EVENTS
SOURCE_COLUMNS = ("opening_year_source", "opening_date_source", "date_source")

METRODB = {
    "beijing": 522,
    "tianjin": 233,
    "shijiazhuang": 63,
    "taiyuan": 23,
    "hohhot": 43,
    "shenyang": 85,
    "dalian": 79,
    "changchun": 100,
    "harbin": 78,
    "shanghai": 510,
    "nanjing": 227,
    "wuxi": 87,
    "xuzhou": 57,
    "changzhou": 43,
    "suzhou": 248,
    "nantong": 28,
    "hangzhou": 241,
    "ningbo": 132,
    "taizhou": 15,
    "wenzhou": 36,
    "shaoxing": 40,
    "hefei": 161,
    "fuzhou": 100,
    "xiamen": 74,
    "nanchang": 121,
    "jinan": 42,
    "qingdao": 156,
    "zhengzhou": 238,
    "luoyang": 34,
    "wuhan": 312,
    "changsha": 140,
    "guangzhou": 290,
    "shenzhen": 284,
    "foshan": 41,
    "dongguan": 15,
    "nanning": 93,
    "chongqing": 305,
    "chengdu": 388,
    "guiyang": 57,
    "kunming": 92,
    "xian": 270,
    "lanzhou": 20,
    "urumqi": 21,
    "jinhua": 32,
}


def _integer_range(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return "unavailable" if values.empty else f"{int(values.min())}-{int(values.max())}"


def audit(events: pd.DataFrame) -> None:
    required = {
        "city_key",
        "station_event_id",
        "canonical_station_name",
        "wgs84_lon",
        "wgs84_lat",
        "opening_year",
        "opening_month",
        "opening_day",
        "date_precision",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Station table lacks required columns: {missing}")

    print("=" * 70)
    print("COMPREHENSIVE DATA QUALITY AUDIT")
    print("=" * 70)
    print("\n[1] BASIC COUNTS")
    print(f"Total stations:    {len(events):,}")
    print(f"Cities:            {events['city_key'].nunique()}")
    print(f"Date precision:    {events['date_precision'].value_counts(dropna=False).to_dict()}")
    source_column = next((column for column in SOURCE_COLUMNS if column in events), None)
    if source_column:
        print(
            f"Date sources ({source_column}): {events[source_column].value_counts(dropna=False).to_dict()}"
        )
    else:
        print("Date sources:      not recorded in this table")

    print("\n[2] NULL CHECKS")
    for column in [
        "city_key",
        "station_event_id",
        "canonical_station_name",
        "wgs84_lon",
        "wgs84_lat",
        "opening_year",
        "opening_month",
    ]:
        count = int(events[column].isna().sum())
        print(f"  {column}: {count} null" + (" -- PROBLEM" if count else ""))

    print("\n[3] COORDINATE VALIDITY")
    lon_ok = pd.to_numeric(events["wgs84_lon"], errors="coerce").between(73, 135)
    lat_ok = pd.to_numeric(events["wgs84_lat"], errors="coerce").between(18, 54)
    bad_coordinates = ~(lon_ok & lat_ok)
    print(f"  Valid coordinates:  {(~bad_coordinates).sum():,} / {len(events):,}")
    if bad_coordinates.any():
        columns = ["city_key", "canonical_station_name", "wgs84_lon", "wgs84_lat"]
        print(events.loc[bad_coordinates, columns].head().to_string())

    print("\n[4] DATE VALIDITY")
    print(f"  Year range:         {_integer_range(events['opening_year'])}")
    print(f"  Month range:        {_integer_range(events['opening_month'])}")
    print(f"  Day range:          {_integer_range(events['opening_day'])}")
    months = pd.to_numeric(events["opening_month"], errors="coerce")
    days = pd.to_numeric(events["opening_day"], errors="coerce")
    print(f"  Invalid months:     {(months.notna() & ~months.between(1, 12)).sum()}")
    print(f"  Invalid days:       {(days.notna() & ~days.between(1, 31)).sum()}")

    print("\n[5] PER-CITY vs METRODB")
    issues = []
    for city_key, quota in sorted(METRODB.items()):
        observed = int((events["city_key"] == city_key).sum())
        gap = observed - quota
        if abs(gap) > 5:
            issues.append((city_key, quota, observed, gap, 100 * observed / quota))
    if issues:
        print("  Cities with gap > 5:")
        for city_key, quota, observed, gap, percent in issues:
            print(
                f"    {city_key:15s} MetroDB={quota:4d}  Ours={observed:4d}  "
                f"Gap={gap:+4d}  ({percent:.0f}%)"
            )
    else:
        print("  All cities within 5 stations of MetroDB quota")

    print("\n[6] DUPLICATE CHECK")
    print(f"  Duplicate IDs:      {events.duplicated('station_event_id').sum()}")
    print(
        f"  Duplicate coords:   {events.duplicated(['city_key', 'wgs84_lon', 'wgs84_lat']).sum()}"
    )

    print("\n[7] COORDINATE SOURCE")
    zero_coordinates = (events["wgs84_lon"] == 0.0) & (events["wgs84_lat"] == 0.0)
    print(f"  Stations with (0,0) coords: {zero_coordinates.sum()}")
    if zero_coordinates.any():
        columns = ["city_key", "canonical_station_name"]
        if "normalized_name" in events:
            columns.append("normalized_name")
        print(events.loc[zero_coordinates, columns].to_string())
    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    args = parser.parse_args(argv)
    audit(pd.read_parquet(args.events))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
