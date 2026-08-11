"""
Wikipedia Metro Station List Fetcher (No API Key needed)
Fetches station name, line, and opening year from zh.wikipedia.org for 4 cities.

Usage:
    python wiki_transit_fetch.py --city beijing
    python wiki_transit_fetch.py --city all
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

# Fix Windows console encoding — Chinese characters (especially
# zero-width spaces like \u200b in Wikipedia HTML) crash GBK stdout.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
import pandas as pd
import requests

from urban_intervention.config.project import get_proxies


# Dynamic URL generation from pipeline_config city names.
# Some cities use "地铁车站列表", others use "轨道交通车站列表".
def _wiki_urls(city_key: str) -> list[str]:
    from urllib.parse import quote

    from urban_intervention.config.project import CITIES

    name = CITIES.get(city_key, {}).get("name", city_key)
    # The entire page name must be URL-encoded together.  Previously only the
    # city name was quoted, leaving the Chinese suffix ("地铁车站列表")
    # un-encoded in the URL path.  Wikipedia's server rejects mixed
    # encoded/unencoded paths with HTTP 403.
    base = f"https://zh.wikipedia.org/wiki/{quote(name)}"
    suffixes = ["地铁车站列表", "轨道交通车站列表", "地铁"]
    return [f"{base}{quote(suffix)}" for suffix in suffixes]


HEADERS = {
    # Wikipedia blocks overly-specific browser User-Agent strings as bot-like.
    # A simple descriptive UA identifying the research project is accepted.
    "User-Agent": "MIT-Summer-Research/1.0 (metro station data collection; contact: research@example.com)",
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def fetch_city(city_key: str) -> pd.DataFrame:
    """Scrape station list from Wikipedia table(s).

    Tries multiple URL patterns: "XX地铁车站列表" and "XX轨道交通车站列表",
    with a zh.m.wikipedia.org mobile fallback for each.
    """
    urls = _wiki_urls(city_key)
    proxies = get_proxies()
    html_text = None

    for base_url in urls:
        for try_url in [base_url, base_url.replace("zh.wikipedia.org", "zh.m.wikipedia.org")]:
            try:
                resp = requests.get(try_url, headers=HEADERS, proxies=proxies, timeout=30)
                if resp.status_code == 200:
                    html_text = resp.text
                    print(f"  URL: {try_url}")
                    break
            except Exception:
                pass
        if html_text:
            break
        else:
            print(f"    No match: {base_url}")

    if html_text is None:
        return pd.DataFrame()

    try:
        tables = pd.read_html(io.StringIO(html_text))
    except Exception as e:
        print(f"    Table parse failed: {e}")
        return pd.DataFrame()

    print(f"    Got {len(tables)} table(s)")

    stations = []
    for table in tables:
        if table.empty or len(table.columns) < 2:
            continue

        # Flatten MultiIndex headers
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = [
                "_".join(str(c) for c in col if "Unnamed" not in str(c)).strip("_")
                for col in table.columns
            ]
        table.columns = [str(c).strip() for c in table.columns]

        # Detect columns
        name_col = next(
            (c for c in table.columns if "站名" in c or "车站名称" in c or "名称" in c),
            table.columns[0],
        )
        line_col = next(
            (c for c in table.columns if "线路" in c or "路线" in c or "所属线路" in c),
            None,
        )
        year_col = next(
            (c for c in table.columns if "开通" in c or "启用" in c or "年份" in c or "运营" in c),
            None,
        )

        for _, row in table.iterrows():
            name = str(row.get(name_col, "")).strip()
            if not name or len(name) < 2:
                continue

            line = str(row.get(line_col, "")).strip() if line_col else ""
            date_info = _extract_opening_date(str(row.get(year_col, "")) if year_col else "")

            # Fallback: scan columns whose NAME suggests they might carry a
            # date/year (开通/启用/运营/年份/时间/建成/首车/通车).  Previously
            # this scanned ALL columns and could match any 4-digit number
            # (e.g. a distance like "2019 m" or a building-year column that
            # actually refers to something else).  Restricting to columns
            # whose header contains a date-related keyword dramatically
            # reduces false positives.
            if date_info["opening_year"] is None:
                date_keywords = (
                    "开通",
                    "启用",
                    "运营",
                    "年份",
                    "时间",
                    "建成",
                    "首车",
                    "通车",
                    "启用",
                    "开业",
                )
                for col in table.columns:
                    if not any(kw in str(col) for kw in date_keywords):
                        continue
                    if col in (name_col, line_col):
                        continue
                    candidate = _extract_opening_date(str(row.get(col, "")))
                    if candidate["opening_year"] is not None:
                        date_info = candidate
                        break

            stations.append(
                {
                    "station_name": name,
                    "line": line,
                    "opening_year": date_info["opening_year"],
                    "opening_month": date_info["opening_month"],
                    "opening_day": date_info["opening_day"],
                    "opening_date": date_info["opening_date"],
                    "date_precision": date_info["date_precision"],
                }
            )

    df = pd.DataFrame(stations)
    if not df.empty:
        df = df.drop_duplicates(subset=["station_name", "line"])

    print(f"    Parsed {len(df)} station-line entries")
    if "opening_year" in df.columns:
        n_years = df["opening_year"].notna().sum()
        if n_years:
            yrs = df["opening_year"].dropna()
            print(
                f"    Year range: {int(yrs.min())}-{int(yrs.max())} ({n_years}/{len(df)} have year)"
            )
        n_lines = df["line"].nunique() if "line" in df.columns else 0
        if n_lines:
            print(f"    Lines: {n_lines}")

    return df


def _extract_year(text: str) -> int | None:
    m = re.search(r"(19|20)\d{2}", text)
    return int(m.group(0)) if m else None


# Match Chinese and ISO opening dates.
# Examples the Wikipedia tables contain:
#   "2014年12月28日"     → 2014, 12, 28
#   "2014-12-28"         → 2014, 12, 28
#   "2014年12月"         → 2014, 12, None
#   "2014年"             → 2014, None, None
#   "2014.12.28"         → 2014, 12, 28
_DATE_CN_DAY = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_DATE_CN_MONTH = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月(?!\s*\d)")
_DATE_CN_YEAR = re.compile(r"(\d{4})\s*年")
_DATE_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_DATE_DOT = re.compile(r"(\d{4})\.(\d{1,2})\.(\d{1,2})")


def _extract_opening_date(text: str) -> dict:
    """Extract year/month/day and precision from a Wikipedia date string.

    Returns a dict with keys ``opening_year``, ``opening_month``,
    ``opening_day``, ``opening_date`` (``YYYY-MM-DD`` or partial) and
    ``date_precision`` (``"day"``, ``"month"``, ``"year"`` or ``""``).
    """
    s = str(text or "").strip()
    if not s:
        return {
            "opening_year": None,
            "opening_month": None,
            "opening_day": None,
            "opening_date": "",
            "date_precision": "",
        }

    # Try day-precision patterns first.
    for pat, _fmt in [(_DATE_CN_DAY, "cn"), (_DATE_ISO, "iso"), (_DATE_DOT, "dot")]:
        m = pat.search(s)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return {
                "opening_year": y,
                "opening_month": mo,
                "opening_day": d,
                "opening_date": f"{y:04d}-{mo:02d}-{d:02d}",
                "date_precision": "day",
            }

    # Month precision: "2014年12月" without a day.
    m = _DATE_CN_MONTH.search(s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return {
            "opening_year": y,
            "opening_month": mo,
            "opening_day": None,
            "opening_date": f"{y:04d}-{mo:02d}-00",
            "date_precision": "month",
        }

    # Year only: "2014年" or a bare 4-digit year.
    m = _DATE_CN_YEAR.search(s)
    if m:
        y = int(m.group(1))
        return {
            "opening_year": y,
            "opening_month": None,
            "opening_day": None,
            "opening_date": f"{y:04d}-00-00",
            "date_precision": "year",
        }

    y = _extract_year(s)
    if y is not None:
        return {
            "opening_year": y,
            "opening_month": None,
            "opening_day": None,
            "opening_date": f"{y:04d}-00-00",
            "date_precision": "year",
        }

    return {
        "opening_year": None,
        "opening_month": None,
        "opening_day": None,
        "opening_date": "",
        "date_precision": "",
    }


# ── Main ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Wikipedia metro station scraper")
    parser.add_argument("--city", default="all")
    args = parser.parse_args()

    from urban_intervention.config.project import ACTIVE_CITIES, CITIES

    if args.city == "all":
        cities = ACTIVE_CITIES
    else:
        cities = [c.strip() for c in args.city.split(",") if c.strip() in CITIES]

    for ck in cities:
        print(f"\n{'─' * 50}\n{ck.upper()}")
        df = fetch_city(ck)

        if df.empty:
            print("  [!] No data")
            continue

        out_dir = BASE_DIR / "data" / "archive" / "raw" / "transit" / "wikipedia" / ck
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{ck}_stations_wiki.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  -> Saved: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
