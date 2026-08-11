"""Housing page parsers and legacy CSV import helpers.

Live Lianjia/Beike collection is intentionally disabled.  The platforms'
access controls must not be bypassed; use an authorized bulk/platform export
with ``import_housing_observations.py``.  The HTML parsers remain available for
licensed page archives and reproducible offline tests.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "scripts"))
from urban_intervention.config.project import ACTIVE_CITIES, CITIES  # noqa: E402
from urban_intervention.data.paths import RAW_WAYBACK_PARSED_DIR  # noqa: E402

OUT_DIR = RAW_WAYBACK_PARSED_DIR
LIANJIA_SUBDOMAIN: dict[str, str | None] = {
    "beijing": "bj",
    "shanghai": "sh",
    "guangzhou": "gz",
    "shenzhen": "sz",
    "chengdu": "cd",
    "hangzhou": "hz",
    "wuhan": "wh",
    "nanjing": "nj",
    "tianjin": "tj",
    "chongqing": "cq",
    "suzhou": "su",
    "xian": "xian",
    "changsha": "cs",
    "dalian": "dl",
    "shenyang": "sy",
    "qingdao": "qd",
    "jinan": "jn",
    "foshan": "fs",
    "dongguan": "dg",
    "xiamen": "xm",
    "hefei": "hf",
    "zhengzhou": "zz",
    "kunming": "km",
    "fuzhou": "fz",
    "nanning": "nn",
    "wuxi": "wx",
    "ningbo": "nb",
    "changchun": "cc",
    "guiyang": "gy",
    "shijiazhuang": "sjz",
    "harbin": "hrb",
    "taiyuan": "ty",
    "nanchang": "nc",
    "lanzhou": "lz",
    "hohhot": "hhht",
    "urumqi": None,
    "wenzhou": None,
    "xuzhou": None,
    "jinhua": None,
    "shaoxing": None,
    "taizhou": None,
    "luoyang": None,
    "nantong": None,
    "changzhou": None,
}
KE_SUBDOMAIN: dict[str, str | None] = {
    **LIANJIA_SUBDOMAIN,
    "xian": "xa",
    "urumqi": "wlmq",
    "wenzhou": "wz",
    "xuzhou": "xz",
    "jinhua": "jh",
    "shaoxing": "sx",
    "taizhou": "tz",
    "luoyang": "ly",
    "nantong": "nt",
    "changzhou": "cz",
}

_EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]
EDGE_EXECUTABLE = next((str(path) for path in _EDGE_CANDIDATES if path.exists()), None)
CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
LIVE_COLLECTION_DISABLED_REASON = (
    "Live Lianjia/Beike collection is disabled. Obtain a platform-authorized "
    "or licensed export, then run scripts/collection/import_housing_observations.py."
)


def _browser_launch_options(login_mode: bool) -> dict:
    """Return ordinary browser options for an explicitly authorized diagnostic."""

    options = {
        "headless": not login_mode,
        "viewport": {"width": 1920, "height": 1080},
        "locale": "zh-CN",
        "timezone_id": "Asia/Shanghai",
        "user_agent": CHROME_UA,
        "extra_http_headers": {"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
    }
    if EDGE_EXECUTABLE:
        options["executable_path"] = EDGE_EXECUTABLE
    return options


def parse_chengjiao_page(html: str) -> list[dict]:
    """Extract transaction rows from a licensed Lianjia/Beike HTML page."""

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    items = []
    for class_name in ["LOGCLICKDATA", "LOGVIEWDATA", "CLEAR"]:
        items = soup.find_all("li", class_=re.compile(class_name))
        if items:
            break
    if not items:
        items = [item for item in soup.find_all("li") if item.find(class_=re.compile("dealDate"))]

    rows: list[dict] = []
    for item in items:
        title_link = item.find("a", href=re.compile(r"/chengjiao/\d+\.html"))
        community = re.sub(r"\s+", " ", title_link.get_text(strip=True)) if title_link else ""
        info_div = item.find(class_=re.compile("houseInfo"))
        info = info_div.get_text(strip=True) if info_div else ""
        parts = [part.strip() for part in info.split("|") if part.strip()]
        layout = parts[0] if parts else ""
        area_m2 = None
        orientation = decoration = floor = ""
        for part in parts:
            area_match = re.search(r"([\d.]+)㎡", part)
            if area_match:
                area_m2 = float(area_match.group(1))
            elif any(word in part for word in ["东", "南", "西", "北"]):
                orientation = part
            elif any(word in part for word in ["精装", "简装", "毛坯", "豪装", "中装"]):
                decoration = part
            elif any(word in part for word in ["楼层", "底层", "顶层"]):
                floor = part

        date_element = item.find(class_=re.compile("dealDate"))
        deal_date = date_element.get_text(strip=True) if date_element else ""
        year_match = re.search(r"(\d{4})", deal_date)
        deal_year = int(year_match.group(1)) if year_match else None

        def price(class_pattern: str, item=item) -> tuple[float | None, str]:
            element = item.find(class_=re.compile(class_pattern))
            if element is None:
                return None, ""
            span = element.find("span")
            text = (span or element).get_text(strip=True).replace(",", "")
            number = re.search(r"([\d.]+)", text)
            return (float(number.group(1)) if number else None), text

        total_price, total_text = price("totalPrice")
        if total_price is not None and "万" not in total_text and total_price > 10_000:
            total_price /= 10_000
        unit_price, _ = price("unitPrice")
        build_match = re.search(r"(\d{4})年建", info)
        build_year = int(build_match.group(1)) if build_match else None
        if community and deal_year:
            rows.append(
                {
                    "community": community,
                    "layout": layout,
                    "area_m2": area_m2,
                    "orientation": orientation,
                    "decoration": decoration,
                    "floor": floor,
                    "build_year": build_year,
                    "unit_price": unit_price,
                    "total_price": total_price,
                    "deal_date": deal_date,
                    "deal_year": deal_year,
                }
            )
    return rows


def diagnose_html(html: str) -> dict[str, int]:
    markers = [
        "LOGVIEWDATA",
        "LOGCLICKDATA",
        "data-lj_action",
        "houseInfo",
        "dealDate",
        "totalPrice",
        "unitPrice",
        "data-total-count",
        "page-data",
        "captcha",
        "验证",
        "人机验证",
        "anti-spider",
        "forbidden",
        "访问已被拦截",
        "账号已被封禁",
        "chengjiao",
        "community",
    ]
    return {marker: html.lower().count(marker.lower()) for marker in markers}


def _is_blocked_url_title(url: str, title: str) -> bool:
    url_text = (url or "").lower()
    title_text = (title or "").lower()
    return any(
        marker in url_text
        for marker in ["hip.lianjia.com/captcha", "hip.lianjia.com/forbidden", "captcha"]
    ) or any(
        marker in title_text
        for marker in ["captcha", "验证", "人机验证", "访问已被拦截", "forbidden"]
    )


def _filter_by_year(rows: list[dict], min_year: int) -> list[dict]:
    if not min_year:
        return rows
    return [row for row in rows if row.get("deal_year") and row["deal_year"] >= min_year]


def _merge_with_existing(
    new_rows: list[dict], out_csv: Path, existing_count: int
) -> tuple[pd.DataFrame, int, int]:
    subset = ["community", "area_m2", "deal_date", "total_price"]
    fresh = pd.DataFrame(new_rows).drop_duplicates(subset=subset)
    if out_csv.exists() and existing_count > 0:
        existing = pd.read_csv(out_csv, encoding="utf-8-sig")
        combined = pd.concat([existing, fresh], ignore_index=True).drop_duplicates(subset=subset)
        return combined, len(combined), len(combined) - len(existing)
    return fresh, len(fresh), len(fresh)


def _resolve_subdomain(city_key: str) -> tuple[str | None, str]:
    subdomain = LIANJIA_SUBDOMAIN.get(city_key)
    if subdomain:
        return subdomain, "lianjia"
    subdomain = KE_SUBDOMAIN.get(city_key)
    if subdomain:
        return subdomain, "beike"
    return None, ""


def scrape_city_playwright(*args: object, **kwargs: object) -> int:
    raise RuntimeError(LIVE_COLLECTION_DISABLED_REASON)


def login_only(*args: object, **kwargs: object) -> int:
    raise RuntimeError(LIVE_COLLECTION_DISABLED_REASON)


def import_csv(city_key: str, csv_path: str) -> int:
    """Legacy compatibility import; new batches must use the canonical importer."""

    if city_key not in CITIES:
        raise ValueError(f"Unknown city: {city_key}")
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "community" not in frame.columns:
        raise ValueError("CSV must contain a community column")
    if "deal_year" not in frame.columns and "deal_date" in frame.columns:
        frame["deal_year"] = pd.to_numeric(
            frame["deal_date"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"{city_key}_lianjia_transactions.csv"
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    return len(frame)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="all")
    parser.add_argument("--list-cities", action="store_true")
    parser.add_argument("--import-csv")
    args = parser.parse_args()
    if args.list_cities:
        for city_key in ACTIVE_CITIES:
            subdomain, site = _resolve_subdomain(city_key)
            print(f"{city_key},{site},{subdomain or ''}")
        return 0
    if args.import_csv:
        if args.city == "all" or "," in args.city:
            raise ValueError("Legacy --import-csv requires exactly one --city")
        print(
            "WARNING: legacy import path; prefer import_housing_observations.py "
            "with a versioned mapping."
        )
        return import_csv(args.city, args.import_csv)
    print(LIVE_COLLECTION_DISABLED_REASON, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
