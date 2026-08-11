"""Fetch and parse the NBS 70-city HPI articles.

Reads the article index CSV produced by nbs_70city_discover.py, downloads
every article, parses the 4 HTML tables (新建商品住宅总指数 / 二手住宅总指数
/ 新建按面积分档 / 二手按面积分档), and writes a long-form CSV:

  year, month, city, housing_type, area_class, mom, yoy, ytd

Where:
  housing_type: new | secondhand
  area_class: total | small (≤90) | medium (90-144) | large (>144)
  mom: month-on-month index (上月=100)
  yoy: year-on-year index (上年同月=100)
  ytd: year-to-date average vs same period last year (上年同期=100)

Usage:
    python scripts/collection/nbs_70city_hpi_fetcher.py
    python scripts/collection/nbs_70city_hpi_fetcher.py --limit 5
    python scripts/collection/nbs_70city_hpi_fetcher.py --since 2024-01
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

from urban_intervention.config.project import get_proxies
from urban_intervention.data.paths import RAW_DIR, STAGING_DIR

INDEX_CSV = STAGING_DIR / "nbs_hpi" / "article_index.csv"
OUT_CSV = STAGING_DIR / "nbs_hpi" / "monthly.csv"
CACHE_DIR = RAW_DIR / "nbs_hpi" / "html_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}
BASE_URL = "https://www.stats.gov.cn/sj/zxfb/"


# Cities with full-width spaces in the HTML like "北　　京", "秦 皇 岛".
# We strip all whitespace (regular + full-width U+3000) to a clean name.
def _clean_city(s: str) -> str:
    return re.sub(r"[\s\u3000]+", "", str(s))


# Map 70-city names to our 44-city keys where they overlap.
# (Our 44 cities ∩ NBS 70 cities = 37 cities. The 7 in our 44 but NOT in NBS 70:
#  changzhou, dongguan, foshan, nantong, shaoxing, suzhou, taizhou.)
# Note: xining (西宁) and yinchuan (银川) are in NBS 70 but NOT in our 44 —
# they are kept in this map for the broader 70-city panel but city_key stays
# None in the output (filtered out by build_hpi_label.py).
CITY_TO_KEY = {
    "beijing": "北京",
    "tianjin": "天津",
    "shijiazhuang": "石家庄",
    "taiyuan": "太原",
    "hohhot": "呼和浩特",
    "shenyang": "沈阳",
    "dalian": "大连",
    "changchun": "长春",
    "harbin": "哈尔滨",
    "shanghai": "上海",
    "nanjing": "南京",
    "wuxi": "无锡",
    "xuzhou": "徐州",
    "hangzhou": "杭州",
    "ningbo": "宁波",
    "hefei": "合肥",
    "fuzhou": "福州",
    "xiamen": "厦门",
    "nanchang": "南昌",
    "jinan": "济南",
    "qingdao": "青岛",
    "zhengzhou": "郑州",
    "luoyang": "洛阳",
    "wuhan": "武汉",
    "changsha": "长沙",
    "guangzhou": "广州",
    "shenzhen": "深圳",
    "nanning": "南宁",
    "chongqing": "重庆",
    "chengdu": "成都",
    "guiyang": "贵阳",
    "kunming": "昆明",
    "xian": "西安",
    "lanzhou": "兰州",
    "xining": "西宁",
    "yinchuan": "银川",
    "urumqi": "乌鲁木齐",
    "wenzhou": "温州",
    "jinhua": "金华",
}
NBS_TO_KEY = {v: k for k, v in CITY_TO_KEY.items()}


def fetch_article(href: str) -> str | None:
    """Download an article, with on-disk cache keyed by href filename."""
    fname = href.rsplit("/", 1)[-1].replace(".html", ".txt")
    cache = CACHE_DIR / fname
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    url = BASE_URL + href
    proxies = get_proxies()
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, proxies=proxies, timeout=30)
            r.raise_for_status()
            html = r.content.decode("utf-8", errors="replace")
            cache.write_text(html, encoding="utf-8")
            return html
        except Exception as e:
            print(f"    attempt {attempt + 1} fail: {e}")
            time.sleep(3)
    return None


def _to_float(s) -> float | None:
    if s is None:
        return None
    m = re.search(r"-?\d+\.?\d*", str(s))
    return float(m.group(0)) if m else None


def parse_article(html: str, year: int, month: int) -> list[dict]:
    """Parse the 4 tables of a 70-city HPI article. Returns long-form rows.

    NBS HTML tables structure (after pd.read_html flattening):
      - Table 0  (8 cols):  表1 新建商品住宅总指数   [城市,环比,同比,定基] ×2
      - Table 1  (8 cols):  表2 二手住宅总指数       [城市,环比,同比,定基] ×2
      - Tables 2-5 (10 cols): 表3 新建按面积分类(一)/(二)/(三)/(四)
                              [城市, (环,同,基)×3 for ≤90, 90-144, >144] ×2
      - Tables 6-9 (10 cols): 表4 二手按面积分类(一)/(二)/(三)/(四)

    Each table has 2-3 header rows that pd.read_html may collapse or keep as
    data rows; we drop rows where col 0 == '城市' (or has no numeric data).
    The two-city-per-row layout is unrolled into two records.
    """
    import io

    rows = []
    try:
        tables = pd.read_html(io.StringIO(html), header=None)
    except Exception as e:
        print(f"    pd.read_html fail: {e}")
        return rows

    for tbl_idx, tbl in enumerate(tables):
        if not isinstance(tbl, pd.DataFrame) or tbl.empty:
            continue
        n_cols = tbl.shape[1]

        # Two publication formats:
        #   OLD (until 2023-12): 8 cols (total, [环,同,基]×2) / 10 cols (by_area)
        #   NEW (from 2024-01):  6 cols (total, [环,同]×2)    / 7 cols (by_area)
        #   — NBS dropped the year-to-date base column from 2024 onward.
        if n_cols not in (6, 7, 8, 10):
            continue

        # The article HTML embeds each logical table twice (likely a
        # thead/tbody split that pandas flattens into two copies). Skip the
        # duplicate half: tables[0..5] are the 6 logical tables, [6..11] are
        # identical copies. Drop anything past index 5 once we've seen ≥6.
        if tbl_idx >= 6:
            break

        # Decide housing_type & area_class from table position:
        #   tbl_idx 0 -> new / total (表1 新建商品住宅总指数)
        #   tbl_idx 1 -> secondhand / total (表2 二手住宅总指数)
        #   tbl_idx 2,3 -> new / by_area (表3 (一)/(二) 新建按面积分类)
        #   tbl_idx 4,5 -> secondhand / by_area (表4 (一)/(二) 二手按面积分类)
        #
        # ── Structural validation ────────────────────────────────────
        # The position-based assignment above is fragile: if NBS adds/removes
        # a table, every subsequent table gets misclassified silently.  We
        # validate by checking that the first non-header cell looks like a
        # real city name (contains a Chinese city character and is NOT a
        # number).  If validation fails, we skip the table with a warning
        # rather than emit misclassified rows.
        if n_cols in (8, 6):  # total index
            housing_type = "new" if tbl_idx == 0 else "secondhand"
            area_class = "total"
            has_ytd = n_cols == 8
            block_size = 4 if has_ytd else 3  # city + (mom, yoy[, ytd])
        else:  # by_area (7 or 10 cols, one city per row)
            housing_type = "new" if tbl_idx in (2, 3) else "secondhand"
            area_class = "by_area"
            has_ytd = n_cols == 10
            metrics_per_area = 3 if has_ytd else 2  # mom, yoy[, ytd]

        # Force column names to canonical positions (drop whatever pd inferred)
        tbl.columns = list(range(n_cols))

        # Skip header rows: drop rows where col 0 isn't a real city (i.e. == '城市'
        # or contains only whitespace or starts with a header token).
        def _is_header_row(v):
            s = str(v).strip()
            return s in ("城市", "城　市", "") or s.startswith("城市")

        data = tbl[~tbl[0].apply(_is_header_row)].copy()
        # Also drop rows where the 2nd column is non-numeric (sub-header rows
        # like "环比=100" survive the first filter sometimes). Require the
        # whole cell to be a number (no trailing text like "100.5abc").
        data = data[data[1].apply(lambda x: bool(re.fullmatch(r"-?\d+\.?\d*", str(x).strip())))]
        if data.empty:
            continue

        # ── Position-validation: confirm first data row's city is plausible ──
        # If NBS shuffles table order, the position-based housing_type/
        # area_class above would be wrong.  We can't fully auto-detect the
        # type from the table content (NBS doesn't label tables in HTML),
        # but we CAN at least sanity-check that the first cell looks like a
        # Chinese city name.  If it doesn't (e.g. a number, a date, or
        # English text), the table structure has likely changed — skip it
        # and warn so the user can investigate.
        first_city_raw = str(data[0].iloc[0]).strip()
        if not re.search(r"[\u4e00-\u9fff]", first_city_raw):
            print(
                f"    [WARN] table {tbl_idx} (n_cols={n_cols}): first cell "
                f"'{first_city_raw}' doesn't look like a city — structure "
                f"may have changed; skipping this table"
            )
            continue

        if area_class == "total":
            # Two cities per row: cols [0:block_size] and [block_size:2*block_size]
            blocks = [(0, block_size), (block_size, 2 * block_size)]
            for lo, _hi in blocks:
                for _, r in data.iterrows():
                    city = _clean_city(r.iloc[lo])
                    if not city:
                        continue
                    mom = _to_float(r.iloc[lo + 1])
                    yoy = _to_float(r.iloc[lo + 2])
                    ytd = _to_float(r.iloc[lo + 3]) if has_ytd else None
                    if mom is None and yoy is None and ytd is None:
                        continue
                    rows.append(
                        {
                            "year": year,
                            "month": month,
                            "city": city,
                            "housing_type": housing_type,
                            "area_class": "total",
                            "mom": mom,
                            "yoy": yoy,
                            "ytd": ytd,
                        }
                    )
        else:  # by_area: one city per row, 3 area sub-classes × metrics_per_area
            for _, r in data.iterrows():
                city = _clean_city(r.iloc[0])
                if not city:
                    continue
                for k, ac in enumerate(["small", "medium", "large"]):
                    mom = _to_float(r.iloc[1 + k * metrics_per_area])
                    yoy = _to_float(r.iloc[1 + k * metrics_per_area + 1])
                    ytd = _to_float(r.iloc[1 + k * metrics_per_area + 2]) if has_ytd else None
                    if mom is None and yoy is None and ytd is None:
                        continue
                    rows.append(
                        {
                            "year": year,
                            "month": month,
                            "city": city,
                            "housing_type": housing_type,
                            "area_class": ac,
                            "mom": mom,
                            "yoy": yoy,
                            "ytd": ytd,
                        }
                    )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="Only fetch first N articles")
    p.add_argument("--since", default=None, help="Only fetch articles on/after YYYY-MM")
    p.add_argument("--refresh", action="store_true", help="Re-download even if cached")
    args = p.parse_args()

    if not INDEX_CSV.exists():
        print(f"[!] Missing index: {INDEX_CSV}. Run nbs_70city_discover.py first.")
        return 1
    idx = pd.read_csv(INDEX_CSV, encoding="utf-8-sig")
    if args.since:
        idx = idx[idx["ym"] >= args.since].reset_index(drop=True)
    if args.limit:
        idx = idx.head(args.limit).reset_index(drop=True)
    print(f"Will process {len(idx)} articles ({idx['ym'].min()} ~ {idx['ym'].max()})")

    all_rows = []
    for i, row in idx.iterrows():
        href = row["href"]
        ym = row["ym"]
        # Guard against NaN year/month (shouldn't happen after discover.py
        # fix, but be defensive in case the CSV was hand-edited).
        if pd.isna(row["year"]) or pd.isna(row["month"]):
            print(f"[{i + 1}/{len(idx)}] {ym} -> SKIP (missing year/month)")
            continue
        year = int(row["year"])
        month = int(row["month"])
        print(f"[{i + 1}/{len(idx)}] {ym} -> {href}")
        if args.refresh:
            cache = CACHE_DIR / href.rsplit("/", 1)[-1].replace(".html", ".txt")
            if cache.exists():
                cache.unlink()
        html = fetch_article(href)
        if html is None:
            print("  [!] fetch failed, skipping")
            continue
        rows = parse_article(html, year, month)
        # Tag housing_type by table number (more robust than guess from text).
        # Use presence of "表1" / "表2" / "表3" / "表4" markers around the
        # html before each table — but pd.read_html doesn't expose that.
        # Simpler heuristic by article position in tables list:
        if rows:
            print(f"  parsed {len(rows)} rows ({len(set(r['city'] for r in rows))} cities)")
        all_rows.extend(rows)
        time.sleep(1.5)  # be polite

    if not all_rows:
        print("[!] No rows parsed")
        return 1
    df = pd.DataFrame(all_rows)
    # Map NBS city -> our city_key where possible (leave unmapped cities in
    # raw form — they're useful for the broader 70-city panel)
    df["city_key"] = df["city"].map(NBS_TO_KEY)
    df = df[
        ["year", "month", "city", "city_key", "housing_type", "area_class", "mom", "yoy", "ytd"]
    ]
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} rows -> {OUT_CSV}")
    print(f"  cities (total): {df['city'].nunique()}")
    print(
        f"  cities (mapped to our 44): {df['city_key'].notna().sum() // (df['housing_type'].nunique() * df['area_class'].nunique())}"
    )
    print(f"  housing_types: {sorted(df['housing_type'].unique())}")
    print(f"  area_classes: {sorted(df['area_class'].unique())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
