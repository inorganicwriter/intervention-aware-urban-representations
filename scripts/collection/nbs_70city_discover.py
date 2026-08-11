"""Step 1: crawl the stats.gov.cn list pages to collect every 70-city HPI
article URL, then save as a CSV. This is a one-time discovery pass.
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from urban_intervention.config.project import get_proxies
from urban_intervention.data.paths import STAGING_DIR

BASE_URL = "https://www.stats.gov.cn/sj/zxfb/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}
PATTERN = re.compile(r'href="\./(\d{6}/t\d{8}_\d+\.html)"[^>]*title=[\'"]([^\'"]+)[\'"]')


def fetch(url, timeout=20):
    proxies = get_proxies()
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, proxies=proxies, timeout=timeout)
            r.raise_for_status()
            return r.content.decode("utf-8", errors="replace")
        except Exception as e:
            print(f"    attempt {attempt + 1} fail: {e}")
            time.sleep(3)
    return None


def main():
    out_dir = STAGING_DIR / "nbs_hpi"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "nbs_70city_article_index.csv"

    found = []
    # try up to 60 pages; stop early if a page fails or is empty
    for i in range(1, 61):
        url = BASE_URL if i == 1 else f"{BASE_URL}index_{i}.html"
        print(f"page {i}: {url}")
        html = fetch(url)
        if html is None:
            print(f"  stop at page {i} (failed)")
            break
        hits = PATTERN.findall(html)
        if not hits:
            print("  no article links; stop")
            break
        n_before = len(found)
        for href, title in hits:
            # keep only 70-city HPI articles
            if "70" in title and ("商品住宅" in title or "住宅销售价格" in title):
                month = href[:6]
                found.append({"month_dir": month, "href": href, "title": title})
        print(
            f"  {len(hits)} links, +{len(found) - n_before} 70-city articles (total {len(found)})"
        )
        time.sleep(1.0)

    if not found:
        print("[!] No 70-city articles found")
        return 1

    df = pd.DataFrame(found).drop_duplicates(subset=["href"])
    # parse year-month from title like "2026年5月份..."
    df["year"] = df["title"].str.extract(r"(\d{4})年")
    df["month"] = df["title"].str.extract(r"年(\d{1,2})月")
    # Drop rows where year or month couldn't be parsed (title format anomaly)
    n_before = len(df)
    df = df.dropna(subset=["year", "month"]).copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  [WARN] Dropped {n_dropped} rows with unparseable year/month")
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["ym"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    df = df.sort_values("ym").reset_index(drop=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} articles -> {out_csv}")
    print(f"Date range: {df['ym'].min()} ~ {df['ym'].max()}")
    print("\nSample rows:")
    print(df.head(3).to_string(index=False))
    print("...")
    print(df.tail(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
