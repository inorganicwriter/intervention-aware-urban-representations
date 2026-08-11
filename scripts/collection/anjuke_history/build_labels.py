"""Build community-month price labels from parsed Anjuke history pages.

Output: ``data/active/labels/housing/listing_price/anjuke_history/{city}/``
  - ``{city}_anjuke_history_monthly.parquet``: city_key, community_id,
    month, price, match_quality, page_count
  - ``{city}_coverage.csv``: per-month series length / coverage summary

Only matched communities (exact/alias/loose) that are grid-bridge admitted
are published; everything else is kept in the staging matched frame for
audit.  Quality gates (documented in the collection plan):
  - price within [500, 200000] 元/㎡ after a coarse sanity window;
  - month strings valid YYYY-MM;
  - series with < 2 points are dropped from the panel (kept in staging).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from . import config
from .matcher import load_registry, match_communities
from .parser import parse_detail_page


def parse_city(city: str) -> pd.DataFrame:
    """Parse all downloaded detail pages for one city into a long frame."""
    pages = sorted(glob.glob(str(config.HTML_DIR / city / "*.html")))
    rows: list[dict] = []
    for page in pages:
        anjuke_id = Path(page).stem
        html = page_read(page)
        result = parse_detail_page(html, city, anjuke_id)
        for month, price in result.series:
            rows.append(
                {
                    "city": city,
                    "anjuke_id": anjuke_id,
                    "name": result.name,
                    "month": month,
                    "price": price,
                    "method": result.method,
                }
            )
    return pd.DataFrame(rows)


def page_read(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def build_city_labels(city: str) -> Path:
    """Full pipeline for one city: parse -> match -> filter -> publish."""
    parsed = parse_city(city)
    registry = load_registry()
    matched = match_communities(parsed, registry)
    matched.to_parquet(config.MATCH_DIR / f"{city}_matched.parquet", index=False)

    if matched.empty:
        summary = {"city": city, "parsed_rows": 0, "matched_communities": 0,
                   "months_total": 0}
        write_summary(city, summary)
        return config.LABEL_DIR / city

    joined = matched.merge(
        registry[["community_id", "aoi_bridge_admitted"]],
        on="community_id",
        how="left",
    )
    panel = joined[
        joined["community_id"].notna()
        & (joined["aoi_bridge_admitted"] == True)  # noqa: E712
        & joined["month"].str.match(r"^\d{4}-\d{2}$")
    ].copy()
    panel["price"] = pd.to_numeric(panel["price"], errors="coerce")
    panel = panel[(panel["price"] >= 500) & (panel["price"] <= 200000)]
    panel = panel.rename(columns={"city": "city_key"})

    city_dir = config.LABEL_DIR / city
    city_dir.mkdir(parents=True, exist_ok=True)
    panel_out = panel[["city_key", "community_id", "month", "price", "match_quality"]]
    panel_out.to_parquet(city_dir / f"{city}_anjuke_history_monthly.parquet", index=False)

    coverage = panel_out.groupby("month").agg(
        communities=("community_id", "nunique"),
        prices=("price", "count"),
    ).reset_index()
    coverage.to_csv(city_dir / f"{city}_coverage.csv", index=False, encoding="utf-8-sig")

    summary = {
        "city": city,
        "parsed_rows": int(len(parsed)),
        "matched_communities": int(panel_out["community_id"].nunique()),
        "months_total": int(len(panel_out)),
        "first_month": str(panel_out["month"].min()) if len(panel_out) else "",
        "last_month": str(panel_out["month"].max()) if len(panel_out) else "",
        "methods": json.loads(matched["method"].value_counts().to_json(), parse_int=str),
    }
    write_summary(city, summary)
    return city_dir


def write_summary(city: str, summary: dict) -> None:
    path = config.LABEL_DIR / city / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
