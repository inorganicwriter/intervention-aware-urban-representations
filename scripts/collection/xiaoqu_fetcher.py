"""Offline parser for authorized Lianjia community-list HTML.

Live collection is disabled pending platform permission.  Anjuke/Lianjia
community exports should enter through ``import_housing_observations.py``.
"""

from __future__ import annotations

import argparse
import re
import sys

from bs4 import BeautifulSoup

LIVE_COLLECTION_DISABLED_REASON = (
    "Live community-page collection is disabled pending platform permission. "
    "Use an authorized export with import_housing_observations.py."
)


def parse_xiaoqu_page(soup: BeautifulSoup) -> list[dict]:
    rows = []
    for item in soup.select("ul.listContent li"):
        title = item.select_one(".title a")
        if not title:
            continue
        price = None
        price_element = item.select_one(".totalPrice span")
        if price_element:
            match = re.search(r"([\d.]+)", price_element.get_text(strip=True).replace(",", ""))
            if match:
                price = float(match.group(1))
        district = bizcircle = ""
        position = item.select_one(".positionInfo")
        if position:
            links = position.select("a")
            district = links[0].get_text(strip=True) if links else ""
            bizcircle = links[1].get_text(strip=True) if len(links) > 1 else ""
        house = item.select_one(".houseInfo")
        metro = ""
        for tag in item.select(".tagList span"):
            text = tag.get_text(strip=True)
            if "号线" in text or "地铁" in text:
                metro = text
        rows.append(
            {
                "name": title.get_text(strip=True),
                "district": district,
                "bizcircle": bizcircle,
                "unit_price": price,
                "build_info": house.get_text(strip=True) if house else "",
                "metro": metro,
            }
        )
    return rows


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(LIVE_COLLECTION_DISABLED_REASON, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
