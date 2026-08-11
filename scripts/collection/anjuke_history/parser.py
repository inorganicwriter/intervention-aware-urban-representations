"""Parser for Anjuke community detail pages.

The exact page structure will be verified in stage 0 (the 58 antibot wall
blocks inspection from this network).  The parser therefore uses several
loose, order-independent strategies and records which one matched, so stage 0
can lock the real selectors without changing the pipeline:

1. Inline JSON blobs containing month labels and prices (e.g. highcharts
   ``categories`` + ``data`` pairs, or ``__INITIAL_STATE__`` payloads).
2. HTML tables / lists with ``YYYY-MM`` (or ``YYYY年MM月``) + price tokens.
3. The current listed price plus any ``历史价格``-style table rows.

Output rows: city, anjuke_id, name, month (YYYY-MM), price (float, 元/㎡),
match_method.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

MONTH_RE = re.compile(
    r"(20(?:0\d|1\d|2\d)[年\-/\.](?:0?[1-9]|1[0-2])月?)"
)
PRICE_RE = re.compile(r"(\d{3,7}(?:\.\d+)?)\s*元[／/]?[平㎡]?")
JSON_MONTH_RE = re.compile(
    r'"(?:categor(?:y|ies)|month|date|time)s?"\s*:\s*(\[[^\]]{5,2000}\])'
)
JSON_SERIES_RE = re.compile(r'"data"\s*:\s*(\[[0-9\.,\[\]"]{20,20000}\])')
# highcharts-style [[timestamp_ms, price], ...] series
NESTED_SERIES_RE = re.compile(
    r"\[\s*(\d{10,13})\s*,\s*(\d+(?:\.\d+)?)\s*\]"
)


@dataclass
class ParseResult:
    city: str
    anjuke_id: str
    name: str
    current_price: float | None
    series: list[tuple[str, float]]
    method: str


def _json_number_list(text: str) -> list[float]:
    try:
        values = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def parse_detail_page(html: str, city: str, anjuke_id: str) -> ParseResult:
    """Best-effort extraction; ``method`` says which strategy hit."""
    name = ""
    title = re.search(r"<title>(.*?)</title>", html, re.S)
    if title:
        name = re.sub(r"\s+", "", title.group(1)).replace("安居客", "").replace(
            "房价", ""
        ).strip()[:40]

    series: list[tuple[str, float]] = []
    method = "none"

    # Strategy 0: highcharts [[timestamp_ms, price], ...] nested series.
    nested = NESTED_SERIES_RE.findall(html)
    if len(nested) >= 2:
        import datetime

        for ts_text, price_text in nested:
            ts = int(ts_text)
            month = datetime.datetime.fromtimestamp(ts / 1000).strftime("%Y-%m")
            series.append((month, float(price_text)))
        series = dedupe_months(series)
        method = "nested_series"

    # Strategy 1: JSON category/series pairs (highcharts-style).
    if not series:
        months_json = JSON_MONTH_RE.findall(html)
        data_json = JSON_SERIES_RE.findall(html)
        for categories, values in zip(months_json, data_json, strict=False):
            months = re.findall(r"20(?:0\d|1\d|2\d)[年\-/\.](?:0?[1-9]|1[0-2])月?", categories)
            prices = _json_number_list(values)
            if months and prices and len(months) >= 2 and len(prices) >= 2:
                for month, price in zip(months, prices, strict=False):
                    series.append((normalize_month(month), price))
                method = "json_series"
                break

    # Strategy 2: month/price regex pairs anywhere in the HTML.
    if not series:
        month_positions = [(m.start(), normalize_month(m.group(0))) for m in MONTH_RE.finditer(html)]
        price_positions = [(m.start(), float(m.group(1))) for m in PRICE_RE.finditer(html)]
        for pos, month in month_positions:
            nearest = min(
                price_positions,
                key=lambda p: abs(p[0] - pos),
                default=None,
            )
            if nearest is not None and abs(nearest[0] - pos) < 300:
                series.append((month, nearest[1]))
        if series:
            series = dedupe_months(series)
            method = "regex_pairs"

    current_price: float | None = None
    if series:
        current_price = series[-1][1]
    elif price_positions := [(m.start(), float(m.group(1))) for m in PRICE_RE.finditer(html)]:
        current_price = price_positions[0][1]
        method = "current_only"

    series = dedupe_months(series)
    return ParseResult(
        city=city,
        anjuke_id=anjuke_id,
        name=name,
        current_price=current_price,
        series=series,
        method=method,
    )


def normalize_month(token: str) -> str:
    match = re.search(r"(20(?:0\d|1\d|2\d))[年\-/\.](\d{1,2})", token)
    if not match:
        return ""
    return f"{match.group(1)}-{int(match.group(2)):02d}"


def dedupe_months(series: list[tuple[str, float]]) -> list[tuple[str, float]]:
    seen: dict[str, float] = {}
    for month, price in series:
        if month and price and price > 0:
            seen[month] = price
    return sorted(seen.items())
