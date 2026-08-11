"""Reproducible Wayback collector for the project housing study.

Unlike the older experimental collectors, this module only enumerates captures
of the exact list endpoint for each platform.  Every downloaded page is the
exact ``timestamp + original`` pair returned by CDX, and a JSONL manifest
records every outcome so a run can be safely resumed and audited.

Examples
--------
python scripts/collection/wayback_research_scraper.py --city beijing --limit-captures 3
python scripts/collection/wayback_research_scraper.py --platform all --city all

Outputs
-------
data/archive/raw/housing/web_archives/wayback/parsed_pages/{city}_wayback_{source}.csv
data/archive/raw/housing/web_archives/wayback/inventories/{source}_{city}_{page}.json
data/archive/raw/housing/web_archives/wayback/manifests/{source}_{city}_{page}.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import signal
import sys
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts"))
import contextlib

from urban_intervention.config.project import ACTIVE_CITIES  # noqa: E402
from urban_intervention.data.paths import (  # noqa: E402
    RAW_WAYBACK_DIR,
    RAW_WAYBACK_INVENTORY_DIR,
    RAW_WAYBACK_MANIFEST_DIR,
    RAW_WAYBACK_PARSED_DIR,
)

CDX_PATH = "web.archive.org/cdx/search/cdx"
WAYBACK_PATH = "web.archive.org/web"
OUTPUT_DIR = RAW_WAYBACK_PARSED_DIR
INVENTORY_DIR = RAW_WAYBACK_INVENTORY_DIR
MANIFEST_DIR = RAW_WAYBACK_MANIFEST_DIR
LOCK_PATH = RAW_WAYBACK_DIR / ".wayback_research_scraper.lock"
USER_AGENT = "MIT-summer-research-wayback-collector/1.0 (research; contact: project-maintainer)"
# ``None`` lets requests honour standard HTTP(S)_PROXY environment variables.
# It is set from --proxy in main when a local proxy is required.
HTTP_PROXIES: dict[str, str] | None = None
# ``auto`` prefers HTTPS but supports the public HTTP endpoint when a proxy
# breaks Wayback's TLS handshake.  This fallback is necessary on some routes
# even though ordinary HTTPS sites work through the same proxy.
WAYBACK_SCHEMES: tuple[str, ...] = ("https", "http")
# SIGINT/SIGTERM set this flag instead of leaving worker threads in blocking
# sleeps.  The remaining in-flight HTTP call has a bounded 30-second read
# timeout, after which shutdown completes.
STOP_EVENT = threading.Event()
REQUEST_TIMEOUT = (10, 30)  # (connect seconds, read seconds)
CDX_TIMEOUT = (5, 10)  # CDX should be small; do not stall a city for minutes


class CrawlInterruptedError(RuntimeError):
    """Raised internally after Ctrl+C / SIGTERM requests a graceful stop."""


def stop_requested() -> bool:
    return STOP_EVENT.is_set()


def interruptible_sleep(seconds: float) -> None:
    if STOP_EVENT.wait(max(seconds, 0.0)):
        raise CrawlInterruptedError("Stop requested")


def _handle_stop_signal(signum: int, frame: object) -> None:
    if not STOP_EVENT.is_set():
        logging.warning(
            "Received signal %s; stopping after active HTTP requests finish (max 30s).", signum
        )
    STOP_EVENT.set()


LIANJIA_SUB = {
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
}
BEIKE_SUB = {
    **LIANJIA_SUB,
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
ANJUKE_SUB = {
    city: {
        "xiamen": "xm",
        "kunming": "km",
        "changchun": "cc",
        "shenyang": "sy",
        "taiyuan": "ty",
        "harbin": "hrb",
        "hohhot": "hhht",
        "urumqi": "wlmq",
    }.get(city, city)
    for city in ACTIVE_CITIES
}


@dataclass(frozen=True)
class Target:
    source: str
    city: str
    page_type: str
    host: str
    path: str
    output_suffix: str

    @property
    def output_path(self) -> Path:
        return OUTPUT_DIR / f"{self.city}_wayback_{self.output_suffix}.csv"

    @property
    def inventory_path(self) -> Path:
        return INVENTORY_DIR / f"{self.source}_{self.city}_{self.page_type}.json"

    @property
    def manifest_path(self) -> Path:
        return MANIFEST_DIR / f"{self.source}_{self.city}_{self.page_type}.jsonl"


class RequestGate:
    """A process-wide request interval shared by worker threads."""

    def __init__(self, interval: float) -> None:
        self.interval = max(interval, 0.0)
        self._lock = threading.Lock()
        self._next_request_at = 0.0

    def wait(self) -> None:
        if stop_requested():
            raise CrawlInterruptedError("Stop requested")
        with self._lock:
            now = time.monotonic()
            pause = max(0.0, self._next_request_at - now)
            self._next_request_at = max(now, self._next_request_at) + self.interval
        if pause:
            interruptible_sleep(pause)

    def defer(self, seconds: float) -> None:
        """Push every worker back after a transient Wayback/proxy failure."""
        with self._lock:
            self._next_request_at = max(self._next_request_at, time.monotonic() + seconds)


class RunLock:
    """Fail closed when a second collector would write the same manifests."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> RunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            details = self.path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"Another collector appears to be running ({self.path}: {details.strip()}). "
                "Stop it before starting a new run."
            ) from exc
        os.write(
            self.fd,
            f"pid={os.getpid()} started={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n".encode(),
        )
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()


def configured_targets(platforms: set[str], cities: set[str]) -> list[Target]:
    targets: list[Target] = []
    specs = {
        "lianjia": (
            LIANJIA_SUB,
            "lianjia.com",
            (("xiaoqu", "/xiaoqu/", "xiaoqu"), ("chengjiao", "/chengjiao/", "chengjiao")),
        ),
        "beike": (
            BEIKE_SUB,
            "ke.com",
            (
                ("xiaoqu", "/xiaoqu/", "beike_xiaoqu"),
                ("chengjiao", "/chengjiao/", "beike_chengjiao"),
            ),
        ),
        "anjuke": (ANJUKE_SUB, "anjuke.com", (("community", "/community/", "anjuke"),)),
    }
    for source in sorted(platforms):
        subdomains, domain, pages = specs[source]
        for city in sorted(cities):
            subdomain = subdomains.get(city)
            if not subdomain:
                continue
            for page_type, path, suffix in pages:
                targets.append(
                    Target(source, city, page_type, f"{subdomain}.{domain}", path, suffix)
                )
    return targets


def _request_json(params: list[tuple[str, str]], gate: RequestGate, retries: int = 3) -> list[Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        for scheme in WAYBACK_SCHEMES:
            gate.wait()
            try:
                response = requests.get(
                    f"{scheme}://{CDX_PATH}",
                    params=params,
                    timeout=CDX_TIMEOUT,
                    headers={"User-Agent": USER_AGENT},
                    proxies=HTTP_PROXIES,
                )
                if response.status_code == 200:
                    payload = response.json()
                    if not isinstance(payload, list):
                        raise ValueError("CDX response is not a JSON array")
                    return payload
                if response.status_code in {429, 502, 503, 504}:
                    last_error = RuntimeError(f"CDX HTTP {response.status_code}")
                    gate.defer(min(30, 3 * (2**attempt)))
                if response.status_code not in {429, 500, 502, 503, 504}:
                    response.raise_for_status()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
        interruptible_sleep(min(15, 2**attempt + random.uniform(0, 1)))
    raise RuntimeError(f"CDX query failed after {retries} attempts: {last_error}")


def _root_url_variants(target: Target) -> Iterable[str]:
    # CDX treats the two forms as distinct original URLs on these sites.
    yield f"{target.host}{target.path}"
    yield f"{target.host}{target.path.rstrip('/')}"


def query_exact_captures(
    target: Target, start_year: int, end_year: int, gate: RequestGate, retries: int
) -> list[dict[str, str]]:
    """Enumerate every successful capture of the exact list URL.

    ``showResumeKey`` is deliberately handled instead of assuming ``limit``
    returns all records.  The returned original URL is retained verbatim and
    later used to construct the Wayback playback URL.
    """
    records: dict[tuple[str, str], dict[str, str]] = {}
    for exact_url in _root_url_variants(target):
        resume_key: str | None = None
        while True:
            if stop_requested():
                raise CrawlInterruptedError("Stop requested")
            params = [
                ("url", exact_url),
                ("matchType", "exact"),
                ("output", "json"),
                ("fl", "timestamp,original,statuscode,mimetype,digest"),
                # Do not exclude ``warc/revisit`` captures here.  Wayback can
                # resolve them at playback time, while an HTML-only CDX filter
                # would silently lose valid list-page captures.
                ("filter", "statuscode:200"),
                ("from", str(start_year)),
                ("to", f"{end_year}1231235959"),
                ("limit", "1000"),
                ("showResumeKey", "true"),
            ]
            if resume_key:
                params.append(("resumeKey", resume_key))
            payload = _request_json(params, gate, retries=retries)
            if not payload:
                break
            header = payload[0] if isinstance(payload[0], list) else []
            positions = {name: idx for idx, name in enumerate(header)}
            next_key: str | None = None
            for row in payload[1:]:
                if not isinstance(row, list):
                    continue
                if len(row) == 1 and isinstance(row[0], str):
                    next_key = row[0]
                    continue
                try:
                    timestamp = str(row[positions["timestamp"]])
                    original = str(row[positions["original"]])
                except (KeyError, IndexError):
                    continue
                if re.fullmatch(r"\d{14}", timestamp):
                    records[(timestamp, original)] = {
                        "timestamp": timestamp,
                        "original": original,
                        "statuscode": str(row[positions.get("statuscode", -1)]),
                        "mimetype": str(row[positions.get("mimetype", -1)]),
                        "digest": str(row[positions.get("digest", -1)]),
                    }
            if not next_key or next_key == resume_key:
                break
            resume_key = next_key
    return [records[key] for key in sorted(records)]


def load_or_query_inventory(
    target: Target, args: argparse.Namespace, gate: RequestGate
) -> list[dict[str, str]]:
    if target.inventory_path.exists() and not args.refresh_inventory:
        try:
            data = json.loads(target.inventory_path.read_text(encoding="utf-8"))
            captures = data["captures"]
            if data.get("start_year") == args.start_year and data.get("end_year") == args.end_year:
                return captures
        except (OSError, ValueError, KeyError, TypeError):
            logging.warning("Ignoring invalid inventory: %s", target.inventory_path)
    captures = query_exact_captures(
        target, args.start_year, args.end_year, gate, retries=args.cdx_retries
    )
    target.inventory_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": target.source,
        "city": target.city,
        "page_type": target.page_type,
        "host": target.host,
        "path": target.path,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "captures": captures,
    }
    temporary = target.inventory_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target.inventory_path)
    return captures


def capture_key(capture: dict[str, str]) -> str:
    return f"{capture['timestamp']}\t{capture['original']}"


TERMINAL_STATUSES = {"ok", "no_rows", "not_list_page", "http_404", "http_410"}


def completed_keys(target: Target, retry_terminal: bool) -> set[str]:
    if retry_terminal or not target.manifest_path.exists():
        return set()
    result: set[str] = set()
    with target.manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
                if entry.get("status") in TERMINAL_STATUSES:
                    result.add(entry["capture_key"])
            except (ValueError, KeyError):
                continue
    return result


def extract_price(text: str) -> float | None:
    match = re.search(r"(\d[\d,]*(?:\.\d+)?)", text.replace("\xa0", " "))
    return float(match.group(1).replace(",", "")) if match else None


def parse_deals(html: str, snapshot_year: int, timestamp: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict[str, Any]] = []
    for item in soup.find_all("li"):
        date_el = item.find(class_=re.compile(r"dealDate", re.I))
        if not date_el:
            continue
        title = item.find("div", class_=re.compile(r"title", re.I))
        anchor = title.find("a") if title else None
        full_title = anchor.get_text(" ", strip=True).replace("\xa0", " ") if anchor else ""
        community_match = re.match(r"^(.+?)\s*\d+室\d+厅", full_title)
        community = community_match.group(1).strip() if community_match else full_title
        date_text = date_el.get_text(" ", strip=True)
        year_match = re.search(r"(\d{4})", date_text)
        unit_el = item.find(class_=re.compile(r"unitPrice", re.I))
        unit_price = extract_price(unit_el.get_text(" ", strip=True)) if unit_el else None
        total_el = item.find(class_=re.compile(r"totalPrice", re.I))
        total_price = extract_price(total_el.get_text(" ", strip=True)) if total_el else None
        info_el = item.find(class_=re.compile(r"houseInfo", re.I))
        info = info_el.get_text(" ", strip=True) if info_el else ""
        area_match = re.search(r"(\d+(?:\.\d+)?)\s*平", info)
        layout_match = re.search(r"\d+室\d+厅", info)
        if community and year_match and unit_price and unit_price > 100:
            rows.append(
                {
                    "community": community,
                    "unit_price": unit_price,
                    "total_price": total_price,
                    "deal_date": date_text,
                    "deal_year": int(year_match.group(1)),
                    "layout": layout_match.group(0) if layout_match else "",
                    "area_m2": float(area_match.group(1)) if area_match else None,
                    "snapshot_year": snapshot_year,
                    "snapshot_date": timestamp,
                }
            )
    return rows


def parse_lianjia_list(html: str, snapshot_year: int, timestamp: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    items = soup.find_all("li", class_=re.compile(r"xiaoquListItem|resblock", re.I))
    rows: list[dict[str, Any]] = []
    for item in items:
        title = item.find("div", class_=re.compile(r"title|resblock-name", re.I))
        anchor = (
            title.find("a") if title else item.find("a", class_=re.compile(r"resblock-name", re.I))
        )
        community = anchor.get_text(" ", strip=True).replace("\xa0", " ") if anchor else ""
        price_el = item.find(
            class_=re.compile(r"totalPrice|xiaoquListItemPrice|resblock-price", re.I)
        )
        unit_price = extract_price(price_el.get_text(" ", strip=True)) if price_el else None
        position_el = item.find(class_=re.compile(r"positionInfo|resblock-location", re.I))
        info_el = item.find(class_=re.compile(r"houseInfo", re.I))
        if community and unit_price and unit_price > 100:
            rows.append(
                {
                    "community": community,
                    "unit_price": unit_price,
                    "position": position_el.get_text(" ", strip=True) if position_el else "",
                    "house_info": info_el.get_text(" ", strip=True) if info_el else "",
                    "snapshot_year": snapshot_year,
                    "snapshot_date": timestamp,
                    "detail_url": anchor.get("href", "") if anchor else "",
                    "source_page": 1,
                }
            )
    return rows


def parse_anjuke_list(html: str, snapshot_year: int, timestamp: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    items = soup.select("ul.list li.list_item")
    if not items:
        items = soup.find_all(
            ["div", "li"], class_=re.compile(r"community-item|comm-item|list-item", re.I)
        )
    rows: list[dict[str, Any]] = []
    for item in items:
        name = item.select_one("div.t_b a.t") or item.find(
            "a", class_=re.compile(r"community|name|title", re.I)
        )
        if not name:
            continue
        community = name.get_text(" ", strip=True).replace("\xa0", " ")
        text = item.get_text(" ", strip=True)
        image_link = item.find("a", class_=re.compile(r"img_content", re.I))
        image_text = " ".join(
            filter(
                None,
                [
                    image_link.get("title", "") if image_link else "",
                    image_link.get("alt", "") if image_link else "",
                ],
            )
        )
        price_match = re.search(
            r"(\d[\d,]+)\s*元\s*[\/／](?:平|㎡|m2|平米)", image_text + " " + text, flags=re.I
        )
        unit_price = float(price_match.group(1).replace(",", "")) if price_match else None
        if community and unit_price and unit_price > 1000:
            position_el = item.find(class_=re.compile(r"position|location|area|district", re.I))
            rows.append(
                {
                    "community": community,
                    "unit_price": unit_price,
                    "position": position_el.get_text(" ", strip=True) if position_el else "",
                    "house_info": "",
                    "snapshot_year": snapshot_year,
                    "snapshot_date": timestamp,
                    "detail_url": name.get("href", ""),
                    "source_page": 1,
                }
            )
    return rows


def playback_urls(capture: dict[str, str]) -> list[str]:
    # id_ asks Wayback for the archived response rather than toolbar-rewritten HTML.
    original = quote(capture["original"], safe=":/?&=%#")
    return [
        f"{scheme}://{WAYBACK_PATH}/{capture['timestamp']}id_/{original}"
        for scheme in WAYBACK_SCHEMES
    ]


def playback_url(capture: dict[str, str]) -> str:
    """Return the preferred playback URL (kept for callers and tests)."""
    return playback_urls(capture)[0]


def fetch_and_parse(
    target: Target, capture: dict[str, str], gate: RequestGate
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    key = capture_key(capture)
    urls = playback_urls(capture)
    url = urls[0]
    response: requests.Response | None = None
    error = ""
    for attempt in range(5):
        if stop_requested():
            raise CrawlInterruptedError("Stop requested")
        terminal_response: requests.Response | None = None
        for candidate_url in urls:
            gate.wait()
            try:
                response = requests.get(
                    candidate_url,
                    timeout=REQUEST_TIMEOUT,
                    headers={"User-Agent": USER_AGENT},
                    proxies=HTTP_PROXIES,
                )
                url = candidate_url
                if response.status_code == 200:
                    break
                if response.status_code in {404, 410}:
                    terminal_response = response
                if response.status_code in {429, 502, 503, 504}:
                    gate.defer(min(120, 5 * (2**attempt)))
                error = f"HTTP {response.status_code}"
            except requests.RequestException as exc:
                error = f"{type(exc).__name__}: {exc}"
        if response is not None and response.status_code == 200:
            break
        if terminal_response is not None:
            return (
                {
                    "capture_key": key,
                    "timestamp": capture["timestamp"],
                    "original": capture["original"],
                    "playback_url": url,
                    "status": f"http_{terminal_response.status_code}",
                    "http_status": terminal_response.status_code,
                    "rows": 0,
                },
                [],
            )
        interruptible_sleep(min(60, 2**attempt + random.uniform(0, 1)))
    if response is None or response.status_code != 200:
        return (
            {
                "capture_key": key,
                "timestamp": capture["timestamp"],
                "original": capture["original"],
                "playback_url": url,
                "status": "request_error",
                "http_status": getattr(response, "status_code", None),
                "error": error,
                "rows": 0,
            },
            [],
        )
    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type.lower() and "<html" not in response.text[:2000].lower():
        return (
            {
                "capture_key": key,
                "timestamp": capture["timestamp"],
                "original": capture["original"],
                "playback_url": url,
                "status": "not_html",
                "http_status": 200,
                "rows": 0,
            },
            [],
        )
    try:
        year = int(capture["timestamp"][:4])
        if target.page_type == "chengjiao":
            rows = parse_deals(response.text, year, capture["timestamp"])
        elif target.source == "anjuke":
            rows = parse_anjuke_list(response.text, year, capture["timestamp"])
        else:
            rows = parse_lianjia_list(response.text, year, capture["timestamp"])
    except Exception as exc:  # Preserve the snapshot for a later parser fix.
        return (
            {
                "capture_key": key,
                "timestamp": capture["timestamp"],
                "original": capture["original"],
                "playback_url": url,
                "status": "parse_error",
                "http_status": 200,
                "error": f"{type(exc).__name__}: {exc}",
                "rows": 0,
            },
            [],
        )
    for row in rows:
        row["city_key"] = target.city
        row["source_platform"] = target.source
        row["original_url"] = capture["original"]
    status = "ok" if rows else "no_rows"
    return (
        {
            "capture_key": key,
            "timestamp": capture["timestamp"],
            "original": capture["original"],
            "playback_url": url,
            "status": status,
            "http_status": 200,
            "rows": len(rows),
        },
        rows,
    )


def append_manifest(path: Path, outcome: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    outcome = {**outcome, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(outcome, ensure_ascii=False) + "\n")


def merge_rows(path: Path, rows: list[dict[str, Any]], page_type: str) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
    combined = existing + rows
    # Keep every capture: the snapshot timestamp is part of the natural key.
    if page_type == "chengjiao":
        key_fields = ("snapshot_date", "original_url", "community", "deal_date", "unit_price")
    else:
        key_fields = ("snapshot_date", "original_url", "community")
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in combined:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    fieldnames = sorted({name for row in deduped for name in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(deduped)
    return len(deduped)


def collect_target(target: Target, args: argparse.Namespace, gate: RequestGate) -> None:
    captures = load_or_query_inventory(target, args, gate)
    if args.limit_captures:
        captures = captures[: args.limit_captures]
    done = completed_keys(target, args.retry_terminal)
    todo = [capture for capture in captures if capture_key(capture) not in done]
    logging.info(
        "%s/%s/%s: %d inventory, %d completed, %d to fetch",
        target.source,
        target.city,
        target.page_type,
        len(captures),
        len(captures) - len(todo),
        len(todo),
    )
    if not todo:
        return
    batch: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = []
    try:
        futures = [executor.submit(fetch_and_parse, target, capture, gate) for capture in todo]
        for index, future in enumerate(as_completed(futures), start=1):
            if stop_requested():
                raise CrawlInterruptedError("Stop requested")
            outcome, rows = future.result()
            append_manifest(target.manifest_path, outcome)
            status_counts[outcome["status"]] = status_counts.get(outcome["status"], 0) + 1
            batch.extend(rows)
            if index % args.save_every == 0 and batch:
                merge_rows(target.output_path, batch, target.page_type)
                batch.clear()
            if index % 25 == 0 or index == len(todo):
                logging.info(
                    "%s/%s/%s: %d/%d %s",
                    target.source,
                    target.city,
                    target.page_type,
                    index,
                    len(todo),
                    status_counts,
                )
    except CrawlInterruptedError:
        for future in futures:
            future.cancel()
        raise
    finally:
        # Do not let context-manager shutdown wait indefinitely on workers
        # after Ctrl+C.  In-flight requests use REQUEST_TIMEOUT above.
        executor.shutdown(wait=not stop_requested(), cancel_futures=True)
    if batch:
        merge_rows(target.output_path, batch, target.page_type)
    logging.info(
        "%s/%s/%s complete: %s -> %s",
        target.source,
        target.city,
        target.page_type,
        status_counts,
        target.output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact-capture Wayback housing collector")
    parser.add_argument(
        "--platform", default="all", help="all, lianjia, beike, anjuke; comma separated"
    )
    parser.add_argument("--city", default="all", help="all or a comma-separated list of city keys")
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent page requests")
    parser.add_argument(
        "--min-interval",
        type=float,
        default=1.5,
        help="Minimum seconds between all Wayback requests",
    )
    parser.add_argument(
        "--proxy", default="", help="Optional HTTP proxy, e.g. http://127.0.0.1:7890"
    )
    parser.add_argument(
        "--wayback-scheme",
        default="auto",
        choices=["auto", "https", "http"],
        help="Wayback transport; auto tries HTTPS then HTTP",
    )
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument(
        "--cdx-retries",
        type=int,
        default=3,
        help="Fast attempts per CDX query before deferring its city",
    )
    parser.add_argument(
        "--inventory-retry-rounds",
        type=int,
        default=3,
        help="How many full deferred-city retry rounds to run",
    )
    parser.add_argument(
        "--inventory-retry-delay",
        type=int,
        default=60,
        help="Seconds between deferred-city retry rounds",
    )
    parser.add_argument(
        "--limit-captures",
        type=int,
        default=0,
        help="Testing only: cap captures per city/page type; 0 means all",
    )
    parser.add_argument(
        "--refresh-inventory", action="store_true", help="Re-query CDX even when inventory exists"
    )
    parser.add_argument(
        "--retry-terminal",
        action="store_true",
        help="Re-fetch manifest entries normally treated as complete",
    )
    parser.add_argument(
        "--break-lock",
        action="store_true",
        help="Delete a stale lock only after confirming no collector is running",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()
    valid_platforms = {"lianjia", "beike", "anjuke"}
    args.platforms = valid_platforms if args.platform == "all" else set(args.platform.split(","))
    args.cities = set(ACTIVE_CITIES) if args.city == "all" else set(args.city.split(","))
    if not args.platforms <= valid_platforms:
        parser.error(f"Unknown platform(s): {sorted(args.platforms - valid_platforms)}")
    if not args.cities <= set(ACTIVE_CITIES):
        parser.error(
            f"Unknown or out-of-scope city key(s): {sorted(args.cities - set(ACTIVE_CITIES))}"
        )
    if (
        args.start_year > args.end_year
        or args.workers < 1
        or args.save_every < 1
        or args.limit_captures < 0
        or args.cdx_retries < 1
        or args.inventory_retry_rounds < 1
        or args.inventory_retry_delay < 0
    ):
        parser.error("Invalid numeric argument")
    return args


def main() -> int:
    global HTTP_PROXIES, WAYBACK_SCHEMES
    args = parse_args()
    STOP_EVENT.clear()
    signal.signal(signal.SIGINT, _handle_stop_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_stop_signal)
    HTTP_PROXIES = {"http": args.proxy, "https": args.proxy} if args.proxy else None
    WAYBACK_SCHEMES = ("https", "http") if args.wayback_scheme == "auto" else (args.wayback_scheme,)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.break_lock:
        try:
            LOCK_PATH.unlink()
            logging.warning("Removed lock requested by --break-lock: %s", LOCK_PATH)
        except FileNotFoundError:
            pass
    gate = RequestGate(args.min_interval)
    targets = configured_targets(args.platforms, args.cities)
    logging.info(
        "Collecting %d targets (%s cities; %s platforms; %d-%d)",
        len(targets),
        len(args.cities),
        ",".join(sorted(args.platforms)),
        args.start_year,
        args.end_year,
    )
    try:
        with RunLock(LOCK_PATH):
            pending = targets
            for retry_round in range(1, args.inventory_retry_rounds + 1):
                deferred: list[Target] = []
                for target in pending:
                    if stop_requested():
                        raise CrawlInterruptedError("Stop requested")
                    try:
                        collect_target(target, args, gate)
                    except CrawlInterruptedError:
                        raise
                    except RuntimeError as exc:
                        # A CDX/proxy failure creates no inventory or CSV;
                        # postpone it so other cities keep making progress.
                        deferred.append(target)
                        logging.warning(
                            "Deferring %s after transient inventory failure: %s", target, exc
                        )
                    except Exception:
                        logging.exception(
                            "Target failed without corrupting its manifest: %s", target
                        )
                if not deferred:
                    pending = []
                    break
                pending = deferred
                if retry_round < args.inventory_retry_rounds:
                    logging.warning(
                        "Retry round %d/%d: %d cities deferred; waiting %ss.",
                        retry_round + 1,
                        args.inventory_retry_rounds,
                        len(pending),
                        args.inventory_retry_delay,
                    )
                    interruptible_sleep(args.inventory_retry_delay)
            if pending:
                logging.error(
                    "Unresolved inventory targets after %d rounds: %s",
                    args.inventory_retry_rounds,
                    ", ".join(f"{t.source}/{t.city}/{t.page_type}" for t in pending),
                )
    except CrawlInterruptedError:
        logging.warning("Stopped cleanly; completed snapshots remain resumable.")
        return 130
    except RuntimeError as exc:
        logging.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
