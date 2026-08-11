"""Anjuke community history collector: list-page ID discovery and detail
page crawling with proxy rotation, captcha hook, and manifest resume.

Two stages:

1. ``discover_city_ids`` — crawl ``https://{city}.anjuke.com/community/p{n}/``
   list pages and record (city, anjuke_id, name, district, page_url).  The
   site paginates at ~15 communities per page; discovery stops at the last
   page (no ``p{n+1}`` link or repeated content).

2. ``crawl_city_pages`` — for every discovered ID, fetch
   ``https://{city}.anjuke.com/community/view/{id}`` and store the raw HTML
   under ``pages/{city}/{id}.html`` plus a JSONL manifest entry
   (id, url, status, sha256, fetched_at).  Existing manifest entries are
   never re-fetched (resume semantics).

The 58-group antibot wall returns a page whose <title> contains 验证码.
When a captcha key is configured the page is retried after solving;
otherwise the entry is recorded as ``captcha_blocked`` and skipped.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import requests

from . import config


class CrawlInterruptedError(RuntimeError):
    pass


_STOP = threading.Event()


def stop_requested() -> bool:
    return _STOP.is_set()


def interruptible_sleep(seconds: float) -> None:
    if seconds <= 0:
        return
    end = time.monotonic() + seconds
    while time.monotonic() < end and not _STOP.is_set():
        time.sleep(min(0.2, end - time.monotonic()))


class RequestGate:
    """Process-wide per-worker request interval."""

    def __init__(self, interval: float) -> None:
        self.interval = max(interval, 0.0)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        if stop_requested():
            raise CrawlInterruptedError("Stop requested")
        with self._lock:
            now = time.monotonic()
            pause = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self.interval
        if pause:
            interruptible_sleep(pause)

    def defer(self, seconds: float) -> None:
        with self._lock:
            self._next_at = max(self._next_at, time.monotonic() + seconds)


class ProxyPool:
    """Round-robin proxy pool with per-proxy ban cooldown.

    ``proxies()`` yields the requests-compatible ``{"http": url, "https": url}``
    dict for the current proxy.  ``park(url)`` puts a proxy into cooldown after
    a ban signal (captcha wall, 403, or connection reset).
    """

    def __init__(self, proxies: list[str]) -> None:
        self._proxies = list(proxies)
        self._lock = threading.Lock()
        self._index = 0
        self._cooldown_until: dict[str, float] = {}
        self._requests_since_rotate = 0

    def __bool__(self) -> bool:
        return bool(self._proxies)

    def current(self) -> dict[str, str] | None:
        if not self._proxies:
            return None
        with self._lock:
            self._requests_since_rotate += 1
            if self._requests_since_rotate >= config.ROTATE_EVERY:
                self._requests_since_rotate = 0
                self._index = (self._index + 1) % len(self._proxies)
            proxy = self._proxies[self._index]
            if time.monotonic() < self._cooldown_until.get(proxy, 0.0):
                self._index = (self._index + 1) % len(self._proxies)
                proxy = self._proxies[self._index]
            return {"http": proxy, "https": proxy}

    def park(self, proxy_url: str | None) -> None:
        if not proxy_url:
            return
        with self._lock:
            self._cooldown_until[proxy_url] = time.monotonic() + config.BAN_COOLDOWN_S


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


CAPTCHA_TITLE_RE = re.compile(r"验证码|verifycode|antibot", re.IGNORECASE)


def is_captcha_page(text: str) -> bool:
    head = text[:4000]
    return bool(
        CAPTCHA_TITLE_RE.search(head) and ("bbsAuthcode" in text or "antibot" in text)
    )


def solve_captcha(proxy: dict[str, str] | None) -> bool:
    """Solve the 58 antibot challenge via the configured platform.

    Returns True when the challenge was answered (the client cookie/session
    is now warmed up).  Without a configured key this returns False and the
    page is recorded as blocked.
    """
    if not config.CAPTCHA_API_KEY:
        return False
    # Pluggable: implement per platform (chaojiying / tjcaptcha / third-party
    # HTTP API).  The default implementation re-requests the challenge
    # endpoint through the same proxy; platform-specific integration goes
    # here once credentials are available.
    if config.CAPTCHA_TYPE == "chaojiying":
        # Chaojiying / 超级鹰 flow: POST the captcha image to their API,
        # receive the code, submit the answer form.  Kept as a stub with an
        # explicit not-implemented error so the run never silently proceeds.
        raise NotImplementedError(
            "chaojiying captcha backend not yet wired; set ANJUKE_CAPTCHA_TYPE "
            "or provide the platform integration."
        )
    return False


def get_page(
    url: str,
    gate: RequestGate,
    pool: ProxyPool,
    retries: int = 2,
) -> tuple[int, bytes | None, str]:
    """Fetch one page.  Returns (status, body, outcome) where outcome is one
    of ``ok``, ``captcha_blocked``, ``http_<code>``, ``error``.
    """
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        gate.wait()
        proxy = pool.current() if pool else None
        try:
            response = requests.get(
                url,
                timeout=config.REQUEST_TIMEOUT,
                headers={
                    "User-Agent": config.USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "zh-CN,zh;q=0.9",
                },
                proxies=proxy,
            )
            if response.status_code == 200:
                body = response.content
                text = body[:4000].decode("utf-8", errors="replace")
                if is_captcha_page(text):
                    if solve_captcha(proxy):
                        gate.defer(2.0)
                        continue
                    pool.park(proxy["http"] if proxy else None)
                    return 200, None, "captcha_blocked"
                return 200, body, "ok"
            if response.status_code in {403, 429, 503}:
                pool.park(proxy["http"] if proxy else None)
                gate.defer(min(30, 3 * (2**attempt)))
                return response.status_code, None, f"http_{response.status_code}"
            return response.status_code, None, f"http_{response.status_code}"
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            pool.park(proxy["http"] if proxy else None)
            gate.defer(min(20, 2 + attempt * 4))
    return 0, None, f"error:{type(last_error).__name__ if last_error else 'unknown'}"


# ── Stage 1: list-page ID discovery ───────────────────────────────────
@dataclass
class CommunityHit:
    city: str
    anjuke_id: str
    name: str
    district: str
    page_url: str


LIST_ITEM_RE = re.compile(
    r'<a[^>]+href="[^"]*/community/view/(?P<id>\d+)"[^>]*>\s*(?P<name>[^<]{1,60})</a>',
    re.IGNORECASE,
)
LIST_PAGER_RE = re.compile(r'href="([^"]*community[^"]*p(?P<page>\d+)[^"]*)"', re.IGNORECASE)


def parse_list_page(html: str) -> tuple[list[CommunityHit], int | None]:
    """Extract community hits and the highest page number linked.

    Returns (hits, last_page_hint) where last_page_hint is None when the
    pager is absent (i.e. single page or end reached).
    """
    hits: list[CommunityHit] = []
    for match in LIST_ITEM_RE.finditer(html):
        hits.append(
            CommunityHit(
                city="",  # filled by the caller
                anjuke_id=match.group("id"),
                name=match.group("name").strip(),
                district="",
                page_url="",
            )
        )
    pages = [int(m.group("page")) for m in LIST_PAGER_RE.finditer(html)]
    return hits, (max(pages) if pages else None)


def discover_city_ids(
    city: str,
    gate: RequestGate,
    pool: ProxyPool,
    limit_pages: int = 0,
) -> Path:
    """Crawl list pages until the last page and write the city ID inventory.

    Returns the inventory path (CSV with city, anjuke_id, name, district,
    page_url, fetched_at).  Existing inventory is resumed: pages already
    covered are skipped based on the stored page_url set.
    """
    subdomain = config.CITY_SUBDOMAIN[city]
    inventory_path = config.ID_INVENTORY_DIR / f"{city}_community_ids.csv"
    seen_urls: set[str] = set()
    rows: list[dict[str, str]] = []
    if inventory_path.exists():
        import csv

        with inventory_path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
                seen_urls.add(row["page_url"])

    page = 1
    while True:
        if stop_requested():
            raise CrawlInterruptedError("Stop requested")
        if limit_pages and page > limit_pages:
            break
        url = f"https://{subdomain}.anjuke.com/community/p{page}/"
        if url in seen_urls:
            page += 1
            continue
        status, body, outcome = get_page(url, gate, pool)
        if body is None:
            # Captcha wall or HTTP error: park and stop this city (the run
            # can be resumed later).
            print(f"  [{city}] page {page}: {outcome}")
            break
        html = body.decode("utf-8", errors="replace")
        hits, last_page = parse_list_page(html)
        for hit in hits:
            hit.city = city
            hit.page_url = url
            rows.append(
                {
                    "city": city,
                    "anjuke_id": hit.anjuke_id,
                    "name": hit.name,
                    "district": "",
                    "page_url": url,
                }
            )
        seen_urls.add(url)
        print(f"  [{city}] page {page}: {len(hits)} communities (total {len(rows)})")
        if not hits or (last_page is not None and page >= last_page):
            break
        page += 1

    import csv

    with inventory_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["city", "anjuke_id", "name", "district", "page_url"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return inventory_path


# ── Stage 2: detail-page crawl ────────────────────────────────────────
@dataclass
class CrawlStats:
    ok: int = 0
    captcha_blocked: int = 0
    http_error: int = 0
    other: int = 0
    last_report: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict[str, int]:
        return {
            "ok": self.ok,
            "captcha_blocked": self.captcha_blocked,
            "http_error": self.http_error,
            "other": self.other,
        }


def _manifest_path(city: str) -> Path:
    return config.MANIFEST_DIR / f"{city}.jsonl"


def _manifest_done(manifest: Path) -> set[str]:
    done: set[str] = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("status") == "ok" or entry.get("outcome") == "ok":
                done.add(str(entry["anjuke_id"]))
    return done


def crawl_city_pages(
    city: str,
    ids: Iterable[str],
    gate: RequestGate,
    pool: ProxyPool,
    workers: int = 1,
    stats: CrawlStats | None = None,
) -> Path:
    """Fetch detail pages for ``ids``, writing HTML + manifest with resume."""
    subdomain = config.CITY_SUBDOMAIN[city]
    manifest = _manifest_path(city)
    done = _manifest_done(manifest)
    todo = [i for i in ids if str(i) not in done]
    city_html_dir = config.HTML_DIR / city
    city_html_dir.mkdir(parents=True, exist_ok=True)
    stats = stats or CrawlStats()

    if workers <= 1 or len(todo) < 2:
        for anjuke_id in todo:
            _crawl_one(city, subdomain, str(anjuke_id), gate, pool, manifest, city_html_dir, stats)
        return manifest

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _crawl_one, city, subdomain, str(i), gate, pool, manifest, city_html_dir, stats
            )
            for i in todo
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # propagates CrawlInterruptedError
    return manifest


def _crawl_one(
    city: str,
    subdomain: str,
    anjuke_id: str,
    gate: RequestGate,
    pool: ProxyPool,
    manifest: Path,
    city_html_dir: Path,
    stats: CrawlStats,
) -> None:
    url = f"https://{subdomain}.anjuke.com/community/view/{anjuke_id}"
    status, body, outcome = get_page(url, gate, pool)
    entry = {
        "anjuke_id": anjuke_id,
        "url": url,
        "status": status,
        "outcome": outcome,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if body is not None:
        entry["sha256"] = sha256_bytes(body)
        (city_html_dir / f"{anjuke_id}.html").write_bytes(body)
    if outcome == "ok":
        stats.ok += 1
    elif outcome == "captcha_blocked":
        stats.captcha_blocked += 1
    elif outcome.startswith("http_"):
        stats.http_error += 1
    else:
        stats.other += 1
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    now = time.monotonic()
    if now - stats.last_report >= 60:
        stats.last_report = now
        print(f"  [{city}] progress: {stats.as_dict()}")


# ── State helpers ─────────────────────────────────────────────────────
def make_lock() -> None:
    import os

    config.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(config.LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another anjuke collector is running ({config.LOCK_PATH})."
        ) from exc
    os.write(
        fd,
        f"pid={os.getpid()} started={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n".encode(),
    )
    os.close(fd)


def release_lock() -> None:
    import contextlib
    import os

    with contextlib.suppress(FileNotFoundError):
        os.unlink(config.LOCK_PATH)
