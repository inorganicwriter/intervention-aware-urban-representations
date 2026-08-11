"""Configuration for the Anjuke community historical-price collector.

Secrets and operational parameters come from environment variables so the
repository never carries credentials:

- ANJUKE_PROXY_FILE: path to a text file with one proxy per line
  (e.g. ``http://user:pass@host:port``).  Alternative: ANJUKE_PROXIES as a
  comma-separated inline list.
- ANJUKE_CAPTCHA_API_KEY: API key of the captcha-solving platform
  (e.g. 图鉴 tjcaptcha / 超级鹰 chaojiying).  When unset, the collector runs
  without captcha solving: pages that hit the 58 antibot wall are recorded as
  ``captcha_blocked`` and skipped (the safe low-cost mode).
- ANJUKE_REQUEST_INTERVAL: minimum seconds between requests per worker
  (default 3).
- ANJUKE_ROTATE_EVERY: rotate proxy after this many requests per worker
  (default 25).
- ANJUKE_BAN_COOLDOWN: seconds a proxy is parked after a ban signal
  (default 1800).
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Directories (canonical two-way layout) ────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[3]

RAW_DIR = BASE_DIR / "data" / "archive" / "raw" / "housing" / "anjuke_history"
ID_INVENTORY_DIR = RAW_DIR / "community_ids"
HTML_DIR = RAW_DIR / "pages"
MANIFEST_DIR = RAW_DIR / "manifests"
PARSED_DIR = BASE_DIR / "data" / "archive" / "staging" / "anjuke_history"
LABEL_DIR = (
    BASE_DIR / "data" / "active" / "labels" / "housing" / "listing_price" / "anjuke_history"
)
MATCH_DIR = PARSED_DIR / "matched"
STATE_DIR = BASE_DIR / ".runtime" / "anjuke_history"

for _d in (RAW_DIR, ID_INVENTORY_DIR, HTML_DIR, MANIFEST_DIR, PARSED_DIR, MATCH_DIR, STATE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LOCK_PATH = STATE_DIR / "collector.lock"

# ── City subdomain map (community URLs use the city pinyin subdomain) ──
CITY_SUBDOMAIN = {
    "beijing": "beijing", "changchun": "changchun", "changsha": "changsha",
    "changzhou": "changzhou", "chengdu": "chengdu", "chongqing": "chongqing",
    "dalian": "dalian", "dongguan": "dongguan", "foshan": "foshan",
    "fuzhou": "fuzhou", "guangzhou": "guangzhou", "guiyang": "guiyang",
    "hangzhou": "hangzhou", "harbin": "harbin", "hefei": "hefei",
    "hohhot": "hohhot", "jinan": "jinan", "jinhua": "jinhua",
    "kunming": "kunming", "lanzhou": "lanzhou", "luoyang": "luoyang",
    "nanchang": "nanchang", "nanjing": "nanjing", "nanning": "nanning",
    "nantong": "nantong", "ningbo": "ningbo", "qingdao": "qingdao",
    "shanghai": "shanghai", "shaoxing": "shaoxing", "shenyang": "shenyang",
    "shenzhen": "shenzhen", "shijiazhuang": "shijiazhuang", "suzhou": "suzhou",
    "taiyuan": "taiyuan", "taizhou": "taizhou", "tianjin": "tianjin",
    "urumqi": "urumqi", "wenzhou": "wenzhou", "wuhan": "wuhan",
    "wuxi": "wuxi", "xiamen": "xiamen", "xian": "xian",
    "xuzhou": "xuzhou", "zhengzhou": "zhengzhou",
}

# ── HTTP defaults ─────────────────────────────────────────────────────
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 25
MIN_LIST_PAGE_INTERVAL = float(os.environ.get("ANJUKE_REQUEST_INTERVAL", "3"))
ROTATE_EVERY = int(os.environ.get("ANJUKE_ROTATE_EVERY", "25"))
BAN_COOLDOWN_S = int(os.environ.get("ANJUKE_BAN_COOLDOWN", "1800"))
LIST_PAGE_SIZE = 15  # communities per list page (site constant; verified in stage 0)

# ── Proxies ───────────────────────────────────────────────────────────
def load_proxies() -> list[str]:
    """Load the residential proxy list from env/file.  Empty means direct."""
    inline = os.environ.get("ANJUKE_PROXIES", "").strip()
    if inline:
        return [p.strip() for p in inline.split(",") if p.strip()]
    proxy_file = os.environ.get("ANJUKE_PROXY_FILE", "").strip()
    if proxy_file:
        path = Path(proxy_file)
        if not path.exists():
            raise FileNotFoundError(f"ANJUKE_PROXY_FILE not found: {path}")
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    return []


# ── Captcha solving (pluggable) ───────────────────────────────────────
CAPTCHA_API_KEY = os.environ.get("ANJUKE_CAPTCHA_API_KEY", "").strip()
CAPTCHA_TYPE = os.environ.get("ANJUKE_CAPTCHA_TYPE", "chaojiying").strip().lower()
CAPTCHA_MAX_RETRIES = int(os.environ.get("ANJUKE_CAPTCHA_MAX_RETRIES", "2"))
