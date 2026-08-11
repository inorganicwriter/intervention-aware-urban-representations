"""CLI for the Anjuke community historical-price collector.

Stages:
  discover   crawl community list pages -> community ID inventories
  collect    crawl detail pages (manifest resume) -> raw HTML + manifest
  parse      parse raw HTML -> long price series (staging)
  build      match to registry + publish labels (per city)

Examples:
  python scripts/collection/run_anjuke_history.py --stage discover --city nanchang --limit-pages 3
  python scripts/collection/run_anjuke_history.py --stage collect --city nanchang --workers 4
  python scripts/collection/run_anjuke_history.py --stage build --city nanchang
  python scripts/collection/run_anjuke_history.py --stage discover --city all
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "scripts"))

from anjuke_history import config  # noqa: E402
from anjuke_history.collector import (  # noqa: E402
    CrawlInterruptedError,
    CrawlStats,
    ProxyPool,
    RequestGate,
    crawl_city_pages,
    discover_city_ids,
    make_lock,
    release_lock,
    stop_requested,
)

from urban_intervention.config.project import ACTIVE_CITIES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=["discover", "collect", "parse", "build", "all"]
    )
    parser.add_argument("--city", default="nanchang", help="city key or 'all'")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-pages", type=int, default=0, help="discover: cap list pages")
    parser.add_argument("--limit-ids", type=int, default=0, help="collect: cap detail pages")
    parser.add_argument("--min-interval", type=float, default=None)
    return parser.parse_args()


def city_ids(city: str) -> list[str]:
    path = config.ID_INVENTORY_DIR / f"{city}_community_ids.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [row["anjuke_id"] for row in csv.DictReader(fh) if row["anjuke_id"]]


def main() -> int:
    args = parse_args()
    cities = list(ACTIVE_CITIES) if args.city == "all" else [args.city]
    if args.city not in ACTIVE_CITIES and args.city != "all":
        print(f"Unknown city '{args.city}'")
        return 2
    if args.min_interval is not None:
        config.MIN_LIST_PAGE_INTERVAL = args.min_interval

    proxies = config.load_proxies()
    pool = ProxyPool(proxies)
    gate = RequestGate(config.MIN_LIST_PAGE_INTERVAL)
    print(f"proxies: {len(proxies)} | captcha: {config.CAPTCHA_API_KEY or 'none'}")

    stages = ["discover", "collect", "parse", "build"] if args.stage == "all" else [args.stage]

    make_lock()
    try:
        for city in cities:
            for stage in stages:
                print(f"== {city} / {stage} ==")
                if stage == "discover":
                    discover_city_ids(city, gate, pool, limit_pages=args.limit_pages)
                elif stage == "collect":
                    ids = city_ids(city)
                    if args.limit_ids:
                        ids = ids[: args.limit_ids]
                    if not ids:
                        print("  no community IDs (run discover first)")
                        continue
                    print(f"  {len(ids)} communities to collect")
                    crawl_city_pages(
                        city, ids, gate, pool, workers=args.workers,
                        stats=CrawlStats(),
                    )
                elif stage == "parse":
                    from anjuke_history.build_labels import parse_city

                    parsed = parse_city(city)
                    print(f"  parsed rows: {len(parsed)}")
                    if not parsed.empty:
                        parsed.to_parquet(
                            config.PARSED_DIR / f"{city}_parsed.parquet", index=False
                        )
                elif stage == "build":
                    from anjuke_history.build_labels import build_city_labels

                    city_dir = build_city_labels(city)
                    summary = city_dir / "summary.json"
                    if summary.exists():
                        print(summary.read_text(encoding="utf-8"))
    except CrawlInterruptedError:
        print("interrupted; state is resumable")
        return 130
    except KeyboardInterrupt:
        stop_requested()
        print("interrupted; state is resumable")
        return 130
    finally:
        release_lock()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
