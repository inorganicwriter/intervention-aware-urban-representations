"""Build 2018+ POI panels by reading nationwide FileGDB assets in city batches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from urban_intervention.config.project import ACTIVE_CITIES, CITIES
from urban_intervention.pipelines.poi.batch import (
    build_batch_year,
    make_city_batches,
    save_batch_year,
)
from urban_intervention.pipelines.poi.cache import cache_status
from urban_intervention.pipelines.poi.pipeline import parse_categories, parse_years


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build grid-year POI panels from nationwide Amap FileGDB assets in spatial city batches."
    )
    parser.add_argument(
        "--city", default="all", help="City key, comma-separated city keys, or 'all'."
    )
    parser.add_argument(
        "--years", default="2018-2024", help="Year list/ranges, e.g. '2018,2020-2024'."
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Optional comma-separated source or analysis categories, e.g. '餐饮服务,购物服务'.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=6, help="Number of cities per spatial batch."
    )
    parser.add_argument(
        "--batch-index", type=int, default=None, help="Run only one 1-based batch index."
    )
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Debug-only cap per FileGDB source/layer."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of threads for parallel GDB processing (default: auto).",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable parquet cache; read GDBs directly."
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild parquet cache for the given years before processing.",
    )
    parser.add_argument(
        "--cache-status",
        action="store_true",
        help="Print cache status for the given years and exit.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Build summaries without saving parquet files."
    )
    return parser.parse_args(argv)


def resolve_cities(value: str) -> list[str]:
    if value == "all":
        return list(ACTIVE_CITIES)
    cities = list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = [city for city in cities if city not in CITIES]
    if unknown:
        raise KeyError(f"Unknown city key(s): {unknown}")
    return cities


def validate_years(years: list[int]) -> None:
    unsupported = [year for year in years if year < 2018]
    if unsupported:
        raise ValueError(
            "poi_batch_panel_builder.py is only for 2018+ FileGDB assets; "
            f"unsupported year(s): {unsupported}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cities = resolve_cities(args.city)
    years = parse_years(args.years)
    validate_years(years)
    categories = parse_categories(args.categories)

    if args.cache_status:
        for year in years:
            status = cache_status(year)
            print(f"\n=== Cache status for {year} ===")
            if status.empty:
                print("  No sources found.")
            else:
                valid = status["valid"].sum()
                print(f"  Valid: {valid}/{len(status)}")
                print(status[["category", "valid", "size_mb"]].to_string(index=False))
        return 0

    use_cache = not args.no_cache

    if use_cache and args.refresh_cache:
        print("--refresh-cache: caches will be rebuilt as batches run.", flush=True)

    batches = make_city_batches(cities, args.batch_size)
    if args.batch_index is not None:
        batches = [batch for batch in batches if batch.index == args.batch_index]
        if not batches:
            raise IndexError(f"No batch with index {args.batch_index}")

    cache_mode = "cached" if use_cache else "direct"
    print(
        f"Running {len(batches)} city batch(es), years={years}, "
        f"batch_size={args.batch_size}, mode={cache_mode}, dry_run={args.dry_run}",
        flush=True,
    )
    for batch in batches:
        print(
            f"{batch.label}: cities={','.join(batch.cities)} config_bbox={batch.bbox}", flush=True
        )
        for year in years:
            panel = build_batch_year(
                batch,
                year,
                categories=categories,
                max_rows=args.max_rows,
                workers=args.workers,
                use_cache=use_cache,
            )
            if args.dry_run:
                city_count = panel["city"].nunique() if not panel.empty else 0
                print(
                    f"{batch.label} {year}: dry run built {len(panel):,} rows "
                    f"for {city_count} city/cities; not saving",
                    flush=True,
                )
            else:
                saved = save_batch_year(panel)
                print(f"{batch.label} {year}: saved {len(saved)} city panel file(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
