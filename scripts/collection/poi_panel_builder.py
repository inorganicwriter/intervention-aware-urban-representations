"""Command line entry point for building grid-year POI features.

The implementation is split across ``scripts/collection/poi/``:

* ``sources`` discovers CSV/FileGDB assets without modifying the raw files.
* ``normalize`` harmonizes schemas and CRS to WGS84.
* ``taxonomy`` maps Amap categories into analysis categories.
* ``aggregate`` joins POIs to grids and writes city panels.
* ``pipeline`` orchestrates city-year processing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from urban_intervention.config.project import ACTIVE_CITIES, CITIES
from urban_intervention.pipelines.poi.aggregate import save_city_panel
from urban_intervention.pipelines.poi.normalize import (
    normalize_crs_to_wgs84,
    normalize_csv_poi_chunk,
)
from urban_intervention.pipelines.poi.pipeline import build_city_year, parse_categories, parse_years
from urban_intervention.pipelines.poi.sources import (
    category_from_gdb_path,
    classify_poi_source,
    fix_zip_name,
    write_inventory,
)
from urban_intervention.pipelines.poi.taxonomy import (
    is_chain_brand,
    map_poi_category,
    shannon_entropy,
)

__all__ = [
    "build_city_year",
    "category_from_gdb_path",
    "classify_poi_source",
    "fix_zip_name",
    "is_chain_brand",
    "map_poi_category",
    "normalize_crs_to_wgs84",
    "normalize_csv_poi_chunk",
    "parse_categories",
    "parse_years",
    "save_city_panel",
    "shannon_entropy",
    "write_inventory",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grid-year POI panels from Amap POI assets.")
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Write asset inventory CSV and exit if no city is given.",
    )
    parser.add_argument(
        "--city", default=None, help="City key, comma-separated city keys, or 'all'."
    )
    parser.add_argument("--years", default="2012", help="Year list/ranges, e.g. '2012,2014-2017'.")
    parser.add_argument(
        "--categories",
        default=None,
        help="Optional comma-separated source or analysis categories, e.g. '餐饮服务,shopping,food'.",
    )
    parser.add_argument("--chunksize", type=int, default=200_000, help="Rows per CSV chunk.")
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Debug-only cap per city-year source."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print summaries without saving parquet files.",
    )
    return parser.parse_args(argv)


def resolve_cities(value: str | None) -> list[str]:
    if value is None:
        return []
    if value == "all":
        return list(ACTIVE_CITIES)
    cities = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [city for city in cities if city not in CITIES]
    if unknown:
        raise KeyError(f"Unknown city key(s): {unknown}")
    return cities


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.inventory:
        write_inventory()

    cities = resolve_cities(args.city)
    if not cities:
        if not args.inventory:
            write_inventory()
        return 0

    years = parse_years(args.years)
    categories = parse_categories(args.categories)
    for city_key in cities:
        frames = [
            build_city_year(
                city_key,
                year,
                chunksize=args.chunksize,
                max_rows=args.max_rows,
                categories=categories,
            )
            for year in years
        ]
        if args.dry_run:
            rows = sum(len(frame) for frame in frames)
            print(f"{city_key}: dry run built {rows:,} grid-year rows; not saving")
        else:
            save_city_panel(city_key, frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
