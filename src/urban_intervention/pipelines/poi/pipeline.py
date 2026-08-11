"""High-level POI panel building workflow."""

from pathlib import Path

import pandas as pd

from .aggregate import aggregate_chunk, finalize_year, load_city_grid
from .config import CSV_YEARS
from .gdb import discover_extracted_gdb_sources
from .normalize import normalize_csv_poi_chunk, read_filegdb
from .sources import (
    choose_archive_tool,
    classify_poi_source,
    extract_gdb_zip_to_temp,
    find_gdb_sources,
    open_city_csv,
)


def iter_gdb_normalized(
    city_key: str,
    year: int,
    categories: set[str] | None,
    max_rows: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
):
    extracted_sources = discover_extracted_gdb_sources(year=year, categories=categories)
    if extracted_sources:
        for source in extracted_sources:
            df, crs_method = read_filegdb(
                source.path,
                year,
                city_key,
                max_rows=max_rows,
                bbox=bbox,
                category_override=source.category,
                infer_category_from_fields=(year == 2020),
                layer=source.layer,
            )
            yield df, str(source.path), crs_method
        return

    archive_sources: list[Path] = find_gdb_sources(year, categories)
    if not archive_sources:
        raise FileNotFoundError(
            f"No GDB POI sources found for year {year}, categories={categories}"
        )
    for gdb_source in archive_sources:
        source_type = classify_poi_source(str(gdb_source))
        if source_type == "filegdb_dir":
            df, crs_method = read_filegdb(gdb_source, year, city_key, max_rows=max_rows, bbox=bbox)
            yield df, str(gdb_source), crs_method
        elif source_type == "filegdb_zip":
            with extract_gdb_zip_to_temp(gdb_source) as (gdb_path, _tmpdir):
                df, crs_method = read_filegdb(
                    gdb_path, year, city_key, max_rows=max_rows, bbox=bbox
                )
                yield df, str(gdb_source), crs_method
        elif source_type == "filegdb_rar":
            tool = choose_archive_tool()
            raise RuntimeError(
                f"Cannot process {gdb_source}: RAR extraction requires 7z/7za/unrar; detected tool={tool!r}"
            )


def build_city_year(
    city_key: str,
    year: int,
    chunksize: int = 200_000,
    max_rows: int | None = None,
    categories: set[str] | None = None,
) -> pd.DataFrame:
    print(f"{city_key} {year}: loading grids", flush=True)
    grids = load_city_grid(city_key)
    parts: list[pd.DataFrame] = []

    if year in CSV_YEARS:
        with open_city_csv(city_key, year) as (fh, label):
            print(f"{city_key} {year}: reading {label}", flush=True)
            reader = pd.read_csv(
                fh,
                encoding="utf-8-sig",
                chunksize=chunksize,
                dtype={"typecode": "string"},
            )
            seen = 0
            for chunk in reader:
                if max_rows is not None and seen >= max_rows:
                    break
                if max_rows is not None:
                    chunk = chunk.head(max_rows - seen)
                seen += len(chunk)
                chunk = normalize_csv_poi_chunk(chunk, year=year, city_key=city_key)
                if categories:
                    chunk = chunk[
                        chunk["cate_A"].isin(categories) | chunk["category"].isin(categories)
                    ]
                part = aggregate_chunk(chunk, grids)
                if not part.empty:
                    parts.append(part)
                print(f"  processed {seen:,} rows; matched parts={len(parts)}", flush=True)
    else:
        x_min, y_min, x_max, y_max = (float(v) for v in grids.total_bounds)
        bbox: tuple[float, float, float, float] = (x_min, y_min, x_max, y_max)
        for chunk, label, crs_method in iter_gdb_normalized(
            city_key, year, categories, max_rows=max_rows, bbox=bbox
        ):
            print(
                f"{city_key} {year}: reading {label} ({len(chunk):,} rows, crs={crs_method})",
                flush=True,
            )
            part = aggregate_chunk(chunk, grids)
            if not part.empty:
                parts.append(part)
            print(f"  matched parts={len(parts)}", flush=True)

    result = finalize_year(city_key, year, parts)
    print(f"{city_key} {year}: {len(result):,} grid rows with POIs", flush=True)
    return result


def parse_years(value: str) -> list[int]:
    years: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            if a > b:
                raise ValueError(f"Reversed year range: {part!r} (expected start <= end)")
            years.extend(range(a, b + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def parse_categories(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {p.strip() for p in value.split(",") if p.strip()}
