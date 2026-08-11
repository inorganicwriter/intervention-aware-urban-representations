"""Batch-oriented POI panel building for nationwide FileGDB assets."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from urban_intervention.config.project import CITIES

from .aggregate import (
    aggregate_chunk_multi_city,
    finalize_multi_city_year,
    load_city_grids,
    save_city_panel,
)
from .gdb import discover_extracted_gdb_sources


@dataclass(frozen=True)
class CityBatch:
    index: int
    cities: tuple[str, ...]
    bbox: tuple[float, float, float, float]

    @property
    def label(self) -> str:
        return f"batch-{self.index:02d}"


def _city_sort_key(city_key: str) -> tuple[float, float, str]:
    cfg = CITIES[city_key]
    return (float(cfg["center_lon"]), float(cfg["center_lat"]), city_key)


def _bbox_for_cities(city_keys: list[str] | tuple[str, ...]) -> tuple[float, float, float, float]:
    boxes = [CITIES[city]["bbox"] for city in city_keys]
    return (
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    )


def make_city_batches(city_keys: list[str], batch_size: int) -> list[CityBatch]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    ordered = sorted(city_keys, key=_city_sort_key)
    batches: list[CityBatch] = []
    for offset in range(0, len(ordered), batch_size):
        cities = tuple(ordered[offset : offset + batch_size])
        batches.append(
            CityBatch(
                index=len(batches) + 1,
                cities=cities,
                bbox=_bbox_for_cities(cities),
            )
        )
    return batches


def _process_gdb(
    source,
    year: int,
    city_label: str,
    bbox,
    grids,
    use_cache: bool = True,
    max_rows: int | None = None,
):
    if use_cache:
        from .cache import read_source_cached

        df, _, crs_method = read_source_cached(
            source, year, city_label, bbox=bbox, max_rows=max_rows
        )
    else:
        from .normalize import read_filegdb

        df, crs_method = read_filegdb(
            source.path,
            year,
            city_label,
            bbox=bbox,
            max_rows=max_rows,
            category_override=source.category,
            infer_category_from_fields=(year == 2020),
            layer=source.layer,
        )
    part = aggregate_chunk_multi_city(df, grids)
    return part, str(source.path), crs_method, len(df)


def build_batch_year(
    batch: CityBatch,
    year: int,
    categories: set[str] | None = None,
    max_rows: int | None = None,
    workers: int | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    if workers is None:
        workers = min(6, (os.cpu_count() or 4))

    print(f"{batch.label} {year}: loading grids for {','.join(batch.cities)}", flush=True)
    grids = load_city_grids(list(batch.cities))
    read_bbox = tuple(float(value) for value in grids.total_bounds)
    print(f"{batch.label} {year}: using grid read_bbox={read_bbox}", flush=True)

    sources = discover_extracted_gdb_sources(year=year, categories=categories)
    if not sources:
        raise FileNotFoundError(f"No GDB sources found for year {year}")

    mode = "cached" if use_cache else "direct"
    print(
        f"{batch.label} {year}: processing {len(sources)} GDBs with {workers} workers ({mode})",
        flush=True,
    )
    parts: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_gdb,
                source,
                year,
                batch.label,
                read_bbox,
                grids,
                use_cache,
                max_rows,
            ): source
            for source in sources
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                part, label, crs_method, n_rows = future.result()
                print(
                    f"{batch.label} {year}: reading {label} ({n_rows:,} rows, crs={crs_method})",
                    flush=True,
                )
                if not part.empty:
                    parts.append(part)
            except Exception as exc:
                print(f"{batch.label} {year}: FAILED {source.path}: {exc}", flush=True)
        print(f"{batch.label} {year}: matched parts={len(parts)}/{len(sources)}", flush=True)

    result = finalize_multi_city_year(year, parts)
    print(f"{batch.label} {year}: {len(result):,} city-grid rows with POIs", flush=True)
    return result


def save_batch_year(panel: pd.DataFrame) -> list[Path]:
    saved: list[Path] = []
    if panel.empty:
        return saved
    for city_key, city_frame in panel.groupby("city"):
        path = save_city_panel(str(city_key), [city_frame.copy()])
        if path is not None:
            saved.append(path)
    return saved
