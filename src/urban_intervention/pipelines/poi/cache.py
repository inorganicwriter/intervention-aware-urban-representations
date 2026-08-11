"""GDB→Parquet cache to avoid repeated FileGDB reads.

FileGDB reads dominate POI pipeline cost (~91% of runtime).  Each batch
re-reads the same 22 nationwide GDBs with a different bbox, so 8 batches
× 22 GDBs = 176 slow reads per year.  This module caches each (year,
source, batch-bbox) slice as a Parquet file; the first batch that touches
a source writes the cache, and subsequent reads hit Parquet instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import pandas as pd

from .config import INTERIM_DIR
from .gdb import ExtractedGdbSource

CACHE_DIR = INTERIM_DIR / "parquet_cache"
META_SUFFIX = ".meta.json"


def _bbox_key(bbox: tuple[float, float, float, float]) -> str:
    """Stable short key from a bbox tuple."""
    return "_".join(f"{v:.4f}" for v in bbox)


def _cache_key(
    source: ExtractedGdbSource,
    year: int,
    bbox: tuple[float, float, float, float],
    max_rows: int | None,
) -> str:
    """Stable cache key from source attributes + bbox."""
    raw = json.dumps(
        {
            "year": year,
            "path": str(source.path),
            "category": source.category or "",
            "layer": source.layer or "",
            "is_nested": source.is_nested,
            "bbox": list(bbox),
            "max_rows": max_rows,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _cache_paths(
    source: ExtractedGdbSource,
    year: int,
    bbox: tuple[float, float, float, float],
    max_rows: int | None = None,
):
    """Return (parquet_path, meta_path) for a source + bbox."""
    key = _cache_key(source, year, bbox, max_rows)
    year_dir = CACHE_DIR / str(year)
    cat = (source.category or "unknown").replace("/", "_")
    name = f"{cat}_{key}.parquet"
    return year_dir / name, year_dir / (name + META_SUFFIX)


def _gdb_mtime(source: ExtractedGdbSource) -> float:
    """Latest modification time of any file inside a GDB directory."""
    path = Path(source.path)
    if path.is_dir():
        files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
        if not files:
            raise ValueError(f"FileGDB directory contains no files: {path}")
        return max(candidate.stat().st_mtime for candidate in files)
    return path.stat().st_mtime


def _is_cache_valid(
    source: ExtractedGdbSource,
    year: int,
    bbox: tuple[float, float, float, float],
    max_rows: int | None = None,
) -> bool:
    """Check if a valid cache exists for this source + bbox."""
    pq_path, meta_path = _cache_paths(source, year, bbox, max_rows)
    if not pq_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return meta.get("gdb_mtime") == _gdb_mtime(source)


def _write_cache(
    source: ExtractedGdbSource, year: int, bbox, gdf, crs_method: str, max_rows: int | None = None
):
    """Persist a GeoDataFrame to parquet cache.

    The parquet is written to a unique temporary name and atomically replaced,
    with the meta file written last, so concurrent readers never observe a
    half-written parquet and concurrent writers cannot corrupt each other.
    """
    pq_path, meta_path = _cache_paths(source, year, bbox, max_rows)
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pq_path.with_name(f".{pq_path.name}.{uuid.uuid4().hex}.tmp")
    gdf.to_parquet(temporary, index=False)
    os.replace(temporary, pq_path)
    meta = {
        "year": year,
        "source_path": str(source.path),
        "category": source.category or "",
        "layer": source.layer or "",
        "bbox": list(bbox),
        "max_rows": max_rows,
        "gdb_mtime": _gdb_mtime(source),
        "crs_method": crs_method,
        "row_count": len(gdf),
        "created_at": time.time(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def read_source_cached(
    source: ExtractedGdbSource,
    year: int,
    city_label: str,
    bbox: tuple[float, float, float, float],
    refresh: bool = False,
    max_rows: int | None = None,
):
    """Read a GDB source, using parquet cache when available.

    Returns (GeoDataFrame, label, crs_method).  The GeoDataFrame is the
    same normalized output as :func:`poi.normalize.read_filegdb`, already
    filtered to *bbox*.
    """
    from .normalize import read_filegdb

    if refresh or not _is_cache_valid(source, year, bbox, max_rows):
        t0 = time.time()
        gdf, crs_method = read_filegdb(
            source.path,
            year,
            city_label,
            bbox=bbox,
            max_rows=max_rows,
            category_override=source.category,
            infer_category_from_fields=(year == 2020),
            layer=source.layer,
        )
        t1 = time.time()
        _write_cache(source, year, bbox, gdf, crs_method, max_rows)
        cat = source.category or source.path.name
        print(
            f"  [cache] WROTE {cat} ({len(gdf):,} rows, read={t1 - t0:.1f}s)",
            flush=True,
        )
    else:
        pq_path, meta_path = _cache_paths(source, year, bbox, max_rows)
        t0 = time.time()
        import geopandas as gpd

        gdf = gpd.read_parquet(pq_path)
        t1 = time.time()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        crs_method = meta.get("crs_method", "cached")
        cat = source.category or source.path.name
        print(
            f"  [cache] HIT {cat} ({len(gdf):,} rows, read={t1 - t0:.1f}s)",
            flush=True,
        )

    return gdf, str(source.path), crs_method


def cache_status(year: int) -> pd.DataFrame:
    """Return a DataFrame summarizing cache status for a year."""
    year_dir = CACHE_DIR / str(year)
    rows = []
    if year_dir.exists():
        for pq in year_dir.glob("*.parquet"):
            meta_path = pq.with_name(pq.name + META_SUFFIX)
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append(
                {
                    "year": year,
                    "category": meta.get("category", ""),
                    "bbox": str(meta.get("bbox", [])),
                    "row_count": meta.get("row_count", 0),
                    "size_mb": pq.stat().st_size / 1e6,
                    "valid": True,
                }
            )
    return pd.DataFrame(rows)
