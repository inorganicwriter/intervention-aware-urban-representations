"""Download the GEE Sentinel-2 sampleRegions exports from GCS and build the
clean curated per-city grid-year panel.

The GEE exports (``run_gee_sample_regions_exports.py``) used
``reduceRegions`` against the uploaded ``mit_grids_v2`` grid assets, so every
CSV row carries the exact ``grid_id`` with one aggregated row per grid per
year.  Raw CSVs are preserved under ``data/archive/staging/gee/s2/``; the curated
panel keeps the same schema as the previous sentinel2 product so downstream
consumers (pretraining dataset, R estimators) need no changes.

Curated output columns: city, grid_id, year, NDVI, NDBI

Usage:
    python scripts/collection/download_gee_s2_exports.py --download-only
    python scripts/collection/download_gee_s2_exports.py --build-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ee
import pandas as pd
from google.cloud import storage

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "scripts"))
from urban_intervention.config.project import ACTIVE_CITIES  # noqa: E402

PROJECT = "macro-city-engine"
BUCKET = "macro-city-engine-mit-summer-gee-staging"
GCS_PREFIX = "MIT_SUMMER_GEE_STAGING/s2"
STAGING_DIR = BASE_DIR / "data" / "archive" / "staging" / "gee" / "s2"
CURATED_DIR = BASE_DIR / "data" / "active" / "curated" / "sentinel2"


def download_all() -> list[Path]:
    ee.Initialize(project=PROJECT)
    client = storage.Client(project=PROJECT, credentials=ee.data.get_persistent_credentials())
    blobs = list(client.bucket(BUCKET).list_blobs(prefix=f"{GCS_PREFIX}/"))
    blobs = [b for b in blobs if b.name.endswith(".csv")]
    print(f"GCS objects: {len(blobs)}")
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for blob in blobs:
        target = STAGING_DIR / Path(blob.name).name
        if target.exists() and target.stat().st_size == blob.size:
            paths.append(target)
            continue
        blob.download_to_filename(target)
        paths.append(target)
    print(f"downloaded/verified {len(paths)} CSVs -> {STAGING_DIR}")
    return paths


def build_city_panel(city: str) -> pd.DataFrame:
    files = sorted(STAGING_DIR.glob(f"{city}_*.csv"))
    if not files:
        return pd.DataFrame(columns=["city", "grid_id", "year", "NDVI", "NDBI"])
    parts = []
    for fp in files:
        df = pd.read_csv(fp, usecols=["grid_id", "year", "NDVI", "NDBI"])
        df = df[df["NDVI"].notna() | df["NDBI"].notna()]
        parts.append(df)
    panel = pd.concat(parts, ignore_index=True)
    panel = panel[panel["grid_id"].notna()]
    dup = int(panel.duplicated(subset=["grid_id", "year"]).sum())
    if dup:
        raise ValueError(f"{city}: {dup} duplicate grid-year rows in S2 staging")
    panel.insert(0, "city", city)
    return panel


def build_all() -> None:
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    for city in ACTIVE_CITIES:
        panel = build_city_panel(city)
        if panel.empty:
            print(f"  [SKIP] {city}: no staging CSVs")
            continue
        target = CURATED_DIR / f"{city}_s2.parquet"
        panel.to_parquet(target, index=False)
        years = f"{panel.year.min()}-{panel.year.max()}"
        print(
            f"  {city}: {len(panel):,} rows, {panel.grid_id.nunique():,} grids, "
            f"years {years} -> {target.name}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    if not args.build_only:
        download_all()
    if not args.download_only:
        build_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
