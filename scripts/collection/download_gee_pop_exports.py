"""Download the GEE WorldPop sampleRegions exports from GCS.

The GEE exports (``run_gee_sample_regions_exports.py``) used
``reduceRegions`` against the uploaded ``mit_grids_v2`` grid assets, so every
CSV row carries the exact ``grid_id`` with one aggregated row per grid per
year.  Raw CSVs are preserved under ``data/archive/staging/gee/pop/`` and are inputs
to ``rebuild_population_panel.py`` (which merges them with R2024B products).

Usage:
    python scripts/collection/download_gee_pop_exports.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import ee
from google.cloud import storage

BASE_DIR = Path(__file__).resolve().parents[2]

PROJECT = "macro-city-engine"
BUCKET = "macro-city-engine-mit-summer-gee-staging"
GCS_PREFIX = "MIT_SUMMER_GEE_STAGING/pop"
STAGING_DIR = BASE_DIR / "data" / "archive" / "staging" / "gee" / "pop"


def main() -> int:
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
