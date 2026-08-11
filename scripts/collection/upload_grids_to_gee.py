"""Upload per-city 500m grid GeoJSON assets to Earth Engine for
``sampleRegions``-based GEE exports (Sentinel-2 NDVI/NDBI, WorldPop POP).

The JS export scripts (``gee_sentinel2_export.js``) sample GEE imagery with
``sampleRegions()`` against per-city grid FeatureCollections so that
``grid_id`` travels inside the exported CSV.  The previous asset folder
``projects/macro-city-engine/assets/mit_grids_v2`` was removed; this script
recreates it from the local immutable grid GeoJSON files.

Two phases (fast for 44 cities):

1. Upload all GeoJSONs (zipped as ESRI Shapefile, required by GEE table
   ingestion) to the project GCS staging bucket, then submit every
   ingestion request without waiting for any of them.
2. Poll all asset IDs until each is queryable.

Usage:
    python scripts/collection/upload_grids_to_gee.py --city beijing
    python scripts/collection/upload_grids_to_gee.py --city all
    python scripts/collection/upload_grids_to_gee.py --city all --submit-only
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import ee
from google.cloud import storage

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "scripts"))
from urban_intervention.config.project import ACTIVE_CITIES, GRID_DIR  # noqa: E402

PROJECT = "macro-city-engine"
BUCKET = "macro-city-engine-mit-summer-gee-staging"
ASSET_ROOT = f"projects/{PROJECT}/assets/mit_grids_v2"
GCS_PREFIX = "mit_grids_geojson"


def make_shp_zip(geojson_path: Path) -> bytes:
    """Convert a grid GeoJSON into a zipped ESRI Shapefile (GEE table
    ingestion accepts only .shp/.csv/.tfrecord primary files)."""
    import geopandas as gpd

    gdf = gpd.read_file(geojson_path)
    buf = io.BytesIO()
    shp_dir = Path("grids_shp")
    shp_dir.mkdir(exist_ok=True)
    gdf.to_file(shp_dir / "grids.shp", driver="ESRI Shapefile")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for part in shp_dir.iterdir():
            zf.write(part, f"grids{part.suffix}")
    for part in shp_dir.iterdir():
        part.unlink()
    shp_dir.rmdir()
    return buf.getvalue()


def submit_city(client: storage.Client, city: str) -> None:
    geojson_path = GRID_DIR / city / f"{city}_grids.geojson"
    if not geojson_path.exists():
        raise FileNotFoundError(f"{geojson_path} missing")
    asset_id = f"{ASSET_ROOT}/{city}"
    blob_name = f"{GCS_PREFIX}/{city}_grids.zip"
    blob = client.bucket(BUCKET).blob(blob_name)
    if not blob.exists():
        zip_bytes = make_shp_zip(geojson_path)
        blob.upload_from_string(zip_bytes, content_type="application/zip")
    print(f"  [submitted] {city}")
    request_id = ee.data.newTaskId()[0]
    ee.data.startTableIngestion(
        request_id,
        {
            "name": asset_id,
            "sources": [{"uris": [f"gs://{BUCKET}/{blob_name}"]}],
        },
        allow_overwrite=True,
    )


def poll_assets(cities: list[str], timeout_min: float = 30) -> None:
    deadline = time.time() + timeout_min * 60
    pending = set(cities)
    while pending and time.time() < deadline:
        for city in list(pending):
            asset_id = f"{ASSET_ROOT}/{city}"
            try:
                n = ee.FeatureCollection(asset_id).size().getInfo()
                print(f"  [ready] {city}: {n} features")
                pending.discard(city)
            except ee.ee_exception.EEException:
                continue
        if pending:
            time.sleep(10)
    if pending:
        print(f"  [WARN] still pending after {timeout_min} min: {sorted(pending)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="all", help="City key or 'all'")
    parser.add_argument(
        "--submit-only", action="store_true", help="Upload + submit ingestions but do not poll"
    )
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    try:
        ee.data.getAsset(ASSET_ROOT)
    except ee.ee_exception.EEException:
        print(f"  [folder] creating {ASSET_ROOT}")
        ee.data.createAsset({"type": "Folder"}, ASSET_ROOT)

    cities = ACTIVE_CITIES if args.city == "all" else [args.city]
    client = storage.Client(project=PROJECT, credentials=ee.data.get_persistent_credentials())
    for city in cities:
        submit_city(client, city)
    print(f"\n  submitted {len(cities)} ingestions")
    if not args.submit_only:
        poll_assets(cities)
    return 0


if __name__ == "__main__":
    sys.exit(main())
