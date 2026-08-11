"""Queue Earth Engine sampleRegions exports for Sentinel-2 NDVI/NDBI and
WorldPop population, using the uploaded per-city grid assets so that
``grid_id`` is carried into every exported CSV.

Root cause: the legacy GEE exports used ``Image.sample()`` whose internal
pixel lattice does not align with the project's 500m UTM grid, forcing a
lossy KD-Tree join downstream and producing duplicate/conflicting grid-year
rows.  ``sampleRegions()`` against ``mit_grids_v2/{city}`` keeps one row
per grid feature with the exact ``grid_id``, eliminating the duplicate
class entirely.

Sources:
  - s2: 2014-2017 Landsat 8 (L2) + 2018-2024 Sentinel-2 (SR harmonized),
         annual median NDVI/NDBI, cloud+snow masked (same recipe as
         ``gee_sentinel2_export.js``).
  - pop: legacy WorldPop (``WorldPop/GP/100m/pop``, 2000-2020) annual
         population, 2014-2020.  The 2019-2020 overlap with R2024B is used
         for calibration; 2021-2024 come from R2024B rasters.

Exports land in the GCS bucket (``MIT_SUMMER_GEE_STAGING`` prefix), which the
project deployment already authorizes for the macro-city-engine project.

Usage:
    python scripts/collection/run_gee_sample_regions_exports.py --source s2 --city all
    python scripts/collection/run_gee_sample_regions_exports.py --source pop --city all
    python scripts/collection/run_gee_sample_regions_exports.py --source s2 --city beijing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ee

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "scripts"))
from urban_intervention.config.project import ACTIVE_CITIES, CITIES  # noqa: E402

PROJECT = "macro-city-engine"
BUCKET = "macro-city-engine-mit-summer-gee-staging"
ASSET_ROOT = f"projects/{PROJECT}/assets/mit_grids_v2"
OUT_PREFIX = "MIT_SUMMER_GEE_STAGING"

S2_YEARS = list(range(2014, 2025))
POP_YEARS = list(range(2010, 2021))


def _grid_asset(city: str) -> ee.FeatureCollection:
    return ee.FeatureCollection(f"{ASSET_ROOT}/{city}")


def extract_s2(city: str, year: int) -> ee.FeatureCollection:
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year + 1, 1, 1)
    if year >= 2018:
        coll = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start, end)
            .filterBounds(_city_bbox(city))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .select(["B2", "B3", "B4", "B8", "B11", "B12", "SCL"])
        )

        def mask_s2(img):
            scl = img.select("SCL")
            m = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
            return img.updateMask(m)

        def add_idx(img):
            return img.addBands(
                [
                    img.normalizedDifference(["B8", "B4"]).rename("NDVI"),
                    img.normalizedDifference(["B11", "B8"]).rename("NDBI"),
                ]
            )

        coll = coll.map(mask_s2).map(add_idx).select(["NDVI", "NDBI"])
        tile = 8
    else:
        coll = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(start, end)
            .filterBounds(_city_bbox(city))
            .select(["SR_B4", "SR_B5", "SR_B6", "QA_PIXEL"])
        )

        def mask_l8(img):
            qa = img.select("QA_PIXEL")
            m = (
                qa.bitwiseAnd(1 << 3)
                .eq(0)
                .And(qa.bitwiseAnd(1 << 4).eq(0))
                .And(qa.bitwiseAnd(1 << 5).eq(0))
            )
            return img.updateMask(m)

        def add_idx_l8(img):
            ndvi = (
                img.select("SR_B5")
                .subtract(img.select("SR_B4"))
                .divide(img.select("SR_B5").add(img.select("SR_B4")))
                .rename("NDVI")
            )
            ndbi = (
                img.select("SR_B6")
                .subtract(img.select("SR_B5"))
                .divide(img.select("SR_B6").add(img.select("SR_B5")))
                .rename("NDBI")
            )
            return img.addBands([ndvi, ndbi])

        coll = coll.map(mask_l8).map(add_idx_l8).select(["NDVI", "NDBI"])
        tile = 4

    annual = coll.median()
    fc = annual.reduceRegions(
        collection=_grid_asset(city),
        reducer=ee.Reducer.mean(),
        scale=500,
        crs="EPSG:4326",
        tileScale=tile,
    )
    return fc.map(lambda f: f.set({"city": city, "year": year}))


def extract_pop(city: str, year: int) -> ee.FeatureCollection:
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year + 1, 1, 1)
    img = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filterDate(start, end)
        .filterBounds(_city_bbox(city))
        .mosaic()
        .select("population")
        .rename("pop_count")
        .unmask(0)
    )
    fc = img.reduceRegions(
        collection=_grid_asset(city),
        reducer=ee.Reducer.mean(),
        scale=500,
        crs="EPSG:4326",
        tileScale=4,
    )
    return fc.map(lambda f: f.set({"city": city, "year": year, "pop_year": year}))


def _city_bbox(city: str) -> ee.Geometry.Rectangle:
    b = CITIES[city]["bbox"] if "bbox" in CITIES[city] else _default_bbox(city)
    return ee.Geometry.Rectangle(b)


def _default_bbox(city: str) -> list[float]:
    bboxes = {
        "beijing": [115.3, 39.35, 117.63, 41.15],
        "changchun": [123.88, 43.07, 127.21, 45.34],
        "changsha": [111.79, 27.76, 114.36, 28.75],
        "changzhou": [119.03, 31.06, 120.31, 32.15],
        "chengdu": [102.88, 30.0, 105.0, 31.53],
        "chongqing": [105.18, 28.07, 110.3, 32.29],
        "dalian": [120.97, 38.63, 123.63, 40.29],
        "dongguan": [113.42, 22.57, 114.35, 23.23],
        "foshan": [112.29, 22.56, 113.49, 23.67],
        "fuzhou": [118.28, 25.01, 120.82, 26.73],
        "guangzhou": [112.85, 22.47, 114.15, 24.03],
        "guiyang": [106.02, 26.1, 107.38, 27.45],
        "hangzhou": [118.24, 29.1, 120.83, 30.65],
        "harbin": [125.55, 43.97, 130.36, 46.76],
        "hefei": [116.58, 30.86, 118.07, 32.63],
        "hohhot": [110.39, 39.5, 112.42, 41.47],
        "jinan": [116.11, 35.9, 118.09, 37.63],
        "jinhua": [119.11, 28.43, 120.88, 29.77],
        "kunming": [102.07, 24.3, 103.77, 26.64],
        "lanzhou": [102.58, 35.48, 104.69, 37.13],
        "luoyang": [111.02, 33.47, 113.08, 35.16],
        "nanchang": [115.33, 28.07, 116.66, 29.23],
        "nanjing": [118.25, 31.14, 119.35, 32.71],
        "nanning": [107.23, 22.12, 109.72, 24.12],
        "nantong": [120.09, 31.54, 122.49, 32.95],
        "ningbo": [120.79, 28.67, 122.97, 30.54],
        "qingdao": [119.4, 35.36, 121.68, 37.24],
        "shanghai": [120.75, 30.58, 123.33, 31.96],
        "shaoxing": [119.78, 29.14, 121.33, 30.39],
        "shenyang": [122.3, 41.11, 123.93, 43.13],
        "shenzhen": [113.58, 21.73, 114.89, 22.95],
        "shijiazhuang": [113.4, 37.35, 115.59, 38.85],
        "suzhou": [119.81, 30.67, 121.49, 32.14],
        "taiyuan": [111.39, 37.35, 113.27, 38.51],
        "taizhou": [120.18, 27.88, 122.55, 29.44],
        "tianjin": [116.59, 38.46, 118.18, 40.34],
        "urumqi": [86.67, 42.83, 89.1, 45.09],
        "wenzhou": [119.52, 26.95, 121.94, 28.71],
        "wuhan": [113.59, 29.88, 115.18, 31.45],
        "wuxi": [119.41, 31.01, 120.71, 32.08],
        "xiamen": [117.78, 24.29, 118.55, 25.0],
        "xian": [107.55, 33.61, 109.93, 34.83],
        "xuzhou": [116.25, 33.62, 118.78, 35.07],
        "zhengzhou": [112.6, 34.17, 114.31, 35.08],
    }
    return bboxes[city]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=["s2", "pop"])
    parser.add_argument("--city", default="all")
    parser.add_argument(
        "--year", type=int, default=None, help="Single year (testing); omit for full range"
    )
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)

    years = S2_YEARS if args.source == "s2" else POP_YEARS
    if args.year is not None:
        years = [args.year]
    cities = ACTIVE_CITIES if args.city == "all" else [args.city]

    extract = extract_s2 if args.source == "s2" else extract_pop
    folder_label = "S2" if args.source == "s2" else "POP"

    total = len(cities) * len(years)
    print(f"{args.source}: {len(cities)} cities x {len(years)} years = {total} tasks")
    print(f"export -> gs://{BUCKET}/{OUT_PREFIX}/")

    count = 0
    for ck in cities:
        if ck not in CITIES:
            continue
        for yr in years:
            fc = extract(ck, yr)
            task = ee.batch.Export.table.toCloudStorage(
                collection=fc,
                description=f"{args.source}_{ck}_{yr}",
                bucket=BUCKET,
                fileNamePrefix=f"{OUT_PREFIX}/{args.source}/{ck}_{yr}",
                fileFormat="CSV",
            )
            task.start()
            count += 1
            if count % 20 == 0 or count == 1:
                print(f"  [{count}/{total}] queued: {ck} {yr}")
            time.sleep(0.3)

    print(f"\nDone. {count} tasks queued for {folder_label}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
