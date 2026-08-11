"""Pipeline: GEE VIIRS + NDVI/NDBI extraction via Python API (no Code Editor).

Extracts annual VIIRS nightlight radiance and NDVI/NDBI indices for all
44 cities at 500m resolution.  Uses ``ee.Image.sample()`` (no asset
upload needed) which produces pixel-centroid points with lat/lon.  A
downstream spatial join matches each point to the nearest grid centroid.

VIIRS:   NOAA monthly cloud-free composite, 2014-2024 (11 years)
NDVI/NDBI: Landsat 8 (2014-2017) + Sentinel-2 (2018-2024), 11 years

Usage:
    python scripts/collection/run_gee_extraction.py --source viirs --city beijing --year 2023
    python scripts/collection/run_gee_extraction.py --source viirs --city all
    python scripts/collection/run_gee_extraction.py --source sentinel --city all
"""

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
import ee

from urban_intervention.config.project import ACTIVE_CITIES, CITIES

# ── City bboxes (admin boundary + 10km) ───────────────────────────
# Synced from admin_boundary_fetcher.py --update-config output.
CITY_BBOXES = {
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

VIIRS_YEARS = list(range(2014, 2025))  # NOAA monthly product starts 2014-01
S2_YEARS = list(range(2014, 2025))  # L8 2014-2017 + S2 2018-2024
# Legacy WorldPop (WorldPop/GP/100m/pop, 2000-2020) — 2014-2020. 2019-2020
# overlap with R2024B (downloaded separately) for calibration. Years before
# 2014 are not re-downloaded: they were dropped by time constraint and most
# early-opening grids route to GSC/MC anyway for lack of pre-treatment data.
POP_YEARS = list(range(2014, 2021))


def _bbox(city_key: str) -> ee.Geometry.Rectangle:
    b = CITY_BBOXES[city_key]
    return ee.Geometry.Rectangle([b[0], b[1], b[2], b[3]])


# ── VIIRS ────────────────────────────────────────────────────────


def extract_viirs(city_key: str, year: int) -> ee.FeatureCollection:
    """Annual mean VIIRS nightlight radiance at 500m (NOAA public product)."""
    bbox = _bbox(city_key)
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year + 1, 1, 1)

    coll = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate(start, end)
        .filterBounds(bbox)
        .select("avg_rad")
    )
    annual = coll.mean().rename("avg_rad").clip(bbox)

    pts = ee.FeatureCollection(
        annual.sample(region=bbox, scale=500, projection="EPSG:4326", geometries=True, tileScale=4)
    )
    return pts.map(
        lambda f: f.set(
            {
                "city": city_key,
                "year": year,
                "longitude": f.geometry().coordinates().get(0),
                "latitude": f.geometry().coordinates().get(1),
            }
        )
    )


# ── NDVI / NDBI ──────────────────────────────────────────────────


def extract_sentinel2(city_key: str, year: int) -> ee.FeatureCollection:
    """Annual median NDVI + NDBI at 500m.

    2014-2017 → Landsat 8 (30m)
    2018-2024 → Sentinel-2 (10m)

    Both sensors use the same spectral regions (NIR ~0.86µm, Red ~0.66µm,
    SWIR1 ~1.61µm) so NDVI/NDBI are comparable at 500m aggregation.
    """
    bbox = _bbox(city_key)
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year + 1, 1, 1)

    if year >= 2018:
        # ── Sentinel-2 ─────────────────────────────────────────
        coll = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start, end)
            .filterBounds(bbox)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .select(["B2", "B3", "B4", "B8", "B11", "B12", "SCL"])
        )

        def mask_img(img):
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

        coll = coll.map(mask_img).map(add_idx).select(["NDVI", "NDBI"])
        tile = 8
    else:
        # ── Landsat 8 ─────────────────────────────────────────
        # SR_B4=Red, SR_B5=NIR, SR_B6=SWIR1
        coll = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(start, end)
            .filterBounds(bbox)
            .select(["SR_B4", "SR_B5", "SR_B6", "QA_PIXEL"])
        )

        def mask_img(img):
            qa = img.select("QA_PIXEL")
            # Bit 3=cloud, 4=shadow, 5=snow — aligns with S2 SCL mask
            m = (
                qa.bitwiseAnd(1 << 3)
                .eq(0)
                .And(qa.bitwiseAnd(1 << 4).eq(0))
                .And(qa.bitwiseAnd(1 << 5).eq(0))
            )
            return img.updateMask(m)

        def add_idx(img):
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

        coll = coll.map(mask_img).map(add_idx).select(["NDVI", "NDBI"])
        tile = 4  # 30m native, lighter than S2

    annual = coll.median().clip(bbox)
    pts = ee.FeatureCollection(
        annual.sample(
            region=bbox, scale=500, projection="EPSG:4326", geometries=True, tileScale=tile
        )
    )
    return pts.map(
        lambda f: f.set(
            {
                "city": city_key,
                "year": year,
                "longitude": f.geometry().coordinates().get(0),
                "latitude": f.geometry().coordinates().get(1),
            }
        )
    )


# ── Population ───────────────────────────────────────────────────


def extract_population(city_key: str, year: int) -> ee.FeatureCollection:
    """Population density at 500m from WorldPop official dataset (2000-2020).

    Re-extracting 2019-2020 from old source for calibration against R2024B.
    """
    bbox = _bbox(city_key)
    img = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filterDate(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year + 1, 1, 1))
        .filterBounds(bbox)
        .mosaic()
        .select("population")
        .rename("pop_count")
        .clip(bbox)
    )
    img = img.unmask(0)

    pts = ee.FeatureCollection(
        img.sample(region=bbox, scale=500, projection="EPSG:4326", geometries=True, tileScale=4)
    )
    return pts.map(
        lambda f: f.set(
            {
                "city": city_key,
                "year": year,
                "pop_year": year,
                "longitude": f.geometry().coordinates().get(0),
                "latitude": f.geometry().coordinates().get(1),
            }
        )
    )


# ── Export ───────────────────────────────────────────────────────


def export_city_year(source: str, city_key: str, year: int):
    """Queue one city-year export to Google Drive."""
    if source == "viirs":
        fc = extract_viirs(city_key, year)
        selectors = ["city", "year", "avg_rad", "latitude", "longitude"]
        folder = "MIT_Summer_VIIRS"
        desc = f"viirs_{city_key}_{year}"
    elif source == "population":
        fc = extract_population(city_key, year)
        selectors = ["city", "year", "pop_year", "pop_count", "latitude", "longitude"]
        folder = "MIT_Summer_POP"
        desc = f"pop_{city_key}_{year}"
    else:
        fc = extract_sentinel2(city_key, year)
        selectors = ["city", "year", "NDVI", "NDBI", "latitude", "longitude"]
        folder = "MIT_Summer_S2"
        desc = f"s2_{city_key}_{year}"

    task = ee.batch.Export.table.toDrive(
        collection=fc, description=desc, folder=folder, fileFormat="CSV", selectors=selectors
    )
    task.start()


def main():
    parser = argparse.ArgumentParser(description="GEE extraction pipeline")
    parser.add_argument("--source", required=True, choices=["viirs", "sentinel", "population"])
    parser.add_argument("--city", default="all")
    parser.add_argument(
        "--year", type=int, default=None, help="Single year (for testing). Omit for full range."
    )
    args = parser.parse_args()

    ee.Initialize(project="macro-city-engine")

    years = (
        VIIRS_YEARS
        if args.source == "viirs"
        else POP_YEARS
        if args.source == "population"
        else S2_YEARS
    )
    if args.year is not None:
        years = [args.year]

    cities = ACTIVE_CITIES if args.city == "all" else [args.city]

    total = len(cities) * len(years)
    print(f"{args.source}: {len(cities)} cities × {len(years)} years = {total} tasks\n")

    count = 0
    for ck in cities:
        if ck not in CITIES:
            continue
        for yr in years:
            export_city_year(args.source, ck, yr)
            count += 1
            if count % 50 == 0 or count == 1:
                print(f"  [{count}/{total}] Queued: {ck} {yr}")
            time.sleep(0.5)

    print(f"\nDone. {count} tasks queued.")
    print("Check https://code.earthengine.google.com/tasks")
    folder_names = {"viirs": "VIIRS", "sentinel": "S2", "population": "POP"}
    folder = folder_names.get(args.source, args.source.upper())
    print(f"Results → Google Drive: MIT_Summer_{folder}/")


if __name__ == "__main__":
    main()
