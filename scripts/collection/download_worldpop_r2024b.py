"""Download WorldPop R2024B (Global 2) annual population rasters and
aggregate them to the project's 500m grids.

Product:   WorldPop Global 2 Population (R2024B), 2015-2030, 100m, UN-adjusted
           (unconstrained).  One GeoTIFF per year covering all of China
           (~1.9 GB each).
Source:    https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/{year}/CHN/v1/100m/unconstrained/
Asset:     chn_pop_{year}_UC_100m_R2024B_v1.tif

Pipeline:
  1. Download the yearly GeoTIFF to ``data/archive/raw/worldpop_r2024b/`` (immutable).
  2. For each city, read only the raster window covering the city bbox,
     project 100m cell centroids into the city UTM zone, and zonal-sum the
     cells whose centroids fall inside each 500m project grid polygon,
     producing ``data/archive/staging/worldpop_r2024b/chn_pop_{year}_grid.parquet``
     with one row per (grid_id, year) and ``source_version='r2024b'``.

The 2019-2020 overlap with the legacy GEE WorldPop series enables
calibration; the curated population panel then carries a ``source_version``
column so the two series can never be mixed silently again.

Usage:
    python scripts/collection/download_worldpop_r2024b.py --years 2019 2020 2021 2022 2023 2024
    python scripts/collection/download_worldpop_r2024b.py --years 2019 2020 --download-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
from shapely.wkt import loads as wkt_loads

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "scripts"))
from urban_intervention.config.project import ACTIVE_CITIES, CITIES, GRID_DIR  # noqa: E402

BASE_URL = "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B"
RAW_DIR = BASE_DIR / "data" / "archive" / "raw" / "worldpop_r2024b"
OUT_DIR = BASE_DIR / "data" / "archive" / "staging" / "worldpop_r2024b"


def download_year(year: int) -> Path:
    url = f"{BASE_URL}/{year}/CHN/v1/100m/unconstrained/chn_pop_{year}_UC_100m_R2024B_v1.tif"
    target = RAW_DIR / f"chn_pop_{year}_UC_100m_R2024B_v1.tif"
    if target.exists():
        print(f"  [cache hit] {target.name}")
        return target
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tif.part")
    print(f"  [download] {year} -> {target.name}  (~1.9 GB)")
    t0 = time.time()
    _download_via_curl(url, tmp)
    tmp.rename(target)
    print(f"    done in {(time.time() - t0) / 60:.1f} min, {target.stat().st_size / 2**30:.2f} GiB")
    return target


def _download_via_curl(url: str, tmp: Path) -> None:
    """Download with the system curl.  The project Python on this machine
    cannot create an SSL context (ASN1: NOT_ENOUGH_DATA), so urllib is not
    usable; curl ships with Windows and uses the OS trust store."""
    import subprocess

    result = subprocess.run(
        ["curl", "-L", "--fail", "--retry", "3", "-o", str(tmp), url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"curl download failed ({result.returncode}): {result.stderr[-500:]}")


def aggregate_city_year(city: str, year: int, tif_path: Path) -> pd.DataFrame:
    grid_path = GRID_DIR / city / f"{city}_grids.parquet"
    grids = pd.read_parquet(grid_path)[["grid_id", "geometry_wkt"]]
    grid_ids = grids["grid_id"].to_numpy()
    polygons = [wkt_loads(wkt) for wkt in grids["geometry_wkt"]]

    min_lon = min(p.bounds[0] for p in polygons)
    min_lat = min(p.bounds[1] for p in polygons)
    max_lon = max(p.bounds[2] for p in polygons)
    max_lat = max(p.bounds[3] for p in polygons)

    epsg = int(CITIES[city]["projected_crs"].split(":")[-1])
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    with rasterio.open(tif_path) as src:
        window = rasterio.windows.from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
        window = window.intersection(src.window(0, 0, src.width, src.height))
        band = src.read(1, window=window)
        win_transform = src.window_transform(window)

    rows, cols = np.indices(band.shape)
    xs, ys = rasterio.transform.xy(win_transform, rows, cols)
    xs = np.asarray(xs).ravel()
    ys = np.asarray(ys).ravel()
    values = band.ravel().astype(np.float64)
    valid = (values > 0) & np.isfinite(values)
    if not valid.any():
        return pd.DataFrame(columns=["city_key", "grid_id", "year", "pop_count", "source_version"])

    cell_x = xs[valid]
    cell_y = ys[valid]
    cell_v = values[valid]
    utm_x, utm_y = to_utm.transform(cell_x, cell_y)

    utm_polys = []
    for poly in polygons:
        px, py = to_utm.transform(
            np.asarray(poly.exterior.coords.xy[0]),
            np.asarray(poly.exterior.coords.xy[1]),
        )
        utm_polys.append(Polygon(list(zip(px, py, strict=True))))
    index = STRtree(utm_polys)

    sums = np.zeros(len(utm_polys), dtype=np.float64)
    for cx, cy, v in zip(utm_x, utm_y, cell_v, strict=True):
        p = Point(cx, cy)
        for i in index.query(p):
            if utm_polys[i].contains(p):
                sums[i] += v
                break

    out = pd.DataFrame(
        {
            "grid_id": grid_ids,
            "pop_count": sums,
        }
    )
    out = out[out["pop_count"] > 0]
    out["city_key"] = city
    out["year"] = year
    out["source_version"] = "r2024b"
    return out


def process_year(year: int, cities: list[str], download_only: bool) -> None:
    tif = download_year(year)
    if download_only:
        return
    parts = []
    for city in cities:
        if city not in CITIES:
            continue
        try:
            part = aggregate_city_year(city, year, tif)
            parts.append(part)
            print(f"  {city} {year}: {len(part)} grids")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {city} {year}: {exc}")
    if parts:
        combined = pd.concat(parts, ignore_index=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"chn_pop_{year}_grid.parquet"
        combined.to_parquet(out_path, index=False)
        print(f"  -> {out_path} ({len(combined)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Years to process (e.g. 2019 2020 2021 2022 2023 2024)",
    )
    parser.add_argument(
        "--download-only", action="store_true", help="Download rasters only; skip grid aggregation"
    )
    parser.add_argument("--cities", nargs="*", default=None, help="City keys (default: all 44)")
    parser.add_argument(
        "--parallel", type=int, default=4, help="Number of parallel year processes (default 4)"
    )
    args = parser.parse_args()

    cities = args.cities or ACTIVE_CITIES
    import multiprocessing as mp

    if args.download_only or args.parallel <= 1 or len(args.years) == 1:
        for year in sorted(args.years):
            process_year(year, cities, args.download_only)
    else:
        ctx = mp.get_context("spawn")
        jobs = [
            ctx.Process(target=process_year, args=(year, list(cities), False))
            for year in sorted(args.years)
        ]
        for job in jobs:
            job.start()
        for job in jobs:
            job.join()

    return 0


if __name__ == "__main__":
    sys.exit(main())
