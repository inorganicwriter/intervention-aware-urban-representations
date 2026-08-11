"""Export VIIRS VNP46A2 samples by city bounding box.

This intentionally follows the project's established acquisition contract:
GEE exports georeferenced sample points without ``grid_id``; the local
``process_viirs_bbox_exports.py`` stage performs nearest-grid matching,
aggregation and primary-key validation.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from urban_intervention.config.project import ACTIVE_CITIES, get_effective_bbox

PROJECT = "macro-city-engine"
PRODUCT = "NASA/VIIRS/002/VNP46A2"
RADIANCE = "Gap_Filled_DNB_BRDF_Corrected_NTL"
SUPPORTED_YEARS = range(2012, 2025)


def _mask(image: ee.Image) -> ee.Image:
    quality = image.select("Mandatory_Quality_Flag")
    snow = image.select("Snow_Flag")
    # Official VNP46A2.002: 0/1 are high quality; 2 is poor quality.
    return image.select(RADIANCE).updateMask(quality.lte(1).And(snow.eq(0))).rename("radiance")


def _dates(year: int, month: int | None) -> tuple[ee.Date, ee.Date, str]:
    if month is None:
        return ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year + 1, 1, 1), str(year)
    start = ee.Date.fromYMD(year, month, 1)
    end = ee.Date.fromYMD(year + (month == 12), 1 if month == 12 else month + 1, 1)
    return start, end, f"{year}-{month:02d}"


def build_samples(city: str, year: int, month: int | None) -> ee.FeatureCollection:
    start, end, period = _dates(year, month)
    bbox = ee.Geometry.Rectangle(get_effective_bbox(city, buffer_km=10.0))
    daily = ee.ImageCollection(PRODUCT).filterDate(start, end).filterBounds(bbox).map(_mask)
    image = daily.mean().addBands(daily.count().rename("valid_days"))
    samples = image.sample(
        region=bbox,
        scale=500,
        projection="EPSG:4326",
        geometries=True,
        tileScale=4,
    )

    def annotate(feature: ee.Feature) -> ee.Feature:
        coordinates = feature.geometry().coordinates()
        return feature.set(
            {
                "city": city,
                "year": year,
                "month": month if month is not None else 0,
                "period": period,
                "product": PRODUCT,
                "longitude": coordinates.get(0),
                "latitude": coordinates.get(1),
            }
        )

    return samples.map(annotate).select(
        [
            "city",
            "year",
            "month",
            "period",
            "product",
            "radiance",
            "valid_days",
            "latitude",
            "longitude",
        ]
    )


def queue(city: str, year: int, month: int | None) -> str:
    _, _, period = _dates(year, month)
    label = period.replace("-", "_")
    description = f"viirs_{city}_{label}"
    task = ee.batch.Export.table.toDrive(
        collection=build_samples(city, year, month),
        description=description,
        folder="MIT_Summer_VIIRS",
        fileNamePrefix=f"viirs_{city}_{label}",
        fileFormat="CSV",
    )
    task.start()
    print(f"[QUEUED] {description}: {task.id}")
    return task.id


def _description(city: str, year: int, month: int | None) -> str:
    _, _, period = _dates(year, month)
    return f"viirs_{city}_{period.replace('-', '_')}"


def _parse_years(value: str) -> list[int]:
    if value == "all":
        return list(SUPPORTED_YEARS)
    if "-" in value and "," not in value:
        lo, hi = (int(x) for x in value.split("-", 1))
        return list(range(lo, hi + 1))
    return [int(x) for x in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, help="city key, comma list, or all")
    parser.add_argument("--year", required=True, help="year, comma list, range, or all")
    parser.add_argument("--frequency", choices=["annual", "monthly"], default="annual")
    parser.add_argument("--month", type=int, choices=range(1, 13))
    parser.add_argument("--max-tasks", type=int, default=500)
    parser.add_argument(
        "--submit-limit",
        type=int,
        default=None,
        help="Maximum number of missing jobs to submit in this invocation",
    )
    parser.add_argument("--submit-delay", type=float, default=0.25)
    parser.add_argument("--submit-workers", type=int, default=1)
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    cities = list(ACTIVE_CITIES) if args.city == "all" else args.city.split(",")
    unknown = sorted(set(cities) - set(ACTIVE_CITIES))
    if unknown:
        raise ValueError(f"Unknown city keys: {unknown}")
    years = _parse_years(args.year)
    unsupported = sorted(set(years) - set(SUPPORTED_YEARS))
    if unsupported:
        raise ValueError(f"Unsupported years: {unsupported}")
    if args.frequency == "annual":
        if args.month:
            raise ValueError("--month requires --frequency monthly")
        months: list[int | None] = [None]
    else:
        months = [args.month] if args.month else list(range(1, 13))
    jobs = [(c, y, m) for c in cities for y in years for m in months]
    if len(jobs) > args.max_tasks:
        raise ValueError(f"Refusing to queue {len(jobs)} tasks; raise --max-tasks explicitly")
    existing = {
        task.get("description")
        for task in ee.data.getTaskList()
        if task.get("state") in {"READY", "RUNNING"}
    }
    missing_jobs = [job for job in jobs if _description(*job) not in existing]
    if args.submit_limit is not None:
        missing_jobs = missing_jobs[: args.submit_limit]

    def submit_job(job: tuple[str, int, int | None]) -> str:
        city, year, month = job
        description = _description(city, year, month)
        for attempt in range(1, 11):
            try:
                queue(city, year, month)
                return description
            except Exception as exc:
                print(f"[RETRY {attempt}/10] {description}: {exc}")
                time.sleep(min(30, attempt * 3))
                try:
                    ee.Initialize(project=PROJECT)
                    remote = {
                        task.get("description")
                        for task in ee.data.getTaskList()
                        if task.get("state") in {"READY", "RUNNING"}
                    }
                    if description in remote:
                        return description
                except Exception:
                    pass
        raise RuntimeError(f"Unable to submit after retries: {description}")

    completed = 0
    with ThreadPoolExecutor(max_workers=args.submit_workers) as pool:
        futures = [pool.submit(submit_job, job) for job in missing_jobs]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 100 == 0:
                print(f"submit progress={completed}/{len(missing_jobs)}", flush=True)
            if args.submit_delay:
                time.sleep(args.submit_delay)
    print(
        f"submission complete: universe={len(jobs)} submitted_batch={completed} "
        f"remaining_before_batch={sum(_description(*job) not in existing for job in jobs)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
