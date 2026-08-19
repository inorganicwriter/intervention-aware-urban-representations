"""Extend the monthly VIIRS cache with 2025 partitions only.

The original exporter (``run_gee_viirs_bbox_exports.py``) pins
``SUPPORTED_YEARS = 2012..2024``; editing it would re-scan the full
2012-2024 history.  This script queues only the requested months (default
2025-01..2025-12) for the 44 cities, skipping months whose partition
already exists under ``data/active/curated/viirs/monthly/``, then hands the
downloaded exports to the existing ``process_viirs_bbox_exports.py`` stage
(nearest-grid match, aggregation, primary-key validation, partition write).

Stages can be run independently on the server:

    python scripts/collection/extend_viirs_monthly_2025.py --submit-only
    python scripts/collection/extend_viirs_monthly_2025.py --status
    python scripts/collection/extend_viirs_monthly_2025.py --process-only

The GEE sampling/export logic below mirrors the original exporter
(``build_samples``/``queue``) for the VNP46A2 monthly mean radiance.

Usage:
    python scripts/collection/extend_viirs_monthly_2025.py [--start-month 2025-01]
        [--end-month 2025-12] [--dry-run] [--submit-only] [--status]
        [--process-only] [--max-tasks 500] [--submit-delay 0.25]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.config.project import ACTIVE_CITIES, get_effective_bbox  # noqa: E402

VIIRS_MONTHLY_DIR = (
    ROOT / "data" / "active" / "curated" / "viirs" / "monthly"
)
DRIVE_FOLDER = "MIT_Summer_VIIRS"
PROJECT = "macro-city-engine"
PRODUCT = "NASA/VIIRS/002/VNP46A2"
RADIANCE = "Gap_Filled_DNB_BRDF_Corrected_NTL"
PROCESS_SCRIPT = ROOT / "scripts" / "collection" / "process_viirs_bbox_exports.py"
REPORT_DIR = ROOT / "outputs" / "collection" / "viirs_monthly_2025"


def _import_ee() -> None:
    global ee
    import ee  # noqa: PLC0415

    ee.Initialize(project=PROJECT)


def _dates(year: int, month: int) -> tuple:
    start = ee.Date.fromYMD(year, month, 1)
    end = ee.Date.fromYMD(year + (month == 12), 1 if month == 12 else month + 1, 1)
    return start, end, f"{year}-{month:02d}"


def _mask(image) -> object:
    quality = image.select("Mandatory_Quality_Flag")
    snow = image.select("Snow_Flag")
    # Official VNP46A2.002: 0/1 are high quality; 2 is poor quality.
    return image.select(RADIANCE).updateMask(quality.lte(1).And(snow.eq(0))).rename("radiance")


def build_samples(city: str, year: int, month: int) -> object:
    start, end, _ = _dates(year, month)
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

    def annotate(feature) -> object:
        coordinates = feature.geometry().coordinates()
        return feature.set(
            {
                "city": city,
                "year": year,
                "month": month,
                "lon": coordinates.get(0),
                "lat": coordinates.get(1),
            }
        )

    return samples.map(annotate)


def queue_export(city: str, year: int, month: int) -> str:
    start, end, period = _dates(year, month)
    label = period.replace("-", "_")
    description = f"viirs_{city}_{label}"
    task = ee.batch.Export.table.toDrive(
        collection=build_samples(city, year, month),
        description=description,
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"viirs_{city}_{label}",
        fileFormat="CSV",
    )
    task.start()
    print(f"[QUEUED] {description}: {task.id}")
    return task.id


def partition_exists(city: str, year: int, month: int) -> bool:
    return (
        VIIRS_MONTHLY_DIR
        / f"city_key={city}"
        / f"year={year}"
        / f"month={month:02d}"
        / "part.parquet"
    ).is_file()


def missing_months(start: str, end: str) -> list[tuple[str, int, int]]:
    import pandas as pd  # noqa: PLC0415

    months = pd.period_range(start, end, freq="M")
    jobs = [
        (city, int(period.year), int(period.month))
        for city in ACTIVE_CITIES
        for period in months
        if not partition_exists(city, int(period.year), int(period.month))
    ]
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-month", default="2025-01", help="First month YYYY-MM")
    parser.add_argument("--end-month", default="2025-12", help="Last month YYYY-MM (inclusive)")
    parser.add_argument("--dry-run", action="store_true", help="List missing months without touching GEE")
    parser.add_argument("--submit-only", action="store_true", help="Queue GEE exports only")
    parser.add_argument("--status", action="store_true", help="Print GEE task states")
    parser.add_argument("--process-only", action="store_true", help="Run the processing stage only")
    parser.add_argument("--max-tasks", type=int, default=500)
    parser.add_argument("--submit-delay", type=float, default=0.25)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = missing_months(args.start_month, args.end_month)
    print(f"缺失分区任务: {len(jobs)}（{args.start_month} ~ {args.end_month}，44 城，跳过已有）")

    if args.dry_run:
        for city, year, month in jobs[:10]:
            print(f"  {city} {year}-{month:02d}")
        print(f"dry-run: 共 {len(jobs)} 个缺失任务（前 10 显示）")
        return 0

    if args.status:
        _import_ee()
        tasks = ee.batch.Task.list()
        states: dict[str, int] = {}
        for task in tasks:
            description = task.config.get("description", "")
            if description.startswith("viirs_"):
                states[task.state] = states.get(task.state, 0) + 1
        print("GEE 任务状态:", states)
        return 0

    if args.submit_only or not args.process_only:
        _import_ee()
        existing: set[str] = set()
        for task in ee.batch.Task.list():
            description = task.config.get("description", "")
            if description.startswith("viirs_") and task.state in {"READY", "RUNNING"}:
                existing.add(description)
        submitted = 0
        for city, year, month in jobs:
            if submitted >= args.max_tasks:
                break
            label = f"{year}-{month:02d}".replace("-", "_")
            description = f"viirs_{city}_{label}"
            if description in existing:
                continue
            queue_export(city, year, month)
            submitted += 1
            time.sleep(args.submit_delay)
        print(f"已提交 {submitted} 个导出任务（跳过已排队/已有分区）")
        print("完成后运行: python scripts/collection/extend_viirs_monthly_2025.py --process-only")
        return 0

    # --process-only (default final stage)
    completed = 0
    failed: list[str] = []
    for city, year, month in jobs:
        period = f"{year}-{month:02d}"
        result = subprocess.run(
            [
                sys.executable,
                str(PROCESS_SCRIPT),
                "--city",
                city,
                "--period",
                period,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0 or not partition_exists(city, year, month):
            failed.append(f"{city} {period}: {result.stderr[-300:]}")
        else:
            completed += 1
        print(f"[{completed + len(failed)}/{len(jobs)}] {city} {period}: "
              f"{'OK' if partition_exists(city, year, month) else 'FAIL'}")
    print(f"物化完成: {completed} 成功, {len(failed)} 失败")
    if failed:
        (REPORT_DIR / "processing_failures.txt").write_text(
            "\n".join(failed), encoding="utf-8"
        )
        print(f"失败详情: {REPORT_DIR / 'processing_failures.txt'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
