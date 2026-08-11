"""Aggregate the monthly VNP46A2 grid radiance cache to clean annual data.

The monthly cache (``data/active/curated/viirs/monthly``) is the production-quality
monthly VIIRS product (NASA/VIIRS/002/VNP46A2) with one row per grid-month,
verified duplicate-free.  This script aggregates it to the annual level so
that annual features and labels share the same product as the monthly causal
series, replacing the legacy per-city annual exports that carried duplicated
geographic samples.

Output columns keep the names consumed by the Python pretraining dataset and
the R estimators (``city_key, grid_id, year, avg_rad``), plus quality
provenance fields describing the monthly support behind each annual cell.

Usage:
    python scripts/analysis/build_viirs_annual_from_monthly.py --city all
    python scripts/analysis/build_viirs_annual_from_monthly.py --city beijing
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.config.project import ACTIVE_CITIES  # noqa: E402
from urban_intervention.data.paths import (  # noqa: E402
    OUTPUT_DATA_QUALITY_DIR,
    VIIRS_ANNUAL_DIR,
    VIIRS_MONTHLY_DIR,
)

REPORT_PATH = OUTPUT_DATA_QUALITY_DIR / "viirs_annual_from_monthly.json"

ANNUAL_COLUMNS = [
    "city_key",
    "grid_id",
    "year",
    "avg_rad",
    "avg_rad_median",
    "avg_rad_min",
    "avg_rad_max",
    "avg_rad_sd",
    "months_available",
    "mean_valid_days",
    "total_source_points",
]


def aggregate_city(city_key: str, output_dir: Path) -> dict:
    pattern = VIIRS_MONTHLY_DIR / f"city_key={city_key}" / "year=*" / "month=*" / "part.parquet"
    files = sorted(glob.glob(str(pattern)))
    if not files:
        msg = f"{city_key}: no monthly VIIRS partitions under {VIIRS_MONTHLY_DIR}"
        raise FileNotFoundError(msg)

    frames = []
    for fp in files:
        year = int(Path(fp).parts[-3].split("=")[1])
        month = int(Path(fp).parts[-2].split("=")[1])
        frame = pd.read_parquet(
            fp,
            columns=[
                "grid_id",
                "avg_rad",
                "valid_days_mean",
                "source_point_count",
            ],
        )
        frame["year"] = year
        frame["month"] = month
        frames.append(frame)
    monthly = pd.concat(frames, ignore_index=True)
    monthly["year"] = monthly["year"].astype("int64")

    null_keys = int(monthly[["grid_id", "year"]].isna().any(axis=1).sum())
    if null_keys:
        raise ValueError(f"{city_key}: {null_keys} monthly rows have null grid_id/year")
    dup_keys = int(monthly.duplicated(subset=["grid_id", "year", "month"]).sum())
    if dup_keys:
        raise ValueError(f"{city_key}: monthly cache has {dup_keys} duplicate grid-year-month rows")

    grouped = (
        monthly.groupby(["grid_id", "year"], sort=True, observed=True)
        .agg(
            avg_rad=("avg_rad", "mean"),
            avg_rad_median=("avg_rad", "median"),
            avg_rad_min=("avg_rad", "min"),
            avg_rad_max=("avg_rad", "max"),
            avg_rad_sd=("avg_rad", "std"),
            months_available=("month", "size"),
            mean_valid_days=("valid_days_mean", "mean"),
            total_source_points=("source_point_count", "sum"),
        )
        .reset_index()
    )
    grouped.insert(0, "city_key", city_key)
    grouped["avg_rad_sd"] = grouped["avg_rad_sd"].fillna(0.0).astype(np.float32)
    grouped["avg_rad"] = grouped["avg_rad"].astype(np.float32)
    grouped["months_available"] = grouped["months_available"].astype("int16")

    keys = ["city_key", "grid_id", "year"]
    duplicate_output_keys = int(grouped.duplicated(keys).sum())
    if duplicate_output_keys:
        raise ValueError(f"{city_key}: aggregation left {duplicate_output_keys} duplicate keys")

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{city_key}_viirs_annual.parquet"
    grouped.to_parquet(target, index=False)

    return {
        "city_key": city_key,
        "output": str(target.relative_to(ROOT)),
        "input_month_rows": len(monthly),
        "output_rows": len(grouped),
        "duplicate_input_keys": dup_keys,
        "duplicate_output_keys": duplicate_output_keys,
        "years": [int(y) for y in sorted(grouped["year"].unique())],
        "grids_per_year": int(grouped.groupby("year")["grid_id"].nunique().min()),
        "max_months": int(grouped["months_available"].max()),
        "cells_with_lt_6_months": int((grouped["months_available"] < 6).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="all", help="city key or 'all' (default: all)")
    args = parser.parse_args()

    cities = ACTIVE_CITIES if args.city == "all" else [args.city]
    reports = []
    for city in cities:
        report = aggregate_city(city, VIIRS_ANNUAL_DIR)
        reports.append(report)
        print(
            f"  {city}: {report['output_rows']} rows, "
            f"years {report['years'][0]}-{report['years'][-1]}, "
            f"{report['max_months']} max months"
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": "scripts/analysis/build_viirs_annual_from_monthly.py",
        "product": "NASA/VIIRS/002/VNP46A2",
        "aggregation": "grid-month mean of monthly grid mean radiance; "
        "months_available counts contributing months",
        "generated_at": datetime.now(UTC).isoformat(),
        "cities": reports,
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
