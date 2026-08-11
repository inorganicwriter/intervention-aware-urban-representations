"""Precompute per-city feature stores for grid-control matching.

Reads the raw monthly/annual parquet partitions once per city, applies the
same transformations as the R estimators (asinh, log1p, median aggregation),
and writes two compact parquet files per city:

  data/active/causal/feature_store/{city}_monthly.parquet
      city_key, grid_id, month, viirs_avg_asinh, viirs_valid_days_mean,
      viirs_source_point_count, housing_log_price

  data/active/causal/feature_store/{city}_annual.parquet
      city_key, grid_id, year, poi_count_log, poi_category_entropy,
      poi_commercial_share, poi_transport_access_log, population_log,
      housing_log_price

The R matching code reads these instead of opening 156 small VIIRS partitions
per task, reducing I/O by ~100x.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from urban_intervention.config.project import ACTIVE_CITIES  # noqa: E402
from urban_intervention.data.paths import (  # noqa: E402
    FEATURE_STORE_DIR as OUT_DIR,
)
from urban_intervention.data.paths import (
    housing_annual_path,
    housing_monthly_panel_path,
    poi_annual_path,
    population_data_path,
    viirs_monthly_city_dir,
)

MONTHLY_COLUMNS = [
    "city_key",
    "grid_id",
    "month",
    "viirs_avg_asinh",
    "viirs_valid_days_mean",
    "viirs_source_point_count",
    "housing_log_price",
]


def build_monthly_store(city_key: str) -> pd.DataFrame:
    """Merge VIIRS 156-month partitions and housing monthly into one frame.

    Every branch returns the full ``MONTHLY_COLUMNS`` schema (missing
    modalities become empty columns) so downstream consumers never see
    per-city column drift.
    """
    viirs_root = viirs_monthly_city_dir(city_key)
    if not viirs_root.is_dir():
        print(f"[skip] VIIRS monthly directory not found: {viirs_root}")
        return pd.DataFrame(columns=MONTHLY_COLUMNS)
    viirs_parts = []
    for year_dir in sorted(viirs_root.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.startswith("year="):
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir() or not month_dir.name.startswith("month="):
                continue
            parquet_path = month_dir / "part.parquet"
            if not parquet_path.exists():
                continue
            year = int(year_dir.name[5:])
            month = int(month_dir.name[6:])
            df = pd.read_parquet(
                parquet_path,
                columns=["grid_id", "avg_rad", "valid_days_mean", "source_point_count"],
            )
            df["city_key"] = city_key
            df["month"] = pd.Timestamp(year=year, month=month, day=1)
            df["viirs_avg_asinh"] = np.arcsinh(df["avg_rad"].astype(np.float32))
            viirs_parts.append(
                df[
                    [
                        "city_key",
                        "grid_id",
                        "month",
                        "viirs_avg_asinh",
                        "valid_days_mean",
                        "source_point_count",
                    ]
                ]
            )
    if viirs_parts:
        viirs = pd.concat(viirs_parts, ignore_index=True)
        viirs.rename(
            columns={
                "valid_days_mean": "viirs_valid_days_mean",
                "source_point_count": "viirs_source_point_count",
            },
            inplace=True,
        )
    else:
        viirs = pd.DataFrame(
            columns=[
                "city_key",
                "grid_id",
                "month",
                "viirs_avg_asinh",
                "viirs_valid_days_mean",
                "viirs_source_point_count",
            ]
        )

    # Housing: read monthly panel, aggregate to grid-month median
    housing_path = housing_monthly_panel_path(city_key)
    if housing_path.exists():
        raw = pd.read_parquet(
            housing_path,
            columns=["city_key", "grid_id", "observed_month", "log_price_raw_median"],
        )
        raw["month"] = pd.to_datetime(raw["observed_month"]).dt.to_period("M").dt.to_timestamp()
        housing = raw.groupby(["city_key", "grid_id", "month"], as_index=False)[
            "log_price_raw_median"
        ].median()
        housing.rename(columns={"log_price_raw_median": "housing_log_price"}, inplace=True)
    else:
        housing = pd.DataFrame(columns=["city_key", "grid_id", "month", "housing_log_price"])

    # Outer merge: not every grid has both VIIRS and housing.  Reindex every
    # branch onto MONTHLY_COLUMNS so missing modalities are explicit empty
    # columns instead of a different column set per city.
    if viirs.empty and housing.empty:
        return pd.DataFrame(columns=MONTHLY_COLUMNS)
    merged = pd.merge(viirs, housing, on=["city_key", "grid_id", "month"], how="outer")
    for column in MONTHLY_COLUMNS:
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged[MONTHLY_COLUMNS]


def build_annual_store(city_key: str) -> pd.DataFrame:
    """Merge POI, population, and housing annual into one frame."""

    frames = []

    # POI
    poi_path = poi_annual_path(city_key)
    if poi_path.exists():
        poi = pd.read_parquet(
            poi_path,
            columns=[
                "city",
                "grid_id",
                "year",
                "poi_count",
                "poi_category_entropy",
                "poi_commercial_share",
                "poi_transport_access_count",
            ],
        )
        poi.rename(columns={"city": "city_key"}, inplace=True)
        poi["poi_count_log"] = np.log1p(np.maximum(poi["poi_count"].astype(np.float32), 0))
        poi["poi_transport_access_log"] = np.log1p(
            np.maximum(poi["poi_transport_access_count"].astype(np.float32), 0)
        )
        frames.append(
            poi[
                [
                    "city_key",
                    "grid_id",
                    "year",
                    "poi_count_log",
                    "poi_category_entropy",
                    "poi_commercial_share",
                    "poi_transport_access_log",
                ]
            ]
        )

    # Population
    pop_path = population_data_path(city_key)
    if pop_path.exists():
        pop = pd.read_parquet(pop_path, columns=["city", "grid_id", "year", "pop_count"])
        pop.rename(columns={"city": "city_key"}, inplace=True)
        pop["population_log"] = np.log1p(np.maximum(pop["pop_count"].astype(np.float32), 0))
        pop = pop.groupby(["city_key", "grid_id", "year"], as_index=False)["population_log"].mean()
        frames.append(pop)

    # Housing annual (already precomputed by build_formal_matching_inputs.R)
    housing_annual_path_val = housing_annual_path(city_key)
    if housing_annual_path_val.exists():
        ha = pd.read_parquet(
            housing_annual_path_val, columns=["city_key", "grid_id", "year", "housing_log_price"]
        )
        frames.append(ha)

    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    result = frames[0]
    for frame in frames[1:]:
        result = pd.merge(result, frame, on=["city_key", "grid_id", "year"], how="outer")
    return result


def process_city(city_key: str) -> dict:
    """Build and write feature stores for one city."""

    monthly = build_monthly_store(city_key)
    annual = build_annual_store(city_key)

    monthly_path = OUT_DIR / f"{city_key}_monthly.parquet"
    annual_path = OUT_DIR / f"{city_key}_annual.parquet"

    monthly.to_parquet(monthly_path, index=False, compression="zstd")
    annual.to_parquet(annual_path, index=False, compression="zstd")

    return {
        "city": city_key,
        "monthly_rows": len(monthly),
        "monthly_mb": monthly_path.stat().st_size / 1e6,
        "annual_rows": len(annual),
        "annual_mb": annual_path.stat().st_size / 1e6,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="all", help="City key or 'all'")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cities = ACTIVE_CITIES if args.city == "all" else [args.city]

    for city_key in cities:
        stats = process_city(city_key)
        print(
            f"[{city_key}] monthly: {stats['monthly_rows']:,} rows "
            f"({stats['monthly_mb']:.1f}MB)  "
            f"annual: {stats['annual_rows']:,} rows ({stats['annual_mb']:.1f}MB)"
        )

    total_mb = sum(f.stat().st_size for f in OUT_DIR.glob("*.parquet")) / 1e6
    print(f"\nTotal feature store: {total_mb:.0f}MB across {len(cities)} cities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
