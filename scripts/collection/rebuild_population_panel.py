"""Rebuild the curated per-city population panel from clean sources.

The legacy curated population files (``data/active/curated/population/*_pop.parquet``)
contain conflicting duplicate rows for 2010-2018 with no source column, so
they cannot be repaired in place.  This script rebuilds them from two clean
inputs with a single product per year:

1. GEE WorldPop exports (``WorldPop/GP/100m/pop``) — used for 2010-2014,
   because the R2024B (Global 2) product starts in 2015.
2. R2024B (Global 2) staging products
   ``data/archive/staging/worldpop_r2024b/chn_pop_{year}_grid.parquet`` — the single
   source for 2015-2024.

Output columns: ``city, grid_id, year, pop_count, source_version`` where
``source_version`` is ``gee`` or ``r2024b``.  The panel is validated to have
exactly one row per (grid_id, year).

Known boundary: the GEE and R2024B products have different per-pixel
estimation scales (grid-level ratios range over ~250x), so the 2014 -> 2015
transition contains a product jump.  This is accepted by design; the two
series must never be merged for the same year.  GEE mean cell values are
scaled to the 500m grid total by multiplying by 25 cells.

Usage:
    python scripts/collection/rebuild_population_panel.py \
        --gee-dir data/archive/staging/gee/pop \
        --r2024b-dir data/archive/staging/worldpop_r2024b \
        --out-dir data/active/curated/population
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "scripts"))
from urban_intervention.config.project import ACTIVE_CITIES  # noqa: E402

GEE_YEARS = (2010, 2011, 2012, 2013, 2014)
GEE_SOURCE = "gee"
R2024B_SOURCE = "r2024b"
CELLS_PER_GRID = 25  # 500m grid / 100m cells


def read_gee_city(city: str, gee_dir: Path) -> pd.DataFrame:
    files = sorted(gee_dir.glob(f"{city}_*.csv"))
    frames = []
    for fp in files:
        year = int(fp.stem.rsplit("_", 1)[-1])
        if year not in GEE_YEARS:
            continue
        df = pd.read_csv(fp, usecols=["grid_id", "mean"])
        df["year"] = year
        df = df.rename(columns={"mean": "pop_count"})
        df["pop_count"] = df["pop_count"] * CELLS_PER_GRID
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["city", "grid_id", "year", "pop_count"])
    out = pd.concat(frames, ignore_index=True)
    out["city"] = city
    out["source_version"] = GEE_SOURCE
    return out


def read_r2024b_city(city: str, r2024b_dir: Path) -> pd.DataFrame:
    frames = []
    for fp in sorted(glob.glob(str(r2024b_dir / "chn_pop_*_grid.parquet"))):
        df = pd.read_parquet(fp, columns=["city_key", "grid_id", "year", "pop_count"])
        df = df[df["city_key"] == city]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"{city}: no R2024B grid products in {r2024b_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["city"] = city
    out["source_version"] = R2024B_SOURCE
    return out[["city", "grid_id", "year", "pop_count", "source_version"]]


def rebuild_city(city: str, gee_dir: Path, r2024b_dir: Path, out_dir: Path) -> dict:
    gee = read_gee_city(city, gee_dir)
    r2024b = read_r2024b_city(city, r2024b_dir)
    combined = pd.concat([gee, r2024b], ignore_index=True)

    dup = int(combined.duplicated(["grid_id", "year", "source_version"]).sum())
    if dup:
        raise ValueError(f"{city}: {dup} duplicate rows within one source version")

    dup_keys = int(combined.duplicated(["grid_id", "year"]).sum())
    if dup_keys:
        raise ValueError(f"{city}: {dup_keys} rows share a (grid_id, year) key")

    combined = combined.sort_values(["grid_id", "year", "source_version"])
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{city}_pop.parquet"
    combined.to_parquet(target, index=False)

    return {
        "city": city,
        "rows": len(combined),
        "years": sorted(combined["year"].unique()),
        "version_years": combined.groupby("source_version")["year"].nunique().to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gee-dir", type=Path, required=True)
    parser.add_argument("--r2024b-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--city", default="all")
    args = parser.parse_args()

    cities = ACTIVE_CITIES if args.city == "all" else [args.city]
    for city in cities:
        report = rebuild_city(city, args.gee_dir, args.r2024b_dir, args.out_dir)
        print(
            f"  {city}: {report['rows']:,} rows, "
            f"years {report['years'][0]}-{report['years'][-1]}, "
            f"versions {report['version_years']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
