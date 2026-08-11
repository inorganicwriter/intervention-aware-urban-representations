"""Audit processed GEE outputs joined to research grids."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR / "src"))
from urban_intervention.data.paths import (
    OUTPUT_GEE_QUALITY_DIR,
    POPULATION_DIR,
    SENTINEL2_DIR,
    VIIRS_ANNUAL_DIR,
)

OUT_DIR = OUTPUT_GEE_QUALITY_DIR


def audit_file(path: Path, source: str, city: str) -> dict:
    df = pd.read_parquet(path)
    row = {
        "source": source,
        "city": city,
        "rows": len(df),
        "n_grids": df["grid_id"].nunique() if "grid_id" in df.columns else 0,
        "min_year": int(df["year"].min()) if "year" in df.columns and not df.empty else None,
        "max_year": int(df["year"].max()) if "year" in df.columns and not df.empty else None,
        "n_years": df["year"].nunique() if "year" in df.columns else 0,
        "duplicate_grid_year_rows": 0,
        "columns": ",".join(df.columns),
    }
    if {"grid_id", "year"}.issubset(df.columns):
        row["duplicate_grid_year_rows"] = int(df.duplicated(["grid_id", "year"]).sum())
    return row


def run_audit() -> pd.DataFrame:
    rows = []
    # Annual VIIRS is no longer kept as per-city grid-year files: the annual
    # series is aggregated from the monthly VNP46A2 cache
    # (data/active/curated/viirs_annual_aggregated). S2 and population still carry
    # per-city grid-year files.
    source_dirs = {"s2": SENTINEL2_DIR, "pop": POPULATION_DIR}
    for source, source_dir in source_dirs.items():
        for path in sorted(source_dir.glob(f"*_{source}.parquet")):
            rows.append(audit_file(path, source, path.name.removesuffix(f"_{source}.parquet")))
    for path in sorted(VIIRS_ANNUAL_DIR.glob("*_viirs_annual.parquet")):
        rows.append(audit_file(path, "viirs", path.name.removesuffix("_viirs_annual.parquet")))
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "gee_processed_audit.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {out_path} ({len(out)} rows)")
    if out.empty:
        return out
    print(
        out.groupby("source")
        .agg(
            files=("city", "count"),
            cities=("city", "nunique"),
            min_year=("min_year", "min"),
            max_year=("max_year", "max"),
            duplicate_grid_year_rows=("duplicate_grid_year_rows", "sum"),
        )
        .to_string()
    )
    short = out[(out["n_years"] < out.groupby("source")["n_years"].transform("max"))]
    print(f"short_coverage_files={len(short)}")
    if not short.empty:
        print(
            short[["source", "city", "min_year", "max_year", "n_years", "rows"]].to_string(
                index=False
            )
        )
    dup = out[out["duplicate_grid_year_rows"] > 0]
    print(f"files_with_duplicate_grid_year={len(dup)}")
    if not dup.empty:
        print(dup[["source", "city", "duplicate_grid_year_rows"]].to_string(index=False))
    return out


def main() -> int:
    run_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
