"""Audit generated POI grid-year panels.

Reads data/active/curated/poi/*_poi_grid_yearly.parquet and writes yearly coverage
summaries to outputs/poi_quality/poi_coverage_by_city_year.csv.
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR / "src"))
from urban_intervention.data.paths import OUTPUT_POI_QUALITY_DIR, POI_DIR

OUT_DIR = OUTPUT_POI_QUALITY_DIR


def audit_city_panel_columns() -> list[str]:
    return [
        "city",
        "year",
        "n_grids_with_poi",
        "poi_count_total",
        "food_total",
        "retail_total",
        "life_service_total",
        "chain_total",
        "median_grid_poi_count",
    ]


def audit_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=audit_city_panel_columns())
    rows = []
    for (city, year), sub in df.groupby(["city", "year"]):
        rows.append(
            {
                "city": city,
                "year": int(year),
                "n_grids_with_poi": int(sub["grid_id"].nunique()),
                "poi_count_total": int(sub["poi_count"].sum()),
                "food_total": int(sub.get("poi_food_count", pd.Series(dtype=float)).sum()),
                "retail_total": int(sub.get("poi_retail_count", pd.Series(dtype=float)).sum()),
                "life_service_total": int(
                    sub.get("poi_life_service_count", pd.Series(dtype=float)).sum()
                ),
                "chain_total": int(sub.get("poi_chain_count", pd.Series(dtype=float)).sum()),
                "median_grid_poi_count": float(sub["poi_count"].median()),
            }
        )
    return pd.DataFrame(rows, columns=audit_city_panel_columns())


def run_audit() -> pd.DataFrame:
    parts = [audit_file(p) for p in sorted(POI_DIR.glob("*_poi_grid_yearly.parquet"))]
    if parts:
        out = pd.concat(parts, ignore_index=True)
    else:
        out = pd.DataFrame(columns=audit_city_panel_columns())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "poi_coverage_by_city_year.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {out_path} ({len(out)} rows)")
    return out


def main() -> int:
    run_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
