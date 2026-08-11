"""Normalize the CC BY Figshare China annual city house-price dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.config.project import CITIES  # noqa: E402
from urban_intervention.data.paths import (  # noqa: E402
    RAW_OPEN_DATASET_DIR,
    STAGING_DIR,
)
from urban_intervention.pipelines.housing.open_research import (  # noqa: E402
    load_figshare_workbooks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_OPEN_DATASET_DIR / "figshare_26968507_v1/extracted/data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STAGING_DIR / "housing/open_research/figshare_26968507_v1",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "outputs/housing_acquisition",
    )
    args = parser.parse_args()

    frame = load_figshare_workbooks(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "city_house_price_year.parquet"
    frame.to_parquet(output_path, index=False)

    observed = frame.loc[frame["is_research_city"]].copy()
    coverage = (
        observed.groupby("city_key", as_index=False)
        .agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
            observed_years=("year", "nunique"),
            observations=("source_record_id", "size"),
        )
        .set_index("city_key")
        .reindex(sorted(CITIES))
        .rename_axis("city_key")
        .reset_index()
    )
    coverage["covered"] = coverage["observations"].fillna(0).gt(0)
    coverage_path = args.report_dir / "figshare_26968507_research_city_coverage.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    report = {
        "schema": "figshare_city_house_price_audit_v1",
        "input_files": sorted(path.name for path in args.input_dir.glob("*.xlsx")),
        "all_city_year_rows": int(len(frame)),
        "all_cities": int(frame["city_cn"].nunique()),
        "first_year": int(frame["year"].min()),
        "last_year": int(frame["year"].max()),
        "research_city_rows": int(frame["is_research_city"].sum()),
        "research_cities_covered": int(coverage["covered"].sum()),
        "research_cities_missing": coverage.loc[~coverage["covered"], "city_key"].tolist(),
        "quality_flag_counts": frame["quality_flags"].value_counts().to_dict(),
        "output_file": str(output_path),
        "coverage_file": str(coverage_path),
    }
    report_path = args.report_dir / "figshare_26968507_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"observations={output_path}")
    print(f"coverage={coverage_path}")
    print(f"audit={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
