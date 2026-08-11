"""Audit whether the public housing replication files can support city/grid outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.data.paths import (  # noqa: E402
    OUTPUT_HOUSING_ACQUISITION_DIR,
    RAW_OPEN_DATASET_DIR,
)

DEFAULT_DATA_DIR = RAW_OPEN_DATASET_DIR / "mendeley_d6m65pyyxd_v2" / "extracted" / "data"
DEFAULT_OUTPUT = OUTPUT_HOUSING_ACQUISITION_DIR / "mendeley_d6m65pyyxd_v2_audit.json"
FILES = ("new_house.dta", "resale_house.dta", "new_spatial.dta", "resale_spatial.dta")


def _numeric_summary(series: pd.Series) -> dict[str, int | float | None]:
    valid = series.dropna()
    return {
        "non_null": int(valid.size),
        "unique": int(valid.nunique()),
        "min": float(valid.min()) if not valid.empty else None,
        "max": float(valid.max()) if not valid.empty else None,
    }


def audit_file(path: Path) -> dict[str, object]:
    frame = pd.read_stata(path, convert_categoricals=False)
    key_columns = [
        column
        for column in (
            "community_id",
            "ym_id",
            "city_year",
            "lnhp",
            "lnvolume",
            "land_id",
        )
        if column in frame.columns
    ]
    prohibited_for_assignment = {
        "city_name",
        "city_id",
        "province",
        "year",
        "month",
        "date",
        "longitude",
        "latitude",
        "lng",
        "lat",
        "community_name",
        "address",
    }
    assignment_columns = sorted(prohibited_for_assignment.intersection(frame.columns))
    community_month = [column for column in ("community_id", "ym_id") if column in frame]
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "key_column_summaries": {column: _numeric_summary(frame[column]) for column in key_columns},
        "city_date_coordinate_columns_present": assignment_columns,
        "duplicate_community_month_rows": (
            int(frame.duplicated(community_month, keep=False).sum())
            if len(community_month) == 2
            else None
        ),
        "value_labels": "absent",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    files = {name: audit_file(args.data_dir / name) for name in FILES}
    output = {
        "schema": "mendeley_d6m65pyyxd_v2_housing_audit_v1",
        "source": {
            "title": "Extrapolative Households and Strategic Firms: Evidence from China's Land and Housing Market",
            "doi": "10.17632/d6m65pyyxd.2",
            "license": "CC BY 4.0",
        },
        "files": files,
        "documentation_claim": {
            "new_house": "approximately 0.6 million records from 35 cities, 2008-2017",
            "resale_house": "approximately 0.7 million transactions from 13 cities, 2010-2017",
        },
        "released_file_findings": {
            "city_mapping": "absent",
            "calendar_mapping_for_ym_id": "absent",
            "coordinates": "absent",
            "community_names_and_addresses": "absent",
            "spatial_files": "contain distance-ring indicators but no coordinates",
            "price_measure": "log price (lnhp), not a documented nominal price level column",
        },
        "decision": "deidentified_replication_only_not_city_or_500m_grid_assignable",
        "permitted_role": "methodological_benchmark_only",
        "not_counted_as_44_city_coverage": True,
        "reason": (
            "The public files cannot be linked to named research cities or 500 m grids. "
            "Any calendar decoding from the reported sample period would be an inference, "
            "and no city decoding is available in the README, do-files, labels, or data."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision": output["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
