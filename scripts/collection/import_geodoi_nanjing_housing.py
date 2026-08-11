"""Standardize the open 2009Q1-2017Q1 Nanjing community housing panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from urban_intervention.data.paths import (
    RAW_OPEN_DATASET_DIR,
    STAGING_HOUSING_STANDARDIZED_DIR,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ID = "geodoi_geodb_2018_04_08_v1"
SOURCE_URL = "https://geodoi.ac.cn/WebCn/geodoi.aspx?Id=913"
SECTION_STARTS = {
    "sale_price_cny_m2": 16,
    "sale_transaction_count": 49,
    "rent_cny_m2": 82,
    "lease_transaction_count": 115,
}
SECTION_WIDTH = 33


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _periods(source: pd.DataFrame, start: int) -> list[tuple[int, int]]:
    values = pd.to_numeric(source.iloc[4, start : start + SECTION_WIDTH], errors="raise")
    result = [(int(value) // 100, int(value) % 100) for value in values]
    expected = [(year, quarter) for year in range(2009, 2017) for quarter in range(1, 5)]
    expected.append((2017, 1))
    if result != expected:
        raise ValueError(f"Unexpected quarter columns at {start}: {result}")
    return result


def _append_flag(flags: pd.Series, mask: pd.Series, value: str) -> pd.Series:
    flags = flags.copy()
    flags.loc[mask] = flags.loc[mask].map(
        lambda existing: ";".join(filter(None, [existing, value]))
    )
    return flags


def normalize(source: pd.DataFrame, source_hash: str) -> pd.DataFrame:
    if source.shape[1] < 148:
        raise ValueError(f"Expected at least 148 columns, found {source.shape[1]}")
    periods_by_section = {name: _periods(source, start) for name, start in SECTION_STARTS.items()}
    if len({tuple(periods) for periods in periods_by_section.values()}) != 1:
        raise ValueError("Quarter columns differ between measure sections")
    periods = next(iter(periods_by_section.values()))

    numeric_id = pd.to_numeric(source.iloc[:, 0], errors="coerce")
    data_rows = source.index[numeric_id.notna() & source.index.to_series().ge(5)]
    base = source.loc[data_rows, :15].copy().reset_index(names="source_index")
    base.columns = [
        "source_index",
        "source_community_id",
        "community_name_cn",
        "community_name_en",
        "community_category_cn",
        "community_category_en",
        "district_cn",
        "district_en",
        "lon",
        "lat",
        "construction_year",
        "plot_ratio",
        "greening_rate_pct",
        "floor_area_m2",
        "number_of_houses",
        "product_type_cn",
        "product_type_en",
    ]
    base["raw_row_number"] = base["source_index"] + 1
    base["source_community_id"] = pd.to_numeric(base["source_community_id"], errors="raise").astype(
        "Int64"
    )
    for column in [
        "lon",
        "lat",
        "construction_year",
        "plot_ratio",
        "greening_rate_pct",
        "floor_area_m2",
        "number_of_houses",
    ]:
        base[column] = pd.to_numeric(base[column], errors="coerce")
    for column in [
        "community_name_cn",
        "community_name_en",
        "community_category_cn",
        "community_category_en",
        "district_cn",
        "district_en",
        "product_type_cn",
        "product_type_en",
    ]:
        base[column] = base[column].astype("string")

    repeated = base.loc[base.index.repeat(len(periods))].reset_index(drop=True)
    repeated["year"] = [year for _ in range(len(base)) for year, _quarter in periods]
    repeated["quarter"] = [quarter for _ in range(len(base)) for _year, quarter in periods]
    repeated["period"] = pd.Series(
        [
            f"{year}Q{quarter}"
            for year, quarter in zip(repeated["year"], repeated["quarter"], strict=True)
        ],
        dtype="string",
    )
    repeated["period_start"] = pd.PeriodIndex(repeated["period"], freq="Q-DEC").start_time

    for name, start in SECTION_STARTS.items():
        values = source.loc[data_rows, start : start + SECTION_WIDTH - 1]
        repeated[name] = values.apply(pd.to_numeric, errors="coerce").to_numpy().reshape(-1)

    repeated["source_record_id"] = [
        hashlib.sha256(f"{SOURCE_ID}|{source_hash}|{community}|{period}".encode()).hexdigest()[:24]
        for community, period in zip(
            repeated["source_community_id"], repeated["period"], strict=True
        )
    ]
    repeated["source_id"] = SOURCE_ID
    repeated["source_url"] = SOURCE_URL
    repeated["source_file_sha256"] = source_hash
    repeated["city_key"] = "nanjing"
    repeated["spatial_unit"] = "residential_community_point"
    repeated["temporal_unit"] = "quarter"
    repeated["source_coordinate_crs"] = "geographic_degrees_crs_unspecified"
    repeated["quality_flags"] = ""
    measures = list(SECTION_STARTS)
    repeated["quality_flags"] = _append_flag(
        repeated["quality_flags"], repeated[measures].isna().all(axis=1), "all_measures_missing"
    )
    valid_coordinate = repeated["lon"].between(118.0, 119.5) & repeated["lat"].between(31.0, 33.0)
    repeated["quality_flags"] = _append_flag(
        repeated["quality_flags"], ~valid_coordinate, "invalid_nanjing_coordinate"
    )
    suspicious_low_price = repeated["sale_price_cny_m2"].between(0, 999, inclusive="both")
    repeated["quality_flags"] = _append_flag(
        repeated["quality_flags"], suspicious_low_price, "suspicious_low_sale_price"
    )
    repeated["quality_flags"] = _append_flag(
        repeated["quality_flags"],
        repeated["sale_price_cny_m2"].notna() & repeated["sale_price_cny_m2"].le(0),
        "nonpositive_sale_price",
    )
    duplicate = repeated.duplicated(["source_community_id", "period"], keep=False)
    repeated["quality_flags"] = _append_flag(
        repeated["quality_flags"], duplicate, "duplicate_community_quarter"
    )
    return repeated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_OPEN_DATASET_DIR
        / "geodoi_geodb_2018_04_08_v1"
        / "extracted"
        / "HousePriceNanjing_2009-2017.xls",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STAGING_HOUSING_STANDARDIZED_DIR / SOURCE_ID,
    )
    args = parser.parse_args()

    source_hash = sha256_file(args.input)
    source = pd.read_excel(args.input, sheet_name="Tab.1", header=None, engine="xlrd")
    standardized = normalize(source, source_hash)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "community_quarter_housing.parquet"
    standardized.to_parquet(output, index=False)

    manifest = {
        "schema": "geodoi_nanjing_housing_import_manifest_v1",
        "source_id": SOURCE_ID,
        "source_url": SOURCE_URL,
        "source_file": str(args.input),
        "source_file_sha256": source_hash,
        "source_community_rows": int(standardized["source_community_id"].nunique()),
        "output_community_quarter_rows": int(len(standardized)),
        "first_period": str(standardized["period"].min()),
        "last_period": str(standardized["period"].max()),
        "quarters": int(standardized["period"].nunique()),
        "rows_with_sale_price": int(standardized["sale_price_cny_m2"].notna().sum()),
        "rows_with_sale_transaction_count": int(
            standardized["sale_transaction_count"].notna().sum()
        ),
        "rows_with_rent": int(standardized["rent_cny_m2"].notna().sum()),
        "rows_with_lease_transaction_count": int(
            standardized["lease_transaction_count"].notna().sum()
        ),
        "coordinate_crs_status": "geographic_degrees_crs_unspecified",
        "semantic_warning": (
            "The panel is quarterly, not monthly. Coordinates are longitude/latitude "
            "degrees but the source file does not declare a geographic CRS. Low source "
            "prices are retained and flagged rather than silently corrected."
        ),
        "monthly_outcome_ready": False,
        "quarterly_outcome_candidate": True,
        "output_file": str(output),
        "quality_flag_counts": standardized["quality_flags"].value_counts().to_dict(),
    }
    manifest_path = args.output_dir / "import_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"output={output}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
