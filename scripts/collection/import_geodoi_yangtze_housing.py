"""Standardize the open 2008-2018 Yangtze River Delta housing panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from urban_intervention.data.paths import (
    RAW_OPEN_DATASET_DIR,
    STAGING_HOUSING_DIR,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ID = "geodoi_geodb_2019_04_17_v1"
SOURCE_URL = "https://geodoi.ac.cn/WebCn/geodoi.aspx?Id=1270"

# Province is part of the key to distinguish Jiangsu Suzhou from Anhui
# Suzhou, and Zhejiang Taizhou from Jiangsu Taizhou.
RESEARCH_CITY_PAIRS: dict[tuple[str, str], str] = {
    ("上海", "上海"): "shanghai",
    ("江苏", "常州"): "changzhou",
    ("江苏", "南京"): "nanjing",
    ("江苏", "南通"): "nantong",
    ("江苏", "苏州"): "suzhou",
    ("江苏", "无锡"): "wuxi",
    ("江苏", "徐州"): "xuzhou",
    ("浙江", "杭州"): "hangzhou",
    ("浙江", "金华"): "jinhua",
    ("浙江", "宁波"): "ningbo",
    ("浙江", "绍兴"): "shaoxing",
    ("浙江", "台州"): "taizhou",
    ("浙江", "温州"): "wenzhou",
    ("安徽", "合肥"): "hefei",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_id(source_hash: str, level: str, raw_row: int, year: int) -> str:
    value = f"{SOURCE_ID}|{source_hash}|{level}|{raw_row}|{year}".encode()
    return hashlib.sha256(value).hexdigest()[:24]


def _city_keys(province: pd.Series, city: pd.Series) -> pd.Series:
    pairs = zip(
        province.astype("string").str.strip(), city.astype("string").str.strip(), strict=True
    )
    return pd.Series(
        [RESEARCH_CITY_PAIRS.get((str(prov), str(name))) for prov, name in pairs],
        dtype="string",
    )


def _add_quality_flags(frame: pd.DataFrame, duplicate_keys: list[str]) -> pd.DataFrame:
    frame["quality_flags"] = ""
    missing = frame["unit_price_cny_m2"].isna()
    frame.loc[missing, "quality_flags"] = "missing_price"
    nonpositive = frame["unit_price_cny_m2"].notna() & frame["unit_price_cny_m2"].le(0)
    frame.loc[nonpositive, "quality_flags"] = frame.loc[nonpositive, "quality_flags"].map(
        lambda value: ";".join(filter(None, [value, "nonpositive_price"]))
    )
    duplicate = frame.duplicated(duplicate_keys, keep=False)
    frame.loc[duplicate, "quality_flags"] = frame.loc[duplicate, "quality_flags"].map(
        lambda value: ";".join(filter(None, [value, "duplicate_spatial_unit_year"]))
    )
    return frame


def normalize_city(source: pd.DataFrame, source_hash: str) -> pd.DataFrame:
    years = [int(value) for value in source.iloc[3, 5:16]]
    if years != list(range(2008, 2019)):
        raise ValueError(f"Unexpected city-year columns: {years}")
    base = source.iloc[4:, :5].copy().reset_index(names="source_index")
    base.columns = [
        "source_index",
        "source_city_number",
        "province_cn",
        "province_en",
        "city_cn",
        "city_en",
    ]
    base["raw_row_number"] = base["source_index"] + 1
    prices = source.iloc[4:, 5:16].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)

    repeated = base.loc[base.index.repeat(len(years))].reset_index(drop=True)
    repeated["year"] = years * len(base)
    repeated["unit_price_cny_m2"] = prices.to_numpy().reshape(-1)
    repeated["city_key"] = _city_keys(repeated["province_cn"], repeated["city_cn"])
    repeated["source_record_id"] = [
        _record_id(source_hash, "city", int(row), int(year))
        for row, year in zip(repeated["raw_row_number"], repeated["year"], strict=True)
    ]
    repeated["source_id"] = SOURCE_ID
    repeated["source_url"] = SOURCE_URL
    repeated["source_file_sha256"] = source_hash
    repeated["spatial_unit"] = "city"
    repeated["temporal_unit"] = "year"
    repeated["unit"] = "cny_per_m2"
    repeated["source_city_number"] = pd.to_numeric(
        repeated["source_city_number"], errors="coerce"
    ).astype("Int64")
    return _add_quality_flags(repeated, ["province_cn", "city_cn", "year"])


def normalize_county(source: pd.DataFrame, source_hash: str) -> pd.DataFrame:
    years = [int(value) for value in source.iloc[3, 6:17]]
    if years != list(range(2008, 2019)):
        raise ValueError(f"Unexpected county-year columns: {years}")
    base = source.iloc[4:, [0, 1, 2, 3, 4, 5, 17, 18, 19]].copy().reset_index(names="source_index")
    base.columns = [
        "source_index",
        "province_cn",
        "province_en",
        "city_cn",
        "city_en",
        "district_county_cn",
        "district_county_en",
        "style_type",
        "note_cn",
        "note_en",
    ]
    base["raw_row_number"] = base["source_index"] + 1
    prices = source.iloc[4:, 6:17].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)

    repeated = base.loc[base.index.repeat(len(years))].reset_index(drop=True)
    repeated["year"] = years * len(base)
    repeated["unit_price_cny_m2"] = prices.to_numpy().reshape(-1)
    repeated["city_key"] = _city_keys(repeated["province_cn"], repeated["city_cn"])
    repeated["source_record_id"] = [
        _record_id(source_hash, "district_county", int(row), int(year))
        for row, year in zip(repeated["raw_row_number"], repeated["year"], strict=True)
    ]
    repeated["source_id"] = SOURCE_ID
    repeated["source_url"] = SOURCE_URL
    repeated["source_file_sha256"] = source_hash
    repeated["spatial_unit"] = "district_or_county"
    repeated["temporal_unit"] = "year"
    repeated["unit"] = "cny_per_m2"
    return _add_quality_flags(
        repeated,
        ["province_cn", "city_cn", "district_county_cn", "year"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_OPEN_DATASET_DIR
        / "geodoi_geodb_2019_04_17_v1"
        / "extracted"
        / "HousingPriceYangtzeRD_2008-2018.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STAGING_HOUSING_DIR / "auxiliary" / SOURCE_ID,
    )
    args = parser.parse_args()

    source_hash = sha256_file(args.input)
    county_source = pd.read_excel(args.input, sheet_name="Tab.1", header=None)
    city_source = pd.read_excel(args.input, sheet_name="Tab.2", header=None)
    county = normalize_county(county_source, source_hash)
    city = normalize_city(city_source, source_hash)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    city_path = args.output_dir / "city_housing_price.parquet"
    county_path = args.output_dir / "district_county_housing_price.parquet"
    city.to_parquet(city_path, index=False)
    county.to_parquet(county_path, index=False)

    research_city = city[city["city_key"].notna()]
    research_county = county[county["city_key"].notna()]
    covered = sorted(research_city["city_key"].dropna().unique().tolist())
    manifest = {
        "schema": "geodoi_yangtze_housing_import_manifest_v1",
        "source_id": SOURCE_ID,
        "source_url": SOURCE_URL,
        "source_file": str(args.input),
        "source_file_sha256": source_hash,
        "city_rows": int(len(city)),
        "source_cities": int(city[["province_cn", "city_cn"]].drop_duplicates().shape[0]),
        "district_county_rows": int(len(county)),
        "source_districts_counties": int(
            county[["province_cn", "city_cn", "district_county_cn"]].drop_duplicates().shape[0]
        ),
        "first_year": 2008,
        "last_year": 2018,
        "research_city_count": len(covered),
        "research_cities": covered,
        "research_city_rows": int(len(research_city)),
        "research_district_county_rows": int(len(research_county)),
        "semantic_warning": (
            "Annual city and district/county average prices are auxiliary trend and "
            "matching variables; they are not monthly 500 m grid outcomes."
        ),
        "monthly_outcome_ready": False,
        "city_output_file": str(city_path),
        "district_county_output_file": str(county_path),
    }
    manifest_path = args.output_dir / "import_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"city={city_path}")
    print(f"district_county={county_path}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
