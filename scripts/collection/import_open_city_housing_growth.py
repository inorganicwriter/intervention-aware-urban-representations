"""Standardize the open 182-city housing-price growth replication data.

The source contains annual *growth rates*, not housing price levels.  All
source rows are retained.  ``city_key`` is populated only when the source
city/province pair unambiguously matches one of the 44 research cities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from urban_intervention.data.paths import (  # noqa: E402
    RAW_OPEN_DATASET_DIR,
    STAGING_DIR,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ID = "mendeley_52kj9yzx5j_v2"
SOURCE_URL = "https://data.mendeley.com/datasets/52kj9yzx5j/2"

# Province is part of the key because the workbook contains two Fuzhous and
# two Suzhous.  Taizhou in the workbook is the Jiangsu city, whereas this
# project's Taizhou research unit is in Zhejiang, so it is intentionally not
# mapped.
RESEARCH_CITY_PAIRS: dict[tuple[str, str], str] = {
    ("Peking", "Peking"): "beijing",
    ("Changchun", "Jilin"): "changchun",
    ("Changsha", "Hunan"): "changsha",
    ("Changzhou", "Jiangsu"): "changzhou",
    ("Chengdu", "Szechwan"): "chengdu",
    ("Chongqing", "Chongqing"): "chongqing",
    ("Dalian", "Liaoning"): "dalian",
    ("Dongguan", "Guangdong"): "dongguan",
    ("Foshan", "Guangdong"): "foshan",
    ("Fuzhou", "Fujian"): "fuzhou",
    ("Guangzhou", "Guangdong"): "guangzhou",
    ("Hangzhou", "Zhejiang"): "hangzhou",
    ("Harbin", "Heilongjiang"): "harbin",
    ("Hefei", "Anhui"): "hefei",
    ("Hohhot", "Inner Mongolia"): "hohhot",
    ("Jinhua", "Zhejiang"): "jinhua",
    ("Luoyang", "Henan"): "luoyang",
    ("Nanchang", "Jiangxi"): "nanchang",
    ("Nanjing", "Jiangsu"): "nanjing",
    ("Nanning", "Guangxi"): "nanning",
    ("Nantong", "Jiangsu"): "nantong",
    ("Ningbo", "Zhejiang"): "ningbo",
    ("Shanghai", "Shanghai"): "shanghai",
    ("Shaoxing", "Zhejiang"): "shaoxing",
    ("Shenyang", "Liaoning"): "shenyang",
    ("Shenzhen", "Guangdong"): "shenzhen",
    ("Suzhou", "Jiangsu"): "suzhou",
    ("Taiyuan", "Shanxi"): "taiyuan",
    ("Tianjin", "Tianjin"): "tianjin",
    ("Wenzhou", "Zhejiang"): "wenzhou",
    ("Wuhan", "Hubei"): "wuhan",
    ("Wuxi", "Jiangsu"): "wuxi",
    ("Xiamen", "Fujian"): "xiamen",
    ("Xuzhou", "Jiangsu"): "xuzhou",
    ("Zhengzhou", "Henan"): "zhengzhou",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(frame: pd.DataFrame, source_sha256: str) -> pd.DataFrame:
    required = {
        "year",
        "city",
        "province",
        "city.1",
        "arbp",
        "crbp",
        "arbp2",
        "crbp2",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Source workbook missing columns: {sorted(missing)}")

    result = pd.DataFrame(
        {
            "source_id": SOURCE_ID,
            "source_url": SOURCE_URL,
            "source_file_sha256": source_sha256,
            "source_city_id": pd.to_numeric(frame["city.1"], errors="coerce").astype("Int64"),
            "city_name_en": frame["city"].astype("string").str.strip(),
            "province_name_en": frame["province"].astype("string").str.strip(),
            "year": pd.to_numeric(frame["year"], errors="coerce").astype("Int64"),
            "commodity_house_price_absolute_growth": pd.to_numeric(frame["arbp"], errors="coerce"),
            "commodity_house_price_relative_growth": pd.to_numeric(frame["crbp"], errors="coerce"),
            "residential_house_price_absolute_growth": pd.to_numeric(
                frame["arbp2"], errors="coerce"
            ),
            "residential_house_price_relative_growth": pd.to_numeric(
                frame["crbp2"], errors="coerce"
            ),
        }
    )
    pairs = zip(result["city_name_en"], result["province_name_en"], strict=True)
    result["city_key"] = pd.Series(
        [RESEARCH_CITY_PAIRS.get((str(city), str(province))) for city, province in pairs],
        dtype="string",
    )
    result["unit"] = "ratio"
    result["temporal_unit"] = "year"
    result["spatial_unit"] = "city"
    result["quality_flags"] = ""
    missing_measure = (
        result[
            [
                "commodity_house_price_absolute_growth",
                "commodity_house_price_relative_growth",
                "residential_house_price_absolute_growth",
                "residential_house_price_relative_growth",
            ]
        ]
        .isna()
        .any(axis=1)
    )
    result.loc[missing_measure, "quality_flags"] = "missing_housing_growth_measure"
    duplicate = result.duplicated(["source_city_id", "year"], keep=False)
    result.loc[duplicate, "quality_flags"] = result.loc[duplicate, "quality_flags"].map(
        lambda value: ";".join(filter(None, [value, "duplicate_source_city_year"]))
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_OPEN_DATASET_DIR / "mendeley_52kj9yzx5j_v2/Data.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STAGING_DIR / "housing/auxiliary/mendeley_52kj9yzx5j_v2",
    )
    args = parser.parse_args()

    source_hash = sha256_file(args.input)
    source = pd.read_excel(args.input)
    standardized = normalize(source, source_hash)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "city_housing_price_growth.parquet"
    standardized.to_parquet(output, index=False)

    research = standardized[standardized["city_key"].notna()]
    covered = sorted(research["city_key"].unique().tolist())
    manifest = {
        "schema": "open_city_housing_growth_manifest_v1",
        "source_id": SOURCE_ID,
        "source_url": SOURCE_URL,
        "license": "CC BY 4.0",
        "source_file": str(args.input),
        "source_file_sha256": source_hash,
        "source_rows": int(len(source)),
        "output_rows": int(len(standardized)),
        "source_city_ids": int(standardized["source_city_id"].nunique()),
        "first_year": int(standardized["year"].min()),
        "last_year": int(standardized["year"].max()),
        "research_city_count": len(covered),
        "research_cities": covered,
        "research_rows": int(len(research)),
        "unfiltered_acquisition": True,
        "semantic_warning": (
            "Four fields are annual real-price growth rates, not price levels; "
            "this city-level source cannot serve as a monthly 500 m outcome."
        ),
        "quality_flag_counts": standardized["quality_flags"].value_counts().to_dict(),
        "output_file": str(output),
    }
    manifest_path = args.output_dir / "import_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output={output}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
