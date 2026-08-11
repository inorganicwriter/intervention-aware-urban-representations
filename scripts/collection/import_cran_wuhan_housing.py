"""Standardize the open 2018 Wuhan second-hand housing cross-section from CRAN."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

from urban_intervention.data.paths import (
    RAW_OPEN_DATASET_DIR,
    STAGING_HOUSING_STANDARDIZED_DIR,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ID = "cran_hgwrr_0.6-2"
SOURCE_URL = "https://cran.r-project.org/package=hgwrr"
BATCH_ID = "cran_hgwrr_0.6-2_wuhan_2018_second_hand_housing"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rdata(path: Path) -> pd.DataFrame:
    try:
        import rdata
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Reading the CRAN .rda file requires the collection dependency 'rdata'."
        ) from exc
    objects = rdata.read_rda(path)
    frame = objects.get("wuhan.hp")
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("wuhan.hp is missing or is not a data frame")
    return frame


def _record_id(source_sha256: str, row_number: int) -> str:
    value = f"{SOURCE_ID}|{source_sha256}|{row_number}".encode()
    return hashlib.sha256(value).hexdigest()[:24]


def normalize(frame: pd.DataFrame, source_sha256: str) -> pd.DataFrame:
    required = {"Price", "BuildingArea", "lon", "lat", "group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Source data missing columns: {sorted(missing)}")

    rows = pd.RangeIndex(1, len(frame) + 1)
    log_price = pd.to_numeric(frame["Price"], errors="coerce")
    log_area = pd.to_numeric(frame["BuildingArea"], errors="coerce")
    group = pd.to_numeric(frame["group"], errors="coerce").astype("Int64")
    result = pd.DataFrame(
        {
            "source_record_id": [_record_id(source_sha256, int(row)) for row in rows],
            "source_platform": "cran_hgwrr",
            "acquisition_method": "open_software_package",
            "batch_id": BATCH_ID,
            "city_key": "wuhan",
            "district": pd.NA,
            "community_name": pd.NA,
            "community_id": group.map(
                lambda value: f"cran_hgwrr_group_{int(value)}" if pd.notna(value) else pd.NA
            ).astype("string"),
            "observation_type": "second_hand_property_observation",
            "price_type": "source_unspecified_second_hand_price",
            "spatial_unit": "community_point",
            "temporal_unit": "year",
            "unit": "cny_per_m2",
            "source_snapshot_date": pd.Timestamp("2018-12-31"),
            "deal_date": pd.NaT,
            "unit_price_cny_m2": log_price.map(
                lambda value: math.exp(value) if pd.notna(value) else math.nan
            ),
            "total_price_10k_cny": math.nan,
            "building_area_m2": log_area.map(
                lambda value: math.exp(value) if pd.notna(value) else math.nan
            ),
            "lon": pd.to_numeric(frame["lon"], errors="coerce"),
            "lat": pd.to_numeric(frame["lat"], errors="coerce"),
            "layout": pd.NA,
            "bedroom_count": math.nan,
            "floor_raw": pd.NA,
            "built_year": math.nan,
            "decoration": pd.NA,
            "property_type": "second_hand_residential",
            "source_url": SOURCE_URL,
            "source_page_id": pd.NA,
            "source_file_sha256": source_sha256,
            "raw_row_number": rows,
            "quality_flags": "source_date_year_only;source_price_log_reversed",
            "pipeline_version": "housing_authorized_import_v1",
            "source_log_unit_price": log_price,
            "source_log_building_area": log_area,
        }
    )

    invalid_coordinate = ~(result["lon"].between(113.5, 115.0) & result["lat"].between(29.5, 31.5))
    result.loc[invalid_coordinate, "quality_flags"] += ";invalid_wuhan_coordinate"
    duplicate = frame.drop(columns=["geometry"], errors="ignore").duplicated(keep=False)
    result.loc[duplicate.to_numpy(), "quality_flags"] += ";duplicate_source_attributes"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_OPEN_DATASET_DIR / "cran_hgwrr_0.6-2" / "hgwrr" / "data" / "wuhan.hp.rda",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STAGING_HOUSING_STANDARDIZED_DIR / BATCH_ID,
    )
    args = parser.parse_args()

    source_hash = sha256_file(args.input)
    source = load_rdata(args.input)
    standardized = normalize(source, source_hash)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "housing_observations.parquet"
    standardized.to_parquet(output, index=False)

    manifest = {
        "schema": "cran_hgwrr_wuhan_import_manifest_v1",
        "source_id": SOURCE_ID,
        "source_url": SOURCE_URL,
        "license": "GPL (>= 2)",
        "source_file": str(args.input),
        "source_file_sha256": source_hash,
        "source_rows": int(len(source)),
        "output_rows": int(len(standardized)),
        "communities": int(standardized["community_id"].nunique()),
        "observation_year": 2018,
        "longitude_range": [float(standardized["lon"].min()), float(standardized["lon"].max())],
        "latitude_range": [float(standardized["lat"].min()), float(standardized["lat"].max())],
        "median_unit_price_cny_m2": float(standardized["unit_price_cny_m2"].median()),
        "duplicate_source_attribute_rows": int(
            standardized["quality_flags"].str.contains("duplicate_source_attributes").sum()
        ),
        "semantic_warning": (
            "The source identifies 2018 only, not transaction months. Price and area "
            "were supplied as natural logarithms and are exponentiated here. The source "
            "does not specify whether each price is a completed transaction or listing."
        ),
        "grid_role": "spatial_cross_section_and_matching_covariate_only",
        "monthly_outcome_ready": False,
        "output_file": str(output),
    }
    manifest_path = args.output_dir / "import_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"output={output}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
