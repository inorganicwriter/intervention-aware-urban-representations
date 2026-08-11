"""Audit and quarantine two public Wuhan listing snapshots from GitHub.

The repository has no explicit data license and no observation date or
coordinates.  The records are standardized for provenance review only and
must not enter a causal outcome table unless those issues are resolved.
"""

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
SOURCE_ID = "github_shiqinhuo_wuhan_house_price_crawler_master"
SOURCE_URL = "https://github.com/ShiqinHuo/wuhan_house_price_crawler"
SNAPSHOT_DATE = pd.Timestamp("2019-04-29")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def normalize(frame: pd.DataFrame, source_hash: str, batch: str) -> pd.DataFrame:
    required = {"单价", "地址", "小区", "年份", "总价", "户型", "朝向", "标题", "楼层", "面积"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Source CSV missing columns: {sorted(missing)}")

    raw_row_number = pd.RangeIndex(2, len(frame) + 2)
    result = pd.DataFrame(
        {
            "source_platform": "github_public_repository",
            "source_id": SOURCE_ID,
            "source_url": SOURCE_URL,
            "source_file_sha256": source_hash,
            "source_batch": batch,
            "raw_row_number": raw_row_number,
            "city_key": "wuhan",
            "district": "qiaokou" if batch == "qiaokou_hanzheng" else pd.NA,
            "submarket": "hanzheng_street" if batch == "qiaokou_hanzheng" else "donghu_high_tech",
            "community_name": frame["小区"].astype("string").str.strip(),
            "address": frame["地址"].astype("string").str.strip(),
            "listing_title": frame["标题"].astype("string").str.strip(),
            "layout": frame["户型"].astype("string").str.strip(),
            "orientation": frame["朝向"].astype("string").str.strip(),
            "floor_raw": frame["楼层"].astype("string").str.strip(),
            "built_year": _numeric(frame["年份"]).astype("Int64"),
            "unit_price_cny_m2": _numeric(frame["单价"]),
            "total_price_10k_cny": _numeric(frame["总价"]),
            "building_area_m2": _numeric(frame["面积"]),
            "bedroom_count": _numeric(
                frame["户型"].astype("string").str.extract(r"(\d+)室", expand=False)
            ),
            "observation_type": "second_hand_listing_snapshot",
            "price_type": "asking_price",
            "source_snapshot_date": SNAPSHOT_DATE,
            "snapshot_date_status": "inferred_from_repository_last_update",
            "deal_date": pd.NaT,
            "lon": pd.NA,
            "lat": pd.NA,
            "license_status": "no_explicit_repository_license",
        }
    )
    result["source_record_id"] = [
        hashlib.sha256(f"{SOURCE_ID}|{source_hash}|{row}".encode()).hexdigest()[:24]
        for row in raw_row_number
    ]
    result["quality_flags"] = (
        "no_coordinates;no_transaction_date;snapshot_date_inferred;no_explicit_license"
    )
    invalid_price = result["unit_price_cny_m2"].isna() | result["unit_price_cny_m2"].le(0)
    result.loc[invalid_price, "quality_flags"] += ";invalid_or_missing_unit_price"
    missing_location = result["community_name"].isna() | result["address"].isna()
    result.loc[missing_location, "quality_flags"] += ";missing_location_text"
    duplicate = frame.drop(columns=["Unnamed: 0"], errors="ignore").duplicated(keep=False)
    result.loc[duplicate.to_numpy(), "quality_flags"] += ";duplicate_source_attributes"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_OPEN_DATASET_DIR / SOURCE_ID,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STAGING_HOUSING_DIR / "quarantine" / SOURCE_ID,
    )
    args = parser.parse_args()

    inputs = {
        "donghu_high_tech": args.input_dir / "donghu_gaoxin_listings.csv",
        "qiaokou_hanzheng": args.input_dir / "qiaokou_hanzheng_listings.csv",
    }
    outputs = []
    hashes: dict[str, str] = {}
    for batch, path in inputs.items():
        hashes[batch] = sha256_file(path)
        source = pd.read_csv(path, encoding="utf-8-sig")
        outputs.append(normalize(source, hashes[batch], batch))
    standardized = pd.concat(outputs, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "housing_listing_snapshot.parquet"
    standardized.to_parquet(output, index=False)
    manifest = {
        "schema": "github_wuhan_listing_snapshot_quarantine_manifest_v1",
        "source_id": SOURCE_ID,
        "source_url": SOURCE_URL,
        "source_file_sha256": hashes,
        "output_rows": int(len(standardized)),
        "rows_by_batch": standardized["source_batch"].value_counts().to_dict(),
        "communities": int(standardized["community_name"].nunique()),
        "duplicate_source_attribute_rows": int(
            standardized["quality_flags"].str.contains("duplicate_source_attributes").sum()
        ),
        "valid_price_rows": int(standardized["unit_price_cny_m2"].gt(0).sum()),
        "formal_use_eligible": False,
        "monthly_outcome_ready": False,
        "blocking_issues": [
            "no_explicit_repository_or_data_license",
            "no_transaction_or_listing_observation_date",
            "no_coordinates",
            "narrow_and_nonrepresentative_submarket_coverage",
        ],
        "semantic_warning": (
            "These are asking-price snapshots, not completed transactions. The inferred "
            "2019-04-29 snapshot date is repository activity metadata, not a source field."
        ),
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
    raise SystemExit(main())
