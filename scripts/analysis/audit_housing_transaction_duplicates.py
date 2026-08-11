"""Audit purchased-Lianjia versus Wayback/Beike transaction overlap."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.data.paths import (  # noqa: E402
    COMMUNITY_SOURCE_CROSSWALK,
    OUTPUT_HOUSING_FUSION_DIR,
    RAW_WAYBACK_PARSED_DIR,
    STAGING_LIANJIA_TRANSACTIONS_DIR,
)

TX_DIR = STAGING_LIANJIA_TRANSACTIONS_DIR
WAYBACK_DIR = RAW_WAYBACK_PARSED_DIR
CROSSWALK_PATH = COMMUNITY_SOURCE_CROSSWALK
DETAIL_PATH = OUTPUT_HOUSING_FUSION_DIR / "cross_source_transaction_duplicates.csv"
SUMMARY_PATH = OUTPUT_HOUSING_FUSION_DIR / "cross_source_transaction_duplicate_summary.json"


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    return re.sub(r"[\s\-—_·•,，、。.;；:：()（）\[\]【】]+", "", text)


def main() -> int:
    crosswalk = pd.read_parquet(CROSSWALK_PATH)
    lj_map = crosswalk[crosswalk["source"] == "lianjia_purchased"][
        ["city_key", "normalized_name", "community_id"]
    ].drop_duplicates(["city_key", "normalized_name"])
    lj = (
        ds.dataset(str(TX_DIR), format="parquet")
        .to_table(
            columns=[
                "source_record_id",
                "city_key",
                "community_name_normalized",
                "deal_date",
                "unit_price_cny_m2",
                "building_area_m2",
                "total_price_10k_cny",
                "is_valid",
            ]
        )
        .to_pandas()
    )
    lj = lj[lj["is_valid"]].merge(
        lj_map.rename(columns={"normalized_name": "community_name_normalized"}),
        on=["city_key", "community_name_normalized"],
        how="left",
        validate="many_to_one",
    )
    lj["deal_date_norm"] = pd.to_datetime(lj["deal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    lj = lj[lj["community_id"].notna() & lj["deal_date_norm"].notna()].copy()

    wb_lookup = crosswalk[crosswalk["source"].fillna("").str.startswith("wayback_")][
        ["city_key", "source", "normalized_name", "community_id"]
    ].drop_duplicates(["city_key", "source", "normalized_name"])
    lookup = {
        (row.city_key, row.source, row.normalized_name): row.community_id
        for row in wb_lookup.itertuples(index=False)
    }
    wb_rows = []
    for path in sorted(WAYBACK_DIR.glob("*_wayback_*chengjiao.csv")):
        city = path.name.split("_wayback_", 1)[0]
        suffix = path.stem.split("_wayback_", 1)[1]
        source = f"wayback_{suffix}"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if not {"community", "deal_date", "unit_price"}.issubset(frame.columns):
            continue
        frame["normalized_name"] = frame["community"].map(normalize_name)
        frame["community_id"] = [
            lookup.get((city, source, name)) for name in frame["normalized_name"]
        ]
        frame["deal_date_norm"] = pd.to_datetime(frame["deal_date"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
        frame["unit_price_wayback"] = pd.to_numeric(frame["unit_price"], errors="coerce")
        frame["area_wayback"] = pd.to_numeric(frame.get("area_m2"), errors="coerce")
        frame["total_price_wayback"] = pd.to_numeric(frame.get("total_price"), errors="coerce")
        frame["wayback_record_id"] = [
            hashlib.sha1(f"{path.relative_to(ROOT)}|{index + 2}".encode()).hexdigest()
            for index in range(len(frame))
        ]
        frame["wayback_source"] = source
        frame["city_key"] = city
        wb_rows.append(
            frame[
                [
                    "wayback_record_id",
                    "wayback_source",
                    "city_key",
                    "community_id",
                    "community",
                    "deal_date_norm",
                    "unit_price_wayback",
                    "area_wayback",
                    "total_price_wayback",
                ]
            ]
        )
    wayback = pd.concat(wb_rows, ignore_index=True) if wb_rows else pd.DataFrame()
    wayback = wayback[
        wayback["community_id"].notna()
        & wayback["deal_date_norm"].notna()
        & wayback["unit_price_wayback"].between(500, 500_000)
    ].copy()

    candidates = wayback.merge(
        lj[
            [
                "source_record_id",
                "city_key",
                "community_id",
                "deal_date_norm",
                "unit_price_cny_m2",
                "building_area_m2",
                "total_price_10k_cny",
            ]
        ],
        on=["city_key", "community_id", "deal_date_norm"],
        how="inner",
    )
    if not candidates.empty:
        candidates["price_relative_difference"] = (
            candidates["unit_price_wayback"] - candidates["unit_price_cny_m2"]
        ).abs() / candidates[["unit_price_wayback", "unit_price_cny_m2"]].mean(axis=1)
        candidates["area_relative_difference"] = np.where(
            candidates["area_wayback"].notna() & candidates["building_area_m2"].notna(),
            (candidates["area_wayback"] - candidates["building_area_m2"]).abs()
            / candidates[["area_wayback", "building_area_m2"]].mean(axis=1),
            np.nan,
        )
        candidates["duplicate_class"] = np.where(
            (candidates["price_relative_difference"] <= 0.001)
            & (
                candidates["area_relative_difference"].isna()
                | (candidates["area_relative_difference"] <= 0.005)
            ),
            "exact_or_near_exact",
            np.where(
                (candidates["price_relative_difference"] <= 0.02)
                & (
                    candidates["area_relative_difference"].isna()
                    | (candidates["area_relative_difference"] <= 0.02)
                ),
                "probable",
                "same_community_date_not_duplicate",
            ),
        )
        rank = {"exact_or_near_exact": 0, "probable": 1, "same_community_date_not_duplicate": 2}
        candidates["class_rank"] = candidates["duplicate_class"].map(rank)
        candidates = candidates.sort_values(
            ["wayback_record_id", "class_rank", "price_relative_difference"]
        ).drop_duplicates("wayback_record_id", keep="first")
    duplicates = candidates[
        candidates["duplicate_class"].isin(["exact_or_near_exact", "probable"])
    ].copy()
    duplicates["canonical_source"] = "lianjia_purchased"
    duplicates["wayback_role"] = "auxiliary_duplicate"
    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    duplicates.drop(columns=["class_rank"], errors="ignore").to_csv(
        DETAIL_PATH, index=False, encoding="utf-8-sig"
    )
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "wayback_transaction_rows_valid_and_resolved": int(len(wayback)),
        "same_community_date_candidates": int(len(candidates)),
        "exact_or_near_exact_duplicates": int(
            (candidates["duplicate_class"] == "exact_or_near_exact").sum()
        ),
        "probable_duplicates": int((candidates["duplicate_class"] == "probable").sum()),
        "canonical_source_for_all_flagged": "lianjia_purchased",
        "detail_rows": int(len(duplicates)),
        "raw_files_modified": False,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
