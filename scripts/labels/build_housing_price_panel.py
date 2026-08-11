"""Build the canonical source-preserving 500 m housing price panel.

Contract
--------
* observed price, location, and time define admission;
* listing and transaction prices are both market-price observations;
* a same-record, same-month completed transaction supersedes its initial
  listing only in the canonical aggregate (both remain in observations);
* aggregation is source-balanced and never performs hedonic adjustment;
* annual/quarterly observations are never copied into artificial months;
* raw inputs are never modified by this builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.config.project import ACTIVE_CITIES, CITIES  # noqa: E402
from urban_intervention.data.paths import (  # noqa: E402
    COMMUNITY_REGISTRY,
    COMMUNITY_SOURCE_CROSSWALK,
    HOUSING_OBSERVATIONS_DIR,
    OUTPUT_HOUSING_FUSION_DIR,
    OUTPUT_HOUSING_PANEL_DIR,
    PANEL_HOUSING_MONTHLY_DIR,
    PANEL_HOUSING_QUARTERLY_DIR,
    PANEL_HOUSING_YEARLY_DIR,
    RAW_ANJUKE_DIR,
    RAW_GRID_PRICE_2023_05_DIR,
    RAW_WAYBACK_PARSED_DIR,
    STAGING_HOUSING_STANDARDIZED_DIR,
    STAGING_LIANJIA_TRANSACTIONS_DIR,
    grid_path,
)
from urban_intervention.pipelines.housing.panel import (  # noqa: E402
    attach_time_fields,
    finalize_observations,
    source_balanced_panel,
)

LIANJIA_DIR = STAGING_LIANJIA_TRANSACTIONS_DIR
STANDARDIZED_DIR = STAGING_HOUSING_STANDARDIZED_DIR
WAYBACK_DIR = RAW_WAYBACK_PARSED_DIR
ANJUKE_DIR = RAW_ANJUKE_DIR
GRID2023_DIR = RAW_GRID_PRICE_2023_05_DIR
REGISTRY_PATH = COMMUNITY_REGISTRY
CROSSWALK_PATH = COMMUNITY_SOURCE_CROSSWALK
WAYBACK_DUPLICATES = OUTPUT_HOUSING_FUSION_DIR / "cross_source_transaction_duplicates.csv"

OBS_DIR = HOUSING_OBSERVATIONS_DIR
MONTH_DIR = PANEL_HOUSING_MONTHLY_DIR
QUARTER_DIR = PANEL_HOUSING_QUARTERLY_DIR
YEAR_DIR = PANEL_HOUSING_YEARLY_DIR
REPORT_DIR = OUTPUT_HOUSING_PANEL_DIR


OPEN_POINT_SOURCES = {
    "beijing": STANDARDIZED_DIR
    / "figshare_14398907_v1_beijing_2020_transactions"
    / "housing_observations.parquet",
    "shanghai": STANDARDIZED_DIR
    / "mendeley_hwfghkygy6_v1_shanghai_transactions"
    / "housing_observations.parquet",
    "chengdu": STANDARDIZED_DIR
    / "mendeley_wpv5zn9rxp_v1_chengdu_fang_transactions"
    / "housing_observations.parquet",
    "wuhan": STANDARDIZED_DIR
    / "cran_hgwrr_0.6-2_wuhan_2018_second_hand_housing"
    / "housing_observations.parquet",
    "nanning": STANDARDIZED_DIR
    / "mendeley_pj2zff4p9m_v4_nanning_2018_community_prices"
    / "housing_observations.parquet",
}
NANJING_QUARTER = (
    STANDARDIZED_DIR / "geodoi_geodb_2018_04_08_v1" / "community_quarter_housing.parquet"
)


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    return "".join(character for character in text if character.isalnum())


def parse_number(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else np.nan


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def stable_row_id(path: Path, row_number: int) -> str:
    relative = path.relative_to(ROOT)
    return hashlib.sha1(f"{relative}|{row_number}".encode()).hexdigest()


def _base_frame(length: int) -> pd.DataFrame:
    return pd.DataFrame(index=pd.RangeIndex(length))


def _map_community_ids(
    frame: pd.DataFrame,
    city: str,
    source: str,
    name_column: str,
    crosswalk: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    result["_normalized_name"] = result[name_column].map(normalize_name)
    source_map = crosswalk[(crosswalk["city_key"] == city) & (crosswalk["source"] == source)][
        ["normalized_name", "community_id"]
    ].drop_duplicates("normalized_name")
    result = result.merge(
        source_map.rename(columns={"normalized_name": "_normalized_name"}),
        on="_normalized_name",
        how="left",
        validate="many_to_one",
    )
    fallback = registry[registry["city_key"] == city][
        ["normalized_name", "community_id"]
    ].drop_duplicates("normalized_name")
    fallback_map = fallback.set_index("normalized_name")["community_id"]
    result["community_id"] = result["community_id"].fillna(
        result["_normalized_name"].map(fallback_map)
    )
    return result


def _attach_registry_coordinates(frame: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    coordinates = registry[["community_id", "centroid_lon", "centroid_lat"]].drop_duplicates(
        "community_id"
    )
    result = frame.merge(coordinates, on="community_id", how="left", validate="many_to_one")
    return result.rename(
        columns={"centroid_lon": "longitude_wgs84", "centroid_lat": "latitude_wgs84"}
    )


def load_lianjia(city: str, crosswalk: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_record_id",
        "city_key",
        "community_name",
        "community_name_normalized",
        "listing_date",
        "deal_date",
        "listing_price_10k_cny",
        "building_area_m2",
        "unit_price_cny_m2",
        "total_price_10k_cny",
        "lon",
        "lat",
        "coordinate_valid",
        "community_valid",
        "is_valid",
        "quality_flags",
    ]
    # The staging layout is batch/city.parquet.  Read only the named city
    # files; scanning the full 2.3 M-row dataset once per city is needlessly
    # expensive because the directory is not Hive-partitioned.
    paths = sorted(LIANJIA_DIR.glob(f"*/{city}.parquet"))
    if not paths:
        return pd.DataFrame()
    frame = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in paths],
        ignore_index=True,
    )
    if frame.empty:
        return pd.DataFrame()
    mapping = crosswalk[
        (crosswalk["city_key"] == city) & (crosswalk["source"] == "lianjia_purchased")
    ][["normalized_name", "community_id"]].drop_duplicates("normalized_name")
    frame = frame.merge(
        mapping.rename(columns={"normalized_name": "community_name_normalized"}),
        on="community_name_normalized",
        how="left",
        validate="many_to_one",
    )

    transaction = _base_frame(len(frame))
    transaction["source_record_id"] = frame["source_record_id"].astype(str)
    transaction["observation_id"] = (
        "lianjia_purchased:transaction:" + transaction["source_record_id"]
    )
    transaction["source_id"] = "lianjia_purchased"
    transaction["city_key"] = city
    transaction["community_id"] = frame["community_id"]
    transaction["community_name"] = frame["community_name"]
    transaction["price_stage"] = "transaction"
    transaction["observation_type"] = "individual_property"
    transaction["price_cny_m2"] = frame["unit_price_cny_m2"]
    transaction["longitude_wgs84"] = frame["lon"]
    transaction["latitude_wgs84"] = frame["lat"]
    transaction["coordinate_source"] = "source_point"
    transaction["location_confidence"] = "high"
    transaction["raw_quality_flags"] = frame["quality_flags"].fillna("")
    transaction["duplicate_class"] = ""
    transaction["canonical_for_aggregation"] = frame["is_valid"].fillna(False)
    transaction["_date"] = frame["deal_date"]
    transaction = attach_time_fields(transaction, "_date", "day")

    transaction_key = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["deal_date"], errors="coerce"),
            "community": frame["community_id"].fillna(frame["community_name_normalized"]),
            "price": pd.to_numeric(frame["unit_price_cny_m2"], errors="coerce").round(0),
            "area": pd.to_numeric(frame["building_area_m2"], errors="coerce").round(1),
            "total": pd.to_numeric(frame["total_price_10k_cny"], errors="coerce").round(2),
        }
    )
    transaction_duplicate = (
        transaction_key.duplicated(keep="first") & transaction["canonical_for_aggregation"]
    )
    transaction.loc[transaction_duplicate, "duplicate_class"] = "within_source_exact"
    transaction.loc[transaction_duplicate, "canonical_for_aggregation"] = False

    listing_date = pd.to_datetime(frame["listing_date"], errors="coerce")
    listing_price = pd.to_numeric(frame["listing_price_10k_cny"], errors="coerce")
    area = pd.to_numeric(frame["building_area_m2"], errors="coerce")
    listing_mask = listing_date.notna() & listing_price.gt(0) & area.gt(0)
    source = frame.loc[listing_mask].copy().reset_index(drop=True)
    listing = _base_frame(len(source))
    listing["source_record_id"] = source["source_record_id"].astype(str).to_numpy()
    listing["observation_id"] = "lianjia_purchased:listing:" + listing["source_record_id"]
    listing["source_id"] = "lianjia_purchased"
    listing["city_key"] = city
    listing["community_id"] = source["community_id"].to_numpy()
    listing["community_name"] = source["community_name"].to_numpy()
    listing["price_stage"] = "listing"
    listing["observation_type"] = "individual_property"
    listing["price_cny_m2"] = (
        pd.to_numeric(source["listing_price_10k_cny"], errors="coerce")
        * 10_000
        / pd.to_numeric(source["building_area_m2"], errors="coerce")
    ).to_numpy()
    listing["longitude_wgs84"] = source["lon"].to_numpy()
    listing["latitude_wgs84"] = source["lat"].to_numpy()
    listing["coordinate_source"] = "source_point"
    listing["location_confidence"] = "high"
    listing["raw_quality_flags"] = ""
    listing["duplicate_class"] = ""
    listing["canonical_for_aggregation"] = (
        source["coordinate_valid"].fillna(False) & source["community_valid"].fillna(False)
    ).to_numpy()
    listing["_date"] = source["listing_date"].to_numpy()
    listing = attach_time_fields(listing, "_date", "day")

    listing_key = pd.DataFrame(
        {
            "date": pd.to_datetime(source["listing_date"], errors="coerce"),
            "community": source["community_id"]
            .fillna(source["community_name_normalized"])
            .to_numpy(),
            "price": pd.to_numeric(listing["price_cny_m2"], errors="coerce").round(0),
            "area": pd.to_numeric(source["building_area_m2"], errors="coerce").round(1).to_numpy(),
        }
    )
    listing_duplicate = listing_key.duplicated(keep="first") & listing["canonical_for_aggregation"]
    listing.loc[listing_duplicate, "duplicate_class"] = "within_source_exact"
    listing.loc[listing_duplicate, "canonical_for_aggregation"] = False

    deal_month = pd.to_datetime(source["deal_date"], errors="coerce").dt.to_period("M")
    list_month = pd.to_datetime(source["listing_date"], errors="coerce").dt.to_period("M")
    same_month = (deal_month == list_month).fillna(False).to_numpy()
    source_transaction_valid = source["is_valid"].fillna(False).to_numpy()
    superseded = (
        same_month & source_transaction_valid & listing["canonical_for_aggregation"].to_numpy()
    )
    listing.loc[superseded, "duplicate_class"] = "lifecycle_superseded_same_month"
    listing.loc[superseded, "canonical_for_aggregation"] = False
    return pd.concat([transaction, listing], ignore_index=True, sort=False)


def load_standardized(city: str) -> pd.DataFrame:
    path = OPEN_POINT_SOURCES.get(city)
    if path is None or not path.exists():
        return pd.DataFrame()
    source = pd.read_parquet(path)
    source = source[source["city_key"].fillna("").eq(city)].copy()
    if source.empty:
        return pd.DataFrame()
    result = _base_frame(len(source))
    result["source_record_id"] = source["source_record_id"].astype(str).to_numpy()
    source_id = source["batch_id"].fillna(source["source_platform"]).astype(str)
    result["source_id"] = source_id.to_numpy()
    result["observation_id"] = source_id.to_numpy() + ":" + result["source_record_id"]
    result["city_key"] = city
    result["community_id"] = source["community_id"].astype("string").to_numpy()
    result["community_name"] = source["community_name"].astype("string").to_numpy()
    price_type = source["price_type"].fillna("").astype(str).str.lower()
    result["price_stage"] = np.where(
        price_type.str.contains("transaction"),
        "transaction",
        np.where(price_type.str.contains("listing"), "listing", "platform_estimate"),
    )
    result["observation_type"] = (
        source["observation_type"].fillna("source_observation").astype(str).to_numpy()
    )
    result["price_cny_m2"] = source["unit_price_cny_m2"].to_numpy()
    result["longitude_wgs84"] = source["lon"].to_numpy()
    result["latitude_wgs84"] = source["lat"].to_numpy()
    result["coordinate_source"] = "source_point"
    unresolved_chengdu = city == "chengdu"
    result["location_confidence"] = "blocked_unresolved_crs" if unresolved_chengdu else "high"
    result["raw_quality_flags"] = source["quality_flags"].fillna("").astype(str).to_numpy()
    result["duplicate_class"] = ""

    temporal = source["temporal_unit"].fillna("").astype(str).str.lower()
    dates = pd.to_datetime(source["deal_date"], errors="coerce").fillna(
        pd.to_datetime(source["source_snapshot_date"], errors="coerce")
    )
    result["_date"] = dates.to_numpy()
    if temporal.eq("year").all():
        result = attach_time_fields(result, "_date", "year")
    elif temporal.eq("month").all():
        result = attach_time_fields(result, "_date", "month")
    else:
        result = attach_time_fields(result, "_date", "day")
    price_valid = pd.to_numeric(result["price_cny_m2"], errors="coerce").gt(0)
    coordinate_valid = pd.to_numeric(result["longitude_wgs84"], errors="coerce").between(
        70, 140
    ) & pd.to_numeric(result["latitude_wgs84"], errors="coerce").between(10, 60)
    fatal = result["raw_quality_flags"].str.contains(
        "cross_city_record|missing_transaction_unit_price|nonpositive_unit_price",
        regex=True,
        na=False,
    )
    result["canonical_for_aggregation"] = (
        price_valid & coordinate_valid & ~fatal & ~unresolved_chengdu
    )
    duplicate_key = pd.DataFrame(
        {
            "date": result["observed_date"],
            "price": pd.to_numeric(result["price_cny_m2"], errors="coerce").round(0),
            "lon": pd.to_numeric(result["longitude_wgs84"], errors="coerce").round(6),
            "lat": pd.to_numeric(result["latitude_wgs84"], errors="coerce").round(6),
            "area": pd.to_numeric(source["building_area_m2"], errors="coerce").round(1).to_numpy(),
        }
    )
    duplicates = duplicate_key.duplicated(keep="first") & result["canonical_for_aggregation"]
    result.loc[duplicates, "duplicate_class"] = "within_source_exact"
    result.loc[duplicates, "canonical_for_aggregation"] = False
    return result


def load_nanjing_quarter() -> pd.DataFrame:
    if not NANJING_QUARTER.exists():
        return pd.DataFrame()
    source = pd.read_parquet(NANJING_QUARTER)
    result = _base_frame(len(source))
    result["source_record_id"] = source["source_record_id"].astype(str).to_numpy()
    result["source_id"] = "geodoi_nanjing_quarter"
    result["observation_id"] = "geodoi_nanjing_quarter:" + result["source_record_id"]
    result["city_key"] = "nanjing"
    result["community_id"] = (
        "geodoi_nanjing:" + source["source_community_id"].astype(str).to_numpy()
    )
    result["community_name"] = source["community_name_cn"].astype("string").to_numpy()
    result["price_stage"] = "transaction"
    result["observation_type"] = "community_quarter_statistic"
    result["price_cny_m2"] = source["sale_price_cny_m2"].to_numpy()
    result["longitude_wgs84"] = source["lon"].to_numpy()
    result["latitude_wgs84"] = source["lat"].to_numpy()
    result["coordinate_source"] = "source_geographic_degrees_crs_unspecified"
    result["location_confidence"] = "medium_unresolved_crs"
    result["raw_quality_flags"] = source["quality_flags"].fillna("").astype(str).to_numpy()
    result["duplicate_class"] = ""
    result["canonical_for_aggregation"] = (
        pd.to_numeric(result["price_cny_m2"], errors="coerce").gt(0)
        & pd.to_numeric(result["longitude_wgs84"], errors="coerce").between(70, 140)
        & pd.to_numeric(result["latitude_wgs84"], errors="coerce").between(10, 60)
    )
    result["_date"] = source["period_start"].to_numpy()
    return attach_time_fields(result, "_date", "quarter")


def load_wayback(
    city: str,
    crosswalk: pd.DataFrame,
    registry: pd.DataFrame,
    duplicate_ids: dict[str, str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(WAYBACK_DIR.glob(f"{city}_wayback_*.csv")):
        source = pd.read_csv(path, encoding="utf-8-sig")
        if not {"community", "unit_price"}.issubset(source.columns):
            continue
        suffix = path.stem.split("_wayback_", 1)[1]
        source_id = f"wayback_{suffix}"
        source = _map_community_ids(source, city, source_id, "community", crosswalk, registry)
        source = _attach_registry_coordinates(source, registry)
        result = _base_frame(len(source))
        ids = [stable_row_id(path, index + 2) for index in range(len(source))]
        result["source_record_id"] = ids
        result["source_id"] = source_id
        result["observation_id"] = source_id + ":" + result["source_record_id"]
        result["city_key"] = city
        result["community_id"] = source["community_id"].astype("string").to_numpy()
        result["community_name"] = source["community"].astype("string").to_numpy()
        is_transaction = suffix.endswith("chengjiao")
        result["price_stage"] = "transaction" if is_transaction else "listing"
        result["observation_type"] = (
            "individual_property" if is_transaction else "community_snapshot"
        )
        result["price_cny_m2"] = pd.to_numeric(source["unit_price"], errors="coerce").to_numpy()
        result["longitude_wgs84"] = source["longitude_wgs84"].to_numpy()
        result["latitude_wgs84"] = source["latitude_wgs84"].to_numpy()
        result["coordinate_source"] = "community_registry_centroid"
        result["location_confidence"] = np.where(
            source["community_id"].notna(), "medium_cross_source_match", "unlocated"
        )
        result["raw_quality_flags"] = ""
        result["duplicate_class"] = [duplicate_ids.get(record_id, "") for record_id in ids]
        if is_transaction:
            result["_date"] = pd.to_datetime(source.get("deal_date"), errors="coerce").to_numpy()
            result = attach_time_fields(result, "_date", "day")
        else:
            snapshot = pd.to_datetime(
                source.get("snapshot_date").astype("string"), format="%Y%m%d%H%M%S", errors="coerce"
            )
            result["_date"] = snapshot.to_numpy()
            result = attach_time_fields(result, "_date", "snapshot_month")
        result["canonical_for_aggregation"] = (
            pd.to_numeric(result["price_cny_m2"], errors="coerce").gt(0)
            & result["observed_date"].notna()
            & pd.to_numeric(result["longitude_wgs84"], errors="coerce").notna()
            & ~result["duplicate_class"].isin(["exact_or_near_exact", "probable"])
        )
        rows.append(result)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def load_anjuke(city: str, registry: pd.DataFrame) -> pd.DataFrame:
    city_name = CITIES[city]["name"]
    paths = sorted(ANJUKE_DIR.glob(f"{city_name}*_house.csv"))
    if not paths:
        return pd.DataFrame()
    path = paths[0]
    source = pd.read_csv(path)
    if source.empty:
        return pd.DataFrame()
    columns = list(source.columns)
    source = source.rename(
        columns={
            columns[0]: "raw_community_id",
            columns[1]: "community_name",
            columns[7]: "unit_price_raw",
        }
    )
    registry_city = registry[registry["city_key"] == city][
        ["community_id", "anjuke_source_id", "centroid_lon", "centroid_lat"]
    ].drop_duplicates("anjuke_source_id")
    source["raw_community_id"] = source["raw_community_id"].fillna("").astype(str)
    source = source.merge(
        registry_city,
        left_on="raw_community_id",
        right_on="anjuke_source_id",
        how="left",
        validate="many_to_one",
    )
    result = _base_frame(len(source))
    result["source_record_id"] = [stable_row_id(path, index + 2) for index in range(len(source))]
    result["source_id"] = "anjuke_cross_section"
    result["observation_id"] = "anjuke_cross_section:" + result["source_record_id"]
    result["city_key"] = city
    result["community_id"] = source["community_id"].astype("string").to_numpy()
    result["community_name"] = source["community_name"].astype("string").to_numpy()
    result["price_stage"] = "listing"
    result["observation_type"] = "individual_property"
    result["price_cny_m2"] = source["unit_price_raw"].map(parse_number).to_numpy()
    result["longitude_wgs84"] = source["centroid_lon"].to_numpy()
    result["latitude_wgs84"] = source["centroid_lat"].to_numpy()
    result["coordinate_source"] = "anjuke_community_registry_centroid"
    result["location_confidence"] = np.where(source["community_id"].notna(), "medium", "unlocated")
    result["raw_quality_flags"] = "assigned_snapshot_year_2025"
    result["duplicate_class"] = ""
    result["canonical_for_aggregation"] = (
        pd.to_numeric(result["price_cny_m2"], errors="coerce").gt(0)
        & pd.to_numeric(result["longitude_wgs84"], errors="coerce").notna()
    )
    result["_date"] = pd.Timestamp("2025-01-01")
    return attach_time_fields(result, "_date", "year")


def load_grid2023(city: str) -> pd.DataFrame:
    city_name = CITIES[city]["name"]
    paths = sorted(GRID2023_DIR.glob(f"{city_name}*/表格/*房价数据.csv"))
    if not paths:
        return pd.DataFrame()
    path = paths[0]
    source = pd.read_csv(path)
    result = _base_frame(len(source))
    result["source_record_id"] = [stable_row_id(path, index + 2) for index in range(len(source))]
    result["source_id"] = "grid_2023_may"
    result["observation_id"] = "grid_2023_may:" + result["source_record_id"]
    result["city_key"] = city
    result["community_id"] = ""
    result["community_name"] = ""
    result["price_stage"] = "platform_estimate"
    result["observation_type"] = "source_grid_average"
    result["price_cny_m2"] = pd.to_numeric(source["avgprice"], errors="coerce").to_numpy()
    result["longitude_wgs84"] = pd.to_numeric(source["centerlon"], errors="coerce").to_numpy()
    result["latitude_wgs84"] = pd.to_numeric(source["centerlat"], errors="coerce").to_numpy()
    result["coordinate_source"] = "source_grid_centroid"
    result["location_confidence"] = "high"
    result["raw_quality_flags"] = ""
    result["duplicate_class"] = ""
    result["canonical_for_aggregation"] = pd.to_numeric(result["price_cny_m2"], errors="coerce").gt(
        0
    )
    result["_date"] = pd.Timestamp("2023-05-01")
    return attach_time_fields(result, "_date", "month")


def map_points_to_reference_grid(frame: pd.DataFrame, city: str) -> tuple[pd.DataFrame, dict]:
    result = frame.copy()
    result["grid_id"] = ""
    lon = pd.to_numeric(result["longitude_wgs84"], errors="coerce")
    lat = pd.to_numeric(result["latitude_wgs84"], errors="coerce")
    valid = lon.between(70, 140) & lat.between(10, 60)
    if not valid.any():
        return result, {"located_rows": 0, "unlocated_rows": int(len(result))}
    unique = pd.DataFrame(
        {"lon": lon[valid].round(7), "lat": lat[valid].round(7)}
    ).drop_duplicates()
    grids = pd.read_parquet(
        grid_path(city), columns=["grid_id", "row", "col", "centroid_lon", "centroid_lat"]
    )
    unique = unique.reset_index(drop=True)
    # Reference grids were generated as exact 500 m cells in the configured
    # projected CRS, then clipped to the municipal boundary.  Recovering the
    # projected origin from retained centroids is exact and avoids parsing tens
    # of thousands of WKT polygons for every city.
    transformer = Transformer.from_crs("EPSG:4326", CITIES[city]["projected_crs"], always_xy=True)
    sample = grids.iloc[:: max(1, len(grids) // 2_000)].copy()
    sample_x, sample_y = transformer.transform(sample["centroid_lon"], sample["centroid_lat"])
    origin_x = float(np.median(sample_x - (sample["col"].to_numpy() + 0.5) * 500.0))
    origin_y = float(np.median(sample_y - (sample["row"].to_numpy() + 0.5) * 500.0))
    residual = max(
        float(np.max(np.abs(sample_x - (origin_x + (sample["col"].to_numpy() + 0.5) * 500.0)))),
        float(np.max(np.abs(sample_y - (origin_y + (sample["row"].to_numpy() + 0.5) * 500.0)))),
    )
    if residual > 1.0:
        raise RuntimeError(
            f"{city} reference grid is not a regular projected 500 m grid: residual={residual}"
        )
    x, y = transformer.transform(unique["lon"].to_numpy(), unique["lat"].to_numpy())
    cols = np.floor((x - origin_x) / 500.0).astype(np.int64)
    rows = np.floor((y - origin_y) / 500.0).astype(np.int64)
    candidates = np.array(
        [f"g{row:05d}x{col:05d}" for row, col in zip(rows, cols, strict=False)], dtype=object
    )
    retained = set(grids["grid_id"].astype(str))
    assigned = np.array(
        [candidate if candidate in retained else "" for candidate in candidates], dtype=object
    )
    unique["grid_id"] = assigned
    lookup = unique.set_index(["lon", "lat"])["grid_id"]
    keys = list(zip(lon.round(7), lat.round(7), strict=False))
    result["grid_id"] = [
        lookup.get(key, "") if ok else "" for key, ok in zip(keys, valid, strict=False)
    ]
    return result, {
        "unique_coordinate_pairs": int(len(unique)),
        "located_coordinate_pairs": int((unique["grid_id"] != "").sum()),
        "located_rows": int(result["grid_id"].ne("").sum()),
        "unlocated_rows": int(result["grid_id"].eq("").sum()),
        "grid_origin_residual_m": residual,
    }


def add_source_signature(source: pd.DataFrame, panel: pd.DataFrame, period: str) -> pd.DataFrame:
    if source.empty or panel.empty:
        return panel
    keys = ["city_key", "grid_id", period]
    signature = source.groupby(keys, as_index=False).agg(
        source_signature=("source_id", lambda x: "|".join(sorted(set(map(str, x))))),
        dominant_source=("source_id", lambda x: str(x.iloc[0])),
    )
    # Dominance is determined by observation count, not alphabetical order.
    dominant = source.sort_values(
        keys + ["n_observations", "source_id"], ascending=[True] * len(keys) + [False, True]
    )
    dominant = dominant.drop_duplicates(keys)[keys + ["source_id"]].rename(
        columns={"source_id": "dominant_source"}
    )
    signature = signature.drop(columns="dominant_source").merge(
        dominant, on=keys, how="left", validate="one_to_one"
    )
    result = panel.merge(signature, on=keys, how="left", validate="one_to_one")
    if period == "observed_month":
        result = result.sort_values(["city_key", "grid_id", period])
        previous = result.groupby(["city_key", "grid_id"])["source_signature"].shift(1)
        result["source_mix_changed"] = previous.notna() & result["source_signature"].ne(previous)
    return result


def build_city(
    city: str,
    registry: pd.DataFrame,
    crosswalk: pd.DataFrame,
    duplicate_ids: dict[str, str],
) -> dict:
    frames = [
        load_lianjia(city, crosswalk),
        load_standardized(city),
        load_wayback(city, crosswalk, registry, duplicate_ids),
        load_anjuke(city, registry),
        load_grid2023(city),
    ]
    if city == "nanjing":
        frames.append(load_nanjing_quarter())
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return {"city_key": city, "observation_rows": 0}
    observations = pd.concat(frames, ignore_index=True, sort=False)
    observations, coordinate_report = map_points_to_reference_grid(observations, city)
    observations = finalize_observations(observations)
    if observations["observation_id"].duplicated().any():
        duplicates = (
            observations.loc[observations["observation_id"].duplicated(), "observation_id"]
            .head()
            .tolist()
        )
        raise RuntimeError(f"Duplicate observation IDs in {city}: {duplicates}")

    monthly_source, monthly = source_balanced_panel(
        observations, ["observed_month"], "analysis_eligible_month"
    )
    monthly = add_source_signature(monthly_source, monthly, "observed_month")
    quarter_source, quarter = source_balanced_panel(
        observations, ["observed_quarter"], "analysis_eligible_quarter"
    )
    quarter = add_source_signature(quarter_source, quarter, "observed_quarter")
    year_source, year = source_balanced_panel(
        observations, ["observed_year"], "analysis_eligible_year"
    )
    year = add_source_signature(year_source, year, "observed_year")

    atomic_parquet(observations, OBS_DIR / f"{city}.parquet")
    for directory, source_frame, panel_frame in [
        (MONTH_DIR, monthly_source, monthly),
        (QUARTER_DIR, quarter_source, quarter),
        (YEAR_DIR, year_source, year),
    ]:
        atomic_parquet(source_frame, directory / f"{city}_source.parquet")
        atomic_parquet(panel_frame, directory / f"{city}.parquet")

    source_counts = observations.groupby("source_id").size().astype(int).to_dict()
    eligible_counts = (
        observations.groupby("source_id")[
            ["analysis_eligible_month", "analysis_eligible_quarter", "analysis_eligible_year"]
        ]
        .sum()
        .astype(int)
        .to_dict(orient="index")
    )
    report = {
        "city_key": city,
        "observation_rows": int(len(observations)),
        "unique_observation_ids": int(observations["observation_id"].nunique()),
        "source_rows": {str(k): int(v) for k, v in source_counts.items()},
        "source_eligible_rows": eligible_counts,
        "canonical_rows": int(observations["canonical_for_aggregation"].sum()),
        "monthly_eligible_rows": int(observations["analysis_eligible_month"].sum()),
        "quarterly_eligible_rows": int(observations["analysis_eligible_quarter"].sum()),
        "annual_eligible_rows": int(observations["analysis_eligible_year"].sum()),
        "monthly_grid_rows": int(len(monthly)),
        "quarterly_grid_rows": int(len(quarter)),
        "annual_grid_rows": int(len(year)),
        "monthly_grids": int(monthly["grid_id"].nunique()) if not monthly.empty else 0,
        "monthly_first": str(monthly["observed_month"].min().date()) if not monthly.empty else None,
        "monthly_last": str(monthly["observed_month"].max().date()) if not monthly.empty else None,
        "coordinate_mapping": coordinate_report,
        "duplicate_class_counts": {
            str(k): int(v) for k, v in observations["duplicate_class"].value_counts().items() if k
        },
    }
    print(
        f"{city}: observations={len(observations):,}, monthly_eligible={report['monthly_eligible_rows']:,}, "
        f"grid_months={len(monthly):,}",
        flush=True,
    )
    return report


def write_global_report(city_reports: list[dict], input_inventory: list[dict]) -> dict:
    active = [row for row in city_reports if row.get("observation_rows", 0)]
    summary = {
        "schema": "housing_panel_summary",
        "created_at": datetime.now(UTC).isoformat(),
        "raw_inputs_modified": False,
        "hedonic_adjustment": False,
        "listing_and_transaction_unified": True,
        "source_balanced_primary_price": True,
        "cities_requested": len(city_reports),
        "cities_with_observations": len(active),
        "observation_rows": int(sum(row.get("observation_rows", 0) for row in active)),
        "monthly_eligible_rows": int(sum(row.get("monthly_eligible_rows", 0) for row in active)),
        "monthly_grid_rows": int(sum(row.get("monthly_grid_rows", 0) for row in active)),
        "quarterly_grid_rows": int(sum(row.get("quarterly_grid_rows", 0) for row in active)),
        "annual_grid_rows": int(sum(row.get("annual_grid_rows", 0) for row in active)),
        "city_reports": city_reports,
        "input_inventory": input_inventory,
        "excluded_replication_package": {
            "source_id": "mendeley_d6m65pyyxd_v2",
            "rows": 2_140_453,
            "reason": "no city mapping, coordinates, community names, or reliable calendar crosswalk",
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "panel_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    city_table = pd.DataFrame(
        [
            {
                "city_key": row.get("city_key"),
                "observation_rows": row.get("observation_rows", 0),
                "canonical_rows": row.get("canonical_rows", 0),
                "monthly_eligible_rows": row.get("monthly_eligible_rows", 0),
                "monthly_grid_rows": row.get("monthly_grid_rows", 0),
                "monthly_grids": row.get("monthly_grids", 0),
                "monthly_first": row.get("monthly_first"),
                "monthly_last": row.get("monthly_last"),
            }
            for row in city_reports
        ]
    )
    city_table.to_csv(REPORT_DIR / "city_month_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(input_inventory).to_csv(
        REPORT_DIR / "source_inventory.csv", index=False, encoding="utf-8-sig"
    )
    return summary


def inventory_inputs() -> list[dict]:
    roots = [LIANJIA_DIR, STANDARDIZED_DIR, WAYBACK_DIR, ANJUKE_DIR, GRID2023_DIR]
    rows = []
    for root in roots:
        files = list(root.rglob("*")) if root.exists() else []
        files = [path for path in files if path.is_file()]
        rows.append(
            {
                "input_root": str(root.relative_to(ROOT)),
                "file_count": len(files),
                "bytes": int(sum(path.stat().st_size for path in files)),
                "latest_modified": max((path.stat().st_mtime for path in files), default=np.nan),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="*", default=list(ACTIVE_CITIES))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cities = [city for city in args.cities if city in ACTIVE_CITIES]
    if not cities:
        raise SystemExit("No valid research cities selected")
    registry = pd.read_parquet(REGISTRY_PATH)
    crosswalk = pd.read_parquet(CROSSWALK_PATH)
    duplicate_ids: dict[str, str] = {}
    if WAYBACK_DUPLICATES.exists():
        duplicates = pd.read_csv(WAYBACK_DUPLICATES)
        duplicate_ids = (
            duplicates.set_index("wayback_record_id")["duplicate_class"].astype(str).to_dict()
        )
    city_reports = [build_city(city, registry, crosswalk, duplicate_ids) for city in cities]
    summary = write_global_report(city_reports, inventory_inputs())
    print(
        json.dumps(
            {
                key: summary[key]
                for key in [
                    "cities_with_observations",
                    "observation_rows",
                    "monthly_eligible_rows",
                    "monthly_grid_rows",
                    "quarterly_grid_rows",
                    "annual_grid_rows",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
