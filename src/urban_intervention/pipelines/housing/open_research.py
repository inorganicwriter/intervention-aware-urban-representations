"""Normalize openly licensed housing datasets used as auxiliary evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from urban_intervention.config.project import CITIES

FIGSHARE_DOI = "10.6084/m9.figshare.26968507.v1"
FIGSHARE_URL = "https://figshare.com/articles/dataset/china_house_price/26968507"
FIGSHARE_LICENSE = "CC BY 4.0"


def normalize_figshare_city_prices(raw: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Normalize one headerless annual workbook without inventing monthly detail."""
    if raw.shape[1] < 8:
        raise ValueError(f"Expected at least 8 columns in {source_file}; found {raw.shape[1]}")
    frame = raw.iloc[:, :8].copy()
    frame.columns = [
        "source_rank",
        "province_cn",
        "city_cn",
        "year",
        "price_label",
        "unit_price_cny_m2",
        "change_label",
        "change_direction",
    ]
    frame["province_cn"] = frame["province_cn"].astype("string").str.strip()
    frame["city_cn"] = frame["city_cn"].astype("string").str.strip()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["unit_price_cny_m2"] = pd.to_numeric(frame["unit_price_cny_m2"], errors="coerce")
    city_name_to_key = {str(value["name"]): key for key, value in CITIES.items()}
    frame["city_key"] = frame["city_cn"].map(city_name_to_key).astype("string")
    frame["is_research_city"] = frame["city_key"].notna()
    frame["raw_row_number"] = range(1, len(frame) + 1)
    frame["source_file"] = source_file
    frame["dataset_doi"] = FIGSHARE_DOI
    frame["source_url"] = FIGSHARE_URL
    frame["license"] = FIGSHARE_LICENSE
    frame["quality_flags"] = ""
    frame.loc[frame["year"].isna(), "quality_flags"] = "missing_year"
    missing_price = frame["unit_price_cny_m2"].isna()
    frame.loc[missing_price, "quality_flags"] = frame.loc[missing_price, "quality_flags"].map(
        lambda value: ";".join(filter(None, [value, "missing_unit_price"]))
    )
    nonpositive_price = frame["unit_price_cny_m2"].notna() & frame["unit_price_cny_m2"].le(0)
    frame.loc[nonpositive_price, "quality_flags"] = frame.loc[
        nonpositive_price, "quality_flags"
    ].map(lambda value: ";".join(filter(None, [value, "nonpositive_unit_price"])))
    frame["source_record_id"] = [
        hashlib.sha256(f"{FIGSHARE_DOI}|{source_file}|{row}".encode()).hexdigest()[:24]
        for row in frame["raw_row_number"]
    ]
    columns = [
        "source_record_id",
        "source_rank",
        "province_cn",
        "city_cn",
        "city_key",
        "is_research_city",
        "year",
        "unit_price_cny_m2",
        "price_label",
        "change_label",
        "change_direction",
        "dataset_doi",
        "source_url",
        "license",
        "source_file",
        "raw_row_number",
        "quality_flags",
    ]
    return frame[columns]


def load_figshare_workbooks(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No Figshare workbooks found in {input_dir}")
    frames = [
        normalize_figshare_city_prices(pd.read_excel(path, header=None), path.name)
        for path in files
    ]
    result = pd.concat(frames, ignore_index=True)
    duplicate = result.duplicated(["city_cn", "year"], keep=False)
    result.loc[duplicate, "quality_flags"] = result.loc[duplicate, "quality_flags"].map(
        lambda value: ";".join(filter(None, [value, "duplicate_city_year"]))
    )
    return result
