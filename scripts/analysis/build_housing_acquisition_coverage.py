"""Summarize the complete source stack for every research city.

Acquisition coverage is deliberately independent of the eventual causal time
window.  Every observed date is retained; downstream DID/SC builders decide
which dates are admissible for a particular station cohort.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.config.project import CITIES  # noqa: E402
from urban_intervention.data.paths import (  # noqa: E402
    OUTPUT_DATA_QUALITY_DIR,
    OUTPUT_HOUSING_ACQUISITION_DIR,
    OUTPUT_HOUSING_DID_PREFLIGHT_DIR,
    OUTPUT_HOUSING_FUSION_DIR,
    STAGING_DIR,
    STAGING_HOUSING_DIR,
    STAGING_HOUSING_STANDARDIZED_DIR,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing-coverage",
        type=Path,
        default=OUTPUT_HOUSING_DID_PREFLIGHT_DIR / "lianjia_city_coverage_summary.csv",
    )
    parser.add_argument(
        "--shanghai-transactions",
        type=Path,
        default=STAGING_HOUSING_STANDARDIZED_DIR
        / "mendeley_hwfghkygy6_v1_shanghai_transactions"
        / "housing_observations.parquet",
    )
    parser.add_argument(
        "--figshare-coverage",
        type=Path,
        default=OUTPUT_HOUSING_ACQUISITION_DIR / "figshare_26968507_research_city_coverage.csv",
    )
    parser.add_argument(
        "--source-inventory",
        type=Path,
        default=OUTPUT_HOUSING_FUSION_DIR / "source_inventory.csv",
    )
    parser.add_argument(
        "--wayback-coverage",
        type=Path,
        default=OUTPUT_DATA_QUALITY_DIR / "wayback_housing_city_distribution.csv",
    )
    parser.add_argument(
        "--nbs-monthly-hpi",
        type=Path,
        default=STAGING_DIR / "nbs_hpi" / "monthly.csv",
    )
    parser.add_argument(
        "--nanning-cross-section",
        type=Path,
        default=STAGING_HOUSING_STANDARDIZED_DIR
        / "mendeley_pj2zff4p9m_v4_nanning_2018_community_prices"
        / "housing_observations.parquet",
    )
    parser.add_argument(
        "--wuhan-cross-section",
        type=Path,
        default=STAGING_HOUSING_STANDARDIZED_DIR
        / "cran_hgwrr_0.6-2_wuhan_2018_second_hand_housing"
        / "housing_observations.parquet",
    )
    parser.add_argument(
        "--beijing-supplemental",
        type=Path,
        default=STAGING_HOUSING_STANDARDIZED_DIR
        / "figshare_14398907_v1_beijing_2020_transactions"
        / "housing_observations.parquet",
    )
    parser.add_argument(
        "--chengdu-supplemental",
        type=Path,
        default=STAGING_HOUSING_STANDARDIZED_DIR
        / "mendeley_wpv5zn9rxp_v1_chengdu_fang_transactions"
        / "housing_observations.parquet",
    )
    parser.add_argument(
        "--early-city-growth",
        type=Path,
        default=STAGING_HOUSING_DIR
        / "auxiliary"
        / "mendeley_52kj9yzx5j_v2"
        / "city_housing_price_growth.parquet",
    )
    parser.add_argument(
        "--nanjing-quarterly-community",
        type=Path,
        default=STAGING_HOUSING_STANDARDIZED_DIR
        / "geodoi_geodb_2018_04_08_v1"
        / "community_quarter_housing.parquet",
    )
    parser.add_argument(
        "--yangtze-annual-city",
        type=Path,
        default=STAGING_HOUSING_DIR
        / "auxiliary"
        / "geodoi_geodb_2019_04_17_v1"
        / "city_housing_price.parquet",
    )
    parser.add_argument(
        "--yangtze-annual-county",
        type=Path,
        default=STAGING_HOUSING_DIR
        / "auxiliary"
        / "geodoi_geodb_2019_04_17_v1"
        / "district_county_housing_price.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_HOUSING_ACQUISITION_DIR,
    )
    args = parser.parse_args()

    existing = pd.read_csv(args.existing_coverage)
    existing_keys = set(existing["city_key"])
    shanghai = pd.read_parquet(
        args.shanghai_transactions,
        columns=["city_key", "deal_date", "quality_flags"],
    )
    shanghai["deal_date"] = pd.to_datetime(shanghai["deal_date"], errors="coerce")
    shanghai_usable = shanghai[
        shanghai["deal_date"].notna()
        & ~shanghai["quality_flags"].str.contains("cross_city_record", na=False)
    ]
    figshare = pd.read_csv(args.figshare_coverage).set_index("city_key")
    inventory = pd.read_csv(args.source_inventory)
    anjuke = inventory.loc[
        inventory["dataset"].eq("anjuke_cross_section"),
        [
            "city_key",
            "boundary_file_present",
            "house_file_present",
            "boundary_rows",
            "house_rows",
        ],
    ].copy()
    anjuke = anjuke.drop_duplicates("city_key").set_index("city_key")
    wayback = pd.read_csv(args.wayback_coverage).rename(columns={"city": "city_key"})
    wayback = wayback.drop_duplicates("city_key").set_index("city_key")
    nbs = pd.read_csv(args.nbs_monthly_hpi)
    nbs["period"] = pd.to_datetime(
        dict(year=nbs["year"], month=nbs["month"], day=1), errors="coerce"
    )
    nbs_city = nbs.groupby("city_key", as_index=True).agg(
        nbs_hpi_rows=("city_key", "size"),
        nbs_hpi_first_month=("period", "min"),
        nbs_hpi_last_month=("period", "max"),
        nbs_hpi_months=("period", "nunique"),
    )
    if args.nanning_cross_section.exists():
        nanning = pd.read_parquet(
            args.nanning_cross_section,
            columns=["city_key", "unit_price_cny_m2", "lon", "lat", "quality_flags"],
        )
        nanning_usable = nanning[
            nanning["unit_price_cny_m2"].gt(0) & nanning[["lon", "lat"]].notna().all(axis=1)
        ]
    else:
        nanning_usable = pd.DataFrame()
    if args.wuhan_cross_section.exists():
        wuhan = pd.read_parquet(
            args.wuhan_cross_section,
            columns=["city_key", "unit_price_cny_m2", "lon", "lat", "quality_flags"],
        )
        wuhan_usable = wuhan[
            wuhan["unit_price_cny_m2"].gt(0)
            & wuhan[["lon", "lat"]].notna().all(axis=1)
            & ~wuhan["quality_flags"].str.contains("invalid_wuhan_coordinate", na=False)
        ]
    else:
        wuhan_usable = pd.DataFrame()
    open_spatial_sources = {
        "nanning": ("mendeley_pj2zff4p9m_v4", 2018, nanning_usable),
        "wuhan": ("cran_hgwrr_0.6-2", 2018, wuhan_usable),
    }
    open_transaction_frames: list[pd.DataFrame] = []
    open_transaction_sources = {
        "beijing": (
            args.beijing_supplemental,
            "figshare_14398907_v1",
            "wgs84_declared_columns",
        ),
        "chengdu": (
            args.chengdu_supplemental,
            "mendeley_wpv5zn9rxp_v1",
            "unresolved_mixed_or_misaligned",
        ),
    }
    for city_key, (path, source_id, coordinate_status) in open_transaction_sources.items():
        if not path.exists():
            continue
        columns = [
            "city_key",
            "deal_date",
            "unit_price_cny_m2",
            "lon",
            "lat",
            "quality_flags",
        ]
        frame = pd.read_parquet(path, columns=columns)
        frame["deal_date"] = pd.to_datetime(frame["deal_date"], errors="coerce")
        frame = frame[
            frame["city_key"].eq(city_key)
            & frame["unit_price_cny_m2"].gt(0)
            & frame["deal_date"].notna()
            & frame[["lon", "lat"]].notna().all(axis=1)
            & ~frame["quality_flags"].str.contains("duplicate_transaction_key", na=False)
        ].copy()
        frame["open_source_id"] = source_id
        frame["coordinate_crs_status"] = coordinate_status
        open_transaction_frames.append(frame)
    open_transactions = (
        pd.concat(open_transaction_frames, ignore_index=True)
        if open_transaction_frames
        else pd.DataFrame()
    )
    if args.early_city_growth.exists():
        early_growth = pd.read_parquet(
            args.early_city_growth,
            columns=["city_key", "year", "quality_flags"],
        )
        early_growth = early_growth[
            early_growth["city_key"].notna() & early_growth["quality_flags"].eq("")
        ]
    else:
        early_growth = pd.DataFrame()
    if args.nanjing_quarterly_community.exists():
        nanjing_quarterly = pd.read_parquet(
            args.nanjing_quarterly_community,
            columns=[
                "city_key",
                "source_community_id",
                "period",
                "sale_price_cny_m2",
                "quality_flags",
            ],
        )
    else:
        nanjing_quarterly = pd.DataFrame()
    if args.yangtze_annual_city.exists():
        yangtze_city = pd.read_parquet(
            args.yangtze_annual_city,
            columns=["city_key", "year", "unit_price_cny_m2", "quality_flags"],
        )
        yangtze_city = yangtze_city[
            yangtze_city["city_key"].notna()
            & yangtze_city["unit_price_cny_m2"].gt(0)
            & yangtze_city["quality_flags"].eq("")
        ]
    else:
        yangtze_city = pd.DataFrame()
    if args.yangtze_annual_county.exists():
        yangtze_county = pd.read_parquet(
            args.yangtze_annual_county,
            columns=[
                "city_key",
                "district_county_cn",
                "year",
                "unit_price_cny_m2",
                "quality_flags",
            ],
        )
        yangtze_county = yangtze_county[
            yangtze_county["city_key"].notna()
            & yangtze_county["unit_price_cny_m2"].gt(0)
            & yangtze_county["quality_flags"].eq("")
        ]
    else:
        yangtze_county = pd.DataFrame()

    rows: list[dict[str, object]] = []
    for city_key in sorted(CITIES):
        prior = existing.loc[existing["city_key"].eq(city_key)]
        is_shanghai = city_key == "shanghai"
        primary = city_key in existing_keys or is_shanghai
        if not prior.empty:
            first_year = int(prior.iloc[0]["first_observed_year"])
            last_year = int(prior.iloc[0]["last_observed_year"])
            primary_rows = int(prior.iloc[0]["valid_transactions"])
            source = "lianjia_purchased"
        elif is_shanghai:
            first_year = int(shanghai_usable["deal_date"].dt.year.min())
            last_year = int(shanghai_usable["deal_date"].dt.year.max())
            primary_rows = int(len(shanghai_usable))
            source = "mendeley_hwfghkygy6_v1"
        else:
            first_year = last_year = primary_rows = pd.NA
            source = pd.NA
        anjuke_row = anjuke.loc[city_key] if city_key in anjuke.index else None
        anjuke_house = bool(anjuke_row is not None and anjuke_row["house_file_present"] is True)
        # CSV round-trips may store booleans as strings.
        if anjuke_row is not None:
            anjuke_house = str(anjuke_row["house_file_present"]).lower() == "true"
            anjuke_boundary = str(anjuke_row["boundary_file_present"]).lower() == "true"
            anjuke_house_rows = (
                int(anjuke_row["house_rows"]) if pd.notna(anjuke_row["house_rows"]) else 0
            )
            anjuke_boundary_rows = (
                int(anjuke_row["boundary_rows"]) if pd.notna(anjuke_row["boundary_rows"]) else 0
            )
        else:
            anjuke_boundary = False
            anjuke_house_rows = 0
            anjuke_boundary_rows = 0
        wayback_row = wayback.loc[city_key] if city_key in wayback.index else None
        wayback_rows = int(wayback_row["rows"]) if wayback_row is not None else 0
        nbs_row = nbs_city.loc[city_key] if city_key in nbs_city.index else None
        nbs_available = nbs_row is not None
        open_spatial_source, open_spatial_year, open_spatial = open_spatial_sources.get(
            city_key, (pd.NA, pd.NA, pd.DataFrame())
        )
        open_spatial_available = not open_spatial.empty
        city_open_transactions = (
            open_transactions[open_transactions["city_key"].eq(city_key)]
            if not open_transactions.empty
            else pd.DataFrame()
        )
        supplemental_open_transaction = not city_open_transactions.empty
        coordinate_crs_status = (
            city_open_transactions["coordinate_crs_status"].iloc[0]
            if supplemental_open_transaction
            else pd.NA
        )
        supplemental_grid_ready = bool(
            supplemental_open_transaction and coordinate_crs_status == "wgs84_declared_columns"
        )
        city_early_growth = (
            early_growth[early_growth["city_key"].eq(city_key)]
            if not early_growth.empty
            else pd.DataFrame()
        )
        early_city_growth_available = not city_early_growth.empty
        city_nanjing_quarterly = (
            nanjing_quarterly[nanjing_quarterly["city_key"].eq(city_key)]
            if not nanjing_quarterly.empty
            else pd.DataFrame()
        )
        quarterly_community_available = not city_nanjing_quarterly.empty
        city_yangtze = (
            yangtze_city[yangtze_city["city_key"].eq(city_key)]
            if not yangtze_city.empty
            else pd.DataFrame()
        )
        city_yangtze_county = (
            yangtze_county[yangtze_county["city_key"].eq(city_key)]
            if not yangtze_county.empty
            else pd.DataFrame()
        )
        yangtze_annual_available = not city_yangtze.empty
        layer_count = sum(
            [
                primary,
                anjuke_house,
                wayback_rows > 0,
                nbs_available,
                bool(figshare.loc[city_key, "covered"]),
                open_spatial_available,
                supplemental_open_transaction,
                early_city_growth_available,
                quarterly_community_available,
                yangtze_annual_available,
            ]
        )
        rows.append(
            {
                "city_key": city_key,
                "primary_transaction_available": primary,
                "primary_source": source,
                "primary_first_year": first_year,
                "primary_last_year": last_year,
                "primary_rows": primary_rows,
                "monthly_point_outcome_available": primary or supplemental_open_transaction,
                "monthly_grid_outcome_ready": primary or supplemental_grid_ready,
                "anjuke_cross_section_available": anjuke_house,
                "anjuke_listing_rows": anjuke_house_rows,
                "anjuke_boundary_available": anjuke_boundary,
                "anjuke_boundary_rows": anjuke_boundary_rows,
                "wayback_available": wayback_rows > 0,
                "wayback_rows": wayback_rows,
                "wayback_anjuke_rows": int(wayback_row["anjuke_rows"])
                if wayback_row is not None
                else 0,
                "wayback_beike_rows": int(wayback_row["beike_rows"])
                if wayback_row is not None
                else 0,
                "wayback_lianjia_rows": int(wayback_row["lianjia_rows"])
                if wayback_row is not None
                else 0,
                "nbs_monthly_hpi_available": nbs_available,
                "nbs_hpi_first_month": nbs_row["nbs_hpi_first_month"].strftime("%Y-%m")
                if nbs_available
                else pd.NA,
                "nbs_hpi_last_month": nbs_row["nbs_hpi_last_month"].strftime("%Y-%m")
                if nbs_available
                else pd.NA,
                "nbs_hpi_months": int(nbs_row["nbs_hpi_months"]) if nbs_available else 0,
                "nbs_hpi_rows": int(nbs_row["nbs_hpi_rows"]) if nbs_available else 0,
                "open_spatial_cross_section_available": open_spatial_available,
                "open_spatial_cross_section_source": open_spatial_source
                if open_spatial_available
                else pd.NA,
                "open_spatial_cross_section_year": open_spatial_year
                if open_spatial_available
                else pd.NA,
                "open_spatial_cross_section_rows": int(len(open_spatial))
                if open_spatial_available
                else 0,
                "supplemental_open_transaction_available": supplemental_open_transaction,
                "supplemental_open_transaction_source": city_open_transactions[
                    "open_source_id"
                ].iloc[0]
                if supplemental_open_transaction
                else pd.NA,
                "supplemental_open_transaction_first_month": city_open_transactions["deal_date"]
                .min()
                .strftime("%Y-%m")
                if supplemental_open_transaction
                else pd.NA,
                "supplemental_open_transaction_last_month": city_open_transactions["deal_date"]
                .max()
                .strftime("%Y-%m")
                if supplemental_open_transaction
                else pd.NA,
                "supplemental_open_transaction_rows": int(len(city_open_transactions))
                if supplemental_open_transaction
                else 0,
                "supplemental_open_transaction_date_precision": "month"
                if supplemental_open_transaction
                else pd.NA,
                "supplemental_open_transaction_coordinate_crs_status": coordinate_crs_status,
                "early_city_growth_available": early_city_growth_available,
                "early_city_growth_source": "mendeley_52kj9yzx5j_v2"
                if early_city_growth_available
                else pd.NA,
                "early_city_growth_first_year": int(city_early_growth["year"].min())
                if early_city_growth_available
                else pd.NA,
                "early_city_growth_last_year": int(city_early_growth["year"].max())
                if early_city_growth_available
                else pd.NA,
                "early_city_growth_rows": int(len(city_early_growth))
                if early_city_growth_available
                else 0,
                "quarterly_community_outcome_available": quarterly_community_available,
                "quarterly_community_outcome_source": "geodoi_geodb_2018_04_08_v1"
                if quarterly_community_available
                else pd.NA,
                "quarterly_community_first_period": city_nanjing_quarterly["period"].min()
                if quarterly_community_available
                else pd.NA,
                "quarterly_community_last_period": city_nanjing_quarterly["period"].max()
                if quarterly_community_available
                else pd.NA,
                "quarterly_community_rows": int(len(city_nanjing_quarterly))
                if quarterly_community_available
                else 0,
                "quarterly_community_sale_price_rows": int(
                    city_nanjing_quarterly["sale_price_cny_m2"].notna().sum()
                )
                if quarterly_community_available
                else 0,
                "yangtze_annual_city_available": yangtze_annual_available,
                "yangtze_annual_first_year": int(city_yangtze["year"].min())
                if yangtze_annual_available
                else pd.NA,
                "yangtze_annual_last_year": int(city_yangtze["year"].max())
                if yangtze_annual_available
                else pd.NA,
                "yangtze_annual_city_rows": int(len(city_yangtze))
                if yangtze_annual_available
                else 0,
                "yangtze_annual_county_rows": int(len(city_yangtze_county))
                if not city_yangtze_county.empty
                else 0,
                "yangtze_annual_county_units": int(
                    city_yangtze_county["district_county_cn"].nunique()
                )
                if not city_yangtze_county.empty
                else 0,
                "annual_city_auxiliary_available": bool(figshare.loc[city_key, "covered"]),
                "annual_city_auxiliary_first_year": int(figshare.loc[city_key, "first_year"]),
                "annual_city_auxiliary_last_year": int(figshare.loc[city_key, "last_year"]),
                "available_source_layers": layer_count,
                "acquisition_status": (
                    "supplement_and_cross_validate"
                    if primary
                    else (
                        "open_monthly_point_outcome_coordinate_crs_audit_required"
                        if supplemental_open_transaction
                        else "monthly_spatial_outcome_required"
                    )
                ),
            }
        )
    coverage = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = args.output_dir / "research_city_housing_source_coverage.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    markdown_path = args.output_dir / "research_city_housing_source_coverage.md"
    markdown_lines = [
        "# Research-city housing source coverage",
        "",
        "Acquisition is open-ended in time: no research-window date filter is applied.",
        "Cross-sections and city indices are retained as auxiliary layers, not treated as",
        "substitutes for a monthly 500 m outcome.",
        "",
        "| City | Primary transactions | Rows / years | Open monthly points | GeoDOI quarterly | Yangtze annual | Anjuke rows | Wayback rows | NBS months | Early growth | Open spatial | Layers | Next acquisition action |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in coverage.itertuples(index=False):
        primary_span = (
            f"{int(row.primary_rows):,} / {int(row.primary_first_year)}-{int(row.primary_last_year)}"
            if row.primary_transaction_available
            else "-"
        )
        markdown_lines.append(
            "| "
            + " | ".join(
                [
                    row.city_key,
                    "yes" if row.primary_transaction_available else "no",
                    primary_span,
                    f"{int(row.supplemental_open_transaction_rows):,}",
                    f"{int(row.quarterly_community_sale_price_rows):,}",
                    f"{int(row.yangtze_annual_city_rows):,}",
                    f"{int(row.anjuke_listing_rows):,}",
                    f"{int(row.wayback_rows):,}",
                    f"{int(row.nbs_hpi_months):,}",
                    f"{int(row.early_city_growth_rows):,}",
                    f"{int(row.open_spatial_cross_section_rows):,}",
                    str(int(row.available_source_layers)),
                    row.acquisition_status,
                ]
            )
            + " |"
        )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    report = {
        "schema": "research_city_housing_source_coverage_v4",
        "research_cities": len(CITIES),
        "primary_transaction_cities": int(coverage["primary_transaction_available"].sum()),
        "primary_transaction_missing_cities": coverage.loc[
            ~coverage["primary_transaction_available"], "city_key"
        ].tolist(),
        "annual_city_auxiliary_cities": int(coverage["annual_city_auxiliary_available"].sum()),
        "anjuke_cross_section_cities": int(coverage["anjuke_cross_section_available"].sum()),
        "wayback_cities": int(coverage["wayback_available"].sum()),
        "anjuke_or_wayback_spatial_web_cities": int(
            (coverage["anjuke_cross_section_available"] | coverage["wayback_available"]).sum()
        ),
        "nbs_monthly_hpi_cities": int(coverage["nbs_monthly_hpi_available"].sum()),
        "open_spatial_cross_section_cities": int(
            coverage["open_spatial_cross_section_available"].sum()
        ),
        "supplemental_open_transaction_cities": int(
            coverage["supplemental_open_transaction_available"].sum()
        ),
        "monthly_point_outcome_cities": int(coverage["monthly_point_outcome_available"].sum()),
        "monthly_grid_outcome_ready_cities": int(coverage["monthly_grid_outcome_ready"].sum()),
        "monthly_grid_outcome_missing_cities": coverage.loc[
            ~coverage["monthly_grid_outcome_ready"], "city_key"
        ].tolist(),
        "early_city_growth_cities": int(coverage["early_city_growth_available"].sum()),
        "quarterly_community_outcome_cities": int(
            coverage["quarterly_community_outcome_available"].sum()
        ),
        "yangtze_annual_city_price_cities": int(coverage["yangtze_annual_city_available"].sum()),
        "all_cities_remain_in_acquisition_scope": True,
        "acquisition_time_filter_applied": False,
        "beijing": coverage.loc[coverage["city_key"].eq("beijing")].iloc[0].to_dict(),
        "warning": "Cross-sections, annual city series, and NBS city indices do not substitute for monthly 500 m grid outcomes. Chengdu open monthly points remain blocked from grid assignment pending coordinate-CRS resolution.",
        "coverage_file": str(coverage_path),
        "coverage_markdown": str(markdown_path),
    }
    report_path = args.output_dir / "research_city_housing_source_coverage.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"coverage={coverage_path}")
    print(f"markdown={markdown_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
