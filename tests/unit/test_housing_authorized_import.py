from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from urban_intervention.pipelines.housing.importer import (
    import_authorized_export,
    import_large_xlsx_authorized_export,
)


def test_authorized_import_preserves_all_rows_and_derives_unit_price(tmp_path: Path) -> None:
    source = tmp_path / "authorized.csv"
    pd.DataFrame(
        {
            "city": ["北京", "北京"],
            "community": ["甲小区", "乙小区"],
            "date": ["2020-01-02", "bad-date"],
            "total": [500.0, 300.0],
            "area": [100.0, 0.0],
            "lon": [116.4, 116.5],
            "lat": [39.9, 39.8],
        }
    ).to_csv(source, index=False, encoding="utf-8-sig")
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(
        yaml.safe_dump(
            {
                "metadata": {
                    "batch_id": "test_batch",
                    "source_platform": "authorized_test",
                    "acquisition_method": "licensed_export",
                    "price_type": "transaction",
                    "observation_type": "individual",
                    "temporal_unit": "deal_date",
                    "spatial_unit": "point",
                    "unit": "cny_per_m2",
                    "source_snapshot_date": "2026-07-22",
                },
                "columns": {
                    "city_key": "city",
                    "community_name": "community",
                    "deal_date": "date",
                    "total_price_10k_cny": "total",
                    "building_area_m2": "area",
                    "lon": "lon",
                    "lat": "lat",
                },
                "city_key_map": {"北京": "beijing"},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    output, manifest = import_authorized_export(
        source,
        mapping,
        tmp_path / "raw",
        tmp_path / "staging",
    )

    result = pd.read_parquet(output)
    assert len(result) == 2
    assert result.loc[0, "unit_price_cny_m2"] == 50_000
    assert result["source_record_id"].notna().all()
    assert result.loc[1, "quality_flags"] == (
        "nonpositive_area;missing_deal_date;missing_transaction_unit_price"
    )
    assert manifest.exists()
    assert (tmp_path / "raw" / "test_batch" / "authorized.csv").exists()


def test_authorized_import_applies_source_quality_rules(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "city": ["上海", "上海"],
            "district": ["浦东", "上海周边"],
            "community": ["甲小区", "乙小区"],
            "date": ["2020-01-02", "2020-01-03"],
            "unit": [50_000, 40_000],
            "total": [500.0, 400.0],
            "area": [100.0, 100.0],
            "lon": [121.5, 121.0],
            "lat": [31.2, 31.0],
        }
    )
    mapping = {
        "metadata": {
            "batch_id": "quality_test",
            "source_platform": "research_test",
            "acquisition_method": "open_research_repository",
            "price_type": "transaction",
            "observation_type": "individual",
            "temporal_unit": "deal_date",
            "spatial_unit": "point",
            "unit": "cny_per_m2",
            "source_snapshot_date": "2026-07-22",
        },
        "columns": {
            "city_key": "city",
            "district": "district",
            "community_name": "community",
            "deal_date": "date",
            "unit_price_cny_m2": "unit",
            "total_price_10k_cny": "total",
            "building_area_m2": "area",
            "lon": "lon",
            "lat": "lat",
        },
        "city_key_map": {"上海": "shanghai"},
        "quality_flag_rules": [
            {
                "source_column": "district",
                "operator": "equals",
                "value": "上海周边",
                "flag": "cross_city_record",
            }
        ],
    }
    from urban_intervention.pipelines.housing.importer import normalize_authorized_export

    result = normalize_authorized_export(source, mapping, "abc")
    assert result.loc[0, "quality_flags"] == ""
    assert result.loc[1, "quality_flags"] == "cross_city_record"


def test_authorized_import_constructs_month_precision_date_parts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "monthly.csv"
    pd.DataFrame(
        {
            "record_id": [1, 2],
            "year": [2020, 2021],
            "month": [9, 13],
            "unit_price": [50_000, 40_000],
        }
    ).to_csv(source, index=False)
    mapping = tmp_path / "monthly_mapping.yaml"
    mapping.write_text(
        yaml.safe_dump(
            {
                "metadata": {
                    "batch_id": "monthly_test",
                    "source_platform": "research_test",
                    "acquisition_method": "open_research_repository",
                    "city_key": "beijing",
                    "price_type": "transaction",
                    "observation_type": "individual",
                    "temporal_unit": "month",
                    "spatial_unit": "point",
                    "unit": "cny_per_m2",
                    "source_snapshot_date": "2026-07-22",
                },
                "columns": {
                    "source_record_id": "record_id",
                    "unit_price_cny_m2": "unit_price",
                },
                "date_parts": {
                    "deal_date": {
                        "year_column": "year",
                        "month_column": "month",
                        "day": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    output, _ = import_authorized_export(source, mapping, tmp_path / "raw", tmp_path / "staging")
    result = pd.read_parquet(output)
    assert result.loc[0, "deal_date"] == pd.Timestamp("2020-09-01")
    assert pd.isna(result.loc[1, "deal_date"])
    assert result.loc[1, "quality_flags"] == "missing_deal_date"


def test_authorized_import_reads_stata_and_formats_yyyymm(tmp_path: Path) -> None:
    source = tmp_path / "monthly.dta"
    pd.DataFrame(
        {
            "community_id": [1.0, 1.0],
            "time": [202001.0, 202013.0],
            "price": [20_000.0, 21_000.0],
        }
    ).to_stata(source, write_index=False)
    mapping = tmp_path / "stata_mapping.yaml"
    mapping.write_text(
        yaml.safe_dump(
            {
                "metadata": {
                    "batch_id": "stata_monthly_test",
                    "source_platform": "research_test",
                    "acquisition_method": "open_research_repository",
                    "city_key": "chengdu",
                    "price_type": "transaction",
                    "observation_type": "individual",
                    "temporal_unit": "month",
                    "spatial_unit": "point",
                    "unit": "cny_per_m2",
                    "source_snapshot_date": "2026-07-22",
                },
                "columns": {
                    "community_id": "community_id",
                    "unit_price_cny_m2": "price",
                },
                "date_formats": {
                    "deal_date": {
                        "source_column": "time",
                        "format": "%Y%m",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    output, _ = import_authorized_export(source, mapping, tmp_path / "raw", tmp_path / "staging")
    result = pd.read_parquet(output)
    assert result.loc[0, "deal_date"] == pd.Timestamp("2020-01-01")
    assert pd.isna(result.loc[1, "deal_date"])
    assert result.loc[1, "quality_flags"] == "missing_deal_date"


def test_large_xlsx_import_flags_duplicates_across_chunks(tmp_path: Path) -> None:
    source = tmp_path / "large.xlsx"
    pd.DataFrame(
        {
            "city": ["上海", "上海", "上海"],
            "community": ["甲小区", "乙小区", "甲小区"],
            "date": ["2020-01-02", "2020-01-03", "2020-01-02"],
            "unit": [50_000, 40_000, 50_000],
            "total": [500.0, 400.0, 500.0],
            "area": [100.0, 100.0, 100.0],
            "lon": [121.5, 121.6, 121.5],
            "lat": [31.2, 31.3, 31.2],
        }
    ).to_excel(source, index=False, sheet_name="transactions")
    mapping = tmp_path / "large_mapping.yaml"
    mapping.write_text(
        yaml.safe_dump(
            {
                "sheet_name": "transactions",
                "metadata": {
                    "batch_id": "large_test",
                    "source_platform": "research_test",
                    "acquisition_method": "open_research_repository",
                    "price_type": "transaction",
                    "observation_type": "individual",
                    "temporal_unit": "deal_date",
                    "spatial_unit": "point",
                    "unit": "cny_per_m2",
                    "source_snapshot_date": "2026-07-22",
                },
                "columns": {
                    "city_key": "city",
                    "community_name": "community",
                    "deal_date": "date",
                    "unit_price_cny_m2": "unit",
                    "total_price_10k_cny": "total",
                    "building_area_m2": "area",
                    "lon": "lon",
                    "lat": "lat",
                },
                "city_key_map": {"上海": "shanghai"},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    output, _ = import_large_xlsx_authorized_export(
        source, mapping, tmp_path / "raw", tmp_path / "staging", chunk_rows=1
    )
    result = pd.read_parquet(output)
    assert result["source_record_id"].nunique() == 3
    assert result["raw_row_number"].tolist() == [2, 3, 4]
    assert result["quality_flags"].tolist() == [
        "duplicate_transaction_key",
        "",
        "duplicate_transaction_key",
    ]
