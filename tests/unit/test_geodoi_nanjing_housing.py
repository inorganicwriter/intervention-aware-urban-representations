from __future__ import annotations

import pandas as pd

from scripts.collection.import_geodoi_nanjing_housing import (
    SECTION_STARTS,
    normalize,
)


def _source() -> pd.DataFrame:
    source = pd.DataFrame(index=range(7), columns=range(148))
    periods = [year * 100 + quarter for year in range(2009, 2017) for quarter in range(1, 5)]
    periods.append(201701)
    for start in SECTION_STARTS.values():
        source.iloc[4, start : start + len(periods)] = periods
    source.iloc[5, :16] = [
        1,
        "测试小区",
        "Test community",
        "物业小区",
        "Property community",
        "鼓楼区",
        "Gulou",
        118.78,
        32.06,
        2000,
        2.0,
        35,
        100_000,
        1_000,
        "住宅",
        "Residence",
    ]
    source.iloc[6, :16] = [
        2,
        "坐标异常小区",
        "Invalid coordinate",
        "物业小区",
        "Property community",
        "鼓楼区",
        "Gulou",
        120.0,
        32.06,
        2005,
        2.5,
        30,
        80_000,
        800,
        "住宅",
        "Residence",
    ]
    for start in SECTION_STARTS.values():
        source.iloc[5:7, start : start + len(periods)] = 1
    source.iloc[5, SECTION_STARTS["sale_price_cny_m2"]] = 800
    return source


def test_normalize_builds_complete_community_quarter_panel_and_flags_quality() -> None:
    result = normalize(_source(), "abc")

    assert len(result) == 66
    assert result["period"].nunique() == 33
    assert result["period"].min() == "2009Q1"
    assert result["period"].max() == "2017Q1"
    first = result.iloc[0]
    assert first["sale_price_cny_m2"] == 800
    assert "suspicious_low_sale_price" in first["quality_flags"]
    assert (
        result.loc[result["source_community_id"].eq(2), "quality_flags"]
        .str.contains("invalid_nanjing_coordinate")
        .all()
    )
