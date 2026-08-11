from __future__ import annotations

import pandas as pd

from scripts.collection.import_geodoi_yangtze_housing import (
    normalize_city,
    normalize_county,
)


def test_city_mapping_uses_province_to_disambiguate_suzhou_and_taizhou() -> None:
    source = pd.DataFrame(index=range(7), columns=range(16))
    source.iloc[3, 5:16] = list(range(2008, 2019))
    source.iloc[4, :5] = [1, "江苏", "Jiangsu", "苏州", "Suzhou"]
    source.iloc[5, :5] = [2, "安徽", "Anhui", "苏州", "Suzhou"]
    source.iloc[6, :5] = [3, "浙江", "Zhejiang", "台州", "Taizhou"]
    source.iloc[4:, 5:16] = 10_000

    result = normalize_city(source, "abc")

    assert result.loc[result["source_city_number"].eq(1), "city_key"].unique().tolist() == [
        "suzhou"
    ]
    assert result.loc[result["source_city_number"].eq(2), "city_key"].isna().all()
    assert result.loc[result["source_city_number"].eq(3), "city_key"].unique().tolist() == [
        "taizhou"
    ]


def test_county_panel_is_long_and_maps_hefei() -> None:
    source = pd.DataFrame(index=range(5), columns=range(20))
    source.iloc[3, 6:17] = list(range(2008, 2019))
    source.iloc[4, [0, 1, 2, 3, 4, 5, 17, 18, 19]] = [
        "安徽",
        "Anhui",
        "合肥",
        "Hefei",
        "蜀山区",
        "Shushan District",
        4,
        None,
        None,
    ]
    source.iloc[4, 6:17] = list(range(8_000, 8_011))

    result = normalize_county(source, "abc")

    assert len(result) == 11
    assert result["city_key"].eq("hefei").all()
    assert result["year"].tolist() == list(range(2008, 2019))
    assert result["unit_price_cny_m2"].tolist() == list(range(8_000, 8_011))
