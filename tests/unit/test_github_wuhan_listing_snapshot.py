from __future__ import annotations

import pandas as pd

from scripts.collection.import_github_wuhan_listing_snapshot import normalize


def test_normalize_parses_mojibake_units_and_quarantines_duplicates() -> None:
    source = pd.DataFrame(
        {
            "单价": ["14706元/�O", "14706元/�O"],
            "地址": ["金融港-测试路", "金融港-测试路"],
            "小区": ["测试小区", "测试小区"],
            "年份": ["2018年建", "2018年建"],
            "总价": [125.0, 125.0],
            "户型": ["2室2厅", "2室2厅"],
            "朝向": ["南北向", "南北向"],
            "标题": ["测试挂牌", "测试挂牌"],
            "楼层": ["中层（共27层）", "中层（共27层）"],
            "面积": ["85�O", "85�O"],
        }
    )

    result = normalize(source, "abc", "donghu_high_tech")

    assert result["unit_price_cny_m2"].tolist() == [14_706, 14_706]
    assert result["building_area_m2"].tolist() == [85, 85]
    assert result["built_year"].tolist() == [2018, 2018]
    assert result["bedroom_count"].tolist() == [2, 2]
    assert result["quality_flags"].str.contains("duplicate_source_attributes").all()
    assert result["license_status"].eq("no_explicit_repository_license").all()
