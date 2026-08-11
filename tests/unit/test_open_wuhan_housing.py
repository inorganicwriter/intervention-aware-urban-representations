from __future__ import annotations

import math

import pandas as pd

from scripts.collection.import_cran_wuhan_housing import normalize


def test_normalize_reverses_logs_and_flags_duplicate_source_rows() -> None:
    source = pd.DataFrame(
        {
            "Price": [math.log(20_000), math.log(20_000), math.log(30_000)],
            "BuildingArea": [math.log(90), math.log(90), math.log(120)],
            "lon": [114.3, 114.3, 120.0],
            "lat": [30.6, 30.6, 30.6],
            "group": [1, 1, 2],
            "geometry": [None, None, None],
        }
    )

    result = normalize(source, "abc")

    assert len(result) == 3
    assert result["unit_price_cny_m2"].round().tolist() == [20_000, 20_000, 30_000]
    assert result["building_area_m2"].round().tolist() == [90, 90, 120]
    assert result["community_id"].tolist() == [
        "cran_hgwrr_group_1",
        "cran_hgwrr_group_1",
        "cran_hgwrr_group_2",
    ]
    assert result.loc[:1, "quality_flags"].str.contains("duplicate_source_attributes").all()
    assert "invalid_wuhan_coordinate" in result.loc[2, "quality_flags"]
