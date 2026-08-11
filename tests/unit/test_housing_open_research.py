from __future__ import annotations

import pandas as pd

from urban_intervention.pipelines.housing.open_research import (
    normalize_figshare_city_prices,
)


def test_normalize_figshare_city_prices_maps_research_city_without_month_invention() -> None:
    raw = pd.DataFrame(
        [
            [1, "上海", "上海", 2023, "60000元/㎡", 60000, "0.1%", "↑"],
            [2, "海南", "三亚", 2023, "30000元/㎡", 30000, "0.2%", "↓"],
        ]
    )
    result = normalize_figshare_city_prices(raw, "2023.xlsx")
    assert result.loc[0, "city_key"] == "shanghai"
    assert bool(result.loc[0, "is_research_city"])
    assert pd.isna(result.loc[1, "city_key"])
    assert not bool(result.loc[1, "is_research_city"])
    assert result.loc[0, "year"] == 2023
    assert "month" not in result.columns
