from __future__ import annotations

import pandas as pd

from scripts.collection.import_open_city_housing_growth import normalize


def test_normalize_distinguishes_same_named_cities_and_preserves_rows() -> None:
    source = pd.DataFrame(
        {
            "year": [2006, 2006, 2006, 2006, 2006],
            "city": ["Fuzhou", "Fuzhou", "Suzhou", "Taizhou", "Chengdu"],
            "province": ["Fujian", "Jiangxi", "Jiangsu", "Jiangsu", "Szechwan"],
            "city.1": [92, 109, 58, 65, 166],
            "arbp": [0.1, 0.2, 0.3, 0.4, 0.5],
            "crbp": [0.0, 0.1, 0.2, 0.3, 0.4],
            "arbp2": [0.11, 0.21, 0.31, 0.41, 0.51],
            "crbp2": [0.01, 0.11, 0.21, 0.31, 0.41],
        }
    )

    result = normalize(source, "abc")

    assert len(result) == 5
    assert result["city_key"].tolist()[:3] == ["fuzhou", pd.NA, "suzhou"]
    assert pd.isna(result.loc[3, "city_key"])
    assert result.loc[4, "city_key"] == "chengdu"
    assert result["quality_flags"].eq("").all()
