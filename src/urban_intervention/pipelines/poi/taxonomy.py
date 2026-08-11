"""POI category mapping and derived semantic flags."""

import math

import pandas as pd

from .config import CATEGORY_MAP, CHAIN_BRANDS, COMMERCIAL_CATEGORIES, COMMUNITY_KEYWORDS

_CHAIN_PATTERN = "|".join(CHAIN_BRANDS)
_COMMUNITY_PATTERN = "|".join(COMMUNITY_KEYWORDS)


def map_poi_category(cate_a: str | None) -> str:
    if not isinstance(cate_a, str) or not cate_a.strip():
        return "other"
    return CATEGORY_MAP.get(cate_a.strip(), "other")


def is_chain_brand(name: str | None) -> bool:
    if not isinstance(name, str):
        return False
    return any(brand.lower() in name.lower() for brand in CHAIN_BRANDS)


def shannon_entropy(counts) -> float:
    vals = [float(c) for c in counts if c and c > 0]
    total = sum(vals)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in vals)


def add_feature_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category"] = df["cate_A"].apply(map_poi_category)
    df["is_commercial"] = df["category"].isin(COMMERCIAL_CATEGORIES).astype(int)
    df["is_chain"] = (
        df["name"].str.contains(_CHAIN_PATTERN, case=False, na=False, regex=True).astype(int)
    )
    cat_mask = df["category"].isin({"food", "retail", "life_service"})
    text = df["name"].fillna("").astype(str)
    for col in ["cate_B", "cate_C"]:
        if col in df.columns:
            text = text + df[col].fillna("").astype(str)
    has_kw = text.str.contains(_COMMUNITY_PATTERN, case=False, na=False, regex=True)
    df["is_community_commerce"] = (cat_mask & has_kw).astype(int)
    return df
