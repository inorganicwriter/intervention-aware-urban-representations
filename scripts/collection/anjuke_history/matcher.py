"""Match parsed Anjuke series to the project community registry.

The registry key is (city_key, normalized_name) with aliases.  Matching order:
1. exact normalized_name within the city;
2. aliases within the city;
3. normalized_name after dropping parentheticals / suffix tokens
   (e.g. "阳光花园(一期)" vs "阳光花园");
4. coordinate fallback is intentionally NOT used here: the registry centroid
   is the community's own geometry, not the Anjuke page location, so any
   distance rule would risk wrong joins.  Unmatched rows are reported
   explicitly and excluded from the labels (they cannot be grid-bridged).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_registry() -> pd.DataFrame:
    path = (
        Path(__file__).resolve().parents[3]
        / "data" / "active" / "reference" / "housing" / "community_registry.parquet"
    )
    registry = pd.read_parquet(
        path,
        columns=[
            "community_id", "city_key", "normalized_name", "aliases",
            "district", "anjuke_source_id", "aoi_bridge_admitted",
        ],
    )
    registry["aliases"] = registry["aliases"].fillna("")
    registry["_alias_set"] = registry["aliases"].apply(
        lambda s: {a.strip() for a in str(s).split(";") if a.strip()}
        if isinstance(s, str)
        else set()
    )
    return registry


def _core(name: str) -> str:
    """Strip parentheticals and common suffixes for loose matching."""
    import re

    out = re.sub(r"[（(].*?[)）]", "", name)
    out = re.sub(r"(小区|公寓|花园|苑|城|大厦|府|居|湾|里|广场)$", "", out)
    return out.strip()


def match_communities(parsed: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Join parsed rows to registry rows.  Returns the matched frame with
    ``community_id``; unmatched rows carry NaN community_id.
    """
    if "_alias_set" not in registry.columns:
        registry = registry.copy()
        registry["_alias_set"] = registry["aliases"].apply(
            lambda s: {a.strip() for a in str(s).split(";") if a.strip()}
            if isinstance(s, str)
            else set()
        )
    if parsed.empty:
        return parsed.assign(community_id=pd.Series(dtype="object"), match_quality="")
    by_city = {
        city: group for city, group in registry.groupby("city_key")
    }
    by_city_aliases = {
        city: {_core(n): cid for n, cid in zip(group["normalized_name"], group["community_id"], strict=False)}
        | {
            alias: cid
            for aliases, cid in zip(group["_alias_set"], group["community_id"], strict=False)
            for alias in aliases
        }
        for city, group in registry.groupby("city_key")
    }
    by_city_core = {
        city: {
            _core(n): cid
            for n, cid in zip(group["normalized_name"], group["community_id"], strict=False)
        }
        for city, group in registry.groupby("city_key")
    }

    rows = []
    for _, row in parsed.iterrows():
        city = row["city"]
        name = str(row.get("name", "")).strip()
        cid: object = None
        quality = "none"
        if name and city in by_city:
            exact = by_city[city]
            exact_map = dict(zip(exact["normalized_name"], exact["community_id"], strict=False))
            if name in exact_map:
                cid, quality = exact_map[name], "exact"
            elif name in by_city_aliases[city]:
                cid, quality = by_city_aliases[city][name], "alias"
            elif _core(name) in by_city_core[city]:
                cid, quality = by_city_core[city][_core(name)], "loose"
        rows.append({"community_id": cid, "match_quality": quality})

    matched = pd.DataFrame(rows)
    return pd.concat([parsed.reset_index(drop=True), matched], axis=1)


def write_matched(parsed: pd.DataFrame, city: str) -> Path:
    """Persist the matched frame and return its path."""
    registry = load_registry()
    out = match_communities(parsed, registry)
    path = config.MATCH_DIR / f"{city}_matched.parquet"
    out.to_parquet(path, index=False)
    return path
