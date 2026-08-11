"""Build the canonical housing community registry and source crosswalk.

Anjuke community IDs anchor the registry where available. Purchased Lianjia
communities are linked by normalized name first and by a conservative
name-plus-distance rule second. Wayback names are admitted by exact normalized
crosswalk and otherwise retained as unresolved source communities. Beijing's
independent AOI is matched as a primary geometry source; no geometry is
silently substituted.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "collection"))
from amap_transit_fetcher import gcj02_to_wgs84  # noqa: E402

from urban_intervention.config.project import ACTIVE_CITIES, CITIES  # noqa: E402
from urban_intervention.data.paths import (  # noqa: E402
    OUTPUT_HOUSING_FUSION_DIR,
    RAW_ANJUKE_DIR,
    RAW_COMMUNITY_AOI_DIR,
    RAW_WAYBACK_PARSED_DIR,
    REFERENCE_HOUSING_DIR,
    STAGING_LIANJIA_TRANSACTIONS_DIR,
)

TX_DIR = STAGING_LIANJIA_TRANSACTIONS_DIR
ANJUKE_DIR = RAW_ANJUKE_DIR
WAYBACK_DIR = RAW_WAYBACK_PARSED_DIR
BEIJING_AOI = RAW_COMMUNITY_AOI_DIR / "baidu_beijing" / "房地产.shp"
OUTPUT_DIR = REFERENCE_HOUSING_DIR
REPORT_DIR = OUTPUT_HOUSING_FUSION_DIR


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    return re.sub(r"[\s\-—_·•,，、。.;；:：()（）\[\]【】]+", "", text)


def parse_point(value: object) -> tuple[float, float] | None:
    match = re.search(r"POINT\(\s*([\d.]+)\s+([\d.]+)\s*\)", str(value or ""))
    if not match:
        return None
    lon, lat = float(match.group(1)), float(match.group(2))
    return gcj02_to_wgs84(lon, lat)


def stable_id(city: str, source: str, value: str) -> str:
    digest = hashlib.sha1(f"{city}|{source}|{value}".encode()).hexdigest()[:16]
    return f"{city}_{source}_{digest}"


def lianjia_communities() -> pd.DataFrame:
    paths_by_city: dict[str, list[Path]] = defaultdict(list)
    for path in TX_DIR.glob("*/*.parquet"):
        paths_by_city[path.stem].append(path)
    frames = []
    for city, paths in sorted(paths_by_city.items()):
        columns = [
            "community_name",
            "community_name_normalized",
            "lon",
            "lat",
            "year",
            "source_record_id",
            "is_valid",
        ]
        frame = pd.concat(
            [pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True
        )
        frame = frame[frame["is_valid"]].copy()
        grouped = frame.groupby("community_name_normalized", as_index=False).agg(
            source_name=("community_name", "first"),
            centroid_lon=("lon", "median"),
            centroid_lat=("lat", "median"),
            transaction_count=("source_record_id", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
        )
        grouped.insert(0, "city_key", city)
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _find_column(columns: list[object], keyword: str, fallback: int | None = None) -> object | None:
    for column in columns:
        if keyword in str(column):
            return column
    if fallback is not None and fallback < len(columns):
        return columns[fallback]
    return None


def anjuke_communities() -> pd.DataFrame:
    rows = []
    for city in ACTIVE_CITIES:
        city_name = CITIES[city]["name"]
        candidates = sorted(ANJUKE_DIR.glob(f"{city_name}*_community_ext.csv"))
        if not candidates:
            continue
        frame = pd.read_csv(candidates[0])
        columns = list(frame.columns)
        id_col = _find_column(columns, "ID", 0)
        name_col = _find_column(columns, "名称", 1)
        coord_col = _find_column(columns, "坐标", 2)
        boundary_col = _find_column(columns, "边界", 5)
        price_col = _find_column(columns, "价格", 3)
        for source_row, record in frame.iterrows():
            name = str(record.get(name_col, "") or "").strip()
            normalized = normalize_name(name)
            if len(normalized) < 2:
                continue
            point = parse_point(record.get(coord_col))
            raw_id = str(record.get(id_col, "") or "").strip()
            community_id = stable_id(city, "anjuke", raw_id or normalized)
            boundary_value = record.get(boundary_col)
            has_boundary = pd.notna(boundary_value) and len(str(boundary_value).strip()) > 10
            rows.append(
                {
                    "community_id": community_id,
                    "city_key": city,
                    "source_name": name,
                    "normalized_name": normalized,
                    "centroid_lon": point[0] if point else np.nan,
                    "centroid_lat": point[1] if point else np.nan,
                    "anjuke_source_id": raw_id,
                    "anjuke_source_row": int(source_row) + 2,
                    "anjuke_file": str(candidates[0].relative_to(ROOT)),
                    "has_anjuke_boundary": bool(has_boundary),
                    "anjuke_listing_price": pd.to_numeric(record.get(price_col), errors="coerce"),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Source IDs should be stable, but duplicated rows are resolved in favor of
    # a row with a usable boundary.
    result = result.sort_values("has_anjuke_boundary", ascending=False).drop_duplicates(
        ["city_key", "normalized_name"], keep="first"
    )
    return result.reset_index(drop=True)


def match_lianjia_to_anjuke(
    lianjia: pd.DataFrame, anjuke: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matched_rows = []
    unresolved_rows = []
    for city, left in lianjia.groupby("city_key", sort=True):
        right = anjuke[anjuke["city_key"] == city].reset_index(drop=True)
        exact = right.set_index("normalized_name")["community_id"].to_dict()
        valid_right = right[right[["centroid_lon", "centroid_lat"]].notna().all(axis=1)].copy()
        tree = None
        if not valid_right.empty:
            lat0 = float(valid_right["centroid_lat"].median())
            cos_lat = max(math.cos(math.radians(lat0)), 0.1)
            tree = cKDTree(
                np.column_stack(
                    [valid_right["centroid_lon"] * cos_lat, valid_right["centroid_lat"]]
                )
            )
        for record in left.itertuples(index=False):
            method = "unmatched"
            community_id = exact.get(record.community_name_normalized)
            match_score = 1.0 if community_id else np.nan
            match_distance_m = 0.0 if community_id else np.nan
            ambiguous = False
            if community_id:
                method = "exact_normalized_name"
            elif (
                tree is not None
                and np.isfinite(record.centroid_lon)
                and np.isfinite(record.centroid_lat)
            ):
                k = min(8, len(valid_right))
                distance, indices = tree.query(
                    [record.centroid_lon * cos_lat, record.centroid_lat], k=k
                )
                distance = np.atleast_1d(distance) * 111_000.0
                indices = np.atleast_1d(indices)
                candidates = []
                for dist, idx in zip(distance, indices, strict=False):
                    if dist > 1_500:
                        continue
                    candidate = valid_right.iloc[int(idx)]
                    name_score = SequenceMatcher(
                        None, record.community_name_normalized, candidate["normalized_name"]
                    ).ratio()
                    distance_score = max(0.0, 1.0 - float(dist) / 1_500.0)
                    combined = 0.8 * name_score + 0.2 * distance_score
                    candidates.append(
                        (combined, name_score, float(dist), candidate["community_id"])
                    )
                candidates.sort(reverse=True)
                if candidates:
                    best = candidates[0]
                    second = candidates[1] if len(candidates) > 1 else None
                    accept = (best[1] >= 0.82 and best[2] <= 1_000) or (
                        best[1] >= 0.72 and best[2] <= 300
                    )
                    ambiguous = bool(second and best[0] - second[0] < 0.04)
                    if accept and not ambiguous:
                        match_score, _, match_distance_m, community_id = best
                        method = "name_plus_distance"
            if not community_id:
                community_id = stable_id(city, "lianjia", record.community_name_normalized)
                unresolved_rows.append(
                    {
                        "city_key": city,
                        "source": "lianjia_purchased",
                        "source_name": record.source_name,
                        "normalized_name": record.community_name_normalized,
                        "centroid_lon": record.centroid_lon,
                        "centroid_lat": record.centroid_lat,
                        "reason": "ambiguous_candidates"
                        if ambiguous
                        else "no_accepted_anjuke_match",
                    }
                )
            matched_rows.append(
                {
                    "community_id": community_id,
                    "city_key": city,
                    "source": "lianjia_purchased",
                    "source_name": record.source_name,
                    "normalized_name": record.community_name_normalized,
                    "match_method": method,
                    "match_score": match_score,
                    "match_distance_m": match_distance_m,
                    "transaction_count": record.transaction_count,
                    "first_year": record.first_year,
                    "last_year": record.last_year,
                    "centroid_lon": record.centroid_lon,
                    "centroid_lat": record.centroid_lat,
                }
            )
    return pd.DataFrame(matched_rows), pd.DataFrame(unresolved_rows)


def match_beijing_aoi(registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import geopandas as gpd

    if not BEIJING_AOI.exists():
        return pd.DataFrame(), pd.DataFrame()
    aoi = gpd.read_file(BEIJING_AOI)
    aoi = aoi[aoi.geometry.notna() & ~aoi.geometry.is_empty & aoi.geometry.is_valid].copy()
    aoi["source_name"] = aoi["name"].fillna("").astype(str)
    aoi["normalized_name"] = aoi["source_name"].map(normalize_name)
    fallback_ids = pd.Series(aoi.index.astype(str), index=aoi.index)
    aoi["aoi_id"] = aoi["uid"].fillna(fallback_ids).astype(str)
    # Beijing lies in UTM zone 50N. Compute centroids in metres and return
    # them to WGS84; geographic-degree centroids are prohibited by the AOI
    # contract even though the numerical difference is often small here.
    centroids = aoi.to_crs("EPSG:32650").geometry.centroid.to_crs("EPSG:4326")
    aoi["centroid_lon"] = centroids.x
    aoi["centroid_lat"] = centroids.y
    aoi = aoi.sort_values("aoi_id").drop_duplicates("normalized_name", keep="first")
    exact = aoi.set_index("normalized_name")["aoi_id"].to_dict()
    by_id = aoi.set_index("aoi_id")
    lat0 = float(aoi["centroid_lat"].median())
    cos_lat = max(math.cos(math.radians(lat0)), 0.1)
    tree = cKDTree(np.column_stack([aoi["centroid_lon"] * cos_lat, aoi["centroid_lat"]]))
    matches = []
    unresolved = []
    for record in registry[registry["city_key"] == "beijing"].itertuples(index=False):
        aoi_id = exact.get(record.normalized_name)
        method = "exact_normalized_name" if aoi_id else "unmatched"
        distance_m = 0.0 if aoi_id else np.nan
        score = 1.0 if aoi_id else np.nan
        if not aoi_id and np.isfinite(record.centroid_lon) and np.isfinite(record.centroid_lat):
            distance, indices = tree.query(
                [record.centroid_lon * cos_lat, record.centroid_lat], k=8
            )
            candidates = []
            for dist, idx in zip(
                np.atleast_1d(distance) * 111_000.0, np.atleast_1d(indices), strict=False
            ):
                if dist > 1_000:
                    continue
                candidate = aoi.iloc[int(idx)]
                name_score = SequenceMatcher(
                    None, record.normalized_name, candidate["normalized_name"]
                ).ratio()
                combined = 0.8 * name_score + 0.2 * max(0.0, 1.0 - float(dist) / 1_000.0)
                candidates.append((combined, name_score, float(dist), candidate["aoi_id"]))
            candidates.sort(reverse=True)
            if candidates:
                best = candidates[0]
                second = candidates[1] if len(candidates) > 1 else None
                if best[1] >= 0.82 and not (second and best[0] - second[0] < 0.04):
                    score, _, distance_m, aoi_id = best
                    method = "name_plus_distance"
        if aoi_id:
            source = by_id.loc[aoi_id]
            matches.append(
                {
                    "community_id": record.community_id,
                    "aoi_id": str(aoi_id),
                    "aoi_source_name": source["source_name"],
                    "match_method": method,
                    "match_score": score,
                    "match_distance_m": distance_m,
                }
            )
        else:
            unresolved.append(
                {
                    "community_id": record.community_id,
                    "city_key": "beijing",
                    "source": "beijing_independent_aoi",
                    "source_name": record.canonical_name,
                    "normalized_name": record.normalized_name,
                    "reason": "no_accepted_beijing_aoi_match",
                }
            )
    return pd.DataFrame(matches), pd.DataFrame(unresolved)


def wayback_crosswalk(registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = {
        (row.city_key, row.normalized_name): row.community_id
        for row in registry.itertuples(index=False)
    }
    rows = []
    unresolved = []
    for path in sorted(WAYBACK_DIR.glob("*_wayback_*.csv")):
        city = path.name.split("_wayback_", 1)[0]
        if city not in CITIES:
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "community" not in frame.columns:
            continue
        source = path.stem.split("_wayback_", 1)[1]
        for name in frame["community"].dropna().astype(str).unique():
            normalized = normalize_name(name)
            community_id = lookup.get((city, normalized))
            proposed_id = stable_id(city, "wayback", normalized)
            record = {
                "community_id": community_id,
                "proposed_community_id": proposed_id if not community_id else "",
                "city_key": city,
                "source": f"wayback_{source}",
                "source_name": name,
                "normalized_name": normalized,
                "match_method": "exact_normalized_name" if community_id else "unresolved",
            }
            rows.append(record)
            if not community_id:
                unresolved.append({**record, "reason": "no_exact_registry_match"})
    return pd.DataFrame(rows), pd.DataFrame(unresolved)


def main() -> int:
    lianjia = lianjia_communities()
    anjuke = anjuke_communities()
    lianjia_xwalk, unresolved_lianjia = match_lianjia_to_anjuke(lianjia, anjuke)

    registry = anjuke.rename(
        columns={"source_name": "canonical_name", "normalized_name": "normalized_name"}
    )[
        [
            "community_id",
            "city_key",
            "canonical_name",
            "normalized_name",
            "centroid_lon",
            "centroid_lat",
            "has_anjuke_boundary",
            "anjuke_source_id",
        ]
    ].copy()
    unmatched = lianjia_xwalk[~lianjia_xwalk["community_id"].isin(registry["community_id"])].copy()
    if not unmatched.empty:
        registry = pd.concat(
            [
                registry,
                unmatched.rename(columns={"source_name": "canonical_name"}).assign(
                    has_anjuke_boundary=False, anjuke_source_id=""
                )[
                    [
                        "community_id",
                        "city_key",
                        "canonical_name",
                        "normalized_name",
                        "centroid_lon",
                        "centroid_lat",
                        "has_anjuke_boundary",
                        "anjuke_source_id",
                    ]
                ],
            ],
            ignore_index=True,
        )
    # Prefer Lianjia transaction centroids when a registry community has one.
    lj_centroids = lianjia_xwalk.groupby("community_id", as_index=False).agg(
        lj_lon=("centroid_lon", "median"),
        lj_lat=("centroid_lat", "median"),
        transaction_count=("transaction_count", "sum"),
        first_year=("first_year", "min"),
        last_year=("last_year", "max"),
    )
    registry = registry.merge(lj_centroids, on="community_id", how="left")
    registry["centroid_lon"] = registry["lj_lon"].combine_first(registry["centroid_lon"])
    registry["centroid_lat"] = registry["lj_lat"].combine_first(registry["centroid_lat"])
    registry = registry.drop(columns=["lj_lon", "lj_lat"])
    registry["transaction_count"] = registry["transaction_count"].fillna(0).astype(int)
    registry["aoi_source"] = np.where(registry["has_anjuke_boundary"], "anjuke", "pending_fallback")
    registry["aoi_quality"] = np.where(registry["has_anjuke_boundary"], "A", "pending")
    registry["match_status"] = np.where(
        registry["has_anjuke_boundary"], "boundary_available", "aoi_pending"
    )

    beijing_matches, unresolved_beijing = match_beijing_aoi(registry)
    if not beijing_matches.empty:
        registry = registry.merge(
            beijing_matches[["community_id", "aoi_id"]].rename(
                columns={"aoi_id": "beijing_aoi_id"}
            ),
            on="community_id",
            how="left",
        )
        mask = registry["beijing_aoi_id"].notna()
        registry.loc[mask, "aoi_source"] = "beijing_independent"
        registry.loc[mask, "aoi_quality"] = "A"
        registry.loc[mask, "match_status"] = "boundary_available"
    else:
        registry["beijing_aoi_id"] = np.nan

    registry["aoi_id"] = registry["beijing_aoi_id"].combine_first(
        registry["anjuke_source_id"].replace("", np.nan)
    )
    registry["district"] = ""
    registry["aliases"] = registry["canonical_name"]
    registry["boundary_area_m2"] = np.nan

    wayback_xwalk, unresolved_wayback = wayback_crosswalk(registry)
    crosswalk = pd.concat(
        [
            lianjia_xwalk,
            anjuke.assign(source="anjuke", match_method="source_anchor").rename(
                columns={"source_name": "source_name"}
            )[
                [
                    "community_id",
                    "city_key",
                    "source",
                    "source_name",
                    "normalized_name",
                    "match_method",
                ]
            ],
            wayback_xwalk,
        ],
        ignore_index=True,
        sort=False,
    )
    unresolved = pd.concat(
        [unresolved_lianjia, unresolved_beijing, unresolved_wayback], ignore_index=True, sort=False
    )

    registry = registry.drop_duplicates("community_id").sort_values(["city_key", "community_id"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(OUTPUT_DIR / "community_registry.parquet", index=False)
    crosswalk.to_parquet(OUTPUT_DIR / "community_source_crosswalk.parquet", index=False)
    unresolved.to_csv(
        REPORT_DIR / "community_unresolved_matches.csv", index=False, encoding="utf-8-sig"
    )
    beijing_matches.to_csv(
        REPORT_DIR / "beijing_aoi_matches.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "registry_communities": len(registry),
        "cities": int(registry["city_key"].nunique()),
        "anjuke_boundary_communities": int(registry["has_anjuke_boundary"].sum()),
        "lianjia_source_communities": len(lianjia_xwalk),
        "lianjia_exact_matches": int(
            (lianjia_xwalk["match_method"] == "exact_normalized_name").sum()
        ),
        "lianjia_spatial_name_matches": int(
            (lianjia_xwalk["match_method"] == "name_plus_distance").sum()
        ),
        "lianjia_unmatched": int((lianjia_xwalk["match_method"] == "unmatched").sum()),
        "beijing_primary_aoi_matches": len(beijing_matches),
        "wayback_source_names": len(wayback_xwalk),
        "wayback_exact_matches": int(
            (wayback_xwalk["match_method"] == "exact_normalized_name").sum()
        )
        if not wayback_xwalk.empty
        else 0,
        "unresolved_rows": len(unresolved),
    }
    (REPORT_DIR / "community_registry_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
