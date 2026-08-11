"""Audit coordinate-system ambiguity in the open Chengdu Fang observations.

The replication package supplies longitude/latitude but does not identify the
coordinate reference system.  This audit compares several common China
coordinate transformations against same-name Anjuke points and against the
reported distance-to-subway field.  It does not mutate source observations.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "collection"))
sys.path.insert(0, str(ROOT / "src"))
from amap_transit_fetcher import gcj02_to_wgs84  # noqa: E402

from urban_intervention.data.paths import (  # noqa: E402
    OUTPUT_HOUSING_ACQUISITION_DIR,
    RAW_ANJUKE_DIR,
    RAW_OPEN_DATASET_DIR,
    RESOLVED_STATION_EVENTS,
)

Transform = Callable[[float, float], tuple[float, float]]
EARTH_RADIUS_M = 6_371_000.0


def bd09_to_gcj02(lon: float, lat: float) -> tuple[float, float]:
    x = lon - 0.0065
    y = lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * math.pi * 3000.0 / 180.0)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * math.pi * 3000.0 / 180.0)
    return z * math.cos(theta), z * math.sin(theta)


def bd09_to_wgs84(lon: float, lat: float) -> tuple[float, float]:
    return gcj02_to_wgs84(*bd09_to_gcj02(lon, lat))


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = re.sub(r"[（(][^）)]*[）)]", "", text)
    return re.sub(r"[\s·・,，。./\\\-_—]+", "", text)


def haversine_m(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> np.ndarray:
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    value = (
        np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(value))


def transform_points(frame: pd.DataFrame, transform: Transform) -> np.ndarray:
    return np.asarray(
        [
            transform(float(lon), float(lat))
            for lon, lat in zip(frame["longitude"], frame["latitude"], strict=True)
        ],
        dtype=float,
    )


def same_name_stats(matches: pd.DataFrame, transform: Transform) -> dict[str, object]:
    transformed = transform_points(matches, transform)
    distance = haversine_m(
        transformed[:, 0],
        transformed[:, 1],
        matches["reference_lon"].to_numpy(float),
        matches["reference_lat"].to_numpy(float),
    )
    candidates = matches[["id"]].copy()
    candidates["distance_m"] = distance
    best = candidates.sort_values("distance_m").drop_duplicates("id")
    return {
        "matched_communities": int(len(best)),
        "median_m": float(best["distance_m"].median()),
        "p75_m": float(best["distance_m"].quantile(0.75)),
        "p90_m": float(best["distance_m"].quantile(0.90)),
        "within_100m": int(best["distance_m"].le(100).sum()),
        "within_500m": int(best["distance_m"].le(500).sum()),
    }


def nearest_station_distance(points: np.ndarray, stations: pd.DataFrame) -> np.ndarray:
    point_lon = points[:, 0][:, None]
    point_lat = points[:, 1][:, None]
    station_lon = stations["wgs84_lon"].to_numpy(float)[None, :]
    station_lat = stations["wgs84_lat"].to_numpy(float)[None, :]
    distance = haversine_m(point_lon, point_lat, station_lon, station_lat)
    return distance.min(axis=1)


def subway_stats(
    communities: pd.DataFrame,
    stations: pd.DataFrame,
    transform: Transform,
) -> dict[str, float]:
    predicted = nearest_station_distance(transform_points(communities, transform), stations)
    observed = communities["subway"].to_numpy(float)
    absolute_error = np.abs(predicted - observed)
    return {
        "communities": int(len(communities)),
        "median_absolute_error_m": float(np.median(absolute_error)),
        "p75_absolute_error_m": float(np.quantile(absolute_error, 0.75)),
        "correlation": float(np.corrcoef(predicted, observed)[0, 1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fang",
        type=Path,
        default=RAW_OPEN_DATASET_DIR / "mendeley_wpv5zn9rxp_v1" / "Final data for Fang.dta",
    )
    parser.add_argument(
        "--anjuke-dir",
        type=Path,
        default=RAW_ANJUKE_DIR,
    )
    parser.add_argument(
        "--stations",
        type=Path,
        default=RESOLVED_STATION_EVENTS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_HOUSING_ACQUISITION_DIR / "chengdu_wpv5zn9rxp_coordinate_audit.json",
    )
    args = parser.parse_args()

    source = pd.read_stata(args.fang, convert_categoricals=False)
    communities = source[["id", "name", "longitude", "latitude", "subway"]].drop_duplicates("id")
    communities = communities.dropna(subset=["longitude", "latitude", "subway"]).copy()
    communities["normalized_name"] = communities["name"].map(normalize_name)

    anjuke_path = next(
        (
            path
            for path in args.anjuke_dir.glob("*_community_ext.csv")
            if path.name.startswith("成都")
        ),
        None,
    )
    if anjuke_path is None:
        raise FileNotFoundError(f"No Chengdu Anjuke community file found in {args.anjuke_dir}")
    anjuke = pd.read_csv(
        anjuke_path,
        encoding="utf-8-sig",
        usecols=["名称", "坐标"],
    )
    coordinates = anjuke["坐标"].str.extract(r"POINT\(([-0-9.]+)\s+([-0-9.]+)\)")
    anjuke["reference_lon"] = pd.to_numeric(coordinates[0], errors="coerce")
    anjuke["reference_lat"] = pd.to_numeric(coordinates[1], errors="coerce")
    anjuke["normalized_name"] = anjuke["名称"].map(normalize_name)
    matches = communities.merge(anjuke, on="normalized_name", how="inner")

    stations = pd.read_parquet(args.stations)
    stations = stations[
        stations["city_key"].eq("chengdu")
        & stations["opening_year"].le(2021)
        & stations[["wgs84_lon", "wgs84_lat"]].notna().all(axis=1)
    ]
    transforms: dict[str, Transform] = {
        "identity": lambda lon, lat: (lon, lat),
        "gcj02_to_wgs84": gcj02_to_wgs84,
        "bd09_to_gcj02": bd09_to_gcj02,
        "bd09_to_wgs84": bd09_to_wgs84,
    }
    id_groups = {
        "all": communities,
        "community_id_le_400": communities[communities["id"].le(400)],
        "community_id_gt_400": communities[communities["id"].gt(400)],
    }
    report = {
        "schema": "chengdu_open_coordinate_audit_v1",
        "source": "mendeley_wpv5zn9rxp_v1",
        "source_coordinate_crs_declared": False,
        "unique_priced_communities": int(len(communities)),
        "same_name_anjuke_matches": int(matches["id"].nunique()),
        "reference_note": (
            "Anjuke published points are a relative cross-source reference; their "
            "own pipeline interpretation as GCJ-02 is not independently verified."
        ),
        "same_name_reference_results": {
            name: same_name_stats(matches, transform) for name, transform in transforms.items()
        },
        "subway_distance_results_open_through_2021": {
            group_name: {
                transform_name: subway_stats(group, stations, transform)
                for transform_name, transform in transforms.items()
            }
            for group_name, group in id_groups.items()
        },
        "decision": "coordinate_crs_unresolved_do_not_assign_to_500m_grid",
        "reason": (
            "No single transformation is uniformly supported across the sample; "
            "the best candidate differs by source community-ID cohort, while the "
            "replication documentation does not state a CRS. Preserve the points "
            "for audit but require author metadata or an independent control-point "
            "crosswalk before grid aggregation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"audit={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
