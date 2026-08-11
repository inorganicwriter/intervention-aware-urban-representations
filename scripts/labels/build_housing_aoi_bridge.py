"""Build audited housing-community AOIs and area-weighted 500 m grid bridges.

Primary geometry hierarchy
--------------------------
1. Beijing independent real-estate AOI (when the registry has an audited ID)
2. Anjuke community boundary (GCJ-02 vertices converted to WGS84)
3. Robust convex hull of distinct purchased-Lianjia transaction points
4. City-calibrated circular buffer, with a 250 m default when calibration is
   unavailable

The script calculates all lengths and areas in a city-specific UTM CRS.  It
publishes raw AOI coverage as well as normalized allocation weights; weights
are normalized only when the reference grid covers at least 99.5% of the AOI.
Old nearest-grid and full-count-copy AOI labels are deliberately not read.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer
from shapely import STRtree, from_wkt, make_valid, to_wkb
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import transform, unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "collection"))
from amap_transit_fetcher import gcj02_to_wgs84  # noqa: E402

from urban_intervention.config.project import ACTIVE_CITIES, CITIES  # noqa: E402
from urban_intervention.data.paths import (  # noqa: E402
    COMMUNITY_REGISTRY,
    COMMUNITY_SOURCE_CROSSWALK,
    OUTPUT_HOUSING_FUSION_DIR,
    RAW_ANJUKE_DIR,
    RAW_COMMUNITY_AOI_DIR,
    REFERENCE_GRID_DIR,
    REFERENCE_HOUSING_DIR,
    STAGING_LIANJIA_TRANSACTIONS_DIR,
    grid_path,
)

REGISTRY_PATH = COMMUNITY_REGISTRY
CROSSWALK_PATH = COMMUNITY_SOURCE_CROSSWALK
TX_DIR = STAGING_LIANJIA_TRANSACTIONS_DIR
ANJUKE_DIR = RAW_ANJUKE_DIR
BEIJING_AOI_PATH = RAW_COMMUNITY_AOI_DIR / "baidu_beijing" / "房地产.shp"
GRID_DIR = REFERENCE_GRID_DIR
OUTPUT_DIR = REFERENCE_HOUSING_DIR
REPORT_DIR = OUTPUT_HOUSING_FUSION_DIR
AOI_PATH = OUTPUT_DIR / "community_aoi.parquet"
BRIDGE_PATH = OUTPUT_DIR / "community_grid_bridge.parquet"
CITY_QA_PATH = REPORT_DIR / "community_aoi_city_qa.csv"
SUMMARY_PATH = REPORT_DIR / "community_aoi_bridge_summary.json"
REPORT_PATH = REPORT_DIR / "community_aoi_bridge_report.md"


MIN_KNOWN_AREA_M2 = 100.0
MAX_KNOWN_AREA_M2 = 5_000_000.0
MIN_HULL_AREA_M2 = 500.0
MAX_HULL_AREA_M2 = 1_000_000.0
HULL_POINT_BUFFER_M = 20.0
HULL_OUTLIER_RADIUS_M = 750.0
MIN_GRID_COVERAGE = 0.995
MAX_GRID_COVERAGE = 1.005
SENSITIVITY_RADII_M = (150.0, 250.0, 400.0)


AOI_COLUMNS = [
    "aoi_variant",
    "community_id",
    "city_key",
    "aoi_source",
    "aoi_quality",
    "source_file",
    "source_feature_id",
    "fallback_reason",
    "buffer_radius_m",
    "n_source_points",
    "area_m2",
    "centroid_lon",
    "centroid_lat",
    "geometry_valid",
    "geometry_crs",
    "projected_crs",
    "grid_coverage_share",
    "bridge_admitted",
    "geometry_wkb",
]

BRIDGE_COLUMNS = [
    "aoi_variant",
    "community_id",
    "city_key",
    "grid_id",
    "aoi_source",
    "aoi_quality",
    "buffer_radius_m",
    "intersection_area_m2",
    "aoi_area_m2",
    "grid_area_m2",
    "community_area_share_raw",
    "community_area_share",
    "grid_area_share",
    "aoi_grid_coverage_share",
    "weights_normalized",
    "bridge_admitted",
]


def _find_column(columns: list[object], keyword: str, fallback: int | None = None) -> object | None:
    for column in columns:
        if keyword in str(column):
            return column
    if fallback is not None and fallback < len(columns):
        return columns[fallback]
    return None


def utm_epsg(lon: float, lat: float) -> int:
    zone = max(1, min(60, int(math.floor((lon + 180.0) / 6.0)) + 1))
    return (32600 if lat >= 0 else 32700) + zone


def polygonal(geometry):
    """Return a valid polygonal geometry or None."""
    if geometry is None or geometry.is_empty:
        return None
    try:
        geometry = make_valid(geometry)
    except Exception:
        try:
            geometry = geometry.buffer(0)
        except Exception:
            return None
    if geometry.is_empty:
        return None
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    parts = []
    if hasattr(geometry, "geoms"):
        parts = [part for part in geometry.geoms if part.geom_type in {"Polygon", "MultiPolygon"}]
    if not parts:
        return None
    merged = unary_union(parts)
    return merged if not merged.is_empty else None


def parse_anjuke_boundary(value: object):
    """Parse semicolon-delimited GCJ-02 vertices and return WGS84 polygon."""
    if pd.isna(value):
        return None
    pairs = re.findall(r"([+-]?\d{2,3}(?:\.\d+)?)\s*,\s*([+-]?\d{1,2}(?:\.\d+)?)", str(value))
    coordinates = []
    for lon_text, lat_text in pairs:
        try:
            lon, lat = float(lon_text), float(lat_text)
            if not (70 <= lon <= 140 and 15 <= lat <= 55):
                continue
            coordinates.append(gcj02_to_wgs84(lon, lat))
        except (TypeError, ValueError):
            continue
    if len(coordinates) < 3:
        return None
    return polygonal(Polygon(coordinates))


def load_anjuke_boundaries(city_key: str) -> dict[str, dict]:
    city_name = CITIES[city_key]["name"]
    candidates = sorted(ANJUKE_DIR.glob(f"{city_name}*_community_ext.csv"))
    if not candidates:
        return {}
    path = candidates[0]
    frame = pd.read_csv(path)
    columns = list(frame.columns)
    id_col = _find_column(columns, "ID", 0)
    boundary_col = _find_column(columns, "边界", 5)
    if id_col is None or boundary_col is None:
        return {}
    result: dict[str, dict] = {}
    for index, record in frame[[id_col, boundary_col]].iterrows():
        source_id = str(record[id_col] or "").strip()
        if not source_id or source_id.lower() == "nan" or source_id in result:
            continue
        geometry = parse_anjuke_boundary(record[boundary_col])
        if geometry is not None:
            result[source_id] = {
                "geometry": geometry,
                "source_file": str(path.relative_to(ROOT)),
                "source_feature_id": source_id,
                "source_row": int(index) + 2,
            }
    return result


def load_beijing_aoi() -> dict[str, dict]:
    if not BEIJING_AOI_PATH.exists():
        return {}
    frame = gpd.read_file(BEIJING_AOI_PATH)
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    elif str(frame.crs).upper() != "EPSG:4326":
        frame = frame.to_crs("EPSG:4326")
    fallback_ids = pd.Series(frame.index.astype(str), index=frame.index)
    ids = frame["uid"].fillna(fallback_ids).astype(str) if "uid" in frame else fallback_ids
    result = {}
    for index, (source_id, geometry) in enumerate(zip(ids, frame.geometry, strict=False)):
        geometry = polygonal(geometry)
        if geometry is None or source_id in result:
            continue
        result[source_id] = {
            "geometry": geometry,
            "source_file": str(BEIJING_AOI_PATH.relative_to(ROOT)),
            "source_feature_id": source_id,
            "source_row": int(index),
        }
    return result


def fallback_transaction_points(
    city_key: str, crosswalk: pd.DataFrame, needed_ids: set[str]
) -> dict[str, np.ndarray]:
    """Return distinct valid transaction coordinates for unresolved AOIs."""
    if not needed_ids:
        return {}
    source = crosswalk[
        (crosswalk["city_key"] == city_key)
        & (crosswalk["source"] == "lianjia_purchased")
        & (crosswalk["community_id"].isin(needed_ids))
    ][["normalized_name", "community_id"]].drop_duplicates("normalized_name")
    if source.empty:
        return {}
    name_to_id = source.set_index("normalized_name")["community_id"]
    paths = sorted(TX_DIR.glob(f"*/{city_key}.parquet"))
    if not paths:
        return {}
    columns = ["community_name_normalized", "lon", "lat", "coordinate_valid", "community_valid"]
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["coordinate_valid"] & frame["community_valid"]].copy()
    frame["community_id"] = frame["community_name_normalized"].map(name_to_id)
    frame = frame[frame["community_id"].notna()]
    frame["lon_round"] = frame["lon"].round(6)
    frame["lat_round"] = frame["lat"].round(6)
    frame = frame.drop_duplicates(["community_id", "lon_round", "lat_round"])
    return {
        community_id: group[["lon_round", "lat_round"]].to_numpy(dtype=float)
        for community_id, group in frame.groupby("community_id", sort=False)
    }


def robust_hull(points_wgs84: np.ndarray, forward: Transformer):
    if points_wgs84 is None or len(points_wgs84) < 3:
        return None, 0
    x, y = forward.transform(points_wgs84[:, 0], points_wgs84[:, 1])
    points = np.column_stack([x, y])
    median = np.median(points, axis=0)
    distance = np.sqrt(((points - median) ** 2).sum(axis=1))
    kept = points[distance <= HULL_OUTLIER_RADIUS_M]
    if len(kept) < 3:
        return None, len(kept)
    # Deterministic cap protects against pathological coordinate-rich groups.
    if len(kept) > 500:
        indices = np.linspace(0, len(kept) - 1, 500, dtype=int)
        kept = kept[indices]
    hull = MultiPoint(kept).convex_hull
    if hull.geom_type != "Polygon" or not (MIN_HULL_AREA_M2 <= hull.area <= MAX_HULL_AREA_M2):
        return None, len(kept)
    buffered = polygonal(hull.buffer(HULL_POINT_BUFFER_M))
    return buffered, len(kept)


def load_grids(city_key: str, forward: Transformer):
    path = grid_path(city_key)
    if not path.exists():
        return path, pd.DataFrame(), [], np.array([])
    frame = pd.read_parquet(path, columns=["grid_id", "geometry_wkt"])
    geometries_wgs84 = list(from_wkt(frame["geometry_wkt"].to_numpy()))
    geometries_projected = [transform(forward.transform, geometry) for geometry in geometries_wgs84]
    areas = np.asarray([geometry.area for geometry in geometries_projected], dtype=float)
    return path, frame, geometries_projected, areas


def geometry_distance_m(
    geometry_projected, lon: object, lat: object, forward: Transformer
) -> float:
    try:
        lon_float, lat_float = float(lon), float(lat)
        if not (np.isfinite(lon_float) and np.isfinite(lat_float)):
            return np.nan
        x, y = forward.transform(lon_float, lat_float)
        return float(geometry_projected.distance(Point(x, y)))
    except (TypeError, ValueError):
        return np.nan


def aoi_record(
    *,
    variant: str,
    community_id: str,
    city_key: str,
    source: str,
    quality: str,
    source_file: str,
    source_feature_id: str,
    fallback_reason: str,
    buffer_radius_m: float | None,
    n_source_points: int,
    geometry_projected,
    inverse: Transformer,
    projected_crs: str,
) -> dict:
    centroid = geometry_projected.centroid
    lon, lat = inverse.transform(centroid.x, centroid.y)
    return {
        "aoi_variant": variant,
        "community_id": community_id,
        "city_key": city_key,
        "aoi_source": source,
        "aoi_quality": quality,
        "source_file": source_file,
        "source_feature_id": source_feature_id,
        "fallback_reason": fallback_reason,
        "buffer_radius_m": np.nan if buffer_radius_m is None else float(buffer_radius_m),
        "n_source_points": int(n_source_points),
        "area_m2": float(geometry_projected.area),
        "centroid_lon": float(lon),
        "centroid_lat": float(lat),
        "geometry_valid": bool(geometry_projected.is_valid and not geometry_projected.is_empty),
        "geometry_crs": "EPSG:4326",
        "projected_crs": projected_crs,
        "grid_coverage_share": np.nan,
        "bridge_admitted": False,
        "geometry_wkb": None,
        "_geometry_projected": geometry_projected,
    }


def build_city_aois(
    city_key: str,
    city_registry: pd.DataFrame,
    crosswalk: pd.DataFrame,
    beijing_aoi: dict[str, dict],
    forward: Transformer,
    inverse: Transformer,
    projected_crs: str,
) -> tuple[list[dict], dict]:
    anjuke = load_anjuke_boundaries(city_key)
    primary: dict[str, dict] = {}
    validation: list[dict] = []
    rejected_known = Counter()

    def admit_known(record, asset, source, quality="A"):
        geometry_wgs84 = polygonal(asset["geometry"])
        if geometry_wgs84 is None:
            rejected_known["invalid_geometry"] += 1
            return None
        geometry_projected = polygonal(transform(forward.transform, geometry_wgs84))
        if geometry_projected is None:
            rejected_known["projection_failure"] += 1
            return None
        area = float(geometry_projected.area)
        if not (MIN_KNOWN_AREA_M2 <= area <= MAX_KNOWN_AREA_M2):
            rejected_known["implausible_area"] += 1
            return None
        distance = geometry_distance_m(
            geometry_projected, record.centroid_lon, record.centroid_lat, forward
        )
        if np.isfinite(distance) and distance > 5_000:
            rejected_known["centroid_over_5km"] += 1
            return None
        return aoi_record(
            variant="primary",
            community_id=record.community_id,
            city_key=city_key,
            source=source,
            quality=quality,
            source_file=asset["source_file"],
            source_feature_id=asset["source_feature_id"],
            fallback_reason="",
            buffer_radius_m=None,
            n_source_points=0,
            geometry_projected=geometry_projected,
            inverse=inverse,
            projected_crs=projected_crs,
        )

    for record in city_registry.itertuples(index=False):
        independent = None
        if city_key == "beijing" and pd.notna(record.beijing_aoi_id):
            independent = beijing_aoi.get(str(record.beijing_aoi_id))
        anjuke_asset = None
        if pd.notna(record.anjuke_source_id) and str(record.anjuke_source_id):
            anjuke_asset = anjuke.get(str(record.anjuke_source_id))

        if independent is not None:
            admitted = admit_known(record, independent, "beijing_independent", "A")
            if admitted is not None:
                primary[record.community_id] = admitted
                if anjuke_asset is not None:
                    validation_record = admit_known(record, anjuke_asset, "anjuke", "A")
                    if validation_record is not None:
                        validation_record["aoi_variant"] = "anjuke_validation"
                        validation.append(validation_record)
                continue
        if anjuke_asset is not None:
            admitted = admit_known(record, anjuke_asset, "anjuke", "A")
            if admitted is not None:
                primary[record.community_id] = admitted

    known_areas = [record["area_m2"] for record in primary.values()]
    if known_areas:
        equivalent_radius = math.sqrt(float(np.median(known_areas)) / math.pi)
        calibrated_radius = float(np.clip(equivalent_radius, 150.0, 500.0))
        calibration_source = "city_known_aoi_median"
    else:
        calibrated_radius = 250.0
        calibration_source = "default_250m"

    missing_ids = set(city_registry["community_id"]) - set(primary)
    tx_points = fallback_transaction_points(city_key, crosswalk, missing_ids)
    registry_by_id = city_registry.set_index("community_id")
    fallback_records: list[dict] = []
    fallback_counts = Counter()

    for community_id in sorted(missing_ids):
        record = registry_by_id.loc[community_id]
        points = tx_points.get(community_id)
        hull, n_points = robust_hull(points, forward)
        if hull is not None:
            main = aoi_record(
                variant="primary",
                community_id=community_id,
                city_key=city_key,
                source="transaction_hull",
                quality="B",
                source_file="",
                source_feature_id=community_id,
                fallback_reason="known_boundary_unavailable",
                buffer_radius_m=HULL_POINT_BUFFER_M,
                n_source_points=n_points,
                geometry_projected=hull,
                inverse=inverse,
                projected_crs=projected_crs,
            )
            fallback_counts["transaction_hull"] += 1
        else:
            lon = pd.to_numeric(record["centroid_lon"], errors="coerce")
            lat = pd.to_numeric(record["centroid_lat"], errors="coerce")
            if not (np.isfinite(lon) and np.isfinite(lat)):
                fallback_counts["missing_centroid"] += 1
                continue
            x, y = forward.transform(float(lon), float(lat))
            geometry = Point(x, y).buffer(calibrated_radius)
            source = (
                "city_calibrated_buffer"
                if calibration_source != "default_250m"
                else "default_buffer"
            )
            quality = "C" if source == "city_calibrated_buffer" else "D"
            main = aoi_record(
                variant="primary",
                community_id=community_id,
                city_key=city_key,
                source=source,
                quality=quality,
                source_file="",
                source_feature_id=community_id,
                fallback_reason="insufficient_or_degenerate_transaction_points",
                buffer_radius_m=calibrated_radius,
                n_source_points=n_points,
                geometry_projected=geometry,
                inverse=inverse,
                projected_crs=projected_crs,
            )
            fallback_counts[source] += 1
        primary[community_id] = main

        # Sensitivity buffers apply only to communities lacking an accepted
        # observed boundary. They never replace the primary AOI silently.
        lon = float(main["centroid_lon"])
        lat = float(main["centroid_lat"])
        x, y = forward.transform(lon, lat)
        for radius in SENSITIVITY_RADII_M:
            fallback_records.append(
                aoi_record(
                    variant=f"buffer_{int(radius)}m",
                    community_id=community_id,
                    city_key=city_key,
                    source="sensitivity_buffer",
                    quality="D",
                    source_file="",
                    source_feature_id=community_id,
                    fallback_reason=f"sensitivity_for_{main['aoi_source']}",
                    buffer_radius_m=radius,
                    n_source_points=n_points,
                    geometry_projected=Point(x, y).buffer(radius),
                    inverse=inverse,
                    projected_crs=projected_crs,
                )
            )

    metadata = {
        "anjuke_boundaries_parsed": len(anjuke),
        "known_primary": len(known_areas),
        "calibrated_radius_m": calibrated_radius,
        "calibration_source": calibration_source,
        "rejected_known": dict(rejected_known),
        "fallback_counts": dict(fallback_counts),
    }
    return list(primary.values()) + validation + fallback_records, metadata


def build_city_bridge(
    city_key: str,
    aois: list[dict],
    grid_frame: pd.DataFrame,
    grid_geometries: list,
    grid_areas: np.ndarray,
    inverse: Transformer,
) -> tuple[list[dict], dict]:
    if grid_frame.empty:
        return [], {"bridge_rows": 0, "admitted_aois": 0, "uncovered_aois": len(aois)}
    tree = STRtree(grid_geometries)
    grid_ids = grid_frame["grid_id"].astype(str).to_numpy()
    bridge_rows: list[dict] = []
    admitted = 0
    uncovered = 0
    partial = 0

    for record in aois:
        geometry = record["_geometry_projected"]
        candidates = np.asarray(tree.query(geometry, predicate="intersects"), dtype=int)
        intersections = []
        for index in candidates:
            intersection = geometry.intersection(grid_geometries[int(index)])
            area = float(intersection.area)
            if area > 0.01:
                intersections.append((int(index), area))
        total_intersection = float(sum(area for _, area in intersections))
        aoi_area = float(record["area_m2"])
        coverage = total_intersection / aoi_area if aoi_area > 0 else 0.0
        is_admitted = MIN_GRID_COVERAGE <= coverage <= MAX_GRID_COVERAGE
        record["grid_coverage_share"] = coverage
        record["bridge_admitted"] = is_admitted
        if not intersections:
            uncovered += 1
            continue
        if is_admitted:
            admitted += 1
        else:
            partial += 1
        for index, area in intersections:
            raw_share = area / aoi_area
            normalized_share = area / total_intersection if is_admitted else np.nan
            bridge_rows.append(
                {
                    "aoi_variant": record["aoi_variant"],
                    "community_id": record["community_id"],
                    "city_key": city_key,
                    "grid_id": grid_ids[index],
                    "aoi_source": record["aoi_source"],
                    "aoi_quality": record["aoi_quality"],
                    "buffer_radius_m": record["buffer_radius_m"],
                    "intersection_area_m2": area,
                    "aoi_area_m2": aoi_area,
                    "grid_area_m2": float(grid_areas[index]),
                    "community_area_share_raw": raw_share,
                    "community_area_share": normalized_share,
                    "grid_area_share": area / float(grid_areas[index]),
                    "aoi_grid_coverage_share": coverage,
                    "weights_normalized": bool(is_admitted),
                    "bridge_admitted": bool(is_admitted),
                }
            )
        geometry_wgs84 = transform(inverse.transform, geometry)
        record["geometry_wkb"] = bytes(to_wkb(geometry_wgs84))

    # AOIs with no intersection still need geometry in the audit asset.
    for record in aois:
        if record["geometry_wkb"] is None:
            geometry_wgs84 = transform(inverse.transform, record["_geometry_projected"])
            record["geometry_wkb"] = bytes(to_wkb(geometry_wgs84))

    return bridge_rows, {
        "bridge_rows": len(bridge_rows),
        "admitted_aois": admitted,
        "partial_aois": partial,
        "uncovered_aois": uncovered,
    }


class AtomicParquetWriter:
    def __init__(self, path: Path, columns: list[str]):
        self.path = path
        self.temp_path = path.with_suffix(path.suffix + ".tmp")
        self.columns = columns
        self.writer: pq.ParquetWriter | None = None
        if self.temp_path.exists():
            self.temp_path.unlink()

    def write(self, rows: list[dict]):
        if not rows:
            return
        frame = pd.DataFrame(rows)
        frame = frame.reindex(columns=self.columns)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.temp_path, table.schema, compression="zstd")
        self.writer.write_table(table)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.temp_path.replace(self.path)


def validate_outputs(aoi_path: Path, bridge_path: Path) -> dict:
    aois = pd.read_parquet(
        aoi_path,
        columns=[
            "aoi_variant",
            "community_id",
            "city_key",
            "aoi_source",
            "aoi_quality",
            "area_m2",
            "geometry_valid",
            "grid_coverage_share",
            "bridge_admitted",
        ],
    )
    bridge = pd.read_parquet(bridge_path)
    admitted = bridge[bridge["bridge_admitted"]].copy()
    unadmitted = aois[~aois["bridge_admitted"]].copy()
    weight_sums = admitted.groupby(["aoi_variant", "community_id"])["community_area_share"].sum()
    raw_sums = admitted.groupby(["aoi_variant", "community_id"])["community_area_share_raw"].sum()
    return {
        "aoi_rows": int(len(aois)),
        "primary_aoi_rows": int((aois["aoi_variant"] == "primary").sum()),
        "unique_aoi_keys": int(aois[["aoi_variant", "community_id"]].drop_duplicates().shape[0]),
        "invalid_geometry_rows": int((~aois["geometry_valid"]).sum()),
        "bridge_rows": int(len(bridge)),
        "unique_bridge_keys": int(
            bridge[["aoi_variant", "community_id", "grid_id"]].drop_duplicates().shape[0]
        ),
        "admitted_aoi_rows": int(aois["bridge_admitted"].sum()),
        "unadmitted_aoi_rows": int((~aois["bridge_admitted"]).sum()),
        "max_normalized_weight_sum_error": float((weight_sums - 1.0).abs().max())
        if len(weight_sums)
        else None,
        "min_admitted_raw_coverage": float(raw_sums.min()) if len(raw_sums) else None,
        "max_admitted_raw_coverage": float(raw_sums.max()) if len(raw_sums) else None,
        "aoi_source_counts": {str(k): int(v) for k, v in aois["aoi_source"].value_counts().items()},
        "aoi_quality_counts": {
            str(k): int(v) for k, v in aois["aoi_quality"].value_counts().items()
        },
        "unadmitted_by_city_variant": {
            f"{city}|{variant}": int(count)
            for (city, variant), count in unadmitted.groupby(["city_key", "aoi_variant"])
            .size()
            .items()
        },
        "unadmitted_primary_community_ids": unadmitted.loc[
            unadmitted["aoi_variant"] == "primary", "community_id"
        ]
        .astype(str)
        .tolist(),
    }


def update_registry_aoi_status(registry_path: Path, aoi_path: Path):
    """Synchronize the derived registry with the published primary AOI."""
    registry = pd.read_parquet(registry_path)
    primary = pd.read_parquet(
        aoi_path,
        columns=[
            "aoi_variant",
            "community_id",
            "aoi_source",
            "aoi_quality",
            "area_m2",
            "centroid_lon",
            "centroid_lat",
            "grid_coverage_share",
            "bridge_admitted",
        ],
    )
    primary = primary[primary["aoi_variant"] == "primary"].drop(columns="aoi_variant")
    if primary["community_id"].duplicated().any():
        raise RuntimeError("Primary AOI community IDs are not unique")
    if set(primary["community_id"]) != set(registry["community_id"]):
        raise RuntimeError("Primary AOI IDs do not close to the community registry")
    status = primary.rename(
        columns={
            "aoi_source": "published_aoi_source",
            "aoi_quality": "published_aoi_quality",
            "area_m2": "published_boundary_area_m2",
            "centroid_lon": "published_aoi_centroid_lon",
            "centroid_lat": "published_aoi_centroid_lat",
        }
    )
    result = registry.drop(
        columns=[
            "aoi_bridge_admitted",
            "aoi_grid_coverage_share",
        ],
        errors="ignore",
    ).merge(status, on="community_id", how="left", validate="one_to_one")
    result["aoi_source"] = result.pop("published_aoi_source")
    result["aoi_quality"] = result.pop("published_aoi_quality")
    result["boundary_area_m2"] = result.pop("published_boundary_area_m2")
    result["aoi_bridge_admitted"] = result.pop("bridge_admitted").astype(bool)
    result["aoi_grid_coverage_share"] = result.pop("grid_coverage_share")
    result["match_status"] = np.where(
        result["aoi_bridge_admitted"], "aoi_bridge_admitted", "aoi_outside_reference_grid"
    )
    # AOI-derived centroids are authoritative for spatial work while the
    # source centroid remains preserved in the AOI/crosswalk provenance.
    result["centroid_lon"] = result.pop("published_aoi_centroid_lon")
    result["centroid_lat"] = result.pop("published_aoi_centroid_lat")
    temp_path = registry_path.with_suffix(registry_path.suffix + ".tmp")
    result.to_parquet(temp_path, index=False)
    temp_path.replace(registry_path)


def write_report(summary: dict, city_qa: pd.DataFrame):
    validation = summary["validation"]
    primary = city_qa[city_qa["aoi_variant"] == "primary"] if "aoi_variant" in city_qa else city_qa
    lines = [
        "# Housing community AOI and grid bridge report",
        "",
        f"Updated: {datetime.now().date().isoformat()}",
        "",
        "## Published assets",
        "",
        "- `data/active/reference/housing/community_aoi.parquet`",
        "- `data/active/reference/housing/community_grid_bridge.parquet`",
        "- `outputs/housing_fusion/community_aoi_city_qa.csv`",
        "- `outputs/housing_fusion/community_aoi_bridge_summary.json`",
        "",
        "## Validation",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| AOI rows (all variants) | {validation['aoi_rows']:,} |",
        f"| Primary AOIs | {validation['primary_aoi_rows']:,} |",
        f"| Duplicate AOI keys | {validation['aoi_rows'] - validation['unique_aoi_keys']:,} |",
        f"| Invalid geometries | {validation['invalid_geometry_rows']:,} |",
        f"| Community-grid bridge rows | {validation['bridge_rows']:,} |",
        f"| Duplicate bridge keys | {validation['bridge_rows'] - validation['unique_bridge_keys']:,} |",
        f"| AOIs admitted to bridge | {validation['admitted_aoi_rows']:,} |",
        f"| AOIs not admitted | {validation['unadmitted_aoi_rows']:,} |",
        f"| Maximum normalized-weight closure error | {validation['max_normalized_weight_sum_error']:.3e} |",
        f"| Admitted raw coverage range | {validation['min_admitted_raw_coverage']:.6f}–{validation['max_admitted_raw_coverage']:.6f} |",
        "",
        "## Method",
        "",
        "All polygon repair, buffers, hulls, areas, and intersections are computed in a city-specific UTM CRS. Anjuke GCJ-02 vertices are converted to WGS84 before projection. The primary hierarchy is Beijing independent AOI, Anjuke polygon, robust transaction-point hull, city-calibrated buffer, and finally a 250 m default buffer.",
        "",
        "Allocation uses intersection area. `community_area_share_raw` records actual grid coverage. `community_area_share` is normalized only for AOIs with 99.5%–100.5% coverage; partial or uncovered AOIs are retained for audit but are not admitted.",
        "",
        "## Primary coverage",
        "",
        f"Primary AOIs published for {int(primary['aoi_rows'].sum()) if len(primary) else 0:,} city-community records. City-level source, quality, fallback, and coverage statistics are in the QA CSV.",
        "",
        "Buffer sensitivity variants (150 m, 250 m, and 400 m) are generated only where an observed boundary was unavailable. Beijing Anjuke polygons matched alongside an independent AOI are retained as `anjuke_validation`, not mixed with the primary geometry.",
    ]
    excluded = validation.get("unadmitted_primary_community_ids", [])
    if excluded:
        lines.extend(
            [
                "",
                "## Excluded primary AOIs",
                "",
                "The following primary AOIs are valid geometries but lie outside the retained reference-grid cells and therefore receive no normalized weight:",
                "",
                *[f"- `{community_id}`" for community_id in excluded],
            ]
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="*", default=list(ACTIVE_CITIES))
    return parser.parse_args()


def main():
    args = parse_args()
    cities = [city for city in args.cities if city in ACTIVE_CITIES]
    if not cities:
        raise SystemExit("No valid research cities selected")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    registry = pd.read_parquet(REGISTRY_PATH)
    crosswalk = pd.read_parquet(CROSSWALK_PATH)
    registry = registry[registry["city_key"].isin(cities)].copy()
    beijing_aoi = load_beijing_aoi() if "beijing" in cities else {}

    aoi_writer = AtomicParquetWriter(AOI_PATH, AOI_COLUMNS)
    bridge_writer = AtomicParquetWriter(BRIDGE_PATH, BRIDGE_COLUMNS)
    city_rows = []
    run_details = {}

    try:
        for city_key in cities:
            city_registry = registry[registry["city_key"] == city_key].copy()
            if city_registry.empty:
                continue
            lon0 = float(pd.to_numeric(city_registry["centroid_lon"], errors="coerce").median())
            lat0 = float(pd.to_numeric(city_registry["centroid_lat"], errors="coerce").median())
            if not (np.isfinite(lon0) and np.isfinite(lat0)):
                raise RuntimeError(f"Cannot determine projected CRS for {city_key}")
            epsg = utm_epsg(lon0, lat0)
            projected_crs = f"EPSG:{epsg}"
            forward = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
            inverse = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)
            grid_path, grid_frame, grid_geometries, grid_areas = load_grids(city_key, forward)
            aois, metadata = build_city_aois(
                city_key, city_registry, crosswalk, beijing_aoi, forward, inverse, projected_crs
            )
            bridge, bridge_metadata = build_city_bridge(
                city_key, aois, grid_frame, grid_geometries, grid_areas, inverse
            )
            aoi_writer.write([{key: row.get(key) for key in AOI_COLUMNS} for row in aois])
            bridge_writer.write(bridge)

            aoi_frame = pd.DataFrame(
                [
                    {key: row.get(key) for key in AOI_COLUMNS if key != "geometry_wkb"}
                    for row in aois
                ]
            )
            for variant, group in aoi_frame.groupby("aoi_variant", sort=True):
                city_rows.append(
                    {
                        "city_key": city_key,
                        "aoi_variant": variant,
                        "registry_communities": int(len(city_registry)),
                        "aoi_rows": int(len(group)),
                        "admitted_aoi_rows": int(group["bridge_admitted"].sum()),
                        "unadmitted_aoi_rows": int((~group["bridge_admitted"]).sum()),
                        "median_grid_coverage": float(group["grid_coverage_share"].median()),
                        "a_quality_rows": int((group["aoi_quality"] == "A").sum()),
                        "b_quality_rows": int((group["aoi_quality"] == "B").sum()),
                        "c_quality_rows": int((group["aoi_quality"] == "C").sum()),
                        "d_quality_rows": int((group["aoi_quality"] == "D").sum()),
                        "calibrated_radius_m": metadata["calibrated_radius_m"],
                        "bridge_rows": int(
                            sum(1 for row in bridge if row["aoi_variant"] == variant)
                        ),
                    }
                )
            run_details[city_key] = {
                **metadata,
                **bridge_metadata,
                "grid_path": str(grid_path.relative_to(ROOT)),
            }
            print(
                f"{city_key}: registry={len(city_registry):,}, aois={len(aois):,}, "
                f"bridge={len(bridge):,}, admitted={bridge_metadata['admitted_aois']:,}",
                flush=True,
            )
    finally:
        aoi_writer.close()
        bridge_writer.close()

    city_qa = pd.DataFrame(city_rows).sort_values(["city_key", "aoi_variant"])
    city_qa.to_csv(CITY_QA_PATH, index=False, encoding="utf-8-sig")
    validation = validate_outputs(AOI_PATH, BRIDGE_PATH)
    update_registry_aoi_status(REGISTRY_PATH, AOI_PATH)
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_files_modified": False,
        "cities": cities,
        "geometry_hierarchy": [
            "beijing_independent",
            "anjuke",
            "transaction_hull",
            "city_calibrated_buffer",
            "default_buffer",
        ],
        "grid_coverage_admission": [MIN_GRID_COVERAGE, MAX_GRID_COVERAGE],
        "sensitivity_radii_m": list(SENSITIVITY_RADII_M),
        "validation": validation,
        "city_details": run_details,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, city_qa)
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
