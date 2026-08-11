"""
Lightweight Grid Builder (No geopandas dependency)
Uses shapely + pyproj for grid generation. Output as GeoJSON via WKT.
"""

import json
import math
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Polygon

try:
    from pyproj import Transformer

    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

# Reuse the canonical GRID_DIR from pipeline_config to avoid drift.
from urban_intervention.config.project import GRID_DIR as _PC_GRID_DIR

GRID_DIR = _PC_GRID_DIR


def wgs84_to_meters(lat: float) -> tuple[float, float]:
    """Approximate meters per degree at a given latitude.

    Returns (m_per_deg_lon, m_per_deg_lat).  Only ``lat`` is needed — the
    longitude-dependent term comes from ``cos(lat)``.

    Previously the signature accepted a ``lon`` argument that was never
    used; callers passed ``0`` as a placeholder.  The signature is now
    honest about its inputs.
    """
    lat_rad = math.radians(lat)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
    return m_per_deg_lon, m_per_deg_lat


def _arange_with_end(start: float, stop: float, step: float) -> np.ndarray:
    """Like np.arange but guarantees the result covers ``stop``.

    np.arange with a float step can drop the last bin due to floating-point
    accumulation error.  We compute the bin count via ceil and build the
    array with ``start + i * step`` so the final bin always straddles or
    exceeds ``stop``.
    """
    if step <= 0 or stop <= start:
        return np.array([start])
    n = max(1, int(math.ceil((stop - start) / step)))
    coords = start + np.arange(n) * step
    return coords


def generate_grids_simple(bbox, cell_size_m=500):
    """
    Generate regular grid cells covering a bounding box.
    Uses per-row latitude-dependent meter conversion for better accuracy.

    Args:
        bbox: [lon_min, lat_min, lon_max, lat_max]
        cell_size_m: grid cell size in meters

    Returns:
        List of dicts with grid_id, lon_min, lat_min, lon_max, lat_max, centroid_lon, centroid_lat
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    # Use minimum latitude for conservative lon-step sizing (wider at equator, narrower at poles)
    m_per_lon_min, m_per_lat_min = wgs84_to_meters(lat_min)
    cell_lon = cell_size_m / m_per_lon_min

    # Lat step: use mid-latitude (less variation in meridian direction)
    mid_lat = (lat_min + lat_max) / 2
    _, m_per_lat = wgs84_to_meters(mid_lat)
    cell_lat = cell_size_m / m_per_lat

    # Generate grid — use ceil-based arange so the last bin is not lost to
    # floating-point accumulation error.
    lon_coords = _arange_with_end(lon_min, lon_max, cell_lon)
    lat_coords = _arange_with_end(lat_min, lat_max, cell_lat)

    cells = []
    for i, x0 in enumerate(lon_coords):
        for j, y0 in enumerate(lat_coords):
            # Use the actual row latitude for lon step (more accurate than single mid-point)
            row_lat = y0 + cell_lat / 2
            m_per_lon_row, _ = wgs84_to_meters(row_lat)
            x1 = x0 + cell_size_m / m_per_lon_row
            y1 = y0 + cell_lat
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2

            cells.append(
                {
                    "grid_id": f"g{j:05d}x{i:05d}",
                    "row": j,
                    "col": i,
                    "lon_min": round(x0, 7),
                    "lat_min": round(y0, 7),
                    "lon_max": round(x1, 7),
                    "lat_max": round(y1, 7),
                    "centroid_lon": round(cx, 7),
                    "centroid_lat": round(cy, 7),
                    "area_km2": round(cell_size_m * cell_size_m / 1e6, 4),
                    "geometry_wkt": Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]).wkt,
                }
            )

    return cells


def generate_grids_proj(bbox, cell_size_m=500, src_crs="EPSG:4326", dst_crs="EPSG:32650"):
    """
    Generate regular grid cells using pyproj for precise metric projection.
    """
    if not HAS_PYPROJ:
        print("  pyproj not available, using simple method")
        return generate_grids_simple(bbox, cell_size_m)

    lon_min, lat_min, lon_max, lat_max = bbox
    transformer_to = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    transformer_from = Transformer.from_crs(dst_crs, src_crs, always_xy=True)

    # Project corners
    x_min, y_min = transformer_to.transform(lon_min, lat_min)
    x_max, y_max = transformer_to.transform(lon_max, lat_max)

    cells = []
    x_coords = _arange_with_end(x_min, x_max, cell_size_m)
    y_coords = _arange_with_end(y_min, y_max, cell_size_m)

    for i, x0 in enumerate(x_coords):
        for j, y0 in enumerate(y_coords):
            x1 = x0 + cell_size_m
            y1 = y0 + cell_size_m

            # Project corners back to WGS84
            c00_lon, c00_lat = transformer_from.transform(x0, y0)
            c10_lon, c10_lat = transformer_from.transform(x1, y0)
            c11_lon, c11_lat = transformer_from.transform(x1, y1)
            c01_lon, c01_lat = transformer_from.transform(x0, y1)
            corners = [
                (c00_lon, c00_lat),
                (c10_lon, c10_lat),
                (c11_lon, c11_lat),
                (c01_lon, c01_lat),
            ]

            cx_lon, cx_lat = transformer_from.transform((x0 + x1) / 2, (y0 + y1) / 2)

            # Create polygon in WGS84
            poly = Polygon(corners)

            # Compute bbox from projected corners (may be non-axis-aligned after reprojection)
            all_lons = [c[0] for c in corners]
            all_lats = [c[1] for c in corners]

            cells.append(
                {
                    "grid_id": f"g{j:05d}x{i:05d}",
                    "row": j,
                    "col": i,
                    "lon_min": round(min(all_lons), 7),
                    "lat_min": round(min(all_lats), 7),
                    "lon_max": round(max(all_lons), 7),
                    "lat_max": round(max(all_lats), 7),
                    "centroid_lon": round(cx_lon, 7),
                    "centroid_lat": round(cx_lat, 7),
                    "area_km2": round(cell_size_m * cell_size_m / 1e6, 4),
                    "geometry_wkt": poly.wkt,
                }
            )

    return cells


def grid_df_to_geojson(cells, output_path):
    """Save grid cells as GeoJSON FeatureCollection."""
    features = []
    for cell in cells:
        try:
            geom = wkt.loads(cell["geometry_wkt"])
        except Exception:
            # Skip cells with unparseable WKT rather than emit invalid GeoJSON.
            continue
        if not isinstance(geom, Polygon):
            # Skip non-polygon geometries (lines, points) — they would
            # produce invalid GeoJSON properties.
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "grid_id": cell["grid_id"],
                    "row": cell["row"],
                    "col": cell["col"],
                    "centroid_lon": cell["centroid_lon"],
                    "centroid_lat": cell["centroid_lat"],
                    "area_km2": cell["area_km2"],
                },
                "geometry": geom.__geo_interface__,
            }
        )

    fc = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    return fc


def build_grids_for_city(
    city_config, city_key="", use_proj=True, use_admin_boundary=True, buffer_km=10.0
):
    """Build and save grids. Optionally clips to admin boundary.

    When use_admin_boundary=True and a cached boundary is available:
      - bbox is derived from the boundary polygon + buffer instead of the
        hardcoded center ±0.6° value.
      - Grid cells whose centroids fall outside the buffered polygon are
        dropped, eliminating ocean / mountain / cross-city grids.
    """
    name = city_config["name"]
    slug = city_key if city_key else name.lower().replace(" ", "_")

    # ── Determine bbox ───────────────────────────────────────
    if use_admin_boundary:
        from urban_intervention.config.project import get_effective_bbox

        bbox = get_effective_bbox(city_key, buffer_km=buffer_km)
        print(
            f"  BBox (admin boundary + {buffer_km}km): lon=[{bbox[0]:.2f}, {bbox[2]:.2f}], lat=[{bbox[1]:.2f}, {bbox[3]:.2f}]"
        )
    else:
        bbox = city_config["bbox"]
        print(
            f"  BBox (hardcoded): lon=[{bbox[0]:.2f}, {bbox[2]:.2f}], lat=[{bbox[1]:.2f}, {bbox[3]:.2f}]"
        )

    if use_proj and HAS_PYPROJ:
        dst = city_config.get("projected_crs", "EPSG:32650")
        cells = generate_grids_proj(bbox, cell_size_m=500, dst_crs=dst)
        print(f"  Using pyproj projection ({dst})")
    else:
        cells = generate_grids_simple(bbox, cell_size_m=500)
        print("  Using simple lat/lon conversion")
    print(f"  Generated {len(cells)} grid cells")

    # ── Clip to admin boundary ────────────────────────────────
    if use_admin_boundary:
        from urban_intervention.config.project import clip_grids_to_boundary, get_admin_boundary

        boundary = get_admin_boundary(city_key)
        if boundary is not None:
            cells_before = len(cells)
            cells = clip_grids_to_boundary(cells, city_key, buffer_km=buffer_km)
            dropped = cells_before - len(cells)
            if dropped > 0:
                pct = dropped / cells_before * 100
                print(
                    f"  Clipped to admin boundary: {len(cells)}/{cells_before} kept "
                    f"({dropped} dropped, {pct:.0f}%)"
                )

    # Save as parquet (DataFrame)
    df = pd.DataFrame(cells)
    city_dir = GRID_DIR / slug
    city_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = city_dir / f"{slug}_grids.parquet"
    df.to_parquet(parquet_path)
    print(f"  Saved parquet: {parquet_path}")

    # Save as GeoJSON
    geojson_path = city_dir / f"{slug}_grids.geojson"
    grid_df_to_geojson(cells, str(geojson_path))
    print(f"  Saved GeoJSON: {geojson_path}")

    # Quick stats
    n_cols = df["col"].nunique()
    n_rows = df["row"].nunique()
    print(f"  Grid dimensions: {n_cols} cols x {n_rows} rows")

    return df


def generate_station_grids(
    stations: pd.DataFrame, half_side_km: float = 3.0, cell_size_m: int = 500
) -> list[dict]:
    """Generate 500m grid cells covering all stations with a buffer.

    Rather than generating per-station patches (which produce overlapping
    cells when stations are close together), this computes a single
    bounding box from the station coordinates + ``half_side_km`` in each
    direction and generates a unified grid over it — exactly like the
    admin-boundary grid but with a tighter, station-driven bbox.

    Parameters
    ----------
    stations : pd.DataFrame
        Must contain ``wgs84_lon`` and ``wgs84_lat`` columns.
    half_side_km : float
        Buffer beyond the station extent in each direction (km).
    cell_size_m : int
        Cell side length in metres.

    Returns
    -------
    list[dict]
        Grid cells with the same schema as ``generate_grids_simple``.
    """
    coords = stations[["wgs84_lon", "wgs84_lat"]].dropna().values
    if len(coords) == 0:
        return []

    # Compute station extent + half_side_km deg buffer (approximate)
    lon_min, lon_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    lat_min, lat_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    mid_lat = (lat_min + lat_max) / 2.0
    m_per_lon, _ = wgs84_to_meters(mid_lat)
    buf_lon = half_side_km * 1000.0 / m_per_lon
    buf_lat = half_side_km * 1000.0 / 111000.0

    bbox = [lon_min - buf_lon, lat_min - buf_lat, lon_max + buf_lon, lat_max + buf_lat]
    return generate_grids_simple(bbox, cell_size_m=cell_size_m)


def build_station_grids_for_city(city_key: str, half_side_km: float = 3.0):
    """Build and save station-centred grids for one city.

    Reads the *merged* station CSV (produced by compare_transit_sources.py)
    so that each city's grid is derived from the best available station set.
    If the merged file doesn't exist, falls back to any available source CSV.
    """
    from urban_intervention.config.project import CITIES

    cfg = CITIES[city_key]
    name = cfg["name"]
    transit_dir = BASE_DIR / "data" / "archive" / "raw" / "transit" / city_key

    # Prefer merged, then try any source with coordinates
    stations = None
    for src_tag in ("merged", "amap", "osm", "wikidata"):
        csv_path = transit_dir / f"{city_key}_metro_stations_{src_tag}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
                if "wgs84_lon" in df.columns and "wgs84_lat" in df.columns:
                    stations = df[["wgs84_lon", "wgs84_lat"]].dropna()
                    print(f"  Using {src_tag} source: {len(stations)} stations")
                    break
            except Exception:
                continue

    if stations is None or stations.empty:
        print(f"  [SKIP] No station coordinates found for {name}")
        return None

    cells = generate_station_grids(stations, half_side_km=half_side_km)
    df = pd.DataFrame(cells)
    city_dir = GRID_DIR / city_key
    city_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = city_dir / f"{city_key}_grids_station.parquet"
    df.to_parquet(parquet_path)
    print(f"  Saved parquet: {parquet_path}")

    geojson_path = city_dir / f"{city_key}_grids_station.geojson"
    grid_df_to_geojson(cells, str(geojson_path))
    print(f"  Saved GeoJSON: {geojson_path}")

    print(
        f"  Station-grid: {len(cells)} cells around {len(stations)} stations "
        f"(half_side={half_side_km}km)"
    )
    return df


def load_grids(city_key="beijing"):
    """Load saved grids for any city by key."""
    path = GRID_DIR / city_key / f"{city_key}_grids.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


if __name__ == "__main__":
    from urban_intervention.config.project import ACTIVE_CITIES, CITIES

    failed = []
    for ck in ACTIVE_CITIES:
        if ck not in CITIES:
            continue
        try:
            city = CITIES[ck]
            print(f"\n{'=' * 50}\n{ck.upper()} — Admin-boundary grid\n{'=' * 50}")
            df = build_grids_for_city(city, ck)
            print(f"  Bbox-grid: {len(df)} cells")

            print(f"\n{ck.upper()} — Station-centred grid")
            df_st = build_station_grids_for_city(ck)
        except Exception as e:
            print(f"\n[ERROR] {ck}: {e}")
            failed.append(ck)
    if failed:
        print(f"\nFailed cities: {failed}")
