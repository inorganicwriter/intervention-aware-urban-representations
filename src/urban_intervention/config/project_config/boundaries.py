"""Administrative-boundary loading, buffering, and grid clipping."""

import json
import math
import warnings

from urban_intervention.data.paths import BOUNDARY_DIR

from .registry import get_city_config

try:
    from shapely import affinity as _shapely_affinity
    from shapely.geometry import MultiPolygon, Point, Polygon, shape  # noqa: F401
    from shapely.ops import unary_union as _shapely_unary_union

    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


def shapely_affine_scale(geom, xfact: float, yfact: float):
    """Wrapper around shapely.affinity.scale that degrades gracefully when
    shapely is unavailable (returns the original geom).

    Uses ``origin=(0, 0)`` so the scaling is a pure coordinate
    transformation (no translation) — required for the latitude-corrected
    buffer in :func:`clip_grids_to_boundary`.
    """
    if not HAS_SHAPELY:
        return geom
    return _shapely_affinity.scale(geom, xfact=xfact, yfact=yfact, origin=(0, 0))


def _load_boundary_geojson(city_key: str) -> dict | None:
    """Load cached admin boundary GeoJSON, or None if unavailable."""
    p = BOUNDARY_DIR / f"{city_key}_boundary.geojson"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def get_admin_boundary(city_key: str):
    """Return shapely Polygon/MultiPolygon for a city, or None."""
    if not HAS_SHAPELY:
        return None
    geojson = _load_boundary_geojson(city_key)
    if geojson is None:
        return None
    try:
        geojson_type = geojson.get("type")
        if geojson_type == "FeatureCollection":
            geometries = [
                shape(feature["geometry"])
                for feature in geojson.get("features", [])
                if feature.get("geometry") is not None
            ]
            if not geometries:
                raise ValueError("FeatureCollection contains no geometries")
            return _shapely_unary_union(geometries)
        if geojson_type == "Feature":
            return shape(geojson["geometry"])
        return shape(geojson)
    except Exception as exc:
        warnings.warn(
            f"Invalid boundary GeoJSON for {city_key}: {exc}; using bbox fallback",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def get_effective_bbox(city_key: str, buffer_km: float = 10.0) -> list[float]:
    """Return bbox from cached admin boundary + buffer, or hardcoded fallback.

    Hardcoded bbox is center ±0.6°; admin-derived bbox is boundary.extent
    + buffer in all directions.  This is the single source of truth for
    grid generation, GEE extraction, and transit queries.

    Buffer conversion accounts for latitude: longitude degrees are scaled
    by ``cos(lat_c)`` so the buffer is approximately square in meters.
    """
    cfg = get_city_config(city_key)
    boundary = get_admin_boundary(city_key)
    if boundary is None:
        return list(cfg["bbox"])

    minx, miny, maxx, maxy = boundary.bounds
    lat_c = (miny + maxy) / 2.0
    buf_lat_deg = buffer_km / 111.0
    buf_lon_deg = buffer_km / (111.0 * max(0.1, math.cos(math.radians(lat_c))))
    return [
        round(minx - buf_lon_deg, 4),
        round(miny - buf_lat_deg, 4),
        round(maxx + buf_lon_deg, 4),
        round(maxy + buf_lat_deg, 4),
    ]


def clip_grids_to_boundary(grids: list[dict], city_key: str, buffer_km: float = 10.0) -> list[dict]:
    """Filter grid cells to those intersecting the (buffered) admin boundary.

    When no boundary is cached, returns grids unchanged.  When the boundary
    is available, keeps only cells whose centroid lies within the buffered
    polygon.

    Buffer conversion accounts for latitude via ``cos(lat_c)``.
    """
    if not HAS_SHAPELY:
        return grids

    boundary = get_admin_boundary(city_key)
    if boundary is None:
        return grids

    # Buffer the boundary outward by buffer_km with latitude-corrected degrees
    minx, miny, maxx, maxy = boundary.bounds
    lat_c = (miny + maxy) / 2.0
    buf_lat_deg = buffer_km / 111.0
    cos_lat = max(0.1, math.cos(math.radians(lat_c)))
    # shapely.buffer is isotropic in the units of the geometry (degrees here),
    # so we approximate a latitude-corrected buffer by transforming to a
    # pseudo-isotropic space, buffering, then transforming back:
    #   1. COMPRESS x by cos(lat)  — now 1 deg-x ≈ 1 deg-y in meters
    #   2. Buffer isotropically by buf_lat_deg
    #   3. STRETCH x by 1/cos(lat)  — back to original coordinate system
    # The x-direction buffer in original space = buf_lat_deg / cos(lat),
    # which equals the desired buf_lon_deg.  (The previous implementation
    # had the scale factors reversed, making the x-buffer too small by
    # a factor of cos²(lat).)
    scaled = shapely_affine_scale(boundary, xfact=cos_lat, yfact=1.0)
    buffered_scaled = scaled.buffer(buf_lat_deg)
    buffered = shapely_affine_scale(buffered_scaled, xfact=1.0 / cos_lat, yfact=1.0)

    kept = []
    for cell in grids:
        pt = Point(cell["centroid_lon"], cell["centroid_lat"])
        if buffered.contains(pt) or buffered.touches(pt):
            kept.append(cell)
    return kept
