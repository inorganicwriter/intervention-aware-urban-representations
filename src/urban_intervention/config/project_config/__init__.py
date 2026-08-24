"""Modular, behavior-compatible project configuration implementation."""

from urban_intervention.data.paths import (
    BOUNDARY_DIR,
    DATA_ROOT,
    HPI_LABEL_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    REFERENCE_GRID_DIR,
    STAGING_DIR,
    TREATMENT_DIR,
)

from .boundaries import (
    HAS_SHAPELY,
    _load_boundary_geojson,
    clip_grids_to_boundary,
    get_admin_boundary,
    get_effective_bbox,
    shapely_affine_scale,
)
from .filesystem import BASE_DIR, DATA_DIR, GRID_DIR, city_dir, ensure_dirs
from .network import detect_proxy, get_proxies, get_proxy, set_proxy
from .pipeline import PIPELINE_CONFIG
from .registry import ACTIVE_CITIES, CITIES, METRO_REFERENCE, CityConfig, get_city_config
from .stations import norm_station_name

__all__ = [
    "ACTIVE_CITIES",
    "BASE_DIR",
    "BOUNDARY_DIR",
    "CITIES",
    "CityConfig",
    "DATA_DIR",
    "DATA_ROOT",
    "GRID_DIR",
    "HAS_SHAPELY",
    "HPI_LABEL_DIR",
    "METRO_REFERENCE",
    "OUTPUT_DIR",
    "PIPELINE_CONFIG",
    "PROJECT_ROOT",
    "RAW_DIR",
    "REFERENCE_GRID_DIR",
    "STAGING_DIR",
    "TREATMENT_DIR",
    "_load_boundary_geojson",
    "city_dir",
    "clip_grids_to_boundary",
    "detect_proxy",
    "ensure_dirs",
    "get_admin_boundary",
    "get_city_config",
    "get_effective_bbox",
    "get_proxies",
    "get_proxy",
    "norm_station_name",
    "set_proxy",
    "shapely_affine_scale",
]
