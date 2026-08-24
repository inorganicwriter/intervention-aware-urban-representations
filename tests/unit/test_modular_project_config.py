from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import pytest

from urban_intervention.config import project as original
from urban_intervention.config import project_modular as modular
from urban_intervention.config.project_config import boundaries as modular_boundaries
from urban_intervention.config.project_config import filesystem as modular_filesystem
from urban_intervention.config.project_config import network as modular_network

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_PATH = ROOT / "src" / "urban_intervention" / "config" / "project.py"
ORIGINAL_NORMALIZED_SHA256 = "033ca82dfa22047dcb102b329723c0dce4c4f204e0d6fe1a43f639e6cb1b814e"

FUNCTION_NAMES = (
    "shapely_affine_scale",
    "_load_boundary_geojson",
    "get_admin_boundary",
    "get_effective_bbox",
    "clip_grids_to_boundary",
    "get_city_config",
    "city_dir",
    "ensure_dirs",
    "norm_station_name",
    "detect_proxy",
    "get_proxy",
    "get_proxies",
    "set_proxy",
)

USED_PUBLIC_NAMES = (
    "ACTIVE_CITIES",
    "CITIES",
    "GRID_DIR",
    "METRO_REFERENCE",
    "TREATMENT_DIR",
    "city_dir",
    "clip_grids_to_boundary",
    "get_admin_boundary",
    "get_city_config",
    "get_effective_bbox",
    "get_proxies",
    "get_proxy",
    "norm_station_name",
    "set_proxy",
)


def _function_dump(function) -> str:
    node = ast.parse(inspect.getsource(function)).body[0]
    assert isinstance(node, ast.FunctionDef)
    return ast.dump(node, include_attributes=False)


def _normalized_signature(function) -> tuple:
    signature = inspect.signature(function)

    def annotation_identity(annotation):
        return getattr(annotation, "__name__", annotation)

    parameters = tuple(
        (
            parameter.name,
            parameter.kind,
            parameter.default,
            annotation_identity(parameter.annotation),
        )
        for parameter in signature.parameters.values()
    )
    return parameters, annotation_identity(signature.return_annotation)


def test_frozen_project_file_is_unchanged() -> None:
    normalized = ORIGINAL_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert hashlib.sha256(normalized.encode("utf-8")).hexdigest() == (
        ORIGINAL_NORMALIZED_SHA256
    )


def test_modular_facade_preserves_used_public_api() -> None:
    for name in USED_PUBLIC_NAMES:
        assert hasattr(original, name), name
        assert hasattr(modular, name), name


def test_registry_pipeline_and_paths_are_equal() -> None:
    assert modular.CITIES == original.CITIES
    assert modular.ACTIVE_CITIES == original.ACTIVE_CITIES
    assert modular.METRO_REFERENCE == original.METRO_REFERENCE
    assert modular.PIPELINE_CONFIG == original.PIPELINE_CONFIG
    assert modular.PIPELINE_CONFIG["cities"] is modular.CITIES
    assert modular.PIPELINE_CONFIG["active_cities"] is modular.ACTIVE_CITIES
    assert modular.CityConfig.__annotations__ == original.CityConfig.__annotations__
    for name in (
        "BASE_DIR",
        "DATA_DIR",
        "GRID_DIR",
        "BOUNDARY_DIR",
        "DATA_ROOT",
        "HPI_LABEL_DIR",
        "OUTPUT_DIR",
        "PROJECT_ROOT",
        "RAW_DIR",
        "REFERENCE_GRID_DIR",
        "STAGING_DIR",
        "TREATMENT_DIR",
    ):
        assert getattr(modular, name) == getattr(original, name), name


@pytest.mark.parametrize("name", FUNCTION_NAMES)
def test_function_signature_and_ast_match(name: str) -> None:
    original_function = getattr(original, name)
    modular_function = getattr(modular, name)
    assert _normalized_signature(modular_function) == _normalized_signature(original_function)
    assert _function_dump(modular_function) == _function_dump(original_function)


def test_all_city_configs_and_effective_bboxes_match() -> None:
    assert len(modular.ACTIVE_CITIES) == 44
    for city in modular.ACTIVE_CITIES:
        assert modular.get_city_config(city) == original.get_city_config(city)
        assert modular.get_effective_bbox(city) == original.get_effective_bbox(city)


def test_boundary_fallback_behavior_matches(monkeypatch) -> None:
    grids = [{"centroid_lon": 116.4, "centroid_lat": 39.9}]
    monkeypatch.setattr(original, "get_admin_boundary", lambda _city: None)
    monkeypatch.setattr(modular_boundaries, "get_admin_boundary", lambda _city: None)
    assert modular.get_effective_bbox("beijing") == original.get_effective_bbox("beijing")
    assert modular.clip_grids_to_boundary(grids, "beijing") == original.clip_grids_to_boundary(
        grids, "beijing"
    )


def test_station_normalization_matches() -> None:
    values = ["建国路站", "西二旗（地铁）", "海淀黄庄·换乘", "五路居", None]
    assert [modular.norm_station_name(value) for value in values] == [
        original.norm_station_name(value) for value in values
    ]


def test_proxy_state_and_override_match(monkeypatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:18080")
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("MIT_AUTO_PROXY_PORT", raising=False)
    monkeypatch.setattr(original, "_PROXY_CACHE", None)
    monkeypatch.setattr(original, "_PROXY_DETECTED", False)
    monkeypatch.setattr(modular_network, "_PROXY_CACHE", None)
    monkeypatch.setattr(modular_network, "_PROXY_DETECTED", False)

    assert modular.detect_proxy() == original.detect_proxy()
    assert modular.get_proxy() == original.get_proxy()
    assert modular.get_proxies() == original.get_proxies()

    original.set_proxy("http://127.0.0.1:19090")
    modular.set_proxy("http://127.0.0.1:19090")
    assert modular.get_proxies() == original.get_proxies()


def test_directory_initialization_matches_in_isolated_roots(tmp_path, monkeypatch) -> None:
    original_root = tmp_path / "original"
    modular_root = tmp_path / "modular"
    original_paths = [original_root / name for name in ("data", "grids", "treatments", "raw", "out")]
    modular_paths = [modular_root / name for name in ("data", "grids", "treatments", "raw", "out")]
    for module, paths in ((original, original_paths), (modular_filesystem, modular_paths)):
        monkeypatch.setattr(module, "DATA_DIR", paths[0])
        monkeypatch.setattr(module, "GRID_DIR", paths[1])
        monkeypatch.setattr(module, "TREATMENT_DIR", paths[2])
        monkeypatch.setattr(module, "RAW_DIR", paths[3])
        monkeypatch.setattr(module, "OUTPUT_DIR", paths[4])

    original.ensure_dirs()
    modular.ensure_dirs()
    original_relative = sorted(path.relative_to(original_root) for path in original_root.rglob("*"))
    modular_relative = sorted(path.relative_to(modular_root) for path in modular_root.rglob("*"))
    assert modular_relative == original_relative
