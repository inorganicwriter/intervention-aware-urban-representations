import tomllib
from pathlib import Path

import yaml

from urban_intervention.config.project import ACTIVE_CITIES, GRID_DIR
from urban_intervention.data.paths import (
    ACTIVE_DIR,
    ARCHIVE_DIR,
    DATA_ROOT,
    LABEL_ROOT,
    PROJECT_ROOT,
)
from urban_intervention.data.registry import iter_datasets, missing_concrete_paths


def test_project_root_and_canonical_layers_exist():
    assert Path(__file__).resolve().parents[2] == PROJECT_ROOT
    for layer in ("catalog", "reference", "curated", "labels", "panels", "causal"):
        assert (ACTIVE_DIR / layer).is_dir()
    for layer in ("raw", "staging"):
        assert (ARCHIVE_DIR / layer).is_dir()


def test_legacy_data_views_are_absent():
    for name in ("grids", "processed", "raw_housing", "external", "labels_canonical"):
        assert not (DATA_ROOT / name).exists()


def test_registry_concrete_paths_exist():
    datasets = list(iter_datasets())
    assert len(datasets) >= 14, "dataset registry should be populated"
    assert missing_concrete_paths() == []


def test_city_and_grid_contract():
    assert len(ACTIVE_CITIES) == 44
    assert GRID_DIR.is_dir()
    assert LABEL_ROOT == ACTIVE_DIR / "labels" / "housing"


def test_declared_development_environment_can_collect_full_test_suite():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = "\n".join(pyproject["project"]["optional-dependencies"]["dev"]).lower()
    for dependency in ("pytest", "openpyxl", "xlrd", "rdata", "pyproj"):
        assert dependency in dev

    environment = yaml.safe_load((PROJECT_ROOT / "environment.yml").read_text(encoding="utf-8"))
    dependencies = [value for value in environment["dependencies"] if isinstance(value, str)]
    assert "python=3.11" in dependencies
    assert any(value.startswith("openpyxl") for value in dependencies)
