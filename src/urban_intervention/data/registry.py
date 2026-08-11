"""Machine-readable dataset registry access and validation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .paths import CATALOG_DIR, PROJECT_ROOT

DEFAULT_REGISTRY = CATALOG_DIR / "datasets.yaml"


@dataclass(frozen=True)
class RegisteredDataset:
    name: str
    specification: dict[str, Any]

    @property
    def path_template(self) -> str | None:
        value = self.specification.get("path")
        return str(value) if value is not None else None

    @property
    def concrete_path(self) -> Path | None:
        template = self.path_template
        if not template or "{" in template:
            return None
        return PROJECT_ROOT / template


def load_registry(path: Path | None = None) -> dict[str, Any]:
    registry_path = path or DEFAULT_REGISTRY
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Dataset registry not found: {registry_path}\n"
            f"Create it from the template or set a custom path."
        )
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise ValueError(f"Invalid dataset registry: {registry_path}")
    return payload


def iter_datasets(path: Path | None = None) -> Iterator[RegisteredDataset]:
    for name, specification in load_registry(path)["datasets"].items():
        yield RegisteredDataset(name=name, specification=specification)


def get_dataset(name: str, path: Path | None = None) -> RegisteredDataset:
    datasets = load_registry(path)["datasets"]
    try:
        specification = datasets[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset {name!r}; available: {sorted(datasets)}") from exc
    return RegisteredDataset(name=name, specification=specification)


def missing_concrete_paths(path: Path | None = None) -> list[RegisteredDataset]:
    return [
        dataset
        for dataset in iter_datasets(path)
        if dataset.concrete_path is not None and not dataset.concrete_path.exists()
    ]
