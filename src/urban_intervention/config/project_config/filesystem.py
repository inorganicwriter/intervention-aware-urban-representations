"""Project filesystem aliases and directory initialization."""

from pathlib import Path

from urban_intervention.data.paths import (
    DATA_ROOT,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    REFERENCE_GRID_DIR,
    TREATMENT_DIR,
)

from .registry import ACTIVE_CITIES, CITIES

BASE_DIR = PROJECT_ROOT

DATA_DIR = DATA_ROOT

GRID_DIR = REFERENCE_GRID_DIR


def city_dir(city_key: str, base: Path = GRID_DIR) -> Path:
    """Return a city sub-directory below a canonical dataset root."""
    return base / city_key


def ensure_dirs():
    for d in [DATA_DIR, GRID_DIR, TREATMENT_DIR, RAW_DIR, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for ck in ACTIVE_CITIES:
        if ck not in CITIES:
            continue
        for base in [GRID_DIR, RAW_DIR, TREATMENT_DIR]:
            city_dir(ck, base).mkdir(parents=True, exist_ok=True)
