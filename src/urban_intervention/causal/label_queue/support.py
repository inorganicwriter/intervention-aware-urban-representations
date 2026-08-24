"""Behavior-preserving component of the modular causal label queue."""

from __future__ import annotations

import subprocess
import sys
from functools import partial

import pandas as pd
import pyarrow.parquet as pq

from urban_intervention.utils import atomic_write_json

from .runtime import (
    DONOR_UNIVERSE,
    OUTCOMES,
    PANEL_HOUSING_MONTHLY_DIR,
    POI_DIR,
    POPULATION_DIR,
    VIIRS_RAW,
    collection_script,
)
from .state import run, task_directory

atomic_json = partial(atomic_write_json, default=str)

_FAMILY_SUPPORT_CACHE: dict[tuple[str, str], set[str]] = {}


def family_signature(row: pd.Series, support: pd.DataFrame) -> str:
    target = support.loc[support["treatment_order"] == int(row["treatment_order"])]
    if len(target) != 1:
        raise ValueError("Treatment support row is not unique")
    names = [
        family
        for family in OUTCOMES
        if pd.notna(target.iloc[0][f"{family}_complete"])
        and bool(target.iloc[0][f"{family}_complete"])
    ]
    return "+".join(sorted(names))


def _family_observed_grids(city: str, family: str) -> set[str]:
    key = (city, family)
    if key in _FAMILY_SUPPORT_CACHE:
        return _FAMILY_SUPPORT_CACHE[key]
    grids: set[str] = set()
    if family == "housing":
        path = PANEL_HOUSING_MONTHLY_DIR / f"{city}.parquet"
        if path.is_file():
            frame = pq.read_table(path, columns=["grid_id", "log_price_raw_median"]).to_pandas()
            grids = set(frame.loc[frame["log_price_raw_median"].notna(), "grid_id"].astype(str))
    elif family in {"poi", "population"}:
        directory = POI_DIR if family == "poi" else POPULATION_DIR
        candidates = sorted(directory.glob(f"{city}*"))
        if candidates:
            frame = pd.read_parquet(candidates[0])
            value_columns = [
                column for column in frame.columns if column not in {"city_key", "grid_id", "year"}
            ]
            if value_columns:
                mask = frame[value_columns].notna().any(axis=1)
                grids = set(frame.loc[mask, "grid_id"].astype(str))
    _FAMILY_SUPPORT_CACHE[key] = grids
    return grids


def family_has_observed_support(row: pd.Series) -> bool:
    """True when the grid appears in the family panel with any observation.

    VIIRS is not pre-screened: the monthly partitions cover every grid, so
    the check would never fire there.  For the other families this turns
    no-data tasks into an instant skip instead of a ~3-minute GSC/MC run
    that must fail.
    """
    family = str(row.outcome_family)
    if family == "viirs":
        return True
    return str(row.grid_id) in _family_observed_grids(str(row.city_key), family)


def ensure_viirs(
    row: pd.Series, require_full_matching_window: bool = True, city_key: str | None = None
) -> subprocess.CompletedProcess[str]:
    requested_city = city_key or str(row.city_key)
    opening = pd.Period(str(row.opening_month), freq="M")
    requested_start = opening - 42
    start = (
        requested_start
        if require_full_matching_window
        else max(requested_start, pd.Period("2012-01", freq="M"))
    )
    end = opening + 24
    command = [
        sys.executable,
        str(collection_script("ensure_viirs_monthly_cache.py")),
        "--city",
        requested_city,
        "--start",
        str(start),
        "--end",
        str(end),
        "--manifest",
        str(
            task_directory(int(row.treatment_order), "viirs") / f"viirs_cache_{requested_city}.json"
        ),
    ]
    if VIIRS_RAW:
        command[2:2] = ["--input-dir", VIIRS_RAW]
    return run(command)


def viirs_has_min_preperiods(opening_month: str) -> bool:
    return pd.Period(str(opening_month), freq="M") >= pd.Period("2012-12", freq="M")


def viirs_has_full_matching_window(opening_month: str) -> bool:
    return pd.Period(str(opening_month), freq="M") >= pd.Period("2015-07", freq="M")


def ensure_cross_city_viirs(row: pd.Series) -> tuple[bool, list[str]]:
    cities = sorted(
        set(pq.read_table(DONOR_UNIVERSE, columns=["city_key"]).column("city_key").to_pylist())
    )
    logs: list[str] = []
    for city in cities:
        completed = ensure_viirs(row, require_full_matching_window=False, city_key=city)
        logs.append(completed.stdout)
        if completed.returncode != 0:
            return False, logs
    return True, logs
