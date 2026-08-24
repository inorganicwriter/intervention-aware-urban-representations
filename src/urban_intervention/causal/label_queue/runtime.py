"""Shared mutable runtime configuration for the modular label queue.

The original script used module-level globals configured by ``main``.  A
single settings object preserves those semantics across the split modules
without duplicating configuration state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from urban_intervention.data.paths import (
    CONTROL_DESIGN_QUEUE as CONTROL_QUEUE,
)
from urban_intervention.data.paths import (
    COUNTERFACTUAL_QUEUE as UNIT_QUEUE,
)
from urban_intervention.data.paths import (
    ELIGIBLE_DONORS as DONOR_UNIVERSE,
)
from urban_intervention.data.paths import (
    FORMAL_TARGET_SUPPORT as SUPPORT,
)
from urban_intervention.data.paths import (
    OUTCOME_FAMILY_QUEUE as DEFAULT_FAMILY_QUEUE,
)
from urban_intervention.data.paths import (
    OUTPUT_CAUSAL_TASKS_DIR as TASK_ROOT,
)
from urban_intervention.data.paths import (
    OUTPUT_COMPLETE_STAGING_DIR as STAGING,
)
from urban_intervention.data.paths import (
    OUTPUT_CONTROL_TASKS_DIR,
    OUTPUT_FIXED_CONTROL_DIR,
    PANEL_HOUSING_MONTHLY_DIR,
    POI_DIR,
    POPULATION_DIR,
    PROJECT_ROOT,
    R_LIB_DIR,
    TREATMENT_UNIT_LIST,
    collection_script,
    r_script,
)

R_SCRIPT = os.environ.get("MIT_RSCRIPT", "Rscript")
R_LIB = Path(os.environ.get("MIT_R_LIB", str(R_LIB_DIR)))
VIIRS_RAW = os.environ.get("MIT_VIIRS_RAW")
ROOT = PROJECT_ROOT

OUTCOMES = {
    "housing": ["housing_log_price"],
    "viirs": ["viirs_avg_asinh"],
    "population": ["population_log"],
    "poi": [
        "poi_count_log",
        "poi_category_entropy",
        "poi_commercial_share",
        "poi_transport_access_log",
    ],
}
HORIZONS = {
    "housing": [1, 3, 6, 12, 18, 24],
    "viirs": [1, 3, 6, 12, 18, 24],
    "population": [1, 2, 3],
    "poi": [1, 2, 3],
}


@dataclass
class QueueSettings:
    """Process-local settings formerly stored as script globals."""

    anticipation_months: int = 6
    price_measure: str = "median"
    label_window: int = 1
    transaction_count_threshold: int = 1
    run_mode: str = "production"
    estimator_backend: str = "r_reference"
    max_gsc_cross_city_donors: int = 50_000
    gsc_donor_sampling_seed: int = 20260823
    qualification_receipt: Path | None = None
    qualification_proof: dict[str, object] = field(default_factory=dict)
    cross_city_design_cache: dict[int, tuple[pd.Series, dict[str, object]]] = field(
        default_factory=dict
    )
    r_timeout_seconds: int = field(
        default_factory=lambda: int(os.environ.get("MIT_R_TIMEOUT_SECONDS", "7200"))
    )
    family_queue: Path = DEFAULT_FAMILY_QUEUE


settings = QueueSettings()

__all__ = [
    "CONTROL_QUEUE",
    "DONOR_UNIVERSE",
    "HORIZONS",
    "OUTCOMES",
    "OUTPUT_CONTROL_TASKS_DIR",
    "OUTPUT_FIXED_CONTROL_DIR",
    "PANEL_HOUSING_MONTHLY_DIR",
    "POI_DIR",
    "POPULATION_DIR",
    "QueueSettings",
    "R_LIB",
    "R_SCRIPT",
    "ROOT",
    "STAGING",
    "SUPPORT",
    "TASK_ROOT",
    "TREATMENT_UNIT_LIST",
    "UNIT_QUEUE",
    "VIIRS_RAW",
    "collection_script",
    "r_script",
    "settings",
]
