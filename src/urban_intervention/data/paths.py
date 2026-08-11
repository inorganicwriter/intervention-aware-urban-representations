"""Canonical filesystem locations for project datasets.

New code should import paths from this module instead of constructing
``data/...`` strings.
"""

import os
from pathlib import Path

_PACKAGE_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = (
    Path(os.environ.get("MIT_PROJECT_ROOT", str(_PACKAGE_PROJECT_ROOT))).expanduser().resolve()
)
DATA_ROOT = PROJECT_ROOT / "data"

# Two-way data layout: archive/ holds immutable originals (raw, staging);
# active/ holds the working datasets consumed by matching, labelling and
# training (curated, reference, causal, panels, labels, catalog).
ARCHIVE_DIR = DATA_ROOT / "archive"
ACTIVE_DIR = DATA_ROOT / "active"

CATALOG_DIR = ACTIVE_DIR / "catalog"
REFERENCE_DIR = ACTIVE_DIR / "reference"
REFERENCE_GRID_DIR = REFERENCE_DIR / "grids"
BOUNDARY_DIR = REFERENCE_DIR / "boundaries"
TRANSIT_REFERENCE_DIR = REFERENCE_DIR / "transit"
CANONICAL_STATION_EVENTS = TRANSIT_REFERENCE_DIR / "canonical_station_events.parquet"
RESOLVED_STATION_EVENTS = TRANSIT_REFERENCE_DIR / "canonical_station_events_resolved.parquet"
COMPETING_TRANSIT_EVENTS = TRANSIT_REFERENCE_DIR / "competing_transit_events.parquet"
EXCLUDED_STATION_EVENTS = TRANSIT_REFERENCE_DIR / "excluded_station_events.csv"
STATION_RESOLUTION_MANIFEST = TRANSIT_REFERENCE_DIR / "station_resolution_manifest.json"
STATION_ISSUE_RESOLUTION = CATALOG_DIR / "quality" / "station_issue_resolution_v1.csv"

RAW_DIR = ARCHIVE_DIR / "raw"
RAW_HOUSING_DIR = RAW_DIR / "housing"
RAW_PLATFORM_EXPORT_DIR = RAW_HOUSING_DIR / "platform_exports"
RAW_WEB_ARCHIVE_DIR = RAW_HOUSING_DIR / "web_archives"
RAW_OPEN_DATA_DIR = RAW_HOUSING_DIR / "open_data"
RAW_HOUSING_SPATIAL_DIR = RAW_HOUSING_DIR / "spatial_support"
RAW_WAYBACK_DIR = RAW_WEB_ARCHIVE_DIR / "wayback"
RAW_WAYBACK_PARSED_DIR = RAW_WAYBACK_DIR / "parsed_pages"
RAW_WAYBACK_INVENTORY_DIR = RAW_WAYBACK_DIR / "inventories"
RAW_WAYBACK_MANIFEST_DIR = RAW_WAYBACK_DIR / "manifests"
RAW_WAYBACK_CDX_DIR = RAW_WAYBACK_DIR / "cdx_cache"
RAW_LIANJIA_DIR = RAW_PLATFORM_EXPORT_DIR / "lianjia" / "purchased_transactions"
RAW_ANJUKE_DIR = RAW_PLATFORM_EXPORT_DIR / "anjuke" / "cross_section"
RAW_OPEN_DATASET_DIR = RAW_OPEN_DATA_DIR / "datasets"
RAW_OPEN_IMPORT_DIR = RAW_OPEN_DATA_DIR / "import_batches"
RAW_COMMUNITY_AOI_DIR = RAW_HOUSING_SPATIAL_DIR / "community_aoi"
RAW_GRID_PRICE_2023_05_DIR = RAW_HOUSING_SPATIAL_DIR / "grid_price_2023_05"
RAW_TRANSIT_DIR = RAW_DIR / "transit"

STAGING_DIR = ARCHIVE_DIR / "staging"
CURATED_DIR = ACTIVE_DIR / "curated"
TREATMENT_DIR = CURATED_DIR / "treatment"
VIIRS_DIR = CURATED_DIR / "viirs"
SENTINEL2_DIR = CURATED_DIR / "sentinel2"
POPULATION_DIR = CURATED_DIR / "population"
POI_DIR = CURATED_DIR / "poi"
ROAD_NETWORK_DIR = CURATED_DIR / "road_network"

LABEL_ROOT = ACTIVE_DIR / "labels" / "housing"
HPI_LABEL_DIR = LABEL_ROOT / "city_hpi"
ANJUKE_LABEL_DIR = LABEL_ROOT / "listing_price" / "anjuke"
GRID2023_LABEL_DIR = LABEL_ROOT / "listing_price" / "grid_2023"
WAYBACK_LABEL_DIR = LABEL_ROOT / "historical_snapshot" / "wayback"
LIANJIA_LABEL_DIR = LABEL_ROOT / "transaction_price" / "lianjia"

PANEL_ROOT = ACTIVE_DIR / "panels"
PANEL_DIR = PANEL_ROOT / "grid_year" / "v2"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RUNTIME_DIR = PROJECT_ROOT / ".runtime"
BROWSER_PROFILE_DIR = RUNTIME_DIR / "browser_profiles" / "playwright_profile"
GEOCODE_CACHE = STAGING_DIR / "housing" / "geocoding" / "geocode_cache.json"


def grid_path(city: str) -> Path:
    return REFERENCE_GRID_DIR / city / f"{city}_grids.parquet"


def treatment_path(city: str) -> Path:
    return TREATMENT_DIR / city / f"{city}_grid_treatment.parquet"


def hpi_label_path(city: str) -> Path:
    return HPI_LABEL_DIR / city / f"{city}_hpi_yearly.parquet"


def anjuke_label_path(city: str) -> Path:
    return ANJUKE_LABEL_DIR / city / f"{city}_anjuke_grid_price.parquet"


def wayback_label_path(city: str) -> Path:
    return WAYBACK_LABEL_DIR / city / f"{city}_wayback_grid_yearly.parquet"


# ── Causal formal matching and feature-store paths ──────────────────

CAUSAL_DIR = ACTIVE_DIR / "causal"
FORMAL_MATCHING_DIR = CAUSAL_DIR / "formal_matching_inputs"


def housing_annual_path(city: str) -> Path:
    return FORMAL_MATCHING_DIR / "housing_annual" / f"{city}.parquet"


def poi_annual_path(city: str) -> Path:
    return POI_DIR / f"{city}_poi_grid_yearly.parquet"


def population_data_path(city: str) -> Path:
    return POPULATION_DIR / f"{city}_pop.parquet"


def sentinel2_data_path(city: str) -> Path:
    return SENTINEL2_DIR / f"{city}_s2.parquet"


# ── Panel data factory functions ────────────────────────────────────

PANEL_HOUSING_MONTHLY_DIR = PANEL_ROOT / "housing_grid_month"
PANEL_HOUSING_QUARTERLY_DIR = PANEL_ROOT / "housing_grid_quarter"
PANEL_HOUSING_YEARLY_DIR = PANEL_ROOT / "housing_grid_year"


def housing_monthly_panel_path(city: str) -> Path:
    return PANEL_HOUSING_MONTHLY_DIR / f"{city}.parquet"


def housing_quarterly_panel_path(city: str) -> Path:
    return PANEL_HOUSING_QUARTERLY_DIR / f"{city}.parquet"


def housing_yearly_panel_path(city: str) -> Path:
    return PANEL_HOUSING_YEARLY_DIR / f"{city}.parquet"


# ── Causal data constants ───────────────────────────────────────────

TREATMENT_UNIT_LIST = CAUSAL_DIR / "treatment_unit_list.parquet"
CONTROL_DESIGN_QUEUE = CAUSAL_DIR / "control_design_queue.csv"
OUTCOME_FAMILY_QUEUE = CAUSAL_DIR / "outcome_family_work_queue.csv"
COUNTERFACTUAL_QUEUE = CAUSAL_DIR / "counterfactual_work_queue.csv"
CAUSAL_RELEASES_DIR = CAUSAL_DIR / "releases"
GRID_UNIVERSE_DIR = CAUSAL_DIR / "grid_universe"
GRID_UNIVERSE_METADATA = CAUSAL_DIR / "grid_universe_metadata.json"
GRID_UNIVERSE_BY_CITY = CAUSAL_DIR / "grid_universe_by_city.csv"
FEATURE_STORE_DIR = CAUSAL_DIR / "feature_store"
FORMAL_TARGET_SUPPORT = FORMAL_MATCHING_DIR / "formal_target_support.parquet"
ELIGIBLE_DONORS = FORMAL_MATCHING_DIR / "eligible_never_treated_donors.parquet"
TREATMENT_PREREND_AVAILABILITY = CAUSAL_DIR / "treatment_pretrend_availability.parquet"
COUNTERFACTUAL_COVERAGE = CAUSAL_DIR / "counterfactual_input_coverage_by_city.csv"

# ── Model data ──────────────────────────────────────────────────────

MODEL_INPUTS_DIR = ACTIVE_DIR / "model_inputs"

# ── VIIRS paths ─────────────────────────────────────────────────────

VIIRS_ANNUAL_DIR = ACTIVE_DIR / "curated" / "viirs_annual_aggregated"
VIIRS_MONTHLY_DIR = VIIRS_DIR / "monthly"


def viirs_annual_path(city: str) -> Path:
    return VIIRS_ANNUAL_DIR / f"{city}_viirs_annual.parquet"


def viirs_monthly_city_dir(city_key: str) -> Path:
    return VIIRS_MONTHLY_DIR / f"city_key={city_key}"


# ── Output directory constants ──────────────────────────────────────

OUTPUT_CAUSAL_LABELS_DIR = OUTPUT_DIR / "causal_labels"
OUTPUT_CAUSAL_TASKS_DIR = OUTPUT_CAUSAL_LABELS_DIR / "tasks"
OUTPUT_FIXED_CONTROL_DIR = OUTPUT_CAUSAL_LABELS_DIR / "fixed_control_staging"
OUTPUT_COMPLETE_ESTIMATORS_DIR = OUTPUT_DIR / "complete_estimators"
OUTPUT_COMPLETE_STAGING_DIR = OUTPUT_COMPLETE_ESTIMATORS_DIR / "staging"
OUTPUT_CONTROL_DESIGN_DIR = OUTPUT_DIR / "control_design"
OUTPUT_CONTROL_TASKS_DIR = OUTPUT_CONTROL_DESIGN_DIR / "tasks"
OUTPUT_HOUSING_FUSION_DIR = OUTPUT_DIR / "housing_fusion"
OUTPUT_HOUSING_PANEL_DIR = OUTPUT_DIR / "housing_panel"
OUTPUT_HOUSING_ACQUISITION_DIR = OUTPUT_DIR / "housing_acquisition"
OUTPUT_HOUSING_DID_DIR = OUTPUT_DIR / "housing_did_spatial_feasibility"
OUTPUT_HOUSING_DID_PREFLIGHT_DIR = OUTPUT_DIR / "housing_did_preflight"
OUTPUT_DATA_QUALITY_DIR = OUTPUT_DIR / "data_quality"
OUTPUT_GEE_QUALITY_DIR = OUTPUT_DIR / "gee_quality"
OUTPUT_POI_QUALITY_DIR = OUTPUT_DIR / "poi_quality"
OUTPUT_TRANSIT_COMPARISON_DIR = OUTPUT_DIR / "transit_comparison"
OUTPUT_VIIRS_MONTHLY_DIR = OUTPUT_DIR / "viirs_monthly"
OUTPUT_VIIRS_PARTITION_AUDITS_DIR = OUTPUT_VIIRS_MONTHLY_DIR / "partition_audits"
OUTPUT_STREETVIEW_DIR = OUTPUT_DIR / "streetview"

# ── Scripts ─────────────────────────────────────────────────────────

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SCRIPTS_CAUSAL_R_DIR = SCRIPTS_DIR / "causal_r"
SCRIPTS_COLLECTION_DIR = SCRIPTS_DIR / "collection"
SCRIPTS_ANALYSIS_DIR = SCRIPTS_DIR / "analysis"
SCRIPTS_LABELS_DIR = SCRIPTS_DIR / "labels"
SCRIPTS_DATA_DIR = SCRIPTS_DIR / "data"

SRC_DIR = PROJECT_ROOT / "src"

# ── Runtime ─────────────────────────────────────────────────────────

R_LIB_DIR = PROJECT_ROOT / ".r-lib"


# ── R scripts factory ───────────────────────────────────────────────


def r_script(name: str) -> Path:
    return SCRIPTS_CAUSAL_R_DIR / name


def collection_script(name: str) -> Path:
    return SCRIPTS_COLLECTION_DIR / name


# ── Housing staging ─────────────────────────────────────────────────

STAGING_HOUSING_DIR = STAGING_DIR / "housing"
STAGING_LIANJIA_TRANSACTIONS_DIR = STAGING_HOUSING_DIR / "lianjia_transactions"
STAGING_HOUSING_STANDARDIZED_DIR = STAGING_HOUSING_DIR / "standardized"
HOUSING_OBSERVATIONS_DIR = CURATED_DIR / "housing" / "housing_observations"

# ── Reference housing ───────────────────────────────────────────────

REFERENCE_HOUSING_DIR = REFERENCE_DIR / "housing"
COMMUNITY_REGISTRY = REFERENCE_HOUSING_DIR / "community_registry.parquet"
COMMUNITY_SOURCE_CROSSWALK = REFERENCE_HOUSING_DIR / "community_source_crosswalk.parquet"


# ── JSON export for R-side consumption ──────────────────────────────


def export_paths_json(output: Path | None = None) -> Path:
    import json

    target = output or RUNTIME_DIR / "paths.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    paths_map = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATA_ROOT": str(DATA_ROOT),
        "CAUSAL_DIR": str(CAUSAL_DIR),
        "FORMAL_MATCHING_DIR": str(FORMAL_MATCHING_DIR),
        "TREATMENT_UNIT_LIST": str(TREATMENT_UNIT_LIST),
        "CONTROL_DESIGN_QUEUE": str(CONTROL_DESIGN_QUEUE),
        "OUTCOME_FAMILY_QUEUE": str(OUTCOME_FAMILY_QUEUE),
        "GRID_UNIVERSE_DIR": str(GRID_UNIVERSE_DIR),
        "FORMAL_TARGET_SUPPORT": str(FORMAL_TARGET_SUPPORT),
        "ELIGIBLE_DONORS": str(ELIGIBLE_DONORS),
        "FEATURE_STORE_DIR": str(FEATURE_STORE_DIR),
        "SCRIPTS_CAUSAL_R_DIR": str(SCRIPTS_CAUSAL_R_DIR),
        "PANEL_HOUSING_MONTHLY_DIR": str(PANEL_HOUSING_MONTHLY_DIR),
        "PANEL_HOUSING_QUARTERLY_DIR": str(PANEL_HOUSING_QUARTERLY_DIR),
        "PANEL_HOUSING_YEARLY_DIR": str(PANEL_HOUSING_YEARLY_DIR),
        "POI_DIR": str(POI_DIR),
        "VIIRS_ANNUAL_DIR": str(VIIRS_ANNUAL_DIR),
        "VIIRS_MONTHLY_DIR": str(VIIRS_MONTHLY_DIR),
        "POPULATION_DIR": str(POPULATION_DIR),
        "SENTINEL2_DIR": str(SENTINEL2_DIR),
        "OUTPUT_DIR": str(OUTPUT_DIR),
        "OUTPUT_CAUSAL_TASKS_DIR": str(OUTPUT_CAUSAL_TASKS_DIR),
        "OUTPUT_COMPLETE_STAGING_DIR": str(OUTPUT_COMPLETE_STAGING_DIR),
        "OUTPUT_CONTROL_TASKS_DIR": str(OUTPUT_CONTROL_TASKS_DIR),
        "HOUSING_ANNUAL_DIR": str(FORMAL_MATCHING_DIR / "housing_annual"),
    }
    target.write_text(json.dumps(paths_map, indent=2), encoding="utf-8")
    return target
