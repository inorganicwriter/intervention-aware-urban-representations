"""Python-native construction of frozen causal queues and formal input audits."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from urban_intervention.utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
)

_atomic_csv = atomic_write_csv
_atomic_parquet = atomic_write_parquet
_atomic_json = partial(atomic_write_json, ensure_ascii=True, default=list)

FAMILIES: dict[str, tuple[str, ...]] = {
    "housing": ("housing_log_price",),
    "poi": (
        "poi_count_log",
        "poi_category_entropy",
        "poi_commercial_share",
        "poi_transport_access_log",
    ),
    "viirs": ("viirs_avg_asinh",),
    "population": ("population_log",),
}
REQUIRED_TREATMENT_COLUMNS = (
    "treatment_order",
    "city_key",
    "grid_id",
    "station_event_id",
    "opening_month",
)
FORMAL_SPEC_SCHEMA = "formal_counterfactual_design_v1"
MINIMUM_COMPLETE_FAMILIES = 1
FORMAL_SPEC_RELATIVE_PATH = Path(
    "data/active/causal/formal_matching_inputs/formal_matching_spec.dput"
)


def validate_frozen_formal_matching_spec(
    root: Path,
    *,
    expected_minimum_complete_families: int = MINIMUM_COMPLETE_FAMILIES,
) -> dict[str, object]:
    """Validate the read-only frozen spec before a formal production run.

    The active input tree is intentionally never rewritten here. A stale dput
    is a release blocker because it cannot prove which matching boundary was
    used to create the control queue.
    """
    path = root / FORMAL_SPEC_RELATIVE_PATH
    if not path.is_file():
        raise ValueError(f"Frozen formal matching spec is missing: {path}")
    text = path.read_text(encoding="utf-8-sig")
    schema_match = re.search(r'schema\s*=\s*"([^"]+)"', text)
    minimum_match = re.search(r"minimum_complete_families\s*=\s*(\d+)L?", text)
    if schema_match is None or minimum_match is None:
        raise ValueError(f"Frozen formal matching spec is not parseable: {path}")
    schema = schema_match.group(1)
    minimum = int(minimum_match.group(1))
    if schema != FORMAL_SPEC_SCHEMA:
        raise ValueError(
            f"Frozen formal matching spec schema {schema!r} does not match "
            f"{FORMAL_SPEC_SCHEMA!r}"
        )
    if minimum != expected_minimum_complete_families:
        raise ValueError(
            "Frozen formal matching spec is stale: "
            f"minimum_complete_families={minimum}, expected "
            f"{expected_minimum_complete_families}. Reconcile the spec in an "
            "isolated server working copy before production; data/active is immutable."
        )
    return {
        "path": str(path),
        "schema": schema,
        "minimum_complete_families": minimum,
    }


def _city_parquet_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.glob("*.parquet")
        if path.stem.isalpha() and path.stem.islower()
    )


def validate_treatments(treatments: pd.DataFrame, *, expected_rows: int = 5_048) -> None:
    missing = set(REQUIRED_TREATMENT_COLUMNS) - set(treatments.columns)
    if missing:
        raise ValueError(f"frozen treatment list lacks columns: {sorted(missing)}")
    if len(treatments) != expected_rows:
        raise ValueError(f"expected {expected_rows:,} treatments, found {len(treatments):,}")
    if treatments["treatment_order"].duplicated().any():
        raise ValueError("treatment_order is not unique")
    if treatments.duplicated(["city_key", "grid_id"]).any():
        raise ValueError("treated city/grid identity is not unique")


def build_pending_queues(treatments: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create the three R-schema-compatible pending queues."""
    validate_treatments(treatments)
    identity = treatments.loc[:, REQUIRED_TREATMENT_COLUMNS].copy()
    identity = identity.sort_values("treatment_order", kind="stable").reset_index(drop=True)
    unit = identity.assign(
        status="pending",
        selected_method=pd.NA,
        selected_control_grid_id=pd.NA,
        failure_reason=pd.NA,
    )
    family = identity.merge(
        pd.DataFrame({"outcome_family": list(FAMILIES)}), how="cross"
    ).assign(status="pending", selected_method=pd.NA, failure_reason=pd.NA)
    family = family.sort_values(
        ["treatment_order", "outcome_family"], kind="stable"
    ).reset_index(drop=True)
    control = identity.assign(
        status="pending",
        active_families=pd.NA,
        selected_method=pd.NA,
        donor_scope=pd.NA,
        control_city_key=pd.NA,
        control_grid_id=pd.NA,
        control_unit_key=pd.NA,
        candidate_count=pd.NA,
        candidate_city_count=pd.NA,
        training_feature_count=pd.NA,
        holdout_feature_count=pd.NA,
        training_distance=np.nan,
        holdout_rms_standardized_gap=np.nan,
        holdout_max_abs_standardized_gap=np.nan,
        training_distance_threshold=np.nan,
        holdout_rms_threshold=np.nan,
        holdout_max_abs_threshold=np.nan,
        control_selection_uses_post_outcome=False,
        failure_reason=pd.NA,
    )
    return {"unit": unit, "family": family, "control": control}


def reset_queues(root: Path) -> dict[str, int]:
    causal = root / "data" / "active" / "causal"
    treatments = pd.read_parquet(causal / "treatment_unit_list.parquet")
    queues = build_pending_queues(treatments)
    _atomic_csv(queues["unit"], causal / "counterfactual_work_queue.csv")
    _atomic_csv(queues["family"], causal / "outcome_family_work_queue.csv")
    _atomic_csv(queues["control"], causal / "control_design_queue.csv")
    return {key: len(value) for key, value in queues.items()}


def build_eligible_donors(universe_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for frame in universe_frames:
        required = {
            "city_key",
            "grid_id",
            "is_nonexperimental_grid",
            "known_station_contamination",
            "primary_spatial_exclusion_reason",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"grid universe lacks columns: {sorted(missing)}")
        mask = (
            frame["is_nonexperimental_grid"].fillna(False).astype(bool)
            & ~frame["known_station_contamination"].fillna(True).astype(bool)
            & frame["primary_spatial_exclusion_reason"].eq("eligible_spatial_donor")
        )
        pieces.append(frame.loc[mask, ["city_key", "grid_id"]])
    donors = pd.concat(pieces, ignore_index=True)
    donors["unit_id"] = donors["city_key"].astype(str) + "::" + donors["grid_id"].astype(str)
    if donors["unit_id"].duplicated().any():
        raise ValueError("formal donor universe is not unique")
    return donors.sort_values(["city_key", "grid_id"], kind="stable").reset_index(drop=True)


def build_housing_annual(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "city_key",
        "grid_id",
        "observed_month",
        "log_price_raw_median",
        "n_observations",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"housing panel lacks columns: {sorted(missing)}")
    values = frame.loc[np.isfinite(pd.to_numeric(frame["log_price_raw_median"], errors="coerce"))].copy()
    values["year"] = pd.to_datetime(values["observed_month"]).dt.year
    values["log_price_raw_median"] = pd.to_numeric(
        values["log_price_raw_median"], errors="coerce"
    )
    values["n_observations"] = pd.to_numeric(values["n_observations"], errors="coerce")
    return (
        values.groupby(["city_key", "grid_id", "year"], as_index=False, sort=False)
        .agg(
            housing_log_price=("log_price_raw_median", "median"),
            housing_observed_months=("observed_month", "nunique"),
            housing_observations=("n_observations", "sum"),
        )
        .sort_values(["grid_id", "year"], kind="stable")
        .reset_index(drop=True)
    )


def _formal_spec() -> dict[str, Any]:
    return {
        "schema": FORMAL_SPEC_SCHEMA,
        "treatment_history_lag_years": 3,
        "pre_year_lags": [1, 2, 3],
        "minimum_complete_families": MINIMUM_COMPLETE_FAMILIES,
        "matching_with_replacement": True,
        "matches_per_treated": 1,
        "post_treatment_data_used_for_matching": False,
        "families": FAMILIES,
        "gsc": {
            "force": "two-way",
            "factor_candidates": list(range(6)),
            "minimum_pre_periods": 5,
            "bootstrap_replications": 200,
        },
    }


def rebuild_formal_inputs(root: Path) -> dict[str, int]:
    causal = root / "data" / "active" / "causal"
    formal = causal / "formal_matching_inputs"
    universe_paths = _city_parquet_paths(causal / "grid_universe")
    if len(universe_paths) != 44:
        raise ValueError(f"expected 44 grid-universe files, found {len(universe_paths)}")
    universe_columns = [
        "city_key",
        "grid_id",
        "is_nonexperimental_grid",
        "known_station_contamination",
        "primary_spatial_exclusion_reason",
    ]
    donors = build_eligible_donors(
        pd.read_parquet(path, columns=universe_columns) for path in universe_paths
    )
    _atomic_parquet(donors, formal / "eligible_never_treated_donors.parquet")

    housing_paths = _city_parquet_paths(
        root / "data" / "active" / "panels" / "housing_grid_month"
    )
    if len(housing_paths) != 44:
        raise ValueError(f"expected 44 housing panel files, found {len(housing_paths)}")
    coverage: list[dict[str, object]] = []
    total_rows = 0
    for path in housing_paths:
        annual = build_housing_annual(
            pd.read_parquet(
                path,
                columns=[
                    "city_key",
                    "grid_id",
                    "observed_month",
                    "log_price_raw_median",
                    "n_observations",
                ],
            )
        )
        city = path.stem
        _atomic_parquet(annual, formal / "housing_annual" / f"{city}.parquet")
        total_rows += len(annual)
        coverage.append(
            {
                "city_key": city,
                "rows": len(annual),
                "grids": annual["grid_id"].nunique(),
                "first_year": annual["year"].min(),
                "last_year": annual["year"].max(),
            }
        )
    _atomic_csv(pd.DataFrame(coverage), formal / "housing_annual_coverage.csv")
    _atomic_json(_formal_spec(), formal / "formal_matching_spec.json")
    metadata = pd.DataFrame(
        [
            {
                "schema": _formal_spec()["schema"],
                "formal_donors": len(donors),
                "cities": donors["city_key"].nunique(),
                "housing_annual_rows": total_rows,
                "created_utc": datetime.now(UTC).isoformat(),
                "builder": "python",
            }
        ]
    )
    _atomic_csv(metadata, formal / "build_metadata.csv")
    return {"formal_donors": len(donors), "housing_annual_rows": total_rows}


def _read_family(root: Path, city: str, family: str) -> pd.DataFrame:
    if family == "housing":
        return pd.read_parquet(
            root
            / "data"
            / "active"
            / "causal"
            / "formal_matching_inputs"
            / "housing_annual"
            / f"{city}.parquet",
            columns=["city_key", "grid_id", "year", "housing_log_price"],
        )
    if family == "poi":
        frame = pd.read_parquet(
            root / "data" / "active" / "curated" / "poi" / f"{city}_poi_grid_yearly.parquet",
            columns=[
                "city",
                "grid_id",
                "year",
                "poi_count",
                "poi_category_entropy",
                "poi_commercial_share",
                "poi_transport_access_count",
            ],
        ).rename(columns={"city": "city_key"})
        frame["poi_count_log"] = np.log1p(
            pd.to_numeric(frame["poi_count"], errors="coerce").clip(lower=0)
        )
        frame["poi_transport_access_log"] = np.log1p(
            pd.to_numeric(frame["poi_transport_access_count"], errors="coerce").clip(lower=0)
        )
        return frame[["city_key", "grid_id", "year", *FAMILIES[family]]]
    if family == "viirs":
        frame = pd.read_parquet(
            root
            / "data"
            / "active"
            / "curated"
            / "viirs_annual_aggregated"
            / f"{city}_viirs_annual.parquet",
            columns=["city_key", "grid_id", "year", "avg_rad"],
        )
        frame["viirs_avg_asinh"] = np.arcsinh(
            pd.to_numeric(frame["avg_rad"], errors="coerce")
        )
        return frame[["city_key", "grid_id", "year", "viirs_avg_asinh"]]
    frame = pd.read_parquet(
        root / "data" / "active" / "curated" / "population" / f"{city}_pop.parquet",
        columns=["city", "grid_id", "year", "pop_count"],
    ).rename(columns={"city": "city_key"})
    frame["population_log"] = np.log1p(
        pd.to_numeric(frame["pop_count"], errors="coerce").clip(lower=0)
    )
    return (
        frame.groupby(["city_key", "grid_id", "year"], as_index=False, sort=False)[
            "population_log"
        ]
        .mean()
    )


def audit_family_support(
    treatments: pd.DataFrame, outcomes: pd.DataFrame, family: str
) -> pd.DataFrame:
    variables = list(FAMILIES[family])
    target = treatments.loc[:, ["treatment_order", "city_key", "grid_id", "opening_year"]]
    merged = target.merge(outcomes, on=["city_key", "grid_id"], how="left")
    years = pd.to_numeric(merged["year"], errors="coerce")
    complete = merged[variables].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    lag = merged["opening_year"] - years

    def counts(mask: pd.Series, prefix: str) -> pd.DataFrame:
        part = merged.loc[mask, ["treatment_order"]].copy()
        part["year"] = years.loc[mask]
        part["complete"] = complete.loc[mask]
        observed = part.groupby("treatment_order")["year"].nunique()
        complete_years = part.loc[part["complete"]].groupby("treatment_order")["year"].nunique()
        return pd.DataFrame(
            {
                f"{prefix}_complete_years": complete_years,
                f"{prefix}_observed_years": observed,
            }
        ).reset_index()

    post = counts(lag.isin([1, 2, 3]), family)
    pre = counts(years.lt(merged["opening_year"]), f"{family}_gsc_pre")
    result = target[["treatment_order"]].merge(post, on="treatment_order", how="left")
    result = result.merge(pre, on="treatment_order", how="left").fillna(0)
    numeric = [column for column in result if column != "treatment_order"]
    result[numeric] = result[numeric].astype(np.int64)
    result[f"{family}_complete"] = result[f"{family}_complete_years"].eq(3)
    result[f"{family}_gsc_ready"] = result[f"{family}_complete"] & result[
        f"{family}_gsc_pre_complete_years"
    ].ge(5)
    return result


def audit_formal_target_support(root: Path) -> pd.DataFrame:
    causal = root / "data" / "active" / "causal"
    treatments = pd.read_parquet(causal / "treatment_unit_list.parquet")
    validate_treatments(treatments)
    treatments = treatments.copy()
    treatments["opening_year"] = treatments["opening_month"].astype(str).str[:4].astype(int)
    audit = treatments[["treatment_order", "city_key", "grid_id", "opening_month"]].copy()
    cities = sorted(treatments["city_key"].astype(str).unique())
    for family in FAMILIES:
        outcomes = pd.concat(
            [_read_family(root, city, family) for city in cities], ignore_index=True
        )
        support = audit_family_support(treatments, outcomes, family)
        audit = audit.merge(support, on="treatment_order", how="left")
    complete_columns = [f"{family}_complete" for family in FAMILIES]
    ready_columns = [f"{family}_gsc_ready" for family in FAMILIES]
    audit["complete_families"] = audit[complete_columns].sum(axis=1).astype(int)
    audit["gsc_ready_families"] = audit[ready_columns].sum(axis=1).astype(int)
    audit = audit.sort_values("treatment_order", kind="stable").reset_index(drop=True)
    formal = causal / "formal_matching_inputs"
    _atomic_parquet(audit, formal / "formal_target_support.parquet")
    summary = (
        audit.groupby("complete_families", as_index=False)
        .size()
        .rename(columns={"size": "N"})
        .sort_values("complete_families")
    )
    _atomic_csv(summary, formal / "formal_target_support_summary.csv")
    return audit
