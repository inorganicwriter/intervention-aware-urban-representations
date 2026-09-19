"""Python/GPU frozen-control design for the six-round causal router."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from urban_intervention.data.paths import (
    ELIGIBLE_DONORS,
    PROJECT_ROOT,
    TREATMENT_UNIT_LIST,
)
from urban_intervention.utils import atomic_write_json

from .contracts import (
    CONTROL_DESIGN_SCHEMA,
    CONTROL_DESIGN_VIIRS_CACHE_CONTRACT,
    FORMAL_IMPLEMENTATION_VERSION,
    MatchingInput,
    MatchingResult,
)
from .matching import MatchingConfig, fit_matching
from .panel_builder import (
    OUTCOMES,
    deterministic_cross_city_gsc_sample,
    monthly_event_calendar,
    read_annual_outcome,
    read_monthly_housing,
    read_monthly_viirs,
)
from .provenance import estimator_code_fingerprint
from .runtime import RuntimeConfig, TorchRuntime

Scope = Literal["same_city", "all_city_standardized"]
ProgressCallback = Callable[[dict[str, object]], None]
VIIRS_CACHE_CONTRACT = CONTROL_DESIGN_VIIRS_CACHE_CONTRACT
VIIRS_PRE_PATH_MONTHS = 36
VIIRS_TRAINING_MONTHS = 24
STATIC_FEATURES = (
    "loc_dist_main_km",
    "loc_dist_nearest_subcentre_km",
    "loc_dist_nearest_centre_km",
    "transit_dist_nearest_station_m",
    "transit_stations_500m",
    "transit_stations_800m",
    "transit_stations_1500m",
    "transit_lines_in_1500m",
    "transit_network_closeness",
)


@dataclass(frozen=True, slots=True)
class ControlDesignResult:
    record: pd.DataFrame
    feature_balance: pd.DataFrame | None
    feature_balance_summary: pd.DataFrame | None
    matching_result: MatchingResult | None


def family_feature_names(family: str) -> tuple[str, ...]:
    if family == "viirs":
        return tuple(
            f"viirs_avg_asinh__pre_m{month:02d}" for month in range(1, VIIRS_PRE_PATH_MONTHS + 1)
        )
    return tuple(f"{outcome}__lag{lag}" for outcome in OUTCOMES[family] for lag in range(1, 4))


def split_preonly_features(features: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split pre-treatment features without collapsing VIIRS monthly dynamics."""
    training: list[str] = []
    holdout: list[str] = []
    for feature in features:
        viirs_month = re.search(r"__pre_m(\d{2})$", feature)
        if viirs_month is not None:
            destination = (
                training if int(viirs_month.group(1)) <= VIIRS_TRAINING_MONTHS else holdout
            )
            destination.append(feature)
        elif feature.endswith(("__lag2", "__lag3")):
            training.append(feature)
        elif feature.endswith("__lag1"):
            holdout.append(feature)
    if not training or not holdout:
        raise ValueError("pre-only design requires non-empty training and holdout features")
    return tuple(training), tuple(holdout)


def _filter_grid_ids(frame: pd.DataFrame, grid_ids: frozenset[str] | None) -> pd.DataFrame:
    if grid_ids is None or frame.empty:
        return frame
    return frame.loc[frame["grid_id"].astype(str).isin(grid_ids)].copy()


def _static_features(
    root: Path,
    city: str,
    opening_month: str,
    grid_ids: frozenset[str] | None = None,
) -> pd.DataFrame:
    location_path = (
        root / "data" / "active" / "curated" / "location_features" / f"{city}_location.parquet"
    )
    transit_path = (
        root
        / "data"
        / "active"
        / "causal"
        / "transit_snapshots"
        / city
        / f"{opening_month}.parquet"
    )
    if not location_path.exists() or not transit_path.exists():
        return pd.DataFrame(columns=["city_key", "grid_id", *STATIC_FEATURES])
    location = pd.read_parquet(
        location_path,
        columns=[
            "grid_id",
            "dist_main_km",
            "dist_nearest_subcentre_km",
            "dist_nearest_centre_km",
        ],
    ).rename(
        columns={
            "dist_main_km": "loc_dist_main_km",
            "dist_nearest_subcentre_km": "loc_dist_nearest_subcentre_km",
            "dist_nearest_centre_km": "loc_dist_nearest_centre_km",
        }
    )
    location = _filter_grid_ids(location, grid_ids)
    transit = pd.read_parquet(
        transit_path,
        columns=[
            "grid_id",
            "dist_nearest_station_m",
            "stations_500m",
            "stations_800m",
            "stations_1500m",
            "lines_in_1500m",
            "network_closeness",
        ],
    ).rename(
        columns={
            "dist_nearest_station_m": "transit_dist_nearest_station_m",
            "stations_500m": "transit_stations_500m",
            "stations_800m": "transit_stations_800m",
            "stations_1500m": "transit_stations_1500m",
            "lines_in_1500m": "transit_lines_in_1500m",
            "network_closeness": "transit_network_closeness",
        }
    )
    transit = _filter_grid_ids(transit, grid_ids)
    result = location.merge(transit, on="grid_id", how="outer")
    result.insert(0, "city_key", city)
    return result


def _monthly_blocks(
    frame: pd.DataFrame,
    outcomes: tuple[str, ...],
    pre: pd.DatetimeIndex,
    minimum: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    mapping = pd.DataFrame(
        {
            "period": pre,
            "lag": 3 - np.arange(len(pre), dtype=int) // 12,
        }
    )
    selected = frame.merge(mapping, on="period", how="inner")
    rows: list[pd.DataFrame] = []
    for outcome in outcomes:
        values = pd.to_numeric(selected[outcome], errors="coerce")
        values = values.where(np.isfinite(values))
        summaries = (
            selected[["city_key", "grid_id", "lag"]]
            .assign(_value=values)
            .groupby(["city_key", "grid_id", "lag"], sort=False)["_value"]
            .agg(["mean", "count"])
        )
        summaries.loc[summaries["count"].lt(minimum), "mean"] = np.nan
        aggregates = summaries["mean"].unstack("lag")
        aggregates.columns = [f"{outcome}__lag{int(value)}" for value in aggregates.columns]
        rows.append(aggregates.reset_index())
    result = rows[0]
    for part in rows[1:]:
        result = result.merge(part, on=["city_key", "grid_id"], how="outer")
    return result


def _monthly_path_features(
    frame: pd.DataFrame,
    outcomes: tuple[str, ...],
    pre: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Preserve each clean-pre month for monthly trajectory matching."""
    if frame.empty:
        return pd.DataFrame()
    mapping = pd.DataFrame(
        {
            "period": pre,
            "pre_month": np.arange(1, len(pre) + 1, dtype=int),
        }
    )
    selected = frame.merge(mapping, on="period", how="inner")
    rows: list[pd.DataFrame] = []
    for outcome in outcomes:
        values = pd.to_numeric(selected[outcome], errors="coerce")
        values = values.where(np.isfinite(values))
        summaries = (
            selected[["city_key", "grid_id", "pre_month"]]
            .assign(_value=values)
            .groupby(["city_key", "grid_id", "pre_month"], sort=False)["_value"]
            .agg(["mean", "count"])
        )
        summaries.loc[summaries["count"].ne(1), "mean"] = np.nan
        path = summaries["mean"].unstack("pre_month").reindex(columns=range(1, len(pre) + 1))
        path.columns = [f"{outcome}__pre_m{int(value):02d}" for value in path.columns]
        rows.append(path.reset_index())
    result = rows[0]
    for part in rows[1:]:
        result = result.merge(part, on=["city_key", "grid_id"], how="outer")
    return result


def _annual_blocks(
    frame: pd.DataFrame, outcomes: tuple[str, ...], opening_year: int
) -> pd.DataFrame:
    selected = frame.loc[
        frame["year"].isin([opening_year - 1, opening_year - 2, opening_year - 3])
    ].copy()
    selected["lag"] = opening_year - selected["year"].astype(int)
    parts: list[pd.DataFrame] = []
    for outcome in outcomes:
        pivot = selected.pivot(index=["city_key", "grid_id"], columns="lag", values=outcome)
        pivot.columns = [f"{outcome}__lag{int(value)}" for value in pivot.columns]
        parts.append(pivot.reset_index())
    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part, on=["city_key", "grid_id"], how="outer")
    return result


def read_city_control_features(
    root: Path,
    city: str,
    target: pd.Series,
    families: tuple[str, ...],
    *,
    grid_ids: frozenset[str] | None = None,
) -> pd.DataFrame:
    opening_month = str(target["opening_month"])
    opening_year = int(opening_month[:4])
    calendar = monthly_event_calendar(opening_month, leads=(1,), anticipation_months=6)
    parts: list[pd.DataFrame] = []
    for family in families:
        try:
            if family == "housing":
                source = read_monthly_housing(
                    root,
                    city,
                    "median",
                    months=calendar["pre"],
                    grid_ids=grid_ids,
                )
                part = _monthly_blocks(source, OUTCOMES[family], calendar["pre"], 1)
            elif family == "viirs":
                source = read_monthly_viirs(
                    root, city, calendar["pre"], grid_ids=set(grid_ids) if grid_ids else None
                )
                part = _monthly_path_features(source, OUTCOMES[family], calendar["pre"])
            else:
                source = read_annual_outcome(root, city, family, grid_ids=grid_ids)
                part = _annual_blocks(source, OUTCOMES[family], opening_year)
            if not part.empty:
                parts.append(part)
        except (FileNotFoundError, ValueError, KeyError):
            continue
    result = parts[0] if parts else pd.DataFrame(columns=["city_key", "grid_id"])
    for part in parts[1:]:
        result = result.merge(part, on=["city_key", "grid_id"], how="outer")
    static = _static_features(root, city, opening_month, grid_ids)
    if result.empty:
        return static
    return result.merge(static, on=["city_key", "grid_id"], how="left")


def active_families(target: pd.Series, features: pd.DataFrame) -> tuple[str, ...]:
    row = features.loc[
        features["city_key"].eq(str(target["city_key"]))
        & features["grid_id"].eq(str(target["grid_id"]))
    ]
    if len(row) != 1:
        return ()
    return tuple(
        family
        for family in OUTCOMES
        if set(family_feature_names(family)).issubset(row.columns)
        and np.isfinite(row.iloc[0][list(family_feature_names(family))].to_numpy(dtype=float)).all()
    )


def _r_mad(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if len(numeric) <= 1:
        return np.nan
    return float(1.4826 * np.median(np.abs(numeric - np.median(numeric))))


def robust_city_standardize(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    donors = result.loc[result["role"].eq("donor")]
    for city, indices in result.groupby("city_key", sort=False).groups.items():
        city_donors = donors.loc[donors["city_key"].eq(city), list(features)].apply(
            pd.to_numeric, errors="coerce"
        )
        centers = city_donors.median(axis=0)
        scales = (city_donors - centers).abs().median(axis=0) * 1.4826
        fallback = city_donors.std(axis=0, ddof=1)
        cutoff = np.sqrt(np.finfo(float).eps)
        valid_scales = np.isfinite(scales) & scales.gt(cutoff)
        scales = scales.where(valid_scales, fallback)
        valid_scales = np.isfinite(scales) & scales.gt(cutoff)
        scales = scales.where(valid_scales, 1.0)
        city_values = result.loc[indices, list(features)].apply(pd.to_numeric, errors="coerce")
        result.loc[indices, list(features)] = ((city_values - centers) / scales).to_numpy()
    return result


@lru_cache(maxsize=4)
def _cached_donor_universe(root_value: str) -> pd.DataFrame:
    root = Path(root_value)
    return pd.read_parquet(
        root / ELIGIBLE_DONORS.relative_to(PROJECT_ROOT),
        columns=["city_key", "grid_id", "unit_id"],
    )


@lru_cache(maxsize=4)
def _cached_treatment_units(root_value: str) -> pd.DataFrame:
    root = Path(root_value)
    return pd.read_parquet(root / TREATMENT_UNIT_LIST.relative_to(PROJECT_ROOT))


def _donor_grid_ids_by_city(donors: pd.DataFrame) -> dict[str, frozenset[str]]:
    return {
        str(city): frozenset(group["grid_id"].astype(str))
        for city, group in donors.groupby("city_key", sort=False)
    }


def usable_static_features(
    treated: pd.Series,
    donors: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[str, ...]:
    """Return static columns that are observed for this matching task.

    A city can have no pre-existing rail station at an early opening date. In
    that case the nearest-station distance is structurally undefined for the
    treated grid and every same-city donor. Such a column carries no
    cross-sectional information and is omitted from static refinement while
    the remaining static columns stay available.
    """
    usable: list[str] = []
    for feature in features:
        target_value = pd.to_numeric(pd.Series([treated[feature]]), errors="coerce").iloc[0]
        donor_values = pd.to_numeric(donors[feature], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(target_value) and np.isfinite(donor_values).all():
            usable.append(feature)
    return tuple(usable)


def build_scope_frame(
    target: pd.Series,
    families: tuple[str, ...],
    scope: Scope,
    *,
    root: Path,
    target_feature_data: pd.DataFrame | None = None,
    donor_universe: pd.DataFrame | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    donors = (
        donor_universe
        if donor_universe is not None
        else _cached_donor_universe(str(root.resolve()))
    )
    donor_grid_ids = _donor_grid_ids_by_city(donors)
    cities = (
        [str(target["city_key"])]
        if scope == "same_city"
        else sorted(donors["city_key"].astype(str).unique())
    )
    features = tuple(
        [feature for family in families for feature in family_feature_names(family)]
        + list(STATIC_FEATURES)
    )
    parts: list[pd.DataFrame] = []
    total_cities = len(cities)
    for city_index, city in enumerate(cities, start=1):
        city_grid_ids = donor_grid_ids.get(city, frozenset())
        if city == str(target["city_key"]):
            city_grid_ids = city_grid_ids | frozenset({str(target["grid_id"])})
        part = (
            target_feature_data
            if city == str(target["city_key"]) and target_feature_data is not None
            else read_city_control_features(root, city, target, families, grid_ids=city_grid_ids)
        )
        if not part.empty and set(features).issubset(part.columns):
            parts.append(part[["city_key", "grid_id", *features]])
        if progress is not None:
            progress(
                {
                    "stage": "feature_city_complete",
                    "city_key": city,
                    "completed_cities": city_index,
                    "total_cities": total_cities,
                }
            )
    if not parts:
        raise ValueError("no city has the required pre-treatment feature blocks")
    feature_data = pd.concat(parts, ignore_index=True)
    donor_features = donors.merge(feature_data, on=["city_key", "grid_id"], how="inner")
    target_features = (
        target_feature_data
        if target_feature_data is not None
        else read_city_control_features(
            root,
            str(target["city_key"]),
            target,
            families,
            grid_ids=frozenset({str(target["grid_id"])}),
        )
    )
    target_features = target_features.loc[
        target_features["city_key"].eq(str(target["city_key"]))
        & target_features["grid_id"].eq(str(target["grid_id"]))
    ]
    if len(target_features) != 1:
        raise ValueError("target pre-treatment feature row is unavailable")
    treated = target_features[["city_key", "grid_id", *features]].copy()
    treated["unit_id"] = treated["city_key"] + "::" + treated["grid_id"]
    treated["role"] = "treated"
    donor_features = donor_features[["city_key", "grid_id", "unit_id", *features]].copy()
    donor_features["role"] = "donor"
    frame = pd.concat([treated, donor_features], ignore_index=True)
    outcome_features = tuple(
        feature for family in families for feature in family_feature_names(family)
    )
    frame = frame.dropna(subset=list(outcome_features))
    finite = np.isfinite(frame[list(outcome_features)].to_numpy(dtype=float)).all(axis=1)
    frame = frame.loc[finite].copy()
    if scope == "same_city":
        frame = frame.loc[frame["city_key"].eq(str(target["city_key"]))].copy()
    if frame["role"].eq("treated").sum() != 1 or frame["role"].eq("donor").sum() < 3:
        raise ValueError("fewer than three complete donors or missing treated feature row")
    if scope == "all_city_standardized":
        frame = robust_city_standardize(frame, features)
    frame["_role"] = pd.Categorical(frame["role"], categories=["donor", "treated"], ordered=True)
    frame = frame.sort_values(["_role", "city_key", "grid_id"], kind="stable").drop(columns="_role")
    return frame.reset_index(drop=True), features


def _feature_balance(
    frame: pd.DataFrame, selected_donor: int, features: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    treated = frame.loc[frame["role"].eq("treated")].iloc[0]
    donors = frame.loc[frame["role"].eq("donor")].reset_index(drop=True)
    control = donors.iloc[selected_donor]
    standard_deviation = donors[list(features)].std(ddof=1)
    raw_gap = treated[list(features)].to_numpy(dtype=float) - control[list(features)].to_numpy(
        dtype=float
    )
    standardized = raw_gap / standard_deviation.to_numpy(dtype=float)
    standardized[~np.isfinite(standardized)] = np.nan
    long = pd.DataFrame(
        {
            "pair_index": 1,
            "treated_unit_id": treated["unit_id"],
            "control_unit_id": control["unit_id"],
            "feature": features,
            "treated_value": treated[list(features)].to_numpy(dtype=float),
            "control_value": control[list(features)].to_numpy(dtype=float),
            "raw_gap": raw_gap,
            "standardized_gap": standardized,
        }
    )
    summary = pd.DataFrame(
        [
            {
                "pair_index": 1,
                "treated_unit_id": treated["unit_id"],
                "control_unit_id": control["unit_id"],
                "preonly_rms_standardized_gap": float(np.sqrt(np.nanmean(np.square(standardized)))),
                "preonly_max_abs_standardized_gap": float(np.nanmax(np.abs(standardized))),
                "active_feature_count": int(np.isfinite(standardized).sum()),
            }
        ]
    )
    return long, summary


def _base_record(
    target: pd.Series, families: tuple[str, ...], status: str, reason: str | None
) -> dict[str, object]:
    return {
        "schema": CONTROL_DESIGN_SCHEMA,
        "implementation_version": FORMAL_IMPLEMENTATION_VERSION,
        "code_fingerprint": estimator_code_fingerprint("matching"),
        "backend": "python_pytorch",
        "viirs_cache_contract": VIIRS_CACHE_CONTRACT,
        "treatment_order": int(target["treatment_order"]),
        "city_key": str(target["city_key"]),
        "grid_id": str(target["grid_id"]),
        "station_event_id": target.get("station_event_id", pd.NA),
        "opening_month": str(target["opening_month"]),
        "status": status,
        "active_families": "+".join(sorted(families)),
        "selected_method": pd.NA,
        "donor_scope": pd.NA,
        "control_city_key": pd.NA,
        "control_grid_id": pd.NA,
        "control_unit_key": pd.NA,
        "candidate_count": pd.NA,
        "candidate_city_count": pd.NA,
        "eligible_scope_donors": pd.NA,
        "donors_after_pre_outcome_cap": pd.NA,
        "pre_outcome_donor_cap": "none",
        "training_feature_count": pd.NA,
        "holdout_feature_count": pd.NA,
        "active_static_features": pd.NA,
        "inactive_static_features": pd.NA,
        "training_distance": np.nan,
        "holdout_rms_standardized_gap": np.nan,
        "holdout_max_abs_standardized_gap": np.nan,
        "training_distance_threshold": np.nan,
        "holdout_rms_threshold": np.nan,
        "holdout_max_abs_threshold": np.nan,
        "control_selection_uses_post_outcome": False,
        "failure_reason": reason,
    }


def design_grid_control(
    treatment_order: int,
    *,
    scope: Scope = "same_city",
    root: Path = PROJECT_ROOT,
    device: str = "auto",
    progress: ProgressCallback | None = None,
    max_cross_city_donors: int = 5_000,
    donor_sampling_seed: int = 20260823,
) -> ControlDesignResult:
    if max_cross_city_donors < 20:
        raise ValueError("max_cross_city_donors must be at least 20")
    started = time.perf_counter()
    if progress is None and scope == "all_city_standardized":
        progress_path = (
            root / "outputs" / "control_design" / "progress" / f"{treatment_order:05d}.json"
        )

        def persistent_progress(details: dict[str, object]) -> None:
            with suppress(OSError):
                atomic_write_json(
                    {
                        "schema": "cross_city_control_progress",
                        "treatment_order": treatment_order,
                        **details,
                    },
                    progress_path,
                    default=str,
                )

        progress = persistent_progress

    def report(stage: str, **details: object) -> None:
        if progress is not None:
            progress(
                {
                    "stage": stage,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    **details,
                }
            )

    report("loading_treatment_and_donors")
    treatments = _cached_treatment_units(str(root.resolve()))
    donor_universe = _cached_donor_universe(str(root.resolve()))
    selected = treatments.loc[treatments["treatment_order"].eq(treatment_order)]
    if len(selected) != 1:
        raise ValueError("treatment order is not unique")
    target = selected.iloc[0]
    if scope == "same_city":
        donor_universe = donor_universe.loc[
            donor_universe["city_key"].astype(str).eq(str(target["city_key"]))
        ].copy()
    eligible_scope_donors = len(donor_universe)
    pre_outcome_donor_cap = "none"
    if scope == "all_city_standardized" and len(donor_universe) > max_cross_city_donors:
        donor_universe = deterministic_cross_city_gsc_sample(
            donor_universe,
            max_cross_city_donors,
            donor_sampling_seed,
        )
        pre_outcome_donor_cap = (
            f"uniform_stable_hash_{max_cross_city_donors}"
            f"_seed_{donor_sampling_seed}_before_outcomes"
        )
    report(
        "donor_scope_ready",
        eligible_scope_donors=eligible_scope_donors,
        donors_after_pre_outcome_cap=len(donor_universe),
        pre_outcome_donor_cap=pre_outcome_donor_cap,
    )

    def add_scope_metadata(record: dict[str, object]) -> None:
        record.update(
            {
                "eligible_scope_donors": eligible_scope_donors,
                "donors_after_pre_outcome_cap": len(donor_universe),
                "pre_outcome_donor_cap": pre_outcome_donor_cap,
            }
        )

    donor_grid_ids = _donor_grid_ids_by_city(donor_universe)
    target_city_grid_ids = donor_grid_ids.get(str(target["city_key"]), frozenset()) | frozenset(
        {str(target["grid_id"])}
    )
    report("loading_target_city_features")
    target_features = read_city_control_features(
        root,
        str(target["city_key"]),
        target,
        tuple(OUTCOMES),
        grid_ids=target_city_grid_ids,
    )
    families = active_families(target, target_features)
    failure_status = "gsc_pending" if scope == "same_city" else "not_matched"
    if not families:
        record = _base_record(
            target, families, failure_status, "fewer_than_1_complete_pre_treatment_families"
        )
        add_scope_metadata(record)
        report("completed", status=failure_status, failure_reason=record["failure_reason"])
        return ControlDesignResult(pd.DataFrame([record]), None, None, None)
    try:
        frame, features = build_scope_frame(
            target,
            families,
            scope,
            root=root,
            target_feature_data=target_features,
            donor_universe=donor_universe,
            progress=lambda details: report(**details),  # type: ignore[arg-type]
        )
        report(
            "matching",
            candidate_rows=int(frame["role"].eq("donor").sum()),
            feature_count=len(features),
        )
        training, holdout = split_preonly_features(features)
        static_candidates = tuple(feature for feature in STATIC_FEATURES if feature in features)
        treated = frame.loc[frame["role"].eq("treated")].iloc[0]
        donors = frame.loc[frame["role"].eq("donor")].reset_index(drop=True)
        static = usable_static_features(treated, donors, static_candidates)
        matching_input = MatchingInput(
            target=treated[list(training)].to_numpy(dtype=float),
            donors=donors[list(training)].to_numpy(dtype=float),
            donor_ids=tuple(donors["unit_id"].astype(str)),
            support_feature_indices=tuple(range(len(training))),
            target_static=treated[list(static)].to_numpy(dtype=float),
            donor_static=donors[list(static)].to_numpy(dtype=float),
            target_holdout=treated[list(holdout)].to_numpy(dtype=float),
            donor_holdout=donors[list(holdout)].to_numpy(dtype=float),
        )
        runtime = TorchRuntime(RuntimeConfig(device=device, seed=20260723))
        match = fit_matching(
            matching_input,
            config=MatchingConfig(candidates=5, placebo_sample=200),
            runtime=runtime,
        )
        report("matching_complete", quality_passed=bool(match.quality_passed))
        if not match.quality_passed:
            record = _base_record(
                target,
                families,
                failure_status,
                f"{scope}:preonly_placebo_quality_gate_failed",
            )
            add_scope_metadata(record)
            report("completed", status=failure_status, failure_reason=record["failure_reason"])
            return ControlDesignResult(pd.DataFrame([record]), None, None, match)
        control = donors.iloc[match.selected_index]
        balance, balance_summary = _feature_balance(
            frame, match.selected_index, tuple([*training, *static, *holdout])
        )
        thresholds = match.placebo_thresholds or {}
        record = _base_record(target, families, "matched", None)
        record.update(
            {
                "selected_method": "python_gpu_M5_static_refine",
                "donor_scope": scope,
                "control_city_key": str(control["city_key"]),
                "control_grid_id": str(control["grid_id"]),
                "control_unit_key": str(control["unit_id"]),
                "candidate_count": len(donors),
                "candidate_city_count": int(donors["city_key"].nunique()),
                "training_feature_count": len(training),
                "holdout_feature_count": len(holdout),
                "active_static_features": "+".join(static),
                "inactive_static_features": "+".join(
                    feature for feature in static_candidates if feature not in static
                ),
                "training_distance": match.training_distance,
                "holdout_rms_standardized_gap": match.holdout_rms_standardized_gap,
                "holdout_max_abs_standardized_gap": match.holdout_max_abs_standardized_gap,
                "training_distance_threshold": thresholds.get("training_distance"),
                "holdout_rms_threshold": thresholds.get("holdout_rms_standardized_gap"),
                "holdout_max_abs_threshold": thresholds.get("holdout_max_abs_standardized_gap"),
            }
        )
        add_scope_metadata(record)
        report("completed", status="matched")
        return ControlDesignResult(pd.DataFrame([record]), balance, balance_summary, match)
    except Exception as error:
        record = _base_record(target, families, failure_status, f"{scope}:{error}")
        add_scope_metadata(record)
        report("completed", status=failure_status, failure_reason=record["failure_reason"])
        return ControlDesignResult(pd.DataFrame([record]), None, None, None)


def write_control_design(result: ControlDesignResult, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "control_record.csv.tmp"
    result.record.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(output / "control_record.csv")
    if result.feature_balance is not None:
        temporary_parquet = output / "feature_balance.parquet.tmp"
        result.feature_balance.to_parquet(temporary_parquet, index=False, compression="zstd")
        temporary_parquet.replace(output / "feature_balance.parquet")
    if result.feature_balance_summary is not None:
        result.feature_balance_summary.to_csv(
            output / "feature_balance_summary.csv", index=False, encoding="utf-8-sig"
        )
    return output / "control_record.csv"
