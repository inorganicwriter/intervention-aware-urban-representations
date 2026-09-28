"""Build leakage-safe, city-split model inputs from a Response Artifact."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from urban_intervention.data.paths import (
    DATA_ROOT,
    housing_annual_path,
    poi_annual_path,
    population_data_path,
    sentinel2_data_path,
    viirs_annual_path,
)
from urban_intervention.utils import sha256_file

from .response_artifact import (
    aggregate_file_fingerprint,
    git_state,
    require_reproducible_code_state,
    runtime_versions,
)

IDENTITY_COLUMNS = [
    "treatment_order",
    "city_key",
    "grid_id",
    "station_event_id",
    "opening_month",
]
DEFAULT_LAGS = (1, 2, 3)
MODALITY_FEATURES: dict[str, tuple[str, ...]] = {
    "housing": ("housing_log_price",),
    "poi": (
        "poi_count_log",
        "poi_category_entropy",
        "poi_commercial_share",
        "poi_transport_access_log",
    ),
    "viirs": ("viirs_avg_asinh",),
    "population": ("population_log",),
    "sentinel2": ("sentinel2_ndvi", "sentinel2_ndbi"),
}


def validate_response_treatment_binding(
    response_manifest: dict[str, object],
    response: pd.DataFrame,
    treatments: pd.DataFrame,
    treatments_path: Path,
    strict_production: bool,
) -> str | None:
    missing_treatments = set(IDENTITY_COLUMNS) - set(treatments.columns)
    missing_response = set(IDENTITY_COLUMNS) - set(response.columns)
    if missing_treatments:
        raise ValueError(f"Treatment list lacks identity columns: {sorted(missing_treatments)}")
    if missing_response:
        raise ValueError(f"Response Artifact lacks treatment identity: {sorted(missing_response)}")
    if treatments["treatment_order"].duplicated().any():
        raise ValueError("Treatment list has duplicate treatment_order")
    if strict_production and len(treatments) != 5_048:
        raise ValueError("Production model inputs require the immutable 5,048 treatments")

    source_files = response_manifest.get("source_files") or {}
    treatment_source = source_files.get("treatments") if isinstance(source_files, dict) else None
    bound_hash = treatment_source.get("sha256") if isinstance(treatment_source, dict) else None
    if strict_production and not bound_hash:
        raise ValueError("Strict Response Artifact manifest lacks its treatment-list hash")
    actual_hash = sha256_file(treatments_path)
    if bound_hash and str(bound_hash) != actual_hash:
        raise ValueError("Treatment file hash disagrees with the Response Artifact manifest")

    response_identity = response[IDENTITY_COLUMNS].drop_duplicates()
    if response_identity["treatment_order"].duplicated().any():
        raise ValueError("Response Artifact maps one treatment_order to multiple identities")
    treatment_identity = treatments[IDENTITY_COLUMNS].copy()
    joined = treatment_identity.merge(
        response_identity,
        on="treatment_order",
        how="outer",
        suffixes=("__treatment", "__response"),
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Treatment orders differ between treatment list and Response Artifact")
    for column in ("city_key", "grid_id", "station_event_id"):
        left = joined[f"{column}__treatment"].astype("string")
        right = joined[f"{column}__response"].astype("string")
        if not left.eq(right).fillna(False).all():
            raise ValueError(f"Treatment identity column {column} disagrees with Response Artifact")
    left_month = joined["opening_month__treatment"].map(
        lambda value: str(pd.Period(str(value)[:7], freq="M"))
    )
    right_month = joined["opening_month__response"].map(
        lambda value: str(pd.Period(str(value)[:7], freq="M"))
    )
    if not left_month.eq(right_month).all():
        raise ValueError("Treatment opening_month disagrees with Response Artifact")
    return str(bound_hash) if bound_hash else None


def deterministic_city_splits(
    cities: Iterable[str],
    seed: str = "mit-urban-v1",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, str]:
    cities = sorted(set(map(str, cities)))
    if len(cities) < 3:
        raise ValueError("At least three cities are required for city-held-out splits")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions leave no test cities")
    ranked = sorted(
        cities,
        key=lambda city: hashlib.sha256(f"{seed}\0{city}".encode()).hexdigest(),
    )
    n = len(ranked)
    train_n = max(1, min(n - 2, round(n * train_fraction)))
    validation_n = max(1, min(n - train_n - 1, round(n * validation_fraction)))
    result: dict[str, str] = {}
    for city in ranked[:train_n]:
        result[city] = "train"
    for city in ranked[train_n : train_n + validation_n]:
        result[city] = "validation"
    for city in ranked[train_n + validation_n :]:
        result[city] = "test"
    return result


def _read_if_exists(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path, columns=columns)


def _annual_city_features(root: Path, city: str) -> tuple[pd.DataFrame, list[Path]]:
    parts: list[pd.DataFrame] = []
    used: list[Path] = []

    def _resolve(absolute: Path) -> Path:
        rel = absolute.relative_to(DATA_ROOT)
        return root / "data" / rel

    housing_path = _resolve(housing_annual_path(city))
    housing = _read_if_exists(housing_path, ["city_key", "grid_id", "year", "housing_log_price"])
    if not housing.empty:
        parts.append(housing)
        used.append(housing_path)

    poi_path = _resolve(poi_annual_path(city))
    poi = _read_if_exists(
        poi_path,
        [
            "city",
            "grid_id",
            "year",
            "poi_count",
            "poi_category_entropy",
            "poi_commercial_share",
            "poi_transport_access_count",
        ],
    )
    if not poi.empty:
        poi = poi.rename(columns={"city": "city_key"})
        poi["poi_count_log"] = np.log1p(poi["poi_count"].clip(lower=0))
        poi["poi_transport_access_log"] = np.log1p(poi["poi_transport_access_count"].clip(lower=0))
        parts.append(
            poi[
                [
                    "city_key",
                    "grid_id",
                    "year",
                    "poi_count_log",
                    "poi_category_entropy",
                    "poi_commercial_share",
                    "poi_transport_access_log",
                ]
            ]
        )
        used.append(poi_path)

    viirs_path = _resolve(viirs_annual_path(city))
    viirs = _read_if_exists(viirs_path, ["city_key", "grid_id", "year", "avg_rad"])
    if not viirs.empty:
        viirs["viirs_avg_asinh"] = np.arcsinh(viirs["avg_rad"])
        parts.append(viirs[["city_key", "grid_id", "year", "viirs_avg_asinh"]])
        used.append(viirs_path)

    population_path = _resolve(population_data_path(city))
    population = _read_if_exists(population_path, ["city", "grid_id", "year", "pop_count"])
    if not population.empty:
        population = population.rename(columns={"city": "city_key"})
        population["population_log"] = np.log1p(population["pop_count"].clip(lower=0))
        population = population.groupby(["city_key", "grid_id", "year"], as_index=False)[
            "population_log"
        ].mean()
        parts.append(population)
        used.append(population_path)

    sentinel_path = _resolve(sentinel2_data_path(city))
    sentinel = _read_if_exists(sentinel_path, ["city", "grid_id", "year", "NDVI", "NDBI"])
    if not sentinel.empty:
        sentinel = sentinel.rename(
            columns={
                "city": "city_key",
                "NDVI": "sentinel2_ndvi",
                "NDBI": "sentinel2_ndbi",
            }
        )
        # GEE may contain several geographic samples in one 500m grid. The model
        # feature is the grid-year mean; source multiplicity is retained as a count.
        sentinel = sentinel.groupby(["city_key", "grid_id", "year"], as_index=False).agg(
            sentinel2_ndvi=("sentinel2_ndvi", "mean"),
            sentinel2_ndbi=("sentinel2_ndbi", "mean"),
            sentinel2_source_points=("sentinel2_ndvi", "size"),
        )
        parts.append(sentinel)
        used.append(sentinel_path)

    if not parts:
        return pd.DataFrame(columns=["city_key", "grid_id", "year"]), used
    combined = parts[0]
    for part in parts[1:]:
        combined = combined.merge(
            part,
            on=["city_key", "grid_id", "year"],
            how="outer",
            validate="one_to_one",
        )
    if combined.duplicated(["city_key", "grid_id", "year"]).any():
        raise ValueError(f"Annual feature key is not unique after aggregation: {city}")
    return combined, used


def build_pretreatment_features(
    treatments: pd.DataFrame,
    root: Path,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    strict_sources: bool = False,
) -> tuple[pd.DataFrame, list[Path]]:
    required = set(IDENTITY_COLUMNS)
    missing = required - set(treatments)
    if missing:
        raise ValueError(f"Treatment list lacks columns: {sorted(missing)}")
    # Positive lags are the leak barrier: rows are selected with
    # lag = opening_year - year in lags, so year is always strictly before
    # opening.  A zero/negative lag would admit opening or post-treatment
    # years into the feature matrix; reject such configuration loudly instead
    # of silently dropping the guard.
    if any(not isinstance(lag, (int, np.integer)) or int(lag) <= 0 for lag in lags):
        raise ValueError(f"lags must be strictly positive integers, got {tuple(lags)}")
    target = treatments[IDENTITY_COLUMNS].copy()
    target["opening_year"] = pd.to_numeric(
        target["opening_month"].astype(str).str[:4], errors="coerce"
    )
    if target["opening_year"].isna().any():
        raise ValueError("Treatment list contains rows without a valid opening_month")
    target["opening_year"] = target["opening_year"].astype(int)
    long_parts: list[pd.DataFrame] = []
    used_paths: list[Path] = []
    for city, city_targets in target.groupby("city_key", sort=True):
        features, paths = _annual_city_features(root, str(city))
        used_paths.extend(paths)
        if features.empty:
            if strict_sources:
                raise FileNotFoundError(f"No admitted feature source for {city}")
            continue
        joined = city_targets[["treatment_order", "city_key", "grid_id", "opening_year"]].merge(
            features, on=["city_key", "grid_id"], how="left"
        )
        # Selection happens purely through the (positive) lags above: opening
        # and post-treatment years present in the annual source panels are
        # deliberately excluded by the lag filter.
        joined["lag"] = joined["opening_year"] - joined["year"]
        joined = joined.loc[joined["lag"].isin(lags)].copy()
        long_parts.append(joined)
    long = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()

    wide = target.copy()
    feature_names = sorted({name for values in MODALITY_FEATURES.values() for name in values})
    if not long.empty:
        for feature in feature_names:
            if feature not in long:
                continue
            pivot = long.pivot(index="treatment_order", columns="lag", values=feature).rename(
                columns={lag: f"{feature}__lag{int(lag)}" for lag in lags}
            )
            wide = wide.merge(pivot, left_on="treatment_order", right_index=True, how="left")
        if "sentinel2_source_points" in long:
            points = long.pivot(
                index="treatment_order", columns="lag", values="sentinel2_source_points"
            ).rename(columns={lag: f"sentinel2_source_points__lag{int(lag)}" for lag in lags})
            wide = wide.merge(points, left_on="treatment_order", right_index=True, how="left")

    for feature in feature_names:
        for lag in lags:
            column = f"{feature}__lag{lag}"
            if column not in wide:
                wide[column] = np.nan
    for modality, features in MODALITY_FEATURES.items():
        columns = [f"{feature}__lag{lag}" for feature in features for lag in lags]
        wide[f"{modality}_available"] = wide[columns].notna().any(axis=1)
    return wide, sorted(set(used_paths))


def attach_streetview_assets(
    features: pd.DataFrame, streetview_index: Path | None
) -> tuple[pd.DataFrame, list[Path]]:
    features = features.copy()
    if streetview_index is None:
        features["streetview_assets"] = "[]"
        features["streetview_available"] = False
        return features, []
    index = (
        pd.read_parquet(streetview_index)
        if streetview_index.suffix == ".parquet"
        else pd.read_csv(streetview_index)
    )
    required = {"city_key", "grid_id", "capture_date", "asset_path"}
    missing = required - set(index)
    if missing:
        raise ValueError(f"Street-view index lacks columns: {sorted(missing)}")
    index["capture_date"] = pd.to_datetime(index["capture_date"], errors="coerce")
    opening = features[["treatment_order", "city_key", "grid_id", "opening_month"]].copy()
    opening["opening_date"] = pd.to_datetime(opening["opening_month"].astype(str) + "-01")
    eligible = opening.merge(index, on=["city_key", "grid_id"], how="left")
    eligible = eligible.loc[eligible["capture_date"] < eligible["opening_date"]]
    assets = (
        eligible.sort_values("capture_date")
        .groupby("treatment_order")["asset_path"]
        .agg(lambda values: json.dumps(list(map(str, values)), ensure_ascii=False))
    )
    features["streetview_assets"] = features["treatment_order"].map(assets).fillna("[]")
    features["streetview_available"] = features["streetview_assets"].ne("[]")
    return features, [streetview_index]


def normalize_from_train(
    features: pd.DataFrame, split_map: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, object]]:
    result = features.copy()
    result["split"] = result["city_key"].map(split_map)
    if result["split"].isna().any():
        raise ValueError("Every city must have an assigned split")
    feature_columns = sorted(
        column
        for column in result.columns
        if "__lag" in column and not column.startswith("sentinel2_source_points")
    )
    train = result["split"].eq("train")
    parameters: dict[str, object] = {}
    for column in feature_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        train_values = values[train & values.notna()]
        mean = float(train_values.mean()) if len(train_values) else 0.0
        sd = float(train_values.std(ddof=0)) if len(train_values) else 1.0
        usable = bool(len(train_values) and np.isfinite(sd) and sd > np.finfo(float).eps)
        if usable:
            result[f"z__{column}"] = (values - mean) / sd
        else:
            # No training-city observations: emitting raw-scale values here
            # would mix unstandardized columns into the model inputs.  The
            # column is published as missing instead (modality availability
            # already reflects the source coverage).
            result[f"z__{column}"] = np.nan
        parameters[column] = {
            "mean": mean,
            "sd": sd if usable else None,
            "train_observations": int(len(train_values)),
            "usable": usable,
        }
    return result, parameters


def _best_quality_grade(targets: pd.DataFrame) -> pd.DataFrame:
    """Return one quality grade per treatment, preferring available labels.

    Same-city-first ordering (matched > GSC > MC within each scope; any
    same-city path ranks above any cross-city path).  ``mc_*_pass`` requires
    more than one pre-treatment support period and is stronger evidence than
    ``mc_*_minimal_pre_support`` (min.T0=1), so the pass grade ranks above the
    minimal-support grade.  When the Response Artifact provides
    ``label_grade``, unavailable labels are never selected.
    """
    grade_rank = {
        "matched_same_city_pass": 8,
        "gsc_same_city_pass": 7,
        "mc_same_city_pass": 6,
        "mc_same_city_minimal_pre_support": 5,
        "matched_cross_city_pass": 4,
        "gsc_cross_city_pass": 3,
        "mc_cross_city_pass": 2,
        "mc_cross_city_minimal_pre_support": 1,
        "pending": 0,
    }
    grade_column = "label_grade" if "label_grade" in targets.columns else "quality_grade"
    ranked = targets.copy()
    ranked["_grade_rank"] = ranked[grade_column].map(grade_rank).fillna(0).astype(int)
    best = (
        ranked.sort_values("_grade_rank", ascending=False)
        .groupby("treatment_order", as_index=False)
        .first()
    )
    return best[["treatment_order", grade_column]].rename(
        columns={grade_column: "quality_grade"}
    )


KNOWN_FAMILIES = ("housing", "viirs", "poi", "population")

CALIBRATION_VERSION = "rev3-ordinal-2026-09"
DESIGN_GAMMA = 1.0
PRECISION_ALPHA = 0.5
DESIGN_NEUTRAL_PERCENTILE = 0.5
WEIGHT_WINSOR_UPPER = 0.99
MIN_DESIGN_STRATUM = 2


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def response_label_weights(targets: pd.DataFrame) -> pd.DataFrame:
    """Attach Rev3 label reliability weights to every label row.

    Structure (deliberately ordinal, not a calibrated variance):

        W = A * exp(-gamma * P_design) * (precision)^alpha

    - ``A`` (``eligible``): zero pre-treatment anchors, a violated matched
      holdout boundary (``max(holdout ratios) > 1``) or unavailable labels get
      weight 0; every eligibility rule is a methodological boundary, not an
      arbitrary quality cutoff.
    - ``P_design`` (``design_unreliability``): within-(method, outcome)
      percentile of the design diagnostic, 0 = best.  Matched labels use
      ``max(holdout_rms_ratio, holdout_max_ratio)``; GSC/MC labels use the
      pre-treatment gap deviation
      ``sqrt(max(pre_rmspe^2 - pre_mean_effect^2, 0))``.  Small strata fall
      back to (method, family); missing diagnostics receive the neutral 0.5.
    - ``precision``: inverse sampling variance ``1/se^2`` winsorized at the
      1%/99% quantiles and mean-normalized **within method and outcome**;
      labels without a standard error (matching) receive exactly 1 so that a
      missing uncertainty estimate never earns a bonus.  ``alpha = 0.5``
      tempers the multiplier (sensitivity: 0, 0.5, 1).
    - Final weights are winsorized at the 99th percentile and normalized to
      mean one within (method, family) among eligible labels.
    """
    frame = targets.copy()
    method = frame.get("method", pd.Series("", index=frame.index)).astype(str)
    matched = method.str.startswith("frozen_matched_change")
    family = frame["outcome_family"].astype(str)
    outcome = frame.get("outcome", pd.Series("", index=frame.index)).astype(str)

    label_available = (
        frame.get("label_available", pd.Series(True, index=frame.index))
        .fillna(False)
        .astype(bool)
    )
    anchors = _numeric_column(frame, "pre_observed_periods")
    holdout_rms = _numeric_column(frame, "holdout_rms_ratio")
    holdout_max = _numeric_column(frame, "holdout_max_ratio")
    matched_holdout = pd.concat([holdout_rms, holdout_max], axis=1).max(axis=1)

    eligible = label_available & ~anchors.eq(0).fillna(False)
    eligible = eligible & ~(matched & matched_holdout.gt(1.0).fillna(False))

    pre_rmspe = _numeric_column(frame, "pre_rmspe")
    pre_mean = _numeric_column(frame, "pre_mean_effect")
    gap_deviation = pd.Series(
        np.sqrt(np.maximum(pre_rmspe**2 - pre_mean**2, 0.0)), index=frame.index
    )
    design_distance = pd.Series(np.nan, index=frame.index, dtype=float)
    design_distance = design_distance.where(~matched, matched_holdout)
    design_distance = design_distance.where(matched, gap_deviation)

    design_stratum = method + "::" + outcome
    family_stratum = method + "::" + family
    design_percentile = design_distance.groupby(design_stratum).rank(pct=True)
    fallback_percentile = design_distance.groupby(family_stratum).rank(pct=True)
    design_sizes = design_distance.groupby(design_stratum).transform("size")
    family_sizes = design_distance.groupby(family_stratum).transform("size")
    design_percentile = design_percentile.where(
        design_sizes.ge(MIN_DESIGN_STRATUM), fallback_percentile
    )
    design_percentile = design_percentile.where(
        family_sizes.ge(MIN_DESIGN_STRATUM), DESIGN_NEUTRAL_PERCENTILE
    )
    design_percentile = design_percentile.where(
        design_distance.notna(), DESIGN_NEUTRAL_PERCENTILE
    )
    design_weight = np.exp(-DESIGN_GAMMA * design_percentile)

    standard_error = _numeric_column(frame, "standard_error")
    inverse_variance = 1.0 / (standard_error**2 + 1e-12)
    lower = inverse_variance.groupby(design_stratum).transform(
        lambda values: values.quantile(0.01)
    )
    upper = inverse_variance.groupby(design_stratum).transform(
        lambda values: values.quantile(0.99)
    )
    clipped = inverse_variance.clip(lower=lower, upper=upper)
    mean_precision = clipped.groupby(design_stratum).transform("mean")
    normalized_precision = clipped / mean_precision
    use_precision = (
        standard_error.notna() & ~matched & mean_precision.gt(0).fillna(False)
    )
    precision_weight = normalized_precision.pow(PRECISION_ALPHA).where(
        use_precision, 1.0
    )

    raw = (design_weight * precision_weight).where(eligible, 0.0)
    cap = raw.groupby(family_stratum).transform(
        lambda values: values.quantile(WEIGHT_WINSOR_UPPER)
    )
    capped = raw.clip(upper=cap)
    eligible_mean = capped.where(eligible).groupby(family_stratum).transform("mean")
    final = capped / eligible_mean.where(eligible_mean.gt(0), 1.0)
    final = final.fillna(0.0).where(eligible, 0.0)

    frame["eligible"] = eligible.astype(bool)
    frame["design_unreliability"] = design_percentile.astype(float)
    frame["design_weight"] = design_weight.astype(float)
    frame["has_precision_weight"] = use_precision.astype(bool)
    frame["precision_weight"] = precision_weight.astype(float)
    frame["weight_raw"] = raw.astype(float)
    frame["response_weight"] = final.astype(float)
    frame["calibration_version"] = CALIBRATION_VERSION
    return frame


def _response_weights(targets: pd.DataFrame) -> pd.DataFrame:
    """Return one conservative response weight per treatment."""
    weighted = response_label_weights(targets)
    if "label_available" in weighted.columns:
        weighted = weighted.loc[weighted["label_available"].fillna(False)]
    return weighted.groupby("treatment_order", as_index=False)["response_weight"].min()


def publish_pretraining_dataset(
    response_release: Path,
    treatments_path: Path,
    project_root: Path,
    output_root: Path,
    dataset_id: str | None = None,
    split_seed: str = "mit-urban-v1",
    min_modalities: int = 2,
    streetview_index: Path | None = None,
    strict_production: bool = True,
    scope_view: str = "same_city",
    families: tuple[str, ...] | None = None,
) -> Path:
    response_manifest_path = response_release / "manifest.json"
    response_path = response_release / "response_artifact.parquet"
    response_manifest = json.loads(response_manifest_path.read_text(encoding="utf-8"))
    if strict_production and not bool(response_manifest.get("strict_production")):
        raise ValueError("Production model inputs require a strict production Response Artifact")
    if sha256_file(response_path) != response_manifest.get("artifact", {}).get("sha256", ""):
        raise ValueError("Response Artifact hash disagrees with its manifest")

    treatments = pd.read_parquet(treatments_path)
    targets = pd.read_parquet(response_path)
    bound_treatment_hash = validate_response_treatment_binding(
        response_manifest, targets, treatments, treatments_path, strict_production
    )
    if strict_production:
        response_code = response_manifest.get("code")
        if not isinstance(response_code, dict):
            raise ValueError("Strict Response Artifact manifest lacks structured code provenance")
        require_reproducible_code_state(response_code)
    state = git_state(project_root)
    if strict_production:
        require_reproducible_code_state(state)
    if families is not None:
        selected = tuple(dict.fromkeys(str(family) for family in families))
        unknown = sorted(set(selected) - set(KNOWN_FAMILIES))
        if unknown:
            raise ValueError(f"unknown outcome families: {unknown}")
        if not selected:
            raise ValueError("--families must select at least one outcome family")
        targets = targets.loc[targets["outcome_family"].astype(str).isin(selected)].copy()
        if targets.empty:
            raise ValueError(
                f"response artifact has no rows for families: {list(selected)}"
            )
    targets = response_label_weights(targets)
    features, source_paths = build_pretreatment_features(
        treatments, project_root, strict_sources=strict_production
    )
    features, asset_paths = attach_streetview_assets(features, streetview_index)
    source_paths.extend(asset_paths)
    split_map = deterministic_city_splits(features["city_key"], seed=split_seed)
    features, normalization = normalize_from_train(features, split_map)
    modality_columns = [f"{name}_available" for name in MODALITY_FEATURES] + [
        "streetview_available"
    ]
    features["available_modality_count"] = features[modality_columns].sum(axis=1)
    features["feature_training_mask"] = features["available_modality_count"] >= int(min_modalities)

    targets = targets.merge(
        features[IDENTITY_COLUMNS + ["split", "feature_training_mask"]],
        on=IDENTITY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    targets["final_training_mask"] = targets["training_mask"].fillna(False) & targets[
        "feature_training_mask"
    ].fillna(False)
    if scope_view in {"same_city", "cross_city"}:
        if "main_spec" not in targets.columns:
            if strict_production:
                raise ValueError(
                    "Response Artifact lacks the main_spec column required for --scope-view"
                )
            # Backward-compatible partial/test artifacts predate donor-scope
            # metadata and contain only same-city fixtures.
            targets["main_spec"] = True
        in_scope = targets["main_spec"].astype("boolean").fillna(False)
        if scope_view == "cross_city":
            in_scope = ~in_scope
        targets["final_training_mask"] = targets["final_training_mask"] & in_scope
    unit_level = targets.groupby("treatment_order", as_index=False).agg(
        training_mask=("training_mask", "max"),
        final_training_mask=("final_training_mask", "max"),
    )
    unit_weights = _response_weights(targets)
    best_grade = _best_quality_grade(targets)
    sample_index = (
        features[["treatment_order", "split", "feature_training_mask"]]
        .merge(unit_level, on="treatment_order", how="left")
        .merge(unit_weights, on="treatment_order", how="left")
        .merge(best_grade, on="treatment_order", how="left")
    )
    sample_index["training_mask"] = sample_index["training_mask"].fillna(False)
    sample_index["final_training_mask"] = sample_index["final_training_mask"].fillna(False)
    sample_index["quality_grade"] = sample_index["quality_grade"].fillna("pending")
    sample_index["response_weight"] = (
        pd.to_numeric(sample_index["response_weight"], errors="coerce").fillna(0.0)
    )

    dataset_id = dataset_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = output_root / dataset_id
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite model-input release: {destination}")
    staging = output_root / f".{dataset_id}.tmp-{uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        features.to_parquet(staging / "unit_features.parquet", index=False, compression="zstd")
        targets.to_parquet(staging / "response_targets.parquet", index=False, compression="zstd")
        sample_index.to_parquet(staging / "sample_index.parquet", index=False, compression="zstd")
        (staging / "normalization.json").write_text(
            json.dumps(normalization, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        split_manifest = {
            split: sorted(city for city, assigned in split_map.items() if assigned == split)
            for split in ("train", "validation", "test")
        }
        manifest = {
            "schema": "urban_intervention_pretraining_dataset",
            "dataset_id": dataset_id,
            "created_utc": datetime.now(UTC).isoformat(),
            "strict_production": strict_production,
            "response_release": str(response_release),
            "response_artifact_sha256": sha256_file(response_path),
            "treatments_sha256": sha256_file(treatments_path),
            "response_bound_treatments_sha256": bound_treatment_hash,
            "source_features": aggregate_file_fingerprint(source_paths, project_root),
            "feature_lags": list(DEFAULT_LAGS),
            "feature_timing": "opening_year minus lag; post/opening-year rows forbidden",
            "split_unit": "city_key",
            "split_seed": split_seed,
            "splits": split_manifest,
            "min_modalities": int(min_modalities),
            "families": list(families) if families is not None else None,
            "response_weight": {
                "definition": (
                    "Rev3 ordinal label reliability: eligible * exp(-gamma * design "
                    "percentile) * tempered precision, winsorized and normalized to "
                    "mean one within method x family"
                ),
                "calibration_version": CALIBRATION_VERSION,
                "gamma": DESIGN_GAMMA,
                "alpha": PRECISION_ALPHA,
                "mean": float(sample_index["response_weight"].mean()),
                "positive": int((sample_index["response_weight"] > 0).sum()),
            },
            "rows": {
                "unit_features": len(features),
                "response_targets": len(targets),
                "sample_index": len(sample_index),
                "final_training": int(sample_index["final_training_mask"].sum()),
            },
            "code": state,
            "runtime": runtime_versions(),
        }
        for name in ("unit_features.parquet", "response_targets.parquet", "sample_index.parquet"):
            outputs_map = manifest.setdefault("outputs", {})
            assert isinstance(outputs_map, dict)
            outputs_map[name] = sha256_file(staging / name)
        outputs_map = manifest.setdefault("outputs", {})
        assert isinstance(outputs_map, dict)
        outputs_map["normalization.json"] = sha256_file(staging / "normalization.json")
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        output_root.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(staging, destination)
        except OSError as exc:
            if destination.exists():
                raise FileExistsError(
                    f"Another publisher created model-input release: {destination}"
                ) from exc
            raise
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
