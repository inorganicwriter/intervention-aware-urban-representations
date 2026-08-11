"""Pure helpers for the canonical source-preserving housing panel.

The panel contract deliberately uses only observed price, location, and time for
the causal outcome.  Listing/transaction labels are retained for provenance
and lifecycle de-duplication, not as admission barriers.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

OBSERVATION_COLUMNS = [
    "observation_id",
    "source_record_id",
    "source_id",
    "city_key",
    "community_id",
    "community_name",
    "price_stage",
    "observation_type",
    "time_precision",
    "observed_date",
    "observed_month",
    "observed_quarter",
    "observed_year",
    "price_cny_m2",
    "longitude_wgs84",
    "latitude_wgs84",
    "coordinate_source",
    "location_confidence",
    "grid_id",
    "raw_quality_flags",
    "duplicate_class",
    "canonical_for_aggregation",
    "analysis_eligible_month",
    "analysis_eligible_quarter",
    "analysis_eligible_year",
]


def join_flags(*values: object) -> str:
    """Join semicolon-delimited flags deterministically without duplicates."""
    flags: set[str] = set()
    for value in values:
        if value is None or pd.isna(value):
            continue
        flags.update(part.strip() for part in str(value).split(";") if part.strip())
    return ";".join(sorted(flags))


def attach_time_fields(
    frame: pd.DataFrame,
    date_column: str,
    precision: str,
) -> pd.DataFrame:
    """Attach real temporal keys without inventing a month for annual data."""
    result = frame.copy()
    dates = pd.to_datetime(result[date_column], errors="coerce")
    result["observed_date"] = dates
    result["time_precision"] = precision
    result["observed_year"] = dates.dt.year.astype("Int16")
    if precision in {"day", "month", "snapshot_month"}:
        result["observed_month"] = dates.dt.to_period("M").dt.to_timestamp()
        result["observed_quarter"] = dates.dt.to_period("Q").astype("string")
    elif precision == "quarter":
        result["observed_month"] = pd.NaT
        result["observed_quarter"] = dates.dt.to_period("Q").astype("string")
    else:
        result["observed_month"] = pd.NaT
        result["observed_quarter"] = pd.Series(pd.NA, index=result.index, dtype="string")
    return result


def finalize_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize v2 dtypes and calculate eligibility flags."""
    result = frame.copy()
    for column in OBSERVATION_COLUMNS:
        if column not in result:
            result[column] = pd.NA

    result["price_cny_m2"] = pd.to_numeric(result["price_cny_m2"], errors="coerce")
    result["longitude_wgs84"] = pd.to_numeric(result["longitude_wgs84"], errors="coerce")
    result["latitude_wgs84"] = pd.to_numeric(result["latitude_wgs84"], errors="coerce")
    result["observed_date"] = pd.to_datetime(result["observed_date"], errors="coerce")
    result["observed_month"] = pd.to_datetime(result["observed_month"], errors="coerce")
    result["observed_year"] = pd.to_numeric(result["observed_year"], errors="coerce").astype(
        "Int16"
    )
    result["canonical_for_aggregation"] = (
        result["canonical_for_aggregation"].fillna(False).astype(bool)
    )

    price_valid = np.isfinite(result["price_cny_m2"]) & result["price_cny_m2"].gt(0)
    located = result["grid_id"].fillna("").astype(str).ne("")
    canonical = result["canonical_for_aggregation"]
    result["analysis_eligible_month"] = (
        price_valid & located & canonical & result["observed_month"].notna()
    )
    result["analysis_eligible_quarter"] = (
        price_valid & located & canonical & result["observed_quarter"].fillna("").ne("")
    )
    result["analysis_eligible_year"] = (
        price_valid & located & canonical & result["observed_year"].notna()
    )

    text_columns = [
        "observation_id",
        "source_record_id",
        "source_id",
        "city_key",
        "community_id",
        "community_name",
        "price_stage",
        "observation_type",
        "time_precision",
        "observed_quarter",
        "coordinate_source",
        "location_confidence",
        "grid_id",
        "raw_quality_flags",
        "duplicate_class",
    ]
    for column in text_columns:
        result[column] = result[column].fillna("").astype("string")
    return result[OBSERVATION_COLUMNS]


def source_balanced_panel(
    observations: pd.DataFrame,
    period_columns: Iterable[str],
    eligibility_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate observations first within source, then equally across sources.

    The primary value is the median of source-specific log-price medians.  A
    source with many scraped listings therefore cannot mechanically dominate a
    source with fewer completed transactions.
    """
    period_columns = list(period_columns)
    eligible = observations[observations[eligibility_column]].copy()
    if eligible.empty:
        return pd.DataFrame(), pd.DataFrame()
    eligible["log_price"] = np.log(eligible["price_cny_m2"])
    eligible["_is_listing"] = eligible["price_stage"].eq("listing").astype("int32")
    eligible["_is_transaction"] = eligible["price_stage"].eq("transaction").astype("int32")
    eligible["_is_platform_estimate"] = (
        eligible["price_stage"].eq("platform_estimate").astype("int32")
    )
    source_keys = ["city_key", "grid_id", *period_columns, "source_id"]
    source_group = eligible.groupby(source_keys, dropna=False, sort=False)
    source = source_group.agg(
        source_log_price_median=("log_price", "median"),
        source_price_median_cny_m2=("price_cny_m2", "median"),
        source_price_mean_cny_m2=("price_cny_m2", "mean"),
        n_observations=("observation_id", "nunique"),
        n_listing=("_is_listing", "sum"),
        n_transaction=("_is_transaction", "sum"),
        n_platform_estimate=("_is_platform_estimate", "sum"),
    ).reset_index()
    quantiles = source_group["price_cny_m2"].quantile([0.25, 0.75]).unstack(level=-1).reset_index()
    quantiles = quantiles.rename(
        columns={0.25: "source_price_p25_cny_m2", 0.75: "source_price_p75_cny_m2"}
    )
    source = source.merge(quantiles, on=source_keys, how="left", validate="one_to_one")

    raw = eligible.groupby(
        ["city_key", "grid_id", *period_columns], as_index=False, dropna=False, sort=False
    ).agg(
        log_price_raw_median=("log_price", "median"),
        price_raw_median_cny_m2=("price_cny_m2", "median"),
        price_raw_mean_cny_m2=("price_cny_m2", "mean"),
        n_observations=("observation_id", "nunique"),
        n_listing=("_is_listing", "sum"),
        n_transaction=("_is_transaction", "sum"),
        n_platform_estimate=("_is_platform_estimate", "sum"),
        n_sources=("source_id", "nunique"),
    )
    balanced = source.groupby(
        ["city_key", "grid_id", *period_columns], as_index=False, dropna=False, sort=False
    ).agg(
        log_price_source_balanced=("source_log_price_median", "median"),
        source_price_dispersion=("source_log_price_median", "std"),
    )
    panel = raw.merge(
        balanced, on=["city_key", "grid_id", *period_columns], how="left", validate="one_to_one"
    )
    panel["price_source_balanced_cny_m2"] = np.exp(panel["log_price_source_balanced"])
    panel["source_price_dispersion"] = panel["source_price_dispersion"].fillna(0.0)
    return source.sort_values(source_keys).reset_index(drop=True), panel.sort_values(
        ["city_key", "grid_id", *period_columns]
    ).reset_index(drop=True)
