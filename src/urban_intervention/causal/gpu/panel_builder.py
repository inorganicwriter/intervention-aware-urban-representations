"""Python-native construction of formal GSC and MC estimation panels.

The builder mirrors the frozen timing, donor-admission, transformation and
cross-city scaling rules in ``complete_estimators_lib.R``.  It intentionally
does not use post-treatment outcomes to decide whether a donor is admitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from urban_intervention.data.paths import (
    ELIGIBLE_DONORS,
    OUTPUT_CAUSAL_LABELS_DIR,
    PANEL_HOUSING_MONTHLY_DIR,
    POI_DIR,
    POPULATION_DIR,
    PROJECT_ROOT,
    TREATMENT_UNIT_LIST,
    VIIRS_ANNUAL_DIR,
    VIIRS_MONTHLY_DIR,
    housing_annual_path,
)

Estimator = Literal["gsc", "mc"]
DonorScope = Literal["same_city", "all_city_standardized"]
Frequency = Literal["annual", "monthly"]

OUTCOMES: dict[str, tuple[str, ...]] = {
    "housing": ("housing_log_price",),
    "viirs": ("viirs_avg_asinh",),
    "population": ("population_log",),
    "poi": (
        "poi_count_log",
        "poi_category_entropy",
        "poi_commercial_share",
        "poi_transport_access_log",
    ),
}


@dataclass(frozen=True, slots=True)
class PanelBuildRequest:
    treatment_order: int
    outcome_family: str
    outcome: str
    estimator: Estimator
    donor_scope: DonorScope = "same_city"
    anticipation_months: int = 6
    price_measure: Literal["median", "hedonic"] = "median"
    max_mc_donors: int = 2000
    max_gsc_cross_city_donors: int = 50_000
    gsc_donor_sampling_seed: int = 20260823
    transaction_count_threshold: int = 1
    root: Path = PROJECT_ROOT

    def __post_init__(self) -> None:
        if self.outcome_family not in OUTCOMES:
            raise ValueError(f"unknown outcome family: {self.outcome_family}")
        if self.outcome not in OUTCOMES[self.outcome_family]:
            raise ValueError(
                f"outcome {self.outcome!r} does not belong to {self.outcome_family!r}"
            )
        if self.estimator not in {"gsc", "mc"}:
            raise ValueError("estimator must be 'gsc' or 'mc'")
        if self.donor_scope not in {"same_city", "all_city_standardized"}:
            raise ValueError("unsupported donor scope")
        if self.anticipation_months < 0:
            raise ValueError("anticipation_months must be non-negative")
        if self.max_mc_donors < 1:
            raise ValueError("max_mc_donors must be positive")
        if self.max_gsc_cross_city_donors < 20:
            raise ValueError("max_gsc_cross_city_donors must be at least 20")
        if self.transaction_count_threshold < 1:
            raise ValueError("transaction_count_threshold must be positive")


def deterministic_cross_city_gsc_sample(
    donors: pd.DataFrame,
    maximum: int,
    seed: int,
) -> pd.DataFrame:
    """Select a stable uniform donor subset without consulting outcomes."""
    if maximum < 20:
        raise ValueError("cross-city GSC donor maximum must be at least 20")
    required = {"city_key", "grid_id", "unit_id"}
    if not required.issubset(donors.columns):
        raise ValueError("donor sample lacks city_key/grid_id/unit_id")
    if len(donors) <= maximum:
        result = donors.copy()
    else:
        keys = (
            donors["unit_id"].astype(str)
            + "::seed="
            + str(int(seed))
        )
        scores = pd.util.hash_pandas_object(keys, index=False, categorize=True).to_numpy(
            dtype=np.uint64
        )
        selected = np.argpartition(scores, maximum - 1)[:maximum]
        result = donors.iloc[selected].copy()
        result["_sample_score"] = scores[selected]
        result = result.sort_values(
            ["_sample_score", "city_key", "grid_id"], kind="stable"
        ).drop(columns="_sample_score")
    return result.reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class BuiltPanel:
    panel: pd.DataFrame
    metadata: dict[str, object]


class MonthlyEventCalendar(TypedDict):
    opening_month: pd.Timestamp
    clean_pre_end: pd.Timestamp
    first_treated_month: pd.Timestamp
    pre: pd.DatetimeIndex
    post: pd.DatetimeIndex
    times: pd.DatetimeIndex
    excluded: pd.DatetimeIndex
    anticipation_months: int


def _month(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value).to_period("M").to_timestamp()
    if pd.isna(result):
        raise ValueError(f"invalid month: {value!r}")
    return result


def monthly_event_calendar(
    opening_month: object,
    *,
    lag: int = 36,
    leads: tuple[int, ...] = tuple(range(1, 25)),
    anticipation_months: int = 6,
) -> MonthlyEventCalendar:
    """Return the exact calendar used by the frozen R specification."""
    if lag < 1 or not leads or min(leads) < 1 or anticipation_months < 0:
        raise ValueError("invalid monthly event calendar arguments")
    opening = _month(opening_month)
    clean_pre_end = opening - pd.DateOffset(months=anticipation_months + 1)
    pre = pd.date_range(end=clean_pre_end, periods=lag, freq="MS")
    post = pd.DatetimeIndex([opening + pd.DateOffset(months=value) for value in leads])
    excluded = pd.date_range(clean_pre_end + pd.DateOffset(months=1), opening, freq="MS")
    return {
        "opening_month": opening,
        "clean_pre_end": clean_pre_end,
        "first_treated_month": opening + pd.DateOffset(months=1),
        "pre": pre,
        "post": post,
        "times": pre.append(post),
        "excluded": excluded,
        "anticipation_months": anticipation_months,
    }


def read_annual_outcome(root: Path, city: str, family: str) -> pd.DataFrame:
    if family == "housing":
        frame = pd.read_parquet(
            root / housing_annual_path(city).relative_to(PROJECT_ROOT),
            columns=["city_key", "grid_id", "year", "housing_log_price"],
        )
    elif family == "poi":
        frame = pd.read_parquet(
            root / POI_DIR.relative_to(PROJECT_ROOT) / f"{city}_poi_grid_yearly.parquet",
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
        frame["poi_count_log"] = np.log1p(frame["poi_count"].clip(lower=0))
        frame["poi_transport_access_log"] = np.log1p(
            frame["poi_transport_access_count"].clip(lower=0)
        )
        frame = frame[["city_key", "grid_id", "year", *OUTCOMES[family]]]
    elif family == "viirs":
        frame = pd.read_parquet(
            root
            / VIIRS_ANNUAL_DIR.relative_to(PROJECT_ROOT)
            / f"{city}_viirs_annual.parquet",
            columns=["city_key", "grid_id", "year", "avg_rad"],
        )
        frame["viirs_avg_asinh"] = np.arcsinh(frame["avg_rad"])
        frame = frame[["city_key", "grid_id", "year", "viirs_avg_asinh"]]
    else:
        frame = pd.read_parquet(
            root / POPULATION_DIR.relative_to(PROJECT_ROOT) / f"{city}_pop.parquet",
            columns=["city", "grid_id", "year", "pop_count"],
        ).rename(columns={"city": "city_key"})
        frame["population_log"] = np.log1p(frame["pop_count"].clip(lower=0))
        frame = (
            frame.groupby(["city_key", "grid_id", "year"], as_index=False, sort=False)[
                "population_log"
            ]
            .mean()
        )
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    return frame


def read_monthly_housing(
    root: Path, city: str, price_measure: Literal["median", "hedonic"]
) -> pd.DataFrame:
    if price_measure == "hedonic":
        path = (
            root
            / OUTPUT_CAUSAL_LABELS_DIR.relative_to(PROJECT_ROOT)
            / "housing_hedonic"
            / f"{city}_monthly.parquet"
        )
        frame = pd.read_parquet(
            path,
            columns=[
                "city_key",
                "grid_id",
                "observed_month",
                "adjusted_price_median",
                "n_transactions",
            ],
        )
        valid = np.isfinite(frame["adjusted_price_median"]) & (
            frame["adjusted_price_median"] > 0
        )
        frame = frame.loc[valid].copy()
        frame["housing_log_price"] = np.log(frame["adjusted_price_median"])
        frame["transaction_count"] = pd.to_numeric(
            frame["n_transactions"], errors="coerce"
        )
    else:
        path = (
            root
            / PANEL_HOUSING_MONTHLY_DIR.relative_to(PROJECT_ROOT)
            / f"{city}.parquet"
        )
        frame = pd.read_parquet(
            path,
            columns=[
                "city_key",
                "grid_id",
                "observed_month",
                "log_price_raw_median",
                "n_transaction",
            ],
        )
        frame["housing_log_price"] = pd.to_numeric(
            frame["log_price_raw_median"], errors="coerce"
        )
        frame["transaction_count"] = pd.to_numeric(
            frame["n_transaction"], errors="coerce"
        )
    frame["period"] = pd.to_datetime(frame["observed_month"]).dt.to_period("M").dt.to_timestamp()
    return frame.groupby(
        ["city_key", "grid_id", "period"], as_index=False, sort=False
    ).agg(
        housing_log_price=("housing_log_price", "median"),
        transaction_count=("transaction_count", "sum"),
    )


def read_monthly_viirs(
    root: Path,
    city: str,
    months: pd.DatetimeIndex,
    grid_ids: set[str] | None = None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    base = root / VIIRS_MONTHLY_DIR.relative_to(PROJECT_ROOT) / f"city_key={city}"
    for period in months:
        path = base / f"year={period.year}" / f"month={period.month:02d}" / "part.parquet"
        if not path.exists():
            continue
        part = pd.read_parquet(path, columns=["grid_id", "avg_rad"])
        if grid_ids is not None:
            part = part.loc[part["grid_id"].astype(str).isin(grid_ids)].copy()
            if part.empty:
                continue
        part["city_key"] = city
        part["period"] = period
        part["viirs_avg_asinh"] = np.arcsinh(part["avg_rad"])
        parts.append(part[["city_key", "grid_id", "period", "viirs_avg_asinh"]])
    if not parts:
        return pd.DataFrame(columns=["city_key", "grid_id", "period", "viirs_avg_asinh"])
    return pd.concat(parts, ignore_index=True)


def read_outcomes_for_request(
    request: PanelBuildRequest,
    cities: list[str],
    monthly_times: pd.DatetimeIndex | None = None,
    units: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Read and transform only the family needed for one formal task."""
    units_by_city: dict[str, set[str]] | None = None
    if units is not None:
        required = {"city_key", "grid_id"}
        if not required.issubset(units.columns):
            raise ValueError("outcome unit filter lacks city_key/grid_id")
        units_by_city = {
            str(city): set(group["grid_id"].astype(str))
            for city, group in units.groupby("city_key", sort=False)
        }
    parts: list[pd.DataFrame] = []
    if request.outcome_family in {"housing", "viirs"}:
        if monthly_times is None:
            raise ValueError("monthly_times are required for monthly outcomes")
        for city in cities:
            grid_ids = units_by_city.get(city, set()) if units_by_city is not None else None
            part = (
                read_monthly_housing(request.root, city, request.price_measure)
                if request.outcome_family == "housing"
                else read_monthly_viirs(
                    request.root, city, monthly_times, grid_ids=grid_ids
                )
            )
            if grid_ids is not None and request.outcome_family == "housing":
                part = part.loc[part["grid_id"].astype(str).isin(grid_ids)].copy()
            if not part.empty:
                parts.append(part)
    else:
        for city in cities:
            part = read_annual_outcome(request.root, city, request.outcome_family)
            if units_by_city is not None:
                part = part.loc[
                    part["grid_id"].astype(str).isin(units_by_city.get(city, set()))
                ].copy()
            if part.empty:
                continue
            part.rename(columns={"year": "period"}, inplace=True)
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _sample_sd(values: pd.Series) -> float:
    return float(values.std(ddof=1))


def build_estimation_panel_from_frames(
    *,
    target: pd.Series,
    donors: pd.DataFrame,
    outcomes: pd.DataFrame,
    request: PanelBuildRequest,
    pre: pd.Index,
    post: pd.Index,
    opening_period_excluded: object,
) -> BuiltPanel:
    """Build one estimator panel from normalized treatment/donor/outcome frames.

    This pure entry point is used by tests and makes every admission rule
    independently auditable without accessing the project data lake.
    """
    eligible_scope_donors = int(
        donors.attrs.get("eligible_scope_donors_before_cap", len(donors))
    )
    pre_outcome_donor_cap = str(donors.attrs.get("pre_outcome_donor_cap", "none"))
    required_target = {"city_key", "grid_id", "treatment_order"}
    if not required_target.issubset(target.index):
        raise ValueError("target lacks city_key/grid_id/treatment_order")
    required_outcomes = {"city_key", "grid_id", "period", request.outcome}
    missing = required_outcomes - set(outcomes.columns)
    if missing:
        raise ValueError(f"outcomes lack columns: {sorted(missing)}")
    times = pd.Index([*pre, *post])
    if times.has_duplicates or len(pre) == 0 or len(post) == 0:
        raise ValueError("pre/post periods must be non-empty and unique")
    selected_columns = list(required_outcomes)
    if "transaction_count" in outcomes.columns:
        selected_columns.append("transaction_count")
    elif request.outcome_family == "housing":
        raise ValueError("housing outcomes lack transaction_count")
    values = outcomes.loc[outcomes["period"].isin(times), selected_columns].copy()
    values["value"] = pd.to_numeric(values[request.outcome], errors="coerce")
    values = values.drop(columns=request.outcome)
    if request.outcome_family == "housing":
        values["transaction_count"] = pd.to_numeric(
            values["transaction_count"], errors="coerce"
        )
        supported = values["transaction_count"].ge(
            request.transaction_count_threshold
        )
        values.loc[~supported, "value"] = np.nan
    if values.duplicated(["city_key", "grid_id", "period"]).any():
        raise ValueError("outcomes contain duplicate city/grid/period rows")

    units = pd.concat(
        [
            pd.DataFrame(
                [{
                    "city_key": str(target["city_key"]),
                    "grid_id": str(target["grid_id"]),
                    "role": "treated",
                    "treatment_order": int(target["treatment_order"]),
                }]
            ),
            donors[["city_key", "grid_id"]].assign(role="donor", treatment_order=pd.NA),
        ],
        ignore_index=True,
    ).drop_duplicates(["city_key", "grid_id"], keep="first")

    pre_values = values.loc[values["period"].isin(pre) & np.isfinite(values["value"])]
    counts = (
        pre_values.groupby(["city_key", "grid_id"], as_index=False, sort=False)
        .size()
        .rename(columns={"size": "pre_finite_count"})
    )
    units = units.merge(counts, on=["city_key", "grid_id"], how="left")
    units["pre_finite_count"] = units["pre_finite_count"].fillna(0).astype(int)
    if request.estimator == "gsc":
        units = units.loc[units["pre_finite_count"].eq(len(pre))].copy()
    else:
        treated_count = units.loc[units["role"].eq("treated"), "pre_finite_count"]
        if treated_count.empty or int(treated_count.iloc[0]) < 1:
            raise ValueError("treated unit lacks enough finite pre-treatment observations for MC")
        eligible_donors = units.loc[
            units["role"].eq("donor") & units["pre_finite_count"].ge(1)
        ].copy()
        donor_capped = len(eligible_donors) > request.max_mc_donors
        eligible_donors = eligible_donors.sort_values(
            ["pre_finite_count", "city_key", "grid_id"],
            ascending=[False, True, True],
            kind="stable",
        ).head(request.max_mc_donors)
        units = pd.concat(
            [units.loc[units["role"].eq("treated")], eligible_donors],
            ignore_index=True,
        )
    if not units["role"].eq("treated").any():
        raise ValueError("no treated unit has the required clean pre-treatment path")
    if not units["role"].eq("donor").any():
        raise ValueError("no donor has the required clean pre-treatment path")
    if request.estimator == "gsc":
        target_post = values.loc[
            values["city_key"].eq(str(target["city_key"]))
            & values["grid_id"].eq(str(target["grid_id"]))
            & values["period"].isin(post)
            & np.isfinite(values["value"]),
            "period",
        ].nunique()
        if target_post != len(post):
            raise ValueError("no complete post-treatment outcome for treated unit")
        donor_capped = False

    role_order = pd.Categorical(units["role"], categories=["donor", "treated"], ordered=True)
    units = (
        units.assign(_role_order=role_order)
        .sort_values(["_role_order", "grid_id", "city_key"], kind="stable")
        .drop(columns="_role_order")
        .reset_index(drop=True)
    )
    unit_column = "gsc_unit_id" if request.estimator == "gsc" else "mc_unit_id"
    units[unit_column] = np.arange(1, len(units) + 1, dtype=np.int64)
    time_map = pd.DataFrame({"period": times, "time_id": np.arange(1, len(times) + 1)})
    panel = units.merge(time_map, how="cross")
    panel = panel.merge(values, on=["city_key", "grid_id", "period"], how="left")
    pre_count = len(pre)
    panel["D"] = (
        panel["role"].eq("treated") & panel["time_id"].gt(pre_count)
    ).astype(np.int8)
    target_center = 0.0
    target_scale = 1.0
    if request.donor_scope == "all_city_standardized":
        donor_pre = panel.loc[
            panel["role"].eq("donor")
            & panel["time_id"].le(pre_count)
            & np.isfinite(panel["value"])
        ]
        stats = donor_pre.groupby("city_key", as_index=False)["value"].agg(
            pre_center="mean", pre_scale=_sample_sd
        )
        stats.loc[
            ~np.isfinite(stats["pre_scale"])
            | stats["pre_scale"].le(np.sqrt(np.finfo(float).eps)),
            "pre_scale",
        ] = 1.0
        panel = panel.merge(stats, on="city_key", how="left")
        if panel[["pre_center", "pre_scale"]].isna().any().any():
            raise ValueError("cross-city panel lacks finite pre-only city scaling parameters")
        panel["model_value"] = (
            panel["value"] - panel["pre_center"]
        ) / panel["pre_scale"]
        target_stats = panel.loc[panel["role"].eq("treated"), ["pre_center", "pre_scale"]]
        target_center = float(target_stats["pre_center"].iloc[0])
        target_scale = float(target_stats["pre_scale"].iloc[0])
    else:
        panel["model_value"] = panel["value"]

    panel = panel.sort_values([unit_column, "time_id"], kind="stable").reset_index(drop=True)
    metadata: dict[str, object] = {
        "schema": "causal_python_panel_v1",
        "estimator": request.estimator,
        "treatment_order": int(target["treatment_order"]),
        "city_key": str(target["city_key"]),
        "grid_id": str(target["grid_id"]),
        "opening_month": str(target.get("opening_month", opening_period_excluded))[:7],
        "outcome_family": request.outcome_family,
        "outcome": request.outcome,
        "frequency": "monthly" if request.outcome_family in {"housing", "viirs"} else "annual",
        "donor_scope": request.donor_scope,
        "opening_period_excluded": str(opening_period_excluded),
        "first_treated_period": str(post[0]),
        "clean_pre_periods": pre_count,
        "post_periods": len(post),
        "eligible_scope_donors": eligible_scope_donors,
        "donors_after_pre_outcome_cap": int(len(donors)),
        "donors_used": int(units["role"].eq("donor").sum()),
        "donor_cap": (
            pre_outcome_donor_cap
            if pre_outcome_donor_cap != "none"
            else f"top_{request.max_mc_donors}_by_pre_finite_count"
            if donor_capped
            else "none"
        ),
        "donor_sampling_uses_outcome": False,
        "donor_admission_uses_post_outcome": False,
        "cross_city_scaling": (
            "city donor pre-period mean/sd; post-period information excluded"
            if request.donor_scope == "all_city_standardized"
            else "none"
        ),
        "target_effect_scale_to_original_units": target_scale,
        "target_center_to_original_units": target_center,
        "anticipation_months": (
            request.anticipation_months
            if request.outcome_family in {"housing", "viirs"}
            else None
        ),
        "annual_anticipation_years": (
            0 if request.outcome_family not in {"housing", "viirs"} else None
        ),
        "price_measure": request.price_measure,
        "transaction_count_threshold": (
            request.transaction_count_threshold
            if request.outcome_family == "housing"
            else None
        ),
        "transaction_count_threshold_unit": (
            "grid_month" if request.outcome_family == "housing" else None
        ),
    }
    return BuiltPanel(panel=panel, metadata=metadata)


def build_estimation_panel(request: PanelBuildRequest) -> BuiltPanel:
    """Build one formal panel directly from the canonical Python data assets."""
    root = request.root.resolve()
    treatments = pd.read_parquet(root / TREATMENT_UNIT_LIST.relative_to(PROJECT_ROOT))
    selected = treatments.loc[
        pd.to_numeric(treatments["treatment_order"], errors="raise").eq(
            request.treatment_order
        )
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one treatment_order={request.treatment_order}")
    target = selected.iloc[0].copy()
    donor_frame = pd.read_parquet(
        root / ELIGIBLE_DONORS.relative_to(PROJECT_ROOT),
        columns=["city_key", "grid_id", "unit_id"],
    )
    if request.donor_scope == "same_city":
        donor_frame = donor_frame.loc[donor_frame["city_key"].eq(target["city_key"])].copy()
    if donor_frame.empty or donor_frame.duplicated(["city_key", "grid_id"]).any():
        raise ValueError("donor scope is empty or contains duplicate units")
    eligible_scope_donors = len(donor_frame)
    pre_outcome_donor_cap = "none"
    if (
        request.estimator == "gsc"
        and request.donor_scope == "all_city_standardized"
        and len(donor_frame) > request.max_gsc_cross_city_donors
    ):
        donor_frame = deterministic_cross_city_gsc_sample(
            donor_frame,
            request.max_gsc_cross_city_donors,
            request.gsc_donor_sampling_seed,
        )
        pre_outcome_donor_cap = (
            f"uniform_stable_hash_{request.max_gsc_cross_city_donors}"
            f"_seed_{request.gsc_donor_sampling_seed}_before_outcomes"
        )
    donor_frame.attrs["eligible_scope_donors_before_cap"] = eligible_scope_donors
    donor_frame.attrs["pre_outcome_donor_cap"] = pre_outcome_donor_cap
    cities = sorted(donor_frame["city_key"].astype(str).unique())
    if str(target["city_key"]) not in cities:
        cities.append(str(target["city_key"]))
        cities.sort()
    requested_units = pd.concat(
        [
            donor_frame[["city_key", "grid_id"]],
            pd.DataFrame(
                [
                    {
                        "city_key": str(target["city_key"]),
                        "grid_id": str(target["grid_id"]),
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).drop_duplicates(["city_key", "grid_id"])

    if request.outcome_family in {"housing", "viirs"}:
        calendar = monthly_event_calendar(
            target["opening_month"], anticipation_months=request.anticipation_months
        )
        pre = calendar["pre"]
        if request.outcome_family == "viirs":
            pre = pre[pre >= pd.Timestamp("2012-01-01")]
        post = calendar["post"]
        times = pre.append(post)
        outcomes = read_outcomes_for_request(
            request, cities, times, units=requested_units
        )
        available = pd.DatetimeIndex(sorted(pd.to_datetime(outcomes["period"].unique())))
        if not pd.Index(pre).isin(available).all():
            raise ValueError("insufficient clean pre-treatment monthly periods")
        if not pd.Index(post).isin(available).all():
            raise ValueError("insufficient post-treatment monthly periods")
        opening = calendar["opening_month"]
    else:
        outcomes = read_outcomes_for_request(request, cities, units=requested_units)
        available = np.sort(pd.to_numeric(outcomes["period"], errors="raise").unique())
        cohort = int(str(target["opening_month"])[:4])
        pre = pd.Index(available[available < cohort])
        post = pd.Index(available[(available > cohort) & (available <= cohort + 3)])
        if len(pre) < (5 if request.estimator == "gsc" else 1):
            raise ValueError("insufficient clean pre-treatment annual periods")
        if len(post) < 3:
            raise ValueError("insufficient clean post-treatment annual periods")
        opening = cohort
    return build_estimation_panel_from_frames(
        target=target,
        donors=donor_frame,
        outcomes=outcomes,
        request=request,
        pre=pd.Index(pre),
        post=pd.Index(post),
        opening_period_excluded=opening,
    )


def request_metadata(request: PanelBuildRequest) -> dict[str, object]:
    """Return a serialisable request payload for manifests and diagnostics."""
    payload = asdict(request)
    payload["root"] = str(request.root)
    return payload
