"""Python implementation of the frozen matched-control label contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from urban_intervention.data.paths import PROJECT_ROOT, TREATMENT_UNIT_LIST

from .panel_builder import (
    OUTCOMES,
    monthly_event_calendar,
    read_annual_outcome,
    read_monthly_housing,
    read_monthly_viirs,
)


def _finite_mean(values: pd.Series, minimum: int = 1) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) >= minimum else float("nan")


def _did_regression(
    treated_values: np.ndarray, control_values: np.ndarray, pre_count: int
) -> dict[str, float | int]:
    treated_values = np.asarray(treated_values, dtype=np.float64)
    control_values = np.asarray(control_values, dtype=np.float64)
    if (
        pre_count < 2
        or not np.isfinite(treated_values).all()
        or not np.isfinite(control_values).all()
        or treated_values.shape != control_values.shape
    ):
        return {
            "regression_beta": np.nan,
            "regression_se": np.nan,
            "regression_p": np.nan,
            "regression_nobs": pd.NA,
        }
    periods = len(treated_values)
    post = (np.arange(periods) >= pre_count).astype(float)
    values = np.concatenate([treated_values, control_values])
    treat = np.concatenate([np.ones(periods), np.zeros(periods)])
    post_two = np.tile(post, 2)
    design = np.column_stack(
        [np.ones(2 * periods), treat, post_two, treat * post_two]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank < design.shape[1]:
        return {
            "regression_beta": np.nan,
            "regression_se": np.nan,
            "regression_p": np.nan,
            "regression_nobs": 2 * periods,
        }
    residual = values - design @ coefficients
    degrees = len(values) - design.shape[1]
    sigma2 = float(residual @ residual / degrees) if degrees > 0 else np.nan
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    standard_error = float(np.sqrt(max(covariance[3, 3], 0)))
    if standard_error > 0 and np.isfinite(standard_error):
        from scipy.stats import t as student_t

        p_value = float(2 * student_t.sf(abs(coefficients[3] / standard_error), degrees))
    else:
        p_value = np.nan
    return {
        "regression_beta": float(coefficients[3]),
        "regression_se": standard_error,
        "regression_p": p_value,
        "regression_nobs": 2 * periods,
    }


def _one_grid_path(
    frame: pd.DataFrame, city: str, grid: str, outcome: str, periods: pd.Index
) -> np.ndarray:
    selected = frame.loc[
        frame["city_key"].eq(city) & frame["grid_id"].eq(grid),
        ["period", outcome],
    ]
    if selected.duplicated("period").any():
        raise ValueError("fixed-control outcome path contains duplicate periods")
    return (
        selected.set_index("period")[outcome]
        .reindex(periods)
        .to_numpy(dtype=np.float64)
    )


def monthly_fixed_control_labels(
    target: pd.Series,
    control_city_key: str,
    control_grid_id: str,
    family: str,
    *,
    root: Path = PROJECT_ROOT,
    window: int = 1,
    price_measure: str = "median",
    transaction_count_threshold: int = 1,
) -> pd.DataFrame:
    if family not in {"housing", "viirs"}:
        raise ValueError("monthly fixed-control family must be housing or viirs")
    if not 1 <= window <= 6:
        raise ValueError("window must be in 1..6")
    if transaction_count_threshold < 1:
        raise ValueError("transaction_count_threshold must be positive")
    horizons = np.asarray([1, 3, 6, 12, 18, 24], dtype=int)
    post_leads = sorted(
        {
            lead
            for horizon in horizons
            for lead in range(max(1, horizon - window + 1), horizon + 1)
        }
    )
    calendar = monthly_event_calendar(
        target["opening_month"], leads=tuple(post_leads), anticipation_months=6
    )
    baseline = pd.DatetimeIndex(calendar["pre"][-12:])
    post = pd.DatetimeIndex(calendar["post"])
    all_periods = baseline.append(post)
    def reader(city: str) -> pd.DataFrame:
        if family == "housing":
            return read_monthly_housing(root, city, price_measure)  # type: ignore[arg-type]
        return read_monthly_viirs(root, city, all_periods)
    treated_frame = reader(str(target["city_key"]))
    control_frame = reader(control_city_key)
    outcome = OUTCOMES[family][0]
    treated_values = _one_grid_path(
        treated_frame, str(target["city_key"]), str(target["grid_id"]), outcome, all_periods
    )
    control_values = _one_grid_path(
        control_frame, control_city_key, control_grid_id, outcome, all_periods
    )
    treated_transactions = (
        _one_grid_path(
            treated_frame,
            str(target["city_key"]),
            str(target["grid_id"]),
            "transaction_count",
            all_periods,
        )
        if family == "housing"
        else np.full(len(all_periods), np.nan)
    )
    control_transactions = (
        _one_grid_path(
            control_frame,
            control_city_key,
            control_grid_id,
            "transaction_count",
            all_periods,
        )
        if family == "housing"
        else np.full(len(all_periods), np.nan)
    )
    if family == "housing":
        treated_values = treated_values.copy()
        control_values = control_values.copy()
        treated_supported = np.isfinite(treated_transactions) & (
            treated_transactions >= transaction_count_threshold
        )
        control_supported = np.isfinite(control_transactions) & (
            control_transactions >= transaction_count_threshold
        )
        treated_values[~treated_supported] = np.nan
        control_values[~control_supported] = np.nan
    minimum_baseline = 12 if family == "viirs" else 1
    treated_baseline = _finite_mean(pd.Series(treated_values[:12]), minimum_baseline)
    control_baseline = _finite_mean(pd.Series(control_values[:12]), minimum_baseline)
    lead_lookup = {lead: 12 + position for position, lead in enumerate(post_leads)}
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        window_leads = range(max(1, int(horizon) - window + 1), int(horizon) + 1)
        positions = [lead_lookup[value] for value in window_leads]
        treated_post = _finite_mean(pd.Series(treated_values[positions]))
        control_post = _finite_mean(pd.Series(control_values[positions]))
        minimum_window = min(window, int(horizon))
        n_treated = int(np.isfinite(treated_values[positions]).sum())
        n_control = int(np.isfinite(control_values[positions]).sum())
        supported = n_treated >= minimum_window and n_control >= minimum_window
        treated_change = treated_post - treated_baseline
        control_change = control_post - control_baseline
        endpoint = lead_lookup[int(horizon)]
        regression = _did_regression(
            treated_values[: endpoint + 1], control_values[: endpoint + 1], 12
        )
        treated_transaction_count = float(np.nansum(treated_transactions[positions]))
        control_transaction_count = float(np.nansum(control_transactions[positions]))
        transaction_supported = bool(
            family != "housing"
            or (
                np.all(
                    treated_transactions[positions]
                    >= transaction_count_threshold
                )
                and np.all(
                    control_transactions[positions]
                    >= transaction_count_threshold
                )
            )
        )
        available = bool(
            supported
            and transaction_supported
            and np.isfinite(treated_change)
            and np.isfinite(control_change)
        )
        rows.append(
            {
                "outcome": outcome,
                "event_time": int(horizon),
                "period": pd.Timestamp(target["opening_month"]) + pd.DateOffset(months=int(horizon)),
                "treated_post": treated_post,
                "control_post": control_post,
                "effective_n_treated": n_treated,
                "effective_n_control": n_control,
                "minimum_window_n": minimum_window,
                "treated_baseline": treated_baseline,
                "control_baseline": control_baseline,
                "treated_change": treated_change,
                "control_change": control_change,
                "observed": treated_post,
                "counterfactual": treated_baseline + control_change,
                "window_supported": supported,
                "label_available": available,
                "causal_response_label": (
                    treated_change - control_change if available else np.nan
                ),
                "transaction_count": (
                    treated_transaction_count if family == "housing" else np.nan
                ),
                "control_transaction_count": (
                    control_transaction_count if family == "housing" else np.nan
                ),
                "transaction_count_threshold": (
                    transaction_count_threshold if family == "housing" else np.nan
                ),
                "transaction_count_supported": bool(
                    family == "housing"
                    and np.all(
                        treated_transactions[positions]
                        >= transaction_count_threshold
                    )
                ),
                "control_transaction_count_supported": bool(
                    family == "housing"
                    and np.all(
                        control_transactions[positions]
                        >= transaction_count_threshold
                    )
                ),
                **regression,
            }
        )
    return pd.DataFrame(rows)


def annual_fixed_control_labels(
    target: pd.Series,
    control_city_key: str,
    control_grid_id: str,
    family: str,
    *,
    root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    opening_year = int(str(target["opening_month"])[:4])
    horizons = np.arange(1, 4, dtype=int)
    baseline_year = opening_year - 1
    post_years = opening_year + horizons
    all_regression_years = pd.Index([*range(baseline_year - 3, baseline_year + 1), *post_years])
    treated = read_annual_outcome(root, str(target["city_key"]), family).rename(
        columns={"year": "period"}
    )
    control = read_annual_outcome(root, control_city_key, family).rename(
        columns={"year": "period"}
    )
    rows: list[dict[str, object]] = []
    for outcome in OUTCOMES[family]:
        treated_values = _one_grid_path(
            treated,
            str(target["city_key"]),
            str(target["grid_id"]),
            outcome,
            all_regression_years,
        )
        control_values = _one_grid_path(
            control, control_city_key, control_grid_id, outcome, all_regression_years
        )
        treated_baseline = float(treated_values[3])
        control_baseline = float(control_values[3])
        for position, (horizon, year) in enumerate(zip(horizons, post_years, strict=True), start=4):
            treated_post = float(treated_values[position])
            control_post = float(control_values[position])
            treated_change = treated_post - treated_baseline
            control_change = control_post - control_baseline
            available = bool(np.isfinite(treated_change) and np.isfinite(control_change))
            regression = _did_regression(
                treated_values[: position + 1], control_values[: position + 1], 4
            )
            rows.append(
                {
                    "outcome": outcome,
                    "event_time": int(horizon),
                    "year": int(year),
                    "treated_post": treated_post,
                    "control_post": control_post,
                    "treated_baseline": treated_baseline,
                    "control_baseline": control_baseline,
                    "treated_change": treated_change,
                    "control_change": control_change,
                    "observed": treated_post,
                    "counterfactual": treated_baseline + control_change,
                    "causal_response_label": (
                        treated_change - control_change if available else np.nan
                    ),
                    "label_available": available,
                    **regression,
                }
            )
    return pd.DataFrame(rows)


def fixed_control_labels(
    treatment_order: int,
    control_city_key: str,
    control_grid_id: str,
    family: str,
    *,
    root: Path = PROJECT_ROOT,
    window: int = 1,
    price_measure: str = "median",
    transaction_count_threshold: int = 1,
) -> pd.DataFrame:
    treatments = pd.read_parquet(root / TREATMENT_UNIT_LIST.relative_to(PROJECT_ROOT))
    selected = treatments.loc[treatments["treatment_order"].eq(treatment_order)]
    if len(selected) != 1:
        raise ValueError("treatment order is not unique")
    target = selected.iloc[0]
    if family not in OUTCOMES:
        raise ValueError(f"unknown outcome family: {family}")
    result = (
        monthly_fixed_control_labels(
            target,
            control_city_key,
            control_grid_id,
            family,
            root=root,
            window=window,
            price_measure=price_measure,
            transaction_count_threshold=transaction_count_threshold,
        )
        if family in {"housing", "viirs"}
        else annual_fixed_control_labels(
            target, control_city_key, control_grid_id, family, root=root
        )
    )
    result = result.assign(
        treatment_order=int(target["treatment_order"]),
        city_key=str(target["city_key"]),
        grid_id=str(target["grid_id"]),
        opening_month=str(target["opening_month"]),
        outcome_family=family,
        control_city_key=control_city_key,
        control_grid_id=control_grid_id,
        control_unit_key=f"{control_city_key}::{control_grid_id}",
        method="frozen_matched_change_12m_baseline",
        specification_id="main_a6_r1km",
        standard_error=np.nan,
        confidence_lower=np.nan,
        confidence_upper=np.nan,
        p_value=np.nan,
        bootstrap_repetitions=0,
        uncertainty_source="preonly_match_design_diagnostics",
    )
    first = [
        "treatment_order",
        "city_key",
        "grid_id",
        "opening_month",
        "outcome_family",
        "outcome",
        "event_time",
        "specification_id",
        "observed",
        "counterfactual",
        "causal_response_label",
        "label_available",
        "treated_baseline",
        "control_baseline",
        "treated_change",
        "control_change",
        "control_city_key",
        "control_grid_id",
        "control_unit_key",
        "method",
    ]
    return result[[*first, *[column for column in result.columns if column not in first]]]
