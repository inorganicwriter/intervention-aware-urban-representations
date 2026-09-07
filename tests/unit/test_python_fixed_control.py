from __future__ import annotations

import numpy as np
import pandas as pd

import urban_intervention.causal.gpu.fixed_control as fixed_control


def test_housing_transaction_threshold_controls_label_availability(monkeypatch) -> None:
    target = pd.Series(
        {
            "city_key": "treated_city",
            "grid_id": "treated_grid",
            "opening_month": "2020-07",
        }
    )
    periods = pd.date_range("2019-01-01", "2022-07-01", freq="MS")

    def city_frame(city: str) -> pd.DataFrame:
        grid = "treated_grid" if city == "treated_city" else "control_grid"
        offset = 1.0 if city == "treated_city" else 0.5
        return pd.DataFrame(
            {
                "city_key": city,
                "grid_id": grid,
                "period": periods,
                "housing_log_price": offset + np.arange(len(periods)) * 0.01,
                "transaction_count": 1,
            }
        )

    monkeypatch.setattr(
        fixed_control,
        "read_monthly_housing",
        lambda _root, city, _measure: city_frame(city),
    )
    admitted = fixed_control.monthly_fixed_control_labels(
        target,
        "control_city",
        "control_grid",
        "housing",
        transaction_count_threshold=1,
    )
    restricted = fixed_control.monthly_fixed_control_labels(
        target,
        "control_city",
        "control_grid",
        "housing",
        transaction_count_threshold=2,
    )
    assert admitted["label_available"].all()
    assert not restricted["label_available"].any()
    assert restricted["causal_response_label"].isna().all()


def test_viirs_never_uses_temporal_moving_window(monkeypatch) -> None:
    target = pd.Series(
        {
            "city_key": "treated_city",
            "grid_id": "treated_grid",
            "opening_month": "2020-07",
        }
    )

    def city_frame(_root, city, periods):
        grid = "treated_grid" if city == "treated_city" else "control_grid"
        offset = 1.0 if city == "treated_city" else 0.5
        values = offset + np.arange(len(periods), dtype=float)
        return pd.DataFrame(
            {
                "city_key": city,
                "grid_id": grid,
                "period": periods,
                "viirs_avg_asinh": values,
            }
        )

    monkeypatch.setattr(fixed_control, "read_monthly_viirs", city_frame)
    labels = fixed_control.monthly_fixed_control_labels(
        target,
        "control_city",
        "control_grid",
        "viirs",
        window=3,
    )

    assert set(labels["minimum_window_n"]) == {1}
    assert labels.loc[labels["event_time"].eq(3), "window_supported"].all()
