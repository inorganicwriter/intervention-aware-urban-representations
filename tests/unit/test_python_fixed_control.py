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
