from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.build_intervention_inputs import FIELDS, build_inputs


def _sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    treatments = pd.DataFrame({"treatment_order": [1, 2], "city_key": ["city", "city"],
                               "grid_id": ["a", "b"], "station_event_id": ["s1", "s2"],
                               "opening_date": ["2020-01-01", "2021-03-01"],
                               "opening_month": ["2020-01", "2021-03"]})
    attrs = treatments.drop(columns=["grid_id", "opening_month"]).copy()
    attrs["is_treated"] = True
    for field in FIELDS[1:]:
        attrs[field] = [0, 1] if field.startswith("is_") else [1, 2]
    attrs["terminal_evidence"] = ["unknown", "station_degree"]
    attrs.loc[0, "is_terminal_at_opening"] = np.nan
    return treatments, attrs


def test_unknown_terminal_and_zero_lines_preserve_event_with_masks() -> None:
    treatments, attrs = _sources()
    attrs.loc[1, "n_lines_at_opening"] = 0
    result = build_inputs(treatments, attrs)
    assert len(result) == 2
    assert not result.is_terminal_at_opening_available.any()
    assert not result.loc[1, "is_transfer_at_opening_available"]
    assert np.isnan(result.loc[1, "n_lines_at_opening"])
    assert result.opening_year_available.all()
    assert result.same_month_openings_available.all()
    assert result.historical_source_review_status.eq("requires_source_review").all()


@pytest.mark.parametrize("fault", ["identity", "date", "duplicate", "missing"])
def test_bad_station_binding_rejected(fault: str) -> None:
    treatments, attrs = _sources()
    if fault == "identity":
        attrs.loc[0, "station_event_id"] = "other"
    elif fault == "date":
        attrs.loc[0, "opening_date"] = "2022-01-01"
    elif fault == "duplicate":
        attrs.loc[0, "treatment_order"] = 2
    else:
        attrs = attrs.iloc[:1]
    with pytest.raises(ValueError):
        build_inputs(treatments, attrs)
