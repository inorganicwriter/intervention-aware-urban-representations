from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import urban_intervention.causal.event_study as event_study
from urban_intervention.causal.event_study import (
    _wald_zero,
    fit_twfe_event_study,
    write_matching_event_study_figure,
)


def _staggered_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    event_times = [-3, -2, -1, 0, 1, 2]
    for pair in range(24):
        cohort = 2010 + pair % 6
        for role in ("treated", "control"):
            unit = f"{role}_{pair}"
            unit_effect = pair * 0.05 + (0.2 if role == "treated" else 0.0)
            for event_time in event_times:
                period = cohort + event_time
                time_effect = (period - 2000) * 0.03
                treatment_effect = (
                    {0: 0.5, 1: 1.25, 2: 2.0}.get(event_time, 0.0) if role == "treated" else 0.0
                )
                noise = ((pair * 7 + event_time * 3 + (role == "treated")) % 11 - 5) * 0.001
                rows.append(
                    {
                        "outcome": unit_effect + time_effect + treatment_effect + noise,
                        "unit": unit,
                        "period": period,
                        "role": role,
                        "event_time": event_time,
                        "grid_cluster": unit,
                        "city_cluster": f"city_{pair % 6}",
                        "treatment_order": pair + 1,
                    }
                )
    return pd.DataFrame(rows)


def test_twfe_event_study_keeps_post_periods_and_clusters() -> None:
    result = fit_twfe_event_study(_staggered_panel(), reference_event_time=-1)
    coefficients = result.coefficients.set_index("event_time")
    assert {0, 1, 2}.issubset(coefficients.index)
    assert abs(coefficients.loc[0, "estimate"] - 0.5) < 0.02
    assert abs(coefficients.loc[1, "estimate"] - 1.25) < 0.02
    assert abs(coefficients.loc[2, "estimate"] - 2.0) < 0.02
    assert np.isfinite(coefficients["standard_error_grid"]).all()
    assert np.isfinite(coefficients["standard_error_city"]).all()
    city_half_width = (
        coefficients["confidence_upper_city"] - coefficients["confidence_lower_city"]
    ) / 2
    assert np.all(city_half_width > 1.96 * coefficients["standard_error_city"])
    assert result.grid_cluster_pretrend.iloc[0]["df1"] == 2
    assert result.diagnostics["treated_events"] == 24
    assert result.diagnostics["grid_ssc_parameters"] > result.diagnostics["parameters"]
    assert result.diagnostics["city_ssc_parameters"] > result.diagnostics["parameters"]
    assert result.diagnostics["pretrend_restrictions"] == 2


def test_monthly_pretrend_cutoff_excludes_visible_anticipation_coefficients() -> None:
    panel = _staggered_panel()
    result = fit_twfe_event_study(
        panel,
        reference_event_time=-3,
        pretrend_max_event_time=-3,
        pretrend_window=3,
    )

    assert {-2, -1}.issubset(set(result.coefficients["event_time"]))
    assert result.diagnostics["pretrend_restrictions"] == 0


def test_monthly_post_window_preserves_t0_and_matches_formal_trailing_mean() -> None:
    frame = pd.DataFrame(
        {
            "event_time": [-1, 0, 1, 2, 3, 4],
            "outcome": [-1.0, 10.0, 1.0, 2.0, 3.0, 4.0],
            "role": ["treated"] * 6,
            "source_city": ["alpha"] * 6,
            "source_grid": ["g1"] * 6,
        }
    )
    result = event_study._apply_monthly_post_window(frame, 3).set_index("event_time")
    assert result.loc[-1, "outcome"] == -1.0
    assert result.loc[0, "outcome"] == 10.0
    assert result.loc[1, "outcome"] == 1.0
    assert result.loc[2, "outcome"] == 1.5
    assert result.loc[3, "outcome"] == 2.0
    assert result.loc[4, "outcome"] == 3.0


def test_matching_event_study_figure_is_written(tmp_path: Path) -> None:
    result = fit_twfe_event_study(_staggered_panel(), reference_event_time=-1)
    output = tmp_path / "event_study.png"
    write_matching_event_study_figure(result, output, title="Synthetic")
    assert output.is_file()
    assert output.with_suffix(".pdf").is_file()


def test_wald_refuses_more_restrictions_than_cluster_degrees_of_freedom() -> None:
    result = _wald_zero(
        np.ones(3),
        np.eye(3),
        np.arange(3),
        cluster_count=3,
    ).iloc[0]
    assert np.isnan(result["p_value"])
    assert result["df1"] == 3
    assert result["df2"] == 2


def test_event_study_builder_requires_accepted_task_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controls = pd.DataFrame(
        {
            "treatment_order": [1, 2],
            "status": ["matched", "matched"],
            "control_unit_key": ["alpha::c1", "alpha::c2"],
        }
    )
    families = pd.DataFrame(
        {
            "treatment_order": [1, 2],
            "city_key": ["alpha", "alpha"],
            "grid_id": ["t1", "t2"],
            "opening_month": ["2020-01", "2020-01"],
            "outcome_family": ["population", "population"],
            "status": ["matched_labelled", "matched_labelled"],
        }
    )
    task = tmp_path / "tasks" / "00001" / "population"
    task.mkdir(parents=True)
    labels = pd.DataFrame({"estimator_backend": ["r_reference"]})
    labels.to_parquet(task / "labels.parquet", index=False)
    manifest = {
        "schema": "causal_response_labels",
        "status": "matched_labelled",
        "method": "frozen_matched_change_12m_baseline",
        "outcome_family": "population",
        "run_mode": "production",
        "production_eligible": True,
        "labels_sha256": event_study.file_sha256(task / "labels.parquet"),
        "price_measure": "median",
        "details": {"donor_scope": "same_city", "control_unit_key": "alpha::c1"},
    }
    (task / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def fake_pair(row, _family, _root, _minimum, _maximum):
        return pd.DataFrame(
            {
                "period": [2019, 2020, 2021, 2019, 2020, 2021],
                "outcome": [1.0, 1.5, 2.0, 1.0, 1.25, 1.5],
                "role": ["treated", "treated", "treated", "control", "control", "control"],
                "source_city": ["alpha"] * 6,
                "source_grid": ["t1", "t1", "t1", "c1", "c1", "c1"],
            }
        )

    monkeypatch.setattr(event_study, "_annual_pair_panel", fake_pair)
    panel, _ = event_study.build_matching_event_study_panel(
        "population",
        root=tmp_path,
        control_queue=controls,
        family_queue=families,
        task_root=tmp_path / "tasks",
        min_pre=-1,
        max_post=1,
    )
    assert set(panel["treatment_order"]) == {1}
    assert 0 in set(panel["event_time"])


def test_matching_event_study_default_requests_full_available_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controls = pd.DataFrame(
        {
            "treatment_order": [1],
            "status": ["matched"],
            "control_unit_key": ["alpha::c1"],
        }
    )
    families = pd.DataFrame(
        {
            "treatment_order": [1],
            "city_key": ["alpha"],
            "grid_id": ["t1"],
            "opening_month": ["2020-01"],
            "outcome_family": ["population"],
            "status": ["matched_labelled"],
        }
    )
    task = tmp_path / "tasks" / "00001" / "population"
    task.mkdir(parents=True)
    labels = pd.DataFrame({"estimator_backend": ["r_reference"]})
    labels.to_parquet(task / "labels.parquet", index=False)
    manifest = {
        "schema": "causal_response_labels",
        "status": "matched_labelled",
        "method": "frozen_matched_change_12m_baseline",
        "outcome_family": "population",
        "run_mode": "production",
        "production_eligible": True,
        "labels_sha256": event_study.file_sha256(task / "labels.parquet"),
        "price_measure": "median",
        "details": {"donor_scope": "same_city", "control_unit_key": "alpha::c1"},
    }
    (task / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    requested_bounds: list[tuple[int | None, int | None]] = []

    def fake_pair(row, _family, _root, minimum, maximum):
        requested_bounds.append((minimum, maximum))
        return pd.DataFrame(
            {
                "period": [2010, 2020, 2025, 2010, 2020, 2025],
                "outcome": [1.0, 2.0, 3.0, 1.0, 1.5, 2.0],
                "role": ["treated"] * 3 + ["control"] * 3,
                "source_city": ["alpha"] * 6,
                "source_grid": ["t1"] * 3 + ["c1"] * 3,
            }
        )

    monkeypatch.setattr(event_study, "_annual_pair_panel", fake_pair)
    panel, _ = event_study.build_matching_event_study_panel(
        "population",
        root=tmp_path,
        control_queue=controls,
        family_queue=families,
        task_root=tmp_path / "tasks",
    )

    assert requested_bounds == [(None, None)]
    assert set(panel["event_time"]) == {-10, 0, 5}
