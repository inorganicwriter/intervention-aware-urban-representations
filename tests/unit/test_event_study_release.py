"""Tests for the frozen four-family event-study release exporter."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from urban_intervention.causal.event_study_release import (
    AVAILABILITY_COLUMNS,
    PANEL_COLUMNS,
    export_event_study_final,
)

FAMILIES = ("housing", "viirs", "poi", "population")
FREQUENCY = {"housing": "monthly", "viirs": "monthly", "poi": "annual", "population": "annual"}
MAIN_OUTCOME = {
    "housing": "housing_log_price",
    "viirs": "viirs_avg_asinh",
    "poi": "poi_count_log",
    "population": "population_log",
}


def _write_matching(root: Path, family: str, *, include: bool = True) -> None:
    if not include:
        return
    directory = root / "matching_twfe" / family
    directory.mkdir(parents=True, exist_ok=True)
    if FREQUENCY[family] == "monthly":
        event_times = [-30, -24, -12, -1, 0, 1, 12, 24, 40]
    else:
        event_times = [-8, -6, -3, -1, 0, 1, 3, 5]
    coefficients = pd.DataFrame(
        {
            "term": [f"event_time::{time}" for time in event_times],
            "event_time": event_times,
            "estimate": [0.01 * time for time in event_times],
            "standard_error_grid": [0.01] * len(event_times),
            "p_value_grid": [0.2] * len(event_times),
            "confidence_lower_grid": [-0.01] * len(event_times),
            "confidence_upper_grid": [0.03] * len(event_times),
            "standard_error_city": [0.02] * len(event_times),
            "p_value_city": [0.1] * len(event_times),
            "confidence_lower_city": [-0.02] * len(event_times),
            "confidence_upper_city": [0.04] * len(event_times),
            "n_treated": [100] * len(event_times),
            "exposure_status": ["post" if time >= 0 else "pre" for time in event_times],
        }
    )
    coefficients.to_csv(directory / "event_study_coefficients.csv", index=False)
    pd.DataFrame(
        [
            {
                "nobs": 1000,
                "units": 200,
                "treated_events": 100,
                "grid_clusters": 180,
                "city_clusters": 20,
                "reference_event_time": -1,
            }
        ]
    ).to_csv(directory / "diagnostics.csv", index=False)
    for filename in ("parallel_trends_wald.csv", "parallel_trends_wald_city_cluster.csv"):
        pd.DataFrame(
            [
                {
                    "statistic": 1.5,
                    "df1": 5,
                    "df2": 19,
                    "covariance_rank": 5,
                    "p_value": 0.2,
                }
            ]
        ).to_csv(directory / filename, index=False)


def _pooled_row(
    family: str,
    method: str,
    scope: str,
    outcome: str,
    event_time: int,
    n_cities: int,
) -> dict[str, object]:
    return {
        "frequency": FREQUENCY[family],
        "outcome_family": family,
        "outcome": outcome,
        "method": method,
        "donor_scope": scope,
        "observation_window": 1,
        "reference_event_time": -1,
        "event_time": event_time,
        "n_grids": 50,
        "n_cities": n_cities,
        "mean_label": 0.02 * event_time,
        "sd_label": 0.5,
        "se_label": 0.03,
        "se_label_city_cluster": 0.05,
        "ci_lower": -0.1,
        "ci_upper": 0.1,
        "ci_lower_city_cluster": -0.15,
        "ci_upper_city_cluster": 0.15,
        "within_var": 0.01,
        "between_var": 0.2,
        "se_available": 50,
        "uncertainty_status": "available",
        "specifications": "fixture_spec",
    }


def _write_pooled(root: Path) -> None:
    directory = root / "gsc_mc_paths"
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    method_name = {"gsc": "xu_2017_gsynth", "mc": "athey_2021_mc"}
    plans = {
        "viirs": [("gsc", "same_city"), ("mc", "same_city"), ("gsc", "all_city_standardized")],
        "poi": [("mc", "same_city"), ("mc", "all_city_standardized")],
        "housing": [("mc", "same_city")],
        "population": [("gsc", "same_city")],
    }
    for family, cells in plans.items():
        if FREQUENCY[family] == "monthly":
            event_times = [-30, -24, -12, 0, 12, 24, 40]
        else:
            event_times = [-8, -6, 0, 3, 5]
        for method, scope in cells:
            n_cities = 30 if scope == "same_city" else 2
            for event_time in event_times:
                rows.append(
                    _pooled_row(
                        family,
                        method_name[method],
                        scope,
                        MAIN_OUTCOME[family],
                        event_time,
                        n_cities,
                    )
                )
    pd.DataFrame(rows).to_csv(directory / "event_study_series.csv", index=False)
    pd.DataFrame(
        [
            {
                "frequency": "annual",
                "outcome_family": "poi",
                "outcome": "poi_count_log",
                "method": "athey_2021_mc",
                "donor_scope": "same_city",
                "observation_window": 1,
                "reference_event_time": -1,
                "n_cities": 30,
                "n_pre_observations": 79,
                "mean_city_mean": 0.001,
                "sd_city_mean": 0.15,
                "t_statistic": 0.01,
                "p_value": 0.99,
                "reject_5pct": False,
                "test_type": "cluster_mean_pre_gap_zero",
            }
        ]
    ).to_csv(directory / "pretrend_city_cluster.csv", index=False)


def _write_support(root: Path) -> None:
    rows = []
    for family in FAMILIES:
        rows.append(
            {
                "outcome_family": family,
                "method": "matching",
                "available": True,
                "method_pre_min": -24.0 if FREQUENCY[family] == "monthly" else -6.0,
                "method_pre_max": -1.0,
                "method_post_min": 1.0,
                "method_post_max": 24.0 if FREQUENCY[family] == "monthly" else 3.0,
                "t0_available": True,
                "common_pre_min": -24,
                "common_pre_max": -1,
                "common_post_min": 1,
                "common_post_max": 24,
            }
        )
    pd.DataFrame(rows).to_csv(root / "event_time_support.csv", index=False)


@pytest.fixture()
def release_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    suite = tmp_path / "suite"
    suite.mkdir()
    for family in FAMILIES:
        _write_matching(suite, family)
    _write_pooled(suite)
    _write_support(suite)
    return suite, suite / "gsc_mc_paths", suite


def test_export_builds_unified_release(release_inputs, tmp_path: Path) -> None:
    matching_root, pooled_root, suite_root = release_inputs
    output = tmp_path / "event_study_final"
    result = export_event_study_final(
        matching_root,
        pooled_root,
        output,
        suite_root=suite_root,
        make_zip=True,
        zip_name="mit_event_study_final.zip",
    )

    panel = pd.read_csv(output / "event_study_panel.csv", encoding="utf-8-sig")
    assert list(panel.columns) == PANEL_COLUMNS
    assert set(panel["outcome_family"]) == set(FAMILIES)
    assert set(panel.loc[panel["method"].eq("matching"), "donor_scope"]) == {"same_city"}
    viirs = panel.loc[panel["outcome_family"].eq("viirs")]
    assert viirs["event_time"].min() >= -24
    assert viirs["event_time"].max() <= 24
    poi = panel.loc[panel["outcome_family"].eq("poi")]
    assert poi["event_time"].min() >= -6
    assert poi["event_time"].max() <= 3
    assert panel["p_city"].notna().any()

    availability = pd.read_csv(output / "availability.csv", encoding="utf-8-sig")
    assert list(availability.columns) == AVAILABILITY_COLUMNS
    matching = availability.loc[availability["method"].eq("matching")]
    assert matching["status"].eq("available").all()
    housing_gsc = availability.loc[
        availability["outcome_family"].eq("housing") & availability["method"].eq("gsc")
    ]
    assert housing_gsc["status"].eq("no_data").all()
    cross = availability.loc[availability["donor_scope"].eq("all_city_standardized")]
    assert cross["warnings"].fillna("").str.contains("insufficient_clusters").any()

    selection = pd.read_csv(output / "family_selection.csv", encoding="utf-8-sig")
    downstream = dict(
        zip(
            selection["outcome_family"],
            selection["in_downstream_supervision"],
            strict=True,
        )
    )
    assert downstream["viirs"] and downstream["poi"]
    assert not downstream["housing"] and not downstream["population"]

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "mit_event_study_final_v1"
    assert manifest["windows"]["viirs"]["canonical_pre_min"] == -24
    assert manifest["windows"]["poi"]["canonical_post_max"] == 3
    assert manifest["downstream_training"]["included_families"] == ["viirs", "poi"]
    assert manifest["downstream_training"]["excluded_families"] == ["housing", "population"]
    assert manifest["input_hashes"]
    assert manifest["caveats"]
    assert manifest["window_filtered_rows"] > 0

    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert validation["ok"], validation["checks"]

    zip_path = tmp_path / "mit_event_study_final.zip"
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    assert any(name.endswith("manifest.json") for name in names)
    assert any(name.endswith("event_study_panel.csv") for name in names)
    assert result["zip"] == str(zip_path)


def test_validation_blocks_missing_family(release_inputs, tmp_path: Path) -> None:
    matching_root, pooled_root, suite_root = release_inputs
    for name in ("event_study_coefficients.csv", "diagnostics.csv"):
        (matching_root / "matching_twfe" / "population" / name).unlink()
    output = tmp_path / "event_study_final_incomplete"
    result = export_event_study_final(
        matching_root,
        pooled_root,
        output,
        suite_root=suite_root,
    )
    validation = result["validation"]
    assert not validation["ok"]
    failing = {check["name"] for check in validation["checks"] if not check["passed"]}
    assert "matching_available" in failing
    availability = pd.read_csv(output / "availability.csv", encoding="utf-8-sig")
    population = availability.loc[
        availability["outcome_family"].eq("population") & availability["method"].eq("matching")
    ]
    assert population["status"].iloc[0] == "no_data"
