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
    normalize_treated_event_columns,
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
    pooled_panel = panel.loc[panel["method"].isin(["gsc", "mc"])]
    assert pooled_panel["p_value_method"].eq("normal_approximation").all()
    assert pooled_panel["se_type"].eq("city_cluster").all()
    assert pooled_panel["n_city_clusters"].notna().all()
    matching_panel = panel.loc[panel["method"].eq("matching")]
    assert matching_panel["p_value_method"].eq("twfe_cluster_robust").all()

    availability = pd.read_csv(output / "availability.csv", encoding="utf-8-sig")
    assert list(availability.columns) == AVAILABILITY_COLUMNS
    assert "treated_events" not in availability.columns
    matching = availability.loc[
        availability["method"].eq("matching")
        & availability["donor_scope"].eq("same_city")
    ]
    assert matching["status"].eq("available").all()
    cross_matching = availability.loc[
        availability["method"].eq("matching")
        & availability["donor_scope"].eq("all_city_standardized")
    ]
    assert len(cross_matching) == len(FAMILIES)
    assert cross_matching["status"].eq("no_data").all()
    matching_viirs = availability.loc[
        availability["outcome_family"].eq("viirs")
        & availability["analysis_type"].eq("matching")
        & availability["donor_scope"].eq("same_city")
    ].iloc[0]
    assert matching_viirs["availability_treated_events"] == pytest.approx(100.0)
    assert matching_viirs["event_study_max_n_treated"] == pytest.approx(100.0)
    assert matching_viirs["coverage"] == pytest.approx(1.0)
    assert pd.isna(matching_viirs["max_supported_treated_grids"])
    pooled_viirs = availability.loc[
        availability["outcome_family"].eq("viirs")
        & availability["method"].eq("gsc")
        & availability["donor_scope"].eq("same_city")
    ].iloc[0]
    assert pooled_viirs["max_supported_treated_grids"] == pytest.approx(50.0)
    assert pd.isna(pooled_viirs["availability_treated_events"])
    assert pd.isna(pooled_viirs["coverage"])
    housing_gsc = availability.loc[
        availability["outcome_family"].eq("housing") & availability["method"].eq("gsc")
    ]
    assert housing_gsc["status"].eq("no_data").all()
    cross = availability.loc[availability["donor_scope"].eq("all_city_standardized")]
    assert cross["warnings"].fillna("").str.contains("insufficient_clusters").any()
    assert cross["source_pre_min"].isna().all()
    assert cross["t0_available"].isna().all()

    selection = pd.read_csv(output / "family_selection.csv", encoding="utf-8-sig")
    downstream = dict(
        zip(
            selection["outcome_family"],
            selection["selected_family"],
            strict=True,
        )
    )
    assert downstream["viirs"] and downstream["poi"]
    assert not downstream["housing"] and not downstream["population"]
    vip = selection.loc[selection["outcome_family"].eq("viirs")].iloc[0]
    assert vip["availability_treated_events"] == pytest.approx(100.0)
    assert vip["event_study_max_n_treated"] == pytest.approx(100.0)
    assert vip["max_supported_treated_grids"] == pytest.approx(50.0)
    assert vip["n_treated_gap"] == pytest.approx(0.0)
    assert vip["coverage"] == pytest.approx(1.0)
    excluded = selection.loc[selection["outcome_family"].eq("housing")].iloc[0]
    assert "excluded" in excluded["selection_reason"]
    assert "selected" in vip["selection_reason"]
    assert (selection["event_study_max_n_treated"] <= selection["availability_treated_events"]).all()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "mit_event_study_final"
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


def test_validation_flags_window_support_above_availability(
    release_inputs, tmp_path: Path
) -> None:
    matching_root, pooled_root, suite_root = release_inputs
    diagnostics_path = matching_root / "matching_twfe" / "viirs" / "diagnostics.csv"
    diagnostics = pd.read_csv(diagnostics_path)
    diagnostics["treated_events"] = 40
    diagnostics.to_csv(diagnostics_path, index=False)

    output = tmp_path / "event_study_final_inconsistent"
    result = export_event_study_final(
        matching_root,
        pooled_root,
        output,
        suite_root=suite_root,
    )
    validation = result["validation"]
    assert not validation["ok"]
    consistency = {
        check["name"]: check
        for check in validation["checks"]
        if check["name"] == "treated_event_consistency"
    }["treated_event_consistency"]
    assert consistency["severity"] == "error"
    assert not consistency["passed"]
    assert "viirs" in consistency["detail"]

    selection = pd.read_csv(output / "family_selection.csv", encoding="utf-8-sig")
    viirs = selection.loc[selection["outcome_family"].eq("viirs")].iloc[0]
    assert viirs["availability_treated_events"] == pytest.approx(40.0)
    assert viirs["event_study_max_n_treated"] == pytest.approx(100.0)
    assert viirs["n_treated_gap"] == pytest.approx(-60.0)
    assert viirs["coverage"] == pytest.approx(2.5)


def test_normalize_legacy_treated_events_column() -> None:
    legacy = pd.DataFrame(
        {
            "outcome_family": ["viirs", "viirs", "poi"],
            "method": ["matching", "gsc", "mc"],
            "treated_events": [110.0, 50.0, 20.0],
        }
    )
    normalized = normalize_treated_event_columns(legacy)
    assert "treated_events" not in normalized.columns
    matching_row = normalized.loc[normalized["method"].eq("matching")].iloc[0]
    pooled_row = normalized.loc[normalized["method"].eq("gsc")].iloc[0]
    mc_row = normalized.loc[normalized["method"].eq("mc")].iloc[0]
    assert matching_row["availability_treated_events"] == pytest.approx(110.0)
    assert pd.isna(matching_row["max_supported_treated_grids"])
    assert pooled_row["max_supported_treated_grids"] == pytest.approx(50.0)
    assert pd.isna(pooled_row["availability_treated_events"])
    assert mc_row["max_supported_treated_grids"] == pytest.approx(20.0)
    assert normalize_treated_event_columns(normalized) is normalized


def test_selection_isolates_same_city_when_cross_matching_present(
    release_inputs, tmp_path: Path
) -> None:
    matching_root, pooled_root, suite_root = release_inputs
    cross_dir = matching_root / "matching_twfe_cross" / "viirs"
    cross_dir.mkdir(parents=True, exist_ok=True)
    times = [-24, 0, 24]
    pd.DataFrame(
        {
            "term": [f"event_time::{t}" for t in times],
            "event_time": times,
            "estimate": [0.05] * len(times),
            "standard_error_grid": [0.01] * len(times),
            "p_value_grid": [0.01] * len(times),
            "confidence_lower_grid": [0.03] * len(times),
            "confidence_upper_grid": [0.07] * len(times),
            "standard_error_city": [0.02] * len(times),
            "p_value_city": [0.05] * len(times),
            "confidence_lower_city": [0.01] * len(times),
            "confidence_upper_city": [0.09] * len(times),
            "n_treated": [150] * len(times),
            "exposure_status": ["post" if t >= 0 else "pre" for t in times],
        }
    ).to_csv(cross_dir / "event_study_coefficients.csv", index=False)
    pd.DataFrame(
        [
            {
                "nobs": 2000,
                "units": 300,
                "treated_events": 150,
                "grid_clusters": 250,
                "city_clusters": 50,
                "reference_event_time": -1,
            }
        ]
    ).to_csv(cross_dir / "diagnostics.csv", index=False)
    for filename in ("parallel_trends_wald.csv", "parallel_trends_wald_city_cluster.csv"):
        pd.DataFrame(
            [
                {
                    "statistic": 0.5,
                    "df1": 2,
                    "df2": 49,
                    "covariance_rank": 2,
                    "p_value": 0.6,
                }
            ]
        ).to_csv(cross_dir / filename, index=False)

    output = tmp_path / "event_study_final_with_cross"
    result = export_event_study_final(
        matching_root,
        pooled_root,
        output,
        suite_root=suite_root,
    )
    assert result["validation"]["ok"]

    selection = pd.read_csv(output / "family_selection.csv", encoding="utf-8-sig")
    viirs = selection.loc[selection["outcome_family"].eq("viirs")].iloc[0]
    # same-city baseline has 100 treated events & 20 city clusters
    assert viirs["availability_treated_events"] == pytest.approx(100.0)
    assert viirs["event_study_max_n_treated"] == pytest.approx(100.0)
    assert viirs["matching_n_cities"] == pytest.approx(20.0)
    assert viirs["n_treated_gap"] == pytest.approx(0.0)
    assert viirs["coverage"] == pytest.approx(1.0)

    availability = pd.read_csv(output / "availability.csv", encoding="utf-8-sig")
    cross_viirs = availability.loc[
        availability["outcome_family"].eq("viirs")
        & availability["donor_scope"].eq("all_city_standardized")
    ].iloc[0]
    assert cross_viirs["status"] == "available"
    assert cross_viirs["availability_treated_events"] == pytest.approx(150.0)
    assert cross_viirs["event_study_max_n_treated"] == pytest.approx(150.0)
