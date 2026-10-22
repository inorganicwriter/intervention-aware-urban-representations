from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urban_intervention.causal.pretraining_dataset import (
    IDENTITY_COLUMNS,
    STC_CONTEXT_TABULAR_COLUMNS,
    _best_quality_grade,
    _enforce_production_supervision_scope,
    _task_sampling_weight,
    build_pretreatment_features,
    deterministic_city_splits,
    publish_pretraining_dataset,
    response_label_weights,
    validate_response_treatment_binding,
)
from urban_intervention.causal.response_artifact import sha256_file


def test_stc_context_excludes_housing_price_fields() -> None:
    assert not any(
        "housing" in column or "price" in column
        for column in STC_CONTEXT_TABULAR_COLUMNS
    )


def test_production_supervision_scope_defaults_and_rejects_excluded_labels() -> None:
    families, methods = _enforce_production_supervision_scope(None, None)
    assert families == ("viirs", "poi")
    assert methods == ("matched", "gsc")
    with pytest.raises(ValueError, match="excluded families"):
        _enforce_production_supervision_scope(("viirs", "housing"), None)
    with pytest.raises(ValueError, match="excluded methods"):
        _enforce_production_supervision_scope(None, ("matched", "mc"))


def test_best_quality_grade_prefers_full_support_mc_and_available_labels() -> None:
    targets = pd.DataFrame(
        {
            "treatment_order": [1, 1, 2, 2, 3, 3],
            "quality_grade": [
                "mc_same_city_minimal_pre_support",
                "mc_same_city_pass",
                "matched_same_city_pass",
                "mc_same_city_pass",
                "mc_cross_city_minimal_pre_support",
                "mc_cross_city_pass",
            ],
            "label_grade": [
                "mc_same_city_minimal_pre_support",
                "mc_same_city_pass",
                "unavailable",
                "mc_same_city_pass",
                "mc_cross_city_minimal_pre_support",
                "mc_cross_city_pass",
            ],
        }
    )
    best = _best_quality_grade(targets).set_index("treatment_order")["quality_grade"]
    # Full-support MC outranks minimal-support MC.
    assert best.loc[1] == "mc_same_city_pass"
    # An unavailable matched label must not outrank an available MC label.
    assert best.loc[2] == "mc_same_city_pass"
    # Cross-city ordering still prefers full support.
    assert best.loc[3] == "mc_cross_city_pass"


def test_best_quality_grade_falls_back_without_label_grade_column() -> None:
    targets = pd.DataFrame(
        {
            "treatment_order": [1, 1],
            "quality_grade": ["mc_same_city_pass", "gsc_same_city_pass"],
        }
    )
    best = _best_quality_grade(targets).set_index("treatment_order")["quality_grade"]
    assert best.loc[1] == "gsc_same_city_pass"


def _write_sources(root: Path, treatments: pd.DataFrame) -> None:
    for directory in (
        "data/active/causal/formal_matching_inputs/housing_annual",
        "data/active/curated/poi",
        "data/active/curated/viirs_annual_aggregated",
        "data/active/curated/population",
        "data/active/curated/sentinel2",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for row in treatments.itertuples(index=False):
        years = [2017, 2018, 2019, 2020]
        base = {"grid_id": [row.grid_id] * 4, "year": years}
        pd.DataFrame(
            {
                **base,
                "city_key": [row.city_key] * 4,
                "housing_log_price": [1.0, 2.0, 3.0, 999.0],
            }
        ).to_parquet(
            root
            / "data/active/causal/formal_matching_inputs/housing_annual"
            / f"{row.city_key}.parquet",
            index=False,
        )
        pd.DataFrame(
            {
                **base,
                "city": [row.city_key] * 4,
                "poi_count": [10, 20, 30, 999],
                "poi_category_entropy": [0.1, 0.2, 0.3, 9.0],
                "poi_commercial_share": [0.2, 0.3, 0.4, 9.0],
                "poi_transport_access_count": [1, 2, 3, 999],
            }
        ).to_parquet(
            root / "data/active/curated/poi" / f"{row.city_key}_poi_grid_yearly.parquet",
            index=False,
        )
        pd.DataFrame(
            {
                **base,
                "city_key": [row.city_key] * 4,
                "avg_rad": [1, 2, 3, 999],
            }
        ).to_parquet(
            root
            / "data/active/curated/viirs_annual_aggregated"
            / f"{row.city_key}_viirs_annual.parquet",
            index=False,
        )
        population = pd.DataFrame(
            {
                **base,
                "city": [row.city_key] * 4,
                "pop_count": [100, 200, 300, 999],
            }
        )
        population.to_parquet(
            root / "data/active/curated/population" / f"{row.city_key}_pop.parquet", index=False
        )
        sentinel = pd.DataFrame(
            {
                **base,
                "city": [row.city_key] * 4,
                "NDVI": [0.1, 0.2, 0.3, 9.0],
                "NDBI": [0.4, 0.3, 0.2, 9.0],
            }
        )
        sentinel.to_parquet(
            root / "data/active/curated/sentinel2" / f"{row.city_key}_s2.parquet", index=False
        )


def _treatments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "treatment_order": [1, 2, 3],
            "city_key": ["a", "b", "c"],
            "grid_id": ["g1", "g2", "g3"],
            "station_event_id": ["s1", "s2", "s3"],
            "opening_month": ["2020-06"] * 3,
        }
    )


def test_pretreatment_features_exclude_opening_and_post_years(tmp_path: Path) -> None:
    treatments = _treatments()
    _write_sources(tmp_path, treatments)
    features, used = build_pretreatment_features(treatments, tmp_path, strict_sources=True)
    assert len(features) == 3
    assert used
    assert np.allclose(features["housing_log_price__lag1"], 3.0)
    assert np.allclose(features["housing_log_price__lag3"], 1.0)
    assert not (features.select_dtypes("number") == 999).any().any()
    assert (
        features[
            ["poi_available", "viirs_available", "population_available", "sentinel2_available"]
        ]
        .all()
        .all()
    )


def test_city_split_is_deterministic_and_disjoint() -> None:
    first = deterministic_city_splits(["a", "b", "c", "d", "e"], seed="frozen")
    second = deterministic_city_splits(reversed(["a", "b", "c", "d", "e"]), seed="frozen")
    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}


def test_pretraining_release_uses_response_hash_and_city_splits(tmp_path: Path) -> None:
    treatments = _treatments()
    _write_sources(tmp_path, treatments)
    treatment_path = tmp_path / "treatments.parquet"
    treatments.to_parquet(treatment_path, index=False)
    response_release = tmp_path / "response"
    response_release.mkdir()
    response = pd.DataFrame(
        {
            "treatment_order": [1, 2, 3],
            "outcome_family": ["population"] * 3,
            "city_key": ["a", "b", "c"],
            "grid_id": ["g1", "g2", "g3"],
            "station_event_id": ["s1", "s2", "s3"],
            "opening_month": ["2020-06"] * 3,
            "outcome": ["population_log"] * 3,
            "event_time": [1] * 3,
            "specification_id": ["main_a6_r1km"] * 3,
            "training_mask": [True] * 3,
            "quality_grade": ["matched_same_city_pass"] * 3,
        }
    )
    response_path = response_release / "response_artifact.parquet"
    response.to_parquet(response_path, index=False)
    (response_release / "manifest.json").write_text(
        json.dumps(
            {
                "strict_production": False,
                "artifact": {"sha256": sha256_file(response_path)},
                "source_files": {"treatments": {"sha256": sha256_file(treatment_path)}},
            }
        ),
        encoding="utf-8",
    )
    output = publish_pretraining_dataset(
        response_release,
        treatment_path,
        tmp_path,
        tmp_path / "model_inputs",
        dataset_id="audit",
        min_modalities=2,
        strict_production=False,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    index = pd.read_parquet(output / "sample_index.parquet")
    assert manifest["response_artifact_sha256"] == sha256_file(response_path)
    assert manifest["response_bound_treatments_sha256"] == sha256_file(treatment_path)
    assert set(index["split"]) == {"train", "validation", "test"}
    assert index["final_training_mask"].all()


def test_pretraining_release_rejects_treatment_file_not_bound_to_response(
    tmp_path: Path,
) -> None:
    response_treatments = _treatments()
    bound_path = tmp_path / "bound_treatments.parquet"
    response_treatments.to_parquet(bound_path, index=False)
    response = pd.DataFrame(
        {
            **{column: response_treatments[column] for column in IDENTITY_COLUMNS},
            "outcome_family": ["population"] * 3,
            "outcome": ["population_log"] * 3,
            "event_time": [1] * 3,
            "specification_id": ["main_a6_r1km"] * 3,
            "training_mask": [True] * 3,
            "quality_grade": ["matched_same_city_pass"] * 3,
        }
    )
    response_release = tmp_path / "response"
    response_release.mkdir()
    response_path = response_release / "response_artifact.parquet"
    response.to_parquet(response_path, index=False)
    (response_release / "manifest.json").write_text(
        json.dumps(
            {
                "strict_production": False,
                "artifact": {"sha256": sha256_file(response_path)},
                "source_files": {"treatments": {"sha256": sha256_file(bound_path)}},
            }
        ),
        encoding="utf-8",
    )

    wrong_treatments = response_treatments.copy()
    wrong_treatments["city_key"] = ["x", "y", "z"]
    wrong_path = tmp_path / "wrong_treatments.parquet"
    wrong_treatments.to_parquet(wrong_path, index=False)
    with pytest.raises(ValueError, match="hash disagrees"):
        publish_pretraining_dataset(
            response_release,
            wrong_path,
            tmp_path,
            tmp_path / "model_inputs",
            dataset_id="must_fail",
            strict_production=False,
        )


def test_treatment_binding_rejects_missing_production_hash_and_identity_drift(
    tmp_path: Path,
) -> None:
    count = 5_048
    treatments = pd.DataFrame(
        {
            "treatment_order": np.arange(1, count + 1),
            "city_key": ["city"] * count,
            "grid_id": [f"g{index}" for index in range(count)],
            "station_event_id": [f"s{index}" for index in range(count)],
            "opening_month": ["2020-01"] * count,
        }
    )
    treatment_path = tmp_path / "treatments.parquet"
    treatments.to_parquet(treatment_path, index=False)
    with pytest.raises(ValueError, match="lacks its treatment-list hash"):
        validate_response_treatment_binding(
            {}, treatments.copy(), treatments, treatment_path, strict_production=True
        )

    drifted = treatments.copy()
    drifted.loc[drifted.index[0], "city_key"] = "wrong_city"
    manifest: dict[str, object] = {
        "source_files": {"treatments": {"sha256": sha256_file(treatment_path)}}
    }
    with pytest.raises(ValueError, match="city_key disagrees"):
        validate_response_treatment_binding(
            manifest, drifted, treatments, treatment_path, strict_production=False
        )


def _weight_frame() -> pd.DataFrame:
    """Synthetic panel with >=30 eligible labels per (method, outcome) stratum."""
    matched = pd.DataFrame(
        {
            "treatment_order": np.arange(1, 41),
            "outcome_family": "viirs",
            "outcome": "viirs_avg_asinh",
            "method": "frozen_matched_change_12m_baseline",
            "label_available": True,
            "holdout_rms_ratio": np.linspace(0.10, 0.90, 40),
            "holdout_max_ratio": np.linspace(0.15, 0.95, 40),
            "pre_rmspe": np.nan,
            "pre_mean_effect": np.nan,
            "mc_cv_mspe": np.nan,
            "standard_error": np.nan,
            "pre_observed_periods": 5,
        }
    )
    gsc = pd.DataFrame(
        {
            "treatment_order": np.arange(101, 141),
            "outcome_family": "viirs",
            "outcome": "viirs_avg_asinh",
            "method": "xu_2017_gsynth_same_city",
            "label_available": True,
            "holdout_rms_ratio": np.nan,
            "holdout_max_ratio": np.nan,
            "pre_rmspe": np.linspace(0.05, 0.60, 40),
            "pre_mean_effect": 0.0,
            "mc_cv_mspe": np.nan,
            "standard_error": np.linspace(0.05, 0.50, 40),
            "pre_observed_periods": 5,
        }
    )
    mc = pd.DataFrame(
        {
            "treatment_order": np.arange(201, 241),
            "outcome_family": "viirs",
            "outcome": "viirs_avg_asinh",
            "method": "athey_2021_mc_same_city",
            "label_available": True,
            "holdout_rms_ratio": np.nan,
            "holdout_max_ratio": np.nan,
            "pre_rmspe": np.linspace(0.05, 0.60, 40),
            "pre_mean_effect": 0.0,
            "mc_cv_mspe": np.linspace(0.01, 0.10, 40),
            "standard_error": 0.20,
            "pre_observed_periods": 5,
        }
    )
    ineligible = pd.DataFrame(
        [
            {
                "treatment_order": 901,
                "outcome_family": "viirs",
                "outcome": "viirs_avg_asinh",
                "method": "frozen_matched_change_12m_baseline",
                "label_available": True,
                "holdout_rms_ratio": 1.50,
                "holdout_max_ratio": 1.50,
                "pre_rmspe": np.nan,
                "pre_mean_effect": np.nan,
                "mc_cv_mspe": np.nan,
                "standard_error": np.nan,
                "pre_observed_periods": 5,
            },
            {
                "treatment_order": 902,
                "outcome_family": "viirs",
                "outcome": "viirs_avg_asinh",
                "method": "xu_2017_gsynth_same_city",
                "label_available": True,
                "holdout_rms_ratio": np.nan,
                "holdout_max_ratio": np.nan,
                "pre_rmspe": 0.10,
                "pre_mean_effect": 0.0,
                "mc_cv_mspe": np.nan,
                "standard_error": 0.10,
                "pre_observed_periods": 0,
            },
            {
                "treatment_order": 903,
                "outcome_family": "viirs",
                "outcome": "viirs_avg_asinh",
                "method": "athey_2021_mc_same_city",
                "label_available": False,
                "holdout_rms_ratio": np.nan,
                "holdout_max_ratio": np.nan,
                "pre_rmspe": 0.10,
                "pre_mean_effect": 0.0,
                "mc_cv_mspe": 0.02,
                "standard_error": 0.20,
                "pre_observed_periods": 5,
            },
        ]
    )
    return pd.concat([matched, gsc, mc, ineligible], ignore_index=True)


def test_response_label_weights_rank_within_strata() -> None:
    weighted = response_label_weights(_weight_frame()).set_index("treatment_order")

    matched = weighted.loc[weighted["method"].eq("frozen_matched_change_12m_baseline")]
    assert matched.loc[1, "response_weight"] > matched.loc[40, "response_weight"]
    assert not matched.loc[1, "has_precision_weight"]
    assert matched.loc[1, "precision_weight"] == pytest.approx(1.0)

    gsc = weighted.loc[weighted["method"].eq("xu_2017_gsynth_same_city")]
    assert gsc.loc[101, "response_weight"] > gsc.loc[140, "response_weight"]
    mc = weighted.loc[weighted["method"].eq("athey_2021_mc_same_city")]
    assert mc.loc[201, "response_weight"] > mc.loc[240, "response_weight"]
    assert gsc.loc[101, "has_precision_weight"]
    assert mc.loc[201, "has_precision_weight"]

    for method in weighted["method"].unique():
        group = weighted[(weighted["method"] == method) & weighted["eligible"]]
        assert group["response_weight"].mean() == pytest.approx(1.0)
    assert set(weighted["calibration_id"]) == {"ordinal-design-precision"}


def test_response_weights_average_available_design_components() -> None:
    weighted = response_label_weights(_weight_frame())

    mc = weighted.loc[
        weighted["method"].eq("athey_2021_mc_same_city") & weighted["eligible"]
    ]
    assert np.allclose(mc["design_unreliability"], (mc["p_trend"] + mc["p_validation"]) / 2)
    matched = weighted.loc[
        weighted["method"].eq("frozen_matched_change_12m_baseline") & weighted["eligible"]
    ]
    assert np.allclose(matched["design_unreliability"], matched["p_holdout"])
    gsc = weighted.loc[
        weighted["method"].eq("xu_2017_gsynth_same_city") & weighted["eligible"]
    ]
    assert np.allclose(gsc["design_unreliability"], gsc["p_trend"])


def test_response_weights_pool_small_outcomes_to_family() -> None:
    outcomes = [
        "poi_count_log",
        "poi_category_entropy",
        "poi_commercial_share",
        "poi_transport_access_log",
    ]
    rows = []
    for index, outcome in enumerate(outcomes):
        for position in range(10):
            rows.append(
                {
                    "treatment_order": 1000 + index * 10 + position,
                    "outcome_family": "poi",
                    "outcome": outcome,
                    "method": "xu_2017_gsynth_same_city",
                    "label_available": True,
                    "pre_rmspe": 0.05 + 0.001 * (index * 10 + position),
                    "pre_mean_effect": 0.0,
                    "standard_error": 0.10,
                    "pre_observed_periods": 5,
                }
            )
    weighted = response_label_weights(pd.DataFrame(rows))
    ordered = weighted.sort_values("p_trend").reset_index(drop=True)
    # n = 10 per outcome (< 30), so percentiles are computed on the pooled
    # (method, family) distribution of 40 labels, not per outcome.
    assert ordered.loc[0, "p_trend"] == pytest.approx(1 / 40)
    assert ordered.loc[9, "p_trend"] == pytest.approx(10 / 40)
    assert set(weighted["weight_stratum"]) == {"xu_2017_gsynth_same_city::poi"}


def test_response_weights_sensitivity_and_eligibility() -> None:
    frame = _weight_frame()
    weighted = response_label_weights(frame).set_index("treatment_order")
    for order in (901, 902, 903):
        assert not weighted.loc[order, "eligible"]
        assert weighted.loc[order, "response_weight"] == 0.0
    assert (weighted.loc[weighted["eligible"], "response_weight"] > 0).all()

    equal = response_label_weights(frame, gamma=0.0, alpha=0.0).set_index(
        "treatment_order"
    )
    eligible = equal.loc[equal["eligible"]]
    assert eligible["response_weight"].nunique() == 1
    assert eligible["response_weight"].iloc[0] == pytest.approx(1.0)


def test_task_sampling_weight_ignores_ineligible_labels() -> None:
    targets = pd.DataFrame(
        {
            "treatment_order": [1, 1, 2],
            "outcome_family": ["poi"] * 3,
            "outcome": ["poi_count_log"] * 3,
            "method": ["athey_2021_mc_same_city"] * 3,
            "label_available": [True, True, True],
            "pre_rmspe": [0.10, 0.05, 0.05],
            "pre_mean_effect": [0.0, 0.0, 0.0],
            "mc_cv_mspe": [0.01, 0.01, 0.01],
            "standard_error": [0.1, 0.1, 0.1],
            "pre_observed_periods": [5, 0, 0],
        }
    )
    per_task = _task_sampling_weight(targets).set_index("treatment_order")[
        "task_sampling_weight"
    ]
    assert per_task.loc[1] > 0
    assert 2 not in per_task.index


def test_response_weights_zero_anchor_and_unavailable_labels() -> None:
    targets = pd.DataFrame(
        {
            "treatment_order": [1, 2, 3],
            "outcome_family": ["poi"] * 3,
            "outcome": ["poi_count_log"] * 3,
            "method": ["athey_2021_mc_same_city"] * 3,
            "label_available": [True, False, True],
            "pre_rmspe": [0.05, 0.01, 0.05],
            "pre_mean_effect": [0.0, 0.0, 0.0],
            "standard_error": [0.10, 0.10, 0.10],
            "pre_observed_periods": [5, 5, 0],
        }
    )
    weighted = response_label_weights(targets).set_index("treatment_order")
    assert weighted.loc[1, "eligible"]
    assert weighted.loc[1, "response_weight"] == pytest.approx(1.0)
    assert not weighted.loc[2, "eligible"]
    assert weighted.loc[2, "response_weight"] == 0.0
    assert not weighted.loc[3, "eligible"]
    assert weighted.loc[3, "response_weight"] == 0.0
    per_task = _task_sampling_weight(targets)
    assert set(per_task["treatment_order"]) == {1}
    assert per_task["task_sampling_weight"].iloc[0] == pytest.approx(1.0)


def test_response_weights_treat_nonfinite_se_as_neutral() -> None:
    frame = _weight_frame()
    gsc = frame["method"].eq("xu_2017_gsynth_same_city")
    first = frame.index[gsc][:3]
    frame.loc[first, "standard_error"] = np.inf

    weighted = response_label_weights(frame, gamma=1.0, alpha=0.5)

    assert weighted.loc[first, "eligible"].all()
    assert not weighted.loc[first, "has_precision_weight"].any()
    assert weighted.loc[first, "precision_weight"].eq(1.0).all()
    eligible = weighted.loc[weighted["eligible"], "response_weight"]
    assert (eligible > 0).all()


def test_pretraining_release_family_filter_and_response_weights(tmp_path: Path) -> None:
    treatments = _treatments()
    _write_sources(tmp_path, treatments)
    treatment_path = tmp_path / "treatments.parquet"
    treatments.to_parquet(treatment_path, index=False)
    response_release = tmp_path / "response"
    response_release.mkdir()
    frames = []
    for family, outcome in (("population", "population_log"), ("viirs", "viirs_avg_asinh")):
        frames.append(
            pd.DataFrame(
                {
                    **{column: treatments[column] for column in IDENTITY_COLUMNS},
                    "outcome_family": family,
                    "outcome": outcome,
                    "event_time": [1, 1, 1],
                    "specification_id": "main_a6_r1km",
                    "training_mask": True,
                    "quality_grade": "matched_same_city_pass",
                    "pretrend_slope_p_value": 0.50,
                    "pre_observed_periods": 4,
                }
            )
        )
    response = pd.concat(frames, ignore_index=True)
    response_path = response_release / "response_artifact.parquet"
    response.to_parquet(response_path, index=False)
    (response_release / "manifest.json").write_text(
        json.dumps(
            {
                "strict_production": False,
                "artifact": {"sha256": sha256_file(response_path)},
                "source_files": {"treatments": {"sha256": sha256_file(treatment_path)}},
            }
        ),
        encoding="utf-8",
    )

    output = publish_pretraining_dataset(
        response_release,
        treatment_path,
        tmp_path,
        tmp_path / "model_inputs",
        dataset_id="viirs_only",
        min_modalities=2,
        strict_production=False,
        families=("viirs",),
    )
    targets = pd.read_parquet(output / "response_targets.parquet")
    index = pd.read_parquet(output / "sample_index.parquet")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert set(targets["outcome_family"]) == {"viirs"}
    assert manifest["families"] == ["viirs"]
    # Without pre-treatment diagnostics every label receives the neutral design
    # score, so the normalized weights are exactly one.
    assert index["task_sampling_weight"].eq(1.0).all()
    assert manifest["response_weight"]["mean"] == pytest.approx(1.0)
    assert manifest["response_weight"]["eligible_rows"] == 3
    assert manifest["response_weight"]["calibration_id"] == "ordinal-design-precision"
    assert manifest["response_weight"]["gamma"] == pytest.approx(1.0)
    assert manifest["response_weight"]["alpha"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="unknown outcome families"):
        publish_pretraining_dataset(
            response_release,
            treatment_path,
            tmp_path,
            tmp_path / "model_inputs",
            dataset_id="must_fail",
            strict_production=False,
            families=("unknown",),
        )
