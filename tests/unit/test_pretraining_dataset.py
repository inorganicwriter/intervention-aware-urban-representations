from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urban_intervention.causal.pretraining_dataset import (
    IDENTITY_COLUMNS,
    build_pretreatment_features,
    deterministic_city_splits,
    publish_pretraining_dataset,
    validate_response_treatment_binding,
)
from urban_intervention.causal.response_artifact import sha256_file


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
