"""Shared test helpers for representation learning tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from urban_intervention.representation.dataset import RESPONSE_CONFIG


def build_synthetic_model_inputs(
    tmp_path: Path, n_grids: int = 12, n_cities: int = 4
) -> tuple[Path, int]:
    """Create a synthetic pretraining dataset on disk; return (dir, n_grids).

    The default 12-grid / 4-city layout matches historical tests.  Larger
    layouts (e.g. 24 grids / 8 cities) exercise evaluation paths that need
    bigger train pools.
    """
    output = tmp_path / "model_inputs" / "test_ds"
    output.mkdir(parents=True)

    cities = sorted(f"city_{i}" for i in range(n_cities))
    n_per_city = n_grids // len(cities)
    city_cycle = (cities * n_per_city)[:n_grids]
    split_map = {
        city: ("train" if index % 4 < 2 else "validation" if index % 4 == 2 else "test")
        for index, city in enumerate(cities)
    }
    units = pd.DataFrame(
        {
            "treatment_order": list(range(1, n_grids + 1)),
            "city_key": city_cycle,
            "grid_id": [f"g{idx}" for idx in range(1, n_grids + 1)],
            "station_event_id": [f"s{idx}" for idx in range(1, n_grids + 1)],
            "opening_month": ["2020-06"] * n_grids,
            "opening_year": [2020] * n_grids,
            "split": [split_map[c] for c in city_cycle],
        }
    )

    for lag in (1, 2, 3):
        units[f"z__housing_log_price__lag{lag}"] = np.random.RandomState(lag * 42).uniform(
            0, 1, n_grids
        )
        units[f"z__poi_count_log__lag{lag}"] = np.random.RandomState(100 + lag * 42).uniform(
            0, 1, n_grids
        )
        units[f"z__viirs_avg_asinh__lag{lag}"] = np.random.RandomState(200 + lag * 42).uniform(
            0, 1, n_grids
        )
        units[f"z__population_log__lag{lag}"] = np.random.RandomState(300 + lag * 42).uniform(
            0, 1, n_grids
        )
        units[f"z__sentinel2_ndvi__lag{lag}"] = np.random.RandomState(400 + lag * 42).uniform(
            -1, 1, n_grids
        )
        units[f"z__sentinel2_ndbi__lag{lag}"] = np.random.RandomState(500 + lag * 42).uniform(
            -1, 1, n_grids
        )

    for mod in ("housing", "poi", "viirs", "population", "sentinel2"):
        units[f"{mod}_available"] = True
    units["streetview_available"] = False
    units["streetview_assets"] = "[]"
    units["available_modality_count"] = 5
    units["feature_training_mask"] = True

    units.to_parquet(output / "unit_features.parquet", index=False)

    response_rows: list[dict] = []
    rng = np.random.RandomState(777)
    training_orders = n_grids * 2 // 3
    for order in range(1, n_grids + 1):
        for family, outcomes in RESPONSE_CONFIG.items():
            for outcome, horizons in outcomes.items():
                for t in horizons:
                    response_rows.append(
                        {
                            "treatment_order": order,
                            "outcome_family": family,
                            "outcome": outcome,
                            "event_time": t,
                            "causal_response_label": rng.normal(0.1, 0.3),
                            "standard_error": 0.05,
                            "label_available": True,
                            "training_mask": order <= training_orders,
                            "quality_grade": (
                                "matched_same_city_pass"
                                if order <= training_orders
                                else "unavailable"
                            ),
                            "specification_id": "main_a6_r1km",
                        }
                    )
    response = pd.DataFrame(response_rows)
    response["split"] = response["treatment_order"].map(
        dict(zip(units["treatment_order"], units["split"], strict=True))
    )
    response["feature_training_mask"] = True
    response["final_training_mask"] = response["training_mask"] & response["feature_training_mask"]
    response.to_parquet(output / "response_targets.parquet", index=False)

    sample_index = (
        response.groupby("treatment_order")
        .agg(
            outcome_family=("outcome_family", "first"),
            split=("split", "first"),
            training_mask=("training_mask", "first"),
            feature_training_mask=("feature_training_mask", "first"),
            final_training_mask=("final_training_mask", "first"),
            quality_grade=("quality_grade", "first"),
        )
        .reset_index()
    )
    sample_index.to_parquet(output / "sample_index.parquet", index=False)

    normalization = {}
    for col in units.columns:
        if col.startswith("z__"):
            vals = units[col].dropna()
            normalization[col] = {
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=0)),
                "train_observations": len(vals),
                "usable": True,
            }
    (output / "normalization.json").write_text(
        json.dumps(normalization, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "schema": "urban_intervention_pretraining_dataset_v1",
        "dataset_id": "test_ds",
        "created_utc": "2026-07-30T00:00:00Z",
        "strict_production": False,
        "response_release": "synthetic",
        "response_artifact_sha256": "0123456789abcdef",
        "treatments_sha256": "fedcba9876543210",
        "source_features": {"sha256": "abcdef", "files": 1, "bytes": 100},
        "feature_lags": [1, 2, 3],
        "feature_timing": "opening_year minus lag",
        "split_unit": "city_key",
        "split_seed": "mit-urban-v1",
        "splits": {
            "train": sorted(cities[:2]),
            "validation": [cities[2]] if len(cities) > 2 else [],
            "test": [cities[3]] if len(cities) > 3 else [],
        },
        "min_modalities": 2,
        "rows": {
            "unit_features": n_grids,
            "response_targets": len(response_rows),
            "sample_index": n_grids,
            "final_training": n_grids * 2 // 3,
        },
        "code": {"commit": "test", "dirty": False, "source": "git"},
        "runtime": {"python": "3.11.0", "platform": "test"},
        "outputs": {
            "unit_features.parquet": "aaaa",
            "response_targets.parquet": "bbbb",
            "sample_index.parquet": "cccc",
            "normalization.json": "dddd",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return output, n_grids
