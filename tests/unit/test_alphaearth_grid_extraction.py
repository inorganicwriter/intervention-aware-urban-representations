from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from scripts.collection import extract_alphaearth_embeddings as collector
from urban_intervention.causal.pretraining_dataset import attach_satellite_embeddings


def _inputs(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    grid_root = tmp_path / "grids"
    city_dir = grid_root / "city"
    city_dir.mkdir(parents=True)
    pd.DataFrame(
        {"grid_id": ["g"], "geometry_wkt": [box(117, 30, 117.005, 30.005).wkt]}
    ).to_parquet(city_dir / "city_grids.parquet", index=False)
    treatments = pd.DataFrame(
        {
            "treatment_order": [1, 2],
            "city_key": ["city", "city"],
            "grid_id": ["g", "g"],
            "wgs84_lon": [117.001, 117.004],
            "wgs84_lat": [30.001, 30.004],
            "opening_month": ["2016-06", "2020-06"],
        }
    )
    return treatments, grid_root


def test_geometry_is_fixed_by_grid_not_station_coordinates(tmp_path: Path) -> None:
    treatments, grid_root = _inputs(tmp_path)
    bound = collector.bind_grid_polygons("city", treatments, grid_root)
    assert bound.geometry_wkt.nunique() == 1
    assert bound.grid_geometry_sha256.nunique() == 1
    assert bound.spatial_support.eq("fixed_grid_polygon_v1").all()


@pytest.mark.parametrize("fault", ["missing_grid", "outside_station", "wrong_city"])
def test_invalid_spatial_binding_is_rejected(tmp_path: Path, fault: str) -> None:
    treatments, grid_root = _inputs(tmp_path)
    if fault == "missing_grid":
        treatments.loc[0, "grid_id"] = "unknown"
    elif fault == "outside_station":
        treatments.loc[0, "wgs84_lon"] = 118
    else:
        treatments.loc[0, "city_key"] = "other"
    with pytest.raises(ValueError):
        collector.bind_grid_polygons("city", treatments, grid_root)


def test_dry_run_never_claims_real_available_embeddings(tmp_path: Path) -> None:
    treatments, grid_root = _inputs(tmp_path)
    result = collector.extract_city_alphaearth(
        "city", treatments, dry_run=True, grid_dir=grid_root
    )
    assert len(result) == 2
    assert not result.alphaearth_available.any()
    assert result[collector.BANDS].isna().all().all()
    assert result.is_dry_run.all()


def test_strict_collection_preserves_early_events_without_querying_future_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    treatments, grid_root = _inputs(tmp_path)
    queried_years = []

    def fake_batch(city: str, year: int, batch: pd.DataFrame, metadata) -> list[dict]:
        queried_years.append(year)
        return [
            {
                **metadata(row),
                **{band: 0.1 for band in collector.BANDS},
                "alphaearth_available": True,
                "availability_reason": "available",
            }
            for _, row in batch.iterrows()
        ]

    monkeypatch.setattr(collector, "_extract_batch", fake_batch)
    result = collector.extract_city_alphaearth("city", treatments, grid_dir=grid_root)
    assert queried_years == [2019]
    early = result.set_index("treatment_order").loc[1]
    assert early.target_year == 2015
    assert not early.alphaearth_available
    assert not early.is_temporal_proxy
    assert np.isnan(early.A00)
    assert len(result) == 2


def test_proxy_requires_explicit_opt_in(tmp_path: Path) -> None:
    treatments, grid_root = _inputs(tmp_path)
    result = collector.extract_city_alphaearth(
        "city", treatments, dry_run=True, grid_dir=grid_root, allow_temporal_proxy=True
    )
    early = result.set_index("treatment_order").loc[1]
    assert early.target_year == 2017
    assert early.is_temporal_proxy
    assert not early.alphaearth_available


def test_publication_preserves_spatial_binding_metadata(tmp_path: Path) -> None:
    treatments, grid_root = _inputs(tmp_path)
    source = collector.extract_city_alphaearth(
        "city", treatments, dry_run=True, grid_dir=grid_root
    )
    path = tmp_path / "source.parquet"
    source.to_parquet(path, index=False)
    attached, _ = attach_satellite_embeddings(treatments, path)
    assert attached.satellite_spatial_support.eq("fixed_grid_polygon_v1").all()
    assert attached.satellite_grid_geometry_sha256.tolist() == source.grid_geometry_sha256.tolist()
    assert not attached.alphaearth_available.any()


def test_publication_rejects_event_id_bound_to_wrong_grid(tmp_path: Path) -> None:
    treatments, grid_root = _inputs(tmp_path)
    source = collector.extract_city_alphaearth(
        "city", treatments, dry_run=True, grid_dir=grid_root
    )
    source.loc[0, "grid_id"] = "different_grid"
    path = tmp_path / "source.parquet"
    source.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="city/grid binding"):
        attach_satellite_embeddings(treatments, path)
