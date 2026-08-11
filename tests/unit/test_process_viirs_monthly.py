from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "collection" / "process_viirs_monthly.py"
SPEC = importlib.util.spec_from_file_location("process_viirs_monthly", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PROJECTED_CRS = "EPSG:32650"
ORIGIN_X = 400_000.0
ORIGIN_Y = 4_300_000.0


def _to_wgs84(x: float, y: float) -> tuple[float, float]:
    lon, lat = Transformer.from_crs(PROJECTED_CRS, "EPSG:4326", always_xy=True).transform(
        float(x), float(y)
    )
    return float(lon), float(lat)


def _grids(include_second: bool = True) -> pd.DataFrame:
    cells = [("g1", 0, 0, ORIGIN_X + 250, ORIGIN_Y + 250)]
    if include_second:
        cells.append(("g2", 0, 1, ORIGIN_X + 750, ORIGIN_Y + 250))
    rows = []
    for grid_id, row, col, x, y in cells:
        lon, lat = _to_wgs84(x, y)
        rows.append(
            {
                "grid_id": grid_id,
                "row": row,
                "col": col,
                "centroid_lon": lon,
                "centroid_lat": lat,
            }
        )
    return pd.DataFrame(rows)


def _points_xy(coordinates: list[tuple[float, float]]) -> pd.DataFrame:
    lon_lat = [_to_wgs84(x, y) for x, y in coordinates]
    return pd.DataFrame({"lon": [p[0] for p in lon_lat], "lat": [p[1] for p in lon_lat]})


def test_exact_lattice_assignment_does_not_fill_clipped_neighbor_cell() -> None:
    # The second point is only 260 m from g1's centroid, so the retired nearest
    # method assigned it to g1 even though it lies in clipped-out column 1.
    points = _points_xy([(ORIGIN_X + 100, ORIGIN_Y + 100), (ORIGIN_X + 510, ORIGIN_Y + 250)])
    lattice = MODULE.build_grid_lattice(_grids(include_second=False), PROJECTED_CRS)

    matched, diagnostics = MODULE.assign_grid_exact(points, lattice)

    assert matched["grid_id"].tolist() == ["g1"]
    assert diagnostics == {
        "source_rows": 2,
        "invalid_coordinate_rows": 0,
        "outside_reference_grid_rows": 1,
        "matched_rows": 1,
    }


def test_exact_lattice_assignment_handles_two_cells_and_invalid_coordinates() -> None:
    points = _points_xy(
        [
            (ORIGIN_X + 100, ORIGIN_Y + 100),
            (ORIGIN_X + 600, ORIGIN_Y + 100),
            (ORIGIN_X + 1_100, ORIGIN_Y + 100),
        ]
    )
    points.loc[len(points)] = [float("nan"), 39.0]
    lattice = MODULE.build_grid_lattice(_grids(), PROJECTED_CRS)

    matched, diagnostics = MODULE.assign_grid_exact(points, lattice)

    assert matched["grid_id"].tolist() == ["g1", "g2"]
    assert matched["cell_edge_distance_m"].min() == pytest.approx(100, abs=0.01)
    assert diagnostics == {
        "source_rows": 4,
        "invalid_coordinate_rows": 1,
        "outside_reference_grid_rows": 1,
        "matched_rows": 2,
    }


def test_process_directory_writes_compact_partition_and_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "downloads" / "nested"
    input_dir.mkdir(parents=True)
    grid_dir = tmp_path / "grids" / "beijing"
    grid_dir.mkdir(parents=True)
    output_dir = tmp_path / "curated"
    audit_dir = tmp_path / "audits"
    _grids().to_parquet(grid_dir / "beijing_grids.parquet", index=False)

    positions = [
        _to_wgs84(ORIGIN_X + 100, ORIGIN_Y + 100),
        _to_wgs84(ORIGIN_X + 200, ORIGIN_Y + 100),
        _to_wgs84(ORIGIN_X + 200, ORIGIN_Y + 100),
    ]
    rows = pd.DataFrame(
        {
            "city": ["beijing"] * 3,
            "year": [2012] * 3,
            "month": [1] * 3,
            "period": ["2012-01"] * 3,
            "product": [MODULE.SOURCE_PRODUCT] * 3,
            "radiance": [10.0, 14.0, 14.0],
            "valid_days": [20, 22, 22],
            "longitude": [p[0] for p in positions],
            "latitude": [p[1] for p in positions],
        }
    )
    rows.to_csv(input_dir / "viirs_beijing_2012_01.csv", index=False)

    monkeypatch.setattr(MODULE, "GRID_DIR", tmp_path / "grids")
    outputs = MODULE.process_directory(
        tmp_path / "downloads", output_dir=output_dir, audit_dir=audit_dir
    )

    expected = output_dir / "city_key=beijing" / "year=2012" / "month=01" / "part.parquet"
    assert outputs == [expected]
    result = pd.read_parquet(expected)
    assert result.columns.tolist() == MODULE.STORED_COLUMNS
    assert len(result) == 1
    row = result.iloc[0]
    assert row["grid_id"] == "g1"
    assert row["avg_rad"] == pytest.approx(12.0)
    assert row["valid_days_mean"] == pytest.approx(21.0)
    assert row["source_point_count"] == 2
    assert result["avg_rad"].dtype == "float32"
    assert result["source_point_count"].dtype == "uint16"
    audit = pd.read_json(audit_dir / "beijing" / "2012-01.json", typ="series")
    assert audit["duplicate_coordinate_rows_removed"] == 1
    assert audit["assignment_method"] == "exact_projected_lattice_half_open_cells"
    assert audit["within_grid_aggregation"] == "unweighted_mean_of_unique_source_coordinates"
    assert audit["reference_grid_rows"] == 2
    assert audit["grid_coverage_share"] == pytest.approx(0.5)


def test_same_coordinate_with_conflicting_radiance_is_rejected() -> None:
    lon, lat = _to_wgs84(ORIGIN_X + 100, ORIGIN_Y + 100)
    points = pd.DataFrame(
        {
            "city_key": ["beijing", "beijing"],
            "year": [2012, 2012],
            "month": [1, 1],
            "lon": [lon, lon],
            "lat": [lat, lat],
            "avg_rad": [10.0, 11.0],
            "valid_days": [20.0, 20.0],
            "grid_id": ["g1", "g1"],
        }
    )
    with pytest.raises(ValueError, match="same-coordinate groups with conflicting values"):
        MODULE.canonicalize_source_coordinates(points)


def test_complete_manifest_fails_before_processing() -> None:
    export = MODULE.ExportFile(Path("viirs_beijing_2012_01.csv"), "beijing", 2012, 1)
    with pytest.raises(ValueError, match="city manifest mismatch"):
        MODULE.validate_complete_batch([export])


def test_empty_export_produces_an_empty_contract_frame() -> None:
    export = MODULE.ExportFile(Path("viirs_beijing_2012_01.csv"), "beijing", 2012, 1)
    empty = pd.DataFrame(
        columns=[
            "city",
            "year",
            "month",
            "radiance",
            "valid_days",
            "longitude",
            "latitude",
        ]
    )
    normalized = MODULE.normalize_export(empty, export)
    lattice = MODULE.build_grid_lattice(_grids(), PROJECTED_CRS)
    matched, diagnostics = MODULE.assign_grid_exact(normalized, lattice)
    result = MODULE.aggregate_grid_month(matched)

    assert result.empty
    assert result.columns.tolist() == MODULE.OUTPUT_COLUMNS
    assert diagnostics["source_rows"] == 0
