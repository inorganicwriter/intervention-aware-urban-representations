from __future__ import annotations

import pandas as pd

from urban_intervention.causal.event_study import _read_monthly_viirs_cached
from urban_intervention.causal.gpu.panel_builder import read_monthly_viirs
from urban_intervention.data.paths import PROJECT_ROOT, VIIRS_MONTHLY_DIR


def _write_partition(
    root, city: str, year: int, month: int, grid_ids: list[str], values: list[float]
) -> None:
    directory = (
        root
        / VIIRS_MONTHLY_DIR.relative_to(PROJECT_ROOT)
        / f"city_key={city}"
        / f"year={year}"
        / f"month={month:02d}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"grid_id": grid_ids, "avg_rad": values}).to_parquet(
        directory / "part.parquet", index=False
    )


def test_cached_viirs_reader_matches_panel_builder(tmp_path) -> None:
    _write_partition(tmp_path, "alpha", 2012, 1, ["g1", "g2"], [1.0, 4.0])
    _write_partition(tmp_path, "alpha", 2012, 2, ["g1"], [2.0])
    months = pd.DatetimeIndex(["2012-01-01", "2012-02-01"])

    expected = read_monthly_viirs(tmp_path, "alpha", months)
    cached = _read_monthly_viirs_cached(tmp_path, "alpha", months)
    pd.testing.assert_frame_equal(cached, expected)

    cached.loc[cached["grid_id"].eq("g1"), "viirs_avg_asinh"] = 0.0
    again = _read_monthly_viirs_cached(tmp_path, "alpha", months)
    assert not again.loc[again["grid_id"].eq("g1"), "viirs_avg_asinh"].eq(0.0).all()


def test_cached_viirs_reader_returns_empty_frame_without_partitions(tmp_path) -> None:
    frame = _read_monthly_viirs_cached(
        tmp_path, "missing", pd.DatetimeIndex(["2012-01-01"])
    )
    assert frame.empty
    assert list(frame.columns) == ["city_key", "grid_id", "period", "viirs_avg_asinh"]
