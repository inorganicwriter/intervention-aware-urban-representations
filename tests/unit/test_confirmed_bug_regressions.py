from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from urban_intervention.data.paths import PROJECT_ROOT
from urban_intervention.pipelines.poi import sources

ROOT = PROJECT_ROOT


def _load_script(name: str, relative_path: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_poi_disk_csv_fallback_has_a_traceable_source(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "city.csv"
    csv_path.write_bytes(b"name,lon,lat\n")
    monkeypatch.setattr(sources, "find_disk_city_csv", lambda city, year: csv_path)

    with sources.open_city_csv("beijing", 2017) as (handle, source_name):
        assert handle.read().startswith(b"name")

    assert source_name == str(csv_path.resolve())


def test_poi_zip_is_closed_when_member_open_fails(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "year.zip"
    archive.write_bytes(b"placeholder")
    opened = []

    class FailingZip:
        def __init__(self, _path):
            self.closed = False
            opened.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

        def infolist(self):
            return [SimpleNamespace(filename="city.csv")]

        def open(self, _info):
            raise OSError("damaged member")

    monkeypatch.setattr(sources, "find_disk_city_csv", lambda *_args: None)
    monkeypatch.setattr(sources, "find_disk_nested_year_zip", lambda _year: archive)
    monkeypatch.setattr(sources, "matches_city_csv", lambda *_args: True)
    monkeypatch.setattr(sources, "ZipFile", FailingZip)

    with pytest.raises(OSError, match="damaged member"), sources.open_city_csv("beijing", 2017):
        pass
    assert len(opened) == 1
    assert opened[0].closed


def test_station_audit_import_has_no_io_and_accepts_absent_source_column(capsys) -> None:
    module = _load_script(
        "audit_station_quality_regression", "scripts/analysis/audit_station_quality.py"
    )
    frame = pd.DataFrame(
        {
            "city_key": ["beijing"],
            "station_event_id": ["s1"],
            "canonical_station_name": ["station"],
            "wgs84_lon": [116.4],
            "wgs84_lat": [39.9],
            "opening_year": [2020],
            "opening_month": [1],
            "opening_day": [1],
            "date_precision": ["day"],
        }
    )
    assert capsys.readouterr().out == ""
    module.audit(frame)
    assert "Date sources:      not recorded in this table" in capsys.readouterr().out


def test_poi_year_filter_is_atomic_and_preserves_backup(tmp_path: Path) -> None:
    module = _load_script(
        "filter_poi_panel_years_regression", "scripts/analysis/filter_poi_panel_years.py"
    )
    path = tmp_path / "panel.parquet"
    original = pd.DataFrame({"year": [2011, 2012, 2013], "value": [1, 2, 3]})
    original.to_parquet(path, index=False)

    count, backup = module.filter_years(path, 2012, 2013)

    assert count == 2
    pd.testing.assert_frame_equal(pd.read_parquet(backup), original)
    pd.testing.assert_frame_equal(
        pd.read_parquet(path).reset_index(drop=True), original.iloc[1:].reset_index(drop=True)
    )


def test_poi_year_filter_refuses_to_overwrite_backup(tmp_path: Path) -> None:
    module = _load_script(
        "filter_poi_panel_years_backup_regression", "scripts/analysis/filter_poi_panel_years.py"
    )
    path = tmp_path / "panel.parquet"
    original = pd.DataFrame({"year": [2011, 2012]})
    original.to_parquet(path, index=False)
    backup = tmp_path / "existing.parquet"
    backup.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="existing backup"):
        module.filter_years(path, 2012, 2012, backup)

    pd.testing.assert_frame_equal(pd.read_parquet(path), original)
    assert backup.read_bytes() == b"keep"


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/causal_python/run_causal_label_queue.py",
        "scripts/causal_r/run_grid_control_design_queue.py",
        "scripts/collection/ensure_viirs_monthly_cache.py",
        "scripts/collection/package_streetview.py",
    ],
)
def test_production_scripts_contain_no_machine_specific_paths(relative_path: str) -> None:
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    forbidden = (r"D:\R-4.6.1", r"E:\Data\MIT_Summer_VIIRS", "/home/nas/lsr")
    assert not any(value in source for value in forbidden)
