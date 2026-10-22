import hashlib
import json

import pandas as pd
import pytest

from scripts.collection import download_satellite_event_sequences as sequences
from scripts.collection.download_satellite_event_sequences import build_queue


def test_full_event_window_includes_boundary_and_future_rows_without_input_leakage() -> None:
    events = pd.DataFrame({"treatment_order": [1], "city_key": ["city"], "grid_id": ["grid"],
                           "opening_month": ["2025-12"]})
    queue = build_queue(events, 2025)
    assert queue.event_time.tolist() == list(range(-6, 4))
    assert queue.target_year.tolist() == list(range(2019, 2029))
    assert queue.state_temporal_allowed.tolist() == [True] * 6 + [False] * 4
    assert queue.calendar_year_complete.tolist() == [True] * 7 + [False] * 3
    assert queue.temporal_phase.tolist() == ["pre"] * 6 + ["opening_year"] + ["post"] * 3


def test_duplicate_grid_cannot_acquire_two_unrelated_sequences() -> None:
    events = pd.DataFrame({"treatment_order": [1, 2], "city_key": ["city", "city"],
                           "grid_id": ["grid", "grid"], "opening_month": ["2020-01", "2021-01"]})
    with pytest.raises(ValueError, match="unique"):
        build_queue(events, 2025)


def test_memory_split_keeps_every_frame_in_original_order(tmp_path, monkeypatch) -> None:
    rows = [{"treatment_order": order, "city_key": "city", "target_year": 2024} for order in range(7)]

    def limited_download(chunk, sensor, raw, masked, output):
        if len(chunk) > 2:
            raise RuntimeError("User memory limit exceeded.")
        return [{"treatment_order": row["treatment_order"], "satellite_image_available": True} for row in chunk]

    monkeypatch.setattr(sequences, "acquire_packed", limited_download)
    result = sequences.acquire_resilient(rows, "sentinel2", None, None, tmp_path)
    assert [record["treatment_order"] for record in result] == list(range(7))
    assert all(record["satellite_image_available"] for record in result)


def test_next_year_refresh_reuses_verified_frames_without_changing_acquisition_date(tmp_path) -> None:
    old_contract = {"version": 2, "latest_complete_year": 2025, "min_valid_fraction": .8}
    new_contract = {**old_contract, "latest_complete_year": 2026}
    sequences.base._write_json(tmp_path / "satellite_sequence_audit.json", {"contract": old_contract})
    old_sha = hashlib.sha256(json.dumps(old_contract, sort_keys=True).encode()).hexdigest()
    new_sha = hashlib.sha256(json.dumps(new_contract, sort_keys=True).encode()).hexdigest()
    assert sequences.compatible_previous_contract(tmp_path, new_contract) == old_sha
    assert sequences.compatible_previous_contract(tmp_path, {**new_contract, "min_valid_fraction": .5}) is None
    for name in ["frame.tif", "rgb.png", "mask.png", "source.json"]:
        (tmp_path / name).write_bytes(b"cached verified asset")
    row = {"treatment_order": 1, "city_key": "city", "grid_id": "grid", "station_event_id": "event",
           "opening_month": "2025-01", "target_year": 2024, "event_time": -1, "temporal_phase": "pre",
           "calendar_year_complete": True, "state_temporal_allowed": True,
           "geometry_wkt": "polygon", "grid_geometry_sha256": "hash"}
    rec = {**sequences.metadata(row, old_sha), "acquired_at_utc": "2026-10-12T10:00:00Z",
           "satellite_image_available": True, "image_path": "frame.tif",
           "image_sha256": sequences.base._checksum(tmp_path / "frame.tif"),
           "preview_path": "rgb.png", "mask_path": "mask.png", "source_manifest_path": "source.json"}
    path = tmp_path / "record.json"
    sequences.base._write_json(path, rec)
    reused = sequences.resume_cached_frame(path, row, new_sha, tmp_path, old_sha)
    assert reused["acquired_at_utc"] == rec["acquired_at_utc"]
    assert reused["contract_sha256"] == new_sha
    assert reused["state_image_available"]
    (tmp_path / "frame.tif").write_bytes(b"corrupt cached frame")
    assert sequences.resume_cached_frame(path, row, new_sha, tmp_path, old_sha) is None
