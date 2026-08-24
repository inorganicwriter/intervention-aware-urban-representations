from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from urban_intervention.causal.gpu.provenance import file_sha256
from urban_intervention.utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    require_columns,
    sha256_file,
)


def test_sha256_implementation_is_independent_of_chunk_size(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    payload = bytes(range(256)) * 9000
    path.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()
    assert sha256_file(path, block_size=1024 * 1024) == expected
    assert sha256_file(path, block_size=8 * 1024 * 1024) == expected
    assert file_sha256(path) == expected


def test_require_columns_reports_only_missing_columns() -> None:
    frame = pd.DataFrame({"a": [1]})
    require_columns(frame, ["a"], "fixture")

    with pytest.raises(ValueError, match=r"fixture.*\['b', 'c'\]"):
        require_columns(frame, ["c", "a", "b"], "fixture")


def test_atomic_dataframe_and_json_writers_publish_complete_files(tmp_path) -> None:
    frame = pd.DataFrame({"name": ["地铁"], "value": [1.5]})
    csv_path = tmp_path / "nested" / "frame.csv"
    parquet_path = tmp_path / "nested" / "frame.parquet"
    json_path = tmp_path / "nested" / "manifest.json"

    atomic_write_csv(frame, csv_path)
    atomic_write_parquet(frame, parquet_path)
    atomic_write_json({"name": "地铁", "value": 1.5}, json_path)

    pd.testing.assert_frame_equal(pd.read_csv(csv_path), frame)
    pd.testing.assert_frame_equal(pd.read_parquet(parquet_path), frame)
    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "name": "地铁",
        "value": 1.5,
    }
    assert not list((tmp_path / "nested").glob("*.tmp"))
