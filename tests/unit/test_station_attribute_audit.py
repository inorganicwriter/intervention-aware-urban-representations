from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analysis/audit_station_attributes.py"
SPEC = importlib.util.spec_from_file_location("station_audit_test", SCRIPT)
station_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(station_audit)


def test_station_audit_checks_coverage_identity_flags_and_keeps_sources(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pd.DataFrame(
        {
            "treatment_order": [1, 2, 3, 4],
            "split": ["train", "test", "test", "test"],
            "final_training_mask": [True, True, True, False],
            "city_key": ["a"] * 4,
        }
    ).to_parquet(dataset / "sample_index.parquet", index=False)
    pd.DataFrame(
        {
            "treatment_order": [1, 2, 3, 4],
            "station_event_id": ["s1", "s2", "s3", "s4"],
            "opening_month": ["2020-01"] * 4,
        }
    ).to_parquet(dataset / "unit_features.parquet", index=False)
    attributes = tmp_path / "attributes.parquet"
    pd.DataFrame(
        {
            "treatment_order": [1, 2],
            "station_event_id": ["s1", "wrong"],
            "opening_date": ["2020-01-01", "2021-01-01"],
            "is_transfer_at_opening": [0, 2],
            "is_new_line_opening": [1, 0],
            "is_extension_opening": [0, 1],
            "is_terminal_at_opening": [1, None],
            "n_lines_at_opening": [1, -1],
            "terminal_evidence": ["station_degree", "unknown"],
        }
    ).to_parquet(attributes, index=False)
    before = attributes.read_bytes()
    result = station_audit.audit_station_attributes(attributes, dataset, tmp_path / "audit")
    assert result["eligible_units"] == 3
    assert result["covered_units"] == 2
    assert result["uncovered_orders"] == [3]
    assert result["flags"]["is_transfer_at_opening"]["invalid"] == 1
    assert result["flags"]["is_terminal_at_opening"]["missing_on_covered_units"] == 1
    assert result["numeric"]["n_lines_at_opening"]["invalid"] == 1
    assert result["identity_checks"]["station_event_id"]["mismatch"] == 1
    assert result["identity_checks"]["opening_month"]["mismatch"] == 1
    assert result["temporal_status"] == "requires_source_review"
    assert attributes.read_bytes() == before
    assert (tmp_path / "audit/station_attribute_review.csv").is_file()


def test_duplicate_attribute_keys_are_not_silently_matched(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pd.DataFrame(
        {
            "treatment_order": [1, 2],
            "split": ["train", "test"],
            "final_training_mask": [True, True],
        }
    ).to_parquet(dataset / "sample_index.parquet", index=False)
    attributes = tmp_path / "attributes.parquet"
    pd.DataFrame(
        {
            "treatment_order": [1, 1, 2, "invalid"],
            "is_transfer_at_opening": [0, 1, 1, 0],
        }
    ).astype({"treatment_order": str}).to_parquet(attributes, index=False)
    result = station_audit.audit_station_attributes(attributes, dataset, tmp_path / "audit")
    assert result["duplicate_attribute_orders"] == [1]
    assert result["covered_units"] == 1
    assert result["invalid_attribute_orders"] == 1
    assert result["uncovered_orders"] == [1]
    assert "is_terminal_at_opening" in result["missing_required_columns"]
