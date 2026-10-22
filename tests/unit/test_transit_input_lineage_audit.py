from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analysis/audit_transit_input_lineage.py"
SPEC = importlib.util.spec_from_file_location("transit_lineage_audit_test", SCRIPT)
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def test_missing_evidence_is_unknown_and_balance_proves_transit_use(tmp_path: Path) -> None:
    root = tmp_path / "production"
    task_root = root / "outputs/control_design/tasks"
    for index in (1, 2):
        folder = task_root / f"{index:05d}"
        folder.mkdir(parents=True)
        pd.DataFrame({"treatment_order": [index], "status": ["matched"]}).to_csv(
            folder / "control_record.csv", index=False
        )
    pd.DataFrame({"feature": ["transit_network_closeness", "viirs__lag2"]}).to_parquet(
        task_root / "00002/feature_balance.parquet", index=False
    )
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    report = audit_module.audit(root, [Path("outputs/control_design/tasks")], max_records=0)
    records = report["control_design"][0]["records"]
    assert records[0]["evidence_status"] == "not_documented_not_proof_of_absence"
    assert records[1]["documented_transit_features"] == ["transit_network_closeness"]
    assert report["control_design"][0]["transit_documented_records"] == 1
    assert report["reference_inputs"]["wikidata_adjacency.parquet"]["exists"] is False
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_explicit_transit_metadata_and_sample_limit(tmp_path: Path) -> None:
    folder = tmp_path / "tasks/00001"
    folder.mkdir(parents=True)
    pd.DataFrame({"active_static_features": ["loc_dist_main_km+transit_stations_500m"]}).to_csv(
        folder / "control_record.csv", index=False
    )
    ref = tmp_path / "data/active/reference/transit"
    ref.mkdir(parents=True)
    pd.DataFrame({"station_name": ["a"], "adj_station_name": ["b"]}).to_parquet(
        ref / "wikidata_adjacency.parquet", index=False
    )
    pd.DataFrame(
        {"opening_date": ["invalid"], "wgs84_lon": [181.0], "wgs84_lat": [40.0]}
    ).to_parquet(ref / "canonical_station_events_resolved.parquet", index=False)
    report = audit_module.audit(tmp_path, [Path("tasks")], max_records=1)
    assert (
        report["reference_inputs"]["wikidata_adjacency.parquet"]["history_status"]
        == "missing_edge_dates"
    )
    assert report["control_design"][0]["records"][0]["evidence_status"] == "transit_documented"
    assert report["reference_inputs"]["canonical_station_events_resolved.parquet"][
        "invalid_values"
    ] == {
        "opening_date": 1,
        "wgs84_lon": 1,
        "wgs84_lat": 0,
    }
