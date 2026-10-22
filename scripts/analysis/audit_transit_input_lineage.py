"""Read-only inventory of transit inputs and saved control-design evidence.

Run against the actual task roots used by the server release. This inventory
does not establish that an archived task was used by a particular label release.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

TRANSIT_PATTERN = re.compile(
    r"transit_|network_closeness|dist_nearest_station|stations_(500|800|1500)m|lines_in_1500m"
)


def audit(root: Path, control_roots: list[Path], max_records: int = 50) -> dict[str, object]:
    if max_records < 0:
        raise ValueError("max_records must be nonnegative; 0 means all records")
    root = root.resolve()
    report: dict[str, object] = {
        "root": str(root),
        "scope": "read_only_inventory_not_release_lineage_certification",
        "reference_inputs": {},
        "model_inputs": [],
        "control_design": [],
    }
    reference = root / "data/active/reference/transit"
    for filename in ("canonical_station_events_resolved.parquet", "wikidata_adjacency.parquet"):
        path = reference / filename
        entry: dict[str, object] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            table = pq.ParquetFile(path)
            columns = table.schema_arrow.names
            entry.update({"rows": table.metadata.num_rows, "columns": columns})
            if filename == "wikidata_adjacency.parquet":
                entry["edge_date_columns"] = [c for c in ("valid_from", "valid_to") if c in columns]
                entry["history_status"] = (
                    "missing_edge_dates"
                    if "valid_from" not in columns
                    else "dates_require_validation"
                )
                if "valid_from" in columns:
                    values = pd.read_parquet(path, columns=["valid_from"])["valid_from"]
                    parsed = pd.to_datetime(values, errors="coerce")
                    entry["valid_from_missing"] = int(values.isna().sum())
                    entry["valid_from_invalid"] = int((values.notna() & parsed.isna()).sum())
            else:
                wanted = [
                    c for c in ("opening_date", "wgs84_lon", "wgs84_lat", "lines") if c in columns
                ]
                frame = pd.read_parquet(path, columns=wanted)
                entry["missing_values"] = {c: int(frame[c].isna().sum()) for c in wanted}
                invalid = {}
                if "opening_date" in frame:
                    dates = pd.to_datetime(frame["opening_date"], errors="coerce")
                    invalid["opening_date"] = int(
                        (frame["opening_date"].notna() & dates.isna()).sum()
                    )
                for column, bound in (("wgs84_lon", 180), ("wgs84_lat", 90)):
                    if column in frame:
                        values = pd.to_numeric(frame[column], errors="coerce")
                        invalid[column] = int(
                            (frame[column].notna() & (values.isna() | (values.abs() > bound))).sum()
                        )
                entry["invalid_values"] = invalid
                entry["line_membership_date_columns"] = [
                    c
                    for c in columns
                    if "line" in c and any(term in c for term in ("date", "valid_from", "valid_to"))
                ]
        report["reference_inputs"][filename] = entry

    snapshots = root / "data/active/causal/transit_snapshots"
    paths = sorted(snapshots.glob("*/*.parquet")) if snapshots.exists() else []
    report["production_snapshots"] = {
        "directory": str(snapshots),
        "parquet_files": len(paths),
        "with_metadata_sidecar": sum(p.with_suffix(".metadata.json").is_file() for p in paths),
        "note": "File presence does not establish code version or server release use.",
    }
    inputs_root = root / "data/active/model_inputs"
    for path in sorted(inputs_root.glob("*/unit_features.parquet")):
        try:
            columns = pq.ParquetFile(path).schema_arrow.names
            report["model_inputs"].append(
                {
                    "dataset": path.parent.name,
                    "transit_feature_columns": [c for c in columns if TRANSIT_PATTERN.search(c)],
                }
            )
        except Exception as error:
            report["model_inputs"].append({"dataset": path.parent.name, "error": str(error)})

    for supplied in control_roots:
        task_root = supplied if supplied.is_absolute() else root / supplied
        records = sorted(task_root.rglob("control_record.csv")) if task_root.is_dir() else []
        sampled = records[:max_records] if max_records else records
        entries = []
        for path in sampled:
            entry: dict[str, object] = {
                "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            }
            try:
                record = pd.read_csv(path)
                if record.empty:
                    raise ValueError("empty control record")
                row = record.iloc[0]
                entry.update(
                    {
                        k: str(row[k])
                        for k in (
                            "treatment_order",
                            "city_key",
                            "opening_month",
                            "status",
                            "selected_method",
                            "donor_scope",
                            "control_unit_key",
                            "code_fingerprint",
                        )
                        if k in record and pd.notna(row[k])
                    }
                )
                evidence = []
                if "active_static_features" in record and pd.notna(row["active_static_features"]):
                    evidence.extend(str(row["active_static_features"]).split("+"))
                balance = path.parent / "feature_balance.parquet"
                if balance.is_file():
                    columns = pq.ParquetFile(balance).schema_arrow.names
                    if "feature" in columns:
                        evidence.extend(
                            pd.read_parquet(balance, columns=["feature"])["feature"]
                            .dropna()
                            .astype(str)
                        )
                entry["documented_transit_features"] = sorted(
                    {c for c in evidence if TRANSIT_PATTERN.search(c)}
                )
                entry["evidence_status"] = (
                    "transit_documented"
                    if entry["documented_transit_features"]
                    else "not_documented_not_proof_of_absence"
                )
            except Exception as error:
                entry["error"] = str(error)
                entry["evidence_status"] = "unknown"
            entries.append(entry)
        report["control_design"].append(
            {
                "task_root": str(task_root),
                "exists": task_root.is_dir(),
                "total_records": len(records),
                "examined_records": len(sampled),
                "sample_complete": len(sampled) == len(records),
                "transit_documented_records": sum(
                    e.get("evidence_status") == "transit_documented" for e in entries
                ),
                "records": entries,
            }
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--control-root",
        type=Path,
        action="append",
        required=True,
        help="Actual production task directory; repeat for cross-city/other scopes",
    )
    parser.add_argument("--max-records", type=int, default=50, help="0 scans all records")
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.root, args.control_root, args.max_records), ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
