"""Audit raw housing acquisition assets without fetching external websites.

The audit treats a Wayback target as complete only when its CDX inventory is
present and every capture has a terminal manifest outcome.  Parsed rows must
also trace back to an exact ``timestamp + original URL`` inventory key.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from collection.wayback_research_scraper import configured_targets  # noqa: E402

from urban_intervention.config.project import ACTIVE_CITIES  # noqa: E402
from urban_intervention.data.paths import (  # noqa: E402
    OUTPUT_HOUSING_ACQUISITION_DIR,
    RAW_HOUSING_DIR,
)

OUTPUT_DIR = OUTPUT_HOUSING_ACQUISITION_DIR
TERMINAL_STATUSES = {"ok", "no_rows", "not_list_page", "http_404", "http_410"}


@dataclass(frozen=True)
class TargetAudit:
    source: str
    city_key: str
    page_type: str
    inventory_file: str
    manifest_file: str
    parsed_file: str
    inventory_exists: bool
    manifest_exists: bool
    parsed_exists: bool
    inventory_captures: int
    unique_inventory_captures: int
    manifest_lines: int
    invalid_manifest_lines: int
    captures_with_any_outcome: int
    terminal_captures: int
    retryable_captures: int
    missing_outcome_captures: int
    parsed_rows: int
    parsed_snapshot_keys: int
    orphan_parsed_rows: int
    target_complete: bool


def _capture_key(timestamp: object, original: object) -> str:
    return f"{str(timestamp).strip()}\t{str(original).strip()}"


def _read_inventory(path: Path) -> tuple[list[dict], int]:
    if not path.exists():
        return [], 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    captures = payload.get("captures", [])
    if not isinstance(captures, list):
        raise ValueError(f"{path}: captures is not a list")
    keys = {
        _capture_key(row.get("timestamp", ""), row.get("original", ""))
        for row in captures
        if isinstance(row, dict)
    }
    return captures, len(keys)


def _read_manifest(path: Path) -> tuple[int, int, dict[str, dict]]:
    if not path.exists():
        return 0, 0, {}
    lines = invalid = 0
    latest: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            lines += 1
            try:
                row = json.loads(raw)
                key = str(row["capture_key"])
            except (ValueError, KeyError, TypeError):
                invalid += 1
                continue
            latest[key] = row
    return lines, invalid, latest


def _parsed_traceability(path: Path, inventory_keys: set[str]) -> tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0
    rows = snapshots = orphan_rows = 0
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"snapshot_date", "original_url"}
        if not required.issubset(reader.fieldnames or []):
            # Legacy files without exact-capture provenance are retained but
            # deliberately fail the traceability contract.
            for _ in reader:
                rows += 1
            return rows, 0, rows
        for row in reader:
            rows += 1
            key = _capture_key(row.get("snapshot_date", ""), row.get("original_url", ""))
            seen.add(key)
            if key not in inventory_keys:
                orphan_rows += 1
    snapshots = len(seen)
    return rows, snapshots, orphan_rows


def audit_target(target: object) -> tuple[TargetAudit, list[dict]]:
    inventory_path = Path(target.inventory_path)
    manifest_path = Path(target.manifest_path)
    parsed_path = Path(target.output_path)
    captures, unique_inventory = _read_inventory(inventory_path)
    inventory_keys = {
        _capture_key(row.get("timestamp", ""), row.get("original", ""))
        for row in captures
        if isinstance(row, dict)
    }
    manifest_lines, invalid_lines, latest = _read_manifest(manifest_path)
    terminal = {key for key, row in latest.items() if row.get("status") in TERMINAL_STATUSES}
    retryable = set(latest) - terminal
    missing = inventory_keys - set(latest)
    parsed_rows, parsed_snapshots, orphan_rows = _parsed_traceability(parsed_path, inventory_keys)
    complete = (
        inventory_path.exists()
        and not missing
        and not retryable
        and invalid_lines == 0
        and orphan_rows == 0
    )
    audit = TargetAudit(
        source=str(target.source),
        city_key=str(target.city),
        page_type=str(target.page_type),
        inventory_file=inventory_path.name,
        manifest_file=manifest_path.name,
        parsed_file=parsed_path.name,
        inventory_exists=inventory_path.exists(),
        manifest_exists=manifest_path.exists(),
        parsed_exists=parsed_path.exists(),
        inventory_captures=len(captures),
        unique_inventory_captures=unique_inventory,
        manifest_lines=manifest_lines,
        invalid_manifest_lines=invalid_lines,
        captures_with_any_outcome=len(set(latest) & inventory_keys),
        terminal_captures=len(terminal & inventory_keys),
        retryable_captures=len(retryable & inventory_keys),
        missing_outcome_captures=len(missing),
        parsed_rows=parsed_rows,
        parsed_snapshot_keys=parsed_snapshots,
        orphan_parsed_rows=orphan_rows,
        target_complete=complete,
    )
    unfinished: list[dict] = []
    capture_lookup = {
        _capture_key(row.get("timestamp", ""), row.get("original", "")): row
        for row in captures
        if isinstance(row, dict)
    }
    for key in sorted(missing | retryable):
        capture = capture_lookup.get(key, {})
        outcome = latest.get(key, {})
        unfinished.append(
            {
                "source": target.source,
                "city_key": target.city,
                "page_type": target.page_type,
                "capture_key": key,
                "timestamp": capture.get("timestamp", ""),
                "original": capture.get("original", ""),
                "latest_status": outcome.get("status", "missing_outcome"),
                "latest_error": outcome.get("error", ""),
            }
        )
    return audit, unfinished


def _raw_file_inventory(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "top_level_source": relative.parts[0] if relative.parts else "",
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def _write_report(audits: pd.DataFrame, unfinished: pd.DataFrame, summary: dict) -> None:
    source = audits.groupby("source", as_index=False).agg(
        targets=("source", "size"),
        complete_targets=("target_complete", "sum"),
        captures=("inventory_captures", "sum"),
        terminal_captures=("terminal_captures", "sum"),
        retryable_captures=("retryable_captures", "sum"),
        missing_outcomes=("missing_outcome_captures", "sum"),
        parsed_rows=("parsed_rows", "sum"),
        orphan_rows=("orphan_parsed_rows", "sum"),
    )
    lines = [
        "# Housing acquisition audit",
        "",
        "This report is generated from local raw assets only; it does not fetch live websites.",
        "",
        "## Wayback acceptance",
        "",
        f"- Expected targets: {summary['expected_targets']:,}",
        f"- Complete targets: {summary['complete_targets']:,}",
        f"- Inventory captures: {summary['inventory_captures']:,}",
        f"- Terminal captures: {summary['terminal_captures']:,}",
        f"- Retryable or missing outcomes: {summary['unfinished_captures']:,}",
        f"- Parsed rows: {summary['parsed_rows']:,}",
        f"- Parsed rows without exact-capture provenance: {summary['orphan_parsed_rows']:,}",
        "",
        "| Source | Targets | Complete | Captures | Terminal | Retryable | Missing outcome | Parsed rows | Orphan rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in source.itertuples(index=False):
        lines.append(
            f"| {row.source} | {row.targets:,} | {row.complete_targets:,} | "
            f"{row.captures:,} | {row.terminal_captures:,} | "
            f"{row.retryable_captures:,} | {row.missing_outcomes:,} | "
            f"{row.parsed_rows:,} | {row.orphan_rows:,} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "A target is complete only when its inventory exists, every capture has a terminal "
            "manifest outcome, the manifest is valid JSONL, and every parsed row traces to an "
            "inventory capture. See `wayback_unfinished_captures.csv` for resumable work.",
        ]
    )
    (OUTPUT_DIR / "housing_acquisition_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run() -> dict:
    targets = configured_targets({"lianjia", "beike", "anjuke"}, set(ACTIVE_CITIES))
    audit_rows: list[dict] = []
    unfinished_rows: list[dict] = []
    for target in targets:
        audit, unfinished = audit_target(target)
        audit_rows.append(asdict(audit))
        unfinished_rows.extend(unfinished)

    audits = pd.DataFrame(audit_rows).sort_values(
        ["source", "city_key", "page_type"], kind="stable"
    )
    unfinished = pd.DataFrame(
        unfinished_rows,
        columns=[
            "source",
            "city_key",
            "page_type",
            "capture_key",
            "timestamp",
            "original",
            "latest_status",
            "latest_error",
        ],
    )
    raw_files = _raw_file_inventory(RAW_HOUSING_DIR)
    status_counts = Counter(unfinished["latest_status"]) if not unfinished.empty else Counter()
    summary = {
        "schema": "housing_acquisition_audit_v1",
        "expected_targets": int(len(audits)),
        "complete_targets": int(audits["target_complete"].sum()),
        "missing_inventory_targets": int((~audits["inventory_exists"]).sum()),
        "inventory_captures": int(audits["inventory_captures"].sum()),
        "terminal_captures": int(audits["terminal_captures"].sum()),
        "unfinished_captures": int(len(unfinished)),
        "unfinished_statuses": dict(sorted(status_counts.items())),
        "parsed_rows": int(audits["parsed_rows"].sum()),
        "orphan_parsed_rows": int(audits["orphan_parsed_rows"].sum()),
        "raw_housing_files": int(len(raw_files)),
        "raw_housing_bytes": int(raw_files["size_bytes"].sum()) if not raw_files.empty else 0,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audits.to_csv(OUTPUT_DIR / "wayback_target_audit.csv", index=False, encoding="utf-8-sig")
    unfinished.to_csv(
        OUTPUT_DIR / "wayback_unfinished_captures.csv", index=False, encoding="utf-8-sig"
    )
    raw_files.to_csv(
        OUTPUT_DIR / "housing_raw_file_inventory.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "housing_acquisition_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(audits, unfinished, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
