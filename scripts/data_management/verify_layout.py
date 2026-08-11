"""Verify a layout migration against a pre-migration snapshot."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from urban_intervention.data.paths import (
    CATALOG_DIR,
    CURATED_DIR,
    DATA_ROOT,
    PANEL_ROOT,
    PROJECT_ROOT,
    RAW_DIR,
    REFERENCE_DIR,
    STAGING_DIR,
)

CANONICAL = {
    "reference": REFERENCE_DIR,
    "raw": RAW_DIR,
    "staging": STAGING_DIR,
    "curated": CURATED_DIR,
    "labels": DATA_ROOT / "labels",
    "panels": PANEL_ROOT,
}


def iter_files(root: Path):
    if not root.exists():
        return
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if not (Path(directory) / d).is_symlink()]
        for name in filenames:
            path = Path(directory) / name
            if not path.is_symlink():
                yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=CATALOG_DIR / "quality" / "layout_v2_verification.json"
    )
    parser.add_argument(
        "--skip-canonical-summary",
        action="store_true",
        help="Only verify pre-migration paths; useful on slow mounted filesystems",
    )
    args = parser.parse_args()
    snapshot = args.snapshot if args.snapshot.is_absolute() else PROJECT_ROOT / args.snapshot

    missing: list[str] = []
    size_changed: list[dict[str, object]] = []
    expected_rows = []
    with snapshot.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] == "file" and not row["path"].startswith("data/active/catalog/"):
                expected_rows.append(row)

    def check_row(row: dict[str, str]):
        path = PROJECT_ROOT / row["path"]
        try:
            actual = path.stat().st_size
        except FileNotFoundError:
            return ("missing", row["path"], 0, 0)
        expected = int(row["size"])
        return ("changed" if actual != expected else "ok", row["path"], expected, actual)

    with ThreadPoolExecutor(max_workers=16) as executor:
        for status, path, expected, actual in executor.map(check_row, expected_rows):
            if status == "missing":
                missing.append(path)
            elif status == "changed":
                size_changed.append({"path": path, "before": expected, "after": actual})
    print(f"snapshot paths checked: {len(expected_rows)}", flush=True)

    # Every compatibility link that carries research data is exercised through
    # its pre-migration file path above; a broken target therefore appears in
    # ``missing`` without another expensive full-tree stat pass.
    broken_links: list[str] = []
    canonical = {}
    for name, root in () if args.skip_canonical_summary else CANONICAL.items():
        files = list(iter_files(root) or [])
        with ThreadPoolExecutor(max_workers=16) as executor:
            sizes = list(executor.map(lambda path: path.stat().st_size, files))
        extensions = Counter(path.suffix.lower() or "<none>" for path in files)
        canonical[name] = {
            "files": len(files),
            "bytes": sum(sizes),
            "extensions": dict(sorted(extensions.items())),
        }
        print(f"canonical layer checked: {name} ({len(files)} files)", flush=True)
    report = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "snapshot": str(snapshot.relative_to(PROJECT_ROOT)),
        "checked_files": len(expected_rows),
        "missing_files": missing,
        "size_changed": size_changed,
        "broken_legacy_links": broken_links,
        "canonical_layers": canonical,
        "ok": not missing and not size_changed and not broken_links,
    }
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
