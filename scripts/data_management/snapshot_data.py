"""Create a deterministic, machine-readable snapshot of project data.

The snapshot is intentionally independent of pandas/pyarrow so it can be run
before an environment is configured.  By default it hashes small files and
research metadata while recording size and nanosecond mtime for every file.
Use ``--hash-mode all`` for a full byte-level checksum audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from urban_intervention.data.paths import CATALOG_DIR, PROJECT_ROOT

DEFAULT_OUTPUT = CATALOG_DIR / "snapshots"
SCAN_ROOTS = ("data", "cache", "outputs")
ALWAYS_HASH = {".csv", ".geojson", ".gpkg", ".jsonl", ".md", ".txt", ".yaml", ".yml"}
RUNTIME_PARTS = {
    "playwright_profile",
    "playwright_profile_old_20260707_015739",
    "playwright_edge_old_20260707_015739",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_hash(path: Path, size: int, mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "none":
        return False
    if any(part in RUNTIME_PARTS for part in path.parts):
        return False
    try:
        if path.relative_to(PROJECT_ROOT).parts[0] == "cache":
            return False
    except ValueError:
        pass
    if "catalog" in path.parts:
        return True
    return size <= 16 * 1024 * 1024


def classify(relative: Path) -> str:
    parts = relative.parts
    if not parts:
        return "unknown"
    if parts[0] == "cache":
        return "raw_cache"
    if parts[0] == "outputs":
        return "report"
    if len(parts) < 2:
        return "data_root"
    return parts[1]


def collect(hash_mode: str) -> list[dict[str, object]]:
    all_paths: list[Path] = []
    for root_name in SCAN_ROOTS:
        scan_root = PROJECT_ROOT / root_name
        if not scan_root.exists():
            continue
        if scan_root.is_symlink():
            all_paths.append(scan_root)
            continue
        paths: list[Path] = []
        for directory, dirnames, filenames in os.walk(scan_root, followlinks=False):
            # Browser profiles are volatile runtime state with junction-like
            # directories on Windows mounts.  They are migrated as directory
            # units but deliberately excluded from the research-data manifest.
            dirnames[:] = [name for name in dirnames if name not in RUNTIME_PARTS]
            if Path(directory) == CATALOG_DIR:
                dirnames[:] = [name for name in dirnames if name != "snapshots"]
            dirnames.sort()
            filenames.sort()
            base = Path(directory)
            paths.extend(base / name for name in dirnames if (base / name).is_symlink())
            paths.extend(base / name for name in filenames)
        all_paths.extend(paths)

    def make_record(path: Path) -> dict[str, object] | None:
        relative = path.relative_to(PROJECT_ROOT)
        file_stat = path.lstat()
        if stat.S_ISLNK(file_stat.st_mode):
            return {
                "path": relative.as_posix(),
                "kind": "symlink",
                "category": classify(relative),
                "size": 0,
                "mtime_ns": file_stat.st_mtime_ns,
                "sha256": "",
                "link_target": os.readlink(path),
            }
        if stat.S_ISREG(file_stat.st_mode):
            return {
                "path": relative.as_posix(),
                "kind": "file",
                "category": classify(relative),
                "size": file_stat.st_size,
                "mtime_ns": file_stat.st_mtime_ns,
                "sha256": sha256(path) if should_hash(path, file_stat.st_size, hash_mode) else "",
                "link_target": "",
            }
        return None

    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for record in executor.map(make_record, sorted(all_paths)):
            if record is not None:
                records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Snapshot name; defaults to UTC timestamp")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hash-mode", choices=("quick", "all", "none"), default="quick")
    args = parser.parse_args()

    name = args.name or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    records = collect(args.hash_mode)

    manifest_path = output / f"{name}_files.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "kind", "category", "size", "mtime_ns", "sha256", "link_target"),
        )
        writer.writeheader()
        writer.writerows(records)

    bytes_by_category: dict[str, int] = defaultdict(int)
    files_by_category: Counter[str] = Counter()
    for row in records:
        if row["kind"] == "file":
            category = str(row["category"])
            files_by_category[category] += 1
            bytes_by_category[category] += int(row["size"])
    summary = {
        "snapshot": name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "root": str(PROJECT_ROOT),
        "hash_mode": args.hash_mode,
        "files": sum(row["kind"] == "file" for row in records),
        "symlinks": sum(row["kind"] == "symlink" for row in records),
        "bytes": sum(int(row["size"]) for row in records),
        "hashed_files": sum(bool(row["sha256"]) for row in records),
        "files_by_category": dict(sorted(files_by_category.items())),
        "bytes_by_category": dict(sorted(bytes_by_category.items())),
        "manifest": manifest_path.name,
    }
    (output / f"{name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
