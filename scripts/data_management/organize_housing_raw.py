"""Organize raw housing assets without changing source-owned filenames.

The migration changes only project-owned directory names. Files supplied by a
platform, repository, archive, or vendor retain their original basename so
that citations, checksums, and provenance remain auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from urban_intervention.data.paths import (
    OUTPUT_HOUSING_ACQUISITION_DIR,
    PROJECT_ROOT,
    RAW_HOUSING_DIR,
)

RAW_ROOT = RAW_HOUSING_DIR
REPORT_DIR = OUTPUT_HOUSING_ACQUISITION_DIR

MOVES = (
    ("anjuke_cross_section/housing", "platform_exports/anjuke/cross_section"),
    ("lianjia_purchased", "platform_exports/lianjia/purchased_transactions"),
    ("wayback", "web_archives/wayback"),
    ("open_research", "open_data/datasets"),
    ("open_research_imports", "open_data/import_batches"),
    ("aoi", "spatial_support/community_aoi"),
    ("grid_2023", "spatial_support/grid_price_2023_05"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(relative_path: Path) -> tuple[str, str]:
    parts = relative_path.parts
    if parts[:1] == ("platform_exports",):
        return "platform_export", parts[1] if len(parts) > 1 else "unknown"
    if parts[:1] == ("web_archives",):
        return "web_archive", parts[1] if len(parts) > 1 else "unknown"
    if parts[:2] == ("open_data", "datasets"):
        return "open_dataset", parts[2] if len(parts) > 2 else "unknown"
    if parts[:2] == ("open_data", "import_batches"):
        return "import_batch", parts[2] if len(parts) > 2 else "unknown"
    if parts[:1] == ("spatial_support",):
        return "spatial_support", parts[1] if len(parts) > 1 else "unknown"
    return "root_metadata", "project"


def migrate(apply: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for old_relative, new_relative in MOVES:
        old = RAW_ROOT / old_relative
        new = RAW_ROOT / new_relative
        if old.exists() and new.exists():
            raise FileExistsError(f"Both migration endpoints exist: {old} and {new}")
        if not old.exists() and not new.exists():
            raise FileNotFoundError(f"Neither migration endpoint exists: {old} or {new}")
        source = old if old.exists() else new
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.relative_to(source)
            rows.append(
                {
                    "old_path": str(Path("data/archive/raw/housing") / old_relative / suffix),
                    "new_path": str(Path("data/archive/raw/housing") / new_relative / suffix),
                    "size_bytes": path.stat().st_size,
                    "migration_status": "planned" if old.exists() else "already_migrated",
                }
            )
        if apply and old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
    return rows


def inventory(with_hashes: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(RAW_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(RAW_ROOT)
        category, source = classify(relative)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "category": category,
                "source_or_dataset": source,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path) if with_hashes else "",
                "filename_policy": (
                    "project_snake_case"
                    if category == "root_metadata"
                    else "source_original_preserved"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply directory moves")
    parser.add_argument("--skip-hashes", action="store_true", help="Skip SHA-256 inventory")
    args = parser.parse_args()

    RAW_ROOT.resolve().relative_to(PROJECT_ROOT.resolve())
    migration_rows = migrate(args.apply)
    if not args.apply:
        print(
            json.dumps(
                {"planned_files": len(migration_rows), "moves": MOVES}, ensure_ascii=False, indent=2
            )
        )
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(migration_rows).to_csv(
        REPORT_DIR / "housing_raw_path_migration.csv", index=False, encoding="utf-8-sig"
    )
    files = inventory(with_hashes=not args.skip_hashes)
    files.to_csv(REPORT_DIR / "housing_raw_inventory.csv", index=False, encoding="utf-8-sig")
    summary = {
        "schema": "housing_raw_organization_manifest",
        "raw_root": "data/archive/raw/housing",
        "directories_moved": len(MOVES),
        "files": int(len(files)),
        "bytes": int(files["size_bytes"].sum()),
        "hashes_computed": not args.skip_hashes,
        "categories": files.groupby("category").size().astype(int).to_dict(),
        "source_filenames_modified": False,
    }
    (REPORT_DIR / "housing_raw_organization.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
