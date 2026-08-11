"""Migrate the legacy data tree into the canonical research-data layout.

Moves are on the same filesystem and therefore do not rewrite large files.
For every moved legacy path, a relative symlink is created so existing scripts
continue to work.  The operation is idempotent and emits a CSV migration log.
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LOG_DIR = DATA / "catalog" / "migrations"


def link_back(old: Path, new: Path) -> None:
    old.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(new, start=old.parent)
    old.symlink_to(relative_target, target_is_directory=new.is_dir())


def directory_size(root: Path) -> int:
    """Return directory bytes without following browser-profile junctions."""
    total = 0
    seen: set[tuple[int, int]] = set()
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            identity = (directory.stat().st_dev, directory.stat().st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def move_path(
    old: Path, new: Path, dry_run: bool, rows: list[dict[str, object]], verbose: bool = False
) -> None:
    if verbose:
        print(f"check {old.relative_to(ROOT)} -> {new.relative_to(ROOT)}", flush=True)
    if old.is_symlink():
        rows.append(
            {
                "old": old.relative_to(ROOT),
                "new": new.relative_to(ROOT),
                "bytes": 0,
                "status": "already_linked",
            }
        )
        return
    if not old.exists():
        rows.append(
            {
                "old": old.relative_to(ROOT),
                "new": new.relative_to(ROOT),
                "bytes": 0,
                "status": "source_missing",
            }
        )
        return
    if new.exists():
        raise FileExistsError(f"Destination already exists: {new}")
    size = old.stat().st_size if old.is_file() else directory_size(old)
    rows.append(
        {
            "old": old.relative_to(ROOT),
            "new": new.relative_to(ROOT),
            "bytes": size,
            "status": "planned" if dry_run else "moved",
        }
    )
    if dry_run:
        return
    new.parent.mkdir(parents=True, exist_ok=True)
    old.rename(new)
    link_back(old, new)


def move_glob(
    pattern: str, destination: Path, dry_run: bool, rows: list[dict[str, object]]
) -> None:
    for old in sorted(ROOT.glob(pattern)):
        if old.is_symlink():
            continue
        move_path(old, destination / old.name, dry_run, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Perform moves; default is dry-run")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    dry_run = not args.execute
    rows: list[dict[str, object]] = []
    # Keep verbose local to the migration process without complicating the
    # declarative call sites below.
    if args.verbose:
        original_move_path = globals()["move_path"]

        def verbose_move(old: Path, new: Path, dry: bool, log: list[dict[str, object]]) -> None:
            original_move_path(old, new, dry, log, verbose=True)

        globals()["move_path"] = verbose_move

    # Reference and source inventories.
    move_path(
        DATA / "external" / "admin_boundaries", DATA / "reference" / "boundaries", dry_run, rows
    )
    move_path(DATA / "北京市AOI", DATA / "raw" / "housing" / "aoi" / "baidu_beijing", dry_run, rows)
    move_path(ROOT / "cache", DATA / "raw" / "road_network" / "osm_overpass_cache", dry_run, rows)

    # NBS source and parsed staging products.
    move_path(
        DATA / "external" / "nbs_70city_html_cache",
        DATA / "raw" / "nbs_hpi" / "html_cache",
        dry_run,
        rows,
    )
    move_path(
        DATA / "external" / "nbs_70city_article_index.csv",
        DATA / "staging" / "nbs_hpi" / "article_index.csv",
        dry_run,
        rows,
    )
    move_path(
        DATA / "external" / "nbs_70city_hpi_monthly.csv",
        DATA / "staging" / "nbs_hpi" / "monthly.csv",
        dry_run,
        rows,
    )
    move_glob("data/external/poi*inventory.csv", DATA / "catalog" / "inventories", dry_run, rows)
    move_path(
        DATA / "external" / "lianjia_sample.html",
        DATA / "staging" / "housing" / "diagnostics" / "lianjia_sample.html",
        dry_run,
        rows,
    )

    # Browser state is runtime state, not research data.
    for name in (
        "playwright_profile",
        "playwright_profile_old_20260707_015739",
        "playwright_edge_old_20260707_015739",
    ):
        move_path(
            DATA / "external" / name, ROOT / ".runtime" / "browser_profiles" / name, dry_run, rows
        )

    # Canonical raw housing sources.
    housing_moves = {
        "anjuke_cross": DATA / "raw" / "housing" / "platform_exports" / "anjuke" / "cross_section",
        "grid_2023may": DATA / "raw" / "housing" / "spatial_support" / "grid_price_2023_05",
        "lianjia_xianyu": DATA
        / "raw"
        / "housing"
        / "platform_exports"
        / "lianjia"
        / "purchased_transactions",
        "wayback_csv": DATA / "raw" / "housing" / "web_archives" / "wayback" / "parsed_pages",
        "wayback_inventory": DATA / "raw" / "housing" / "web_archives" / "wayback" / "inventories",
        "wayback_manifests": DATA / "raw" / "housing" / "web_archives" / "wayback" / "manifests",
    }
    for name, destination in housing_moves.items():
        move_path(DATA / "raw_housing" / name, destination, dry_run, rows)
    move_path(
        DATA / "raw" / "housing" / "_anjuke_raw_html",
        DATA / "raw" / "housing" / "web_archives" / "wayback" / "raw_html" / "anjuke",
        dry_run,
        rows,
    )
    move_path(
        DATA / "raw" / "housing" / "_wayback_raw_html",
        DATA / "raw" / "housing" / "web_archives" / "wayback" / "raw_html" / "legacy_wayback",
        dry_run,
        rows,
    )

    # Transit acquisition artifacts are source-first in the canonical layout.
    transit_root = DATA / "raw" / "transit"
    for city_dir in sorted(transit_root.glob("*")):
        if not city_dir.is_dir() or city_dir.is_symlink():
            continue
        for old in sorted(city_dir.glob("*.csv")):
            if old.is_symlink():
                continue
            name = old.name
            if "_amap" in name:
                source = "amap"
            elif "_osm" in name:
                source = "osm"
            elif "_wikidata" in name:
                source = "wikidata"
            elif "_wiki" in name:
                source = "wikipedia"
            elif "_merged" in name:
                source = "merged"
            else:
                source = "unclassified"
            move_path(old, transit_root / source / city_dir.name / name, dry_run, rows)

    # Split grids from treatment outputs while preserving every old filename.
    for city_dir in sorted((DATA / "grids").glob("*")):
        if not city_dir.is_dir() or city_dir.is_symlink():
            continue
        for old in sorted(city_dir.iterdir()):
            if old.is_symlink() or not old.is_file():
                continue
            if "treatment" in old.name:
                destination = DATA / "curated" / "treatment" / city_dir.name / old.name
            else:
                destination = DATA / "reference" / "grids" / city_dir.name / old.name
            move_path(old, destination, dry_run, rows)

    # Curated covariates and final panels.
    move_glob("data/processed/*_viirs.parquet", DATA / "curated" / "viirs", dry_run, rows)
    move_glob("data/processed/*_s2.parquet", DATA / "curated" / "sentinel2", dry_run, rows)
    move_glob("data/processed/*_pop.parquet", DATA / "curated" / "population", dry_run, rows)
    move_path(DATA / "processed" / "poi", DATA / "curated" / "poi", dry_run, rows)
    move_path(DATA / "processed" / "road_network", DATA / "curated" / "road_network", dry_run, rows)
    move_path(
        DATA / "processed" / "panel", DATA / "panels" / "grid_year_housing" / "v1", dry_run, rows
    )

    # Research labels: preserve source and observation semantics in the path.
    move_path(
        DATA / "labels" / "all_cities_hpi_yearly.parquet",
        DATA / "labels_canonical" / "housing" / "city_hpi" / "all_cities_hpi_yearly.parquet",
        dry_run,
        rows,
    )
    move_path(
        DATA / "labels" / "hpi_city_yearly.parquet",
        DATA / "labels_canonical" / "housing" / "city_hpi" / "hpi_city_yearly.parquet",
        dry_run,
        rows,
    )
    move_path(
        DATA / "labels" / "geocode_cache.json",
        DATA / "staging" / "housing" / "geocoding" / "geocode_cache.json",
        dry_run,
        rows,
    )
    for city_dir in sorted((DATA / "labels").glob("*")):
        if not city_dir.is_dir() or city_dir.is_symlink():
            continue
        for old in sorted(city_dir.iterdir()):
            if old.is_symlink() or not old.is_file():
                continue
            name = old.name
            if "_hpi_yearly" in name:
                base = DATA / "labels_canonical" / "housing" / "city_hpi"
            elif "_anjuke_grid_price" in name:
                base = DATA / "labels_canonical" / "housing" / "listing_price" / "anjuke"
            elif "_grid2023_yearly" in name:
                base = DATA / "labels_canonical" / "housing" / "listing_price" / "grid_2023"
            elif "_wayback_grid_yearly" in name:
                base = DATA / "labels_canonical" / "housing" / "historical_snapshot" / "wayback"
            elif "_lianjia_" in name:
                base = DATA / "labels_canonical" / "housing" / "transaction_price" / "lianjia"
            else:
                base = DATA / "labels_canonical" / "housing" / "unclassified"
            move_path(old, base / city_dir.name / name, dry_run, rows)

    # Runtime logs at project root.
    for name in ("wayback_scrape.log", "wayback_stderr.log", "wayback_stdout.log"):
        source = ROOT / name
        move_path(source, ROOT / ".runtime" / "logs" / name, dry_run, rows)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if dry_run:
        log_path = ROOT / ".runtime" / f"migration_dry_run_{timestamp}.csv"
    else:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"layout_v2_{timestamp}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("old", "new", "bytes", "status"))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"{'DRY RUN' if dry_run else 'MIGRATED'}: {len(rows)} paths; log={log_path.relative_to(ROOT)}"
    )
    for status, count in sorted(
        {s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}}.items()
    ):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
