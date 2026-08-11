"""Audit extracted Amap FileGDB assets for 2018-2024 POI processing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR / "scripts" / "collection"))

from urban_intervention.data.paths import CATALOG_DIR
from urban_intervention.pipelines.poi.gdb import GDB_ARCHIVE_DIR, build_extracted_gdb_inventory


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit extracted Amap FileGDB assets.")
    parser.add_argument("--base-dir", default=str(GDB_ARCHIVE_DIR))
    parser.add_argument(
        "--out",
        default=str(CATALOG_DIR / "inventories" / "poi_gdb_extracted_inventory.csv"),
    )
    parser.add_argument(
        "--inspect", action="store_true", help="Open each FileGDB and record layer/CRS/columns."
    )
    args = parser.parse_args()

    inventory = build_extracted_gdb_inventory(Path(args.base_dir), inspect=args.inspect)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {out_path} ({len(inventory)} rows)")
    if not inventory.empty:
        print(inventory.groupby("year").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
