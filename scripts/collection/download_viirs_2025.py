"""Download the 2025 VIIRS Drive exports to the local staging layout.

The 528 monthly exports (44 cities x 2025-01..2025-12) were queued to the
Drive folder ``MIT_Summer_VIIRS`` by ``extend_viirs_monthly_2025.py``.  This
script pulls them into ``data/active/staging/gee/viirs/{city}/raw/`` — the
same layout the processing stage expects — and reports any missing months.

Usage:
    python scripts/collection/download_viirs_2025.py [--dry-run] [--workers 4]
"""

from __future__ import annotations

import argparse
import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.config.project import ACTIVE_CITIES  # noqa: E402
from urban_intervention.data.paths import STAGING_DIR  # noqa: E402

FOLDER_NAME = "MIT_Summer_VIIRS"
PROJECT = "macro-city-engine"
OUT_ROOT = STAGING_DIR / "gee" / "viirs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    import ee
    from googleapiclient.discovery import build

    ee.Initialize(project=PROJECT)
    service = build(
        "drive", "v3",
        credentials=ee.data.get_persistent_credentials(),
        cache_discovery=False,
    )

    def execute(request):
        for attempt in range(20):
            try:
                return request.execute()
            except Exception as exc:  # noqa: BLE001
                if attempt == 19:
                    raise
                import time

                time.sleep(min(30, attempt * 3))

    folders = execute(
        service.files().list(
            q=f"name='{FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' "
            "and trashed=false",
            fields="files(id, name)",
        )
    ).get("files", [])
    if not folders:
        raise RuntimeError(f"Drive folder {FOLDER_NAME} not found")
    folder_id = folders[0]["id"]

    page = execute(
        service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id, name, size)",
            pageSize=1000,
        )
    )
    files = page.get("files", [])
    viirs_files = [f for f in files if f["name"].startswith("viirs_") and "_2025_" in f["name"]]
    print(f"Drive 中 2025 VIIRS 文件: {len(viirs_files)}")

    if args.dry_run:
        for f in sorted(viirs_files, key=lambda x: x["name"])[:8]:
            print(" ", f["name"], f.get("size", "?"))
        return 0

    def download_one(f):
        name = f["name"]  # viirs_{city}_{YYYY_MM}.csv
        city = name.split("_")[1]
        out_dir = OUT_ROOT / city / "raw"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / name
        if target.exists() and target.stat().st_size > 0:
            return name, "already"
        data = execute(service.files().get_media(fileId=f["id"]))
        target.write_bytes(data.execute())
        return name, "downloaded"

    ok, already, failed = 0, 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, f) for f in viirs_files]
        for future in as_completed(futures):
            try:
                name, status = future.result()
                if status == "downloaded":
                    ok += 1
                else:
                    already += 1
            except Exception as exc:  # noqa: BLE001
                failed.append(str(exc)[:120])
    print(f"下载完成: {ok} 新下载, {already} 已存在, {len(failed)} 失败")
    if failed:
        print("\n".join(failed[:10]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
