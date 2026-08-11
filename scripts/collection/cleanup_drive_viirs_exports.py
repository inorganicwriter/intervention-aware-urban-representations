"""Remove stale VIIRS CSV exports from the dedicated Drive folder."""

import argparse
import time

import ee
from googleapiclient.discovery import build

FOLDER_NAME = "MIT_Summer_VIIRS"


def _execute(request):
    for attempt in range(20):
        try:
            return request.execute()
        except Exception:
            if attempt == 19:
                raise
            time.sleep(2)


def _list(service, query: str, fields: str) -> list[dict]:
    rows = []
    token = None
    while True:
        response = _execute(
            service.files().list(
                q=query,
                fields=f"nextPageToken, files({fields})",
                pageToken=token,
                pageSize=1000,
            )
        )
        rows.extend(response.get("files", []))
        token = response.get("nextPageToken")
        if not token:
            return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    ee.Initialize(project="macro-city-engine")
    service = build(
        "drive",
        "v3",
        credentials=ee.data.get_persistent_credentials(),
        cache_discovery=False,
    )
    folders = _list(
        service,
        "name='MIT_Summer_VIIRS' and mimeType='application/vnd.google-apps.folder' "
        "and trashed=false",
        "id,name",
    )
    files = []
    for folder in folders:
        files.extend(
            _list(
                service,
                f"'{folder['id']}' in parents and name contains 'viirs_' and trashed=false",
                "id,name,mimeType",
            )
        )
    print(f"folders={len(folders)} stale_viirs_files={len(files)}", flush=True)
    if args.execute:
        for item in files:
            _execute(service.files().delete(fileId=item["id"]))
        print(f"deleted={len(files)}", flush=True)


if __name__ == "__main__":
    main()
