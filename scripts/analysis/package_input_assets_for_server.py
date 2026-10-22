"""Package the five new input directories with a per-file content manifest."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

DATASETS = ["alphaearth_grid_v2", "satellite_grid_images_v1", "satellite_event_sequences_v2",
            "intervention_grid_v1", "multimodal_inputs_v1"]
ROOT = Path(__file__).resolve().parents[2]


def files_under(root: Path):
    with os.scandir(root) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            if entry.is_symlink():
                raise ValueError(f"Symlink in input asset tree: {entry.path}")
            if entry.is_dir(follow_symlinks=False):
                yield from files_under(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                yield Path(entry.path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "latest_input_assets.tar.gz"
    temporary = archive.with_suffix(".gz.packing")
    records, counts, sizes = [], Counter(), Counter()
    support = [(ROOT / "data/active/causal/treatment_unit_list.parquet", "support/treatment_unit_list.parquet"),
               (ROOT / "outputs/satellite_event_sequences_audit/2026-10-13/audit.json", "support/satellite_image_audit.json"),
               (ROOT / "outputs/satellite_event_sequences_audit/2026-10-13/image_quality_by_frame.csv", "support/satellite_image_quality.csv"),
               (ROOT / "outputs/spatial_support_audit/2026-10-13/alphaearth_spatial_audit.json", "support/alphaearth_spatial_audit.json"),
               (ROOT / "outputs/causal_labels/station_attributes/station_attributes.parquet", "support/station_attributes.parquet")]
    with tarfile.open(temporary, "w:gz", compresslevel=1, format=tarfile.GNU_FORMAT) as tar:
        for name in DATASETS:
            for source in files_under(ROOT / "data/active" / name):
                before = source.stat()
                data = source.read_bytes()
                after = source.stat()
                if before.st_size != len(data) or before.st_mtime_ns != after.st_mtime_ns:
                    raise ValueError(f"Source changed while packing: {source}")
                relative = source.relative_to(ROOT).as_posix()
                info = tarfile.TarInfo(relative)
                info.size, info.mode = len(data), 0o644
                tar.addfile(info, io.BytesIO(data))
                records.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                counts[name] += 1
                sizes[name] += len(data)
                if len(records) % 10000 == 0:
                    print(json.dumps({"packed_files": len(records), "dataset": name}), flush=True)
            print(json.dumps({"dataset_completed": name, "files": counts[name], "bytes": sizes[name]}), flush=True)
        for source, relative in support:
            if not source.is_file():
                raise ValueError(f"Required support evidence is missing: {source}")
            data = source.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
            records.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        manifest = {"schema": "input_asset_server_transfer_v1", "created_at_utc": datetime.now(UTC).isoformat(),
                    "datasets": DATASETS, "source_project": str(ROOT), "files": records,
                    "file_count": len(records), "file_bytes": sum(record["bytes"] for record in records),
                    "dataset_file_counts": dict(counts), "dataset_bytes": dict(sizes),
                    "frozen_treatment_sha256": sha256(ROOT / "data/active/causal/treatment_unit_list.parquet")}
        body = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        info = tarfile.TarInfo("transfer_manifest.json")
        info.size, info.mode = len(body), 0o644
        tar.addfile(info, io.BytesIO(body))
    os.replace(temporary, archive)
    (args.output / "transfer_manifest.json").write_bytes(body)
    receipt = {"archive": str(archive), "archive_bytes": archive.stat().st_size,
               "archive_sha256": sha256(archive), "manifest_sha256": hashlib.sha256(body).hexdigest(),
               "file_count": manifest["file_count"], "file_bytes": manifest["file_bytes"],
               "datasets": DATASETS, "dataset_file_counts": dict(counts), "dataset_bytes": dict(sizes),
               "frozen_treatment_sha256": manifest["frozen_treatment_sha256"]}
    (args.output / "package_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
