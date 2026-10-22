"""Verify and publish a scoped input package into the existing server project."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tarfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

DATASETS = ["alphaearth_grid_v2", "satellite_grid_images_v1", "satellite_event_sequences_v2",
            "intervention_grid_v1", "multimodal_inputs_v1"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    if path.parts[:2] == ("data", "active") and len(path.parts) >= 4 and path.parts[2] in DATASETS:
        return
    if len(path.parts) == 2 and path.parts[0] == "support":
        return
    raise ValueError(f"Archive path outside the authorized input assets: {name}")


def write_verified(destination: Path, data: bytes, record: dict) -> None:
    if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"Source package content checksum mismatch: {record['path']}")
    temporary = destination.with_name(destination.name + ".receiving")
    with temporary.open("wb") as stream:
        stream.write(data)
    if sha256(temporary) != record["sha256"]:
        raise ValueError(f"Stored file checksum mismatch: {record['path']}")
    os.replace(temporary, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--native-root", type=Path)
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    run = args.run_root.resolve()
    if project != Path("/home/nas/lsr/MIT_summer") or not run.is_relative_to(project / "outputs"):
        raise ValueError("Unexpected project or publication directory")
    receipt = json.loads((args.package_root / "package_receipt.json").read_text())
    manifest_path = args.package_root / "transfer_manifest.json"
    if sha256(manifest_path) != receipt["manifest_sha256"]:
        raise ValueError("Transfer manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    archive = args.package_root / "latest_input_assets.tar.gz"
    if archive.stat().st_size != receipt["archive_bytes"] or sha256(archive) != receipt["archive_sha256"]:
        raise ValueError("Uploaded archive checksum mismatch")
    treatment = project / "data/active/causal/treatment_unit_list.parquet"
    if sha256(treatment) != receipt["frozen_treatment_sha256"]:
        raise ValueError("Server uses a different frozen treatment list")
    records = {record["path"]: record for record in manifest["files"]}
    if len(records) != manifest["file_count"]:
        raise ValueError("Duplicate or missing transfer entries")
    for name in records:
        validate_name(name)
    native = args.native_root
    if native is not None:
        if not native.absolute().is_relative_to(Path("/tmp/mit_input_sync_20261013")):
            raise ValueError("Native extraction escaped the designated temporary directory")
        native_receipt = args.package_root / "native_extraction_receipt.json"
        if args.native_only:
            native.mkdir(exist_ok=True)
            parents = {native / PurePosixPath(name).parent for name in records}
            for parent in sorted(parents):
                parent.mkdir(parents=True, exist_ok=True)
            seen = set()
            with tarfile.open(archive, "r|gz") as tar:
                for member in tar:
                    if not member.isfile():
                        raise ValueError("Non-regular native archive member")
                    stream = tar.extractfile(member)
                    if member.name == "transfer_manifest.json":
                        if hashlib.sha256(stream.read()).hexdigest() != receipt["manifest_sha256"]:
                            raise ValueError("Embedded native manifest mismatch")
                        continue
                    validate_name(member.name)
                    if member.name not in records or member.name in seen:
                        raise ValueError("Unexpected native member")
                    write_verified(native / member.name, stream.read(), records[member.name])
                    seen.add(member.name)
                    if len(seen) % 10000 == 0:
                        print(json.dumps({"native_files_verified": len(seen), "total": len(records)}), flush=True)
            if seen != set(records):
                raise ValueError("Native extraction incomplete")
            native_receipt.write_text(json.dumps({"completed": True, "files": len(seen),
                                                  "archive_sha256": receipt["archive_sha256"]}))
            print("NATIVE_EXTRACTION_COMPLETE", flush=True)
            return 0
        status = json.loads(native_receipt.read_text())
        if not status["completed"] or status["archive_sha256"] != receipt["archive_sha256"]:
            raise ValueError("Native source extraction is not verified")
    for name in DATASETS:
        if (project / "data/active" / name).exists():
            raise ValueError(f"Destination already exists; preserve its version: {name}")
    incoming = run / "incoming"
    if incoming.exists() and native is None:
        raise ValueError("Incoming directory already exists; inspect the previous receiver before resuming")
    incoming.mkdir(exist_ok=True)
    parents = {incoming / PurePosixPath(name).parent for name in records}
    for parent in sorted(parents):
        parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"package_sha256_verified": True, "expected_files": len(records),
                      "expected_bytes": manifest["file_bytes"]}), flush=True)
    if native is not None:
        names = list(records)
        random.Random(20261013).shuffle(names)

        def copy_native(name):
            destination = incoming / name
            # Directory listings provide positive entries; resume does not issue
            # a negative stat for every new small file on NFS.
            parent_names = inventory[destination.parent]
            if destination.name in parent_names:
                if sha256(destination) != records[name]["sha256"]:
                    raise ValueError("Previously copied content differs from the source")
                return name
            write_verified(destination, (native / name).read_bytes(), records[name])
            return name

        inventory = {}
        for parent in parents:
            with os.scandir(parent) as entries:
                inventory[parent] = {entry.name for entry in entries}
        complete = 0
        with ThreadPoolExecutor(args.workers) as pool:
            for _ in pool.map(copy_native, names):
                complete += 1
                if complete % 5000 == 0:
                    print(json.dumps({"stored_and_sha256_verified": complete, "total": len(records)}), flush=True)
        seen = set(names)
    else:
        complete, seen = extract_stream(archive, incoming, receipt, records, args.workers)
    if complete != len(records) or seen != set(records):
        raise ValueError("Incomplete archive extraction")
    if sha256(incoming / "support/treatment_unit_list.parquet") != sha256(treatment):
        raise ValueError("Source treatment snapshot differs from the frozen server list")
    if any((project / "data/active" / name).exists() for name in DATASETS):
        raise ValueError("A destination appeared during the upload; no existing version will be replaced")
    published = []
    for name in DATASETS:
        source, target = incoming / "data/active" / name, project / "data/active" / name
        if not source.resolve().is_relative_to(run) or not target.parent.resolve() == project / "data/active":
            raise ValueError("Publication paths escaped the intended project")
        os.rename(source, target)
        published.append(name)
        (run / "publication_progress.json").write_text(json.dumps({"published": published}))
    result = {"schema": "server_input_transfer_receipt_v1", "transfer_completed": True,
              "published_at_utc": datetime.now(UTC).isoformat(), "project_root": str(project),
              "published_datasets": published, "verified_files": complete, "verified_bytes": manifest["file_bytes"],
              "archive_sha256": receipt["archive_sha256"], "manifest_sha256": receipt["manifest_sha256"],
              "frozen_treatment_sha256": sha256(treatment), "source_manifest_path": str(manifest_path),
              "support_root": str(incoming / "support"), "existing_streetview_release_preserved": True,
              "integration_completed": False}
    (run / "transfer_receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result), flush=True)
    return 0


def extract_stream(archive: Path, incoming: Path, receipt: dict, records: dict, workers: int):
    complete, pending, seen = 0, set(), set()
    with ThreadPoolExecutor(workers) as pool, tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                raise ValueError(f"Non-regular package member: {member.name}")
            stream = tar.extractfile(member)
            if member.name == "transfer_manifest.json":
                if hashlib.sha256(stream.read()).hexdigest() != receipt["manifest_sha256"]:
                    raise ValueError("Embedded manifest differs from the supplied manifest")
                continue
            validate_name(member.name)
            if member.name not in records or member.name in seen:
                raise ValueError("Unexpected or duplicate archive member")
            seen.add(member.name)
            data = stream.read()
            pending.add(pool.submit(write_verified, incoming / member.name, data, records[member.name]))
            if len(pending) >= workers * 4:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for task in done:
                    task.result()
                old = complete
                complete += len(done)
                if complete // 5000 > old // 5000:
                    print(json.dumps({"stored_and_sha256_verified": complete, "total": len(records)}), flush=True)
        for task in pending:
            task.result()
        complete += len(pending)
    return complete, seen


if __name__ == "__main__":
    raise SystemExit(main())
