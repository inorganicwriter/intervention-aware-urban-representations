"""Fetch only the verified pretreatment training selection through SFTP.

SSH password is prompted interactively and is never saved or passed as an
argument. A server fingerprint is required. Each downloaded original is
checked against the server audit's SHA256 before being published locally.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from threading import local

import paramiko


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", default="root")
    parser.add_argument("--host-key-sha256", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = Path(manifest["local_image_root"]).resolve()
    remote_root = PurePosixPath(manifest["remote_image_root"])

    class PinnedKey(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, key):
            actual = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
            if actual != args.host_key_sha256:
                raise ValueError("SSH server fingerprint changed")
            client.get_host_keys().add(hostname, key.get_name(), key)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(PinnedKey())
    password = getpass.getpass("SSH password: ")
    client.connect(args.host, port=args.port, username=args.user, password=password,
                   look_for_keys=False, allow_agent=False, timeout=20)
    del password
    state = local()
    channels = []

    def fetch(record):
        city, pano = record["city_key"], record["panoid"]
        if not re.fullmatch(r"[a-z0-9_]+", city) or not re.fullmatch(r"[A-Za-z0-9_-]+", pano):
            raise ValueError("Unsafe asset identifier")
        remote = PurePosixPath(record["remote_asset_path"])
        if ".." in remote.parts or not remote.is_relative_to(remote_root):
            raise ValueError("Remote asset escapes the authorized image root")
        target = root / city / f"{pano}.jpg"
        if target.is_file() and target.stat().st_size == record["asset_bytes"] and digest(target) == record["asset_sha256"]:
            return {"panoid": pano, "local_path": str(target), "status": "cached"}
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".jpg.partial")
        if not hasattr(state, "sftp"):
            state.sftp = client.open_sftp()
            channels.append(state.sftp)
        last_error = None
        for _ in range(3):
            try:
                state.sftp.get(str(remote), str(partial))
                if partial.stat().st_size != record["asset_bytes"] or digest(partial) != record["asset_sha256"]:
                    raise ValueError("Image checksum disagrees with the server audit")
                partial.replace(target)
                return {"panoid": pano, "local_path": str(target), "status": "downloaded"}
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Could not fetch {pano}: {last_error}")

    records = []
    try:
        with ThreadPoolExecutor(args.workers) as pool:
            for record in pool.map(fetch, manifest["assets"]):
                records.append(record)
                if len(records) % 100 == 0:
                    print(json.dumps({"verified_images": len(records), "total": len(manifest["assets"])}), flush=True)
    finally:
        for channel in channels:
            channel.close()
        client.close()
    receipt = {"run_finished": True, "host_key_sha256": args.host_key_sha256,
               "images": len(records), "manifest_sha256": digest(args.manifest), "assets": records}
    args.manifest.with_name("download_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"run_finished": True, "verified_images": len(records)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
