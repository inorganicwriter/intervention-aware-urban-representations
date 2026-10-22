"""Upload the prepared package with a pinned host key and resumable SFTP writes."""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import posixpath
import time
from pathlib import Path

import paramiko


class PinnedHostKey(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        fingerprint = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
        if fingerprint != "cenKYeEFnQlpCno9O9Itd3tJqybVSsaacSY7/Ghcewg":
            raise paramiko.SSHException("Server host key does not match the previously verified key")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    args = parser.parse_args()
    if args.remote_root != "/tmp/mit_input_sync_20261013":
        raise ValueError("Unexpected upload staging directory")
    receipt = json.loads((args.package_dir / "package_receipt.json").read_text())
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(PinnedHostKey())
    password = getpass.getpass("SSH password: ")
    client.connect("222.29.102.166", port=7007, username="root", password=password,
                   look_for_keys=False, allow_agent=False, timeout=20)
    del password
    transport = client.get_transport()
    transport.set_keepalive(30)
    with paramiko.SFTPClient.from_transport(transport, window_size=64 * 1024 * 1024) as sftp:
        for name in ["transfer_manifest.json", "package_receipt.json", "latest_input_assets.tar.gz"]:
            local = args.package_dir / name
            final = posixpath.join(args.remote_root, name)
            partial = final + ".uploading"
            size = local.stat().st_size
            try:
                if sftp.stat(final).st_size == size:
                    print(json.dumps({"already_uploaded": name, "bytes": size}), flush=True)
                    continue
                raise ValueError(f"Existing upload differs in size: {final}")
            except FileNotFoundError:
                pass
            try:
                offset = sftp.stat(partial).st_size
            except FileNotFoundError:
                offset = 0
            if offset > size:
                raise ValueError("Remote partial upload is larger than the source")
            start, last = time.monotonic(), offset
            print(json.dumps({"upload_started": name, "bytes": size, "resume_offset": offset}), flush=True)
            with local.open("rb") as source, sftp.open(partial, "ab" if offset else "wb") as target:
                target.set_pipelined(True)
                source.seek(offset)
                while block := source.read(1024 * 1024):
                    target.write(block)
                    offset += len(block)
                    if offset - last >= 16 * 1024 * 1024 or offset == size:
                        print(json.dumps({"file": name, "uploaded_bytes": offset, "total_bytes": size,
                                          "elapsed_seconds": round(time.monotonic() - start, 1)}), flush=True)
                        last = offset
            if sftp.stat(partial).st_size != size:
                raise ValueError("Uploaded byte count mismatch")
            sftp.rename(partial, final)
            print(json.dumps({"upload_completed": name, "bytes": size}), flush=True)
    client.close()
    print(json.dumps({"package_uploaded": True, "archive_sha256": receipt["archive_sha256"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
