"""Finish this running transfer when its verified receiver has published all files."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver-pid", type=int, required=True)
    parser.add_argument("--receiver-log", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"waiting_for_existing_receiver": args.receiver_pid}), flush=True)
    process = Path("/proc") / str(args.receiver_pid)
    while process.exists():
        command = (process / "cmdline").read_bytes()
        if b"/tmp/mit_input_sync_20261013/receive_server_input_assets.py" not in command:
            raise ValueError("Receiver PID no longer belongs to the authorized transfer")
        time.sleep(10)
    completed = False
    for line in reversed(args.receiver_log.read_text().splitlines()):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("transfer_completed") is True:
            completed = True
            break
    if not completed:
        raise ValueError("The receiver exited before a verified publication; integration was not started")
    script = Path(__file__).with_name("integrate_server_input_assets.py")
    print("VERIFIED_TRANSFER_COMPLETE_STARTING_INPUT_BINDING", flush=True)
    subprocess.run([sys.executable, "-u", str(script), "--project-root", str(args.project_root),
                    "--run-root", str(args.run_root)], check=True)
    print("SERVER_INPUT_SYNC_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
