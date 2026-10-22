"""Copy verified street-view assets into the server project without client downloads.

The server's system Python writes Parquet; a separate geometry runtime performs
the polygon audit. Neither runtime nor the original collector archive is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_verified_image(record: dict, image_root: Path, raw_root: Path) -> dict:
    city, pano = record["city_key"], record["panoid"]
    if not re.fullmatch(r"[a-z]+", city) or not re.fullmatch(r"[A-Za-z0-9_-]+", pano):
        raise ValueError("Unsafe image identity")
    source = Path(record["asset_path"]).resolve()
    if not source.is_relative_to(raw_root.resolve() / "images"):
        raise ValueError(f"Source image outside the collector archive: {source}")
    destination = image_root / city / f"{pano}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jpg.copying")
    if destination.exists():
        if destination.stat().st_size != record["asset_bytes"] or sha256(destination) != record["asset_sha256"]:
            raise ValueError(f"Existing destination differs from the verified source: {destination}")
    else:
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size != record["asset_bytes"] or sha256(temporary) != record["asset_sha256"]:
            raise ValueError(f"Image changed during copying: {source}")
        os.replace(temporary, destination)
    with Image.open(destination) as image:
        image.load()
        if image.size != (record["width"], record["height"]):
            raise ValueError(f"Copied image dimensions changed: {destination}")
    return {"city_key": city, "panoid": pano, "asset_path": str(destination),
            "asset_bytes": record["asset_bytes"], "asset_sha256": record["asset_sha256"],
            "copied_and_verified": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--geometry-python", type=Path, required=True)
    parser.add_argument("--binding-validator", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "data/active/streetview_grid_v2"
    stage = args.snapshot.parent / "published"
    stage.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.snapshot / "audit.json").read_text())
    contract = json.loads(args.contract.read_text())
    treatment_path = root / "data/active/causal/treatment_unit_list.parquet"
    if not audit["run_finished"] or audit["cities"] != 44 or audit["events"] != 5048:
        raise ValueError("A completed 44-city rebuild is required")
    if sha256(treatment_path) != audit["treatment_sha256"] or sha256(args.contract) != audit["contract_sha256"]:
        raise ValueError("Server project and rebuild use different frozen inputs")
    for name, digest in audit["file_sha256"].items():
        if Path(name).name != name or sha256(args.snapshot / name) != digest:
            raise ValueError(f"Rebuild snapshot checksum mismatch: {name}")
    treatments = pd.read_parquet(treatment_path)
    identity = ["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"]
    events = pd.DataFrame(contract["events"])
    if not treatments[identity].equals(events[identity]):
        raise ValueError("Frozen event identities or order changed")
    geometry_audit = stage / "binding_audit.json"
    geometry_code = (
        "import sys,json,pandas as pd; from pathlib import Path; "
        "sys.path.insert(0,sys.argv[1]); from publish_streetview_inputs import assert_binding; "
        "s=Path(sys.argv[2]); c=json.loads(Path(sys.argv[3]).read_text()); "
        "dtype={'panoid':str,'capture_month':str,'opening_month':str}; "
        "i=pd.read_csv(s/'streetview_index.csv',dtype=dtype); "
        "p=pd.read_csv(s/'streetview_pre_selected.csv',dtype=dtype); "
        "checks=assert_binding(i,p,pd.DataFrame(c['events']),c); "
        "Path(sys.argv[4]).write_text(json.dumps(checks,indent=2)); "
        "print(json.dumps({'geometry_checks':len(checks),'passed':all(checks.values())}))"
    )
    subprocess.run([str(args.geometry_python), "-c", geometry_code,
                    str(args.binding_validator.parent), str(args.snapshot),
                    str(args.contract), str(geometry_audit)], check=True)
    checks = json.loads(geometry_audit.read_text())
    dtype = {"panoid": str, "capture_month": str, "opening_month": str}
    index = pd.read_csv(args.snapshot / "streetview_index.csv", dtype=dtype)
    selected = pd.read_csv(args.snapshot / "streetview_pre_selected.csv", dtype=dtype)
    unique = selected.drop_duplicates(["city_key", "panoid"])
    if len(unique) != audit["selected_unique_images"]:
        raise ValueError("Selected image identity count differs from the rebuild audit")
    print(json.dumps({"copy_started": True, "images": len(unique), "output": str(output)}), flush=True)
    receipts = []
    with ThreadPoolExecutor(args.workers) as pool:
        tasks = [pool.submit(copy_verified_image, row, output / "images", Path(audit["raw_root"]))
                 for row in unique.to_dict("records")]
        for task in as_completed(tasks):
            receipts.append(task.result())
            if len(receipts) % 100 == 0 or len(receipts) == len(tasks):
                print(json.dumps({"copied_verified": len(receipts), "total": len(tasks)}), flush=True)
    selected = selected.rename(columns={"asset_path": "source_asset_path"})
    selected["asset_path"] = [str(output / "images" / r.city_key / f"{r.panoid}.jpg")
                              for r in selected.itertuples()]
    selected["capture_date"] = pd.to_datetime(selected.capture_date)
    selected_index = selected.drop_duplicates(["city_key", "grid_id", "panoid"]).drop(
        columns=["treatment_order", "station_event_id", "opening_month"])
    counts = selected.groupby("treatment_order").size()
    coverage = treatments[identity].copy()
    coverage["streetview_image_count"] = coverage.treatment_order.map(counts).fillna(0).astype(int)
    coverage["streetview_available"] = coverage.streetview_image_count.gt(0)
    coverage["streetview_verification_complete"] = True
    candidates = coverage[identity].merge(index[["city_key", "grid_id", "capture_month"]],
                                          on=["city_key", "grid_id"], how="inner")
    any_grid = set(candidates.treatment_order)
    any_pre = set(candidates.loc[candidates.capture_month < candidates.opening_month.str[:7], "treatment_order"])
    available = set(selected.treatment_order)
    coverage["streetview_status"] = [
        "server_project_pretreatment_verified" if order in available else
        "preopening_candidates_but_no_verified_image" if order in any_pre else
        "only_opening_or_postopening_captures" if order in any_grid else
        "no_pano_metadata_in_formal_grid" for order in coverage.treatment_order]
    paths = selected.groupby("treatment_order").asset_path.agg(lambda values: json.dumps(values.tolist()))
    coverage["streetview_assets"] = coverage.treatment_order.map(paths).fillna("[]")
    expected_coverage = pd.read_csv(args.snapshot / "event_coverage.csv")
    checks["all_frozen_events_preserved"] = coverage[identity].equals(treatments[identity])
    checks["event_availability_matches_rebuild"] = coverage.streetview_available.equals(expected_coverage.streetview_available)
    checks["availability_agrees_with_asset_paths"] = coverage.streetview_available.equals(coverage.streetview_assets.ne("[]"))
    checks["all_selected_images_copied_decoded_and_sha256_verified"] = len(receipts) == len(unique)
    if not all(checks.values()):
        raise ValueError("Final server publication audit failed")
    archive = index.rename(columns={"asset_path": "source_asset_path"})
    archive["capture_date"] = pd.to_datetime(archive.capture_date)
    archive["asset_bytes"] = pd.Series(pd.NA, index=archive.index, dtype="Int64")
    archive["image_verification_status"] = "metadata_candidate_image_unverified"
    tables = {"streetview_server_index.parquet": archive,
              "streetview_index.parquet": selected_index,
              "streetview_state_assets.parquet": selected,
              "event_coverage.parquet": coverage,
              "copy_receipt.parquet": pd.DataFrame(receipts)}
    for name, frame in tables.items():
        frame.to_parquet(stage / name, index=False, compression="zstd")
    (stage / "server_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    (stage / "binding_audit.json").write_text(json.dumps(checks, indent=2))
    shutil.copyfile(args.contract, stage / "frozen_grid_contract.json")
    shutil.copyfile(Path(__file__), stage / "publish_server_streetview_inputs.py")
    file_names = [*tables, "server_audit.json", "binding_audit.json", "frozen_grid_contract.json",
                  "publish_server_streetview_inputs.py"]
    manifest = {"schema": "streetview_fixed_treated_grid_v2",
                "release_status": "server_project_pretreatment_assets_verified",
                "published_at_utc": datetime.now(UTC).isoformat(),
                "project_root": str(root), "asset_root": str(output), "role": "state_only",
                "full_collection_completed": True, "pending_verification_events": 0,
                "events": len(coverage), "cities": audit["cities"],
                "available_events": int(coverage.streetview_available.sum()),
                "missing_events": int((~coverage.streetview_available).sum()),
                "event_status_counts": coverage.streetview_status.value_counts().to_dict(),
                "metadata_candidate_rows": len(archive), "candidate_images_fully_verified": False,
                "selected_rows": len(selected), "selected_unique_images": len(unique),
                "selected_unique_bytes": sum(r["asset_bytes"] for r in receipts),
                "max_images_per_event": int(counts.max()), "server_images_verified": True,
                "treatment_sha256": audit["treatment_sha256"], "grid_contract_sha256": audit["contract_sha256"],
                "coordinate_source": audit["coordinate_source"], "spatial_support": audit["spatial_support"],
                "capture_date_precision": "month", "temporal_rule": audit["temporal_rule"],
                "original_collector_assets_preserved": True, "client_image_downloads": False,
                "model_migration_completed": False,
                "integration_table": str(output / "event_coverage.parquet"),
                "file_sha256": {name: sha256(stage / name) for name in file_names}}
    (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    # Publish the manifest last, after rechecking every destination table.
    for name in [*file_names, "manifest.json"]:
        destination = output / name
        temporary = output / f".{name}.publishing"
        shutil.copyfile(stage / name, temporary)
        if sha256(temporary) != sha256(stage / name):
            raise ValueError(f"Published file checksum mismatch: {name}")
        os.replace(temporary, destination)
    print(json.dumps({k: manifest[k] for k in ["release_status", "asset_root", "events", "cities",
                                             "available_events", "missing_events", "selected_unique_images"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
