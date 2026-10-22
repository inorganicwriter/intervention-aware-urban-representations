"""Copy annual post-opening street views into the existing server project release."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

try:
    from .publish_server_streetview_inputs import copy_verified_image, sha256
    from .publish_streetview_inputs import metadata_values_match
except ImportError:
    from publish_server_streetview_inputs import copy_verified_image, sha256
    from publish_streetview_inputs import metadata_values_match


IDENTITY = ["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"]


def prepare_post_images(root: Path, snapshot: Path, workers: int) -> int:
    """Stage disjoint post images while the pre-opening tables are published."""
    audit = json.loads((snapshot / "post_audit.json").read_text())
    if not audit["run_finished"] or not all(audit["checks"].values()):
        raise ValueError("Completed post-image verification required")
    treatment_path = root / "data/active/causal/treatment_unit_list.parquet"
    if sha256(treatment_path) != audit["treatment_sha256"]:
        raise ValueError("Frozen treatment identity mismatch")
    for name, digest in audit["file_sha256"].items():
        if Path(name).name != name or sha256(snapshot / name) != digest:
            raise ValueError("Post snapshot changed")
    selected = pd.read_csv(snapshot / "streetview_post_selected.csv", dtype={"panoid": str})
    pre = pd.read_csv(snapshot.parent / "index/streetview_pre_selected.csv", dtype={"panoid": str})
    shared = set(zip(pre.city_key, pre.panoid, strict=True))
    unique = selected.drop_duplicates(["city_key", "panoid"])
    unique = unique.loc[[key not in shared for key in zip(unique.city_key, unique.panoid, strict=True)]]
    print(json.dumps({"post_image_staging_started": True, "images": len(unique)}), flush=True)
    receipts = []
    with ThreadPoolExecutor(workers) as pool:
        tasks = [pool.submit(copy_verified_image, record, root / "data/active/streetview_grid_v2/images", Path(audit["raw_root"]))
                 for record in unique.to_dict("records")]
        for task in as_completed(tasks):
            receipts.append(task.result())
            if len(receipts) % 500 == 0 or len(receipts) == len(tasks):
                print(json.dumps({"post_images_staged_verified": len(receipts), "total": len(tasks)}), flush=True)
    pd.DataFrame(receipts).to_parquet(snapshot.parent / "post_image_staging_receipt.parquet", index=False, compression="zstd")
    print("POST_IMAGE_STAGING_COMPLETE", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prepare-images-only", action="store_true")
    parser.add_argument("--verified-staging-receipt", type=Path,
                        help="Reuse copies already decoded and hashed by the preparation step.")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.prepare_images_only:
        return prepare_post_images(root, args.snapshot, args.workers)
    output = root / "data/active/streetview_grid_v2"
    stage = args.snapshot.parent / "post_published"
    stage.mkdir(parents=True, exist_ok=True)
    parent_path = output / "manifest.json"
    parent_sha = sha256(parent_path)
    parent = json.loads(parent_path.read_text())
    audit = json.loads((args.snapshot / "post_audit.json").read_text())
    if not audit["run_finished"] or audit["events"] != 5048 or audit["cities"] != 44:
        raise ValueError("Completed post-opening archive required")
    treatments = pd.read_parquet(root / "data/active/causal/treatment_unit_list.parquet")
    if sha256(root / "data/active/causal/treatment_unit_list.parquet") != audit["treatment_sha256"]:
        raise ValueError("Frozen treatment hash mismatch")
    if parent["treatment_sha256"] != audit["treatment_sha256"] or parent["grid_contract_sha256"] != audit["contract_sha256"]:
        raise ValueError("Post archive differs from the published frozen grid contract")
    source_audit = json.loads((output / "server_audit.json").read_text())
    if source_audit["file_sha256"]["streetview_index.csv"] != audit["source_index_sha256"]:
        raise ValueError("Post archive uses a different metadata source")
    for name, digest in audit["file_sha256"].items():
        if Path(name).name != name or sha256(args.snapshot / name) != digest:
            raise ValueError(f"Post snapshot checksum mismatch: {name}")
    selected = pd.read_csv(args.snapshot / "streetview_post_selected.csv",
                           dtype={"panoid": str, "capture_month": str, "opening_month": str})
    selected["capture_date"] = pd.to_datetime(selected.capture_date)
    bound = selected.merge(treatments[IDENTITY], on="treatment_order", suffixes=("", "_frozen"), validate="many_to_one")
    archive_path = output / "streetview_server_index.parquet"
    if sha256(archive_path) != parent["file_sha256"][archive_path.name]:
        raise ValueError("Published source index checksum mismatch")
    index = pd.read_parquet(archive_path, filters=[("panoid", "in", selected.panoid.unique().tolist())])
    lookup = selected.merge(index, on=["city_key", "grid_id", "panoid"], suffixes=("", "_index"), validate="many_to_one")
    common = [name for name in index if name in selected and name not in
              ["city_key", "grid_id", "panoid", "asset_bytes", "asset_path"]]
    checks = {"selected_frozen_identities_match": all(bound[name].equals(bound[name + "_frozen"]) for name in IDENTITY[1:]),
              "selected_rows_match_source_index": len(lookup) == len(selected),
              "capture_metadata_matches_audited_grid_index": all(metadata_values_match(lookup[name], lookup[name + "_index"]) for name in common),
              "strictly_after_opening_month": bool((selected.capture_month > selected.opening_month.str[:7]).all()),
              "correct_event_year": bool(selected.event_time.eq(selected.capture_month.str[:4].astype(int) - selected.opening_month.str[:4].astype(int)).all()),
              "event_times_zero_to_three": bool(selected.event_time.between(0, 3).all()),
              "post_images_never_state_inputs": bool((~selected.state_allowed).all()),
              "all_images_source_decode_passed": bool(selected.image_valid.all()),
              "same_pano_has_consistent_checksum": bool(selected.groupby(["city_key", "panoid"]).asset_sha256.nunique().le(1).all())}
    if not all(checks.values()):
        raise ValueError(f"Post-opening binding failed: {checks}")
    unique = selected.drop_duplicates(["city_key", "panoid"])
    if args.verified_staging_receipt:
        receipt = pd.read_parquet(args.verified_staging_receipt)
        bound_receipt = unique.merge(receipt, on=["city_key", "panoid"],
                                      suffixes=("", "_receipt"), validate="one_to_one")
        expected_paths = [str(output / "images" / row.city_key / f"{row.panoid}.jpg")
                          for row in bound_receipt.itertuples()]
        if (len(receipt) != len(unique) or len(bound_receipt) != len(unique)
                or not receipt.copied_and_verified.all()
                or not bound_receipt.asset_sha256.eq(bound_receipt.asset_sha256_receipt).all()
                or not bound_receipt.asset_bytes.eq(bound_receipt.asset_bytes_receipt).all()
                or bound_receipt.asset_path_receipt.tolist() != expected_paths):
            raise ValueError("Verified staging receipt differs from the selected images")

        def size_unchanged(row):
            return Path(row.asset_path).stat().st_size == row.asset_bytes

        with ThreadPoolExecutor(args.workers) as pool:
            if not all(pool.map(size_unchanged, receipt.itertuples())):
                raise ValueError("A staged image changed size before publication")
        receipts = receipt.to_dict("records")
        print(json.dumps({"verified_project_copies_reused": len(receipts)}), flush=True)
    else:
        print(json.dumps({"post_copy_started": True, "unique_images": len(unique)}), flush=True)
        receipts = []
        with ThreadPoolExecutor(args.workers) as pool:
            tasks = [pool.submit(copy_verified_image, record, output / "images", Path(audit["raw_root"]))
                     for record in unique.to_dict("records")]
            for task in as_completed(tasks):
                receipts.append(task.result())
                if len(receipts) % 500 == 0 or len(receipts) == len(tasks):
                    print(json.dumps({"post_images_copied_verified": len(receipts), "total": len(tasks)}), flush=True)
    selected = selected.rename(columns={"asset_path": "source_asset_path"})
    selected["asset_path"] = [str(output / "images" / row.city_key / f"{row.panoid}.jpg")
                              for row in selected.itertuples()]
    coverage = pd.read_csv(args.snapshot / "post_event_coverage.csv")
    paths = selected.groupby(["treatment_order", "event_time"]).asset_path.agg(lambda values: json.dumps(values.tolist()))
    coverage = coverage.merge(paths.rename("post_streetview_assets").reset_index(),
                              on=["treatment_order", "event_time"], how="left", validate="one_to_one")
    coverage["post_streetview_assets"] = coverage.post_streetview_assets.fillna("[]")
    expected = treatments[IDENTITY].merge(pd.DataFrame({"event_time": [0, 1, 2, 3]}), how="cross")
    checks["all_frozen_event_years_preserved"] = coverage[[*IDENTITY, "event_time"]].equals(expected[[*IDENTITY, "event_time"]])
    checks["coverage_masks_agree_with_project_paths"] = coverage.post_streetview_available.equals(coverage.post_streetview_assets.ne("[]"))
    checks["coverage_never_state_inputs"] = bool((~coverage.state_allowed).all())
    checks["all_post_images_copied_decoded_and_sha256_verified"] = len(receipts) == len(unique)
    if not all(checks.values()):
        raise ValueError(f"Post publication failed: {checks}")
    tables = {"streetview_post_assets.parquet": selected, "streetview_post_event_coverage.parquet": coverage,
              "post_copy_receipt.parquet": pd.DataFrame(receipts)}
    for name, frame in tables.items():
        frame.to_parquet(stage / name, index=False, compression="zstd")
    (stage / "post_server_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    (stage / "post_binding_audit.json").write_text(json.dumps(checks, indent=2))
    shutil.copyfile(Path(__file__), stage / "publish_server_streetview_post_archive.py")
    names = [*tables, "post_server_audit.json", "post_binding_audit.json", "publish_server_streetview_post_archive.py"]
    manifest = {"schema": audit["schema"], "release_status": "server_project_postopening_archive_verified",
                "published_at_utc": datetime.now(UTC).isoformat(), "asset_root": str(output),
                "events": 5048, "cities": 44, "event_times": [0, 1, 2, 3],
                "expected_event_years": len(coverage), "available_event_years": int(coverage.post_streetview_available.sum()),
                "available_events": int(selected.treatment_order.nunique()), "selected_rows": len(selected),
                "selected_unique_images": len(unique), "selected_unique_bytes": sum(row["asset_bytes"] for row in receipts),
                "by_event_time": coverage.groupby("event_time").post_streetview_available.sum().to_dict(),
                "status_counts": coverage.post_streetview_status.value_counts().to_dict(),
                "treatment_sha256": audit["treatment_sha256"], "grid_contract_sha256": audit["contract_sha256"],
                "capture_date_precision": "month", "temporal_rule": audit["temporal_rule"],
                "opening_month_relation": "unknown_excluded_from_verified_post_selection",
                "state_allowed": False, "client_image_downloads": False, "max_images_per_event_year": audit["max_images_per_event_year"],
                "file_sha256": {name: sha256(stage / name) for name in names}}
    (stage / "post_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    for name in [*names, "post_manifest.json"]:
        temporary = output / f".{name}.publishing"
        shutil.copyfile(stage / name, temporary)
        if sha256(temporary) != sha256(stage / name):
            raise ValueError(f"Published post file checksum mismatch: {name}")
        os.replace(temporary, output / name)
    if sha256(parent_path) != parent_sha:
        raise ValueError("Parent release changed while post archive was published")
    parent.update(post_archive_manifest="post_manifest.json", post_archive_manifest_sha256=sha256(output / "post_manifest.json"),
                  post_archive_available_event_years=manifest["available_event_years"],
                  post_archive_selected_images=manifest["selected_unique_images"], post_archive_state_allowed=False)
    temporary = output / ".manifest.json.postpublishing"
    temporary.write_text(json.dumps(parent, ensure_ascii=False, indent=2))
    os.replace(temporary, parent_path)
    print(json.dumps({k: manifest[k] for k in ["release_status", "expected_event_years", "available_event_years",
                                             "selected_unique_images", "by_event_time"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
