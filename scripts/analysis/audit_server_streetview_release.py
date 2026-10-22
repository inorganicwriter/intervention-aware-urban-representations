"""Verify published server manifests, frozen identities and temporal role masks."""
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

try:
    from .publish_server_streetview_inputs import sha256
except ImportError:
    from publish_server_streetview_inputs import sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "data/active/streetview_grid_v2"
    primary = json.loads((output / "manifest.json").read_text())
    post = json.loads((output / "post_manifest.json").read_text())
    checks = {}
    for prefix, manifest in [("pre", primary), ("post", post)]:
        for name, digest in manifest["file_sha256"].items():
            path = output / name
            if ".." in Path(name).parts or not path.is_relative_to(output):
                raise ValueError("Manifest path escapes the dataset directory")
            checks[f"{prefix}_sha256_{name}"] = sha256(path) == digest
    checks["parent_post_manifest_sha256"] = primary["post_archive_manifest_sha256"] == sha256(output / "post_manifest.json")
    treatment_path = root / "data/active/causal/treatment_unit_list.parquet"
    treatment_sha = sha256(treatment_path)
    checks["frozen_treatment_sha256"] = primary["treatment_sha256"] == post["treatment_sha256"] == treatment_sha
    checks["frozen_geometry_contract_sha256"] = primary["grid_contract_sha256"] == post["grid_contract_sha256"] == sha256(output / "frozen_grid_contract.json")
    treatments = pd.read_parquet(treatment_path)
    pre = pd.read_parquet(output / "streetview_state_assets.parquet")
    post_assets = pd.read_parquet(output / "streetview_post_assets.parquet")
    pre_coverage = pd.read_parquet(output / "event_coverage.parquet")
    post_coverage = pd.read_parquet(output / "streetview_post_event_coverage.parquet")
    identity = ["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"]
    checks["all_pre_frozen_events_preserved"] = pre_coverage[identity].equals(treatments[identity])
    expected_post = treatments[identity].merge(pd.DataFrame({"event_time": [0, 1, 2, 3]}), how="cross")
    checks["all_post_frozen_event_years_preserved"] = post_coverage[[*identity, "event_time"]].equals(expected_post[[*identity, "event_time"]])
    checks["pre_images_strictly_before_opening_month"] = bool((pre.capture_month < pre.opening_month.str[:7]).all())
    checks["post_images_strictly_after_opening_month"] = bool((post_assets.capture_month > post_assets.opening_month.str[:7]).all())
    checks["post_images_never_state_inputs"] = bool((~post_assets.state_allowed).all() and (~post_coverage.state_allowed).all())
    checks["pre_mask_matches_paths"] = pre_coverage.streetview_available.equals(pre_coverage.streetview_assets.ne("[]"))
    checks["post_mask_matches_paths"] = post_coverage.post_streetview_available.equals(post_coverage.post_streetview_assets.ne("[]"))
    checks["pre_available_count_matches_manifest"] = int(pre_coverage.streetview_available.sum()) == primary["available_events"]
    checks["post_available_count_matches_manifest"] = int(post_coverage.post_streetview_available.sum()) == post["available_event_years"]
    for prefix, frame, receipt_name in [("pre", pre, "copy_receipt.parquet"), ("post", post_assets, "post_copy_receipt.parquet")]:
        receipt = pd.read_parquet(output / receipt_name)
        unique = frame.drop_duplicates("asset_path")
        checks[f"{prefix}_receipt_covers_all_selected_images"] = set(unique.asset_path) == set(receipt.asset_path)
        checks[f"{prefix}_all_copy_receipts_verified"] = bool(receipt.copied_and_verified.all())
        compare = unique.merge(receipt, on="asset_path", suffixes=("", "_receipt"), validate="one_to_one")
        checks[f"{prefix}_copy_bytes_and_hashes_match_selection"] = bool(compare.asset_bytes.eq(compare.asset_bytes_receipt).all() and compare.asset_sha256.eq(compare.asset_sha256_receipt).all())
    expected_images = set(pd.concat([pre.asset_path, post_assets.asset_path]))
    actual_images = set()
    for city in set(treatments.city_key):
        directory = output / "images" / city
        if directory.exists():
            with os.scandir(directory) as entries:
                actual_images.update(entry.path for entry in entries if entry.name.endswith(".jpg"))
    checks["physical_project_image_inventory_matches_selection"] = actual_images == expected_images
    for name in ["binding_audit.json", "post_binding_audit.json"]:
        binding = json.loads((output / name).read_text())
        checks[f"all_{name}_checks_passed"] = all(binding.values())
    summary = {"schema": "server_streetview_publication_verification_v1",
               "verified_at_utc": datetime.now(UTC).isoformat(), "project_root": str(root),
               "all_checks_passed": all(checks.values()), "check_count": len(checks), "checks": checks,
               "pre_available_events": primary["available_events"], "pre_missing_events": primary["missing_events"],
               "pre_unique_images": primary["selected_unique_images"], "post_available_events": post["available_events"],
               "post_available_event_years": post["available_event_years"], "post_unique_images": post["selected_unique_images"],
               "physical_project_images": len(actual_images), "post_by_event_time": post["by_event_time"],
               "primary_manifest_sha256": sha256(output / "manifest.json"),
               "post_manifest_sha256": sha256(output / "post_manifest.json"),
               "frozen_treatment_sha256": treatment_sha, "client_images_downloaded": False,
               "other_latest_modalities_synchronized_to_server": False, "model_migration_completed": False}
    (output / "publication_verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_checks_passed"]:
        raise ValueError(f"Server release verification failed: {[name for name, passed in checks.items() if not passed]}")
    print(json.dumps({k:v for k,v in summary.items() if k != "checks"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
