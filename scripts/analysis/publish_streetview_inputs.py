"""Audit a server rebuild, stage the selected images and publish role references."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TREATMENTS = ROOT / "data/active/causal/treatment_unit_list.parquet"
DEFAULT_OUTPUT = ROOT / "data/active/streetview_grid_v2"
ROLES = ROOT / "data/active/multimodal_inputs_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata_values_match(left: pd.Series, right: pd.Series) -> bool:
    """Allow only floating-point round-trip noise, not changed capture metadata."""
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        return bool(np.isclose(left.to_numpy(), right.to_numpy(),
                               rtol=4 * np.finfo(float).eps, atol=0, equal_nan=False).all())
    return bool(left.eq(right).all())


def assert_binding(index: pd.DataFrame, selected: pd.DataFrame, treatments: pd.DataFrame,
                   contract: dict) -> dict:
    import shapely

    checks = {}
    keys = ["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"]
    bound = selected.merge(treatments[keys], on="treatment_order", suffixes=("", "_frozen"), validate="many_to_one")
    checks["selected_identity_matches_frozen_events"] = all(bound[k].equals(bound[f"{k}_frozen"]) for k in keys[1:])
    checks["selected_months_strictly_before_opening"] = bool((bound.capture_month < bound.opening_month.str[:7]).all())
    checks["selected_capture_precision_month"] = bool(selected.capture_date_precision.eq("month").all())
    checks["selected_images_server_decode_passed"] = bool(selected.image_valid.all())
    checks["selected_sha256_present"] = bool(selected.asset_sha256.str.fullmatch("[0-9a-f]{64}").all())
    checks["same_image_has_consistent_checksum"] = bool(selected.groupby("asset_path").asset_sha256.nunique().le(1).all())
    checks["index_unique_city_pano"] = not index.duplicated(["city_key", "panoid"]).any()
    checks["selected_keys_exist_in_index"] = bool(selected.set_index(["city_key", "grid_id", "panoid"]).index.isin(index.set_index(["city_key", "grid_id", "panoid"]).index).all())
    lookup = selected.merge(index, on=["city_key", "grid_id", "panoid"], suffixes=("", "_index"), validate="many_to_one")
    shared = [name for name in index if name not in ["city_key", "grid_id", "panoid", "asset_path", "asset_bytes"]]
    checks["selected_metadata_equal_server_index"] = len(lookup) == len(selected) and all(
        metadata_values_match(lookup[name], lookup[f"{name}_index"]) for name in shared)
    for city, spec in contract["cities"].items():
        frame = index.loc[index.city_key.eq(city)]
        geometry = {g["grid_id"]: g for g in spec["grids"]}
        checks[f"{city}_grid_identity"] = bool(frame.grid_id.isin(geometry).all())
        hashes = frame.grid_id.map({k: v["grid_geometry_sha256"] for k, v in geometry.items()})
        checks[f"{city}_geometry_hash"] = hashes.equals(frame.grid_geometry_sha256)
        if not checks[f"{city}_grid_identity"]:
            continue
        # Parse each frozen polygon once, instead of millions of identical WKT
        # strings. Prepared polygons retain the same boundary-inclusive rule.
        polygons = shapely.from_wkt([v["geometry_wkt"] for v in geometry.values()])
        shapely.prepare(polygons)
        positions = frame.grid_id.map({key: position for position, key in enumerate(geometry)}).to_numpy()
        points = shapely.points(frame.wgs_lon.to_numpy(), frame.wgs_lat.to_numpy())
        checks[f"{city}_actual_capture_inside_grid"] = bool(shapely.covers(polygons[positions], points).all())
    if not all(checks.values()):
        raise ValueError(f"Street-view binding failed: {[k for k,v in checks.items() if not v]}")
    return checks


def publish_verified_pilot(snapshot: Path, contract_path: Path, output: Path) -> int:
    """Publish verified local assets without calling unvisited cities unavailable."""
    audit = json.loads((snapshot / "audit.json").read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not audit["run_finished"] or audit["cities"] != 1 or set(audit["per_city"]) != {"beijing"}:
        raise ValueError("This recovery path requires the audited Beijing pilot")
    if audit["treatment_sha256"] != sha256(TREATMENTS) or audit["contract_sha256"] != sha256(contract_path):
        raise ValueError("Pilot frozen identity mismatch")
    source = snapshot / "streetview_pre_selected.csv"
    if sha256(source) != audit["file_sha256"][source.name]:
        raise ValueError("Pilot selection checksum mismatch")
    treatments = pd.read_parquet(TREATMENTS)
    selected = pd.read_csv(source, dtype={"panoid": str, "capture_month": str, "opening_month": str})
    extra = ["treatment_order", "station_event_id", "opening_month", "image_valid", "width", "height",
             "image_mode", "image_format", "asset_sha256", "error"]
    server_index = selected.drop(columns=extra).drop_duplicates(["city_key", "grid_id", "panoid"])
    checks = assert_binding(server_index, selected, treatments,
                            {"cities": {"beijing": contract["cities"]["beijing"]}})
    selected = selected.rename(columns={"asset_path": "remote_asset_path"})
    selected["asset_path"] = [str((output / "images" / r.city_key / f"{r.panoid}.jpg").resolve()) for r in selected.itertuples()]
    for row in selected.itertuples():
        path = Path(row.asset_path)
        if not path.is_file() or path.stat().st_size != row.asset_bytes or sha256(path) != row.asset_sha256:
            raise ValueError(f"Pilot local image missing or changed: {path}")
        with Image.open(path) as image:
            image.load()
            if image.size != (row.width, row.height):
                raise ValueError(f"Pilot image dimensions mismatch: {path}")
    checks["all_local_pilot_images_decoded_and_sha256_verified"] = True
    coverage = treatments[["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"]].copy()
    counts = selected.groupby("treatment_order").size()
    coverage["streetview_image_count"] = coverage.treatment_order.map(counts).fillna(0).astype(int)
    coverage["streetview_available"] = coverage.streetview_image_count.gt(0)
    coverage["streetview_verification_complete"] = coverage.city_key.eq("beijing")
    coverage["streetview_status"] = "pending_full_verification_nas_io_blocked"
    coverage.loc[coverage.city_key.eq("beijing"), "streetview_status"] = "not_available_in_verified_beijing_pilot"
    coverage.loc[coverage.streetview_available, "streetview_status"] = "local_pretreatment_verified"
    selected["capture_date"] = pd.to_datetime(selected.capture_date)
    index = selected.drop(columns=["treatment_order", "station_event_id", "opening_month"]).drop_duplicates(["city_key", "grid_id", "panoid"])
    output.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output / "streetview_state_assets.parquet", index=False, compression="zstd")
    index.to_parquet(output / "streetview_index.parquet", index=False, compression="zstd")
    coverage.to_parquet(output / "event_coverage.parquet", index=False, compression="zstd")
    (output / "server_pilot_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"schema": "streetview_fixed_treated_grid_v2", "release_status": "partial_verified_pilot_pending_nas_io",
                "full_collection_completed": False, "published_at_utc": datetime.now(UTC).isoformat(),
                "treatment_sha256": sha256(TREATMENTS), "grid_contract_sha256": sha256(contract_path),
                "role": "state_only", "events": len(coverage), "verified_cities": ["beijing"],
                "verified_city_events": int(coverage.streetview_verification_complete.sum()),
                "available_events": int(coverage.streetview_available.sum()),
                "pending_verification_events": int((~coverage.streetview_verification_complete).sum()),
                "verified_pilot_events_without_available_image": int((coverage.streetview_verification_complete & ~coverage.streetview_available).sum()),
                "selected_rows": len(selected), "selected_unique_images": len(index),
                "selected_unique_bytes": int(index.asset_bytes.sum()), "max_images_per_event": 2,
                "spatial_support": "unique_actual_pano_point_in_frozen_treated_grid_polygon",
                "coordinate_source": audit["coordinate_source"], "capture_date_precision": "month",
                "temporal_rule": "capture_month < opening_month", "original_server_images_preserved": True,
                "selected_local_images_verified": True, "model_migration_completed": False,
                "file_sha256": {name: sha256(output / name) for name in ["streetview_index.parquet", "streetview_state_assets.parquet", "event_coverage.parquet", "server_pilot_audit.json"]}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    state = pd.read_parquet(ROLES / "state_inputs.parquet")
    before = state.drop(columns=[k for k in state if k.startswith("streetview_")]).copy()
    state = before.merge(coverage[["treatment_order", "streetview_available", "streetview_status",
                                  "streetview_image_count", "streetview_verification_complete"]], on="treatment_order", validate="one_to_one")
    paths = selected.groupby("treatment_order").asset_path.agg(lambda s: json.dumps(s.tolist()))
    state["streetview_assets"] = state.treatment_order.map(paths).fillna("[]")
    checks["all_frozen_events_preserved"] = state.treatment_order.tolist() == treatments.treatment_order.tolist()
    checks["other_state_fields_unchanged"] = before.equals(state[before.columns])
    checks["local_availability_agrees_with_paths"] = state.streetview_available.equals(state.streetview_assets.ne("[]"))
    if not all(checks.values()):
        raise ValueError("Pilot role binding failed")
    state.to_parquet(ROLES / "state_inputs.parquet", index=False, compression="zstd")
    role_path = ROLES / "manifest.json"
    role = json.loads(role_path.read_text(encoding="utf-8"))
    role.update(streetview_status=manifest["release_status"], streetview_root=output.relative_to(ROOT).as_posix(),
                streetview_state_assets=(output / "streetview_state_assets.parquet").relative_to(ROOT).as_posix(),
                streetview_available_events=manifest["available_events"], streetview_verified_cities=["beijing"],
                streetview_pending_verification_events=manifest["pending_verification_events"],
                streetview_full_collection_completed=False, streetview_manifest_sha256=sha256(output / "manifest.json"),
                bundle_status="tabular_annual_satellite_and_partial_streetview_assets_published",
                updated_at_utc=datetime.now(UTC).isoformat())
    role["availability_counts"]["state"]["streetview_available"] = manifest["available_events"]
    role["input_table_sha256"]["state_inputs.parquet"] = sha256(ROLES / "state_inputs.parquet")
    role_path.write_text(json.dumps(role, ensure_ascii=False, indent=2), encoding="utf-8")
    checks["state_hash_matches_role_manifest"] = sha256(ROLES / "state_inputs.parquet") == role["input_table_sha256"]["state_inputs.parquet"]
    (output / "binding_audit.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    path = ROLES / "table_binding_audit.json"
    combined = json.loads(path.read_text(encoding="utf-8"))
    for name, digest in role["input_table_sha256"].items():
        combined[f"sha256_{name}"] = sha256(ROLES / name) == digest
    combined.update({f"streetview_pilot_{k}": bool(v) for k,v in checks.items()})
    if not all(combined.values()):
        raise ValueError("Combined role audit failed")
    path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"release_status": manifest["release_status"], "available_events": manifest["available_events"],
                      "local_images": manifest["selected_unique_images"], "pending_events": manifest["pending_verification_events"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--publish-verified-pilot", action="store_true")
    args = parser.parse_args()
    if args.publish_verified_pilot:
        return publish_verified_pilot(args.snapshot, args.contract, args.output)
    audit = json.loads((args.snapshot / "audit.json").read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if not audit["run_finished"] or audit["events"] != 5048 or audit["cities"] != 44:
        raise ValueError("A complete 44-city rebuild is required")
    if audit["treatment_sha256"] != sha256(TREATMENTS) or audit["contract_sha256"] != sha256(args.contract):
        raise ValueError("Frozen contract does not match the server rebuild")
    for filename, digest in audit["file_sha256"].items():
        if sha256(args.snapshot / filename) != digest:
            raise ValueError(f"Server snapshot checksum mismatch: {filename}")
    treatments = pd.read_parquet(TREATMENTS)
    dtype = {"panoid": str, "capture_month": str, "opening_month": str}
    index = pd.read_csv(args.snapshot / "streetview_index.csv", dtype=dtype)
    selected = pd.read_csv(args.snapshot / "streetview_pre_selected.csv", dtype=dtype)
    checks = assert_binding(index, selected, treatments, contract)
    args.output.mkdir(parents=True, exist_ok=True)
    selected = selected.rename(columns={"asset_path": "remote_asset_path"})
    image_root = (args.output / "images").resolve()
    selected["asset_path"] = [str(image_root / r.city_key / f"{r.panoid}.jpg") for r in selected.itertuples()]
    transport = {"local_image_root": str(image_root),
                 "remote_image_root": audit["raw_root"] + "/images",
                 "assets": selected.drop_duplicates(["city_key", "panoid"])[
                     ["city_key", "panoid", "remote_asset_path", "asset_sha256", "asset_bytes"]].to_dict("records")}
    (args.output / "download_manifest.json").write_text(json.dumps(transport, indent=2), encoding="utf-8")
    if args.stage_only:
        print(json.dumps({"stage_ready": True, "images": len(transport["assets"]),
                          "bytes": sum(r["asset_bytes"] for r in transport["assets"])}))
        return 0
    for record in selected.drop_duplicates("asset_path").itertuples():
        path = Path(record.asset_path)
        if not path.is_file() or path.stat().st_size != record.asset_bytes or sha256(path) != record.asset_sha256:
            raise ValueError(f"Missing or changed local training image: {path}")
        with Image.open(path) as image:
            image.load()
            if image.size != (record.width, record.height):
                raise ValueError(f"Image dimensions changed: {path}")
    checks["all_selected_local_images_decoded_and_checksums_match"] = True
    counts = selected.groupby("treatment_order").size()
    coverage = treatments[["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"]].copy()
    coverage["streetview_image_count"] = coverage.treatment_order.map(counts).fillna(0).astype(int)
    coverage["streetview_available"] = coverage.streetview_image_count.gt(0)
    available_ids = set(selected.treatment_order)
    candidates = treatments[["treatment_order", "city_key", "grid_id", "opening_month"]].merge(
        index[["city_key", "grid_id", "capture_month"]], on=["city_key", "grid_id"], how="inner")
    any_grid = set(candidates.treatment_order)
    any_pre = set(candidates.loc[candidates.capture_month < candidates.opening_month.str[:7], "treatment_order"])
    coverage["streetview_status"] = ["local_pretreatment_verified" if o in available_ids else
                                     "preopening_candidates_but_no_verified_image" if o in any_pre else
                                     "only_opening_or_postopening_captures" if o in any_grid else
                                     "no_pano_metadata_in_formal_grid" for o in coverage.treatment_order]
    grid_index = selected.drop_duplicates(["city_key", "grid_id", "panoid"]).drop(columns=[
        "treatment_order", "station_event_id", "opening_month"])
    for frame in [index, selected, grid_index]:
        frame["capture_date"] = pd.to_datetime(frame.capture_date)
    archive = index.rename(columns={"asset_path": "remote_asset_path"}).copy()
    archive["asset_bytes"] = archive.asset_bytes.astype("Int64")
    archive["image_verification_status"] = "file_presence_checked_content_unverified"
    if audit.get("index_image_availability") == "unverified_metadata_candidates":
        archive["asset_bytes"] = pd.NA
        archive["image_verification_status"] = "metadata_candidate_image_unverified"
    archive.to_parquet(args.output / "streetview_server_index.parquet", index=False, compression="zstd")
    selected.to_parquet(args.output / "streetview_state_assets.parquet", index=False, compression="zstd")
    grid_index.to_parquet(args.output / "streetview_index.parquet", index=False, compression="zstd")
    coverage.to_parquet(args.output / "event_coverage.parquet", index=False, compression="zstd")
    source_audit = args.output / "server_audit.json"
    source_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"schema": "streetview_fixed_treated_grid_v2", "published_at_utc": datetime.now(UTC).isoformat(),
                "treatment_sha256": sha256(TREATMENTS), "grid_contract_sha256": sha256(args.contract),
                "role": "state_only", "coordinate_source": audit["coordinate_source"],
                "spatial_support": audit["spatial_support"], "capture_date_precision": "month",
                "temporal_rule": audit["temporal_rule"], "remote_index_rows": len(index),
                "remote_index_image_availability": audit.get("index_image_availability", "file_presence_verified"),
                "selected_rows": len(selected), "selected_unique_images": len(grid_index),
                "events": len(coverage), "available_events": int(coverage.streetview_available.sum()),
                "missing_events": int((~coverage.streetview_available).sum()),
                "event_status_counts": coverage.streetview_status.value_counts().to_dict(),
                "remote_originals_preserved": True, "all_remote_images_downloaded": False,
                "selected_local_images_verified": True, "max_images_per_event": int(counts.max()),
                "training_preprocessing_fitted": False,
                "file_sha256": {name: sha256(args.output / name) for name in [
                    "streetview_server_index.parquet", "streetview_state_assets.parquet", "streetview_index.parquet",
                    "event_coverage.parquet", "server_audit.json"]}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    state = pd.read_parquet(ROLES / "state_inputs.parquet")
    before = state.drop(columns=[k for k in state if k.startswith("streetview_")]).copy()
    state = state.drop(columns=[k for k in state if k.startswith("streetview_")]).merge(
        coverage[["treatment_order", "streetview_available", "streetview_status", "streetview_image_count"]],
        on="treatment_order", how="left", validate="one_to_one")
    paths = selected.groupby("treatment_order").asset_path.agg(lambda s: json.dumps(s.tolist()))
    state["streetview_assets"] = state.treatment_order.map(paths).fillna("[]")
    checks["other_state_fields_unchanged"] = before.equals(state[before.columns])
    checks["all_events_preserved"] = state.treatment_order.tolist() == treatments.treatment_order.tolist()
    checks["missing_masks_agree_with_paths"] = bool(state.streetview_available.equals(state.streetview_assets.ne("[]")))
    if not all(checks.values()):
        raise ValueError("Final role binding failed")
    state.to_parquet(ROLES / "state_inputs.parquet", index=False, compression="zstd")
    role_path = ROLES / "manifest.json"
    role = json.loads(role_path.read_text(encoding="utf-8"))
    role.update(streetview_status="local_pretreatment_assets_verified",
                streetview_full_collection_completed=True, streetview_pending_verification_events=0,
                streetview_verified_cities=sorted(coverage.city_key.unique().tolist()),
                streetview_root=args.output.relative_to(ROOT).as_posix(),
                streetview_state_assets=(args.output / "streetview_state_assets.parquet").relative_to(ROOT).as_posix(),
                streetview_available_events=int(coverage.streetview_available.sum()),
                streetview_manifest_sha256=sha256(args.output / "manifest.json"),
                bundle_status="tabular_annual_satellite_and_streetview_assets_published",
                updated_at_utc=datetime.now(UTC).isoformat())
    role["availability_counts"]["state"]["streetview_available"] = int(state.streetview_available.sum())
    role["input_table_sha256"]["state_inputs.parquet"] = sha256(ROLES / "state_inputs.parquet")
    role_path.write_text(json.dumps(role, ensure_ascii=False, indent=2), encoding="utf-8")
    checks["state_table_hash_matches_role_manifest"] = sha256(ROLES / "state_inputs.parquet") == role["input_table_sha256"]["state_inputs.parquet"]
    (args.output / "binding_audit.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    prior_path = ROLES / "table_binding_audit.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    # Existing checks remain valid because all old state values and the other
    # role tables are preserved. Recheck every published table hash now.
    for filename, digest in role["input_table_sha256"].items():
        prior[f"sha256_{filename}"] = sha256(ROLES / filename) == digest
    prior.update({f"streetview_{name}": bool(value) for name, value in checks.items()})
    if not all(prior.values()):
        raise ValueError("Combined input binding audit failed")
    prior_path.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["remote_index_rows", "selected_rows", "available_events", "missing_events"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
