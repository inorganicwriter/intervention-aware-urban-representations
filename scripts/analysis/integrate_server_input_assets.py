"""Bind uploaded S/T/C tables to the verified server imagery without changing features."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shutil
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

IDENTITY = ["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"]
PATH_FIELDS = ["image_path", "preview_path", "mask_path", "source_manifest_path"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def rebase_asset(value, asset_root: str, project_root: str):
    if value is None or pd.isna(value) or value == "":
        return value
    text = str(value).replace("\\", "/")
    local = "D:/User/Desktop/MIT_summer/"
    if text.startswith(local):
        text = project_root + "/" + text[len(local):]
    elif not text.startswith("/"):
        if ":" in text:
            raise ValueError("Unrecognized asset drive or URI")
        text = asset_root + "/" + text
    result = PurePosixPath(posixpath.normpath(text))
    if not result.is_relative_to(PurePosixPath(project_root)):
        raise ValueError("Asset reference escapes the server project")
    return str(result)


def preserve_source_metadata(roles: Path, backup: Path, records: dict) -> dict[str, Path]:
    """Keep the transferred metadata intact across interrupted binding attempts."""
    backup.mkdir(exist_ok=True)
    paths = {}
    for name in ["state_inputs.parquet", "satellite_state_sequences.parquet", "manifest.json", "table_binding_audit.json"]:
        expected = records[f"data/active/multimodal_inputs_v1/{name}"]["sha256"]
        preserved = backup / name
        if not preserved.exists():
            source = roles / name
            if sha256(source) != expected:
                raise ValueError(f"Original metadata is unavailable for backup: {name}")
            temporary = backup / f".{name}.sourcecopying"
            shutil.copyfile(source, temporary)
            if sha256(temporary) != expected:
                raise ValueError(f"Source metadata backup checksum mismatch: {name}")
            os.replace(temporary, preserved)
        if sha256(preserved) != expected:
            raise ValueError(f"Preserved source metadata checksum mismatch: {name}")
        paths[name] = preserved
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    run = args.run_root.resolve()
    if project != Path("/home/nas/lsr/MIT_summer") or not run.is_relative_to(project / "outputs"):
        raise ValueError("Unexpected server deployment root")
    transfer = json.loads((run / "transfer_receipt.json").read_text())
    if not transfer["transfer_completed"]:
        raise ValueError("Verified file transfer is not complete")
    source_manifest = Path(transfer["source_manifest_path"])
    records = {record["path"]: record for record in json.loads(source_manifest.read_text())["files"]}
    shutil.copyfile(source_manifest, run / "source_transfer_manifest.json")
    roles = project / "data/active/multimodal_inputs_v1"
    backup = run / "original_role_metadata"
    sources = preserve_source_metadata(roles, backup, records)
    role = json.loads(sources["manifest.json"].read_text())
    source_audit = json.loads(sources["table_binding_audit.json"].read_text())
    checks = {"source_binding_audit_all_passed": all(source_audit.values()),
              "frozen_treatment_hash_matches_transfer": sha256(project / "data/active/causal/treatment_unit_list.parquet") == transfer["frozen_treatment_sha256"]}
    for name, digest in role["input_table_sha256"].items():
        checks[f"original_role_sha256_{name}"] = sha256(sources.get(name, roles / name)) == digest
    treatments = pd.read_parquet(project / "data/active/causal/treatment_unit_list.parquet")
    state = pd.read_parquet(sources["state_inputs.parquet"])
    context = pd.read_parquet(roles / "context_inputs.parquet")
    intervention = pd.read_parquet(roles / "intervention_inputs.parquet")
    sequence = pd.read_parquet(sources["satellite_state_sequences.parquet"])
    annual_root = project / "data/active/satellite_event_sequences_v2"
    annual = pd.read_parquet(annual_root / "satellite_sequence_index.parquet")
    baseline = pd.read_parquet(project / "data/active/satellite_grid_images_v1/satellite_image_index.parquet")
    pixel_audit = json.loads((Path(transfer["support_root"]) / "satellite_image_audit.json").read_text())
    checks["full_annual_index_matches_pixel_audit"] = sha256(annual_root / "satellite_sequence_index.parquet") == pixel_audit["image_index_sha256"]
    checks["original_pixel_audit_has_zero_errors"] = pixel_audit["error_count"] == 0 and pixel_audit["verified_images"] == 48470
    for label, frame in [("state", state), ("context", context), ("intervention", intervention)]:
        checks[f"{label}_frozen_identity"] = frame[IDENTITY].equals(treatments[IDENTITY])
    checks["annual_all_ten_event_years_preserved"] = len(annual) == 50480 and all(
        len(part) == 5048 and set(part.treatment_order) == set(treatments.treatment_order)
        for _, part in annual.groupby("event_time")) and set(annual.event_time) == set(range(-6, 4))
    checks["state_only_six_preopening_years"] = len(sequence) == 30288 and sequence.event_time.between(-6, -1).all()
    checks["annual_state_flags_exclude_opening_and_post"] = annual.state_temporal_allowed.equals(annual.event_time.lt(0))
    checks["annual_available_and_future_masks_preserved"] = int(annual.satellite_image_available.sum()) == 48465 and int((~annual.calendar_year_complete).sum()) == 2010
    checks["state_available_frame_count_preserved"] = int(sequence.state_image_available.sum()) == 30283
    hashes = context.set_index("treatment_order").alphaearth_grid_geometry_sha256
    checks["annual_geometry_matches_context"] = annual.grid_geometry_sha256.equals(annual.treatment_order.map(hashes))
    checks["annual_target_year_is_event_year"] = annual.target_year.eq(annual.opening_month.str[:4].astype(int) + annual.event_time).all()
    checks["context_alphaearth_legal_coverage"] = int(context.alphaearth_available.sum()) == 3297
    bands = [f"alphaearth_A{i:02d}" for i in range(64)]
    checks["available_alphaearth_all_64_values_finite"] = bool(np.isfinite(context.loc[context.alphaearth_available, bands].to_numpy()).all())
    checks["alphaearth_stays_out_of_state"] = not any(name.startswith("alphaearth_") for name in state)

    def verify_paths(frame, base):
        for row in frame.itertuples():
            for field in PATH_FIELDS:
                value = getattr(row, field)
                if value is None or pd.isna(value) or value == "":
                    continue
                absolute = rebase_asset(value, str(base), str(project))
                relative = PurePosixPath(absolute).relative_to(PurePosixPath(str(project))).as_posix()
                if relative not in records:
                    raise ValueError(f"Image or sidecar omitted from the transferred package: {relative}")
                if field == "image_path" and records[relative]["sha256"] != row.image_sha256:
                    raise ValueError("Transferred satellite content differs from the audited index")
        return True

    checks["all_annual_imagery_and_sidecars_transferred"] = verify_paths(annual, annual_root)
    checks["all_baseline_imagery_and_sidecars_transferred"] = verify_paths(baseline, project / "data/active/satellite_grid_images_v1")
    before_state = state.copy()
    baseline_base = project / role["satellite_image_root"]
    state_paths = ["satellite_image_path", "satellite_mask_path", "satellite_source_manifest_path"]
    for field in state_paths:
        state[field] = state[field].map(lambda value: rebase_asset(value, str(baseline_base), str(project)))
    before_sequence = sequence.copy()
    for field in PATH_FIELDS:
        sequence[field] = sequence[field].map(lambda value: rebase_asset(value, str(annual_root), str(project)))
    checks["sequence_numeric_values_masks_and_pixel_hashes_unchanged"] = before_sequence.drop(columns=PATH_FIELDS).equals(sequence.drop(columns=PATH_FIELDS))
    street_root = project / "data/active/streetview_grid_v2"
    street_manifest = json.loads((street_root / "manifest.json").read_text())
    street_verification = json.loads((street_root / "publication_verification.json").read_text())
    coverage = pd.read_parquet(street_root / "event_coverage.parquet")
    street = pd.read_parquet(street_root / "streetview_state_assets.parquet")
    checks["streetview_release_verified"] = street_verification["all_checks_passed"] and sha256(street_root / "manifest.json") == street_verification["primary_manifest_sha256"]
    checks["streetview_frozen_identity_matches"] = coverage[IDENTITY].equals(treatments[IDENTITY])
    checks["streetview_strictly_preopening"] = bool((street.capture_month < street.opening_month.str[:7]).all())
    street_columns = [name for name in state if name.startswith("streetview_")]
    state = state.drop(columns=street_columns).merge(coverage[["treatment_order", "streetview_available", "streetview_status",
        "streetview_image_count", "streetview_verification_complete", "streetview_assets"]], on="treatment_order", validate="one_to_one")
    fixed = [name for name in before_state if name not in [*street_columns, *state_paths]]
    checks["state_numeric_features_and_other_masks_unchanged"] = state[fixed].equals(before_state[fixed])
    checks["all_state_events_preserved"] = state[IDENTITY].equals(treatments[IDENTITY])
    checks["full_server_streetview_mask_matches_paths"] = state.streetview_available.equals(state.streetview_assets.ne("[]"))
    checks["full_server_streetview_count_matches_manifest"] = int(state.streetview_available.sum()) == street_manifest["available_events"] == 3778
    if not all(checks.values()):
        raise ValueError(f"Server input binding failed: {[name for name, passed in checks.items() if not passed]}")
    for name, frame in [("state_inputs.parquet", state), ("satellite_state_sequences.parquet", sequence)]:
        temporary = roles / f".{name}.serverpublishing"
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, roles / name)
    role.update(server_project_root=str(project), runtime_image_paths="absolute_server_paths",
                satellite_image_root=str(annual_root), satellite_paths_relative_to="absolute_server_paths",
                baseline_satellite_asset_root=str(project / "data/active/satellite_grid_images_v1"),
                satellite_sequence_asset_root=str(annual_root),
                satellite_state_sequence_sha256=sha256(roles / "satellite_state_sequences.parquet"),
                streetview_status="server_project_pretreatment_assets_verified", streetview_counts_scope="full_server_release",
                streetview_available_events=3778, streetview_verified_cities=sorted(treatments.city_key.unique()),
                streetview_pending_verification_events=0, streetview_full_collection_completed=True,
                streetview_manifest_sha256=sha256(street_root / "manifest.json"),
                streetview_post_archive_manifest=str(street_root / "post_manifest.json"),
                streetview_post_archive_predictor_allowed=False,
                bundle_status="server_tabular_annual_satellite_and_streetview_assets_published",
                other_latest_modalities_synchronized_to_server=True,
                model_migration_completed=False, updated_at_utc=datetime.now(UTC).isoformat())
    role["availability_counts"]["state"]["streetview_available"] = 3778
    role["input_table_sha256"] = {name: sha256(roles / name) for name in role["input_table_sha256"]}
    if "remote_streetview_release" in role:
        role["remote_streetview_release"]["latest_other_modalities_synchronized_to_server"] = True
    role["deployment_source_reference_copies"] = {"outputs/causal_labels/station_attributes/station_attributes.parquet":
                                                str(Path(transfer["support_root"]) / "station_attributes.parquet")}
    for name, digest in role["input_table_sha256"].items():
        checks[f"published_role_sha256_{name}"] = sha256(roles / name) == digest
    audit_body = {name: bool(passed) for name, passed in checks.items()}
    (roles / "table_binding_audit.json").write_text(json.dumps(audit_body, indent=2))
    temporary = roles / ".manifest.json.serverpublishing"
    temporary.write_text(json.dumps(role, ensure_ascii=False, indent=2))
    os.replace(temporary, roles / "manifest.json")
    summary = {"schema": "server_multimodal_input_sync_v1", "completed": True,
               "completed_at_utc": datetime.now(UTC).isoformat(), "project_root": str(project),
               "all_checks_passed": all(audit_body.values()), "check_count": len(audit_body),
               "source_binding_check_count": len(source_audit), "verified_transfer_files": transfer["verified_files"],
               "verified_transfer_bytes": transfer["verified_bytes"], "events": 5048, "cities": 44,
               "alphaearth_available": 3297, "alphaearth_historical_missing": 1751,
               "satellite_annual_rows": 50480, "satellite_verified_physical_frames": 48470,
               "satellite_available_frames": 48465, "satellite_low_coverage_frames": 5, "satellite_incomplete_year_frames": 2010,
               "satellite_state_rows": 30288, "satellite_state_available_frames": 30283,
               "streetview_pre_available_events": 3778, "streetview_pre_images": 7543,
               "streetview_post_available_event_years": 5489, "streetview_post_images": 10854,
               "intervention_historical_source_review_status": "requires_source_review",
               "role_manifest_sha256": sha256(roles / "manifest.json"),
               "input_table_sha256": role["input_table_sha256"], "frozen_treatment_sha256": transfer["frozen_treatment_sha256"],
               "model_migration_completed": False}
    (run / "integration_receipt.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
