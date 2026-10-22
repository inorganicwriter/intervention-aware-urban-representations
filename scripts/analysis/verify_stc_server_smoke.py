#!/usr/bin/env python3
"""Server real-asset smoke verification script for S/T/C multimodal representation model.

Runs an end-to-end verification pipeline:
1. Selects 64-128 events across diverse cities covering mixed modality availability
   (e.g. missing streetview, missing satellite years, missing alphaearth).
2. Builds an isolated STC training slice in a candidate directory.
3. Performs:
   - Reading and dataset indexing
   - Batch collation and mask validation
   - Forward pass (finite z, fused, predictions)
   - Backward pass (finite gradients across all branches)
   - Invariance verification (fixed S, changing T/C preserves exact z)
   - Mask perturbation verification (perturbed masked inputs produce 0.0 change)
   - Short training (2 epochs) with soft causal anchoring loss
   - Checkpoint saving with STC contract and metadata
   - Checkpoint restoration and numerical equivalence check
   - Representation export with pure state representation z and fused representation
4. Outputs comprehensive audit log and JSON report.

Usage:
    python scripts/analysis/verify_stc_server_smoke.py --outcome-family viirs --output-dir outputs/stc_server_smoke/viirs
    python scripts/analysis/verify_stc_server_smoke.py --outcome-family poi --output-dir outputs/stc_server_smoke/poi
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.causal.pretraining_dataset import (  # noqa: E402
    STC_CONTEXT_TABULAR_COLUMNS,
    STC_INTERVENTION_COLUMNS,
    STC_STATE_TABULAR_COLUMNS,
    publish_stc_dataset,
)
from urban_intervention.data.paths import (  # noqa: E402
    ACTIVE_DIR,
    PROJECT_ROOT as DATA_PROJECT_ROOT,
    TREATMENT_UNIT_LIST,
)
from urban_intervention.representation.dataset import (  # noqa: E402
    RESPONSE_CONFIG,
    RepresentationDataset,
    collate_samples,
)
from urban_intervention.representation.export import export_embeddings  # noqa: E402
from urban_intervention.representation.model import STCResponseModel  # noqa: E402
from urban_intervention.representation.training.runner import train_representation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stc_server_smoke")


def _resolve_stc_input_directory(project_root: Path) -> Path:
    """Find the unique active directory containing the published role tables."""
    active_root = Path(project_root) / "data" / "active"
    required = (
        "manifest.json",
        "state_inputs.parquet",
        "context_inputs.parquet",
        "intervention_inputs.parquet",
        "satellite_state_sequences.parquet",
    )
    matches = [
        child
        for child in active_root.iterdir()
        if child.is_dir() and all((child / name).is_file() for name in required)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected exactly one active S/T/C role package containing "
            f"{', '.join(required)}; found {len(matches)} under {active_root}. "
            "Pass --stc-inputs-dir explicitly to resolve the ambiguity."
        )
    return matches[0]


def _resolve_active_asset_file(project_root: Path, filename: str) -> Path | None:
    """Find one active asset index by its stable filename, rejecting ambiguity."""
    active_root = Path(project_root) / "data" / "active"
    if not active_root.is_dir():
        return None
    matches = sorted(
        child / filename
        for child in active_root.iterdir()
        if child.is_dir() and (child / filename).is_file()
    )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple active files named {filename}: {matches}; "
            "pass the intended path explicitly."
        )
    return matches[0] if matches else None


def check_response_artifact_provenance(response_release: Path | None) -> tuple[bool, str]:
    """Verify whether response_release meets strict server production acceptance criteria."""
    if response_release is None:
        return False, "No response release path provided"
    artifact_path = response_release / "response_artifact.parquet"
    if not artifact_path.is_file():
        return False, f"Response artifact table not found at {artifact_path}"
    manifest_path = response_release / "manifest.json"
    if not manifest_path.is_file():
        return False, f"Response release lacks manifest.json at {manifest_path}"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Failed to parse manifest: {exc}"
    if not bool(manifest.get("strict_production", False)):
        return False, "Response release manifest strict_production is not True"
    expected_sha = (
        manifest.get("artifact", {}).get("sha256")
        or manifest.get("outputs", {}).get("response_artifact.parquet")
    )
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        return False, "Response release manifest lacks a valid artifact SHA256"
    actual_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        return False, f"Artifact SHA mismatch: expected {expected_sha}, got {actual_sha}"
    try:
        df = pd.read_parquet(artifact_path)
        if "quality_grade" in df.columns:
            grades = df["quality_grade"].dropna().astype(str).unique().tolist()
            if any("synthetic" in g or "mock" in g for g in grades):
                return False, f"Artifact contains synthetic/mock quality grades: {grades}"
    except Exception as exc:
        return False, f"Failed to inspect quality grades: {exc}"
    return True, "Valid production response release"


def _round_robin_orders(frame: pd.DataFrame, limit: int) -> list[int]:
    """Choose event orders across cities without changing their published splits."""
    queues = {
        str(city): group.sort_values("treatment_order")["treatment_order"].astype(int).tolist()
        for city, group in frame.groupby("city_key", sort=True)
    }
    selected: list[int] = []
    while len(selected) < limit:
        progressed = False
        for city in sorted(queues):
            if queues[city] and len(selected) < limit:
                selected.append(queues[city].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def select_real_smoke_orders(
    state_df: pd.DataFrame,
    sample_index: pd.DataFrame,
    targets_df: pd.DataFrame,
    outcome_family: str,
    n_samples: int,
) -> list[int]:
    """Select labeled events across cities, including streetview present and missing cases."""
    if not 64 <= n_samples <= 128:
        raise ValueError(f"Strict server acceptance requires 64-128 events, got {n_samples}")
    required = {"treatment_order", "city_key", "streetview_available"}
    if not required.issubset(state_df.columns):
        raise ValueError(f"State inputs lack smoke-selection columns: {sorted(required - set(state_df.columns))}")
    active = targets_df["final_training_mask"].fillna(False).astype(bool)
    family_rows = targets_df["outcome_family"].astype(str).eq(outcome_family)
    valid_orders = set(targets_df.loc[active & family_rows, "treatment_order"].astype(int))
    eligible = state_df[state_df["treatment_order"].astype(int).isin(valid_orders)].copy()
    eligible["streetview_available"] = eligible["streetview_available"].fillna(False).astype(bool)
    with_streetview = eligible[eligible["streetview_available"]]
    without_streetview = eligible[~eligible["streetview_available"]]
    if with_streetview.empty or without_streetview.empty:
        raise ValueError(
            "Strict smoke selection needs valid supervised events with both declared streetview "
            f"availability and legal streetview missingness; available={len(with_streetview)}, "
            f"missing={len(without_streetview)}"
        )

    n_with = min(len(with_streetview), max(1, n_samples // 4))
    selected = _round_robin_orders(with_streetview, n_with)
    selected.extend(_round_robin_orders(without_streetview, n_samples - len(selected)))
    if len(selected) < n_samples:
        remaining = eligible[~eligible["treatment_order"].astype(int).isin(selected)]
        selected.extend(_round_robin_orders(remaining, n_samples - len(selected)))
    if len(selected) != n_samples:
        raise ValueError(
            f"Only {len(selected)} valid supervised events are available; expected {n_samples}"
        )
    if len({str(state_df.set_index("treatment_order").loc[o, "city_key"]) for o in selected}) < 3:
        raise ValueError("Strict smoke selection did not cover at least three cities")
    # Keep the published city assignment and confirm that all selected events exist in it.
    indexed_sample = sample_index.set_index("treatment_order")
    missing = [o for o in selected if o not in indexed_sample.index]
    if missing:
        raise ValueError(f"Selected response events are absent from sample_index: {missing[:10]}")
    not_final = [
        o for o in selected
        if not bool(indexed_sample.loc[o, "final_training_mask"])
    ]
    if not_final:
        raise ValueError(f"Selected response events lack the published final training mask: {not_final[:10]}")
    return sorted(selected)


def build_smoke_slice(
    stc_inputs_dir: Path,
    response_release: Path | None,
    candidate_dir: Path,
    outcome_family: str,
    n_samples: int = 96,
    synthetic_test: bool = False,
    strict_production: bool = False,
    treatments_path: Path | None = None,
    streetview_index_path: Path | None = None,
    project_root: Path | None = None,
    satellite_root: Path | None = None,
) -> Path:
    """Prepare a 64-128 unit slice with real features, ensuring cross-city coverage and missing modalities."""
    candidate_dir.mkdir(parents=True, exist_ok=True)
    data_root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(os.environ.get("MIT_PROJECT_ROOT", DATA_PROJECT_ROOT)).resolve()
    )

    is_real_prod, prod_reason = check_response_artifact_provenance(response_release)
    if strict_production and not is_real_prod:
        raise ValueError(
            f"Strict server production acceptance requested but response provenance check failed: {prod_reason}"
        )

    targets_path = response_release / "response_artifact.parquet" if response_release else None
    if targets_path is None or not targets_path.is_file():
        if not synthetic_test:
            raise ValueError(
                f"Real Response Artifact not found (provided: {targets_path}). "
                "Server production acceptance requires real Response Artifact with true W-design/Matching labels. "
                "To execute in local offline test mode, explicitly pass --synthetic-test."
            )
        logger.warning(
            "[SYNTHETIC TEST MODE] Synthesizing mock response targets for %s on %d units. "
            "This run does NOT constitute server production acceptance.",
            outcome_family, n_samples,
        )
        state_df = pd.read_parquet(stc_inputs_dir / "state_inputs.parquet")
        context_df = pd.read_parquet(stc_inputs_dir / "context_inputs.parquet")
        interv_df = pd.read_parquet(stc_inputs_dir / "intervention_inputs.parquet")
        seq_df = pd.read_parquet(stc_inputs_dir / "satellite_state_sequences.parquet")

        cities = state_df["city_key"].value_counts().index.tolist()[:6]
        selected_pool = state_df[state_df["city_key"].isin(cities)].copy()
        sv_avail = selected_pool[selected_pool["streetview_available"].fillna(False).astype(bool)]
        sv_missing = selected_pool[~selected_pool["streetview_available"].fillna(False).astype(bool)]
        n_sv = min(len(sv_avail), n_samples // 4)
        n_non_sv = n_samples - n_sv
        chosen_orders = sorted(pd.concat([
            sv_avail.head(n_sv),
            sv_missing.head(n_non_sv),
        ])["treatment_order"].unique())[:n_samples]

        sub_state = state_df[state_df["treatment_order"].isin(chosen_orders)].reset_index(drop=True)
        sub_context = context_df[context_df["treatment_order"].isin(chosen_orders)].reset_index(drop=True)
        sub_interv = interv_df[interv_df["treatment_order"].isin(chosen_orders)].reset_index(drop=True)
        sub_seq = seq_df[seq_df["treatment_order"].isin(chosen_orders)].reset_index(drop=True)

        horizons = [1, 3, 6, 12, 18, 24] if outcome_family == "viirs" else [1, 2, 3]
        outcome_name = "viirs_avg_asinh" if outcome_family == "viirs" else "poi_count_log"
        t_records = []
        for order in chosen_orders:
            for h in horizons:
                t_records.append({
                    "treatment_order": int(order),
                    "outcome_family": outcome_family,
                    "outcome": outcome_name,
                    "event_time": int(h),
                    "causal_response_label": float(np.random.normal(0.2, 0.5)),
                    "standard_error": float(np.random.uniform(0.05, 0.2)),
                    "response_weight": float(np.random.uniform(0.5, 1.5)),
                    "final_training_mask": True,
                    "quality_grade": "synthetic_mock_test",
                })
        sub_targets = pd.DataFrame(t_records)
        city_list = sorted(sub_state["city_key"].unique())
        split_map = {c: ("test" if i == len(city_list) - 1 else "validation" if i == len(city_list) - 2 else "train")
                     for i, c in enumerate(city_list)}
        sub_state["split"] = sub_state["city_key"].map(split_map)
        sub_context["split"] = sub_context["city_key"].map(split_map)
        sub_interv["split"] = sub_interv["city_key"].map(split_map)

        sample_index = pd.DataFrame({
            "treatment_order": sub_state["treatment_order"],
            "city_key": sub_state["city_key"],
            "split": sub_state["split"],
            "final_training_mask": True,
            "quality_grade": "synthetic_mock_test",
            "task_sampling_weight": 1.0,
        })
        train_mask = sub_state["split"].eq("train")
        normalization: dict[str, dict] = {"state": {}, "context": {}, "intervention": {}}
        for col in STC_STATE_TABULAR_COLUMNS:
            vals = pd.to_numeric(sub_state.loc[train_mask, col], errors="coerce").dropna().values
            m = float(vals.mean()) if len(vals) else 0.0
            s = float(vals.std(ddof=0)) if len(vals) else 1.0
            usable = bool(len(vals) > 0 and s > 1e-6)
            sub_state[f"z__{col}"] = (pd.to_numeric(sub_state[col], errors="coerce") - m) / (s if usable else 1.0)
            normalization["state"][col] = {"mean": m, "sd": s if usable else None, "usable": usable}
        for col in STC_CONTEXT_TABULAR_COLUMNS:
            vals = pd.to_numeric(sub_context.loc[train_mask, col], errors="coerce").dropna().values
            m = float(vals.mean()) if len(vals) else 0.0
            s = float(vals.std(ddof=0)) if len(vals) else 1.0
            usable = bool(len(vals) > 0 and s > 1e-6)
            sub_context[f"z__{col}"] = (pd.to_numeric(sub_context[col], errors="coerce") - m) / (s if usable else 1.0)
            normalization["context"][col] = {"mean": m, "sd": s if usable else None, "usable": usable}
        for col in ("opening_year", "n_lines_at_opening", "same_month_openings"):
            vals = pd.to_numeric(sub_interv.loc[train_mask, col], errors="coerce").dropna().values
            m = float(vals.mean()) if len(vals) else 0.0
            s = float(vals.std(ddof=0)) if len(vals) else 1.0
            usable = bool(len(vals) > 0 and s > 1e-6)
            sub_interv[f"z__{col}"] = (pd.to_numeric(sub_interv[col], errors="coerce") - m) / (s if usable else 1.0)
            normalization["intervention"][col] = {"mean": m, "sd": s if usable else None, "usable": usable}

        sub_state.to_parquet(candidate_dir / "state_inputs.parquet", index=False)
        sub_context.to_parquet(candidate_dir / "context_inputs.parquet", index=False)
        sub_interv.to_parquet(candidate_dir / "intervention_inputs.parquet", index=False)
        sub_seq.to_parquet(candidate_dir / "satellite_state_sequences.parquet", index=False)
        sub_targets.to_parquet(candidate_dir / "response_targets.parquet", index=False)
        sample_index.to_parquet(candidate_dir / "sample_index.parquet", index=False)
        (candidate_dir / "normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")

        # Preserve streetview state assets if present
        sv_table_path = None
        if streetview_index_path is not None and Path(streetview_index_path).is_file():
            sv_table_path = Path(streetview_index_path)
        elif (stc_inputs_dir / "streetview_state_assets.parquet").is_file():
            sv_table_path = stc_inputs_dir / "streetview_state_assets.parquet"
        else:
            sv_table_path = _resolve_active_asset_file(
                data_root, "streetview_state_assets.parquet"
            )
        if sv_table_path is not None and sv_table_path.is_file():
            raw_sv_df = pd.read_parquet(sv_table_path)
            if "treatment_order" in raw_sv_df.columns:
                sub_sv_df = raw_sv_df[raw_sv_df["treatment_order"].isin(chosen_orders)].reset_index(drop=True)
                sub_sv_df.to_parquet(candidate_dir / "streetview_state_assets.parquet", index=False)

        manifest = {
            "schema": "urban_intervention_stc_dataset",
            "dataset_id": f"smoke_stc_{outcome_family}_n{len(chosen_orders)}",
            "strict_production": False,
            "created_utc": datetime.now(UTC).isoformat(),
            "state_tabular_columns": list(STC_STATE_TABULAR_COLUMNS),
            "intervention_columns": list(STC_INTERVENTION_COLUMNS),
            "context_tabular_columns": list(STC_CONTEXT_TABULAR_COLUMNS),
            "satellite_relative_years": [-6, -5, -4, -3, -2, -1],
            "families": [outcome_family],
            "is_synthetic": True,
            "execution_mode": "SYNTHETIC_TEST",
        }
        outputs_map = {}
        for fname in [
            "state_inputs.parquet",
            "context_inputs.parquet",
            "intervention_inputs.parquet",
            "satellite_state_sequences.parquet",
            "response_targets.parquet",
            "sample_index.parquet",
            "normalization.json",
            "streetview_state_assets.parquet",
        ]:
            if (candidate_dir / fname).is_file():
                outputs_map[fname] = hashlib.sha256((candidate_dir / fname).read_bytes()).hexdigest()
        manifest["outputs"] = outputs_map
        if "streetview_state_assets.parquet" in outputs_map:
            manifest["streetview_assets_sha256"] = outputs_map["streetview_state_assets.parquet"]
        (candidate_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return candidate_dir

    # Real mode: publish through official publish_stc_dataset pipeline
    tmp_staging = candidate_dir.parent / f".tmp_pub_{candidate_dir.name}_{int(time.time() * 1000)}"
    tmp_staging.mkdir(parents=True, exist_ok=True)
    try:
        if treatments_path is not None and Path(treatments_path).is_file():
            treatments_cand = Path(treatments_path)
        else:
            treatments_cand = data_root / "data/active/causal/treatment_unit_list.parquet"
            if not treatments_cand.is_file() and TREATMENT_UNIT_LIST.is_file():
                treatments_cand = TREATMENT_UNIT_LIST
        pub_path = publish_stc_dataset(
            response_release=response_release,
            treatments_path=treatments_cand,
            stc_inputs_dir=stc_inputs_dir,
            project_root=data_root,
            output_root=tmp_staging,
            dataset_id="stc_published",
            scope_view="same_city",
            families=(outcome_family,),
            methods=("matched", "gsc"),
            strict_production=is_real_prod,
            streetview_assets_path=streetview_index_path,
        )

        pub_state = pd.read_parquet(pub_path / "state_inputs.parquet")
        pub_sample_index = pd.read_parquet(pub_path / "sample_index.parquet")
        pub_targets = pd.read_parquet(pub_path / "response_targets.parquet")
        chosen_orders = select_real_smoke_orders(
            pub_state,
            pub_sample_index,
            pub_targets,
            outcome_family,
            n_samples,
        )

        for fname in [
            "state_inputs.parquet",
            "context_inputs.parquet",
            "intervention_inputs.parquet",
            "satellite_state_sequences.parquet",
            "response_targets.parquet",
            "sample_index.parquet",
        ]:
            df = pd.read_parquet(pub_path / fname)
            df_sub = df[df["treatment_order"].astype(int).isin(chosen_orders)].reset_index(drop=True)
            df_sub.to_parquet(candidate_dir / fname, index=False)

        # Preserve exact normalization and only the selected official image records.
        if (pub_path / "normalization.json").is_file():
            shutil.copy(str(pub_path / "normalization.json"), str(candidate_dir / "normalization.json"))
        if (pub_path / "streetview_state_assets.parquet").is_file():
            raw_sv_df = pd.read_parquet(pub_path / "streetview_state_assets.parquet")
            if "treatment_order" in raw_sv_df.columns:
                sub_sv_df = raw_sv_df[raw_sv_df["treatment_order"].astype(int).isin(chosen_orders)].reset_index(drop=True)
                sub_sv_df.to_parquet(candidate_dir / "streetview_state_assets.parquet", index=False)
            else:
                raise ValueError("Streetview selection table lacks treatment_order identity binding")

        manifest = json.loads((pub_path / "manifest.json").read_text(encoding="utf-8"))
        manifest["dataset_id"] = f"smoke_stc_{outcome_family}_n{len(pd.read_parquet(candidate_dir / 'state_inputs.parquet'))}"
        manifest["is_synthetic"] = not is_real_prod
        manifest["strict_production"] = is_real_prod
        manifest["execution_mode"] = "SERVER_PRODUCTION_ACCEPTANCE" if is_real_prod else "FIXTURE_TEST_MODE"
        outputs_map = {}
        for fname in [
            "state_inputs.parquet",
            "context_inputs.parquet",
            "intervention_inputs.parquet",
            "satellite_state_sequences.parquet",
            "response_targets.parquet",
            "sample_index.parquet",
            "normalization.json",
            "streetview_state_assets.parquet",
        ]:
            if (candidate_dir / fname).is_file():
                outputs_map[fname] = hashlib.sha256((candidate_dir / fname).read_bytes()).hexdigest()
        manifest["outputs"] = outputs_map
        if "streetview_state_assets.parquet" in outputs_map:
            manifest["streetview_assets_sha256"] = outputs_map["streetview_state_assets.parquet"]
        (candidate_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Verify real supervision
        targets_df = pd.read_parquet(candidate_dir / "response_targets.parquet")
        active_cells = targets_df["final_training_mask"].fillna(False).astype(bool)
        if int(active_cells.sum()) == 0:
            raise ValueError(f"Smoke dataset contains 0 valid training supervision cells for {outcome_family}")
        dim_counts = targets_df[active_cells].groupby(["outcome", "event_time"]).size().to_dict()
        logger.info("Supervision support per dimension: %s", dim_counts)
    finally:
        shutil.rmtree(tmp_staging, ignore_errors=True)

    logger.info("Smoke dataset packaged into %s", candidate_dir)
    return candidate_dir



def run_smoke_verification(
    dataset_dir: Path,
    output_dir: Path,
    outcome_family: str,
    device: str = "cpu",
    is_synthetic: bool = False,
    streetview_index_path: Path | str | None = None,
    satellite_root: Path | str | None = None,
) -> dict[str, object]:
    """Execute complete smoke verification steps and return structured audit results."""
    audit_results: dict[str, object] = {
        "outcome_family": outcome_family,
        "device": device,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "tests": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    candidate_src = (ROOT / "src").resolve()
    imported_module_paths: dict[str, str] = {}
    for module_name in (
        "urban_intervention.causal.pretraining_dataset",
        "urban_intervention.data.paths",
        "urban_intervention.representation.dataset",
        "urban_intervention.representation.export",
        "urban_intervention.representation.model",
        "urban_intervention.representation.training.runner",
    ):
        module_path = Path(sys.modules[module_name].__file__).resolve()
        imported_module_paths[module_name] = str(module_path)
        if not module_path.is_relative_to(candidate_src):
            raise RuntimeError(
                f"Candidate import guard failed: {module_name} loaded from {module_path}, "
                f"expected below {candidate_src}"
            )

    # 1. Dataset loading and verification
    t0 = time.time()
    ds = RepresentationDataset(
        dataset_dir,
        load_images=True,
        load_satellite_images=True,
        verify_bundle=True,
        streetview_index_path=streetview_index_path,
        satellite_root=satellite_root,
    )
    load_time = time.time() - t0
    assert ds.is_stc, "Dataset failed to recognize STC manifest schema"
    if list(ds.satellite_relative_years) != [-6, -5, -4, -3, -2, -1]:
        raise ValueError(f"State satellite years must be fixed at -6..-1, got {ds.satellite_relative_years}")
    alphaearth_schemas = (
        [f"A{i:02d}" for i in range(64)],
        [f"alphaearth_A{i:02d}" for i in range(64)],
    )
    alphaearth_columns = next(
        (columns for columns in alphaearth_schemas if all(c in ds.context_inputs.columns for c in columns)),
        [],
    )
    if not alphaearth_columns or int(ds.manifest.get("alphaearth_dimension", 0)) != 64:
        raise ValueError(
            "Context input contract must carry all 64 AlphaEarth dimensions under either "
            "Axx or alphaearth_Axx names; "
            f"found={len(alphaearth_columns)}, manifest_dim={ds.manifest.get('alphaearth_dimension')}"
        )
    alphaearth_available_events = int(
        ds.context_inputs.get("alphaearth_available", pd.Series(False, index=ds.context_inputs.index))
        .fillna(False).astype(bool).sum()
    )
    if alphaearth_available_events == 0:
        raise ValueError("Selected strict response events contain no real AlphaEarth context observations")
    expected_response_dim = sum(len(horizons) for horizons in RESPONSE_CONFIG.get(outcome_family, {}).values())
    assert ds.response_dim() == expected_response_dim, (
        f"{outcome_family} response dimension is {ds.response_dim()}, expected {expected_response_dim}"
    )
    package_manifest_path = dataset_dir / "manifest.json"
    package_manifest_sha = hashlib.sha256(package_manifest_path.read_bytes()).hexdigest()
    release_path = Path(str(ds.manifest.get("response_release", "")))
    release_manifest_path = release_path / "manifest.json" if str(release_path) else None
    release_manifest_sha = (
        hashlib.sha256(release_manifest_path.read_bytes()).hexdigest()
        if release_manifest_path is not None and release_manifest_path.is_file()
        else None
    )
    role_root = os.environ.get("MIT_STC_INPUTS_DIR")
    role_manifest_path = Path(role_root) / "manifest.json" if role_root else None
    role_manifest_sha = (
        hashlib.sha256(role_manifest_path.read_bytes()).hexdigest()
        if role_manifest_path is not None and role_manifest_path.is_file()
        else None
    )
    identity_columns = [
        column for column in ("treatment_order", "city_key", "grid_id", "station_event_id", "opening_month")
        if column in ds.state_inputs.columns
    ]
    audit_results["provenance"] = {
        "host": os.environ.get("MIT_ACCEPTANCE_HOST"),
        "run_id": os.environ.get("MIT_ACCEPTANCE_RUN_ID"),
        "project_root": os.environ.get("MIT_PROJECT_ROOT"),
        "stc_role_inputs_dir": os.environ.get("MIT_STC_INPUTS_DIR"),
        "smoke_dataset_dir": str(dataset_dir),
        "streetview_index_path": str(streetview_index_path) if streetview_index_path else None,
        "satellite_root": str(satellite_root) if satellite_root else None,
        "execution_mode": ds.manifest.get("execution_mode"),
        "strict_production": bool(ds.manifest.get("strict_production", False)),
        "candidate_root": str(ROOT.resolve()),
        "candidate_code_tree_sha256": os.environ.get("MIT_CANDIDATE_TREE_SHA256"),
        "imported_module_paths": imported_module_paths,
        "smoke_script": str(Path(__file__).resolve()),
        "response_release": str(release_path),
        "response_release_manifest_sha256": release_manifest_sha,
        "response_artifact_sha256": ds.manifest.get("response_artifact_sha256"),
        "treatments_sha256": ds.manifest.get("treatments_sha256"),
        "response_bound_treatments_sha256": ds.manifest.get("response_bound_treatments_sha256"),
        "role_input_manifest_sha256": role_manifest_sha,
        "source_features": ds.manifest.get("source_features"),
        "dataset_manifest_sha256": package_manifest_sha,
        "dataset_output_sha256": dict(ds.manifest.get("outputs", {})),
        "alphaearth_dimension": 64,
        "alphaearth_column_schema": (
            "alphaearth_Axx" if alphaearth_columns[0].startswith("alphaearth_") else "Axx"
        ),
        "alphaearth_available_events": alphaearth_available_events,
        "selected_event_identities": json.loads(
            ds.state_inputs[identity_columns].to_json(orient="records", date_format="iso")
        ),
        "selected_city_splits": json.loads(
            ds.sample_index[["treatment_order", "city_key", "split"]].to_json(orient="records", date_format="iso")
        ),
    }
    audit_results["tests"]["dataset_load"] = {
        "status": "PASS",
        "units": len(ds),
        "response_dim": ds.response_dim(),
        "elapsed_sec": round(load_time, 4),
    }
    logger.info("[PASS] Dataset loaded: %d units, response_dim=%d", len(ds), ds.response_dim())

    # 1b. Read every selected event once and distinguish declared missingness from failed reads.
    loaded_samples = [ds[i] for i in range(len(ds))]
    sat_seq = ds.satellite_sequences.copy()
    sat_seq["treatment_order"] = sat_seq["treatment_order"].astype(int)
    sat_seq["event_time"] = sat_seq["event_time"].astype(int)
    sat_lookup = {
        (int(row.treatment_order), int(row.event_time)): row
        for row in sat_seq.itertuples(index=False)
    }
    sat_declared = 0
    sat_loaded = 0
    sat_legal_missing = 0
    sat_failures: list[dict[str, object]] = []
    sat_temporal_failures: list[dict[str, object]] = []
    sv_declared_events = 0
    sv_legal_missing_events = 0
    sv_declared_images = 0
    sv_loaded_images = 0
    sv_failures: list[dict[str, object]] = []
    sv_source_inconsistencies: list[dict[str, object]] = []
    events_with_streetview = 0
    events_with_satellite = 0
    city_split_counts: dict[str, dict[str, int]] = {}
    for order, sample in zip(ds._treatment_orders, loaded_samples, strict=True):
        order = int(order)
        meta = ds._stc_state_info.get(order, {})
        city = str(meta.get("city_key", ""))
        split = str(sample.split)
        city_split_counts.setdefault(city, {}).setdefault(split, 0)
        if bool(sample.final_training_mask):
            city_split_counts[city][split] += 1

        sv_paths = ds._stc_streetview_assets.get(order, [])
        sv_declared = bool(ds._stc_streetview_avail.get(order, False))
        sv_count = int(sample.streetview_mask.sum().item()) if sample.streetview_mask is not None else 0
        if sv_declared:
            sv_declared_events += 1
            expected_images = min(len(sv_paths), 2)
            sv_declared_images += expected_images
            sv_loaded_images += sv_count
            if sv_count:
                events_with_streetview += 1
            if expected_images == 0 or sv_count < expected_images:
                sv_failures.append({
                    "treatment_order": order,
                    "city_key": city,
                    "declared_images": expected_images,
                    "loaded_images": sv_count,
                    "paths": [str(path) for path in sv_paths],
                })
        else:
            sv_legal_missing_events += 1
            if sv_paths:
                sv_source_inconsistencies.append({
                    "treatment_order": order,
                    "city_key": city,
                    "reason": "assets_present_but_streetview_marked_unavailable",
                    "paths": [str(path) for path in sv_paths],
                })

        for year_index, rel_year in enumerate(ds.satellite_relative_years):
            if int(rel_year) >= 0:
                sat_temporal_failures.append({
                    "treatment_order": order,
                    "relative_year": int(rel_year),
                    "reason": "non_pretreatment_year_in_state_contract",
                })
                continue
            row = sat_lookup.get((order, int(rel_year)))
            if row is None:
                sat_legal_missing += 1
                continue
            temporal_allowed = bool(getattr(row, "state_temporal_allowed", False))
            is_available = bool(getattr(row, "state_image_available", False))
            loaded = bool(sample.satellite_year_mask[year_index].item())
            if not temporal_allowed:
                sat_temporal_failures.append({
                    "treatment_order": order,
                    "relative_year": int(rel_year),
                    "target_year": (
                        int(getattr(row, "target_year"))
                        if pd.notna(getattr(row, "target_year", None)) else None
                    ),
                    "reason": "pre_opening_year_not_state_allowed",
                })
            elif is_available:
                sat_declared += 1
                if loaded:
                    sat_loaded += 1
                else:
                    sat_failures.append({
                        "treatment_order": order,
                        "city_key": city,
                        "relative_year": int(rel_year),
                        "path": str(getattr(row, "image_path", "")),
                    })
            else:
                sat_legal_missing += 1
        if bool(sample.satellite_year_mask.any().item()):
            events_with_satellite += 1

    modality_results = {
        "status": "PASS",
        "streetview": {
            "declared_available_events": sv_declared_events,
            "declared_images": sv_declared_images,
            "images_read_successfully": sv_loaded_images,
            "legal_missing_events": sv_legal_missing_events,
            "failed_images": sum(
                max(0, int(item["declared_images"]) - int(item["loaded_images"]))
                for item in sv_failures
            ),
            "failed_events": len(sv_failures),
            "events_with_loaded_images": events_with_streetview,
            "read_failures": sv_failures,
            "source_inconsistencies": sv_source_inconsistencies,
        },
        "satellite": {
            "relative_years": list(ds.satellite_relative_years),
            "declared_available_frames": sat_declared,
            "frames_read_successfully": sat_loaded,
            "legal_missing_frames": sat_legal_missing,
            "failed_frames": len(sat_failures),
            "events_with_loaded_frames": events_with_satellite,
            "read_failures": sat_failures,
            "temporal_failures": sat_temporal_failures,
            "valid_pixels": int(sum(int(sample.satellite_pixel_mask.sum().item()) for sample in loaded_samples)),
        },
    }
    if (
        sv_loaded_images == 0
        or sat_loaded == 0
        or sv_legal_missing_events == 0 and sat_legal_missing == 0
        or sv_failures
        or sv_source_inconsistencies
        or sat_failures
        or sat_temporal_failures
    ):
        modality_results["status"] = "FAIL"
        audit_results["tests"]["modality_ingestion"] = modality_results
        audit_results["summary"] = "FAIL_MODALITY_INGESTION"
        report_file = output_dir / "smoke_verification_report.json"
        report_file.write_text(json.dumps(audit_results, indent=2), encoding="utf-8")
        raise RuntimeError(
            "Real modality audit failed; read-failure, legal-missing and temporal counts "
            f"were saved to {report_file}"
        )
    audit_results["tests"]["modality_ingestion"] = modality_results

    # Keep the historical top-level key for existing audit readers.
    sat_seq = ds.satellite_sequences
    audit_results["tests"]["satellite_ingestion"] = {
        "status": "PASS",
        "declared_available_images": sat_declared,
        "actually_loaded_images": sat_loaded,
    }
    logger.info(
        "[PASS] Satellite ingestion verified: %d declared available images, %d actually loaded",
        sat_declared,
        sat_loaded,
    )

    # 1c. Check real response supervision across all configured dimensions
    targets_df = pd.read_parquet(dataset_dir / "response_targets.parquet")
    sample_idx = pd.read_parquet(dataset_dir / "sample_index.parquet")
    if bool(ds.manifest.get("strict_production", False)):
        if outcome_family not in (ds.manifest.get("families") or []):
            raise ValueError(f"Strict response release does not declare family {outcome_family}")
        if "quality_grade" not in targets_df.columns:
            raise ValueError("Strict response targets lack quality_grade")
        grades = targets_df["quality_grade"].dropna().astype(str).str.lower()
        if grades.str.contains("synthetic|mock", regex=True).any():
            raise ValueError("Strict response targets contain synthetic/mock quality grades")
    valid_mask = targets_df["final_training_mask"].fillna(False).astype(bool)
    if "outcome_family" in targets_df.columns:
        valid_mask &= targets_df["outcome_family"].astype(str).eq(outcome_family)
    total_valid = int(valid_mask.sum())
    assert total_valid > 0, f"0 valid training supervision cells found in {dataset_dir}"
    if "response_weight" not in targets_df.columns:
        raise ValueError("Strict smoke targets lack per-label response_weight")
    active_weights = pd.to_numeric(targets_df.loc[valid_mask, "response_weight"], errors="coerce")
    if active_weights.isna().any() or not active_weights.gt(0).all():
        raise ValueError("Eligible response rows contain missing or non-positive design/Matching weights")
    dim_counts = targets_df[valid_mask].groupby(["outcome", "event_time"]).size().to_dict()

    per_dim_serializable: dict[str, int] = {}
    for out_name, horizons in RESPONSE_CONFIG.get(outcome_family, {}).items():
        for h in horizons:
            per_dim_serializable[f"{out_name}_t{h}"] = int(dim_counts.get((out_name, h), 0))
    if len(per_dim_serializable) != expected_response_dim or any(value == 0 for value in per_dim_serializable.values()):
        raise ValueError(f"Missing valid supervision in one or more {outcome_family} dimensions: {per_dim_serializable}")

    sample_meta = sample_idx[["treatment_order", "city_key", "split"]].rename(
        columns={"city_key": "_sample_city_key", "split": "_sample_split"}
    )
    merged = targets_df[valid_mask].merge(
        sample_meta, on="treatment_order", how="left", validate="many_to_one"
    )
    if merged[["_sample_city_key", "_sample_split"]].isna().any().any():
        raise ValueError("Eligible response rows are not fully bound to the official sample split")
    if "city_key" in merged and not merged["city_key"].astype(str).eq(merged["_sample_city_key"].astype(str)).all():
        raise ValueError("Response target city_key disagrees with the official sample index")
    if "split" in merged and not merged["split"].astype(str).eq(merged["_sample_split"].astype(str)).all():
        raise ValueError("Response target split disagrees with the official sample index")
    merged["city_key"] = merged["_sample_city_key"]
    merged["split"] = merged["_sample_split"]
    merged = merged.drop(columns=["_sample_city_key", "_sample_split"])
    split_dim_counts = merged.groupby(["split", "outcome", "event_time"]).size().to_dict()
    dimension_event_counts = (
        merged.groupby(["outcome", "event_time"])["treatment_order"].nunique().to_dict()
    )
    split_event_counts = {
        str(split): int(count)
        for split, count in merged.groupby("split")["treatment_order"].nunique()
        .reindex(["train", "validation", "test"], fill_value=0)
        .items()
    }
    city_split_event_counts = (
        merged.groupby(["city_key", "split"])["treatment_order"].nunique().astype(int).to_dict()
    )
    per_split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        per_split_counts[split] = {}
        for out_name, horizons in RESPONSE_CONFIG.get(outcome_family, {}).items():
            for h in horizons:
                per_split_counts[split][f"{out_name}_t{h}"] = int(
                    split_dim_counts.get((split, out_name, h), 0)
                )

    audit_results["tests"]["response_supervision"] = {
        "status": "PASS",
        "total_valid_cells": total_valid,
        "per_dimension_counts": per_dim_serializable,
        "per_dimension_effective_events": {
            f"{outcome}_t{horizon}": int(dimension_event_counts.get((outcome, horizon), 0))
            for outcome, horizons in RESPONSE_CONFIG.get(outcome_family, {}).items()
            for horizon in horizons
        },
        "per_split_counts": per_split_counts,
        "valid_events_by_split": split_event_counts,
        "valid_events_by_city_and_split": {
            f"{city}|{split}": int(count) for (city, split), count in city_split_event_counts.items()
        },
        "active_response_weight": {
            "count": int(active_weights.count()),
            "min": float(active_weights.min()),
            "mean": float(active_weights.mean()),
            "max": float(active_weights.max()),
        },
        "valid_labels_by_method": (
            {str(method): int(count) for method, count in merged.groupby("method")["treatment_order"].nunique().items()}
            if "method" in merged.columns else {}
        ),
        "selected_city_supervision_events": city_split_counts,
    }
    logger.info(
        "[PASS] Response supervision verified: %d valid cells across dimensions: %s",
        total_valid,
        per_dim_serializable,
    )

    # 2. Collate samples selected by actual reads, not availability metadata alone.
    sv_loaded_idx = [i for i, sample in enumerate(loaded_samples) if sample.streetview_mask is not None and bool(sample.streetview_mask.any().item())]
    sv_missing_idx = [i for i, sample in enumerate(loaded_samples) if not sample.streetview_mask.any().item()]
    sat_loaded_idx = [i for i, sample in enumerate(loaded_samples) if bool(sample.satellite_year_mask.any().item())]
    sat_missing_idx = [i for i, sample in enumerate(loaded_samples) if not sample.satellite_year_mask.any().item()]
    alphaearth_loaded_idx = [i for i, sample in enumerate(loaded_samples) if bool(sample.alphaearth_available)]
    chosen_idx: list[int] = []
    for candidates in (sv_loaded_idx, sv_missing_idx, sat_loaded_idx, sat_missing_idx, alphaearth_loaded_idx):
        if candidates and candidates[0] not in chosen_idx:
            chosen_idx.append(candidates[0])
    for idx in range(len(loaded_samples)):
        if len(chosen_idx) >= 8:
            break
        if idx not in chosen_idx:
            chosen_idx.append(idx)
    batch_raw = [loaded_samples[i] for i in chosen_idx]
    batch = collate_samples(batch_raw, use_images=True)
    assert batch.get("is_stc") is True, "Batch collation missing is_stc flag"
    assert bool(batch["streetview_mask"].any().item()), "Verification batch contains no successfully read streetview image"
    assert bool(batch["satellite_year_mask"].any().item()), "Verification batch contains no successfully read satellite frame"
    assert bool(batch["alphaearth_available"].any().item()), "Verification batch contains no real AlphaEarth observation"
    audit_results["tests"]["batch_collate"] = {
        "status": "PASS",
        "batch_size": len(batch_raw),
        "state_tabular_shape": list(batch["state_tabular"].shape),
        "satellite_images_shape": list(batch["satellite_images"].shape),
    }
    logger.info("[PASS] Collation verified: state shape %s, satellite shape %s",
                batch["state_tabular"].shape, batch["satellite_images"].shape)

    # 3. Model forward and backward pass
    model = STCResponseModel(
        state_tabular_dim=len(ds.state_tabular_columns),
        intervention_dim=len(ds.intervention_columns),
        context_tabular_dim=len(ds.context_tabular_columns),
        alphaearth_dim=64,
        z_dim=64,
        t_dim=32,
        b_dim=64,
        predictor_hidden_dim=128,
        outcome_dim=ds.response_dim(),
        outcome_family=outcome_family,
        use_streetview=True,
        use_satellite=True,
    ).to(dev)

    device_batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
    z, preds, fused = model.forward_batch(device_batch)
    assert torch.isfinite(z).all(), "Non-finite values in state representation z"
    assert torch.isfinite(fused).all(), "Non-finite values in fused representation"
    assert torch.isfinite(preds[outcome_family]).all(), "Non-finite values in predictions"

    # Backward pass
    loss = preds[outcome_family].sum()
    loss.backward()
    has_sv_in_batch = batch.get("streetview_mask") is not None and bool(batch["streetview_mask"].any().item())
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
            if "streetview_encoder" in name and not has_sv_in_batch:
                continue
            if "missing" in name:
                continue
            assert p.grad.abs().sum().item() > 0, f"Zero gradient detected in active parameter {name}"

    audit_results["tests"]["forward_backward"] = {
        "status": "PASS",
        "z_shape": list(z.shape),
        "fused_shape": list(fused.shape),
        "gradients_finite": True,
        "active_gradients_nonzero": True,
    }
    logger.info("[PASS] Forward/backward verified: z shape %s, active branch gradients all non-zero and finite", z.shape)

    # 4. Invariance verification: fix S, perturb T and C -> z must remain unchanged
    model.eval()
    with torch.no_grad():
        eval_z, eval_preds, eval_fused = model.forward_batch(device_batch)
        z_base = eval_z

        # Altered batch: only T, C, AlphaEarth altered; S remains strictly identical
        altered_batch = dict(device_batch)
        altered_batch["intervention"] = altered_batch["intervention"] + 50.0
        altered_batch["context"] = altered_batch["context"] - 30.0
        altered_batch["alphaearth"] = altered_batch["alphaearth"] * 2.0
        z_altered, pred_altered, fused_altered = model.forward_batch(altered_batch)

        diff_z = (z_base - z_altered).abs().max().item()
        assert diff_z < 1e-6, f"State representation z changed when T/C altered! diff: {diff_z}"
        assert not torch.allclose(eval_fused, fused_altered, atol=1e-3), (
            "Fused representation failed to reflect T/C changes"
        )
    audit_results["tests"]["stc_invariance"] = {
        "status": "PASS",
        "max_z_diff": diff_z,
    }
    logger.info("[PASS] S/T/C separation verified: max z diff = %.2e", diff_z)

    with torch.no_grad():
        alphaearth_rows = device_batch["alphaearth_available"].bool()
        no_alphaearth = dict(device_batch)
        no_alphaearth["alphaearth"] = torch.zeros_like(device_batch["alphaearth"])
        _, _, fused_without_alphaearth = model.forward_batch(no_alphaearth)
        alphaearth_consumption_diff = float(
            (eval_fused[alphaearth_rows] - fused_without_alphaearth[alphaearth_rows]).abs().max().item()
        )
    assert alphaearth_consumption_diff > 1e-8, "Context encoder did not consume real 64-dimensional AlphaEarth values"
    audit_results["tests"]["alphaearth_consumption"] = {
        "status": "PASS",
        "dimension": 64,
        "events_in_batch": int(alphaearth_rows.sum().item()),
        "max_fused_change_when_zeroed": alphaearth_consumption_diff,
    }

    # 4b. Confirm that successfully read real visual tensors affect the state representation.
    with torch.no_grad():
        streetview_rows = device_batch["streetview_mask"].any(dim=1)
        satellite_rows = device_batch["satellite_year_mask"].any(dim=1)
        no_streetview = dict(device_batch)
        no_streetview["streetview_images"] = torch.zeros_like(device_batch["streetview_images"])
        z_without_streetview, _, _ = model.forward_batch(no_streetview)
        no_satellite = dict(device_batch)
        no_satellite["satellite_images"] = torch.zeros_like(device_batch["satellite_images"])
        z_without_satellite, _, _ = model.forward_batch(no_satellite)
        streetview_consumption_diff = float(
            (eval_z[streetview_rows] - z_without_streetview[streetview_rows]).abs().max().item()
        )
        satellite_consumption_diff = float(
            (eval_z[satellite_rows] - z_without_satellite[satellite_rows]).abs().max().item()
        )
    assert streetview_consumption_diff > 1e-8, "State encoder did not consume successfully read streetview images"
    assert satellite_consumption_diff > 1e-8, "State encoder did not consume successfully read satellite frames"
    audit_results["tests"]["visual_feature_consumption"] = {
        "status": "PASS",
        "streetview_events_in_batch": int(streetview_rows.sum().item()),
        "satellite_events_in_batch": int(satellite_rows.sum().item()),
        "max_z_change_when_streetview_zeroed": streetview_consumption_diff,
        "max_z_change_when_satellite_zeroed": satellite_consumption_diff,
    }

    # 4c. Change only masked inputs and require exact-equivalent model outputs.
    masked_batch = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in device_batch.items()
    }
    masked_counts: dict[str, int] = {}
    for value_key, mask_key in (
        ("state_tabular", "state_field_mask"),
        ("intervention", "intervention_mask"),
        ("context", "context_field_mask"),
    ):
        value = masked_batch[value_key]
        valid = masked_batch[mask_key]
        masked_counts[value_key] = int((~valid).sum().item())
        masked_batch[value_key] = torch.where(valid, value, value + 37.0)
    alphaearth_valid = masked_batch["alphaearth_available"].bool().unsqueeze(-1)
    masked_counts["alphaearth"] = int((~alphaearth_valid).sum().item())
    masked_batch["alphaearth"] = torch.where(
        alphaearth_valid,
        masked_batch["alphaearth"],
        masked_batch["alphaearth"] + 37.0,
    )
    streetview_valid = masked_batch["streetview_mask"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    masked_counts["streetview_images"] = int((~masked_batch["streetview_mask"]).sum().item())
    masked_batch["streetview_images"] = torch.where(
        streetview_valid,
        masked_batch["streetview_images"],
        masked_batch["streetview_images"] + 37.0,
    )
    if masked_batch.get("streetview_features") is not None:
        streetview_feature_valid = masked_batch["streetview_mask"].unsqueeze(-1)
        masked_counts["streetview_features"] = int((~masked_batch["streetview_mask"]).sum().item())
        masked_batch["streetview_features"] = torch.where(
            streetview_feature_valid,
            masked_batch["streetview_features"],
            masked_batch["streetview_features"] + 37.0,
        )
    satellite_valid = (
        masked_batch["satellite_pixel_mask"]
        & masked_batch["satellite_year_mask"].view(-1, len(ds.satellite_relative_years), 1, 1, 1)
    ).expand_as(masked_batch["satellite_images"])
    masked_counts["satellite_images"] = int((~satellite_valid).sum().item())
    masked_batch["satellite_images"] = torch.where(
        satellite_valid,
        masked_batch["satellite_images"],
        masked_batch["satellite_images"] + 37.0,
    )
    with torch.no_grad():
        z_masked, pred_masked, fused_masked = model.forward_batch(masked_batch)
    max_masked_z_diff = float((eval_z - z_masked).abs().max().item())
    max_masked_fused_diff = float((eval_fused - fused_masked).abs().max().item())
    if isinstance(eval_preds, dict):
        max_masked_pred_diff = max(
            float((eval_preds[key] - pred_masked[key]).abs().max().item())
            for key in eval_preds
        )
    else:
        max_masked_pred_diff = float((eval_preds - pred_masked).abs().max().item())
    assert max(max_masked_z_diff, max_masked_fused_diff, max_masked_pred_diff) < 1e-5, (
        "Perturbing inputs hidden by masks changed model outputs: "
        f"z={max_masked_z_diff}, fused={max_masked_fused_diff}, prediction={max_masked_pred_diff}"
    )
    audit_results["tests"]["masked_input_invariance"] = {
        "status": "PASS",
        "perturbed_masked_values": masked_counts,
        "max_z_diff": max_masked_z_diff,
        "max_fused_diff": max_masked_fused_diff,
        "max_prediction_diff": max_masked_pred_diff,
    }

    # 5. Short training loop (2 epochs)
    train_dir = output_dir / "training"
    ckpt_path = train_representation(
        model_inputs_dir=dataset_dir,
        output_dir=train_dir,
        embedding_dim=64,
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        run_baselines=False,
        run_transfer=False,
        seed=101,
        device=device,
        use_images=True,
        use_satellite=True,
        verify_bundle=True,
        streetview_index_path=streetview_index_path,
        satellite_root=satellite_root,
    )
    assert ckpt_path.is_file(), f"Checkpoint not produced at {ckpt_path}"
    audit_results["tests"]["short_training"] = {
        "status": "PASS",
        "checkpoint_path": str(ckpt_path),
        "device": device,
        "epochs": 2,
        "batch_size": 8,
        "used_streetview_images": True,
        "used_satellite_sequences": True,
        "training_history_path": str(train_dir / "training_history.json"),
        "training_config_path": str(train_dir / "training_config.json"),
    }
    logger.info("[PASS] Short training verified: checkpoint saved at %s", ckpt_path)

    # 6. Checkpoint restoration and numerical equivalence check
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
    assert ckpt.get("is_stc") is True, "Checkpoint missing is_stc flag"
    assert ckpt.get("model_type") == "stc", "Checkpoint missing model_type 'stc'"
    assert "stc_contract" in ckpt, "Checkpoint missing stc_contract"
    contract = ckpt["stc_contract"]
    expected_contract = {
        "state_tabular_columns": list(ds.state_tabular_columns),
        "intervention_columns": list(ds.intervention_columns),
        "context_tabular_columns": list(ds.context_tabular_columns),
        "satellite_relative_years": list(ds.satellite_relative_years),
        "families": list(ds.families) if ds.families else None,
        "response_dim": ds.response_dim(),
        "outcome_family": outcome_family,
        "normalization": ds.normalization,
        "bundle_shas": dict(ds.manifest.get("outputs", {})),
    }
    for key, expected_value in expected_contract.items():
        if contract.get(key) != expected_value:
            raise ValueError(f"Checkpoint S/T/C contract mismatch for {key}")
    if ckpt.get("bundle_shas") != expected_contract["bundle_shas"]:
        raise ValueError("Checkpoint top-level bundle hashes disagree with the input package")
    if ckpt.get("response_artifact_sha256") != ds.manifest.get("response_artifact_sha256"):
        raise ValueError("Checkpoint response-artifact fingerprint disagrees with the input package")
    expected_norm_sha = hashlib.sha256(
        json.dumps(ds.normalization, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    if ckpt.get("normalization_sha256") != expected_norm_sha:
        raise ValueError("Checkpoint normalization fingerprint disagrees with the input package")

    # Load the saved state into two independent model instances and compare z and predictions.
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        z_saved, pred_saved, _ = model.forward_batch(device_batch)

    model_restored = STCResponseModel(
        state_tabular_dim=len(ds.state_tabular_columns),
        intervention_dim=len(ds.intervention_columns),
        context_tabular_dim=len(ds.context_tabular_columns),
        alphaearth_dim=64,
        z_dim=64,
        t_dim=32,
        b_dim=64,
        predictor_hidden_dim=128,
        outcome_dim=ds.response_dim(),
        outcome_family=outcome_family,
        use_streetview=bool(ckpt.get("use_image_encoder", True)),
        use_satellite=bool(ckpt.get("use_satellite_encoder", True)),
    ).to(dev)
    model_restored.load_state_dict(ckpt["model_state_dict"])
    model_restored.eval()

    with torch.no_grad():
        z_restored, pred_restored, fused_restored = model_restored.forward_batch(device_batch)
    max_restore_z_diff = float((z_saved - z_restored).abs().max().item())
    if isinstance(pred_saved, dict):
        max_restore_prediction_diff = max(
            float((pred_saved[key] - pred_restored[key]).abs().max().item())
            for key in pred_saved
        )
    else:
        max_restore_prediction_diff = float((pred_saved - pred_restored).abs().max().item())
    assert max_restore_z_diff <= 1e-6, f"Checkpoint-restored z differs from saved model state: {max_restore_z_diff}"
    assert max_restore_prediction_diff <= 1e-6, (
        "Checkpoint-restored predictions differ from saved model state: "
        f"{max_restore_prediction_diff}"
    )
    assert bool(torch.isfinite(z_restored).all().item()), "Restored z contains non-finite values"
    if isinstance(pred_restored, dict):
        assert all(bool(torch.isfinite(v).all().item()) for v in pred_restored.values()), "Restored pred contains non-finite values"
    else:
        assert bool(torch.isfinite(pred_restored).all().item()), "Restored pred contains non-finite values"

    # 7. Embedding export
    export_path = output_dir / "embeddings.parquet"
    res_path = export_embeddings(
        checkpoint_path=ckpt_path,
        model_inputs_dir=dataset_dir,
        output_path=export_path,
        batch_size=8,
        device=device,
        streetview_index_path=streetview_index_path,
        satellite_root=satellite_root,
    )
    assert res_path.is_file(), "Exported embeddings file not found"
    exp_df = pd.read_parquet(export_path)
    assert len(exp_df) == len(ds), f"Export row count ({len(exp_df)}) != dataset ({len(ds)})"
    if exp_df["treatment_order"].duplicated().any():
        raise ValueError("Export contains duplicate treatment identities")
    if set(exp_df["treatment_order"].astype(int)) != set(ds._treatment_orders):
        raise ValueError("Export treatment identities do not match the selected input events exactly")
    assert "emb_0" in exp_df.columns, "Export missing pure state representation columns (emb_0)"
    assert "fused_0" in exp_df.columns, "Export missing fused representation columns (fused_0)"

    # Verify numerical consistency between restored model z and exported embeddings
    batch_orders = device_batch["treatment_order"].cpu().tolist()
    exp_sub = exp_df.set_index("treatment_order").loc[batch_orders]
    emb_cols = [f"emb_{i}" for i in range(64)]
    exp_z = torch.from_numpy(exp_sub[emb_cols].to_numpy(dtype=np.float32)).to(dev)
    max_export_z_diff = float((z_restored - exp_z).abs().max().item())
    assert max_export_z_diff < 1e-4, f"Exported z differs from restored model z: diff={max_export_z_diff}"

    audit_results["tests"]["checkpoint_restore"] = {
        "status": "PASS",
        "contract_verified": True,
        "restored_z_finite": True,
        "restored_pred_finite": True,
        "max_restore_z_diff": max_restore_z_diff,
        "max_restore_prediction_diff": max_restore_prediction_diff,
        "max_export_z_diff": max_export_z_diff,
        "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
    }
    logger.info("[PASS] Checkpoint restoration verified with full contract and exact export match (max diff: %.6e)", max_export_z_diff)

    audit_results["tests"]["embedding_export"] = {
        "status": "PASS",
        "exported_rows": len(exp_df),
        "unique_treatment_orders": int(exp_df["treatment_order"].nunique()),
        "exported_columns": len(exp_df.columns),
        "emb_dim": 64,
        "fused_dim": 160,  # 64 + 32 + 64
        "export_file": str(export_path),
        "export_sha256": hashlib.sha256(export_path.read_bytes()).hexdigest(),
    }
    logger.info("[PASS] Embeddings exported: %d rows, pure z (emb_0..63) and fused (fused_0..159) separated", len(exp_df))

    # Determine execution mode and summary from dataset manifest
    manifest_file = dataset_dir / "manifest.json"
    if manifest_file.is_file():
        ds_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        is_synth = bool(ds_manifest.get("is_synthetic", False)) or not bool(
            ds_manifest.get("strict_production", False)
        )
        exec_mode = ds_manifest.get(
            "execution_mode",
            "SYNTHETIC_TEST" if is_synth else "SERVER_PRODUCTION_ACCEPTANCE",
        )
    else:
        is_synth = bool(is_synthetic)
        exec_mode = "SYNTHETIC_TEST" if is_synth else "SERVER_PRODUCTION_ACCEPTANCE"

    if exec_mode == "SYNTHETIC_TEST":
        summary = "SYNTHETIC_PASS"
    elif exec_mode == "FIXTURE_TEST_MODE":
        summary = "FIXTURE_VERIFICATION_PASS"
    else:
        summary = "ALL_PASS"

    audit_results["is_synthetic"] = is_synth
    audit_results["execution_mode"] = exec_mode
    audit_results["summary"] = summary
    report_file = output_dir / "smoke_verification_report.json"
    report_file.write_text(json.dumps(audit_results, indent=2), encoding="utf-8")
    logger.info("Complete verification report saved to %s", report_file)
    return audit_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify S/T/C model on server real assets.")
    parser.add_argument("--stc-inputs-dir", type=Path, default=None)
    parser.add_argument("--response-release", type=Path, default=None)
    parser.add_argument("--outcome-family", choices=["viirs", "poi"], default="viirs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/stc_server_smoke/viirs")
    parser.add_argument("--n-samples", type=int, default=96)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--synthetic-test",
        action="store_true",
        default=False,
        help="Allow synthesizing mock response targets for local offline testing. Does NOT constitute server acceptance.",
    )
    parser.add_argument(
        "--strict-production",
        action="store_true",
        default=False,
        help="Enforce strict production response artifact provenance checks.",
    )
    parser.add_argument("--treatments-path", type=Path, default=None, help="Explicit path to treatments parquet table")
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit path to data root directory")
    parser.add_argument("--streetview-index-path", type=Path, default=None, help="Explicit path to streetview_state_assets.parquet")
    parser.add_argument("--satellite-root", type=Path, default=None, help="Explicit path to satellite images root directory")
    args = parser.parse_args()

    project_root = Path(args.project_root or os.environ.get("MIT_PROJECT_ROOT") or ROOT).resolve()
    stc_inputs_dir = args.stc_inputs_dir

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slice_dir = args.output_dir / "data"
    try:
        if stc_inputs_dir is None:
            stc_inputs_dir = _resolve_stc_input_directory(project_root)
        build_smoke_slice(
            stc_inputs_dir=stc_inputs_dir,
            response_release=args.response_release,
            candidate_dir=slice_dir,
            outcome_family=args.outcome_family,
            n_samples=args.n_samples,
            synthetic_test=args.synthetic_test,
            strict_production=args.strict_production,
            treatments_path=args.treatments_path,
            streetview_index_path=args.streetview_index_path,
            project_root=project_root,
            satellite_root=args.satellite_root,
        )

        res = run_smoke_verification(
            dataset_dir=slice_dir,
            output_dir=args.output_dir,
            outcome_family=args.outcome_family,
            device=args.device,
            is_synthetic=args.synthetic_test,
            streetview_index_path=args.streetview_index_path,
            satellite_root=args.satellite_root,
        )
    except Exception as exc:
        failure_report = {
            "status": "FAIL",
            "outcome_family": args.outcome_family,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "host": os.environ.get("MIT_ACCEPTANCE_HOST"),
            "run_id": os.environ.get("MIT_ACCEPTANCE_RUN_ID"),
            "project_root": str(project_root),
            "candidate_code_tree_sha256": os.environ.get("MIT_CANDIDATE_TREE_SHA256"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "arguments": {
                "stc_inputs_dir": str(stc_inputs_dir) if stc_inputs_dir else None,
                "response_release": str(args.response_release) if args.response_release else None,
                "output_dir": str(args.output_dir),
                "n_samples": args.n_samples,
                "device": args.device,
                "strict_production": args.strict_production,
                "project_root": str(project_root),
                "treatments_path": str(args.treatments_path) if args.treatments_path else None,
                "streetview_index_path": str(args.streetview_index_path) if args.streetview_index_path else None,
                "satellite_root": str(args.satellite_root) if args.satellite_root else None,
            },
        }
        failure_path = args.output_dir / f"failure_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        failure_path.write_text(json.dumps(failure_report, indent=2), encoding="utf-8")
        logger.exception("S/T/C smoke verification failed; evidence saved to %s", failure_path)
        raise
    print("\n" + "=" * 60)
    print(f"S/T/C SMOKE VERIFICATION RESULT: {res['summary']} ({args.outcome_family.upper()})")
    print(f"MODE: {res['execution_mode']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
