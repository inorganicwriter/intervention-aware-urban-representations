from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from urban_intervention.causal.response_artifact import (
    ArtifactInputs,
    build_response_frame,
    collect_task_products,
    git_state,
    publish_response_artifact,
    require_reproducible_code_state,
    sha256_file,
    validate_control_design_provenance,
)


def _fixture(tmp_path: Path) -> ArtifactInputs:
    treatments = pd.DataFrame(
        {
            "treatment_order": [1, 2],
            "city_key": ["alpha", "beta"],
            "grid_id": ["g1", "g2"],
            "station_event_id": ["s1", "s2"],
            "opening_month": ["2020-01", "2020-01"],
        }
    )
    treatment_path = tmp_path / "treatments.parquet"
    treatments.to_parquet(treatment_path, index=False)
    queue_rows = []
    for order in (1, 2):
        for family in ("housing", "poi", "population", "viirs"):
            success = family == "population"
            queue_rows.append(
                {
                    "treatment_order": order,
                    "outcome_family": family,
                    "status": ("matched_labelled" if order == 1 else "gsc_labelled")
                    if success
                    else "skipped",
                    "selected_method": (
                        "frozen_matched_change_12m_baseline"
                        if order == 1
                        else "xu_2017_gsynth_same_city"
                    )
                    if success
                    else pd.NA,
                    "failure_reason": pd.NA if success else "no_outcome_support",
                }
            )
    family_path = tmp_path / "family.csv"
    pd.DataFrame(queue_rows).to_csv(family_path, index=False)
    controls = pd.DataFrame(
        {
            "treatment_order": [1, 2],
            "status": ["matched", "gsc_pending"],
            "active_families": ["poi+population", "population"],
            "selected_method": ["Matching::Match_M5_static_refine", pd.NA],
            "donor_scope": ["same_city", pd.NA],
            "control_city_key": ["alpha", pd.NA],
            "control_grid_id": ["c1", pd.NA],
            "control_unit_key": ["alpha::c1", pd.NA],
            "failure_reason": [pd.NA, "no_match"],
            "candidate_count": [100, pd.NA],
            "candidate_city_count": [1, pd.NA],
            "training_feature_count": [8, pd.NA],
            "holdout_feature_count": [4, pd.NA],
            "training_distance": [0.5, pd.NA],
            "training_distance_threshold": [1.0, pd.NA],
            "holdout_rms_standardized_gap": [0.2, pd.NA],
            "holdout_rms_threshold": [0.5, pd.NA],
            "holdout_max_abs_standardized_gap": [0.3, pd.NA],
            "holdout_max_abs_threshold": [0.7, pd.NA],
            "control_selection_uses_post_outcome": [False, False],
        }
    )
    control_path = tmp_path / "control.csv"
    controls.to_csv(control_path, index=False)
    task_root = tmp_path / "tasks"
    for order, method in (
        (1, "frozen_matched_change_12m_baseline"),
        (2, "xu_2017_gsynth_same_city"),
    ):
        directory = task_root / f"{order:05d}" / "population"
        directory.mkdir(parents=True)
        labels = pd.DataFrame(
            {
                "treatment_order": [order] * 3,
                "outcome_family": ["population"] * 3,
                "city_key": ["alpha" if order == 1 else "beta"] * 3,
                "grid_id": ["g1" if order == 1 else "g2"] * 3,
                "opening_month": ["2020-01"] * 3,
                "outcome": ["population_log"] * 3,
                "event_time": [1, 2, 3],
                "specification_id": ["main_a6_r1km"] * 3,
                "observed": [2.0, 2.1, 2.2],
                "counterfactual": [1.9, 1.95, 2.0],
                "causal_response_label": [0.1, 0.15, 0.2],
                "label_available": [True] * 3,
                "transformed_scale": [True] * 3,
                "method": [method] * 3,
                "control_unit_key": ["alpha::c1" if order == 1 else pd.NA] * 3,
                "standard_error": [pd.NA] * 3 if order == 1 else [0.05] * 3,
                "confidence_lower": [pd.NA] * 3 if order == 1 else [0.0, 0.05, 0.1],
                "confidence_upper": [pd.NA] * 3 if order == 1 else [0.2, 0.25, 0.3],
                "bootstrap_repetitions": [0] * 3 if order == 1 else [200] * 3,
            }
        )
        label_path = directory / "labels.parquet"
        labels.to_parquet(label_path, index=False)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "causal_response_labels_v1",
                    "status": "matched_labelled" if order == 1 else "gsc_labelled",
                    "treatment_order": order,
                    "outcome_family": "population",
                    "city_key": "alpha" if order == 1 else "beta",
                    "grid_id": "g1" if order == 1 else "g2",
                    "station_event_id": "s1" if order == 1 else "s2",
                    "opening_month": "2020-01",
                    "label_rows": len(labels),
                    "labels_sha256": sha256_file(label_path),
                    "production_eligible": True,
                }
            ),
            encoding="utf-8",
        )
    return ArtifactInputs(treatment_path, family_path, control_path, task_root)


def test_response_artifact_preserves_expected_failures_and_quality(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    artifact, diagnostics, _ = build_response_frame(
        inputs, "main_a6_r1km", strict_production=False, workers=2
    )
    assert len(artifact) == 2 * 27
    matched = artifact[(artifact.treatment_order == 1) & (artifact.outcome_family == "population")]
    gsc = artifact[(artifact.treatment_order == 2) & (artifact.outcome_family == "population")]
    unavailable = artifact[artifact.outcome_family == "housing"]
    assert matched.training_mask.all()
    assert not matched.uncertainty_available.any()
    assert gsc.training_mask.all()
    assert gsc.uncertainty_available.all()
    assert set(matched.quality_grade) == {"matched_same_city_pass"}
    assert set(gsc.quality_grade) == {"gsc_same_city_pass"}
    assert not unavailable.training_mask.any()
    assert not unavailable.main_training_mask.any()
    assert not unavailable.cross_city_extension_mask.any()
    assert set(unavailable.failure_reason) == {"no_outcome_support"}
    assert diagnostics["training_labels"] == 6


def test_response_artifact_accepts_cross_validated_mc_labels(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    queue = pd.read_csv(inputs.family_queue)
    mask = (queue["treatment_order"] == 2) & (queue["outcome_family"] == "population")
    queue.loc[mask, ["status", "selected_method"]] = ["mc_labelled", "athey_2021_mc_same_city"]
    queue.to_csv(inputs.family_queue, index=False)
    directory = inputs.task_root / "00002" / "population"
    label_path = directory / "labels.parquet"
    labels = pd.read_parquet(label_path)
    labels["method"] = "athey_2021_mc_same_city"
    labels["pre_observed_periods"] = 1
    labels["pre_rmspe"] = 0.1
    labels["mc_lambda"] = 0.0
    labels["mc_regularized"] = False
    labels["mc_cv_mspe"] = 0.05
    labels["uncertainty_source"] = "mc_nonparametric_bootstrap"
    labels.to_parquet(label_path, index=False)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "mc_labelled"
    manifest["method"] = "athey_2021_mc_same_city"
    manifest["labels_sha256"] = sha256_file(label_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    artifact, diagnostics, _ = build_response_frame(
        inputs, "main_a6_r1km", strict_production=False, workers=1
    )
    mc = artifact[(artifact.treatment_order == 2) & (artifact.outcome_family == "population")]
    assert mc.training_mask.all()
    assert mc.uncertainty_available.all()
    assert set(mc.quality_grade) == {"mc_same_city_minimal_pre_support"}
    assert set(mc.mc_lambda) == {0.0}
    assert not mc.mc_regularized.any()
    task_status_counts = diagnostics["task_status_counts"]
    assert isinstance(task_status_counts, dict)
    assert task_status_counts["mc_labelled"] == 1


def test_response_artifact_infers_cross_city_scope_from_task_method(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    directory = inputs.task_root / "00002" / "population"
    label_path = directory / "labels.parquet"
    labels = pd.read_parquet(label_path)
    labels["method"] = "xu_2017_gsynth_all_city_standardized"
    labels["donor_scope"] = "all_city_standardized"
    labels["transaction_count"] = [1, 2, 3]
    labels.to_parquet(label_path, index=False)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["labels_sha256"] = sha256_file(label_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    artifact, diagnostics, _ = build_response_frame(
        inputs, "main_a6_r1km", strict_production=False, workers=1
    )
    cross = artifact[
        (artifact.treatment_order == 2) & (artifact.outcome_family == "population")
    ]
    assert set(cross.donor_scope) == {"all_city_standardized"}
    assert not cross.main_spec.any()
    assert not cross.main_training_mask.any()
    assert cross.cross_city_extension_mask.all()
    assert cross.transaction_count.tolist() == [1, 2, 3]
    assert diagnostics["cross_city_extension_labels"] == 3


def test_response_artifact_rejects_mc_status_without_estimator_proof(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    queue = pd.read_csv(inputs.family_queue)
    mask = (queue["treatment_order"] == 2) & (queue["outcome_family"] == "population")
    queue.loc[mask, ["status", "selected_method"]] = ["mc_labelled", "athey_2021_mc_same_city"]
    queue.to_csv(inputs.family_queue, index=False)
    directory = inputs.task_root / "00002" / "population"
    label_path = directory / "labels.parquet"
    labels = pd.read_parquet(label_path)
    labels["method"] = "athey_2021_mc_same_city"
    labels.to_parquet(label_path, index=False)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "mc_labelled"
    manifest["method"] = "athey_2021_mc_same_city"
    manifest["labels_sha256"] = sha256_file(label_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="cross-validated estimator proof"):
        build_response_frame(inputs, "main_a6_r1km", strict_production=False, workers=1)


def test_response_artifact_release_is_immutable_and_hashed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    output = publish_response_artifact(
        inputs,
        tmp_path / "releases",
        release_id="audit",
        strict_production=False,
        project_root=tmp_path,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "urban_response_artifact_release_v1"
    assert manifest["strict_production"] is False
    assert manifest["artifact"]["rows"] == 54
    with pytest.raises(FileExistsError):
        publish_response_artifact(
            inputs,
            tmp_path / "releases",
            release_id="audit",
            strict_production=False,
            project_root=tmp_path,
        )


def test_production_release_refuses_small_or_unfinished_queue(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    with pytest.raises(ValueError, match="5,048"):
        build_response_frame(inputs, "main_a6_r1km", strict_production=True)


def test_control_provenance_rejects_legacy_record(tmp_path: Path) -> None:
    control_root = tmp_path / "control_tasks" / "00001"
    control_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "treatment_order": 1,
                "status": "matched",
                "schema": "grid_control_design_v1",
                "implementation_version": "r-reference-grid-v1",
                "backend": "r_matching",
                "viirs_cache_contract": "complete_44_city_2012_2024_monthly_v1",
                "selected_method": "Matching::Match_M1",
                "control_selection_uses_post_outcome": False,
            }
        ]
    ).to_csv(control_root / "control_record.csv", index=False)
    queue = pd.DataFrame({"treatment_order": [1], "status": ["matched"]})

    with pytest.raises(ValueError, match="stale control-design schema"):
        validate_control_design_provenance(queue, control_task_root=tmp_path / "control_tasks")


def test_control_provenance_accepts_current_gpu_record(tmp_path: Path) -> None:
    control_root = tmp_path / "control_tasks" / "00001"
    control_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "treatment_order": 1,
                "status": "matched",
                "schema": "grid_control_design_v3_exact_stable_ties",
                "implementation_version": "python-causal-v3",
                "backend": "python_pytorch",
                "viirs_cache_contract": "complete_44_city_2012_2024_monthly_v1",
                "selected_method": "python_gpu_M5_static_refine",
                "control_selection_uses_post_outcome": False,
            }
        ]
    ).to_csv(control_root / "control_record.csv", index=False)
    queue = pd.DataFrame({"treatment_order": [1], "status": ["matched"]})

    validate_control_design_provenance(
        queue,
        control_task_root=tmp_path / "control_tasks",
        expected_backend="python_gpu",
    )


def test_strict_task_collection_rejects_unqualified_python_manifest(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    treatments = pd.read_parquet(inputs.treatments)
    family_queue = pd.read_csv(inputs.family_queue)
    for order in (1, 2):
        directory = inputs.task_root / f"{order:05d}" / "population"
        label_path = directory / "labels.parquet"
        labels = pd.read_parquet(label_path)
        labels["estimator_backend"] = "python_gpu"
        labels["implementation_version"] = "python-causal-v3"
        labels.to_parquet(label_path, index=False)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["details"] = {"run_id": f"run-{order}"}
        manifest["labels_sha256"] = sha256_file(label_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="qualification proof"):
        collect_task_products(
            family_queue,
            treatments,
            inputs.task_root,
            "main_a6_r1km",
            strict_production=True,
            workers=1,
        )


def test_response_artifact_rejects_labels_swapped_between_task_directories(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    first = inputs.task_root / "00001" / "population" / "labels.parquet"
    labels = pd.read_parquet(first)
    labels["treatment_order"] = 2
    labels.to_parquet(first, index=False)
    manifest_path = first.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["labels_sha256"] = sha256_file(first)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="treatment_order"):
        build_response_frame(inputs, "main_a6_r1km", strict_production=False, workers=1)


def test_strict_release_rejects_unknown_code_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fixture(tmp_path)
    monkeypatch.setattr(
        "urban_intervention.causal.response_artifact.git_state",
        lambda _root: {"commit": "unknown", "dirty": False, "source": "unavailable"},
    )
    with pytest.raises(ValueError, match="known code version"):
        publish_response_artifact(
            inputs,
            tmp_path / "releases",
            release_id="must_fail",
            strict_production=True,
            project_root=tmp_path,
        )
    with pytest.raises(ValueError, match="known code version"):
        require_reproducible_code_state({"commit": "unknown", "dirty": False})


def test_source_tree_hash_is_a_known_version_without_git(tmp_path: Path) -> None:
    source = tmp_path / "src" / "package"
    source.mkdir(parents=True)
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    state = git_state(tmp_path)
    assert state["source"] == "source_tree_sha256"
    assert str(state["commit"]).startswith("tree-sha256:")
    require_reproducible_code_state(state)
