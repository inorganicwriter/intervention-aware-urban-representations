from __future__ import annotations

import ast
import hashlib
import inspect
import json
import typing
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.helpers.representation import build_synthetic_model_inputs
from urban_intervention.representation import trainer as original
from urban_intervention.representation import trainer_modular as modular

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_PATH = ROOT / "src" / "urban_intervention" / "representation" / "trainer.py"
ORIGINAL_NORMALIZED_SHA256 = "cdcf1b0fff7082eddc2c597760c05e811c7ce36fcbada4a4419f728bafa9ea78"

CALLABLE_NAMES = (
    "_collate_fn",
    "_evaluate_retrieval",
    "_run_epoch",
    "_collect_pool",
    "_visualize_embeddings",
    "build_evaluation_report",
    "train_representation",
    "_append_run_record",
)


class _NormalizeMovedRelativeImports(ast.NodeTransformer):
    """Ignore only the package-depth change required by moving a function."""

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:  # noqa: N802
        node.level = 0
        return node


def _function_dump(function) -> str:
    node = ast.parse(inspect.getsource(function)).body[0]
    assert isinstance(node, ast.FunctionDef)
    normalized = _NormalizeMovedRelativeImports().visit(node)
    return ast.dump(normalized, include_attributes=False)


def _normalized_signature(function) -> tuple:
    signature = inspect.signature(function)

    def annotation_identity(annotation):
        return getattr(annotation, "__name__", annotation)

    parameters = tuple(
        (
            parameter.name,
            parameter.kind,
            parameter.default,
            annotation_identity(parameter.annotation),
        )
        for parameter in signature.parameters.values()
    )
    return parameters, annotation_identity(signature.return_annotation)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run_record(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    record.pop("created_utc", None)
    config = record.get("config")
    if isinstance(config, dict):
        config.pop("created_utc", None)
    return record


def test_frozen_original_trainer_is_unchanged() -> None:
    normalized = ORIGINAL_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert hashlib.sha256(normalized.encode("utf-8")).hexdigest() == (
        ORIGINAL_NORMALIZED_SHA256
    )


def test_modular_facade_preserves_complete_callable_surface() -> None:
    assert typing.get_type_hints(modular.Pool) == typing.get_type_hints(original.Pool)
    for name in CALLABLE_NAMES:
        assert hasattr(modular, name), name


@pytest.mark.parametrize("name", CALLABLE_NAMES)
def test_function_signature_and_ast_match(name: str) -> None:
    original_function = getattr(original, name)
    modular_function = getattr(modular, name)
    assert _normalized_signature(modular_function) == _normalized_signature(original_function)
    assert _function_dump(modular_function) == _function_dump(original_function)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 1}, "batch_size must be at least 2"),
        ({"epochs": 0}, "epochs must be positive"),
        ({"temperature": 0.0}, "temperature must be positive"),
        ({"rep_alpha": 1.1}, "rep_alpha must be between 0 and 1"),
        ({"pred_weight": -0.1}, "pred_weight must be between 0 and 1"),
        ({"queue_size": -1}, "queue_size must be non-negative"),
    ],
)
def test_runner_validation_matches(tmp_path: Path, kwargs: dict, message: str) -> None:
    common = {
        "model_inputs_dir": tmp_path / "missing-inputs",
        "output_dir": tmp_path / "output",
        **kwargs,
    }
    for implementation in (original.train_representation, modular.train_representation):
        with pytest.raises(ValueError, match=message):
            implementation(**common)


def test_deterministic_cpu_training_artifacts_match(tmp_path: Path) -> None:
    model_inputs_dir, _ = build_synthetic_model_inputs(tmp_path)
    original_output = tmp_path / "original-output"
    modular_output = tmp_path / "modular-output"
    common = {
        "model_inputs_dir": model_inputs_dir,
        "embedding_dim": 8,
        "hidden_dims": (8,),
        "dropout": 0.0,
        "batch_size": 4,
        "epochs": 1,
        "use_images": False,
        "device": "cpu",
        "seed": 712,
        "eval_k": 2,
        "eval_n_perm": 4,
        "eval_n_boot": 4,
        "probe_min_obs": 2,
        "run_baselines": False,
        "run_transfer": False,
        "visualize": False,
    }

    original_checkpoint = original.train_representation(
        output_dir=original_output,
        **common,
    )
    modular_checkpoint = modular.train_representation(
        output_dir=modular_output,
        **common,
    )

    for name in (
        "training_history.json",
        "test_metrics.json",
        "evaluation_report.json",
    ):
        assert _load_json(modular_output / name) == _load_json(original_output / name), name

    original_config = _load_json(original_output / "training_config.json")
    modular_config = _load_json(modular_output / "training_config.json")
    original_config.pop("created_utc")
    modular_config.pop("created_utc")
    assert modular_config == original_config

    assert _load_run_record(modular_output / "runs.jsonl") == _load_run_record(
        original_output / "runs.jsonl"
    )

    original_state = torch.load(original_checkpoint, map_location="cpu", weights_only=True)
    modular_state = torch.load(modular_checkpoint, map_location="cpu", weights_only=True)
    assert original_state.keys() == modular_state.keys()
    assert original_state["epoch"] == modular_state["epoch"]
    assert original_state["val_loss"] == modular_state["val_loss"]
    assert np.isfinite(original_state["val_loss"])
    assert original_state["feature_columns"] == modular_state["feature_columns"]
    assert original_state["model_state_dict"].keys() == modular_state["model_state_dict"].keys()
    for name, original_tensor in original_state["model_state_dict"].items():
        assert torch.equal(modular_state["model_state_dict"][name], original_tensor), name
