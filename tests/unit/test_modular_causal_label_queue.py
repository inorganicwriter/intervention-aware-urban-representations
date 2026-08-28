from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

import urban_intervention.causal.label_queue as modular
from urban_intervention.causal.label_queue import cli as modular_cli
from urban_intervention.causal.label_queue import orchestrator as modular_orchestrator

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_PATH = ROOT / "scripts" / "causal_python" / "run_causal_label_queue.py"
MODULAR_PATH = ROOT / "scripts" / "causal_python" / "run_causal_label_queue_modular.py"
ORIGINAL_NORMALIZED_SHA256 = "b3f0886a75bd0ad8e27a72610d47ff2b6ba2debcecfd704765af72978624e355"

SPEC = importlib.util.spec_from_file_location("frozen_causal_label_queue", ORIGINAL_PATH)
assert SPEC is not None and SPEC.loader is not None
ORIGINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORIGINAL)

SETTINGS_TO_ORIGINAL = {
    "family_queue": "FAMILY_QUEUE",
    "anticipation_months": "_ANTICIPATION_MONTHS",
    "price_measure": "_PRICE_MEASURE",
    "label_window": "_LABEL_WINDOW",
    "transaction_count_threshold": "_TRANSACTION_COUNT_THRESHOLD",
    "run_mode": "_RUN_MODE",
    "estimator_backend": "_ESTIMATOR_BACKEND",
    "max_gsc_cross_city_donors": "_MAX_GSC_CROSS_CITY_DONORS",
    "gsc_donor_sampling_seed": "_GSC_DONOR_SAMPLING_SEED",
    "qualification_receipt": "_QUALIFICATION_RECEIPT",
    "qualification_proof": "_QUALIFICATION_PROOF",
    "cross_city_design_cache": "_CROSS_CITY_DESIGN_CACHE",
    "r_timeout_seconds": "_R_TIMEOUT_SECONDS",
}


class _NormalizeSettings(ast.NodeTransformer):
    def visit_Attribute(self, node: ast.Attribute):  # noqa: N802
        node = self.generic_visit(node)
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "settings"
            and node.attr in SETTINGS_TO_ORIGINAL
        ):
            return ast.copy_location(
                ast.Name(id=SETTINGS_TO_ORIGINAL[node.attr], ctx=node.ctx), node
            )
        return node

    def visit_Global(self, node: ast.Global):  # noqa: N802
        # The modular implementation mutates one settings object and therefore
        # has no function-level global declarations.
        return None


def _normalized_function_dump(node: ast.FunctionDef) -> str:
    normalized = _NormalizeSettings().visit(ast.fix_missing_locations(node))
    assert isinstance(normalized, ast.FunctionDef)
    return ast.dump(normalized, include_attributes=False)


def test_frozen_original_file_is_unchanged() -> None:
    normalized = ORIGINAL_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert hashlib.sha256(normalized.encode("utf-8")).hexdigest() == (
        ORIGINAL_NORMALIZED_SHA256
    )


def test_all_original_functions_have_ast_equivalent_modular_implementations() -> None:
    tree = ast.parse(ORIGINAL_PATH.read_text(encoding="utf-8"))
    original_functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert len(original_functions) == 49

    for name, original_node in original_functions.items():
        modular_function = getattr(modular, name)
        modular_node = ast.parse(inspect.getsource(modular_function)).body[0]
        assert isinstance(modular_node, ast.FunctionDef)
        assert _normalized_function_dump(modular_node) == _normalized_function_dump(
            original_node
        ), name


def test_cli_option_surface_matches_frozen_entrypoint() -> None:
    outputs = []
    for path in (ORIGINAL_PATH, MODULAR_PATH):
        completed = subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stdout
        outputs.append(set(re.findall(r"--[a-z][a-z0-9-]*", completed.stdout)))
    assert outputs[0] == outputs[1]


def test_specs_paths_commands_and_selection_match(monkeypatch) -> None:
    values = {
        "_ANTICIPATION_MONTHS": 12,
        "_PRICE_MEASURE": "median",
        "_LABEL_WINDOW": 3,
        "_TRANSACTION_COUNT_THRESHOLD": 2,
        "_RUN_MODE": "preview",
        "_ESTIMATOR_BACKEND": "python_gpu",
        "_MAX_GSC_CROSS_CITY_DONORS": 1234,
        "_GSC_DONOR_SAMPLING_SEED": 77,
        "_QUALIFICATION_RECEIPT": None,
        "_QUALIFICATION_PROOF": {},
    }
    for name, value in values.items():
        monkeypatch.setattr(ORIGINAL, name, value)
    for field, original_name in SETTINGS_TO_ORIGINAL.items():
        if original_name in values:
            monkeypatch.setattr(modular.settings, field, values[original_name])

    row = pd.Series(
        {
            "treatment_order": 7,
            "city_key": "alpha",
            "grid_id": "g7",
            "opening_month": "2020-07",
            "outcome_family": "housing",
        }
    )
    assert ORIGINAL.specification_fingerprint(row) == modular.specification_fingerprint(row)
    assert ORIGINAL.fixed_control_output(row) == modular.fixed_control_output(row)
    assert ORIGINAL.gsc_output(row, "housing_log_price") == modular.gsc_output(
        row, "housing_log_price"
    )
    assert ORIGINAL.mc_output(row, "housing_log_price") == modular.mc_output(
        row, "housing_log_price"
    )
    assert ORIGINAL.mc_family_run_output(row) == modular.mc_family_run_output(row)
    assert ORIGINAL.python_estimator_command(
        row, "gsc", "same_city", "run-7"
    ) == modular.python_estimator_command(row, "gsc", "same_city", "run-7")

    queue = pd.DataFrame(
        {
            "treatment_order": [7, 7, 8, 9],
            "outcome_family": pd.Series(
                ["housing", "poi", "housing", "population"], dtype="string"
            ),
            "status": pd.Series(
                ["pending", "gsc_pending", "mc_pending", "skipped"], dtype="string"
            ),
        }
    )
    original_indices = ORIGINAL.eligible_indices(
        queue, 7, 9, None, "all", 10, retry_skipped=True
    )
    modular_indices = modular.eligible_indices(
        queue, 7, 9, None, "all", 10, retry_skipped=True
    )
    assert original_indices.tolist() == modular_indices.tolist()


def test_dry_run_task_observation_matches_frozen_implementation(
    monkeypatch, capsys
) -> None:
    queue = pd.DataFrame(
        {
            "treatment_order": [7],
            "outcome_family": pd.Series(["population"], dtype="string"),
            "city_key": ["alpha"],
            "grid_id": ["g7"],
            "opening_month": ["2020-01"],
            "status": pd.Series(["pending"], dtype="string"),
        }
    )
    support = pd.DataFrame(
        {
            "treatment_order": [7],
            "housing_complete": [False],
            "viirs_complete": [True],
            "population_complete": [True],
            "poi_complete": [False],
        }
    )
    controls = pd.DataFrame(
        {
            "treatment_order": [7],
            "status": ["matched"],
            "control_unit_key": ["alpha::control"],
        }
    )
    monkeypatch.setattr(ORIGINAL, "recover_completed_task", lambda *_args, **_kwargs: False)
    ORIGINAL.process_one(queue.copy(), 0, support, controls, dry_run=True)
    original_output = capsys.readouterr().out

    monkeypatch.setattr(
        modular_orchestrator,
        "recover_completed_task",
        lambda *_args, **_kwargs: False,
    )
    modular.process_one(queue.copy(), 0, support, controls, dry_run=True)
    modular_output = capsys.readouterr().out
    assert modular_output == original_output


def test_modular_cli_configures_shared_runtime_and_dispatches_dry_run(
    tmp_path, monkeypatch
) -> None:
    queue = pd.DataFrame(
        {
            "treatment_order": [7],
            "outcome_family": pd.Series(["population"], dtype="string"),
            "status": pd.Series(["pending"], dtype="string"),
        }
    )
    support = pd.DataFrame({"treatment_order": [7]})
    controls = pd.DataFrame({"treatment_order": [7], "status": ["matched"]})

    class _Table:
        def to_pandas(self):
            return support

    observed = []
    monkeypatch.setattr(modular.settings, "family_queue", tmp_path / "family.csv")
    monkeypatch.setattr(modular_cli, "read_family_queue", lambda _path: queue)
    monkeypatch.setattr(modular_cli, "read_control_queue", lambda _path: controls)
    monkeypatch.setattr(modular_cli.pq, "read_table", lambda _path: _Table())
    monkeypatch.setattr(
        modular_cli,
        "invalidate_stale_terminal_tasks",
        lambda _queue, _orders: 0,
    )
    monkeypatch.setattr(
        modular_cli,
        "process_one",
        lambda *args, **kwargs: observed.append((args, kwargs)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULAR_PATH),
            "--orders",
            "7",
            "--family",
            "population",
            "--dry-run",
            "--anticipation-months",
            "12",
            "--window",
            "3",
            "--transaction-count-threshold",
            "2",
        ],
    )

    assert modular_cli.main() == 0
    assert len(observed) == 1
    assert observed[0][0][4] is True
    assert modular.settings.estimator_backend == "python_gpu"
    assert modular.settings.anticipation_months == 12
    assert modular.settings.label_window == 3
    assert modular.settings.transaction_count_threshold == 2


def test_modular_fallback_routes_matching_and_gsc_failures_to_mc(
    tmp_path, monkeypatch
) -> None:
    queue = pd.DataFrame(
        {
            "treatment_order": [7],
            "outcome_family": pd.Series(["population"], dtype="string"),
            "city_key": ["alpha"],
            "grid_id": ["g7"],
            "opening_month": ["2020-01"],
            "status": pd.Series(["pending"], dtype="string"),
            "selected_method": pd.Series([pd.NA], dtype="string"),
            "failure_reason": pd.Series([pd.NA], dtype="string"),
        }
    )
    support = pd.DataFrame(
        {
            "treatment_order": [7],
            "housing_complete": [False],
            "viirs_complete": [False],
            "population_complete": [True],
            "poi_complete": [False],
        }
    )
    controls = pd.DataFrame(
        {
            "treatment_order": [7],
            "status": ["matched"],
            "control_unit_key": ["alpha::control"],
        }
    )
    mc_calls = []
    monkeypatch.setattr(
        modular_orchestrator,
        "recover_completed_task",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        modular_orchestrator,
        "family_has_observed_support",
        lambda _row: True,
    )
    monkeypatch.setattr(
        modular_orchestrator,
        "run_frozen_control",
        lambda *_args: (False, [], {"reason": "matching_failed"}),
    )
    monkeypatch.setattr(
        modular_orchestrator,
        "run_gsc_scope",
        lambda *_args: (False, [], {"reason": "gsc_failed"}),
    )
    monkeypatch.setattr(
        modular_orchestrator,
        "run_mc_stage",
        lambda *args, **kwargs: mc_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(modular_orchestrator, "atomic_csv", lambda *_args: None)
    monkeypatch.setattr(modular_orchestrator, "atomic_json", lambda *_args: None)
    monkeypatch.setattr(
        modular_orchestrator,
        "task_directory",
        lambda order, family: tmp_path / str(order) / family,
    )

    modular_orchestrator.process_one(
        queue,
        0,
        support,
        controls,
        dry_run=False,
        phase="all",
        control_queue_path=tmp_path / "controls.csv",
    )

    assert len(mc_calls) == 1
    assert queue.loc[0, "status"] == "mc_pending"
    assert queue.loc[0, "failure_reason"] == "gsc_failed"
