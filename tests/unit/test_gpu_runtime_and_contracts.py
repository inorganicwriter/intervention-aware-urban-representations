from __future__ import annotations

import numpy as np
import pytest

from urban_intervention.causal.gpu.contracts import MatchingInput, PanelData
from urban_intervention.causal.gpu.runtime import RuntimeConfig, TorchRuntime
from urban_intervention.causal.gpu.scheduler import GpuTask, sticky_partitions


def test_runtime_is_deterministic_float64_on_cpu() -> None:
    runtime = TorchRuntime(RuntimeConfig(device="cpu"))
    tensor = runtime.tensor([1.0, 2.0])
    assert str(tensor.dtype) == "torch.float64"
    assert runtime.metadata()["resolved_device"] == "cpu"
    assert runtime.metadata()["allow_tf32"] is False


def test_runtime_copies_readonly_numpy_arrays() -> None:
    source = np.array([1.0, 2.0], dtype=np.float64)
    source.flags.writeable = False
    runtime = TorchRuntime(RuntimeConfig(device="cpu"))

    tensor = runtime.tensor(source)
    tensor[0] = 9.0

    assert source[0] == 1.0
    assert tensor[0].item() == 9.0


def test_matching_contract_rejects_nonfinite_features() -> None:
    with pytest.raises(ValueError, match="finite"):
        MatchingInput(target=np.array([1.0, np.nan]), donors=np.ones((3, 2)))


def test_panel_contract_allows_unobserved_treated_cells() -> None:
    y = np.arange(12, dtype=float).reshape(4, 3)
    observed = np.ones_like(y, dtype=bool)
    treated = np.zeros_like(y, dtype=bool)
    observed[3, 0] = False
    treated[3, 0] = True
    panel = PanelData(y=y, observed=observed, treated=treated)
    assert not panel.observed[3, 0]
    assert panel.treated[3, 0]


def test_panel_contract_finds_absorbing_treatment() -> None:
    y = np.arange(15, dtype=float).reshape(5, 3)
    treated = np.zeros_like(y, dtype=bool)
    treated[2:, 1] = True
    panel = PanelData(y=y, treated=treated)
    assert panel.single_treated_unit() == 1
    assert panel.treatment_start() == 2
    assert not panel.untreated_observed[3, 1]


def test_sticky_partitions_keep_cache_keys_together() -> None:
    tasks = [
        GpuTask("a1", "a", {}),
        GpuTask("b1", "b", {}),
        GpuTask("a2", "a", {}),
        GpuTask("c1", "c", {}),
    ]
    partitions = sticky_partitions(tasks, [0, 1])
    owners = {
        task.task_id: gpu_id
        for gpu_id, assigned in partitions.items()
        for task in assigned
    }
    assert owners["a1"] == owners["a2"]
    assert set(owners.values()) == {0, 1}


def test_sticky_partitions_balance_estimated_cost() -> None:
    tasks = [
        GpuTask("large", "large", {}, cost=10),
        GpuTask("small-1", "small-1", {}, cost=1),
        GpuTask("small-2", "small-2", {}, cost=1),
    ]
    partitions = sticky_partitions(tasks, [0, 1])
    assert [task.task_id for task in partitions[0]] == ["large"]
    assert [task.task_id for task in partitions[1]] == ["small-1", "small-2"]
