"""Tests for the local weighted-training memory smoke script."""

from __future__ import annotations

import torch

from scripts.representation.smoke_weighted_training import run_smoke


def test_run_smoke_completes_on_cpu() -> None:
    result = run_smoke(
        batch_size=4,
        queue_size=8,
        steps=3,
        warmup=1,
        feature_dim=8,
        embedding_dim=16,
        hidden_dims=(16, 8),
        learning_rate=1e-3,
        seed=0,
        device=torch.device("cpu"),
    )
    assert result["status"] == "completed"
    assert result["steps_recorded"] == 3
    assert result["step_seconds"]["mean"] is not None
    assert "step_memory_mb" not in result
    assert "largest_step_time_jump_seconds" in result
