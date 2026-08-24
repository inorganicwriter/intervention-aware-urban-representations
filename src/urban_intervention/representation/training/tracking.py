"""Experiment-run tracking for representation training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _append_run_record(
    output_dir: Path,
    training_config: dict[str, object],
    history: list[dict],
    test_metrics: dict[str, float],
    evaluation_report: dict[str, object],
) -> None:
    """Append one JSONL run record for experiment tracking.

    The record carries a content hash of the training configuration plus the
    headline test metrics and the chance-level baselines, so every run is
    self-describing and comparable without loading the full report.
    """
    # The digest covers only the reproducibility-relevant configuration:
    # created_utc, the absolute model_inputs_dir and the device are run
    # metadata, not hyperparameters, so identical configurations hash equal.
    hash_config = {
        key: value
        for key, value in training_config.items()
        if key not in {"created_utc", "model_inputs_dir", "device", "runtime"}
    }
    config_digest = hashlib.sha256(
        json.dumps(hash_config, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()
    best_val = min((entry["val_total"] for entry in history), default=None)
    record: dict[str, object] = {
        "created_utc": training_config["created_utc"],
        "dataset_id": training_config.get("dataset_id", ""),
        "config_sha256": config_digest,
        "config": training_config,
        "best_val_loss": best_val,
        "test_metrics": test_metrics,
    }
    baselines = evaluation_report.get("baselines")
    if isinstance(baselines, dict):
        test_baselines = baselines.get("test")
        if isinstance(test_baselines, dict):
            from ..baselines import headline_metric

            record["baseline_nn_corr@k"] = {
                name: headline_metric(entry)
                for name, entry in test_baselines.items()
                if isinstance(entry, dict)
            }
    test_entry = evaluation_report.get("test")
    if isinstance(test_entry, dict):
        retrieval = test_entry.get("retrieval")
        if isinstance(retrieval, dict):
            overall = retrieval.get("overall")
            if isinstance(overall, dict):
                record["test_nn_corr@k"] = overall.get("nn_corr@k")
    runs_path = output_dir / "runs.jsonl"
    with runs_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
