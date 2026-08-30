#!/usr/bin/env python
"""Build a model card for a representation-training run.

Reads ``training_config.json``, ``training_history.json``,
``test_metrics.json``, ``evaluation_report.json`` and ``best_model.pt`` from
a run directory produced by ``urban-train-representation`` and writes:

- ``model_card.json``: machine-readable card;
- ``model_card.md``: human-readable report with per-family evaluation tables
  and explicit limitation notes.

Usage:
    python -m urban_intervention.representation.model_card_cli OUTPUT_DIR [--out model_card.md]

Also exposed as the ``urban-build-model-card`` console script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# numpy must be imported before torch: on Windows, torch's MKL init sets FPU
# flags that make numpy's blas_fpe_check abort when numpy loads afterwards.
import numpy as np  # noqa: F401
import torch

REQUIRED_FILES = (
    "training_config.json",
    "training_history.json",
    "test_metrics.json",
    "evaluation_report.json",
    "best_model.pt",
)


def _family_rows(evaluation: dict) -> list[dict]:
    rows = []
    retrieval = evaluation.get("retrieval", {})
    overall = retrieval.get("overall", {})
    if overall:
        rows.append({"family": "overall", **overall})
    for family, block in retrieval.items():
        if family != "overall":
            rows.append({"family": family, **block})
    return rows


def _split_summary(name: str, evaluation: dict) -> dict:
    probe = evaluation.get("probe", {})
    embeddings_probe = probe.get("embeddings", {}) if isinstance(probe, dict) else {}
    raw_probe = probe.get("raw_features", {}) if isinstance(probe, dict) else {}
    permutation = evaluation.get("permutation", {})
    return {
        "split": name,
        "n_units": evaluation.get("n_units", 0),
        "nn_corr@k": (evaluation.get("retrieval", {}).get("overall", {}).get("nn_corr@k")),
        "baseline_corr": (evaluation.get("retrieval", {}).get("overall", {}).get("baseline_corr")),
        "ratio": (evaluation.get("retrieval", {}).get("overall", {}).get("ratio")),
        "bootstrap_ci": evaluation.get("bootstrap_ci", {}),
        "permutation_p_value": permutation.get("p_value"),
        "permutation_mean_null": permutation.get("mean_null"),
        "raw_feature_baseline_nn_corr@k": (
            evaluation.get("raw_feature_baseline", {}).get("overall", {}).get("nn_corr@k")
        ),
        "probe_embeddings_rmse": embeddings_probe.get("rmse_overall"),
        "probe_raw_rmse": raw_probe.get("rmse_overall"),
        "families": _family_rows(evaluation),
    }


def build_model_card(run_dir: Path, out_path: Path | None = None) -> dict:
    """Compose the model card from an existing run directory."""
    run_dir = Path(run_dir)
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Run directory {run_dir} lacks required files: {', '.join(missing)}"
        )

    config = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
    history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
    test_metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)

    best_epoch = min(history, key=lambda entry: entry["val_total"]) if history else None
    card: dict = {
        "schema": "urban_intervention_model_card",
        "dataset_id": config.get("dataset_id", ""),
        "strict_production": config.get("strict_production", False),
        "architecture": config.get("architecture", {}),
        "optimization": config.get("optimization", {}),
        "algorithm": config.get("algorithm", {}),
        "data_splits": config.get("data_splits", {}),
        "seed": config.get("seed"),
        "device": config.get("device"),
        "created_utc": config.get("created_utc"),
        "runtime": config.get("runtime", {}),
        "checkpoint": {
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
            "embedding_dim": checkpoint.get("embedding_dim"),
            "hidden_dims": checkpoint.get("hidden_dims"),
        },
        "training_history": {
            "epochs_logged": len(history),
            "best_epoch": best_epoch.get("epoch") if best_epoch else None,
            "best_val_total_loss": best_epoch.get("val_total") if best_epoch else None,
            "best_val_nn_corr@5": best_epoch.get("val_nn_corr@5") if best_epoch else None,
            "final_val_total_loss": history[-1].get("val_total") if history else None,
        },
        "test_metrics": test_metrics,
        "evaluation": {
            name: _split_summary(name, entry)
            for name, entry in evaluation.items()
            if name in ("validation", "test")
        },
        "baselines": _baseline_summary(evaluation),
        "limitations": _limitations(config, evaluation),
    }

    if out_path is None:
        out_path = run_dir / "model_card.json"
    out_path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = out_path if out_path.suffix.lower() == ".md" else run_dir / "model_card.md"
    md_path.write_text(_render_markdown(card), encoding="utf-8")
    return card


def _baseline_summary(evaluation: dict) -> dict:
    """Flat nn_corr@k per chance/appearance baseline for paper tables."""
    summary: dict = {}
    baselines = evaluation.get("baselines")
    if not isinstance(baselines, dict):
        return summary
    for pool_name, pool in baselines.items():
        if not isinstance(pool, dict):
            continue
        for baseline_name, entry in pool.items():
            if not isinstance(entry, dict):
                continue
            retrieval = entry.get("retrieval")
            overall = retrieval.get("overall") if isinstance(retrieval, dict) else None
            value = overall.get("nn_corr@k") if isinstance(overall, dict) else None
            summary[f"{pool_name}.{baseline_name}.nn_corr@k"] = value
            if isinstance(overall, dict):
                summary[f"{pool_name}.{baseline_name}.baseline_corr"] = overall.get("baseline_corr")
    return summary


def _limitations(config: dict, evaluation: dict) -> list[str]:
    notes = [
        "Checkpoint is selected by validation total loss, not by retrieval "
        "metric; nn_corr@k should be read together with its bootstrap CI and "
        "permutation p-value.",
        "Retrieval metrics compare against the random-neighbour baseline; the "
        "raw-feature retrieval baseline and linear-probe RMSE must be reported "
        "alongside to claim representation gains.",
    ]
    test_units = config.get("data_splits", {}).get("test_units", 0)
    if test_units < 30:
        notes.append(
            f"Test split contains only {test_units} units; retrieval metrics on "
            "such a small pool are unreliable for claims."
        )
    if not config.get("strict_production", True):
        notes.append(
            "Model trained on a non-production dataset release; results are "
            "not admissible as final paper evidence."
        )
    for name in ("validation", "test"):
        entry = evaluation.get(name, {})
        if entry.get("n_units", 0) < 2:
            notes.append(f"Evaluation pool '{name}' has fewer than 2 units.")
    return notes


def _render_markdown(card: dict) -> str:
    lines = ["# Model Card", ""]
    lines.append(
        f"**Dataset**: `{card['dataset_id']}` "
        f"(production: {str(card['strict_production']).lower()})"
    )
    architecture = card.get("architecture", {})
    lines.append(
        "**Architecture**: tabular encoder with hidden dims "
        f"{architecture.get('hidden_dims')} -> embedding "
        f"{architecture.get('embedding_dim')}; "
        f"image encoder: {'DINOv2' if architecture.get('use_image_encoder') else 'off'}"
        f" (pooling {architecture.get('image_pooling', 'max')}); "
        f"conditioning: {architecture.get('conditioning') or 'implicit'}"
    )
    optimization = card.get("optimization", {})
    lines.append(
        "**Training**: "
        f"{optimization.get('epochs')} epochs, batch {optimization.get('batch_size')}, "
        f"lr {optimization.get('learning_rate')}, weight decay {optimization.get('weight_decay')}, "
        f"temperature {optimization.get('temperature')}, rep_alpha "
        f"{optimization.get('rep_alpha')}, pred_weight {optimization.get('pred_weight')}, "
        f"seed {card.get('seed')}"
    )
    algorithm = card.get("algorithm", {})
    if algorithm:
        lines.append(
            "**Algorithm options**: "
            f"se_shrinkage {algorithm.get('se_shrinkage')}, "
            f"queue_size {algorithm.get('queue_size')}, "
            f"learnable_temperature {algorithm.get('learnable_temperature')}, "
            f"uncertainty_weighted {algorithm.get('uncertainty_weighted')}"
        )
    splits = card.get("data_splits", {})
    lines.append(
        "**Data splits**: "
        f"train {splits.get('train_units')} / validation "
        f"{splits.get('validation_units')} / test {splits.get('test_units')} units; "
        f"feature dim {card.get('architecture', {}).get('feature_dim')}, "
        f"response dim {splits.get('response_dim')}"
    )
    history = card.get("training_history", {})
    lines.append(
        "**History**: best epoch "
        f"{history.get('best_epoch')} (val total loss {history.get('best_val_total_loss')}, "
        f"val nn_corr@5 {history.get('best_val_nn_corr@5')}); "
        f"final val total loss {history.get('final_val_total_loss')}"
    )
    lines.append("")

    for split_name, summary in card.get("evaluation", {}).items():
        lines.append(f"## {split_name.capitalize()} split ({summary['n_units']} units)")
        lines.append("")
        lines.append("| family | nn_corr@k | baseline_corr | ratio |")
        lines.append("|---|---|---|---|")
        for row in summary["families"]:
            ratio = row.get("ratio")
            ratio_text = "—" if ratio is None else f"{ratio:.3f}"
            lines.append(
                f"| {row['family']} | {row['nn_corr@k']} | {row['baseline_corr']} | {ratio_text} |"
            )
        lines.append("")
        ci = summary.get("bootstrap_ci", {})
        lines.append(
            f"- nn_corr@k bootstrap 95% CI: "
            f"[{ci.get('lower')}, {ci.get('upper')}] (n_boot={ci.get('n_boot')})"
        )
        lines.append(
            f"- Permutation test (response-shuffle null): p = "
            f"{summary.get('permutation_p_value')} "
            f"(null mean {summary.get('permutation_mean_null')})"
        )
        lines.append(
            f"- Raw-feature baseline nn_corr@k: {summary.get('raw_feature_baseline_nn_corr@k')}"
        )
        lines.append(
            f"- Linear probe RMSE: embeddings "
            f"{summary.get('probe_embeddings_rmse')} vs raw features "
            f"{summary.get('probe_raw_rmse')}"
        )
        lines.append("")

    baselines = card.get("baselines", {})
    if baselines:
        lines.append("## Chance / appearance baselines (held-out test pool)")
        lines.append("")
        lines.append("| baseline | nn_corr@k | random-neighbour baseline |")
        lines.append("|---|---|---|")
        for key in sorted(baselines):
            if key.endswith(".nn_corr@k"):
                pool, name, _ = key.split(".")
                nn_corr = baselines.get(key)
                neighbour = baselines.get(f"{pool}.{name}.baseline_corr")
                lines.append(f"| {name} | {nn_corr} | {neighbour} |")
        lines.append("")

    lines.append("## Limitations")
    for note in card.get("limitations", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Training output directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Model card output path (default: run_dir/model_card.json + .md)",
    )
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        print(f"Error: run directory not found: {args.run_dir}", file=sys.stderr)
        return 1
    card = build_model_card(args.run_dir, args.out)
    print(
        f"Wrote model card for dataset {card['dataset_id']} to {args.run_dir / 'model_card.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
