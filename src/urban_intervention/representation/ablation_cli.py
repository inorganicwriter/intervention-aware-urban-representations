#!/usr/bin/env python
"""Run an ablation grid over representation-training configurations.

Each spec in the JSON specs file is one full training + evaluation run
(``train_representation``) into ``<output>/<name>/``; failed specs are
recorded and do not stop the grid. The runner finally writes:

- ``ablation_summary.json``: one row per spec with headline metrics;
- ``ablation_summary.md``: the same as a markdown table.

Specs file format (JSON list of dicts):
    [
      {"name": "baseline", "overrides": {}},
      {"name": "no_prediction_head", "overrides": {"pred_weight": 0.0}},
      {"name": "pure_prediction", "overrides": {"pred_weight": 1.0}}
    ]

Allowed override keys are the parameters of
``urban_intervention.representation.trainer.train_representation``:
``embedding_dim``, ``hidden_dims`` (list), ``dropout``, ``temperature``,
``rep_alpha``, ``pred_weight``, ``learning_rate``, ``weight_decay``,
``batch_size``, ``epochs``, ``use_images``, ``max_images_per_grid``,
``image_pooling``, ``conditioning``, ``se_shrinkage``, ``queue_size``,
``learnable_temperature``, ``uncertainty_weighted``, ``transfer_n_seeds``,
``seed``.

Usage:
    python -m urban_intervention.representation.ablation_cli DATA_DIR --specs specs.json
        --output outputs/ablation

Also exposed as the ``urban-run-ablation`` console script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


from .trainer import train_representation


def _collect_row(run_dir: Path) -> dict:
    """Flatten headline metrics from a completed run directory."""
    evaluation = json.loads((run_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
    best_epoch = min(history, key=lambda entry: entry["val_total"]) if history else None
    row: dict = {
        "best_val_loss": best_epoch.get("val_total") if best_epoch else None,
    }
    for split in ("validation", "test"):
        entry = evaluation.get(split, {})
        row[f"{split}_units"] = entry.get("n_units", 0)
        row[f"{split}_nn_corr@k"] = entry.get("retrieval", {}).get("overall", {}).get("nn_corr@k")
        row[f"{split}_baseline_corr"] = (
            entry.get("retrieval", {}).get("overall", {}).get("baseline_corr")
        )
        row[f"{split}_permutation_p"] = entry.get("permutation", {}).get("p_value")
        row[f"{split}_raw_baseline_nn_corr@k"] = (
            entry.get("raw_feature_baseline", {}).get("overall", {}).get("nn_corr@k")
        )
        row[f"{split}_probe_emb_rmse"] = (
            entry.get("probe", {}).get("embeddings", {}).get("rmse_overall")
        )
        row[f"{split}_probe_raw_rmse"] = (
            entry.get("probe", {}).get("raw_features", {}).get("rmse_overall")
        )
    return row


def run_ablation(
    data_dir: Path,
    specs: list[dict],
    output_dir: Path,
    device: str | None = None,
    eval_n_perm: int = 100,
    eval_n_boot: int = 200,
) -> Path:
    """Run the spec grid; returns the path of ``ablation_summary.json``."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    rows: list[dict] = []
    for spec in specs:
        name = str(spec.get("name", "unnamed"))
        overrides = dict(spec.get("overrides", {}))
        if not isinstance(overrides, dict):
            raise ValueError(f"Spec '{name}' overrides must be a dict")
        run_dir = output_dir / name
        try:
            if "hidden_dims" in overrides:
                overrides["hidden_dims"] = tuple(overrides["hidden_dims"])
            ckpt = train_representation(
                model_inputs_dir=data_dir,
                output_dir=run_dir,
                device=device,
                eval_n_perm=eval_n_perm,
                eval_n_boot=eval_n_boot,
                **overrides,
            )
            row = _collect_row(run_dir)
            row.update(
                {"name": name, "checkpoint": str(ckpt), "error": None, "overrides": overrides}
            )
            rows.append(row)
            print(f"[ok] {name}: val nn_corr@k={row['validation_nn_corr@k']}")
        except Exception as exc:  # noqa: BLE001 — one bad spec must not kill the grid
            rows.append(
                {"name": name, "error": str(exc), "overrides": overrides, "checkpoint": None}
            )
            print(f"[fail] {name}: {exc}", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ablation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "specs": len(specs),
                "completed": sum(r.get("error") is None for r in rows),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "ablation_summary.md").write_text(_render_markdown(rows), encoding="utf-8")
    return summary_path


def _render_markdown(rows: list[dict]) -> str:
    columns = [
        "name",
        "best_val_loss",
        "validation_nn_corr@k",
        "validation_baseline_corr",
        "validation_permutation_p",
        "validation_raw_baseline_nn_corr@k",
        "validation_probe_emb_rmse",
        "validation_probe_raw_rmse",
        "test_nn_corr@k",
        "test_units",
        "error",
    ]
    lines = [
        "# Ablation Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append("—" if value is None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path, help="Path to data/model_inputs/<dataset_id>/")
    parser.add_argument(
        "--specs", type=Path, required=True, help="JSON file with the ablation spec list"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/ablation"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-n-perm", type=int, default=100)
    parser.add_argument("--eval-n-boot", type=int, default=200)
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"Error: data directory not found: {args.data_dir}", file=sys.stderr)
        return 1
    if not args.specs.is_file():
        print(f"Error: specs file not found: {args.specs}", file=sys.stderr)
        return 1
    specs = json.loads(args.specs.read_text(encoding="utf-8"))
    if not isinstance(specs, list) or not specs:
        print("Error: specs file must be a non-empty JSON list", file=sys.stderr)
        return 1

    summary_path = run_ablation(
        args.data_dir,
        specs,
        args.output,
        device=args.device,
        eval_n_perm=args.eval_n_perm,
        eval_n_boot=args.eval_n_boot,
    )
    print(f"Ablation summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
