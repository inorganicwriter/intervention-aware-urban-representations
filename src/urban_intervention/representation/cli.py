#!/usr/bin/env python
"""Train an intervention-conditioned urban representation model.

Usage:
    python -m urban_intervention.representation.cli DATA_DIR --output OUTPUT_DIR [options]
    python -m urban_intervention.representation.cli DATA_DIR --output OUTPUT_DIR --seeds 1 2 3

Each run writes ``training_history.json``, ``test_metrics.json`` and
``evaluation_report.json`` (retrieval metrics + bootstrap CI + permutation
p-value + raw-feature baseline + linear-probe transfer) into its output
directory. With ``--seeds``, one run per seed is written to
``<output>/seed_<n>/`` and a ``seed_summary.json`` with mean/std across seeds
is written to ``<output>/``.

Also exposed as the ``urban-train-representation`` console script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Windows: torch and numpy can both load libiomp5md.dll, crashing the process
# on import.  Allow the duplicate OpenMP runtime (safe for this workload; the
# alternative is a hard crash before any training starts).
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .trainer import train_representation


def _mean_std(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
    return {"mean": round(mean, 6), "std": round(variance**0.5, 6)}


def aggregate_seed_metrics(summaries: list[dict]) -> dict[str, object]:
    """Mean/std of every numeric test metric across seeds."""
    numeric_keys: set[str] = set()
    for summary in summaries:
        numeric_keys.update(
            key
            for key, value in summary.get("test_metrics", {}).items()
            if isinstance(value, (int, float))
        )
    aggregated: dict[str, object] = {
        "seeds": [summary["seed"] for summary in summaries],
        "metrics": {},
    }
    for key in sorted(numeric_keys):
        values = [float(summary["test_metrics"][key]) for summary in summaries]
        metrics_map = aggregated["metrics"]
        assert isinstance(metrics_map, dict)
        metrics_map[key] = _mean_std(values)
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Train urban representation model")
    parser.add_argument("data_dir", type=Path, help="Path to data/model_inputs/<dataset_id>/")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/representation"),
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--training-mode", choices=("joint", "viirs_only", "poi_only"),
                        default=None, help="Shared STC mode; auto-selected for joint STC bundles")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Global gradient clip threshold; 0 disables clipping")
    parser.add_argument("--gradient-diagnostics-interval", type=int, default=1,
                        help="Record task gradients every N batches; 0 disables diagnostics")
    parser.add_argument("--prediction-inputs", choices=("full", "tc_only", "z_only"), default="full")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128], help="Legacy tabular model MLP widths; ignored by STC")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--rep-alpha", type=float, default=0.5, help="Legacy contrastive_legacy only; ignored by multimodal_causal"
    )
    parser.add_argument(
        "--pred-weight",
        type=float,
        default=0.5,
        help="Legacy contrastive_legacy only; use loss-anchor-weight/loss-align-weight for multimodal_causal",
    )
    parser.add_argument(
        "--loss-type",
        choices=("multimodal_causal", "contrastive_legacy"),
        default="multimodal_causal",
        help="Loss objective: multimodal_causal (default, Soft Causal Anchoring Huber + InfoNCE) "
        "or contrastive_legacy (historical InfoNCE + MSE baseline)",
    )
    parser.add_argument(
        "--loss-anchor-weight",
        type=float,
        default=0.85,
        help="Weight for causal trajectory Huber anchoring loss (default: 0.85)",
    )
    parser.add_argument(
        "--loss-align-weight",
        type=float,
        default=0.15,
        help="Weight for response alignment InfoNCE loss (default: 0.15)",
    )
    parser.add_argument(
        "--loss-huber-delta",
        type=float,
        default=1.0,
        help="Huber delta threshold for outlier resistance (default: 1.0)",
    )
    parser.add_argument(
        "--allow-satellite-proxy",
        action="store_true",
        default=False,
        help="Allow contemporary/post-treatment satellite proxy embeddings (default: False, strict pre-treatment only)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, dest="learning_rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--no-se-shrinkage",
        action="store_false",
        dest="se_shrinkage",
        default=True,
        help="Disable SE-based response-similarity shrinkage (label-reliability "
        "weighting; on by default)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=0,
        help="Response-aware FIFO queue of cross-batch contrastive pairs "
        "(0 = disabled; e.g. 4096)",
    )
    parser.add_argument(
        "--learnable-temperature",
        action="store_true",
        default=False,
        help="Learn the InfoNCE temperature (CLIP-style logit scale, init 1/0.07)",
    )
    parser.add_argument(
        "--uncertainty-weighted",
        action="store_true",
        default=False,
        help="Learn rep/pred task weights via uncertainty weighting "
        "(Kendall et al. 2018) instead of the fixed pred_weight",
    )
    parser.add_argument(
        "--use-images",
        action="store_true",
        help="Enable DINOv2 image encoder (requires streetview data)",
    )
    parser.add_argument(
        "--use-satellite",
        action="store_true",
        help="Legacy: AlphaEarth encoder; STC: independent 6-year satellite sequence encoder",
    )
    parser.add_argument(
        "--use-multimodal",
        action="store_true",
        default=False,
        help="Enable full multimodal mode: sets --use-images, --use-satellite, and --use-cross-view-synergy",
    )
    parser.add_argument(
        "--require-multimodal",
        action="store_true",
        default=False,
        help="Enforce mandatory multimodal inputs: require both valid pre-treatment satellite and street-view images for training units",
    )
    parser.add_argument(
        "--require-satellite",
        action="store_true",
        default=False,
        help="Require valid pre-treatment satellite embeddings for training units",
    )
    parser.add_argument(
        "--require-streetview",
        action="store_true",
        default=False,
        help="Require valid pre-treatment street-view images for training units",
    )
    parser.add_argument(
        "--satellite-embeddings",
        type=Path,
        default=None,
        help="Path to AlphaEarth satellite embeddings parquet file",
    )
    parser.add_argument(
        "--use-cross-view-synergy",
        action="store_true",
        help="Legacy VIGOR cross-view module only; STC uses StateEncoder fusion",
    )
    parser.add_argument(
        "--synergy-heads",
        type=int,
        default=4,
        help="Number of attention heads in cross-view synergy module (default: 4)",
    )
    parser.add_argument(
        "--image-pooling",
        choices=("max", "mean", "meanmax"),
        default="max",
        help="Legacy image pooling only; STC uses time-aware attention",
    )
    parser.add_argument(
        "--conditioning",
        choices=("none", "opening_year", "station", "opening_year_station"),
        default="none",
        help="Explicit intervention conditioning token (default: none = implicit "
        "response-aligned conditioning); station modes require "
        "--station-attributes",
    )
    parser.add_argument(
        "--station-attributes",
        type=Path,
        default=None,
        help="Station attribute parquet (outputs/causal_labels/station_attributes/"
        "station_attributes.parquet) for station conditioning tokens",
    )
    parser.add_argument("--max-images", type=int, default=4, help="Max streetview images per grid")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run one full training+eval per seed into seed_<n>/ subdirs "
        "and write a seed_summary.json with mean/std",
    )
    parser.add_argument(
        "--eval-k", type=int, default=5, help="Neighbour count for nn_corr@k retrieval metrics"
    )
    parser.add_argument(
        "--eval-n-perm",
        type=int,
        default=100,
        help="Permutations for the response-shuffle chance test",
    )
    parser.add_argument(
        "--eval-n-boot",
        type=int,
        default=200,
        help="Bootstrap resamples for nn_corr@k confidence intervals",
    )
    parser.add_argument(
        "--probe-ridge",
        type=float,
        default=1.0,
        help="Ridge penalty for the linear-probe transfer metric",
    )
    parser.add_argument(
        "--probe-min-obs",
        type=int,
        default=16,
        help="Minimum train observations per response cell for the probe",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_false",
        dest="baselines",
        default=True,
        help="Skip chance/appearance-only baselines (random projection, PCA, DINOv2)",
    )
    parser.add_argument(
        "--no-transfer",
        action="store_false",
        dest="transfer",
        default=True,
        help="Skip cross-city transfer evaluation (per-city retrieval, few-shot probes)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Write PCA-2D embedding scatter plots (needs matplotlib)",
    )

    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"Error: data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    args.data_dir.resolve()
    if args.use_multimodal:
        args.use_images = True
        args.use_satellite = True
        args.use_cross_view_synergy = True

    common = dict(
        model_inputs_dir=args.data_dir,
        training_mode=args.training_mode,
        max_grad_norm=args.max_grad_norm or None,
        gradient_diagnostics_interval=args.gradient_diagnostics_interval,
        prediction_inputs=args.prediction_inputs,
        embedding_dim=args.embedding_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        temperature=args.temperature,
        rep_alpha=args.rep_alpha,
        pred_weight=args.pred_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_images=args.use_images,
        max_images_per_grid=args.max_images,
        image_pooling=args.image_pooling,
        use_satellite=args.use_satellite,
        satellite_embeddings_path=args.satellite_embeddings,
        use_cross_view_synergy=args.use_cross_view_synergy,
        synergy_heads=args.synergy_heads,
        conditioning=args.conditioning if args.conditioning != "none" else None,
        station_attributes_path=args.station_attributes,
        se_shrinkage=args.se_shrinkage,
        queue_size=args.queue_size,
        learnable_temperature=args.learnable_temperature,
        uncertainty_weighted=args.uncertainty_weighted,
        device=args.device,
        eval_k=args.eval_k,
        eval_n_perm=args.eval_n_perm,
        eval_n_boot=args.eval_n_boot,
        probe_ridge=args.probe_ridge,
        probe_min_obs=args.probe_min_obs,
        run_baselines=args.baselines,
        run_transfer=args.transfer,
        visualize=args.visualize,
        loss_type=args.loss_type,
        loss_anchor_weight=args.loss_anchor_weight,
        loss_align_weight=args.loss_align_weight,
        loss_huber_delta=args.loss_huber_delta,
        allow_satellite_proxy=args.allow_satellite_proxy,
        require_multimodal=args.require_multimodal,
        require_satellite=args.require_satellite,
        require_streetview=args.require_streetview,
    )

    try:
        if args.seeds:
            summaries: list[dict] = []
            for seed in args.seeds:
                run_dir = args.output / f"seed_{seed}"
                ckpt = train_representation(output_dir=run_dir, seed=seed, **common)
                test_metrics = json.loads(
                    (run_dir / "test_metrics.json").read_text(encoding="utf-8")
                )
                summaries.append(
                    {"seed": seed, "checkpoint": str(ckpt), "test_metrics": test_metrics}
                )
                print(f"Seed {seed} done. Best checkpoint: {ckpt}")
                print(f"  Evaluation report: {run_dir / 'evaluation_report.json'}")
            summary = aggregate_seed_metrics(summaries)
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "seed_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Multi-seed summary: {args.output / 'seed_summary.json'}")
        else:
            ckpt = train_representation(output_dir=args.output, seed=args.seed, **common)
            print(f"Training complete. Best checkpoint: {ckpt}")
            print(f"Metrics: {args.output / 'test_metrics.json'}")
            print(f"Evaluation report: {args.output / 'evaluation_report.json'}")
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
