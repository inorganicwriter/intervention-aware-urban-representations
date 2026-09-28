#!/usr/bin/env python3
"""Local memory/time smoke test for the weighted response training loop.

Runs the real model, loss, queue and optimizer for a number of steps on
synthetic tensors shaped like the production batches, and reports CUDA peak
memory plus step times.  This is the hard pre-training check for the pairwise
``(batch x queue x 27)`` design-weight tensors.

Example:

    python scripts/representation/smoke_weighted_training.py \
        --batch-size 256 --queue-size 4096 --steps 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from urban_intervention.representation.dataset import RESPONSE_DIM  # noqa: E402
from urban_intervention.representation.loss import total_loss  # noqa: E402
from urban_intervention.representation.model import ResponseEmbeddingModel  # noqa: E402
from urban_intervention.representation.queue import EmbeddingQueue  # noqa: E402


def _make_batch(
    batch_size: int,
    feature_dim: int,
    device: torch.device,
    generator: torch.Generator,
    start_order: int,
) -> dict[str, torch.Tensor]:
    features = torch.randn(batch_size, feature_dim, generator=generator, device="cpu")
    responses = torch.randn(batch_size, RESPONSE_DIM, generator=generator, device="cpu")
    mask = torch.rand(batch_size, RESPONSE_DIM, generator=generator, device="cpu") > 0.3
    standard_errors = torch.rand(
        batch_size, RESPONSE_DIM, generator=generator, device="cpu"
    ) * 0.3 + 0.02
    standard_errors = standard_errors.masked_fill(~mask, float("nan"))
    weights = torch.rand(batch_size, RESPONSE_DIM, generator=generator, device="cpu") * 4 + 0.2
    weights = weights.masked_fill(~mask, 0.0)
    responses = responses.masked_fill(~mask, float("nan"))
    ids = torch.arange(start_order, start_order + batch_size, dtype=torch.long)
    return {
        "features": features.to(device),
        "responses": responses.to(device),
        "response_mask": mask.to(device),
        "response_se": standard_errors.to(device),
        "response_weight": weights.to(device),
        "ids": ids.to(device),
    }


def _device_description(device: torch.device) -> dict[str, object]:
    description: dict[str, object] = {"device": str(device)}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        description.update(
            {
                "name": properties.name,
                "total_memory_mb": round(properties.total_memory / 1024**2, 1),
                "capability": f"{properties.major}.{properties.minor}",
            }
        )
    return description


def run_smoke(
    *,
    batch_size: int,
    queue_size: int,
    steps: int,
    warmup: int,
    feature_dim: int,
    embedding_dim: int,
    hidden_dims: tuple[int, ...],
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = ResponseEmbeddingModel(
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        embedding_dim=embedding_dim,
        dropout=0.1,
        use_image_encoder=False,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    queue = EmbeddingQueue(
        dim=embedding_dim, capacity=queue_size, device=torch.device("cpu")
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    step_records: list[dict[str, float]] = []
    failure: str | None = None
    start_order = 0
    for step in range(steps):
        batch = _make_batch(batch_size, feature_dim, device, generator, start_order)
        start_order += batch_size
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        try:
            embedding, predictions = model(batch["features"], None, None)
            queue_state = queue.labeled_state()
            losses = total_loss(
                embedding,
                predictions,
                batch["responses"],
                batch["response_mask"],
                batch["response_se"],
                temperature=0.07,
                rep_alpha=0.5,
                pred_weight=0.5,
                se_shrinkage=True,
                queue_embeddings=queue_state["embeddings"] if queue_state else None,
                queue_responses=queue_state["responses"] if queue_state else None,
                queue_response_mask=queue_state["response_mask"] if queue_state else None,
                queue_response_se=queue_state["response_se"] if queue_state else None,
                anchor_ids=batch["ids"],
                queue_ids=queue_state["ids"] if queue_state else None,
                queue_response_weight=(
                    queue_state["response_weight"] if queue_state else None
                ),
                response_weight=batch["response_weight"],
            )
            loss = losses["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            queue.enqueue(
                embedding.detach(),
                batch["responses"],
                batch["response_mask"],
                batch["response_se"],
                batch["ids"],
                response_weight=batch["response_weight"],
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            failure = f"cuda_oom_at_step_{step}"
            step_records.append({"step": step, "status": "oom"})
            torch.cuda.empty_cache()
            break
        elapsed = time.perf_counter() - started
        record: dict[str, float] = {"step": step, "seconds": elapsed}
        if device.type == "cuda":
            record["max_memory_allocated_mb"] = round(
                torch.cuda.max_memory_allocated(device) / 1024**2, 1
            )
            record["max_memory_reserved_mb"] = round(
                torch.cuda.max_memory_reserved(device) / 1024**2, 1
            )
        step_records.append(record)

    measured = [record for record in step_records if "seconds" in record]
    measured = measured[warmup:] if len(measured) > warmup else measured
    seconds = [float(record["seconds"]) for record in measured]
    result: dict[str, object] = {
        "config": {
            "batch_size": batch_size,
            "queue_size": queue_size,
            "steps": steps,
            "warmup": warmup,
            "feature_dim": feature_dim,
            "embedding_dim": embedding_dim,
            "hidden_dims": list(hidden_dims),
            "learning_rate": learning_rate,
            "seed": seed,
        },
        "device": _device_description(device),
        "status": failure or ("completed" if len(step_records) == steps else "incomplete"),
        "steps_recorded": len(step_records),
        "step_seconds": {
            "mean": round(sum(seconds) / len(seconds), 4) if seconds else None,
            "median": round(sorted(seconds)[len(seconds) // 2], 4) if seconds else None,
            "max": round(max(seconds), 4) if seconds else None,
        },
    }
    if device.type == "cuda":
        result["step_memory_mb"] = {
            "max_allocated": max(
                (record.get("max_memory_allocated_mb", 0.0) for record in step_records),
                default=0.0,
            ),
            "max_reserved": max(
                (record.get("max_memory_reserved_mb", 0.0) for record in step_records),
                default=0.0,
            ),
        }
    if any("seconds" in record for record in step_records[1:]):
        largest_jump = 0.0
        previous = None
        for record in step_records:
            if "seconds" not in record:
                continue
            jump = record["seconds"] - previous if previous is not None else 0.0
            largest_jump = max(largest_jump, jump)
            previous = record["seconds"]
        result["largest_step_time_jump_seconds"] = round(largest_jump, 4)
    result["per_step"] = step_records
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--json", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run_smoke(
        batch_size=args.batch_size,
        queue_size=args.queue_size,
        steps=args.steps,
        warmup=args.warmup,
        feature_dim=args.feature_dim,
        embedding_dim=args.embedding_dim,
        hidden_dims=tuple(args.hidden_dims),
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "per_step"}, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Report: {args.json}")
    return 1 if result["status"] != "completed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
