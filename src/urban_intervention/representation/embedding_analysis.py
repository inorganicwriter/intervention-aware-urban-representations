"""Descriptive clustering/manifold analysis of the exported shared z128."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .evaluation import latent_manifold_umap


def analyze_embeddings(path: Path, output: Path, clusters: int = 4, seed: int = 42):
    frame = pd.read_parquet(path)
    columns = [f"emb_{index}" for index in range(128)]
    actual = {c for c in frame if c.startswith("emb_")}
    if actual != set(columns):
        raise ValueError("Shared STC analysis requires exactly emb_0..emb_127")
    z = frame[columns].to_numpy(dtype=np.float64)
    if len(z) < 2 or not np.isfinite(z).all():
        raise ValueError("Embedding analysis requires at least two finite events")
    count = min(max(1, clusters), len(z))
    _, labels = kmeans2(z, count, minit="++", seed=seed)
    manifold = latent_manifold_umap(
        z,
        frame.treatment_order.tolist(),
        frame.city_key.tolist(),
        n_neighbors=min(15, max(2, len(z) - 1)),
        random_state=seed,
    )
    result = {
        "embedding_source": "shared_state_z",
        "dimensions": 128,
        "embedding_columns": columns,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "descriptive_only": True,
        "seed": seed,
        "clusters": [
            {"treatment_order": int(order), "cluster": int(label)}
            for order, label in zip(frame.treatment_order, labels, strict=True)
        ],
        "manifold": manifold,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = analyze_embeddings(args.embeddings, args.output, args.clusters, args.seed)
    print(f"Analyzed z128: {result['manifold']['method']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
