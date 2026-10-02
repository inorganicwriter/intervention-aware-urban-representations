"""Diagnose the representation-learning evaluation chain on a built dataset.

Run from the repository root on the server, for example::

    conda run -n mit python scripts/analysis/representation_diagnostics.py \
        --dataset data/active/model_inputs/viirs-mg-design \
        --output-directory outputs/representation_diagnostics/viirs-mg-design

Add a checkpoint to also profile the trained embedding geometry::

    conda run -n mit python scripts/analysis/representation_diagnostics.py \
        --dataset data/active/model_inputs/poi-mg-design \
        --checkpoint outputs/representation/poi-var-default/seed_1/best_model.pt \
        --split test \
        --output-directory outputs/representation_diagnostics/poi-var-default-seed1

For a station-conditioned checkpoint also pass
``--station-attributes outputs/causal_labels/station_attributes/station_attributes_treated.parquet``.

The script only reads existing artifacts; it never retrains or writes into the
dataset or checkpoint directories.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.representation.dataset import (  # noqa: E402
    RESPONSE_OFFSETS,
    RepresentationDataset,
)
from urban_intervention.representation.evaluation import (  # noqa: E402
    cosine_similarity,
    response_similarity_between,
    retrieval_metrics,
)
from urban_intervention.representation.model import ResponseEmbeddingModel  # noqa: E402
from urban_intervention.representation.trainer import _collate_fn  # noqa: E402
from urban_intervention.representation.transfer import per_city_retrieval  # noqa: E402


def _records_to_arrays(ds: RepresentationDataset) -> dict[str, np.ndarray]:
    records = ds._records  # diagnostics-only access to the label tensors
    responses = np.stack(records["response"].to_numpy()).astype(np.float64)
    weights = np.stack(records["response_weight"].to_numpy()).astype(np.float64)
    orders = records.index.to_numpy()
    city = records["city_key"].to_numpy()
    return {
        "response": responses,
        "weight": weights,
        "order": orders,
        "city": city,
    }


def _family_columns(family: str) -> list[int]:
    start, end, _ = RESPONSE_OFFSETS[family]
    return list(range(start, end))


def _outcome_groups(family: str) -> dict[str, list[int]]:
    """Outcome -> relative column indices inside the family segment."""
    start, _, columns = RESPONSE_OFFSETS[family]
    del start
    groups: dict[str, list[int]] = {}
    for offset, name in enumerate(columns):
        outcome = name.split("__t")[0]
        groups.setdefault(outcome, []).append(offset)
    return groups


def _support_histogram(mask: np.ndarray) -> dict[str, float]:
    """Exact pairwise common-support histogram for one family segment."""
    n, h = mask.shape
    m = mask.astype(np.float32)
    common = m @ m.T
    upper = common[np.triu_indices(n, k=1)]
    total = max(len(upper), 1)
    hist: dict[str, float] = {str(k): float((upper == k).sum()) for k in range(h + 1)}
    hist["share_ge_4"] = float((upper >= 4).sum()) / total
    hist["share_eq_2"] = float((upper == 2).sum()) / total
    return hist


def label_diagnostics(ds: RepresentationDataset, split: str | None) -> dict[str, object]:
    arrays = _records_to_arrays(ds)
    responses, weights = arrays["response"], arrays["weight"]
    mask_all = np.isfinite(responses)
    if split is not None:
        split_mask = ds._records["split"].to_numpy() == split
        responses = responses[split_mask]
        weights = weights[split_mask]
        mask_all = mask_all[split_mask]
    result: dict[str, object] = {"n_grids": int(responses.shape[0]), "families": {}}
    families: dict[str, object] = {}
    for family in RESPONSE_OFFSETS:
        cols = _family_columns(family)
        r = responses[:, cols]
        w = weights[:, cols]
        valid = mask_all[:, cols]
        per_grid_support = valid.sum(axis=1)
        grids_with_any = int((per_grid_support > 0).sum())
        support_dist = Counter(per_grid_support.tolist())
        constant_grids = 0
        eligible_grids = 0
        ratios: list[float] = []
        for row in range(valid.shape[0]):
            row_valid = valid[row]
            if row_valid.sum() < 2:
                continue
            eligible_grids += 1
            row_w = w[row][row_valid]
            if float(row_w.max() - row_w.min()) < 1e-9:
                constant_grids += 1
            vals = r[row][row_valid]
            sd = float(vals.std(ddof=1))
            level = float(abs(vals.mean()))
            if sd > 0:
                ratios.append(level / sd)
        ratios_arr = np.array(ratios, dtype=np.float64) if ratios else np.array([])
        outcomes = {}
        for outcome, group in _outcome_groups(family).items():
            gmask = valid[:, group]
            per_grid = gmask.sum(axis=1)
            outcomes[outcome] = {
                "grids_any": int((per_grid > 0).sum()),
                "support_distribution": {str(int(k)): int(v) for k, v in Counter(per_grid.tolist()).items()},
            }
        families[family] = {
            "grids_any": grids_with_any,
            "support_distribution": {str(int(k)): int(v) for k, v in support_dist.items()},
            "pair_support": _support_histogram(valid),
            "constant_weight_grids": constant_grids,
            "eligible_grids": eligible_grids,
            "constant_weight_share": round(constant_grids / max(eligible_grids, 1), 4),
            "level_over_sd_median": float(np.median(ratios_arr)) if ratios_arr.size else None,
            "level_over_sd_p90": float(np.percentile(ratios_arr, 90)) if ratios_arr.size else None,
            "level_over_sd_share_gt100": float((ratios_arr > 100).mean()) if ratios_arr.size else None,
            "outcomes": outcomes,
        }
    result["families"] = families
    return result


def precision_diagnostics(ds: RepresentationDataset, n_pairs: int = 2000, seed: int = 0) -> dict[str, object]:
    arrays = _records_to_arrays(ds)
    responses64 = arrays["response"]
    mask = np.isfinite(responses64)
    rng = np.random.default_rng(seed)
    out: dict[str, object] = {}
    for family in RESPONSE_OFFSETS:
        cols = _family_columns(family)
        eligible = np.where(mask[:, cols].sum(axis=1) >= 2)[0]
        if len(eligible) < 4:
            out[family] = {"note": "not enough grids"}
            continue
        i = rng.choice(eligible, size=min(n_pairs, len(eligible) * (len(eligible) - 1) // 2), replace=True)
        j = rng.choice(eligible, size=len(i), replace=True)
        keep = i != j
        i, j = i[keep], j[keep]
        r32 = torch.from_numpy(responses64[:, cols]).float()
        m32 = torch.from_numpy(mask[:, cols])
        diffs = []
        mismatch = 0
        chunk = 256

        for start in range(0, len(i), chunk):
            ia = torch.from_numpy(i[start : start + chunk])
            jb = torch.from_numpy(j[start : start + chunk])
            s32, v32 = response_similarity_between(r32[ia], m32[ia], r32[jb], m32[jb])
            s64, v64 = response_similarity_between(
                r32[ia].double(), m32[ia], r32[jb].double(), m32[jb]
            )
            diag32 = torch.diagonal(s32)
            diag64 = torch.diagonal(s64)
            v32d = torch.diagonal(v32)
            v64d = torch.diagonal(v64)
            both = v64d
            if both.any():
                diffs.append((diag32.double() - diag64)[both].abs().cpu().numpy())
            mismatch += int((v64d & ~v32d).sum())
        diff = np.concatenate(diffs) if diffs else np.array([0.0])
        out[family] = {
            "pairs": int(len(i)),
            "float64_valid": int(sum(d.shape[0] for d in diffs)),
            "float32_invalid_but_float64_valid": mismatch,
            "mean_abs_similarity_error": float(diff.mean()),
            "max_abs_similarity_error": float(diff.max()),
        }
    return out


def _load_model(checkpoint_path: Path, ds: RepresentationDataset) -> ResponseEmbeddingModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    conditioning = checkpoint.get("conditioning") or None
    conditioning_schema = str(
        checkpoint.get("conditioning_schema") or ("v1" if conditioning else "v2")
    )
    model = ResponseEmbeddingModel(
        feature_dim=ds.feature_dim(),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        dropout=float(checkpoint.get("dropout", 0.1)),
        use_image_encoder=False,
        image_pooling=checkpoint.get("image_pooling", "max"),
        conditioning=conditioning,
        conditioning_schema=conditioning_schema,
        learnable_temperature=any(key.startswith("logit_scale") for key in state),
        uncertainty_weighted=any(key.startswith("log_var") for key in state),
    )
    model.load_state_dict(state)
    model.eval()
    return model


class _EmbeddingPool(TypedDict):
    embeddings: torch.Tensor
    responses: torch.Tensor
    masks: torch.Tensor
    features: torch.Tensor
    cities: list[str]


def _embed_pool(
    ds: RepresentationDataset, model: ResponseEmbeddingModel, batch_size: int = 256
) -> _EmbeddingPool:
    from torch.utils.data import DataLoader

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: _collate_fn(batch, ds, False),
    )
    embeddings: list[torch.Tensor] = []
    responses: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    cities: list[str] = []
    with torch.no_grad():
        for batch in loader:
            embedding, _ = model(
                batch["features"],
                None,
                None,
                conditioning_tokens=batch.get("conditioning_tokens"),
                station_tokens=batch.get("station_tokens"),
            )
            embeddings.append(embedding)
            responses.append(batch["responses"])
            masks.append(batch["response_mask"])
            features.append(batch["features"])
            orders = batch["treatment_order"].tolist()
            cities.extend(str(ds._records.loc[order]["city_key"]) for order in orders)
    return {
        "embeddings": torch.cat(embeddings),
        "responses": torch.cat(responses),
        "masks": torch.cat(masks),
        "features": torch.cat(features),
        "cities": cities,
    }


def embedding_diagnostics(
    ds: RepresentationDataset,
    checkpoint: Path,
    split: str,
    k: int = 5,
) -> dict[str, object]:
    model = _load_model(checkpoint, ds)
    pool = _embed_pool(ds, model)
    emb = pool["embeddings"]
    resp = pool["responses"]
    mask = pool["masks"]
    raw = pool["features"]
    cities = pool["cities"]
    n = emb.shape[0]
    cos = cosine_similarity(emb)
    off = ~torch.eye(n, dtype=torch.bool)
    centered = emb - emb.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered.double())
    energy = singular**2
    participation = float((energy.sum() ** 2) / (energy**2).sum())
    cumulative = torch.cumsum(energy / energy.sum(), dim=0)
    dims_90 = int(torch.searchsorted(cumulative, torch.tensor(0.9)) + 1)
    raw_cos = cosine_similarity(raw)
    result: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "split": split,
        "n_units": int(n),
        "mean_pairwise_cosine": float(cos[off].mean()),
        "std_pairwise_cosine": float(cos[off].std()),
        "raw_mean_pairwise_cosine": float(raw_cos[off].mean()),
        "effective_rank_participation": participation,
        "effective_dims_90pct": dims_90,
        "retrieval": retrieval_metrics(emb, resp, mask, k=k)["overall"],
        "raw_retrieval": retrieval_metrics(raw, resp, mask, k=k)["overall"],
    }
    raw_city = per_city_retrieval(raw, resp, mask, cities, k=k)
    model_city = per_city_retrieval(emb, resp, mask, cities, k=k)
    result["per_city"] = {
        city: {
            "n_units": (model_city[city].get("overall") or {}).get("n_units"),
            "model_nn_corr": (model_city[city].get("overall") or {}).get("nn_corr@k"),
            "raw_nn_corr": (raw_city.get(city, {}).get("overall") or {}).get("nn_corr@k"),
            "baseline_corr": (model_city[city].get("overall") or {}).get("baseline_corr"),
        }
        for city in sorted(model_city)
    }
    per_outcome: dict[str, object] = {}
    for family in RESPONSE_OFFSETS:
        family_start = RESPONSE_OFFSETS[family][0]
        for outcome, rel_cols in _outcome_groups(family).items():
            cols = [family_start + rel for rel in rel_cols]
            if mask[:, cols].sum() < 2 or mask[:, cols].sum(dim=1).lt(2).all():
                continue
            model_block = retrieval_metrics(emb, resp[:, cols], mask[:, cols], k=k)["overall"]
            raw_block = retrieval_metrics(raw, resp[:, cols], mask[:, cols], k=k)["overall"]
            per_outcome[f"{family}/{outcome}"] = {
                "n_units": model_block.get("n_units"),
                "model_nn_corr": model_block.get("nn_corr@k"),
                "raw_nn_corr": raw_block.get("nn_corr@k"),
                "baseline_corr": model_block.get("baseline_corr"),
            }
    result["per_outcome"] = per_outcome
    station_embedding = getattr(model, "station_embedding", None)
    if station_embedding is not None:
        result["station_embedding_norms"] = {
            "mean": float(station_embedding.weight.norm(dim=1).mean()),
            "min": float(station_embedding.weight.norm(dim=1).min()),
            "max": float(station_embedding.weight.norm(dim=1).max()),
        }
    return result


def station_token_distribution(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    import pandas as pd

    frame = pd.read_parquet(path)
    required = {
        "treatment_order",
        "is_transfer_at_opening",
        "is_new_line_opening",
        "is_extension_opening",
        "is_terminal_at_opening",
    }
    if not required.issubset(frame.columns):
        return {"note": "station attributes lack expected columns"}
    treated = frame.loc[frame["treatment_order"].notna()].copy()
    flags: dict[str, object] = {}
    for column in ("is_terminal_at_opening",):
        flags[f"{column}_missing"] = int(
            pd.to_numeric(treated[column], errors="coerce").isna().sum()
        )
    flags["n_treated"] = int(len(treated))
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--station-attributes", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    label_ds = RepresentationDataset(args.dataset, split=None, load_images=False)
    report: dict[str, object] = {
        "dataset": str(args.dataset),
        "split": args.split,
        "labels": label_diagnostics(label_ds, split=None),
        "labels_split": label_diagnostics(label_ds, split=args.split),
        "precision": precision_diagnostics(label_ds),
        "station_tokens": station_token_distribution(args.station_attributes),
    }
    if args.checkpoint is not None:
        checkpoint_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        checkpoint_conditioning = checkpoint_meta.get("conditioning") or None
        if args.station_attributes is None and "station" in str(checkpoint_conditioning or ""):
            print("Checkpoint is station-conditioned; pass --station-attributes", file=sys.stderr)
            return 2
        conditioning_schema = str(
            checkpoint_meta.get("conditioning_schema")
            or ("v1" if checkpoint_conditioning else "v2")
        )
        split_ds = RepresentationDataset(
            args.dataset,
            split=args.split,
            only_training_mask=True,
            load_images=False,
            station_attributes_path=args.station_attributes,
            station_schema=conditioning_schema,
        )
        report["embedding"] = embedding_diagnostics(split_ds, args.checkpoint, args.split, k=args.k)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    out_path = args.output_directory / "diagnostics.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False)[:4000])
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
