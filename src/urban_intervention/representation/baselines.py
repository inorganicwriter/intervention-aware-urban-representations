"""Chance-level and appearance-only baselines for representation evaluation.

The trained model is only meaningful if it beats these null/weak baselines on
the held-out (unseen-city) pool:

- ``random_projection_baseline``: a fixed random Gaussian projection of the raw
  pre-treatment features.  No learning, no response signal; it is the pure
  chance level for retrieval quality given the feature geometry.
- ``pca_projection_baseline``: centred SVD projection of the raw features.
  Unsupervised ("appearance-only") geometry: it uses no response labels, so any
  response similarity it captures is a property of the input features alone.
- ``dinov2_image_baseline``: frozen DINOv2 backbone pooled over street-view
  images (mean over valid images per grid).  The strongest available
  appearance-only baseline: no response supervision, but rich visual features.

Every baseline returns the same metric structure as the trained-model entries
in ``evaluation_report.json`` (``retrieval`` + ``bootstrap_ci`` +
``permutation``) so they can be compared row for row.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as functional

from .evaluation import (
    bootstrap_ci,
    cosine_similarity,
    permutation_test,
    response_similarity_with_validity,
    retrieval_metrics,
)

# Number of components for the projection baselines.  The raw feature space is
# typically ~30-60 dimensions; a 64-dim projection preserves that geometry.
DEFAULT_PROJECTION_DIM = 64


def baseline_metric_entry(
    embeddings: torch.Tensor,
    responses: torch.Tensor,
    masks: torch.Tensor,
    k: int = 5,
    n_perm: int = 100,
    n_boot: int = 200,
    seed: int = 42,
    max_perm_units: int = 512,
) -> dict[str, object]:
    """Standard metric block shared by all baselines (mirrors model entries)."""
    n = embeddings.shape[0]
    if n < 2:
        return {"note": "pool smaller than 2 units", "n_units": n}
    emb_cos = cosine_similarity(embeddings)
    sim_resp, valid_pairs = response_similarity_with_validity(responses, masks)
    return {
        "n_units": int(n),
        "retrieval": retrieval_metrics(embeddings, responses, masks, k=k),
        "bootstrap_ci": bootstrap_ci(
            emb_cos,
            sim_resp,
            k=k,
            n_boot=n_boot,
            seed=seed,
            valid_pairs=valid_pairs,
        ),
        "permutation": permutation_test(
            emb_cos, responses, masks, k=k, n_perm=n_perm, seed=seed, max_units=max_perm_units
        ),
    }


def random_projection_baseline(
    features: torch.Tensor,
    responses: torch.Tensor,
    masks: torch.Tensor,
    k: int = 5,
    n_perm: int = 100,
    n_boot: int = 200,
    seed: int = 42,
    projection_seed: int = 2026,
    projection_dim: int = DEFAULT_PROJECTION_DIM,
) -> dict[str, object]:
    """Random Gaussian projection of the raw features (chance level)."""
    n, dim = features.shape
    if n < 2:
        return {"note": "pool smaller than 2 units", "n_units": n}
    rng = np.random.RandomState(projection_seed)
    matrix = torch.from_numpy(rng.randn(dim, projection_dim).astype(np.float32))
    matrix = matrix / matrix.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
    embeddings = functional.normalize(features @ matrix, p=2, dim=1)
    entry = baseline_metric_entry(embeddings, responses, masks, k=k, n_perm=n_perm, n_boot=n_boot, seed=seed)
    entry["projection"] = {"seed": projection_seed, "dim": projection_dim}
    return entry


def pca_projection_baseline(
    features: torch.Tensor,
    responses: torch.Tensor,
    masks: torch.Tensor,
    k: int = 5,
    n_perm: int = 100,
    n_boot: int = 200,
    seed: int = 42,
    components: int = DEFAULT_PROJECTION_DIM,
) -> dict[str, object]:
    """Centred SVD projection of the raw features (appearance-only geometry).

    The SVD is fitted on this pool itself; it is unsupervised and never touches
    response labels, so it stays an honest no-response baseline.
    """
    n = features.shape[0]
    if n < 2:
        return {"note": "pool smaller than 2 units", "n_units": n}
    centered = features - features.mean(dim=0, keepdim=True)
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    basis = vt[: min(components, vt.shape[0])].T
    embeddings = functional.normalize(centered @ basis, p=2, dim=1)
    entry = baseline_metric_entry(embeddings, responses, masks, k=k, n_perm=n_perm, n_boot=n_boot, seed=seed)
    entry["components"] = int(basis.shape[1])
    return entry


def dinov2_image_baseline(
    batches: Iterable[dict[str, object]],
    device: torch.device,
    k: int = 5,
    n_perm: int = 100,
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, object]:
    """Frozen DINOv2 backbone pooled over street-view images per grid.

    ``batches`` must yield the same dicts as ``collate_samples`` with
    ``use_images=True`` (``images``/``image_mask`` keys).  The backbone is the
    same frozen ``_ensure_dinov2`` checkpoint used by the image encoder, but
    without the trainable projection head.
    """
    try:
        from .encoder import _ensure_dinov2
    except (ImportError, OSError, RuntimeError) as exc:
        return {"note": f"dinov2_unavailable: {exc}"}
    pooled: list[torch.Tensor] = []
    responses: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    seen_batch = False
    for batch in batches:
        images = batch.get("images")
        image_mask = batch.get("image_mask")
        batch_responses = batch.get("responses")
        batch_masks = batch.get("response_mask")
        if not isinstance(images, torch.Tensor) or not isinstance(image_mask, torch.Tensor):
            continue
        if not isinstance(batch_responses, torch.Tensor) or not isinstance(batch_masks, torch.Tensor):
            continue
        seen_batch = True
        batch_size, max_images, c, h, w = images.shape
        flat = images.reshape(batch_size * max_images, c, h, w).to(device)
        mask_2d = image_mask.reshape(batch_size, max_images).to(device)
        with torch.no_grad():
            features = _ensure_dinov2()(flat)
        features = features.reshape(batch_size, max_images, -1)
        features = features.masked_fill(~mask_2d.unsqueeze(-1), float("nan"))
        valid = mask_2d.sum(dim=1)
        emb = features.nanmean(dim=1)
        # NaN * 0 is NaN: an all-masked row must be replaced with zeros
        # explicitly, not masked by multiplication.
        emb = torch.where(valid.gt(0).unsqueeze(-1), emb, torch.zeros_like(emb))
        pooled.append(emb.cpu())
        responses.append(batch_responses.cpu())
        masks.append(batch_masks.cpu())
    if not seen_batch:
        return {"note": "no_image_batches_available"}
    embeddings = functional.normalize(torch.cat(pooled, dim=0), p=2, dim=1)
    return baseline_metric_entry(
        embeddings,
        torch.cat(responses, dim=0),
        torch.cat(masks, dim=0),
        k=k,
        n_perm=n_perm,
        n_boot=n_boot,
        seed=seed,
    )


def run_baselines(
    features: torch.Tensor,
    responses: torch.Tensor,
    masks: torch.Tensor,
    image_batches: Iterable[dict[str, object]] | None = None,
    device: torch.device | None = None,
    k: int = 5,
    n_perm: int = 100,
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, object]:
    """Compute all baselines on one pool; returns a report section.

    ``image_batches`` is optional: the DINOv2 baseline is only included when
    image batches (with actual street-view tensors) are supplied.
    """
    baselines: dict[str, object] = {
        "random_projection": random_projection_baseline(
            features, responses, masks, k=k, n_perm=n_perm, n_boot=n_boot, seed=seed
        ),
        "pca_features": pca_projection_baseline(
            features, responses, masks, k=k, n_perm=n_perm, n_boot=n_boot, seed=seed
        ),
    }
    if image_batches is not None:
        baselines["dinov2_images"] = dinov2_image_baseline(
            image_batches,
            device or torch.device("cpu"),
            k=k,
            n_perm=n_perm,
            n_boot=n_boot,
            seed=seed,
        )
    return baselines


def headline_metric(entry: dict[str, object]) -> float | None:
    """nn_corr@k from a baseline/model entry, or None when unavailable."""
    retrieval = entry.get("retrieval")
    if not isinstance(retrieval, dict):
        return None
    overall = retrieval.get("overall")
    if not isinstance(overall, dict):
        return None
    value = overall.get("nn_corr@k")
    return float(value) if isinstance(value, (int, float)) else None


class AppearanceAutoencoder(torch.nn.Module):
    """Same-encoder-shape appearance-only baseline.

    A mirror MLP autoencoder trained to reconstruct the raw pre-treatment
    features.  It never sees response labels, so any response structure in its
    embeddings is a property of the input features alone — the strongest
    *trainable* appearance-only comparison for the supervised model.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        embedding_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim

        encoder_layers: list[torch.nn.Module] = []
        in_dim = feature_dim
        for h_dim in hidden_dims:
            encoder_layers.append(torch.nn.Linear(in_dim, h_dim))
            encoder_layers.append(torch.nn.ReLU())
            if dropout > 0:
                encoder_layers.append(torch.nn.Dropout(dropout))
            in_dim = h_dim
        encoder_layers.append(torch.nn.Linear(in_dim, embedding_dim))
        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers: list[torch.nn.Module] = []
        in_dim = embedding_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(torch.nn.Linear(in_dim, h_dim))
            decoder_layers.append(torch.nn.ReLU())
            in_dim = h_dim
        decoder_layers.append(torch.nn.Linear(in_dim, feature_dim))
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(features)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction


def appearance_autoencoder_baseline(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    responses: torch.Tensor,
    masks: torch.Tensor,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_dims: tuple[int, ...] = (64, 32),
    embedding_dim: int = 32,
    device: torch.device | None = None,
    k: int = 5,
    n_perm: int = 100,
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, object]:
    """Train an appearance-only autoencoder on the train pool, evaluate on test.

    The autoencoder is fitted on ``train_features`` only (no response labels),
    then its frozen encoder embeds ``test_features`` for the standard metric
    block.  Training is deterministic via ``seed``.
    """
    n_train = train_features.shape[0]
    if n_train < 8 or test_features.shape[0] < 2:
        return {"note": "train or test pool too small for autoencoder baseline"}
    dev = device or torch.device("cpu")
    torch.manual_seed(seed)
    model = AppearanceAutoencoder(
        feature_dim=train_features.shape[1],
        hidden_dims=hidden_dims,
        embedding_dim=embedding_dim,
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_tensor = train_features.to(dev)
    for _ in range(epochs):
        permutation = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            batch_indices = permutation[start : start + batch_size]
            batch = train_tensor[batch_indices]
            _, reconstruction = model(batch)
            loss = functional.mse_loss(reconstruction, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        embeddings = model.encoder(test_features.to(dev)).cpu()
    embeddings = functional.normalize(embeddings, p=2, dim=1)
    entry = baseline_metric_entry(
        embeddings, responses, masks, k=k, n_perm=n_perm, n_boot=n_boot, seed=seed
    )
    entry["training"] = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "hidden_dims": list(hidden_dims),
        "embedding_dim": int(embedding_dim),
        "train_units": int(n_train),
        "response_supervision": False,
        "seed": int(seed),
    }
    return entry
