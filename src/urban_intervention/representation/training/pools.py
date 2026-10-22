"""Evaluation-pool collection for representation training."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..model import ResponseEmbeddingModel
from .types import Pool


def _collect_pool(
    model: ResponseEmbeddingModel,
    loader: DataLoader,
    device: torch.device,
) -> Pool | None:
    """Run the model in eval mode and return full-pool tensors (CPU)."""
    model.eval()
    embeddings: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    responses: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    city_keys: list[str] = []
    quality_grades: list[str] = []
    predictions: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in loader:
            feat = batch["features"].to(device)
            resp = batch["responses"].to(device)
            mask = batch["response_mask"].to(device)
            images = batch.get("images")
            image_mask = batch.get("image_mask")
            conditioning_tokens = batch.get("conditioning_tokens")
            station_tokens = batch.get("station_tokens")
            satellite_vectors = batch.get("satellite_vectors")
            satellite_mask = batch.get("satellite_mask")
            if images is not None:
                images = images.to(device)
            if image_mask is not None:
                image_mask = image_mask.to(device)
            if conditioning_tokens is not None:
                conditioning_tokens = conditioning_tokens.to(device)
            if satellite_vectors is not None:
                satellite_vectors = satellite_vectors.to(device)
            if satellite_mask is not None:
                satellite_mask = satellite_mask.to(device)
            if getattr(model, "is_stc_model", False):
                device_batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                embedding, preds, _ = model.forward_batch(device_batch)
                raw_features = torch.cat(
                    [
                        batch["state_tabular"],
                        batch["context"],
                        batch["intervention"],
                        batch["alphaearth"],
                    ],
                    dim=1,
                )
            else:
                raw_features = feat
                embedding, preds = model(
                    feat,
                    images,
                    image_mask,
                    conditioning_tokens=conditioning_tokens,
                    station_tokens=station_tokens,
                    satellite_vectors=satellite_vectors,
                    satellite_mask=satellite_mask,
                )
            embeddings.append(embedding.cpu())
            features.append(raw_features.cpu())
            responses.append(resp.cpu())
            masks.append(mask.cpu())
            city_keys.extend(str(value) for value in batch.get("city_keys", []))
            quality_grades.extend(str(value) for value in batch.get("quality_grades", []))
            if isinstance(preds, dict):
                for k, v in preds.items():
                    predictions.setdefault(k, []).append(v.cpu())
    if not embeddings:
        return None
    return {
        "embeddings": torch.cat(embeddings, dim=0),
        "features": torch.cat(features, dim=0),
        "responses": torch.cat(responses, dim=0),
        "masks": torch.cat(masks, dim=0),
        "city_keys": city_keys,
        "quality_grades": quality_grades,
        "predictions": {k: torch.cat(v, dim=0) for k, v in predictions.items()} if predictions else {},
    }
