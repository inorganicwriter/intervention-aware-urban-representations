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
    with torch.no_grad():
        for batch in loader:
            feat = batch["features"].to(device)
            resp = batch["responses"].to(device)
            mask = batch["response_mask"].to(device)
            images = batch.get("images")
            image_mask = batch.get("image_mask")
            conditioning_tokens = batch.get("conditioning_tokens")
            station_tokens = batch.get("station_tokens")
            if images is not None:
                images = images.to(device)
            if image_mask is not None:
                image_mask = image_mask.to(device)
            if conditioning_tokens is not None:
                conditioning_tokens = conditioning_tokens.to(device)
            embedding, _ = model(
                feat, images, image_mask,
                conditioning_tokens=conditioning_tokens,
                station_tokens=station_tokens,
            )
            embeddings.append(embedding.cpu())
            features.append(feat.cpu())
            responses.append(resp.cpu())
            masks.append(mask.cpu())
            city_keys.extend(str(value) for value in batch.get("city_keys", []))
            quality_grades.extend(str(value) for value in batch.get("quality_grades", []))
    if not embeddings:
        return None
    return {
        "embeddings": torch.cat(embeddings, dim=0),
        "features": torch.cat(features, dim=0),
        "responses": torch.cat(responses, dim=0),
        "masks": torch.cat(masks, dim=0),
        "city_keys": city_keys,
        "quality_grades": quality_grades,
    }
