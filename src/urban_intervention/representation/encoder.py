"""Multi-modal encoders for urban grid features."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional

logger = logging.getLogger(__name__)

_DINO_CACHE: dict[tuple[str, str], nn.Module] = {}


def _hub_dir() -> Path:
    if path := os.environ.get("TORCH_HUB"):
        return Path(path)
    return Path(torch.hub.get_dir())


def _ensure_dinov2(
    repo: str = "facebookresearch/dinov2", model_name: str = "dinov2_vits14"
) -> nn.Module:
    cache_key = (repo, model_name)
    if cache_key in _DINO_CACHE:
        return _DINO_CACHE[cache_key]
    try:
        model = torch.hub.load(repo, model_name, source="github", skip_validation=True)
        for param in model.parameters():
            param.requires_grad = False
        _DINO_CACHE[cache_key] = model
        return model
    except (OSError, RuntimeError) as exc:
        local_root = _hub_dir() / "facebookresearch_dinov2_main"
        if local_root.is_dir():
            logger.warning("DINOv2 hub load failed (%s), retrying with local cache", exc)
            model = torch.hub.load(
                str(local_root),
                model_name,
                source="local",
                skip_validation=True,
            )
            for param in model.parameters():
                param.requires_grad = False
            _DINO_CACHE[cache_key] = model
            return model
        raise RuntimeError(f"Failed to load DINOv2: {exc}") from exc


class TabularEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim

        layers: list[nn.Module] = []
        in_dim = feature_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, embedding_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return functional.normalize(emb, p=2, dim=1)


class ImageEncoder(nn.Module):
    DINO_EMBED_DIM = 384
    IMAGE_SIZE = 224
    pixel_mean: torch.Tensor
    pixel_std: torch.Tensor

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self._backbone: nn.Module | None = None
        self._register_mean_std()
        self.projection = nn.Sequential(
            nn.Linear(self.DINO_EMBED_DIM, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    @property
    def backbone(self) -> nn.Module:
        if self._backbone is None:
            backbone = _ensure_dinov2()
            backbone.eval()
            # The frozen, globally cached backbone is intentionally not
            # registered as a child module. This keeps it out of checkpoints
            # and prevents ``model.train()`` from enabling stochastic training
            # behaviour in the frozen feature extractor.
            object.__setattr__(self, "_backbone", backbone)
        assert self._backbone is not None
        return self._backbone

    def _register_mean_std(self) -> None:
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.pixel_mean.to(images.device)) / self.pixel_std.to(images.device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images)
        backbone = self.backbone
        if next(backbone.parameters()).device != images.device:
            backbone = backbone.to(images.device)
        backbone.eval()
        with torch.no_grad():
            features: torch.Tensor = backbone(images)
        emb = self.projection(features)
        return functional.normalize(emb, p=2, dim=1)
