"""Multi-modal urban representation model with response prediction heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .dataset import RESPONSE_OFFSETS
from .encoder import ImageEncoder, TabularEncoder

FAMILY_DIMS = {family: end - start for family, (start, end, _) in RESPONSE_OFFSETS.items()}


class MultiHeadPredictor(nn.Module):
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.heads = nn.ModuleDict()
        for family, dim in FAMILY_DIMS.items():
            self.heads[family] = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, dim),
            )

    def forward(self, emb: torch.Tensor) -> dict[str, torch.Tensor]:
        return {family: head(emb) for family, head in self.heads.items()}


class ResponseEmbeddingModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        embedding_dim: int = 128,
        dropout: float = 0.1,
        use_image_encoder: bool = True,
        predictor_hidden_dim: int = 64,
        image_pooling: str = "max",
        conditioning: str | None = None,
        learnable_temperature: bool = False,
        uncertainty_weighted: bool = False,
    ):
        super().__init__()
        if image_pooling not in {"max", "mean", "meanmax"}:
            raise ValueError(f"Unknown image_pooling: {image_pooling!r}")
        if conditioning not in {None, "opening_year", "station", "opening_year_station"}:
            raise ValueError(f"Unknown conditioning: {conditioning!r}")
        self.embedding_dim = embedding_dim
        self.use_image_encoder = use_image_encoder
        self.image_pooling = image_pooling
        self.conditioning = conditioning
        self.learnable_temperature = learnable_temperature
        self.uncertainty_weighted = uncertainty_weighted

        self.tab_encoder = TabularEncoder(
            feature_dim=feature_dim,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.image_encoder = (
            ImageEncoder(embedding_dim=embedding_dim) if use_image_encoder else None
        )

        # meanmax pooling concatenates mean and max pooled image embeddings
        # before a projection back to the embedding dimension.
        pooling_out = embedding_dim * 2 if image_pooling == "meanmax" else embedding_dim
        self.pool_projection = (
            nn.Linear(pooling_out, embedding_dim) if image_pooling == "meanmax" else None
        )

        self.fusion = nn.Linear(embedding_dim * 2, embedding_dim) if use_image_encoder else None

        # Explicit intervention conditioning: per-token embeddings injected
        # into the tabular stream.  ``opening_year`` is pre-treatment
        # information available at inference for any new station; ``station``
        # tokens encode treatment-level attributes (transfer / new-line /
        # extension / terminal, 4 bits) that are likewise known ex ante.
        # Disabled by default; the response-aligned objective alone is the
        # "implicit conditioning".
        self.conditioning_embedding = (
            nn.Embedding(30, embedding_dim) if conditioning in {"opening_year", "opening_year_station"} else None
        )
        self.station_embedding = (
            nn.Embedding(16, embedding_dim) if conditioning in {"station", "opening_year_station"} else None
        )

        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.predictor = MultiHeadPredictor(embedding_dim, predictor_hidden_dim)

        # Learnable InfoNCE temperature (CLIP-style): the stored parameter is
        # log(1/τ) initialised to the default 1/0.07, so exp() is the logit
        # scale; the scale is clamped to <= 100, i.e. the temperature never
        # falls below 0.01.
        self.logit_scale = (
            nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
            if learnable_temperature
            else None
        )
        # Uncertainty weights for the representation/prediction task balance
        # (Kendall et al. 2018): log vars initialised at log(2) so the
        # starting weights exp(-log_var) match the default 0.5/0.5 balance.
        self.log_var_rep = (
            nn.Parameter(torch.tensor(math.log(2.0))) if uncertainty_weighted else None
        )
        self.log_var_pred = (
            nn.Parameter(torch.tensor(math.log(2.0))) if uncertainty_weighted else None
        )

    def temperature(self) -> float | torch.Tensor:
        """InfoNCE temperature = 1/logit scale, with the scale clamped <= 100."""
        if self.logit_scale is None:
            return 0.07
        scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        return 1.0 / scale

    def encode_tabular(self, x: torch.Tensor) -> torch.Tensor:
        return self.tab_encoder(x)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if self.image_encoder is None:
            return torch.zeros(images.shape[0], self.embedding_dim, device=images.device)
        return self.image_encoder(images)

    def _pool_images(
        self, images: torch.Tensor, image_mask: torch.Tensor | None
    ) -> torch.Tensor:
        batch_size, max_images, c, h, w = images.shape
        images_flat = images.reshape(batch_size * max_images, c, h, w)
        img_emb_flat = self.image_encoder(images_flat) if self.image_encoder else None
        if img_emb_flat is None:
            return torch.zeros(batch_size, self.embedding_dim, device=images.device)
        img_emb_flat = img_emb_flat.reshape(batch_size, max_images, self.embedding_dim)
        if image_mask is None:
            image_mask = torch.ones(batch_size, max_images, dtype=torch.bool, device=images.device)

        if self.image_pooling == "max":
            masked = img_emb_flat.masked_fill(~image_mask.unsqueeze(-1), float("-inf"))
            pooled = masked.max(dim=1).values
            pooled = torch.where(
                pooled.isfinite().all(dim=1, keepdim=True),
                pooled,
                torch.zeros_like(pooled),
            )
        elif self.image_pooling == "mean":
            nan_masked = img_emb_flat.masked_fill(~image_mask.unsqueeze(-1), float("nan"))
            valid = image_mask.sum(dim=1)
            pooled = nan_masked.nanmean(dim=1)
            # NaN * 0 is NaN: an all-masked row must be replaced with zeros
            # explicitly, not masked by multiplication.
            pooled = torch.where(valid.gt(0).unsqueeze(-1), pooled, torch.zeros_like(pooled))
        else:  # meanmax
            nan_masked = img_emb_flat.masked_fill(~image_mask.unsqueeze(-1), float("nan"))
            valid = image_mask.sum(dim=1)
            mean_pooled = nan_masked.nanmean(dim=1)
            mean_pooled = torch.where(
                valid.gt(0).unsqueeze(-1), mean_pooled, torch.zeros_like(mean_pooled)
            )
            max_masked = img_emb_flat.masked_fill(~image_mask.unsqueeze(-1), float("-inf"))
            max_pooled = max_masked.max(dim=1).values
            max_pooled = torch.where(
                max_pooled.isfinite().all(dim=1, keepdim=True),
                max_pooled,
                torch.zeros_like(max_pooled),
            )
            combined = torch.cat([mean_pooled, max_pooled], dim=-1)
            pool_projection = self.pool_projection
            if pool_projection is None:
                raise RuntimeError("meanmax pooling requires pool_projection")
            pooled = functional.normalize(pool_projection(combined), p=2, dim=1)
        return pooled

    def forward(
        self,
        features: torch.Tensor,
        images: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        conditioning_tokens: torch.Tensor | None = None,
        station_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tab_emb = self.tab_encoder(features)

        if self.conditioning_embedding is not None or self.station_embedding is not None:
            if conditioning_tokens is None and self.conditioning_embedding is not None:
                raise ValueError("Model is conditioning-enabled but no conditioning tokens given")
            if station_tokens is None and self.station_embedding is not None:
                raise ValueError("Model is station-conditioning-enabled but no station tokens given")
            added = torch.zeros_like(tab_emb)
            if self.conditioning_embedding is not None:
                added = added + self.conditioning_embedding(
                    conditioning_tokens.to(tab_emb.device)
                )
            if self.station_embedding is not None:
                added = added + self.station_embedding(station_tokens.to(tab_emb.device))
            tab_emb = functional.normalize(tab_emb + added, p=2, dim=1)

        if self.image_encoder is not None and images is not None:
            img_emb = self._pool_images(images, image_mask)
        else:
            img_emb = torch.zeros_like(tab_emb)

        if self.fusion is not None:
            fused = self.fusion(torch.cat([tab_emb, img_emb], dim=-1))
        else:
            fused = tab_emb

        embedding = functional.normalize(self.projector(fused), p=2, dim=1)

        predictions = self.predictor(embedding)
        return embedding, predictions

    def forward_features_only(self, features: torch.Tensor) -> torch.Tensor:
        return self.tab_encoder(features)
