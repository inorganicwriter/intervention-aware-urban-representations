"""Multi-modal urban representation model with response prediction heads and cross-view synergy."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .dataset import RESPONSE_OFFSETS
from .encoder import (
    ContextEncoder,
    CrossViewSynergyModule,
    ImageEncoder,
    InterventionEncoder,
    SatelliteAlphaEarthEncoder,
    StateEncoder,
    TabularEncoder,
)

FAMILY_DIMS = {family: end - start for family, (start, end, _) in RESPONSE_OFFSETS.items()}

# Fixed initial mixing scale for explicit conditioning tokens.  The token
# embedding direction is learned, but this bounds how much a random
# initialization may distort the tabular geometry (see forward()).
CONDITIONING_MIX_SCALE = 0.1


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
        use_satellite_encoder: bool = False,
        use_cross_view_synergy: bool = False,
        synergy_heads: int = 4,
        predictor_hidden_dim: int = 64,
        image_pooling: str = "max",
        conditioning: str | None = None,
        conditioning_schema: str = "v2",
        learnable_temperature: bool = False,
        uncertainty_weighted: bool = False,
    ):
        super().__init__()
        if image_pooling not in {"max", "mean", "meanmax"}:
            raise ValueError(f"Unknown image_pooling: {image_pooling!r}")
        if conditioning not in {None, "opening_year", "station", "opening_year_station"}:
            raise ValueError(f"Unknown conditioning: {conditioning!r}")
        if conditioning_schema not in {"v1", "v2"}:
            raise ValueError(f"Unknown conditioning_schema: {conditioning_schema!r}")
        self.embedding_dim = embedding_dim
        self.hidden_dims = tuple(hidden_dims)
        self.use_image_encoder = use_image_encoder
        self.use_satellite_encoder = use_satellite_encoder
        self.use_cross_view_synergy = use_cross_view_synergy
        self.synergy_heads = synergy_heads
        self.image_pooling = image_pooling
        self.conditioning = conditioning
        self.conditioning_schema = conditioning_schema
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
        self.satellite_encoder = (
            SatelliteAlphaEarthEncoder(embedding_dim=embedding_dim, dropout=dropout)
            if (use_satellite_encoder or use_cross_view_synergy)
            else None
        )
        self.cross_view_synergy = (
            CrossViewSynergyModule(embedding_dim=embedding_dim, num_heads=synergy_heads, dropout=dropout)
            if use_cross_view_synergy
            else None
        )

        # meanmax pooling concatenates mean and max pooled image embeddings
        # before a projection back to the embedding dimension.
        pooling_out = embedding_dim * 2 if image_pooling == "meanmax" else embedding_dim
        self.pool_projection = (
            nn.Linear(pooling_out, embedding_dim) if image_pooling == "meanmax" else None
        )

        has_visual = use_image_encoder or use_satellite_encoder or use_cross_view_synergy
        self.fusion = nn.Linear(embedding_dim * 2, embedding_dim) if has_visual else None

        # Explicit intervention conditioning
        station_codes = 16 if conditioning_schema == "v1" else 32
        self.conditioning_embedding = (
            nn.Embedding(30, embedding_dim)
            if conditioning in {"opening_year", "opening_year_station"}
            else None
        )
        self.station_embedding = (
            nn.Embedding(station_codes, embedding_dim)
            if conditioning in {"station", "opening_year_station"}
            else None
        )

        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.predictor = MultiHeadPredictor(embedding_dim, predictor_hidden_dim)

        self.logit_scale = (
            nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
            if learnable_temperature
            else None
        )
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

    def encode_satellite(self, satellite_embeddings: torch.Tensor) -> torch.Tensor:
        if self.satellite_encoder is None:
            return torch.zeros(satellite_embeddings.shape[0], self.embedding_dim, device=satellite_embeddings.device)
        return self.satellite_encoder(satellite_embeddings)

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
        satellite_embeddings: torch.Tensor | None = None,
        satellite_mask: torch.Tensor | None = None,
        satellite_vectors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if satellite_embeddings is None and satellite_vectors is not None:
            satellite_embeddings = satellite_vectors
        tab_emb = self.tab_encoder(features)

        if self.conditioning_embedding is not None or self.station_embedding is not None:
            if self.conditioning_schema == "v1":
                added = torch.zeros_like(tab_emb)
                if self.conditioning_embedding is not None:
                    if conditioning_tokens is None:
                        raise ValueError(
                            "Model is conditioning-enabled but no conditioning tokens given"
                        )
                    added = added + self.conditioning_embedding(
                        conditioning_tokens.to(tab_emb.device)
                    )
                if self.station_embedding is not None:
                    if station_tokens is None:
                        raise ValueError(
                            "Model is station-conditioning-enabled but no station tokens given"
                        )
                    added = added + self.station_embedding(station_tokens.to(tab_emb.device))
                tab_emb = functional.normalize(tab_emb + added, p=2, dim=1)
            else:
                added_v2: torch.Tensor | None = None
                if self.conditioning_embedding is not None:
                    if conditioning_tokens is None:
                        raise ValueError(
                            "Model is conditioning-enabled but no conditioning tokens given"
                        )
                    added_v2 = functional.normalize(
                        self.conditioning_embedding(conditioning_tokens.to(tab_emb.device)),
                        p=2,
                        dim=1,
                    )
                if self.station_embedding is not None:
                    if station_tokens is None:
                        raise ValueError(
                            "Model is station-conditioning-enabled but no station tokens given"
                        )
                    token_emb = functional.normalize(
                        self.station_embedding(station_tokens.to(tab_emb.device)),
                        p=2,
                        dim=1,
                    )
                    added_v2 = token_emb if added_v2 is None else added_v2 + token_emb
                assert added_v2 is not None
                tab_emb = functional.normalize(
                    tab_emb + CONDITIONING_MIX_SCALE * added_v2, p=2, dim=1
                )

        visual_emb: torch.Tensor | None = None

        if self.use_cross_view_synergy and self.cross_view_synergy is not None:
            # VIGOR Cross-View Synergy
            if self.image_encoder is not None and images is not None:
                img_emb = self._pool_images(images, image_mask)
                has_img = image_mask.any(dim=1) if image_mask is not None else torch.ones(images.shape[0], dtype=torch.bool, device=images.device)
            else:
                img_emb = torch.zeros_like(tab_emb)
                has_img = torch.zeros(tab_emb.shape[0], dtype=torch.bool, device=tab_emb.device)

            if self.satellite_encoder is not None and satellite_embeddings is not None:
                sat_emb = self.satellite_encoder(satellite_embeddings)
                has_sat = satellite_mask if satellite_mask is not None else torch.ones(satellite_embeddings.shape[0], dtype=torch.bool, device=satellite_embeddings.device)
            else:
                sat_emb = torch.zeros_like(tab_emb)
                has_sat = torch.zeros(tab_emb.shape[0], dtype=torch.bool, device=tab_emb.device)

            visual_emb = self.cross_view_synergy(
                pano_emb=img_emb,
                sat_emb=sat_emb,
                pano_mask=has_img,
                sat_mask=has_sat,
            )
        elif self.use_image_encoder and images is not None:
            visual_emb = self._pool_images(images, image_mask)
        elif self.use_satellite_encoder and satellite_embeddings is not None and self.satellite_encoder is not None:
            visual_emb = self.satellite_encoder(satellite_embeddings)

        if visual_emb is not None and self.fusion is not None:
            fused = self.fusion(torch.cat([tab_emb, visual_emb], dim=-1))
        else:
            fused = tab_emb

        embedding = functional.normalize(self.projector(fused), p=2, dim=1)
        predictions = self.predictor(embedding)
        return embedding, predictions

    def forward_features_only(self, features: torch.Tensor) -> torch.Tensor:
        return self.tab_encoder(features)


class STCResponseModel(nn.Module):
    """S/T/C multimodal response model.

    - State S: POI, VIIRS, streetview, 6-year satellite -> StateEncoder -> z
    - Intervention T: 7 intervention fields + masks -> InterventionEncoder -> t
    - Context C: AlphaEarth64, housing, population, city proxies -> ContextEncoder -> b
    - Response Prediction: concat(z, t, b) -> ResponsePredictor -> response paths

    z is the learned multimodal state embedding: it is generated solely from S.
    T and C condition response prediction but are not folded into z.
    """

    is_stc_model = True

    def __init__(
        self,
        state_tabular_dim: int = 15,
        context_tabular_dim: int = 12,
        intervention_dim: int = 7,
        alphaearth_dim: int = 64,
        z_dim: int = 128,
        t_dim: int = 32,
        b_dim: int = 64,
        predictor_hidden_dim: int = 64,
        outcome_dim: int = 6,
        outcome_family: str | None = None,
        use_streetview: bool = True,
        use_satellite: bool = True,
        dropout: float = 0.1,
        learnable_temperature: bool = False,
        uncertainty_weighted: bool = False,
        streetview_time_aware: bool = True,
    ):
        super().__init__()
        self.state_tabular_dim = state_tabular_dim
        self.context_tabular_dim = context_tabular_dim
        self.intervention_dim = intervention_dim
        self.alphaearth_dim = alphaearth_dim
        self.z_dim = z_dim
        self.t_dim = t_dim
        self.b_dim = b_dim
        self.outcome_dim = outcome_dim
        self.outcome_family = outcome_family
        self.use_streetview = use_streetview
        self.use_satellite = use_satellite
        self.streetview_time_aware = streetview_time_aware
        self.learnable_temperature = learnable_temperature
        self.uncertainty_weighted = uncertainty_weighted

        # 1. State Encoder (S -> z)
        self.state_encoder = StateEncoder(
            tabular_dim=state_tabular_dim,
            embedding_dim=z_dim,
            use_streetview=use_streetview,
            use_satellite=use_satellite,
            dropout=dropout,
            streetview_time_aware=streetview_time_aware,
        )

        # 2. Intervention Encoder (T -> t)
        self.intervention_encoder = InterventionEncoder(
            in_features=intervention_dim,
            embedding_dim=t_dim,
            dropout=dropout,
        )

        # 3. Context Encoder (C -> b)
        self.context_encoder = ContextEncoder(
            alphaearth_dim=alphaearth_dim,
            tabular_dim=context_tabular_dim,
            embedding_dim=b_dim,
            dropout=dropout,
        )

        # 4. Predictor Head (concat(z, t, b) -> response)
        fusion_dim = z_dim + t_dim + b_dim
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, predictor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden_dim, outcome_dim),
        )

        self.logit_scale = (
            nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
            if learnable_temperature
            else None
        )
        self.log_var_rep = (
            nn.Parameter(torch.tensor(math.log(2.0))) if uncertainty_weighted else None
        )
        self.log_var_pred = (
            nn.Parameter(torch.tensor(math.log(2.0))) if uncertainty_weighted else None
        )

    def temperature(self) -> float | torch.Tensor:
        if self.logit_scale is None:
            return 0.07
        scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        return 1.0 / scale

    def encode_state(
        self,
        state_tabular: torch.Tensor,
        state_field_mask: torch.Tensor | None = None,
        streetview_features: torch.Tensor | None = None,
        streetview_mask: torch.Tensor | None = None,
        streetview_images: torch.Tensor | None = None,
        streetview_gap_compressed: torch.Tensor | None = None,
        satellite_images: torch.Tensor | None = None,
        satellite_pixel_mask: torch.Tensor | None = None,
        satellite_year_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.state_encoder(
            state_tabular=state_tabular,
            state_field_mask=state_field_mask,
            streetview_features=streetview_features,
            streetview_mask=streetview_mask,
            streetview_images=streetview_images,
            streetview_gap_compressed=streetview_gap_compressed,
            satellite_images=satellite_images,
            satellite_pixel_mask=satellite_pixel_mask,
            satellite_year_mask=satellite_year_mask,
        )

    def encode_intervention(
        self,
        intervention: torch.Tensor,
        intervention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.intervention_encoder(
            intervention=intervention,
            intervention_mask=intervention_mask,
        )

    def encode_context(
        self,
        context_tabular: torch.Tensor,
        context_field_mask: torch.Tensor | None = None,
        alphaearth: torch.Tensor | None = None,
        alphaearth_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.context_encoder(
            context_tabular=context_tabular,
            context_field_mask=context_field_mask,
            alphaearth=alphaearth,
            alphaearth_available=alphaearth_available,
        )

    def _map_predictions(self, pred_out: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.outcome_family:
            return {self.outcome_family: pred_out}
        if self.outcome_dim == 27:
            result: dict[str, torch.Tensor] = {}
            for family, (start, end, _) in RESPONSE_OFFSETS.items():
                result[family] = pred_out[:, start:end]
            return result
        return {"response": pred_out}

    def encode_components(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode_state(
            state_tabular=batch["state_tabular"],
            state_field_mask=batch.get("state_field_mask"),
            streetview_features=batch.get("streetview_features"),
            streetview_mask=batch.get("streetview_mask"),
            streetview_images=batch.get("streetview_images"),
            streetview_gap_compressed=batch.get("streetview_gap_compressed"),
            satellite_images=batch.get("satellite_images"),
            satellite_pixel_mask=batch.get("satellite_pixel_mask"),
            satellite_year_mask=batch.get("satellite_year_mask"),
        )
        t = self.encode_intervention(
            intervention=batch["intervention"],
            intervention_mask=batch.get("intervention_mask"),
        )
        b = self.encode_context(
            context_tabular=batch["context"],
            context_field_mask=batch.get("context_field_mask"),
            alphaearth=batch.get("alphaearth"),
            alphaearth_available=batch.get("alphaearth_available"),
        )
        return z, t, b

    def forward_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        z, t, b = self.encode_components(batch)
        fused = torch.cat([z, t, b], dim=-1)
        pred_out = self.predictor(fused)
        predictions = self._map_predictions(pred_out)
        return z, predictions, fused

    def forward(
        self,
        state_tabular: torch.Tensor,
        state_field_mask: torch.Tensor | None = None,
        intervention: torch.Tensor | None = None,
        intervention_mask: torch.Tensor | None = None,
        context_tabular: torch.Tensor | None = None,
        context_field_mask: torch.Tensor | None = None,
        alphaearth: torch.Tensor | None = None,
        alphaearth_available: torch.Tensor | None = None,
        streetview_features: torch.Tensor | None = None,
        streetview_mask: torch.Tensor | None = None,
        streetview_images: torch.Tensor | None = None,
        streetview_gap_compressed: torch.Tensor | None = None,
        satellite_images: torch.Tensor | None = None,
        satellite_pixel_mask: torch.Tensor | None = None,
        satellite_year_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        B = state_tabular.shape[0]
        device = state_tabular.device
        if intervention is None:
            intervention = torch.zeros(B, self.intervention_dim, device=device)
        if context_tabular is None:
            context_tabular = torch.zeros(B, self.context_tabular_dim, device=device)

        batch = {
            "state_tabular": state_tabular,
            "state_field_mask": state_field_mask,
            "intervention": intervention,
            "intervention_mask": intervention_mask,
            "context": context_tabular,
            "context_field_mask": context_field_mask,
            "alphaearth": alphaearth,
            "alphaearth_available": alphaearth_available,
            "streetview_features": streetview_features,
            "streetview_mask": streetview_mask,
            "streetview_images": streetview_images,
            "streetview_gap_compressed": streetview_gap_compressed,
            "satellite_images": satellite_images,
            "satellite_pixel_mask": satellite_pixel_mask,
            "satellite_year_mask": satellite_year_mask,
        }
        return self.forward_batch(batch)


class JointSTCResponseModel(STCResponseModel):
    """One z128, direct b96, and independent VIIRS/POI response heads."""

    architecture_version = "stc_joint_v1"

    def __init__(
        self,
        state_tabular_dim: int = 15,
        context_tabular_dim: int = 12,
        intervention_dim: int = 7,
        use_streetview: bool = True,
        use_satellite: bool = True,
        dropout: float = 0.1,
        prediction_inputs: str = "full",
    ):
        from .encoder import DirectContextEncoder

        if prediction_inputs not in {"full", "tc_only", "z_only"}:
            raise ValueError("prediction_inputs must be full, tc_only or z_only")
        super().__init__(
            state_tabular_dim=state_tabular_dim,
            context_tabular_dim=context_tabular_dim,
            intervention_dim=intervention_dim,
            z_dim=128,
            t_dim=32,
            b_dim=96,
            predictor_hidden_dim=128,
            outcome_dim=18,
            use_streetview=use_streetview,
            use_satellite=use_satellite,
            dropout=dropout,
        )
        self.context_encoder = DirectContextEncoder(context_tabular_dim, dropout)
        del self.predictor
        self.embedding_dim = 128
        self.hidden_dims = (128,)
        self.prediction_inputs = prediction_inputs

        def head(dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, dim)
            )

        self.viirs_head = head(6)
        self.poi_head = head(12)

    def forward_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        z, t, b = self.encode_components(batch)
        predictor_z = torch.zeros_like(z) if self.prediction_inputs == "tc_only" else z
        if self.prediction_inputs == "z_only":
            t, b = torch.zeros_like(t), torch.zeros_like(b)
        fused = torch.cat([predictor_z, t, b], dim=-1)
        return z, {"viirs": self.viirs_head(fused), "poi": self.poi_head(fused)}, fused
