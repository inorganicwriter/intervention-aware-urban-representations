"""Multi-modal encoders for urban grid features with cross-view visual synergy.

Encoders:
- TabularEncoder: 27-d pre-treatment tabular covariates (location, network, historical trends).
- SatelliteAlphaEarthEncoder: 64-d Google AlphaEarth continuous geospatial embeddings.
- StreetviewPanoramaEncoder: Ground-level 360° panoramas via frozen DINOv2 backbone.
- CrossViewSynergyModule: VIGOR-inspired cross-view cross-attention where ground
  panorama queries overhead satellite embeddings.
"""

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
    local_root = _hub_dir() / "facebookresearch_dinov2_main"
    if local_root.is_dir():
        try:
            model = torch.hub.load(
                str(local_root),
                model_name,
                source="local",
                skip_validation=True,
            )
        except (OSError, RuntimeError) as exc:
            logger.warning("DINOv2 local cache load failed (%s), trying GitHub", exc)
            model = torch.hub.load(
                repo, model_name, source="github", skip_validation=True
            )
    else:
        model = torch.hub.load(repo, model_name, source="github", skip_validation=True)
    for param in model.parameters():
        param.requires_grad = False
    _DINO_CACHE[cache_key] = model
    return model


class TabularEncoder(nn.Module):
    """Encodes 27-d tabular pre-treatment priors."""

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


class SatelliteAlphaEarthEncoder(nn.Module):
    """Encodes 64-dimensional Google AlphaEarth continuous foundation embeddings."""

    ALPHAEARTH_DIM = 64

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Accepts:
            (B, 64) -> returns (B, embedding_dim)
            (B, S, 64) -> returns (B, S, embedding_dim)
        """
        emb = self.net(x)
        return functional.normalize(emb, p=2, dim=-1)


class ImageEncoder(nn.Module):
    """Standard DINOv2 feature extractor for single/multi perspective images."""

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


class StreetviewPanoramaEncoder(nn.Module):
    """Encoder for ground 360° panoramas.

    Supports:
    - Pre-extracted DINOv2 feature tensors (B, DINO_DIM) or (B, K, DINO_DIM)
    - Raw image tensors (B, N, C, H, W) through frozen DINOv2 backbone
    """

    DINO_EMBED_DIM = 384

    def __init__(self, embedding_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.image_encoder = ImageEncoder(embedding_dim=embedding_dim)
        self.feature_projection = nn.Sequential(
            nn.Linear(self.DINO_EMBED_DIM, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project pre-extracted DINOv2 features (B, K, 384) or (B, 384)."""
        emb = self.feature_projection(features)
        return functional.normalize(emb, p=2, dim=-1)

    def forward(
        self,
        images: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features is not None:
            return self.forward_features(features)
        if images is not None:
            return self.image_encoder(images)
        raise ValueError("Either images or features must be provided to StreetviewPanoramaEncoder")


class CrossViewSynergyModule(nn.Module):
    """VIGOR-inspired Cross-View Synergy Module.

    Ground-level streetview panorama tokens act as Query to retrieve and interact
    with overhead satellite embedding tokens (Key, Value) via Multi-Head Cross-Attention.
    Gracefully handles missing modalities via fallback projections and learned mask tokens.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_query = nn.LayerNorm(embedding_dim)
        self.norm_kv = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        # Learnable missing modality embeddings
        self.missing_pano_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.missing_sat_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.normal_(self.missing_pano_token, std=0.02)
        nn.init.normal_(self.missing_sat_token, std=0.02)

    def forward(
        self,
        pano_emb: torch.Tensor,
        sat_emb: torch.Tensor,
        pano_mask: torch.Tensor | None = None,
        sat_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-view interaction.

        Args:
            pano_emb: (B, D) or (B, K, D) ground perspective embeddings.
            sat_emb: (B, D) or (B, S, D) overhead satellite embeddings.
            pano_mask: (B,) boolean tensor indicating whether streetview is available.
            sat_mask: (B,) boolean tensor indicating whether satellite is available.

        Returns:
            (B, D) synergistic visual embedding.
        """
        # Ensure 3D shapes (B, Seq, D)
        if pano_emb.ndim == 2:
            pano_emb = pano_emb.unsqueeze(1)
        if sat_emb.ndim == 2:
            sat_emb = sat_emb.unsqueeze(1)

        batch_size = pano_emb.shape[0]

        # Substitute missing modalities with learned tokens if mask is provided
        if pano_mask is not None:
            pano_valid = pano_mask.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            pano_emb = torch.where(pano_valid, pano_emb, self.missing_pano_token.expand(batch_size, pano_emb.shape[1], -1))

        if sat_mask is not None:
            sat_valid = sat_mask.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            sat_emb = torch.where(sat_valid, sat_emb, self.missing_sat_token.expand(batch_size, sat_emb.shape[1], -1))

        # Query = Ground, Key/Value = Overhead
        q = self.norm_query(pano_emb)
        kv = self.norm_kv(sat_emb)

        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        x = pano_emb + attn_out
        x = x + self.ffn(x)

        # Pool over sequence dimension to obtain (B, D)
        pooled = x.mean(dim=1)
        return functional.normalize(pooled, p=2, dim=-1)


class SatelliteSixBandCNN(nn.Module):
    """6-band CNN for spatial encoding of a single annual satellite patch.

    Channels: (red, green, blue, nir, swir1, swir2).
    Accepts pixel_mask (B, 1, H, W) to strictly zero out unobserved/invalid pixels.
    """

    def __init__(self, in_channels: int = 6, embedding_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
        )
        self.missing_token = nn.Parameter(torch.zeros(1, embedding_dim))
        nn.init.normal_(self.missing_token, std=0.02)

    def forward(
        self,
        images: torch.Tensor,
        pixel_mask: torch.Tensor | None = None,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = images.shape[0]
        # Invariance rule: strictly zero-out masked pixels before convolution
        if pixel_mask is not None:
            images = images * pixel_mask.float()

        x = self.relu(self.bn1(self.conv1(images)))
        if x.shape[-1] >= 4 and x.shape[-2] >= 4:
            x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.adaptive_pool(x).flatten(1)
        out = self.fc(x)

        if valid_patch_mask is not None:
            out = torch.where(
                valid_patch_mask.unsqueeze(-1),
                out,
                self.missing_token.expand(B, -1),
            )
        return functional.normalize(out, p=2, dim=-1)


class SatelliteSequenceEncoder(nn.Module):
    """Encodes a 6-year annual satellite sequence (relative years -6 to -1).

    Uses SatelliteSixBandCNN for each annual patch, relative year positional embeddings,
    and a masked GRU that skips hidden state updates for unobserved years.
    """

    def __init__(self, embedding_dim: int = 128, num_years: int = 6, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_years = num_years
        self.cnn = SatelliteSixBandCNN(in_channels=6, embedding_dim=embedding_dim, dropout=dropout)
        self.year_embedding = nn.Embedding(num_years, embedding_dim)
        self.gru_cell = nn.GRUCell(embedding_dim, embedding_dim)
        self.missing_seq_token = nn.Parameter(torch.zeros(1, embedding_dim))
        nn.init.normal_(self.missing_seq_token, std=0.02)

    def forward(
        self,
        satellite_images: torch.Tensor,
        satellite_pixel_mask: torch.Tensor | None = None,
        satellite_year_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C, H, W = satellite_images.shape
        if satellite_year_mask is None:
            satellite_year_mask = torch.ones(B, T, dtype=torch.bool, device=satellite_images.device)

        # Invariance rule: zero-out masked years in input tensor
        ymask_expanded = satellite_year_mask.view(B, T, 1, 1, 1)
        satellite_images = satellite_images * ymask_expanded.float()

        img_flat = satellite_images.view(B * T, C, H, W)
        pmask_flat = (
            satellite_pixel_mask.view(B * T, 1, H, W)
            if satellite_pixel_mask is not None
            else None
        )
        ymask_flat = satellite_year_mask.view(B * T)

        patch_embs = self.cnn(img_flat, pixel_mask=pmask_flat, valid_patch_mask=ymask_flat)
        patch_embs = patch_embs.view(B, T, self.embedding_dim)

        year_ids = torch.arange(T, device=satellite_images.device)
        year_pos = self.year_embedding(year_ids).unsqueeze(0)  # (1, T, D)
        patch_embs = patch_embs + year_pos

        # Masked GRU step: only update hidden state when the year is valid
        h = torch.zeros(B, self.embedding_dim, device=satellite_images.device)
        for t in range(T):
            mask_t = satellite_year_mask[:, t].unsqueeze(-1)  # (B, 1)
            h_next = self.gru_cell(patch_embs[:, t], h)
            h = torch.where(mask_t, h_next, h)

        # Handle full sequence missing
        has_any_year = satellite_year_mask.any(dim=1, keepdim=True)
        h = torch.where(has_any_year, h, self.missing_seq_token.expand(B, -1))
        return functional.normalize(h, p=2, dim=-1)


class InterventionEncoder(nn.Module):
    """Encodes 7 intervention features with availability masks.

    Fields: opening_year, is_transfer, is_new_line, is_extension, is_terminal,
            n_lines, same_month_openings.
    """

    def __init__(self, in_features: int = 7, embedding_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(in_features * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(
        self,
        intervention: torch.Tensor,
        intervention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intervention_mask is None:
            intervention_mask = torch.ones_like(intervention, dtype=torch.bool)
        masked_interv = intervention * intervention_mask.float()
        x = torch.cat([masked_interv, intervention_mask.float()], dim=-1)
        emb = self.net(x)
        return functional.normalize(emb, p=2, dim=-1)


class ContextEncoder(nn.Module):
    """Encodes context C: 64-d AlphaEarth, historical housing, population, city proxies.

    AlphaEarth enters ONLY the context branch.
    """

    def __init__(
        self,
        alphaearth_dim: int = 64,
        tabular_dim: int = 12,
        embedding_dim: int = 64,
        ae_proj_dim: int = 32,
        tab_proj_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.alphaearth_dim = alphaearth_dim
        self.tabular_dim = tabular_dim
        self.embedding_dim = embedding_dim

        self.ae_proj = nn.Sequential(
            nn.Linear(alphaearth_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, ae_proj_dim),
            nn.LayerNorm(ae_proj_dim),
        )
        self.missing_ae_token = nn.Parameter(torch.zeros(1, alphaearth_dim))
        nn.init.normal_(self.missing_ae_token, std=0.02)

        self.tab_proj = nn.Sequential(
            nn.Linear(tabular_dim * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, tab_proj_dim),
            nn.LayerNorm(tab_proj_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(ae_proj_dim + tab_proj_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(
        self,
        context_tabular: torch.Tensor,
        context_field_mask: torch.Tensor | None = None,
        alphaearth: torch.Tensor | None = None,
        alphaearth_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = context_tabular.shape[0]

        if context_field_mask is None:
            context_field_mask = torch.ones_like(context_tabular, dtype=torch.bool)
        masked_tab = context_tabular * context_field_mask.float()
        tab_in = torch.cat([masked_tab, context_field_mask.float()], dim=-1)
        tab_emb = self.tab_proj(tab_in)

        if alphaearth is None or alphaearth_available is None:
            ae_in = self.missing_ae_token.expand(B, -1)
        else:
            avail = alphaearth_available.unsqueeze(-1)
            masked_ae = alphaearth * avail.float()
            ae_in = torch.where(avail, masked_ae, self.missing_ae_token.expand(B, -1))
        ae_emb = self.ae_proj(ae_in)

        fused = self.fusion(torch.cat([ae_emb, tab_emb], dim=-1))
        return functional.normalize(fused, p=2, dim=-1)


class DirectContextEncoder(nn.Module):
    """Preserve AlphaEarth64 and concatenate a learned 32-value context.

    Availability is explicitly encoded in the tabular branch. No projection
    or normalization changes the valid AlphaEarth coordinates.
    """

    def __init__(self, tabular_dim: int = 12, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = 96
        self.missing_ae_token = nn.Parameter(torch.zeros(1, 64))
        nn.init.normal_(self.missing_ae_token, std=0.02)
        self.tab_proj = nn.Sequential(
            nn.Linear(tabular_dim * 2 + 1, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
        )

    def forward(
        self,
        context_tabular: torch.Tensor,
        context_field_mask: torch.Tensor | None = None,
        alphaearth: torch.Tensor | None = None,
        alphaearth_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = context_tabular.shape[0]
        mask = torch.isfinite(context_tabular)
        if context_field_mask is not None:
            mask = mask & context_field_mask.bool()
        values = torch.where(mask, context_tabular, 0.0)
        available = torch.zeros(B, dtype=torch.bool, device=values.device)
        if alphaearth is not None and alphaearth_available is not None:
            if alphaearth.shape != (B, 64):
                raise ValueError("AlphaEarth must have shape (batch, 64)")
            available = alphaearth_available.to(device=values.device, dtype=torch.bool)
            if not torch.isfinite(alphaearth[available]).all():
                raise ValueError("Available AlphaEarth vectors must be finite")
            ae = torch.where(available[:, None], alphaearth, self.missing_ae_token)
        else:
            ae = self.missing_ae_token.expand(B, -1)
        tab = self.tab_proj(torch.cat([values, mask.float(), available[:, None].float()], -1))
        return torch.cat([ae, tab], dim=-1)


class StateEncoder(nn.Module):
    """Encodes state S: POI, VIIRS history, streetview, 6-year satellite sequence.

    Produces state representation z strictly from S. NEVER takes T or C.
    """

    def __init__(
        self,
        tabular_dim: int = 15,
        embedding_dim: int = 128,
        use_streetview: bool = True,
        use_satellite: bool = True,
        dropout: float = 0.1,
        streetview_time_aware: bool = True,
    ):
        super().__init__()
        self.tabular_dim = tabular_dim
        self.embedding_dim = embedding_dim
        self.use_streetview = use_streetview
        self.use_satellite = use_satellite
        self.streetview_time_aware = streetview_time_aware

        self.tab_encoder = nn.Sequential(
            nn.Linear(tabular_dim * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        if use_streetview:
            self.streetview_encoder = StreetviewPanoramaEncoder(embedding_dim=embedding_dim, dropout=dropout)
            self.missing_streetview_token = nn.Parameter(torch.zeros(1, embedding_dim))
            nn.init.normal_(self.missing_streetview_token, std=0.02)
            if streetview_time_aware:
                self.streetview_gap_encoder = nn.Sequential(
                    nn.Linear(1, embedding_dim),
                    nn.GELU(),
                    nn.Linear(embedding_dim, embedding_dim),
                )
                self.streetview_time_fusion = nn.Sequential(
                    nn.Linear(embedding_dim * 2, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.streetview_attention_score = nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.Tanh(),
                    nn.Linear(embedding_dim, 1, bias=False),
                )
        else:
            self.streetview_encoder = None

        if use_satellite:
            self.satellite_encoder = SatelliteSequenceEncoder(embedding_dim=embedding_dim, dropout=dropout)
        else:
            self.satellite_encoder = None

        n_branches = 1 + (1 if use_streetview else 0) + (1 if use_satellite else 0)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * n_branches, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(
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
        B = state_tabular.shape[0]

        if state_field_mask is None:
            state_field_mask = torch.ones_like(state_tabular, dtype=torch.bool)
        masked_tab = state_tabular * state_field_mask.float()
        tab_in = torch.cat([masked_tab, state_field_mask.float()], dim=-1)
        tab_emb = self.tab_encoder(tab_in)

        branches = [tab_emb]

        if self.use_streetview and self.streetview_encoder is not None:
            sv_emb: torch.Tensor | None = None
            if streetview_images is not None:
                if streetview_images.ndim != 5:
                    raise ValueError("streetview_images must have shape (batch, images, 3, height, width)")
                K = streetview_images.shape[1]
                if streetview_mask is None:
                    image_mask = torch.ones(B, K, dtype=torch.bool, device=streetview_images.device)
                else:
                    image_mask = streetview_mask.to(device=streetview_images.device, dtype=torch.bool)
                if image_mask.shape != (B, K):
                    raise ValueError("streetview_mask must have shape (batch, images)")
                if K:
                    flat_mask = image_mask.reshape(-1)
                    flat_emb = streetview_images.new_zeros(B * K, self.embedding_dim)
                    if flat_mask.any():
                        flat_imgs = streetview_images.reshape(B * K, *streetview_images.shape[2:])[flat_mask]
                        flat_emb[flat_mask] = self.streetview_encoder(images=flat_imgs)
                    sv_emb = flat_emb.view(B, K, self.embedding_dim)
            elif streetview_features is not None:
                sv_emb = self.streetview_encoder.forward_features(streetview_features)
                if sv_emb.ndim == 2:
                    sv_emb = sv_emb.unsqueeze(1)
                K = sv_emb.shape[1]
                image_mask = (
                    torch.ones(B, K, dtype=torch.bool, device=sv_emb.device)
                    if streetview_mask is None
                    else streetview_mask.to(device=sv_emb.device, dtype=torch.bool)
                )
                if image_mask.ndim == 1 and K == 1:
                    image_mask = image_mask.unsqueeze(-1)
                if image_mask.shape != (B, K):
                    raise ValueError("streetview_mask must have shape (batch, images)")
            else:
                image_mask = torch.zeros(B, 0, dtype=torch.bool, device=state_tabular.device)

            if sv_emb is None or sv_emb.shape[1] == 0:
                pooled = self.missing_streetview_token.expand(B, -1)
            elif self.streetview_time_aware:
                K = sv_emb.shape[1]
                if streetview_gap_compressed is None:
                    gap_compressed = sv_emb.new_zeros(B, K)
                else:
                    gap_compressed = streetview_gap_compressed.to(device=sv_emb.device, dtype=sv_emb.dtype)
                    if gap_compressed.ndim == 1 and K == 1:
                        gap_compressed = gap_compressed.unsqueeze(-1)
                    if gap_compressed.shape != (B, K):
                        raise ValueError("streetview_gap_compressed must have shape (batch, images)")
                valid_gaps = gap_compressed[image_mask]
                if not torch.isfinite(valid_gaps).all():
                    raise ValueError("Valid streetview images must have finite compressed time gaps")
                if ((valid_gaps < 0) | (valid_gaps >= 1)).any():
                    raise ValueError("Compressed streetview time gaps must be in [0, 1)")

                gap_emb = self.streetview_gap_encoder(gap_compressed.unsqueeze(-1))
                time_conditioned = self.streetview_time_fusion(torch.cat([sv_emb, gap_emb], dim=-1))
                scores = self.streetview_attention_score(time_conditioned).squeeze(-1)
                scores = scores.masked_fill(~image_mask, torch.finfo(scores.dtype).min)
                weights = torch.softmax(scores, dim=1).masked_fill(~image_mask, 0.0)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                pooled = torch.sum(time_conditioned * weights.unsqueeze(-1), dim=1)
                has_valid = image_mask.any(dim=1, keepdim=True)
                pooled = torch.where(
                    has_valid,
                    pooled,
                    self.missing_streetview_token.expand(B, -1),
                )
            else:
                valid_k = image_mask.unsqueeze(-1).to(dtype=sv_emb.dtype)
                mean_sv = (sv_emb * valid_k).sum(dim=1) / valid_k.sum(dim=1).clamp(min=1.0)
                has_valid = image_mask.any(dim=1, keepdim=True)
                pooled = torch.where(
                    has_valid,
                    mean_sv,
                    self.missing_streetview_token.expand(B, -1),
                )
            branches.append(pooled)

        if self.use_satellite and self.satellite_encoder is not None:
            if satellite_images is not None:
                sat_emb = self.satellite_encoder(
                    satellite_images,
                    satellite_pixel_mask=satellite_pixel_mask,
                    satellite_year_mask=satellite_year_mask,
                )
            else:
                sat_emb = self.satellite_encoder.missing_seq_token.expand(B, -1)
            branches.append(sat_emb)

        fused = self.fusion(torch.cat(branches, dim=-1))
        return functional.normalize(fused, p=2, dim=-1)
