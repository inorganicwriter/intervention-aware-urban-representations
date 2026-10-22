"""Streetview inputs for event-level visual state-transition learning.

Each sample keeps every verified pre-opening image and its own opening-time
gap. Batches pad variable-size image sets and carry an explicit image mask.
Post-opening images are returned in a separate target-only branch.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.functional import resize

from .encoder import StreetviewPanoramaEncoder
from .time_encoding import compress_time_gap_months


IMAGE_SIZE = 224
PAIR_INDEX_REQUIRED_COLUMNS = {
    "pair_id",
    "treatment_order",
    "station_event_id",
    "modality",
    "target_event_time",
    "pre_state_assets_json",
    "post_target_assets_json",
    "pre_state_mask",
    "post_target_mask",
    "identity_mask",
    "pre_temporal_mask",
    "post_temporal_mask",
    "transition_supervision_mask",
    "post_target_predictor_use_allowed",
}


def _parse_assets(value: Any, field: str, pair_id: str) -> list[dict[str, Any]]:
    if value is None or pd.isna(value):
        return []
    try:
        assets = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {field} JSON for pair_id={pair_id}") from exc
    if not isinstance(assets, list) or any(not isinstance(asset, dict) for asset in assets):
        raise ValueError(f"{field} must be a JSON list of asset objects for pair_id={pair_id}")
    return assets


class StreetviewTransitionDataset(torch.utils.data.Dataset[dict[str, Any]]):
    """Load supervised streetview transitions without reducing pre-images to one.

    Each pre-image must carry the agreed compressed age. ``asset_root`` can
    remap project-relative or archived absolute paths; paths that already
    exist are used as written.
    """

    def __init__(
        self,
        pair_index_path: str | Path,
        *,
        asset_root: str | Path | None = None,
        treatment_orders: set[int] | None = None,
        target_event_times: set[int] | None = None,
        verify_sha256: bool = True,
    ) -> None:
        self.pair_index_path = Path(pair_index_path)
        if not self.pair_index_path.is_file():
            raise FileNotFoundError(f"Visual transition pair index not found: {self.pair_index_path}")
        self.asset_root = Path(asset_root).resolve() if asset_root is not None else None
        self.verify_sha256 = verify_sha256

        frame = pd.read_parquet(self.pair_index_path)
        missing = PAIR_INDEX_REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Pair index lacks required columns: {sorted(missing)}")
        frame = frame.loc[frame.modality.eq("streetview")].copy()
        for column in (
            "pre_state_mask",
            "post_target_mask",
            "identity_mask",
            "pre_temporal_mask",
            "post_temporal_mask",
            "transition_supervision_mask",
        ):
            frame[column] = frame[column].fillna(False).astype(bool)

        invalid_supervision = frame.transition_supervision_mask & ~(
            frame.pre_state_mask
            & frame.post_target_mask
            & frame.identity_mask
            & frame.pre_temporal_mask
            & frame.post_temporal_mask
        )
        if invalid_supervision.any():
            raise ValueError("Streetview pair index marks a temporally or spatially invalid pair supervised")
        if frame.post_target_predictor_use_allowed.fillna(True).astype(bool).any():
            raise ValueError("Post-opening streetview targets must never be allowed as predictor inputs")

        frame = frame.loc[frame.transition_supervision_mask]
        if treatment_orders is not None:
            frame = frame.loc[frame.treatment_order.isin(treatment_orders)]
        if target_event_times is not None:
            frame = frame.loc[frame.target_event_time.isin(target_event_times)]
        if frame.empty:
            raise ValueError("No supervised streetview transition pairs remain after filtering")
        self.rows = frame.sort_values(
            ["treatment_order", "target_event_time"], kind="stable"
        ).to_dict("records")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, asset_path: Any, pair_id: str) -> Path:
        if not isinstance(asset_path, str) or not asset_path.strip():
            raise ValueError(f"Missing asset_path for pair_id={pair_id}")
        source = Path(asset_path)
        if source.is_file():
            return source
        if self.asset_root is not None:
            candidates = [self.asset_root / source] if not source.is_absolute() else []
            parts = source.parts
            if "MIT_summer" in parts:
                project_relative = parts[parts.index("MIT_summer") + 1 :]
                candidates.insert(0, self.asset_root.joinpath(*project_relative))
            if source.is_absolute() and len(parts) > 1:
                candidates.append(self.asset_root.joinpath(*parts[1:]))
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(f"Streetview asset not found for pair_id={pair_id}: {asset_path}")

    def _load_asset_set(
        self,
        assets: list[dict[str, Any]],
        *,
        role: str,
        pair_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not assets:
            raise ValueError(f"Supervised pair_id={pair_id} has no {role} assets")

        images: list[torch.Tensor] = []
        gaps_months: list[float] = []
        gaps_compressed: list[float] = []
        gap_field = "age_before_opening_months" if role == "pre" else "months_after_opening"
        compressed_field = "age_before_opening_compressed"

        for asset in assets:
            try:
                gap = float(asset[gap_field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {gap_field} for pair_id={pair_id}") from exc
            if gap <= 0 or not math.isfinite(gap):
                raise ValueError(f"{role} image gap must be finite and positive for pair_id={pair_id}")

            expected_compressed = compress_time_gap_months(gap)
            if role == "pre":
                if compressed_field not in asset:
                    raise ValueError(
                        "Streetview pair index is missing per-image compressed age; rebuild it with the current pair builder"
                    )
                try:
                    stored_compressed = float(asset[compressed_field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid {compressed_field} for pair_id={pair_id}") from exc
                if abs(stored_compressed - expected_compressed) > 1e-7:
                    raise ValueError(f"Compressed streetview age does not match raw gap for pair_id={pair_id}")
                compressed = stored_compressed
            else:
                compressed = expected_compressed

            path = self._resolve_path(asset.get("asset_path"), pair_id)
            expected_sha = str(asset.get("asset_sha256") or "").strip()
            if self.verify_sha256 and expected_sha:
                actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual_sha != expected_sha:
                    raise ValueError(f"Streetview image SHA mismatch for pair_id={pair_id}: {path}")
            try:
                image = decode_image(str(path), mode=ImageReadMode.RGB).float().div_(255.0)
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError(f"Could not decode streetview image for pair_id={pair_id}: {path}") from exc
            if image.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
                image = resize(image, [IMAGE_SIZE, IMAGE_SIZE], antialias=True)
            images.append(image)
            gaps_months.append(gap)
            gaps_compressed.append(compressed)

        return (
            torch.stack(images),
            torch.tensor(gaps_months, dtype=torch.float32),
            torch.tensor(gaps_compressed, dtype=torch.float32),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        pair_id = str(row["pair_id"])
        pre_assets = _parse_assets(row["pre_state_assets_json"], "pre_state_assets_json", pair_id)
        post_assets = _parse_assets(row["post_target_assets_json"], "post_target_assets_json", pair_id)
        pre_images, pre_gap, pre_gap_compressed = self._load_asset_set(
            pre_assets, role="pre", pair_id=pair_id
        )
        post_images, post_gap, post_gap_compressed = self._load_asset_set(
            post_assets, role="post", pair_id=pair_id
        )
        return {
            "pair_id": pair_id,
            "treatment_order": int(row["treatment_order"]),
            "station_event_id": str(row["station_event_id"]),
            "target_event_time": int(row["target_event_time"]),
            "pre_images": pre_images,
            "pre_gap_months": pre_gap,
            "pre_gap_compressed": pre_gap_compressed,
            "post_images": post_images,
            "post_gap_months": post_gap,
            "post_gap_compressed": post_gap_compressed,
            "transition_supervision_mask": True,
        }


def _pad_image_set(
    samples: list[dict[str, Any]], prefix: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_key = f"{prefix}_images"
    gap_key = f"{prefix}_gap_months"
    compressed_key = f"{prefix}_gap_compressed"
    max_count = max(sample[image_key].shape[0] for sample in samples)
    batch_size = len(samples)
    images = samples[0][image_key].new_zeros(batch_size, max_count, 3, IMAGE_SIZE, IMAGE_SIZE)
    gaps = torch.zeros(batch_size, max_count, dtype=torch.float32)
    compressed = torch.zeros(batch_size, max_count, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_count, dtype=torch.bool)
    for row, sample in enumerate(samples):
        count = sample[image_key].shape[0]
        images[row, :count] = sample[image_key]
        gaps[row, :count] = sample[gap_key]
        compressed[row, :count] = sample[compressed_key]
        mask[row, :count] = True
    return images, gaps, compressed, mask


def collate_streetview_transitions(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad pre/target image sets independently and preserve per-image time gaps."""
    if not samples:
        raise ValueError("Cannot collate an empty streetview transition batch")
    pre_images, pre_gaps, pre_compressed, pre_mask = _pad_image_set(samples, "pre")
    post_images, post_gaps, post_compressed, post_mask = _pad_image_set(samples, "post")
    return {
        "pair_id": [sample["pair_id"] for sample in samples],
        "treatment_order": torch.tensor(
            [sample["treatment_order"] for sample in samples], dtype=torch.long
        ),
        "station_event_id": [sample["station_event_id"] for sample in samples],
        "target_event_time": torch.tensor(
            [sample["target_event_time"] for sample in samples], dtype=torch.long
        ),
        "pre_images": pre_images,
        "pre_gap_months": pre_gaps,
        "pre_gap_compressed": pre_compressed,
        "pre_image_mask": pre_mask,
        "post_images": post_images,
        "post_gap_months": post_gaps,
        "post_gap_compressed": post_compressed,
        "post_image_mask": post_mask,
        "transition_supervision_mask": torch.tensor(
            [sample["transition_supervision_mask"] for sample in samples], dtype=torch.bool
        ),
    }


class TimeAwareStreetviewSetEncoder(nn.Module):
    """Encode a variable-size streetview image set with learned time-aware pooling."""

    def __init__(self, embedding_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.image_encoder = StreetviewPanoramaEncoder(
            embedding_dim=embedding_dim, dropout=dropout
        )
        self.gap_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.image_time_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention_score = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1, bias=False),
        )
        self.missing_state = nn.Parameter(torch.zeros(1, embedding_dim))
        nn.init.normal_(self.missing_state, std=0.02)

    def forward(
        self,
        images: torch.Tensor,
        gap_compressed: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError("images must have shape (batch, images, 3, height, width)")
        batch_size, image_count = images.shape[:2]
        if image_count == 0:
            return functional.normalize(self.missing_state.expand(batch_size, -1), dim=-1)
        if gap_compressed.shape != (batch_size, image_count):
            raise ValueError("gap_compressed must have shape (batch, images)")
        if image_mask.shape != (batch_size, image_count):
            raise ValueError("image_mask must have shape (batch, images)")
        image_mask = image_mask.to(device=images.device, dtype=torch.bool)
        gap_compressed = gap_compressed.to(device=images.device, dtype=images.dtype)
        if not torch.isfinite(gap_compressed[image_mask]).all():
            raise ValueError("Valid images must have finite compressed time gaps")
        if ((gap_compressed[image_mask] < 0) | (gap_compressed[image_mask] >= 1)).any():
            raise ValueError("Compressed time gaps for valid images must be in [0, 1)")

        flat_mask = image_mask.reshape(-1)
        image_vectors = images.new_zeros(batch_size * image_count, self.embedding_dim)
        if flat_mask.any():
            flat_images = images.reshape(
                batch_size * image_count, *images.shape[2:]
            )[flat_mask]
            valid_vectors = self.image_encoder(images=flat_images)
            image_vectors[flat_mask] = valid_vectors
        image_vectors = image_vectors.view(batch_size, image_count, self.embedding_dim)

        gap_vectors = self.gap_encoder(gap_compressed.unsqueeze(-1))
        fused = self.image_time_fusion(torch.cat([image_vectors, gap_vectors], dim=-1))
        scores = self.attention_score(fused).squeeze(-1)
        scores = scores.masked_fill(~image_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).masked_fill(~image_mask, 0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = torch.sum(fused * weights.unsqueeze(-1), dim=1)
        has_images = image_mask.any(dim=1, keepdim=True)
        pooled = torch.where(has_images, pooled, self.missing_state.expand(batch_size, -1))
        return functional.normalize(pooled, p=2, dim=-1)
