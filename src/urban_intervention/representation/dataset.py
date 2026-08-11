"""Pretraining dataset from published model inputs.

Reads a `data/model_inputs/<dataset_id>/` release and provides per-grid feature and
response vectors for representation learning.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.functional import resize

logger = logging.getLogger(__name__)

RESPONSE_CONFIG: dict[str, dict[str, tuple[int, ...]]] = {
    "housing": {"housing_log_price": (1, 3, 6, 12, 18, 24)},
    "viirs": {"viirs_avg_asinh": (1, 3, 6, 12, 18, 24)},
    "population": {"population_log": (1, 2, 3)},
    "poi": {
        "poi_count_log": (1, 2, 3),
        "poi_category_entropy": (1, 2, 3),
        "poi_commercial_share": (1, 2, 3),
        "poi_transport_access_log": (1, 2, 3),
    },
}
RESPONSE_DIM = 27
IMAGE_SIZE = 224


def _response_offsets() -> dict[str, tuple[int, int, list[str]]]:
    offsets: dict[str, tuple[int, int, list[str]]] = {}
    cursor = 0
    for family, outcomes in RESPONSE_CONFIG.items():
        family_outcomes: list[str] = []
        family_start = cursor
        for outcome, horizons in outcomes.items():
            for h in horizons:
                family_outcomes.append(f"{outcome}__t{h}")
                cursor += 1
        offsets[family] = (family_start, cursor, family_outcomes)
    if cursor != RESPONSE_DIM:
        raise ValueError(
            f"Response dimension mismatch: RESPONSE_DIM={RESPONSE_DIM}, "
            f"computed cursor={cursor}. Update RESPONSE_CONFIG or RESPONSE_DIM."
        )
    return offsets


RESPONSE_OFFSETS = _response_offsets()
RESPONSE_COLUMNS = [col for _, _, cols in RESPONSE_OFFSETS.values() for col in cols]


@dataclass
class GridSample:
    treatment_order: int
    city_key: str
    split: str
    feature_vector: torch.Tensor
    response_vector: torch.Tensor
    response_mask: torch.Tensor
    response_se: torch.Tensor
    modality_available: dict[str, bool]
    final_training_mask: bool
    quality_grade: str
    conditioning_token: int = 0
    image_paths: list[str] = field(default_factory=list)


def _load_images(paths: list[str], max_images: int) -> tuple[torch.Tensor, torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for path in paths:
        try:
            img = decode_image(path, mode=ImageReadMode.RGB)
            img = img.float() / 255.0
            _, h, w = img.shape
            if h != IMAGE_SIZE or w != IMAGE_SIZE:
                img = resize(img, [IMAGE_SIZE, IMAGE_SIZE], antialias=True)
            tensors.append(img)
        except (OSError, RuntimeError, ValueError):
            continue
    if not tensors:
        return (
            torch.zeros(max_images, 3, IMAGE_SIZE, IMAGE_SIZE),
            torch.zeros(max_images, dtype=torch.bool),
        )
    n = min(len(tensors), max_images)
    padded = torch.zeros(max_images, 3, IMAGE_SIZE, IMAGE_SIZE)
    mask = torch.zeros(max_images, dtype=torch.bool)
    padded[:n] = torch.stack(tensors[:n])
    mask[:n] = True
    return padded, mask


class RepresentationDataset(torch.utils.data.Dataset[GridSample]):
    def __init__(
        self,
        model_inputs_dir: Path | str,
        split: str | None = None,
        max_images_per_grid: int = 4,
        load_images: bool = False,
        only_training_mask: bool = False,
    ):
        root = Path(model_inputs_dir)
        self.root = root
        self.max_images_per_grid = max_images_per_grid
        self.load_images = load_images
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema") != "urban_intervention_pretraining_dataset_v1":
            raise ValueError(f"Unsupported manifest schema: {self.manifest.get('schema')}")

        self.unit_features = pd.read_parquet(root / "unit_features.parquet")
        self.response_targets = pd.read_parquet(root / "response_targets.parquet")
        self.sample_index = pd.read_parquet(root / "sample_index.parquet")
        self.normalization = json.loads((root / "normalization.json").read_text(encoding="utf-8"))

        self._validate()
        self._build_feature_columns()
        self._build_response_tensor()

        if split is not None:
            mask = self.sample_index["split"].eq(split)
            self.sample_index = self.sample_index.loc[mask].reset_index(drop=True)
        if only_training_mask:
            self.sample_index = self.sample_index.loc[
                self.sample_index["final_training_mask"].astype(bool)
            ].reset_index(drop=True)

        self._treatment_orders = self.sample_index["treatment_order"].tolist()
        # Bounded LRU: caching the whole dataset's street-view tensors would
        # need ~12 GB for 5,048 grids (4 × 224×224×3 float32 each); evicted
        # orders are re-decoded on demand.
        self._image_cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._image_cache_max = 128
        self._feat_index = self.unit_features.set_index("treatment_order", drop=False)
        missing_records = [
            order for order in self._treatment_orders if order not in self._records.index
        ]
        if missing_records:
            shown = sorted(missing_records)[:10]
            suffix = "..." if len(missing_records) > 10 else ""
            raise ValueError(
                f"sample_index references {len(missing_records)} treatment orders "
                f"absent from response_targets: {shown}{suffix}"
            )

    def _validate(self) -> None:
        if self.sample_index["treatment_order"].duplicated().any():
            raise ValueError("sample_index has duplicate treatment_order")
        required = {"treatment_order", "split", "final_training_mask", "quality_grade"}
        missing = required - set(self.sample_index.columns)
        if missing:
            raise ValueError(f"sample_index lacks columns: {sorted(missing)}")

    def _build_feature_columns(self) -> None:
        skip_prefixes = (
            "treatment_order",
            "city_key",
            "grid_id",
            "station_event_id",
            "opening_month",
            "opening_year",
            "split",
            "streetview_assets",
        )
        skip_suffixes = ("_available",)
        # Post-processing columns written by the release publisher, not raw
        # pre-treatment features: feeding them to the model leaks the training
        # selection rule and their scale is not normalized.
        skip_exact = {"feature_training_mask", "available_modality_count"}
        z_scored = {col for col in self.unit_features.columns if col.startswith("z__")}
        self.feature_columns: list[str] = []
        for col in self.unit_features.columns:
            if col in skip_exact:
                continue
            if col.startswith(skip_prefixes):
                continue
            if any(col.endswith(suffix) for suffix in skip_suffixes):
                continue
            if "source_points" in col:
                continue
            if not col.startswith("z__") and f"z__{col}" in z_scored:
                continue
            self.feature_columns.append(col)
        self.feature_columns = sorted(self.feature_columns)

    def _build_response_tensor(self) -> None:
        lookup = self.unit_features.set_index("treatment_order")[["city_key", "split"]].to_dict(
            "index"
        )

        grouped = self.response_targets.groupby("treatment_order", sort=False)
        records: list[dict] = []
        sample_lookup = self.sample_index.set_index("treatment_order")
        for order, group in grouped:
            row: dict[str, object] = {"treatment_order": order}
            meta = lookup.get(order, {})
            row["city_key"] = str(meta.get("city_key", ""))
            row["split"] = str(meta.get("split", ""))

            if order in sample_lookup.index:
                idx_row = sample_lookup.loc[order]
                row["final_training_mask"] = bool(idx_row.get("final_training_mask", False))
                row["quality_grade"] = str(idx_row.get("quality_grade", "pending"))
            else:
                row["final_training_mask"] = False
                row["quality_grade"] = "unknown"

            response = np.full(RESPONSE_DIM, np.nan, dtype=np.float64)
            response_se = np.full(RESPONSE_DIM, np.nan, dtype=np.float64)

            for family, (start, end, cols) in RESPONSE_OFFSETS.items():
                del end
                family_group = group[group["outcome_family"].eq(family)]
                for _, task_row in family_group.iterrows():
                    outcome = str(task_row["outcome"])
                    horizon = int(task_row["event_time"])
                    col_name = f"{outcome}__t{horizon}"
                    if col_name in cols:
                        idx = cols.index(col_name) + start
                        response[idx] = pd.to_numeric(
                            task_row.get("causal_response_label", np.nan), errors="coerce"
                        )
                        response_se[idx] = pd.to_numeric(
                            task_row.get("standard_error", np.nan), errors="coerce"
                        )
                    else:
                        logger.warning(
                            "Skipping unknown outcome/horizon: %s in family %s",
                            col_name,
                            family,
                        )

            row["response"] = response.astype(np.float32)
            row["response_se"] = response_se.astype(np.float32)
            records.append(row)

        self._records = pd.DataFrame(records)
        self._records.set_index("treatment_order", inplace=True)

    def _build_modality_available(self, treatment_order: int) -> dict[str, bool]:
        modality_names = ["housing", "poi", "viirs", "population", "sentinel2", "streetview"]
        result: dict[str, bool] = {}
        if treatment_order not in self._feat_index.index:
            return {name: False for name in modality_names}
        row = self._feat_index.loc[treatment_order]
        for name in modality_names:
            col = f"{name}_available"
            if col in row:
                val = row[col]
                result[name] = bool(val) if pd.notna(val) else False
            else:
                result[name] = False
        return result

    def _get_image_paths(self, order: int) -> list[str]:
        if (
            order not in self._feat_index.index
            or "streetview_assets" not in self.unit_features.columns
        ):
            return []
        raw = str(self._feat_index.loc[order]["streetview_assets"])
        if raw in ("", "[]", "null", "None"):
            return []
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []

    def _get_images(self, order: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.load_images:
            return (
                torch.zeros(self.max_images_per_grid, 3, IMAGE_SIZE, IMAGE_SIZE),
                torch.zeros(self.max_images_per_grid, dtype=torch.bool),
            )
        if order in self._image_cache:
            self._image_cache.move_to_end(order)
            return self._image_cache[order]
        paths = self._get_image_paths(order)
        result = _load_images(paths, self.max_images_per_grid)
        if len(self._image_cache) >= self._image_cache_max:
            self._image_cache.popitem(last=False)
        self._image_cache[order] = result
        return result

    def __len__(self) -> int:
        return len(self._treatment_orders)

    def __getitem__(self, idx: int) -> GridSample:
        order = self._treatment_orders[idx]
        features = np.zeros(len(self.feature_columns), dtype=np.float32)
        conditioning_token = 0
        if order in self._feat_index.index:
            feat_row = self._feat_index.loc[order]
            for i, col in enumerate(self.feature_columns):
                if col in feat_row:
                    val = feat_row[col]
                    if pd.notna(val):
                        numeric_val = pd.to_numeric(val, errors="coerce")
                        features[i] = float(numeric_val) if pd.notna(numeric_val) else 0.0
            opening_year = feat_row.get("opening_year", pd.NA)
            if pd.notna(opening_year):
                conditioning_token = int(
                    min(max(int(pd.to_numeric(opening_year, errors="coerce")) - 2005, 0), 29)
                )

        rec = self._records.loc[order]
        response = rec["response"]
        response_mask = ~np.isnan(response)
        response_se = rec["response_se"]

        return GridSample(
            treatment_order=order,
            city_key=str(rec["city_key"]),
            split=str(rec["split"]),
            feature_vector=torch.from_numpy(features),
            response_vector=torch.from_numpy(response),
            response_mask=torch.from_numpy(response_mask),
            response_se=torch.from_numpy(response_se),
            modality_available=self._build_modality_available(order),
            final_training_mask=bool(rec["final_training_mask"]),
            quality_grade=str(rec["quality_grade"]),
            conditioning_token=conditioning_token,
            image_paths=self._get_image_paths(order),
        )

    def response_similarity(
        self, idx_a: int, idx_b: int, within_family: str | None = None
    ) -> float:
        sample_a = self[idx_a]
        sample_b = self[idx_b]
        families = [within_family] if within_family else list(RESPONSE_OFFSETS)
        similarities: list[float] = []
        for family in families:
            if family not in RESPONSE_OFFSETS:
                continue
            start, _, _ = RESPONSE_OFFSETS[family]
            end = RESPONSE_OFFSETS[family][1]
            a = sample_a.response_vector[start:end]
            b = sample_b.response_vector[start:end]
            mask = sample_a.response_mask[start:end] & sample_b.response_mask[start:end]
            common = int(mask.sum().item())
            if common < 2:
                continue
            a_valid = a[mask]
            b_valid = b[mask]
            a_mean, b_mean = a_valid.mean(), b_valid.mean()
            a_std, b_std = a_valid.std(), b_valid.std()
            if a_std < 1e-10 or b_std < 1e-10:
                continue
            a_norm = (a_valid - a_mean) / a_std
            b_norm = (b_valid - b_mean) / b_std
            corr = float((a_norm * b_norm).sum() / (len(a_valid) - 1))
            similarities.append(corr)
        if not similarities:
            return 0.0
        return sum(similarities) / len(similarities)

    def feature_dim(self) -> int:
        return len(self.feature_columns)

    def response_dim(self) -> int:
        return RESPONSE_DIM

    def response_offsets(self) -> dict[str, tuple[int, int, list[str]]]:
        return dict(RESPONSE_OFFSETS)

    def treatment_orders(self) -> list[int]:
        return list(self._treatment_orders)

    def training_orders(self) -> list[int]:
        return [
            o
            for o in self._treatment_orders
            if bool(self._records.loc[o].get("final_training_mask", False))
        ]


def collate_samples(
    samples: list[GridSample],
    load_images_fn=None,
    max_images_per_grid: int = 4,
    use_images: bool = False,
) -> dict[str, object]:
    batch: dict[str, object] = {
        "treatment_order": torch.tensor([s.treatment_order for s in samples], dtype=torch.long),
        "features": torch.stack([s.feature_vector for s in samples]),
        "responses": torch.stack([s.response_vector for s in samples]),
        "response_mask": torch.stack([s.response_mask for s in samples]),
        "response_se": torch.stack([s.response_se for s in samples]),
        "conditioning_tokens": torch.tensor(
            [s.conditioning_token for s in samples], dtype=torch.long
        ),
        "final_training_mask": torch.tensor(
            [s.final_training_mask for s in samples], dtype=torch.bool
        ),
        "city_keys": [s.city_key for s in samples],
        "splits": [s.split for s in samples],
        "quality_grades": [s.quality_grade for s in samples],
        "modality_available": [s.modality_available for s in samples],
    }

    if use_images and load_images_fn is not None:
        images_list: list[torch.Tensor] = []
        masks_list: list[torch.Tensor] = []
        for s in samples:
            imgs, img_mask = load_images_fn(s.treatment_order)
            images_list.append(imgs)
            masks_list.append(img_mask)
        batch["images"] = torch.stack(images_list)
        batch["image_mask"] = torch.stack(masks_list)

    return batch
