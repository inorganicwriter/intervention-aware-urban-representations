"""Pretraining dataset from published model inputs.

Reads a `data/model_inputs/<dataset_id>/` release and provides per-grid feature and
response vectors for representation learning.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.functional import resize

from ..causal.pretraining_dataset import (
    STC_CONTEXT_TABULAR_COLUMNS,
    STC_INTERVENTION_COLUMNS,
    STC_STATE_TABULAR_COLUMNS,
)
from .time_encoding import compress_time_gap_months

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
_IMAGE_SHA_CACHE: dict[str, str] = {}
_REPO_ROOT = Path(__file__).resolve().parents[3]
_STREETVIEW_REGISTRY: dict[str, dict[str, object]] | None = None


def _load_streetview_registry(
    custom_path: Path | None = None,
    dataset_root: Path | None = None,
    raise_if_missing: bool = False,
) -> dict[str, dict[str, object]]:
    global _STREETVIEW_REGISTRY
    if _STREETVIEW_REGISTRY is not None and custom_path is None and dataset_root is None:
        return _STREETVIEW_REGISTRY

    candidates: list[Path] = []
    if custom_path is not None:
        candidates.append(Path(custom_path))
    if dataset_root is not None:
        candidates.append(Path(dataset_root) / "streetview_state_assets.parquet")

    # Locate formal active datasets via MIT_PROJECT_ROOT / data.paths.PROJECT_ROOT
    env_root = os.environ.get("MIT_PROJECT_ROOT")
    if env_root:
        p_env = Path(env_root).resolve()
        candidates.extend([
            p_env / "data/active/streetview_grid_v2/streetview_state_assets.parquet",
            p_env / "data/active/streetview_fixed_treated_grid_v2/streetview_state_assets.parquet",
        ])

    try:
        from urban_intervention.data.paths import ACTIVE_DIR, PROJECT_ROOT

        candidates.extend([
            ACTIVE_DIR / "streetview_grid_v2/streetview_state_assets.parquet",
            ACTIVE_DIR / "streetview_fixed_treated_grid_v2/streetview_state_assets.parquet",
            PROJECT_ROOT / "data/active/streetview_grid_v2/streetview_state_assets.parquet",
            PROJECT_ROOT / "data/active/streetview_fixed_treated_grid_v2/streetview_state_assets.parquet",
        ])
    except Exception:
        pass

    candidates.extend([
        _REPO_ROOT / "data/active/streetview_grid_v2/streetview_state_assets.parquet",
        _REPO_ROOT / "data/active/streetview_fixed_treated_grid_v2/streetview_state_assets.parquet",
    ])

    seen = set()
    unique_candidates: list[Path] = []
    for cand in candidates:
        try:
            cand_key = str(cand.resolve()) if cand.exists() else str(cand)
        except Exception:
            cand_key = str(cand)
        if cand_key not in seen:
            seen.add(cand_key)
            unique_candidates.append(cand)

    table_path = None
    for cand in unique_candidates:
        if cand.is_file():
            table_path = cand
            break

    if table_path is None and raise_if_missing:
        raise FileNotFoundError(
            "Streetview asset registry (streetview_state_assets.parquet) not found. "
            f"Searched candidates: {[str(c) for c in unique_candidates]}. "
            "Ensure streetview_state_assets.parquet is packaged in the dataset, "
            "provide streetview_index_path, or configure MIT_PROJECT_ROOT."
        )

    registry: dict[str, dict[str, object]] = {}
    if table_path is not None:
        df = pd.read_parquet(table_path)
        for _, row in df.iterrows():
            entry = {
                "treatment_order": int(row["treatment_order"]) if pd.notna(row.get("treatment_order")) else None,
                "city_key": str(row.get("city_key", "")),
                "grid_id": str(row.get("grid_id", "")),
                "opening_month": str(row.get("opening_month", ""))[:7],
                "capture_month": str(row.get("capture_month", ""))[:7],
                "asset_sha256": str(row.get("asset_sha256", "")),
                "asset_path": str(row.get("asset_path", "")),
            }
            p_str = str(row.get("asset_path", ""))
            if p_str:
                p_obj = Path(p_str)
                registry[str(p_obj)] = entry
                registry[p_obj.name] = entry
                with contextlib.suppress(Exception):
                    registry[str(p_obj.resolve())] = entry
            panoid = str(row.get("panoid", ""))
            if panoid:
                registry[panoid] = entry

    if custom_path is None and dataset_root is None:
        _STREETVIEW_REGISTRY = registry
    return registry


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


def compute_response_offsets(
    families: tuple[str, ...] | list[str] | None = None,
) -> tuple[dict[str, tuple[int, int, list[str]]], list[str], int]:
    selected_families = families if families is not None else list(RESPONSE_CONFIG.keys())
    offsets: dict[str, tuple[int, int, list[str]]] = {}
    cursor = 0
    all_columns: list[str] = []
    for family in selected_families:
        if family not in RESPONSE_CONFIG:
            continue
        outcomes = RESPONSE_CONFIG[family]
        family_cols: list[str] = []
        family_start = cursor
        for outcome, horizons in outcomes.items():
            for h in horizons:
                col_name = f"{outcome}__t{h}"
                family_cols.append(col_name)
                all_columns.append(col_name)
                cursor += 1
        offsets[family] = (family_start, cursor, family_cols)
    return offsets, all_columns, cursor


@dataclass
class GridSample:
    treatment_order: int
    city_key: str
    split: str
    feature_vector: torch.Tensor
    response_vector: torch.Tensor
    response_mask: torch.Tensor
    response_se: torch.Tensor
    response_weight: torch.Tensor
    modality_available: dict[str, bool]
    final_training_mask: bool
    quality_grade: str
    conditioning_token: int = 0
    station_token: int = 0
    image_paths: list[str] = field(default_factory=list)
    satellite_vector: torch.Tensor = field(default_factory=lambda: torch.zeros(64, dtype=torch.float32))
    satellite_available: bool = False
    satellite_is_temporal_proxy: bool = False
    satellite_target_year: float = float("nan")
    # S/T/C multimodal fields
    is_stc: bool = False
    state_tabular: torch.Tensor = field(default_factory=lambda: torch.zeros(15, dtype=torch.float32))
    state_field_mask: torch.Tensor = field(default_factory=lambda: torch.zeros(15, dtype=torch.bool))
    streetview_features: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 384, dtype=torch.float32))
    streetview_mask: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.bool))
    streetview_images: torch.Tensor | None = None
    streetview_gap_months: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.float32))
    streetview_gap_compressed: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.float32))
    satellite_images: torch.Tensor = field(default_factory=lambda: torch.zeros(6, 6, 17, 17, dtype=torch.float32))
    satellite_pixel_mask: torch.Tensor = field(default_factory=lambda: torch.zeros(6, 1, 17, 17, dtype=torch.bool))
    satellite_year_mask: torch.Tensor = field(default_factory=lambda: torch.zeros(6, dtype=torch.bool))
    satellite_event_times: torch.Tensor = field(
        default_factory=lambda: torch.tensor([-6, -5, -4, -3, -2, -1], dtype=torch.long)
    )
    context_tabular: torch.Tensor = field(default_factory=lambda: torch.zeros(12, dtype=torch.float32))
    context_field_mask: torch.Tensor = field(default_factory=lambda: torch.zeros(12, dtype=torch.bool))
    alphaearth: torch.Tensor = field(default_factory=lambda: torch.zeros(64, dtype=torch.float32))
    alphaearth_available: bool = False
    intervention: torch.Tensor = field(default_factory=lambda: torch.zeros(7, dtype=torch.float32))
    intervention_mask: torch.Tensor = field(default_factory=lambda: torch.zeros(7, dtype=torch.bool))


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
        station_attributes_path: Path | str | None = None,
        station_schema: str = "v2",
        satellite_embeddings_path: Path | str | None = None,
        allow_satellite_proxy: bool = False,
        require_multimodal: bool = False,
        require_satellite: bool = False,
        require_streetview: bool = False,
        load_satellite_images: bool | None = None,
        satellite_image_size: tuple[int, int] = (17, 17),
        satellite_root: Path | str | None = None,
        verify_bundle: bool = False,
        streetview_index_path: Path | str | None = None,
    ):
        root = Path(model_inputs_dir)
        if station_schema not in {"v1", "v2"}:
            raise ValueError(f"Unknown station_schema: {station_schema!r}")
        self.root = root
        self.verify_bundle = bool(verify_bundle)
        self.allow_satellite_proxy = allow_satellite_proxy
        self.require_multimodal = require_multimodal
        self.require_satellite = require_satellite or require_multimodal
        self.require_streetview = require_streetview or require_multimodal
        self.max_images_per_grid = max_images_per_grid
        # A mandatory street-view/multimodal cohort must actually decode the
        # images; otherwise the model's street-view branch only sees the
        # missing-modality token even though the flags are enabled.
        self.load_images = bool(load_images or require_streetview or require_multimodal)
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        schema = str(self.manifest.get("schema", ""))
        self.is_stc = schema in {"urban_intervention_stc_dataset", "urban_intervention_stc_dataset_v1"}
        if not self.is_stc and schema != "urban_intervention_pretraining_dataset" and not schema.startswith(
            "urban_intervention_pretraining_dataset_v"
        ):
            raise ValueError(f"Unsupported manifest schema: {self.manifest.get('schema')}")

        if streetview_index_path is not None:
            self.streetview_index_path = Path(streetview_index_path)
        elif (root / "streetview_state_assets.parquet").is_file():
            self.streetview_index_path = root / "streetview_state_assets.parquet"
        else:
            self.streetview_index_path = None

        if self.streetview_index_path is not None and self.streetview_index_path.is_file():
            self.streetview_assets_sha256 = hashlib.sha256(self.streetview_index_path.read_bytes()).hexdigest()
        else:
            self.streetview_assets_sha256 = self.manifest.get("streetview_assets_sha256") or self.manifest.get("outputs", {}).get("streetview_state_assets.parquet")
        self._streetview_registry: dict[str, dict[str, object]] | None = None

        if load_satellite_images is None:
            self.load_satellite_images = bool(self.is_stc)
        else:
            self.load_satellite_images = bool(load_satellite_images)
        self.satellite_image_size = satellite_image_size
        if satellite_root is not None:
            self.satellite_root = Path(satellite_root).resolve()
        else:
            default_sat_root = None
            if (root / "satellite_event_sequences_v2").is_dir():
                default_sat_root = (root / "satellite_event_sequences_v2").resolve()
            else:
                env_root = os.environ.get("MIT_PROJECT_ROOT")
                if env_root and (Path(env_root) / "data/active/satellite_event_sequences_v2").is_dir():
                    default_sat_root = (Path(env_root) / "data/active/satellite_event_sequences_v2").resolve()
                else:
                    try:
                        from urban_intervention.data.paths import ACTIVE_DIR, PROJECT_ROOT

                        if (PROJECT_ROOT / "data/active/satellite_event_sequences_v2").is_dir():
                            default_sat_root = (PROJECT_ROOT / "data/active/satellite_event_sequences_v2").resolve()
                        elif ACTIVE_DIR.is_dir() and (ACTIVE_DIR / "satellite_event_sequences_v2").is_dir():
                            default_sat_root = (ACTIVE_DIR / "satellite_event_sequences_v2").resolve()
                    except Exception:
                        pass
                if default_sat_root is None and (_REPO_ROOT / "data/active/satellite_event_sequences_v2").is_dir():
                    default_sat_root = (_REPO_ROOT / "data/active/satellite_event_sequences_v2").resolve()
            self.satellite_root = default_sat_root
        self._image_cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._image_cache_max = 128

        if self.is_stc:
            self.state_inputs = pd.read_parquet(root / "state_inputs.parquet")
            self.context_inputs = pd.read_parquet(root / "context_inputs.parquet")
            self.intervention_inputs = pd.read_parquet(root / "intervention_inputs.parquet")
            self.satellite_sequences = pd.read_parquet(root / "satellite_state_sequences.parquet")
            self.response_targets = pd.read_parquet(root / "response_targets.parquet")
            self.sample_index = pd.read_parquet(root / "sample_index.parquet")
            self.normalization = json.loads((root / "normalization.json").read_text(encoding="utf-8"))

            self.state_tabular_columns = self.manifest.get("state_tabular_columns") or list(STC_STATE_TABULAR_COLUMNS)
            self.intervention_columns = self.manifest.get("intervention_columns") or list(STC_INTERVENTION_COLUMNS)
            self.context_tabular_columns = self.manifest.get("context_tabular_columns") or list(STC_CONTEXT_TABULAR_COLUMNS)
            self.satellite_relative_years = self.manifest.get("satellite_relative_years") or [-6, -5, -4, -3, -2, -1]
            self.families = self.manifest.get("families")

            if self.families:
                self._response_offsets, self.response_columns, self._response_dim = compute_response_offsets(self.families)
            else:
                self._response_offsets = RESPONSE_OFFSETS
                self.response_columns = RESPONSE_COLUMNS
                self._response_dim = RESPONSE_DIM

            self._validate_stc()
            if split is not None:
                self.sample_index = self.sample_index.loc[self.sample_index["split"].eq(split)].reset_index(drop=True)
            if only_training_mask:
                self.sample_index = self.sample_index.loc[
                    self.sample_index["final_training_mask"].fillna(False).astype(bool)
                ].reset_index(drop=True)

            self._treatment_orders = self.sample_index["treatment_order"].tolist()
            self._build_stc_lookup()
            if self.require_satellite:
                valid_orders = {
                    order for order in self._treatment_orders if self._has_stc_satellite(order)
                }
                self.sample_index = self.sample_index.loc[
                    self.sample_index["treatment_order"].isin(valid_orders)
                ].reset_index(drop=True)
                self._treatment_orders = self.sample_index["treatment_order"].tolist()
            if self.require_streetview:
                valid_orders = {
                    order
                    for order in self._treatment_orders
                    if len(self._stc_streetview_assets.get(order, [])) > 0
                }
                self.sample_index = self.sample_index.loc[
                    self.sample_index["treatment_order"].isin(valid_orders)
                ].reset_index(drop=True)
                self._treatment_orders = self.sample_index["treatment_order"].tolist()
            if not self._treatment_orders:
                raise ValueError(
                    "No STC units satisfy the requested mandatory modality filters"
                )
            self._build_stc_response_tensor()
            self._station_tokens = {}
            self.station_attributes_binding = None
            self.feature_columns = list(self.state_tabular_columns)
            return

        self.unit_features = pd.read_parquet(root / "unit_features.parquet")
        self.response_targets = pd.read_parquet(root / "response_targets.parquet")
        self.sample_index = pd.read_parquet(root / "sample_index.parquet")
        self.normalization = json.loads((root / "normalization.json").read_text(encoding="utf-8"))

        self._validate()
        self._treatment_orders = self.sample_index["treatment_order"].tolist()
        self._build_feature_columns()
        self._build_response_tensor()
        self._station_tokens: dict[int, int] = {}
        self.station_attributes_binding: dict[str, object] | None = None
        if station_attributes_path is not None:
            # Hash exactly the bytes parsed, so the identity cannot diverge
            # from the tokens if the external file is replaced during loading.
            attribute_bytes = Path(station_attributes_path).read_bytes()
            attributes = pd.read_parquet(io.BytesIO(attribute_bytes))
            required_attr = {
                "treatment_order",
                "is_transfer_at_opening",
                "is_new_line_opening",
                "is_extension_opening",
                "is_terminal_at_opening",
            }
            if not required_attr.issubset(attributes.columns):
                raise ValueError(
                    f"Station attributes lack columns: {sorted(required_attr - set(attributes.columns))}"
                )
            attr_frame = attributes[list(required_attr)].copy()
            attr_frame["treatment_order"] = pd.to_numeric(
                attr_frame["treatment_order"], errors="coerce"
            )
            # The attributes table covers every canonical station; only rows
            # that carry a treatment_order can join dataset units.
            attr_frame = attr_frame.loc[attr_frame["treatment_order"].notna()].copy()
            attr_frame["treatment_order"] = attr_frame["treatment_order"].astype(int)
            if attr_frame["treatment_order"].duplicated().any():
                raise ValueError("Station attributes have duplicate treatment_order")
            terminal_values = pd.to_numeric(attr_frame["is_terminal_at_opening"], errors="coerce")
            attr_frame["_terminal_missing"] = terminal_values.isna().astype(int)
            for column in (
                "is_transfer_at_opening",
                "is_new_line_opening",
                "is_extension_opening",
                "is_terminal_at_opening",
            ):
                attr_frame[column] = (
                    pd.to_numeric(attr_frame[column], errors="coerce").fillna(0).astype(int)
                )
            if station_schema == "v1":
                # Legacy 4-bit codes: terminal-missing is not represented.
                attr_frame["_code"] = (
                    attr_frame["is_transfer_at_opening"] * 8
                    + attr_frame["is_new_line_opening"] * 4
                    + attr_frame["is_extension_opening"] * 2
                    + attr_frame["is_terminal_at_opening"]
                )
            else:
                # Four attribute bits plus one missing indicator: an unknown
                # terminal status must not be conflated with "not terminal".
                attr_frame["_code"] = (
                    attr_frame["is_transfer_at_opening"] * 16
                    + attr_frame["is_new_line_opening"] * 8
                    + attr_frame["is_extension_opening"] * 4
                    + attr_frame["is_terminal_at_opening"] * 2
                    + attr_frame["_terminal_missing"]
                )
            self._station_tokens = dict(
                zip(
                    attr_frame["treatment_order"].astype(int),
                    attr_frame["_code"].astype(int),
                    strict=False,
                )
            )
            orders = sorted(int(order) for order in self._treatment_orders)
            token_rows = [
                [order, self._station_tokens.get(order, 0), order in self._station_tokens]
                for order in orders
            ]
            self.station_attributes_binding = {
                "sha256": hashlib.sha256(attribute_bytes).hexdigest(),
                "schema": station_schema,
                "token_sha256": hashlib.sha256(
                    json.dumps(token_rows, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "dataset_units": len(orders),
                "covered_units": sum(order in self._station_tokens for order in orders),
            }

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

        # Satellite AlphaEarth embedding vector loading
        self._satellite_vectors: dict[int, torch.Tensor] = {}
        self._satellite_is_proxy: dict[int, bool] = {}
        self._satellite_target_year: dict[int, float] = {}
        bands = [f"A{i:02d}" for i in range(64)]
        if set(bands).issubset(self.unit_features.columns) and "treatment_order" in self.unit_features.columns:
            has_proxy_col = "satellite_is_temporal_proxy" in self.unit_features.columns or "is_temporal_proxy" in self.unit_features.columns
            proxy_col = "satellite_is_temporal_proxy" if "satellite_is_temporal_proxy" in self.unit_features.columns else "is_temporal_proxy"
            has_ty_col = "satellite_target_year" in self.unit_features.columns or "target_year" in self.unit_features.columns
            ty_col = "satellite_target_year" if "satellite_target_year" in self.unit_features.columns else "target_year"
            has_avail_col = "alphaearth_available" in self.unit_features.columns

            for _, s_row in self.unit_features.iterrows():
                order = int(s_row["treatment_order"])
                is_proxy = bool(s_row[proxy_col]) if has_proxy_col and pd.notna(s_row[proxy_col]) else False
                ty_val = pd.to_numeric(s_row[ty_col], errors="coerce") if has_ty_col else float("nan")
                target_yr = float(ty_val) if pd.notna(ty_val) else float("nan")
                self._satellite_is_proxy[order] = is_proxy
                self._satellite_target_year[order] = target_yr

                if is_proxy and not self.allow_satellite_proxy:
                    continue
                if has_avail_col and not bool(s_row["alphaearth_available"]):
                    continue

                vals = [float(s_row[b]) if pd.notna(s_row[b]) else 0.0 for b in bands]
                if any(v != 0.0 for v in vals):
                    self._satellite_vectors[order] = torch.tensor(
                        vals, dtype=torch.float32
                    )

        if not self._satellite_vectors:
            sat_path = Path(satellite_embeddings_path) if satellite_embeddings_path is not None else None
            if sat_path is None:
                candidate_paths = [
                    root / "alphaearth_treated_embeddings.parquet",
                    root.parents[1] / "alphaearth" / "alphaearth_treated_embeddings.parquet",
                ]
                for candidate in candidate_paths:
                    if candidate.is_file():
                        sat_path = candidate
                        break
            if sat_path is not None and sat_path.is_file():
                try:
                    sat_df = pd.read_parquet(sat_path)
                    has_proxy_col = "is_temporal_proxy" in sat_df.columns
                    has_ty_col = "target_year" in sat_df.columns
                    has_avail_col = "alphaearth_available" in sat_df.columns

                    if set(bands).issubset(sat_df.columns) and "treatment_order" in sat_df.columns:
                        for _, s_row in sat_df.iterrows():
                            order = int(s_row["treatment_order"])
                            is_proxy = bool(s_row["is_temporal_proxy"]) if has_proxy_col and pd.notna(s_row["is_temporal_proxy"]) else False
                            ty_val = pd.to_numeric(s_row["target_year"], errors="coerce") if has_ty_col else float("nan")
                            target_yr = float(ty_val) if pd.notna(ty_val) else float("nan")
                            self._satellite_is_proxy[order] = is_proxy
                            self._satellite_target_year[order] = target_yr

                            if is_proxy and not self.allow_satellite_proxy:
                                continue
                            if has_avail_col and not bool(s_row["alphaearth_available"]):
                                continue

                            vals = [float(s_row[b]) if pd.notna(s_row[b]) else 0.0 for b in bands]
                            if any(v != 0.0 for v in vals):
                                self._satellite_vectors[order] = torch.tensor(
                                    vals, dtype=torch.float32
                                )
                except Exception as exc:
                    logger.warning("Failed to load satellite embeddings from %s: %s", sat_path, exc)

        if self.require_satellite:
            sat_valid_orders = {
                order for order, vec in self._satellite_vectors.items()
                if torch.is_tensor(vec) and (vec != 0).any()
            }
            self.sample_index = self.sample_index.loc[
                self.sample_index["treatment_order"].isin(sat_valid_orders)
            ].reset_index(drop=True)
            self._treatment_orders = self.sample_index["treatment_order"].tolist()

        if self.require_streetview:
            streetview_valid_orders = {
                order for order in self.sample_index["treatment_order"]
                if len(self._get_image_paths(order)) > 0
            }
            self.sample_index = self.sample_index.loc[
                self.sample_index["treatment_order"].isin(streetview_valid_orders)
            ].reset_index(drop=True)
            self._treatment_orders = self.sample_index["treatment_order"].tolist()

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

    def _validate_stc(self) -> None:
        required_sample_cols = {"treatment_order", "city_key", "split", "final_training_mask"}
        missing_sample = required_sample_cols - set(self.sample_index.columns)
        if missing_sample:
            raise ValueError(f"sample_index lacks required columns: {sorted(missing_sample)}")
        if self.sample_index["treatment_order"].duplicated().any():
            raise ValueError("sample_index has duplicate treatment_order")

        sample_orders = set(self.sample_index["treatment_order"])
        sample_lookup = self.sample_index.set_index("treatment_order")

        required_id_cols = ["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"]
        for name, df in [
            ("state_inputs", self.state_inputs),
            ("context_inputs", self.context_inputs),
            ("intervention_inputs", self.intervention_inputs),
        ]:
            missing_id = set(required_id_cols) - set(df.columns)
            if missing_id:
                raise ValueError(f"{name} lacks identity columns: {sorted(missing_id)}")
            if df["treatment_order"].duplicated().any():
                raise ValueError(f"{name} has duplicate treatment_order")
            if set(df["treatment_order"]) != sample_orders:
                raise ValueError(f"{name} treatment_orders do not match sample_index")

            df_idx = df.set_index("treatment_order")
            if not (df_idx["city_key"].astype(str) == sample_lookup.loc[df_idx.index, "city_key"].astype(str)).all():
                raise ValueError(f"{name} city_key does not match sample_index")

        state_idx = self.state_inputs.set_index("treatment_order")
        context_idx = self.context_inputs.set_index("treatment_order")
        interv_idx = self.intervention_inputs.set_index("treatment_order")

        for col in ("city_key", "grid_id", "station_event_id"):
            if not (state_idx[col].astype(str) == context_idx.loc[state_idx.index, col].astype(str)).all():
                raise ValueError(f"context_inputs identity column {col} does not match state_inputs")
            if not (state_idx[col].astype(str) == interv_idx.loc[state_idx.index, col].astype(str)).all():
                raise ValueError(f"intervention_inputs identity column {col} does not match state_inputs")

        state_month = state_idx["opening_month"].astype(str).str[:7]
        context_month = context_idx.loc[state_idx.index, "opening_month"].astype(str).str[:7]
        interv_month = interv_idx.loc[state_idx.index, "opening_month"].astype(str).str[:7]
        if not (state_month == context_month).all():
            raise ValueError("context_inputs opening_month does not match state_inputs")
        if not (state_month == interv_month).all():
            raise ValueError("intervention_inputs opening_month does not match state_inputs")

        # Verify physical file integrity on disk against manifest outputs when strict or requested
        if bool(self.manifest.get("strict_production", False)) or getattr(self, "verify_bundle", False):
            manifest_outputs = self.manifest.get("outputs")
            if isinstance(manifest_outputs, dict):
                for fname, expected_hash in manifest_outputs.items():
                    fpath = self.root / fname
                    if fpath.is_file():
                        actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
                        if actual_hash != str(expected_hash):
                            raise ValueError(
                                f"Dataset file {fname} SHA256 on disk ({actual_hash}) does not match manifest outputs ({expected_hash})"
                            )


    def _resolve_satellite_image_path(self, path_str: str) -> Path | None:
        p = Path(path_str)
        if p.is_file():
            return p.resolve()

        search_dirs: list[Path] = []
        if self.satellite_root is not None:
            search_dirs.append(self.satellite_root)
            if self.satellite_root.name != "images" and (self.satellite_root / "images").is_dir():
                search_dirs.append(self.satellite_root / "images")
            if self.satellite_root.name == "images":
                search_dirs.append(self.satellite_root.parent)

        search_dirs.append(self.root)
        if (self.root / "satellite_event_sequences_v2").is_dir():
            search_dirs.append(self.root / "satellite_event_sequences_v2")

        env_root = os.environ.get("MIT_PROJECT_ROOT")
        if env_root:
            env_p = Path(env_root).resolve()
            search_dirs.append(env_p)
            search_dirs.append(env_p / "data/active/satellite_event_sequences_v2")
            search_dirs.append(env_p / "data/active/satellite_event_sequences_v2/images")

        try:
            from urban_intervention.data.paths import ACTIVE_DIR, PROJECT_ROOT

            search_dirs.append(PROJECT_ROOT)
            search_dirs.append(PROJECT_ROOT / "data/active/satellite_event_sequences_v2")
            search_dirs.append(ACTIVE_DIR / "satellite_event_sequences_v2")
            search_dirs.append(ACTIVE_DIR)
        except Exception:
            pass

        search_dirs.append(_REPO_ROOT)
        search_dirs.append(_REPO_ROOT / "data/active/satellite_event_sequences_v2")

        sub_paths = [p]
        norm_str = path_str.replace("\\", "/")
        prefix = "data/active/satellite_event_sequences_v2/"
        if norm_str.startswith(prefix):
            sub_paths.append(Path(norm_str[len(prefix):]))
        if norm_str.startswith("images/"):
            sub_paths.append(Path(norm_str[len("images/"): ]))

        for base in search_dirs:
            for sub in sub_paths:
                cand = base / sub
                if cand.is_file():
                    return cand.resolve()

        return None

    def _read_satellite_tiff(self, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = self.satellite_image_size
        try:
            import rasterio
            with rasterio.open(path) as src:
                # Bands 1..6: reflectance
                bands = src.read([1, 2, 3, 4, 5, 6]).astype(np.float32)
                if src.count >= 7:
                    p_mask = src.read(7).astype(np.float32) > 0
                else:
                    p_mask = np.ones((src.height, src.width), dtype=bool)
                img = torch.from_numpy(bands)
                raw_mask = torch.from_numpy(p_mask).unsqueeze(0)

                # Mask out invalid, infinite or NaN raw pixels immediately before resampling
                valid_finite = torch.isfinite(img).all(dim=0, keepdim=True)
                valid_raw = (raw_mask & valid_finite).float()
                img_clean = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0) * valid_raw

                if img.shape[1] == H and img.shape[2] == W:
                    return img_clean, valid_raw.bool()

                # Normalized convolution resampling:
                weighted_img = resize(img_clean, [H, W], antialias=True)
                weight = resize(valid_raw, [H, W], antialias=True)
                out_mask = weight >= 0.5
                norm_img = torch.where(
                    out_mask,
                    weighted_img / weight.clamp(min=1e-6),
                    torch.zeros_like(weighted_img),
                )
                return norm_img, out_mask
        except Exception as exc:
            logger.debug("Failed reading satellite tiff %s: %s", path, exc)
            return torch.zeros(6, H, W, dtype=torch.float32), torch.zeros(1, H, W, dtype=torch.bool)

    def _get_satellite_sequence(
        self, order: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_years = len(self.satellite_relative_years)
        H, W = self.satellite_image_size
        images = torch.zeros(num_years, 6, H, W, dtype=torch.float32)
        pixel_masks = torch.zeros(num_years, 1, H, W, dtype=torch.bool)
        year_mask = torch.zeros(num_years, dtype=torch.bool)
        event_times = torch.tensor(self.satellite_relative_years, dtype=torch.long)

        seq_df = self._stc_sat_seq_dict.get(order)
        if seq_df is None or not self.load_satellite_images:
            return images, pixel_masks, year_mask, event_times
        strict_satellite = bool(
            self.require_satellite or self.manifest.get("strict_production", False)
        )

        for i, rel_yr in enumerate(self.satellite_relative_years):
            if rel_yr in seq_df.index:
                row = seq_df.loc[rel_yr]
                state_allowed = bool(row.get("state_temporal_allowed", False))
                if rel_yr >= 0 or not state_allowed:
                    continue
                is_avail = bool(row.get("state_image_available", False)) or bool(row.get("satellite_image_available", False))
                if not is_avail:
                    continue
                # Temporal boundary verification against opening year
                opening_year = self._stc_opening_years.get(order)
                row_ty = row.get("target_year")
                if pd.notna(row_ty):
                    target_year = int(row_ty)
                    if opening_year is not None:
                        if target_year >= opening_year or target_year != opening_year + rel_yr:
                            continue
                    elif target_year >= 2026:
                        continue
                img_path_str = str(row.get("image_path", ""))
                if not img_path_str or img_path_str in ("nan", "None"):
                    if strict_satellite:
                        raise FileNotFoundError(
                            f"Satellite image marked available but path is empty for order={order}, rel_yr={rel_yr}"
                        )
                    continue
                resolved = self._resolve_satellite_image_path(img_path_str)
                if resolved is None or not resolved.is_file():
                    msg = (
                        f"Satellite image file not found for treatment_order={order}, "
                        f"relative_year={rel_yr}, path={img_path_str!r}. "
                        f"Configured satellite_root={self.satellite_root}"
                    )
                    if strict_satellite:
                        raise FileNotFoundError(msg)
                    logger.warning(msg)
                    continue
                if resolved and resolved.is_file():
                    # 1. Physical file hash verification against expected image_sha256 if recorded
                    exp_sha = row.get("image_sha256")
                    if pd.notna(exp_sha) and str(exp_sha).strip():
                        exp_sha_str = str(exp_sha).strip()
                        res_str = str(resolved)
                        if res_str not in _IMAGE_SHA_CACHE:
                            _IMAGE_SHA_CACHE[res_str] = hashlib.sha256(resolved.read_bytes()).hexdigest()
                        if _IMAGE_SHA_CACHE[res_str] != exp_sha_str:
                            if strict_satellite:
                                raise ValueError(
                                    "Satellite image SHA256 mismatch: "
                                    f"actual {_IMAGE_SHA_CACHE[res_str]} != expected "
                                    f"{exp_sha_str} for {resolved}"
                                )
                            logger.warning(
                                "Satellite image %s hash mismatch: actual %s != expected %s",
                                resolved, _IMAGE_SHA_CACHE[res_str], exp_sha_str,
                            )
                            continue

                    # 2. Path/filename temporal verification
                    import re
                    m_year = re.search(r"[_\-](\d{4})[_\-\.]", resolved.name)
                    if m_year:
                        file_year = int(m_year.group(1))
                        if opening_year is not None:
                            if file_year >= opening_year:
                                continue
                            if pd.notna(row_ty) and file_year != int(row_ty):
                                continue
                        elif file_year >= 2026:
                            continue

                    img, p_mask = self._read_satellite_tiff(resolved)
                    if p_mask.any():
                        images[i] = img
                        pixel_masks[i] = p_mask
                        year_mask[i] = True

        return images, pixel_masks, year_mask, event_times

    def _get_streetview(
        self, order: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        empty_features = torch.zeros(0, 384, dtype=torch.float32)
        empty_mask = torch.zeros(0, dtype=torch.bool)
        empty_gaps = torch.zeros(0, dtype=torch.float32)

        if not self.load_images:
            return empty_features, empty_mask, None, empty_gaps, empty_gaps

        paths = self._stc_streetview_assets.get(order, [])
        if not paths:
            declared_available = bool(self._stc_streetview_avail.get(order, False))
            if declared_available and (
                self.require_streetview or self.manifest.get("strict_production", False)
            ):
                raise ValueError(
                    f"Streetview is declared available but has no asset paths for treatment_order={order}"
                )
            return empty_features, empty_mask, None, empty_gaps, empty_gaps

        # Verify streetview assets against the frozen selection table
        if getattr(self, "_streetview_registry", None) is None:
            self._streetview_registry = _load_streetview_registry(
                self.streetview_index_path, dataset_root=self.root, raise_if_missing=True
            )
        registry = self._streetview_registry
        event_info = self._stc_state_info.get(order, {})
        ev_city = str(event_info.get("city_key", ""))
        ev_grid = str(event_info.get("grid_id", ""))
        ev_op_month = str(event_info.get("opening_month", ""))[:7]
        if not ev_op_month:
            raise ValueError(f"Missing opening_month for streetview event treatment_order={order}")
        try:
            opening_period = pd.Period(ev_op_month, freq="M")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid opening_month for streetview event treatment_order={order}: {ev_op_month!r}"
            ) from exc

        strict_assets = bool(self.require_streetview or self.manifest.get("strict_production", False))
        valid_assets: list[tuple[Path, int, float]] = []
        for p in paths:
            p_obj = Path(p)
            name_key = p_obj.name
            resolved_key = str(p_obj.resolve()) if p_obj.is_file() else str(p_obj)
            rec = registry.get(resolved_key) or registry.get(str(p_obj)) or registry.get(name_key)
            if rec is None:
                if strict_assets:
                    raise ValueError(
                        f"Streetview asset is absent from the verified selection table: {p_obj}"
                    )
                continue
            rec_city = str(rec.get("city_key", ""))
            rec_grid = str(rec.get("grid_id", ""))
            rec_order = rec.get("treatment_order")
            # 1. Identity binding: event order / city / grid
            if strict_assets and (rec_order is None or not rec_city or not rec_grid):
                raise ValueError(f"Streetview asset lacks event/grid identity fields: {p_obj}")
            if rec_order is not None and rec_order != order:
                if strict_assets:
                    raise ValueError(f"Streetview treatment_order mismatch for asset {p_obj}")
                continue
            if ev_city and rec_city and rec_city != ev_city:
                if strict_assets:
                    raise ValueError(f"Streetview city_key mismatch for asset {p_obj}")
                continue
            if ev_grid and rec_grid and rec_grid != ev_grid:
                if strict_assets:
                    raise ValueError(f"Streetview grid_id mismatch for asset {p_obj}")
                continue
            rec_open_month = str(rec.get("opening_month", ""))[:7]
            if strict_assets and not rec_open_month:
                raise ValueError(f"Streetview opening_month is missing for asset {p_obj}")
            if rec_open_month and rec_open_month != ev_op_month:
                if strict_assets:
                    raise ValueError(f"Streetview opening_month mismatch for asset {p_obj}")
                continue

            # 2. Temporal binding and per-image age relative to opening month.
            rec_cap_month = str(rec.get("capture_month", ""))[:7]
            if not rec_cap_month:
                if strict_assets:
                    raise ValueError(f"Streetview capture_month is missing for asset {p_obj}")
                continue
            try:
                capture_period = pd.Period(rec_cap_month, freq="M")
            except (TypeError, ValueError) as exc:
                if strict_assets:
                    raise ValueError(
                        f"Invalid capture_month for streetview asset {p_obj}: {rec_cap_month!r}"
                    ) from exc
                continue
            gap_months = int(opening_period.ordinal - capture_period.ordinal)
            if gap_months <= 0:
                if strict_assets:
                    raise ValueError(
                        f"Streetview image is not strictly pre-opening for treatment_order={order}: "
                        f"capture_month={rec_cap_month}, opening_month={ev_op_month}, path={p_obj}"
                    )
                continue
            # 3. Physical image SHA verification
            if p_obj.is_file():
                actual_sha = _IMAGE_SHA_CACHE.get(resolved_key)
                if actual_sha is None:
                    actual_sha = hashlib.sha256(p_obj.read_bytes()).hexdigest()
                    _IMAGE_SHA_CACHE[resolved_key] = actual_sha
                exp_sha = rec.get("asset_sha256")
                if exp_sha and actual_sha != exp_sha:
                    if strict_assets:
                        raise ValueError(f"Streetview image SHA mismatch for asset {p_obj}")
                    continue
                if strict_assets and not exp_sha:
                    raise ValueError(f"Streetview asset registry lacks SHA256 for asset {p_obj}")
            else:
                if strict_assets:
                    raise FileNotFoundError(f"Streetview image file not found: {p_obj}")
                continue
            valid_assets.append((p_obj, gap_months, compress_time_gap_months(gap_months)))

        if not valid_assets:
            return empty_features, empty_mask, None, empty_gaps, empty_gaps
        valid_assets.sort(key=lambda asset: (asset[1], str(asset[0])))

        images: list[torch.Tensor] = []
        gap_months_values: list[float] = []
        gap_compressed_values: list[float] = []
        for path, gap_months, gap_compressed in valid_assets:
            try:
                image = decode_image(str(path), mode=ImageReadMode.RGB).float().div_(255.0)
                if image.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
                    image = resize(image, [IMAGE_SIZE, IMAGE_SIZE], antialias=True)
            except (OSError, RuntimeError, ValueError) as exc:
                if strict_assets:
                    raise RuntimeError(f"Could not decode streetview image: {path}") from exc
                logger.warning("Skipping unreadable streetview image %s: %s", path, exc)
                continue
            images.append(image)
            gap_months_values.append(float(gap_months))
            gap_compressed_values.append(gap_compressed)

        if not images:
            return empty_features, empty_mask, None, empty_gaps, empty_gaps

        image_tensor = torch.stack(images)
        image_mask = torch.ones(len(images), dtype=torch.bool)
        features = torch.zeros(len(images), 384, dtype=torch.float32)
        return (
            features,
            image_mask,
            image_tensor,
            torch.tensor(gap_months_values, dtype=torch.float32),
            torch.tensor(gap_compressed_values, dtype=torch.float32),
        )

    def _build_stc_lookup(self) -> None:
        state_idx = self.state_inputs.set_index("treatment_order")
        interv_idx = self.intervention_inputs.set_index("treatment_order")
        context_idx = self.context_inputs.set_index("treatment_order")

        self._stc_opening_years: dict[int, int] = {}
        self._stc_state_info: dict[int, dict[str, object]] = {}
        self._stc_state_tab: dict[int, torch.Tensor] = {}
        self._stc_state_mask: dict[int, torch.Tensor] = {}
        self._stc_interv: dict[int, torch.Tensor] = {}
        self._stc_interv_mask: dict[int, torch.Tensor] = {}
        self._stc_context_tab: dict[int, torch.Tensor] = {}
        self._stc_context_mask: dict[int, torch.Tensor] = {}
        self._stc_alphaearth: dict[int, torch.Tensor] = {}
        self._stc_alphaearth_avail: dict[int, bool] = {}
        self._stc_streetview_assets: dict[int, list[str]] = {}
        self._stc_streetview_avail: dict[int, bool] = {}

        # The published AlphaEarth role table uses ``alphaearth_Axx`` names,
        # while early STC fixtures used the shorter ``Axx`` form. Preserve
        # either on-disk schema and map the selected columns into the same
        # 64-value model vector.
        ae_column_schemas = (
            [f"A{i:02d}" for i in range(64)],
            [f"alphaearth_A{i:02d}" for i in range(64)],
        )
        ae_cols = next(
            (
                columns
                for columns in ae_column_schemas
                if all(column in self.context_inputs.columns for column in columns)
            ),
            [],
        )
        if not ae_cols and "alphaearth_available" in self.context_inputs.columns:
            has_alphaearth = self.context_inputs["alphaearth_available"].fillna(False).astype(bool)
            if has_alphaearth.any():
                raise ValueError(
                    "context_inputs marks AlphaEarth available but does not contain a complete "
                    "64-dimensional Axx or alphaearth_Axx column schema"
                )

        for order in self._treatment_orders:
            # 1. State tabular
            if order in state_idx.index:
                s_row = state_idx.loc[order]
                self._stc_state_info[order] = {
                    "city_key": str(s_row.get("city_key", "")),
                    "grid_id": str(s_row.get("grid_id", "")),
                    "opening_month": str(s_row.get("opening_month", "")),
                }
                op_m = str(s_row.get("opening_month", ""))
                if len(op_m) >= 4 and op_m[:4].isdigit():
                    self._stc_opening_years[order] = int(op_m[:4])
                tab_vals = []
                tab_masks = []
                for col in self.state_tabular_columns:
                    z_col = f"z__{col}"
                    avail_col = f"{col}_available"
                    is_avail = bool(s_row[avail_col]) if avail_col in s_row and pd.notna(s_row[avail_col]) else False
                    val = pd.to_numeric(s_row.get(z_col, np.nan), errors="coerce")
                    if is_avail and pd.notna(val) and np.isfinite(val):
                        tab_vals.append(float(val))
                        tab_masks.append(True)
                    else:
                        tab_vals.append(0.0)
                        tab_masks.append(False)
                self._stc_state_tab[order] = torch.tensor(tab_vals, dtype=torch.float32)
                self._stc_state_mask[order] = torch.tensor(tab_masks, dtype=torch.bool)

                # Streetview assets
                sv_raw = s_row.get("streetview_assets", "[]") if "streetview_assets" in s_row else "[]"
                if isinstance(sv_raw, list):
                    sv_paths = sv_raw
                else:
                    try:
                        sv_paths = json.loads(str(sv_raw)) if pd.notna(sv_raw) else []
                    except Exception:
                        sv_paths = []
                self._stc_streetview_assets[order] = sv_paths
                self._stc_streetview_avail[order] = bool(s_row.get("streetview_available", False)) if "streetview_available" in s_row and pd.notna(s_row.get("streetview_available")) else False
            else:
                self._stc_state_info[order] = {}
                self._stc_state_tab[order] = torch.zeros(len(self.state_tabular_columns), dtype=torch.float32)
                self._stc_state_mask[order] = torch.zeros(len(self.state_tabular_columns), dtype=torch.bool)
                self._stc_streetview_assets[order] = []
                self._stc_streetview_avail[order] = False

            # 2. Intervention
            if order in interv_idx.index:
                t_row = interv_idx.loc[order]
                t_vals = []
                t_masks = []
                for col in self.intervention_columns:
                    z_col = f"z__{col}" if f"z__{col}" in t_row else col
                    avail_col = f"{col}_available"
                    is_avail = bool(t_row[avail_col]) if avail_col in t_row and pd.notna(t_row[avail_col]) else False
                    val = pd.to_numeric(t_row.get(z_col, np.nan), errors="coerce")
                    if is_avail and pd.notna(val) and np.isfinite(val):
                        t_vals.append(float(val))
                        t_masks.append(True)
                    else:
                        t_vals.append(0.0)
                        t_masks.append(False)
                self._stc_interv[order] = torch.tensor(t_vals, dtype=torch.float32)
                self._stc_interv_mask[order] = torch.tensor(t_masks, dtype=torch.bool)
            else:
                self._stc_interv[order] = torch.zeros(len(self.intervention_columns), dtype=torch.float32)
                self._stc_interv_mask[order] = torch.zeros(len(self.intervention_columns), dtype=torch.bool)

            # 3. Context tabular & AlphaEarth
            if order in context_idx.index:
                c_row = context_idx.loc[order]
                c_vals = []
                c_masks = []
                for col in self.context_tabular_columns:
                    z_col = f"z__{col}"
                    avail_col = f"{col}_available"
                    is_avail = bool(c_row[avail_col]) if avail_col in c_row and pd.notna(c_row[avail_col]) else False
                    val = pd.to_numeric(c_row.get(z_col, np.nan), errors="coerce")
                    if is_avail and pd.notna(val) and np.isfinite(val):
                        c_vals.append(float(val))
                        c_masks.append(True)
                    else:
                        c_vals.append(0.0)
                        c_masks.append(False)
                self._stc_context_tab[order] = torch.tensor(c_vals, dtype=torch.float32)
                self._stc_context_mask[order] = torch.tensor(c_masks, dtype=torch.bool)

                # AlphaEarth
                ae_avail = bool(c_row.get("alphaearth_available", False)) if "alphaearth_available" in c_row and pd.notna(c_row.get("alphaearth_available")) else False
                if ae_avail and ae_cols:
                    ae_vals = [float(pd.to_numeric(c_row[b], errors="coerce")) if pd.notna(pd.to_numeric(c_row[b], errors="coerce")) else 0.0 for b in ae_cols]
                    self._stc_alphaearth[order] = torch.tensor(ae_vals, dtype=torch.float32)
                    self._stc_alphaearth_avail[order] = True
                else:
                    self._stc_alphaearth[order] = torch.zeros(64, dtype=torch.float32)
                    self._stc_alphaearth_avail[order] = False
            else:
                self._stc_context_tab[order] = torch.zeros(len(self.context_tabular_columns), dtype=torch.float32)
                self._stc_context_mask[order] = torch.zeros(len(self.context_tabular_columns), dtype=torch.bool)
                self._stc_alphaearth[order] = torch.zeros(64, dtype=torch.float32)
                self._stc_alphaearth_avail[order] = False

        self._stc_sat_seq_dict = {}
        if "treatment_order" in self.satellite_sequences.columns:
            for order, group in self.satellite_sequences.groupby("treatment_order"):
                self._stc_sat_seq_dict[int(order)] = group.set_index("event_time")

    def _has_stc_satellite(self, order: int) -> bool:
        """Return True when a pre-treatment satellite frame is declared available."""
        sequence = self._stc_sat_seq_dict.get(order)
        if sequence is None:
            return False
        for event_time, row in sequence.iterrows():
            if int(event_time) >= 0:
                continue
            if not bool(row.get("state_temporal_allowed", False)):
                continue
            if bool(row.get("state_image_available", False)) or bool(
                row.get("satellite_image_available", False)
            ):
                return True
        return False

    def _build_stc_response_tensor(self) -> None:
        sample_lookup = self.sample_index.set_index("treatment_order")
        grouped = self.response_targets.groupby("treatment_order", sort=False)
        grouped_orders = set(grouped.groups)

        self._stc_records: dict[int, dict[str, object]] = {}

        for order in self._treatment_orders:
            row: dict[str, object] = {"treatment_order": order}
            if order in sample_lookup.index:
                s_row = sample_lookup.loc[order]
                row["city_key"] = str(s_row.get("city_key", ""))
                row["split"] = str(s_row.get("split", ""))
                row["final_training_mask"] = bool(s_row.get("final_training_mask", False))
                row["quality_grade"] = str(s_row.get("quality_grade", "pending"))
            else:
                row["city_key"] = ""
                row["split"] = ""
                row["final_training_mask"] = False
                row["quality_grade"] = "unknown"

            response = np.full(self._response_dim, np.nan, dtype=np.float32)
            response_se = np.full(self._response_dim, np.nan, dtype=np.float32)
            response_weight = np.zeros(self._response_dim, dtype=np.float32)

            if order in grouped_orders:
                group = grouped.get_group(order)
                has_weight = "response_weight" in group.columns
                for family, (start, end, cols) in self._response_offsets.items():
                    del end
                    family_group = group[group["outcome_family"].eq(family)]
                    for _, task_row in family_group.iterrows():
                        outcome = str(task_row["outcome"])
                        horizon = int(task_row["event_time"])
                        col_name = f"{outcome}__t{horizon}"
                        if col_name in cols:
                            idx = cols.index(col_name) + start
                            label_eligible = task_row.get("final_training_mask", False)
                            label_eligible = bool(label_eligible) if pd.notna(label_eligible) else False
                            if label_eligible:
                                response[idx] = pd.to_numeric(task_row.get("causal_response_label", np.nan), errors="coerce")
                                response_se[idx] = pd.to_numeric(task_row.get("standard_error", np.nan), errors="coerce")
                                if has_weight:
                                    w_val = pd.to_numeric(task_row.get("response_weight", np.nan), errors="coerce")
                                    response_weight[idx] = float(w_val) if pd.notna(w_val) else 0.0
                                else:
                                    response_weight[idx] = 1.0

            row["response"] = torch.from_numpy(response)
            row["response_mask"] = torch.from_numpy(np.isfinite(response))
            row["response_se"] = torch.from_numpy(response_se)
            row["response_weight"] = torch.from_numpy(response_weight)
            self._stc_records[order] = row

    def _getitem_stc(self, idx: int) -> GridSample:
        order = self._treatment_orders[idx]
        rec = self._stc_records[order]
        city_key = str(rec["city_key"])
        split = str(rec["split"])

        sat_imgs, sat_pixel_mask, sat_year_mask, sat_times = self._get_satellite_sequence(order)
        sv_feats, sv_mask, sv_imgs, sv_gap_months, sv_gap_compressed = self._get_streetview(order)

        state_tab = self._stc_state_tab[order]
        state_mask = self._stc_state_mask[order]
        interv = self._stc_interv[order]
        interv_mask = self._stc_interv_mask[order]
        context_tab = self._stc_context_tab[order]
        context_mask = self._stc_context_mask[order]
        ae = self._stc_alphaearth[order]
        ae_avail = self._stc_alphaearth_avail[order]

        return GridSample(
            treatment_order=order,
            city_key=city_key,
            split=split,
            feature_vector=state_tab,
            response_vector=rec["response"],
            response_mask=rec["response_mask"],
            response_se=rec["response_se"],
            response_weight=rec["response_weight"],
            modality_available={
                "poi": bool(state_mask[:12].any().item()) if len(state_mask) >= 12 else False,
                "viirs": bool(state_mask[12:].any().item()) if len(state_mask) > 12 else False,
                "streetview": self._stc_streetview_avail.get(order, False),
                "satellite": bool(sat_year_mask.any().item()),
                "alphaearth": ae_avail,
            },
            final_training_mask=bool(rec["final_training_mask"]),
            quality_grade=str(rec["quality_grade"]),
            image_paths=self._stc_streetview_assets.get(order, []),
            satellite_vector=ae,
            satellite_available=ae_avail,
            is_stc=True,
            state_tabular=state_tab,
            state_field_mask=state_mask,
            streetview_features=sv_feats,
            streetview_mask=sv_mask,
            streetview_images=sv_imgs,
            streetview_gap_months=sv_gap_months,
            streetview_gap_compressed=sv_gap_compressed,
            satellite_images=sat_imgs,
            satellite_pixel_mask=sat_pixel_mask,
            satellite_year_mask=sat_year_mask,
            satellite_event_times=sat_times,
            context_tabular=context_tab,
            context_field_mask=context_mask,
            alphaearth=ae,
            alphaearth_available=ae_avail,
            intervention=interv,
            intervention_mask=interv_mask,
        )

    def _validate(self) -> None:
        if self.sample_index["treatment_order"].duplicated().any():
            raise ValueError("sample_index has duplicate treatment_order")
        required = {"treatment_order", "split", "final_training_mask", "quality_grade"}
        missing = required - set(self.sample_index.columns)
        if missing:
            raise ValueError(f"sample_index lacks columns: {sorted(missing)}")
        response_required = {
            "treatment_order",
            "outcome_family",
            "outcome",
            "event_time",
            "causal_response_label",
            "final_training_mask",
        }
        response_missing = response_required - set(self.response_targets.columns)
        if response_missing:
            raise ValueError(
                "response_targets lacks per-label training eligibility columns: "
                f"{sorted(response_missing)}"
            )

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
            "satellite_",
            "target_year",
            "is_temporal_proxy",
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
            if len(col) == 3 and col.startswith("A") and col[1:].isdigit():
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
        grouped_orders = set(grouped.groups)
        records: list[dict] = []
        sample_lookup = self.sample_index.set_index("treatment_order")
        # Iterate the sample index, not the labels: a filtered response domain
        # (for example a VIIRS-only build) can leave sampled units without any
        # label row, and those units must still yield an all-missing record so
        # the dataset, loaders and exports stay aligned.
        for order in self._treatment_orders:
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
            response_weight = np.zeros(RESPONSE_DIM, dtype=np.float64)

            if order in grouped_orders:
                group = grouped.get_group(order)
                has_weight_column = "response_weight" in group.columns
                for family, (start, end, cols) in RESPONSE_OFFSETS.items():
                    del end
                    family_group = group[group["outcome_family"].eq(family)]
                    for _, task_row in family_group.iterrows():
                        outcome = str(task_row["outcome"])
                        horizon = int(task_row["event_time"])
                        col_name = f"{outcome}__t{horizon}"
                        if col_name in cols:
                            idx = cols.index(col_name) + start
                            # The sample-level mask says whether this grid can be
                            # used at all; it must not promote every response
                            # dimension.  A release may contain a valid label for
                            # one outcome and an ineligible/held-out label for
                            # another outcome on the same grid.
                            label_eligible = task_row.get("final_training_mask", False)
                            label_eligible = (
                                bool(label_eligible) if pd.notna(label_eligible) else False
                            )
                            if label_eligible:
                                response[idx] = pd.to_numeric(
                                    task_row.get("causal_response_label", np.nan),
                                    errors="coerce",
                                )
                                response_se[idx] = pd.to_numeric(
                                    task_row.get("standard_error", np.nan), errors="coerce"
                                )
                                if has_weight_column:
                                    weight_value = pd.to_numeric(
                                        task_row.get("response_weight", np.nan),
                                        errors="coerce",
                                    )
                                    response_weight[idx] = (
                                        float(weight_value) if pd.notna(weight_value) else 0.0
                                    )
                                else:
                                    # Datasets built before the weight columns
                                    # existed carry no weight column; treat every
                                    # eligible label as equally reliable so the
                                    # loader stays compatible.
                                    response_weight[idx] = 1.0
                        else:
                            logger.warning(
                                "Skipping unknown outcome/horizon: %s in family %s",
                                col_name,
                                family,
                            )

            row["response"] = response.astype(np.float32)
            row["response_se"] = response_se.astype(np.float32)
            row["response_weight"] = response_weight.astype(np.float32)
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

    @property
    def station_token_count(self) -> int:
        """Number of units with a resolved station-conditioning token."""
        return sum(order in self._station_tokens for order in self._treatment_orders)

    def __len__(self) -> int:
        return len(self._treatment_orders)

    def __getitem__(self, idx: int) -> GridSample:
        if self.is_stc:
            return self._getitem_stc(idx)

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
        response_mask = np.isfinite(response)
        response_se = rec["response_se"]

        sat_vec = self._satellite_vectors.get(order)
        if sat_vec is None:
            sat_vec = torch.zeros(64, dtype=torch.float32)
            sat_avail = False
        else:
            sat_avail = True

        return GridSample(
            treatment_order=order,
            city_key=str(rec["city_key"]),
            split=str(rec["split"]),
            feature_vector=torch.from_numpy(features),
            response_vector=torch.from_numpy(response),
            response_mask=torch.from_numpy(response_mask),
            response_se=torch.from_numpy(response_se),
            response_weight=torch.from_numpy(rec["response_weight"]),
            modality_available=self._build_modality_available(order),
            final_training_mask=bool(rec["final_training_mask"]),
            quality_grade=str(rec["quality_grade"]),
            conditioning_token=conditioning_token,
            station_token=self._station_tokens.get(order, 0),
            image_paths=self._get_image_paths(order),
            satellite_vector=sat_vec,
            satellite_available=sat_avail,
            satellite_is_temporal_proxy=self._satellite_is_proxy.get(order, False),
            satellite_target_year=self._satellite_target_year.get(order, float("nan")),
        )

    def response_similarity(
        self, idx_a: int, idx_b: int, within_family: str | None = None
    ) -> float:
        sample_a = self[idx_a]
        sample_b = self[idx_b]
        offsets_dict = self._response_offsets if self.is_stc else RESPONSE_OFFSETS
        families = [within_family] if within_family else list(offsets_dict)
        similarities: list[float] = []
        for family in families:
            if family not in offsets_dict:
                continue
            start, end, _ = offsets_dict[family]
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
        if self.is_stc:
            return self._response_dim
        return RESPONSE_DIM

    def response_offsets(self) -> dict[str, tuple[int, int, list[str]]]:
        if self.is_stc:
            return dict(self._response_offsets)
        return dict(RESPONSE_OFFSETS)

    def treatment_orders(self) -> list[int]:
        return list(self._treatment_orders)

    def training_orders(self) -> list[int]:
        if self.is_stc:
            return [
                o
                for o in self._treatment_orders
                if bool(self._stc_records[o].get("final_training_mask", False))
            ]
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
        "response_weight": torch.stack([s.response_weight for s in samples]),
        "conditioning_tokens": torch.tensor(
            [s.conditioning_token for s in samples], dtype=torch.long
        ),
        "station_tokens": torch.tensor([s.station_token for s in samples], dtype=torch.long),
        "final_training_mask": torch.tensor(
            [s.final_training_mask for s in samples], dtype=torch.bool
        ),
        "city_keys": [s.city_key for s in samples],
        "splits": [s.split for s in samples],
        "quality_grades": [s.quality_grade for s in samples],
        "modality_available": [s.modality_available for s in samples],
        "satellite_embeddings": torch.stack([s.satellite_vector for s in samples]),
        "satellite_vectors": torch.stack([s.satellite_vector for s in samples]),
        "satellite_mask": torch.tensor([s.satellite_available for s in samples], dtype=torch.bool),
        "satellite_is_temporal_proxy": torch.tensor(
            [s.satellite_is_temporal_proxy for s in samples], dtype=torch.bool
        ),
    }

    if any(s.is_stc for s in samples):
        batch["is_stc"] = True
        batch["state_tabular"] = torch.stack([s.state_tabular for s in samples])
        batch["state_field_mask"] = torch.stack([s.state_field_mask for s in samples])
        batch["intervention"] = torch.stack([s.intervention for s in samples])
        batch["intervention_mask"] = torch.stack([s.intervention_mask for s in samples])
        batch["context"] = torch.stack([s.context_tabular for s in samples])
        batch["context_field_mask"] = torch.stack([s.context_field_mask for s in samples])
        batch["alphaearth"] = torch.stack([s.alphaearth for s in samples])
        batch["alphaearth_available"] = torch.tensor(
            [s.alphaearth_available for s in samples], dtype=torch.bool
        )
        streetview_count = max((int(s.streetview_mask.numel()) for s in samples), default=0)
        batch["streetview_mask"] = torch.zeros(len(samples), streetview_count, dtype=torch.bool)
        batch["streetview_gap_months"] = torch.zeros(len(samples), streetview_count, dtype=torch.float32)
        batch["streetview_gap_compressed"] = torch.zeros(len(samples), streetview_count, dtype=torch.float32)
        streetview_features = torch.zeros(len(samples), streetview_count, 384, dtype=torch.float32)
        for row_index, sample in enumerate(samples):
            count = int(sample.streetview_mask.numel())
            if (
                sample.streetview_features.shape != (count, 384)
                or sample.streetview_gap_months.numel() != count
                or sample.streetview_gap_compressed.numel() != count
                or (sample.streetview_images is not None and sample.streetview_images.shape[0] != count)
            ):
                raise ValueError(
                    f"Streetview image, mask, feature, and time-gap counts disagree for "
                    f"treatment_order={sample.treatment_order}"
                )
            if count:
                batch["streetview_mask"][row_index, :count] = sample.streetview_mask
                streetview_features[row_index, :count] = sample.streetview_features
                batch["streetview_gap_months"][row_index, :count] = sample.streetview_gap_months
                batch["streetview_gap_compressed"][row_index, :count] = sample.streetview_gap_compressed
        batch["streetview_features"] = streetview_features
        if any(s.streetview_images is not None for s in samples):
            image_count = max(
                (int(s.streetview_images.shape[0]) for s in samples if s.streetview_images is not None),
                default=0,
            )
            streetview_images = torch.zeros(len(samples), image_count, 3, 224, 224, dtype=torch.float32)
            for row_index, sample in enumerate(samples):
                if sample.streetview_images is not None:
                    count = int(sample.streetview_images.shape[0])
                    streetview_images[row_index, :count] = sample.streetview_images
            batch["streetview_images"] = streetview_images
        batch["satellite_images"] = torch.stack([s.satellite_images for s in samples])
        batch["satellite_pixel_mask"] = torch.stack([s.satellite_pixel_mask for s in samples])
        batch["satellite_year_mask"] = torch.stack([s.satellite_year_mask for s in samples])
        batch["satellite_event_times"] = torch.stack([s.satellite_event_times for s in samples])

    if not any(s.is_stc for s in samples) and use_images and load_images_fn is not None:
        images_list: list[torch.Tensor] = []
        masks_list: list[torch.Tensor] = []
        for s in samples:
            imgs, img_mask = load_images_fn(s.treatment_order)
            images_list.append(imgs)
            masks_list.append(img_mask)
        batch["images"] = torch.stack(images_list)
        batch["image_mask"] = torch.stack(masks_list)

    return batch
