"""Explicitly non-production joint labels and optional synthetic image assets."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tests.unit.test_stc_model import _create_mock_stc_dataset_on_disk
from urban_intervention.representation.dataset import RESPONSE_CONFIG


def build_joint_fixture(base: Path, n_units: int = 24, images: bool = False) -> Path:
    np.random.seed(20261022)
    root = _create_mock_stc_dataset_on_disk(base, n_units=n_units)
    targets = pd.read_parquet(root / "response_targets.parquet")
    targets["causal_response_label"] = targets.treatment_order * 0.002 + targets.event_time * 0.01
    targets.loc[targets.treatment_order.mod(5).eq(0), "final_training_mask"] = False
    poi = []
    for order in range(1, n_units + 1):
        for index, (outcome, horizons) in enumerate(RESPONSE_CONFIG["poi"].items()):
            for horizon in horizons:
                poi.append(
                    dict(
                        treatment_order=order,
                        outcome_family="poi",
                        outcome=outcome,
                        event_time=horizon,
                        causal_response_label=0.02 * index + 0.01 * horizon + order * 0.003,
                        standard_error=0.1,
                        response_weight=1.0,
                        final_training_mask=order % 7 != 0,
                    )
                )
    targets = pd.concat([targets, pd.DataFrame(poi)], ignore_index=True)
    targets.to_parquet(root / "response_targets.parquet", index=False)
    sample = pd.read_parquet(root / "sample_index.parquet")
    sample["final_training_mask"] = sample.treatment_order.map(
        targets.groupby("treatment_order").final_training_mask.any()
    )
    sample.to_parquet(root / "sample_index.parquet", index=False)
    if images:
        _attach_images(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        families=["viirs", "poi"],
        dataset_id="joint_fixture_20261022",
        strict_production=False,
        production_eligible=False,
        source_kind="synthetic_fixture_not_scientific_evidence",
    )
    manifest["outputs"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.iterdir()
        if p.suffix in {".parquet", ".json"} and p.name != "manifest.json"
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def _attach_images(root):
    import rasterio
    from PIL import Image
    from rasterio.transform import from_origin

    rng = np.random.default_rng(20261022)
    assets = root / "fixture_assets"
    assets.mkdir()
    state = pd.read_parquet(root / "state_inputs.parquet")
    sequence = pd.read_parquet(root / "satellite_state_sequences.parquet")
    registry = []
    for index, row in state.iterrows():
        order = int(row.treatment_order)
        paths = []
        if order % 3:
            for image_index in range(1 + order % 2):
                path = assets / f"street_{order}_{image_index}.png"
                Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)).save(path)
                paths.append(str(path.resolve()))
                registry.append(
                    dict(
                        treatment_order=order,
                        city_key=row.city_key,
                        grid_id=row.grid_id,
                        opening_month=row.opening_month,
                        capture_month=f"2017-{image_index + 1:02d}",
                        asset_path=str(path.resolve()),
                        asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
        state.at[index, "streetview_assets"] = json.dumps(paths)
        state.at[index, "streetview_available"] = bool(paths)
    for index, row in sequence.iterrows():
        if int(row.treatment_order) % 4 == 0 and int(row.event_time) == -3:
            continue
        path = assets / f"sat_{row.treatment_order}_{row.target_year}_pre.tif"
        bands = rng.uniform(0.05, 0.5, (8, 17, 17)).astype(np.float32)
        bands[6:] = 1
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=17,
            height=17,
            count=8,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(116, 40, 0.0001, 0.0001),
        ) as dst:
            dst.write(bands)
        sequence.at[index, "image_path"] = str(path.resolve())
        sequence.at[index, "image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        sequence.at[index, "state_image_available"] = True
        sequence.at[index, "satellite_image_available"] = True
    state.to_parquet(root / "state_inputs.parquet", index=False)
    sequence.to_parquet(root / "satellite_state_sequences.parquet", index=False)
    pd.DataFrame(registry).to_parquet(root / "streetview_state_assets.parquet", index=False)
