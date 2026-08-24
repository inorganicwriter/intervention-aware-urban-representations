"""Batch collation for representation training."""

from __future__ import annotations

from ..dataset import RepresentationDataset, collate_samples


def _collate_fn(batch, ds: RepresentationDataset, use_images: bool = False):
    return collate_samples(
        batch,
        load_images_fn=ds._get_images if use_images else None,
        max_images_per_grid=ds.max_images_per_grid,
        use_images=use_images,
    )
