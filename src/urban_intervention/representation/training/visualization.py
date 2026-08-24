"""Best-effort embedding visualizations."""

from __future__ import annotations

from pathlib import Path

import torch


def _visualize_embeddings(
    embeddings: torch.Tensor,
    city_keys: list[str],
    quality_grades: list[str] | None,
    out_path: Path,
) -> dict[str, object]:
    """PCA-2D scatter of a pool coloured by city (best-effort).

    Uses an SVD projection so the coordinate export needs no extra dependency;
    the PNG plot additionally needs matplotlib and is skipped when it is
    missing.
    """
    if embeddings.shape[0] < 3:
        return {"note": "pool smaller than 3 units", "path": None}
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    coords = (centered @ vt[:2].T).numpy()
    points = [
        {"x": float(x), "y": float(y), "city": city}
        for x, y, city in zip(coords[:, 0], coords[:, 1], city_keys, strict=False)
    ]
    if quality_grades:
        for point, grade in zip(points, quality_grades, strict=False):
            point["quality_grade"] = grade
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {"points": points, "plot_written": False, "note": "matplotlib_not_installed"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cities = sorted(set(city_keys))
    palette = [plt.get_cmap("tab20")(index) for index in range(20)]
    figure, axis = plt.subplots(figsize=(7, 6))
    for index, city in enumerate(cities):
        keep = [i for i, value in enumerate(city_keys) if value == city]
        axis.scatter(
            coords[keep, 0],
            coords[keep, 1],
            s=24,
            label=city,
            color=palette[index % len(palette)],
            alpha=0.8,
        )
    axis.set_title("PCA-2D projection of learned embeddings (by city)")
    axis.legend(fontsize=7, loc="best", frameon=False)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return {"points": points, "plot_written": True, "path": out_path.as_posix()}
