"""Tests for the literature-faithful city-centre identification pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))
sys.path.insert(0, str(ROOT / "src"))

from build_city_centres import (  # noqa: E402
    _expand_monomial,
    _multiply_expansions,
    benjamini_hochberg,
    build_city_centres,
    haversine_km,
    lwr_fit_dense,
)


def _gaussian_2d(
    rows: int, cols: int, center_r: int, center_c: int, peak: float, sigma: float, background: float
) -> np.ndarray:
    rr, cc = np.mgrid[0:rows, 0:cols]
    return background + peak * np.exp(
        -((rr - center_r) ** 2 + (cc - center_c) ** 2) / (2.0 * sigma**2)
    )


def _write_synthetic_city(tmp_path: Path, density_grid: np.ndarray) -> Path:
    """Write synthetic grid + POI-yearly files for one city; return poi_dir."""
    rows, cols = density_grid.shape
    lon0, lat0 = 116.0, 39.0
    lon_step, lat_step = 0.006, 0.0045
    grid_dir = tmp_path / "grids" / "synthetic"
    grid_dir.mkdir(parents=True)
    records = []
    for r in range(rows):
        for c in range(cols):
            records.append(
                {
                    "grid_id": f"g{r:05d}x{c:05d}",
                    "row": r,
                    "col": c,
                    "lon_min": lon0 + c * lon_step,
                    "lat_min": lat0 + r * lat_step,
                    "lon_max": lon0 + (c + 1) * lon_step,
                    "lat_max": lat0 + (r + 1) * lat_step,
                    "centroid_lon": lon0 + (c + 0.5) * lon_step,
                    "centroid_lat": lat0 + (r + 0.5) * lat_step,
                    "area_km2": 0.25,
                    "geometry_wkt": "",
                }
            )
    grids = pd.DataFrame(records)
    grids.to_parquet(grid_dir / "synthetic_grids.parquet", index=False)

    poi_dir = tmp_path / "poi"
    poi_dir.mkdir(parents=True)
    rows_long = []
    for year in (2012, 2013, 2014, 2015):
        for r in range(rows):
            for c in range(cols):
                rows_long.append(
                    {
                        "city": "synthetic",
                        "grid_id": f"g{r:05d}x{c:05d}",
                        "year": year,
                        "poi_count": float(density_grid[r, c]),
                    }
                )
    pd.DataFrame(rows_long).to_parquet(poi_dir / "synthetic_poi_grid_yearly.parquet", index=False)
    return poi_dir


def test_haversine_known_distance() -> None:
    distance = float(
        haversine_km(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([1.0]))[0]
    )
    assert distance == pytest.approx(157.2, rel=0.01)


def test_monomial_expansion_and_multiplication() -> None:
    terms = _expand_monomial(2, 1)
    # (u - ui)^2 (v - vi) = u^2 v - 2 ui u v + ui^2 v - vi u^2 + 2 ui vi u - ui^2 vi
    total = 0.0
    for coeff, r, s, p, q in terms:
        total += coeff * (-3.0) ** r * (-2.0) ** s * 5.0**p * 4.0**q
    expected = ((5.0 - 3.0) ** 2) * (4.0 - 2.0)
    assert total == pytest.approx(expected, rel=1e-12)


def test_multiply_expansions_consistent() -> None:
    left = _expand_monomial(1, 0)
    right = _expand_monomial(0, 1)
    product = _multiply_expansions(left, right)
    value = 0.0
    for coeff, r, s, p, q in product:
        value += coeff * (-1.5) ** r * (-0.5) ** s * 2.0**p * 3.0**q
    expected = (2.0 - 1.5) * (3.0 - 0.5)
    assert value == pytest.approx(expected, rel=1e-12)


def test_benjamini_hochberg() -> None:
    p_values = np.array([0.01, 0.02, 0.5, 0.6])
    passing = benjamini_hochberg(p_values, 0.05)
    assert passing.tolist() == [True, True, False, False]
    assert benjamini_hochberg(np.array([]), 0.05).tolist() == []


def test_lwr_matches_brute_force(tmp_path: Path) -> None:
    density = _gaussian_2d(40, 40, 20, 20, 500.0, 3.0, 5.0)
    poi_dir = _write_synthetic_city(tmp_path, density)
    grid_path = tmp_path / "grids" / "synthetic" / "synthetic_grids.parquet"
    poi_path = poi_dir / "synthetic_poi_grid_yearly.parquet"
    from build_city_centres import _dense_array, _projected_coords, load_density_surface

    surface = load_density_surface("synthetic", grid_path, poi_path, (2012, 2015))
    log_dense, row_min, col_min = _dense_array(surface, "log_density")
    lon_dense, _, _ = _dense_array(surface, "centroid_lon")
    lat_dense, _, _ = _dense_array(surface, "centroid_lat")
    valid = ~np.isnan(log_dense)
    u_dense, v_dense = _projected_coords(lon_dense, lat_dense)
    sigma_cells = 3.0
    fitted, var_fitted, loo_fitted, _ = lwr_fit_dense(
        log_dense, u_dense, v_dense, valid, sigma_cells, 1.5, need_loo=True
    )

    # Brute force with the exact separable discrete kernel of gaussian_filter.
    from scipy import ndimage

    radius = int(np.ceil(3.0 * sigma_cells))
    impulse = np.zeros(2 * radius + 1)
    impulse[radius] = 1.0
    kernel_1d = ndimage.gaussian_filter1d(impulse, sigma_cells, truncate=3.0, mode="constant")
    samples = [(20, 20), (24, 24), (18, 25)]
    for sr, sc in samples:
        a_brute = np.zeros((6, 6))
        b_brute = np.zeros(6)
        b2_brute = np.zeros((6, 6))
        u_i, v_i = u_dense[sr, sc], v_dense[sr, sc]
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rn, cn = sr + dr, sc + dc
                if not (0 <= rn < log_dense.shape[0] and 0 <= cn < log_dense.shape[1]):
                    continue
                if not valid[rn, cn]:
                    continue
                w = kernel_1d[dr + radius] * kernel_1d[dc + radius]
                du = (u_dense[rn, cn] - u_i) / 1.5
                dv = (v_dense[rn, cn] - v_i) / 1.5
                x = np.array([1, du, dv, du**2, dv**2, du * dv])
                a_brute += w * np.outer(x, x)
                b_brute += w * x * log_dense[rn, cn]
                b2_brute += (w**2) * np.outer(x, x)
        a_brute += 1e-8 * np.eye(6)
        beta = np.linalg.solve(a_brute, b_brute)
        assert fitted[sr, sc] == pytest.approx(beta[0], rel=1e-6, abs=1e-6)

        # LOO: remove self contribution (local centring: exactly e1 e1', e1*y)
        a_loo = a_brute - np.eye(6)[:, :1] @ np.eye(6)[:1, :]
        b_loo = b_brute - np.eye(6)[:, 0] * log_dense[sr, sc]
        beta_loo = np.linalg.solve(a_loo, b_loo)
        assert loo_fitted[sr, sc] == pytest.approx(beta_loo[0], rel=1e-6, abs=1e-6)

        # Sandwich variance up to the discrete-integral approximation.
        residuals = log_dense - fitted
        s2 = float(
            np.sum(
                (w := kernel_1d[None, :] * kernel_1d[:, None])
                * (residuals[sr - radius : sr + radius + 1, sc - radius : sc + radius + 1] ** 2)
                * valid[sr - radius : sr + radius + 1, sc - radius : sc + radius + 1]
            )
            / max(
                np.sum(w * valid[sr - radius : sr + radius + 1, sc - radius : sc + radius + 1]) - 6,
                1.0,
            )
        )
        var_brute = s2 * float(
            np.eye(6)[0]
            @ np.linalg.solve(a_brute, b2_brute)
            @ np.linalg.solve(a_brute, np.eye(6)[:, 0])
        )
        assert var_fitted[sr, sc] == pytest.approx(var_brute, rel=0.06, abs=1e-8)


def test_synthetic_bimodal_recovers_two_centres(tmp_path: Path) -> None:
    rows, cols = 60, 60
    density = _gaussian_2d(rows, cols, 30, 30, 500.0, 2.5, 5.0)
    density += _gaussian_2d(rows, cols, 30, 44, 300.0, 2.0, 0.0)
    poi_dir = _write_synthetic_city(tmp_path, density)
    result = build_city_centres(
        "synthetic",
        tmp_path / "grids",
        poi_dir,
        (2012, 2015),
        (0.75, 1.0, 1.5, 2.0),
        1.5,
        0.05,
        0.5,
        5.0,
        3.0,
        8,
        bandwidth_km=1.5,
        store_surface=True,
    )
    centres = result["centres"]
    main = centres.loc[centres["role"].eq("main")]
    assert len(main) == 1
    # Main centre within ~2 cells of the larger bump (row 30, col 30).
    assert abs(int(main["row"].iloc[0]) - 30) <= 2
    assert abs(int(main["col"].iloc[0]) - 30) <= 2
    subs = centres.loc[centres["role"].str.startswith("subcenter")]
    assert len(subs) >= 1
    sub = subs.iloc[0]
    assert abs(int(sub["row"]) - 30) <= 3
    assert abs(int(sub["col"]) - 44) <= 3


def test_synthetic_flat_has_no_subcentres(tmp_path: Path) -> None:
    density = np.full((40, 40), 20.0)
    poi_dir = _write_synthetic_city(tmp_path, density)
    result = build_city_centres(
        "synthetic",
        tmp_path / "grids",
        poi_dir,
        (2012, 2015),
        (0.75, 1.0, 1.5, 2.0),
        1.5,
        0.05,
        0.5,
        5.0,
        3.0,
        8,
        bandwidth_km=1.5,
        store_surface=True,
    )
    centres = result["centres"]
    assert len(centres) == 1
    assert centres["role"].iloc[0] == "main"
    assert result["n_peaks_significant"] == 0
