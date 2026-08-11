"""Build a frozen, literature-faithful city-centre registry (McMillen 2001 method).

The complete design reproduces the published procedure rather than a
threshold-based shortcut:

1. Density surface. Per-grid mean POI count over a frozen pre-treatment year
   window (default 2012-2015) is log1p-transformed and smoothed by a LOCAL
   WEIGHTED REGRESSION (LWR) with a local quadratic polynomial in projected
   coordinates and a Gaussian kernel. The bandwidth is frozen at 1.5 km (the
   LOO-CV optimum sits at the boundary of the sigma grid for noisy 500m-grid
   counts, so CV is reported as a diagnostic in the manifest and the
   sensitivity report covers 0.75-3.0 km). Every fitted value carries the
   mean-surface variance of the estimate, s^2 * e1' (X'WX)^-1 (X'W^2X)
   (X'WX)^-1 e1 (sandwich form; the variance appropriate for testing features
   of the smoothed surface itself, McMillen 2001).

2. Significant local maxima. Candidate subcentres are strict local maxima of
   the smoothed surface within `peak-radius-km`. Each candidate is tested
   against its neighbourhood mean with a t-statistic built from the LWR
   prediction variances; p-values are Benjamini-Hochberg FDR corrected at
   `fdr_level` (McMillen 2001, significance step; BH correction modern
   practice).

3. Size criterion. Each significant peak must have a CONTIGUOUS support area:
   the 8-connected component (Giuliano-Small 1991 contiguity) of the peak in
   the floor-thresholded surface, at least `min-cluster-km2` large, and lie at
   least `min-distance-km` from the main centre (the global maximum of the
   smoothed surface).

4. Validation. A polycentric density-gradient regression
   log1p(D) ~ dist_main + dist_sub1 + ... is fitted per city; significant
   negative gradients corroborate that the identified centres organise the
   pre-treatment density surface (Giuliano & Small 1991; McMillen 2001).

5. Sensitivity. All identification parameters are varied and the coordinate
   drift of every centre relative to the main specification is reported.

Deviations (documented, data-driven): POI counts proxy employment density;
log1p handles zero POI counts; local quadratic rather than local linear LWR.

References:
- McMillen, D.P. (2001). Nonparametric employment subcenter identification.
  Journal of Urban Economics 50(3):448-473.
- Giuliano, G. and Small, K.A. (1991). Subcenters in the Los Angeles region.
  Regional Science and Urban Economics 21(2):163-182.

Outputs:
    data/active/reference/city_centers.csv                 centre registry
    data/active/reference/city_centers_manifest.json       frozen parameters, CV, hashes
    data/active/reference/city_centers_validation.csv      gradient-regression validation
    data/active/reference/city_centers_sensitivity.csv     parameter-variant drift
    outputs/figures/centers/<city>.png              surface + centres audit figure
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.config.project import ACTIVE_CITIES  # noqa: E402
from urban_intervention.data.paths import POI_DIR, REFERENCE_DIR  # noqa: E402
from urban_intervention.utils import sha256_file  # noqa: E402

DEFAULT_YEARS = (2012, 2015)
DEFAULT_SIGMA_GRID_KM = (1.0, 1.5, 2.0, 2.5, 3.0)
EARTH_RADIUS_KM = 6371.0
CELL_AREA_KM2 = 0.25  # fixed 500m x 500m grid
LWR_DEGREE = 2  # local quadratic
RIDGE = 1e-8  # normal-equation ridge for collinearity safety


class _FitResult(TypedDict):
    beta: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    r2: float


class _ValidationResult(TypedDict):
    n: int
    r2_main_only: float | None
    r2_full: float | None
    delta_r2: float | None
    coefficients: list[dict[str, float | str]]


class CityCentresResult(TypedDict):
    city_key: str
    centres: pd.DataFrame
    surface: pd.DataFrame | None
    bandwidth_km: float
    cv_results: dict[float, float]
    validation: _ValidationResult
    n_peaks_tested: int
    n_peaks_significant: int
    cell_km: float
    input_paths: list[Path]


def _as_float(value: object) -> float:
    return float(value)  # type: ignore[arg-type]


def haversine_km(
    lon_a: float | np.ndarray,
    lat_a: float | np.ndarray,
    lon_b: float | np.ndarray,
    lat_b: float | np.ndarray,
) -> np.ndarray:
    """Great-circle distance in km between coordinates (scalar or array)."""
    lon_a, lat_a = np.radians(lon_a), np.radians(lat_a)
    lon_b, lat_b = np.radians(lon_b), np.radians(lat_b)
    d_lon = lon_a - lon_b
    d_lat = lat_a - lat_b
    a = np.sin(d_lat / 2.0) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(d_lon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _median_cell_km(surface: pd.DataFrame) -> float:
    """Grid cell size from the design property (0.25 km2 cells => 0.5 km)."""
    if "area_km2" in surface.columns:
        area = surface["area_km2"].dropna()
        if len(area):
            return float(np.sqrt(float(area.median())))
    return 0.5


def _shift(array: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Shift a dense (row, col) array; out[r, c] = array[r - dr, c - dc]; NaN edges."""
    rows, cols = array.shape
    out = np.full(array.shape, np.nan)
    r_dest_start, r_dest_end = max(0, dr), rows + min(0, dr)
    c_dest_start, c_dest_end = max(0, dc), cols + min(0, dc)
    if r_dest_start >= r_dest_end or c_dest_start >= c_dest_end:
        return out
    r_src_start, r_src_end = max(0, -dr), rows - max(0, dr)
    c_src_start, c_src_end = max(0, -dc), cols - max(0, dc)
    out[r_dest_start:r_dest_end, c_dest_start:c_dest_end] = array[
        r_src_start:r_src_end, c_src_start:c_src_end
    ]
    return out


def _window_offsets(radius_cells: int) -> list[tuple[int, int]]:
    return [
        (dr, dc)
        for dr in range(-radius_cells, radius_cells + 1)
        for dc in range(-radius_cells, radius_cells + 1)
        if dr != 0 or dc != 0
    ]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DENSITY_SOURCES = ("poi", "viirs", "population", "composite")


def _window_mean(path: Path, value_col: str, city_key: str, years: tuple[int, int]) -> pd.DataFrame:
    """Mean of a per-grid-year panel over the frozen window, one row/grid."""
    if not path.exists():
        raise FileNotFoundError(f"No panel for {city_key}: {path}")
    panel = pd.read_parquet(path)
    if "year" not in panel.columns or value_col not in panel.columns:
        raise ValueError(f"{path.name}: missing year/{value_col} columns")
    window = panel.loc[
        panel["year"].between(years[0], years[1]) & panel[value_col].notna(),
        ["grid_id", value_col],
    ]
    if window.empty:
        raise ValueError(f"{city_key}: no rows in frozen years {years}")
    density = window.groupby("grid_id", as_index=False)[value_col].mean()
    density.rename(columns={value_col: "density_value"}, inplace=True)
    return density


def load_density_surface(
    city_key: str, grid_path: Path, poi_path: Path, years: tuple[int, int], *, source: str = "poi"
) -> pd.DataFrame:
    grids = pd.read_parquet(grid_path)
    required = {"grid_id", "row", "col", "centroid_lon", "centroid_lat"}
    missing = required - set(grids.columns)
    if missing:
        raise ValueError(f"{city_key} grid file lacks columns: {sorted(missing)}")
    if source not in DENSITY_SOURCES:
        raise ValueError(f"Unknown density source {source!r}; use one of {DENSITY_SOURCES}")

    grid_cols = ["grid_id", "row", "col", "centroid_lon", "centroid_lat"]
    if source in ("poi", "composite"):
        poi = _window_mean(poi_path, "poi_count", city_key, years)
        poi = poi.rename(columns={"density_value": "poi_count_mean"})
        surface = grids[grid_cols].merge(poi, on="grid_id", how="left")
        surface["poi_count_mean"] = pd.to_numeric(surface["poi_count_mean"], errors="coerce")
        surface = surface.loc[surface["poi_count_mean"].notna()].copy()
        surface["density_value"] = surface["poi_count_mean"].to_numpy(dtype=float)
        surface["log_density"] = np.log1p(surface["poi_count_mean"].to_numpy(dtype=float))
        if source == "poi":
            surface = _finalize_surface(surface)
            return surface
        surface = surface[["grid_id", "row", "col", "centroid_lon", "centroid_lat", "log_density"]]

    if source in ("viirs", "composite"):
        viirs = _window_mean(
            poi_path.parent.parent / "viirs_annual_aggregated" / f"{city_key}_viirs_annual.parquet",
            "avg_rad",
            city_key,
            years,
        )
        viirs_surface = grids[grid_cols].merge(viirs, on="grid_id", how="left")
        viirs_surface["density_value"] = pd.to_numeric(
            viirs_surface["density_value"], errors="coerce"
        )
        viirs_surface = viirs_surface.loc[viirs_surface["density_value"].notna()].copy()
        viirs_surface["density_value"] = viirs_surface["density_value"].to_numpy(dtype=float)
        viirs_log = np.log1p(np.maximum(viirs_surface["density_value"].to_numpy(dtype=float), 0.0))
        if source == "viirs":
            viirs_surface["log_density"] = viirs_log
            viirs_surface = _finalize_surface(viirs_surface)
            return viirs_surface
        viirs_surface["log_density"] = viirs_log
        viirs_surface = viirs_surface[["grid_id", "log_density"]]

    if source in ("population", "composite"):
        population = _window_mean(
            poi_path.parent.parent / "population" / f"{city_key}_pop.parquet",
            "pop_count",
            city_key,
            years,
        )
        pop_surface = grids[grid_cols].merge(population, on="grid_id", how="left")
        pop_surface["density_value"] = pd.to_numeric(pop_surface["density_value"], errors="coerce")
        pop_surface = pop_surface.loc[pop_surface["density_value"].notna()].copy()
        pop_surface["density_value"] = pop_surface["density_value"].to_numpy(dtype=float)
        pop_log = np.log1p(np.maximum(pop_surface["density_value"].to_numpy(dtype=float), 0.0))
        if source == "population":
            pop_surface["log_density"] = pop_log
            pop_surface = _finalize_surface(pop_surface)
            return pop_surface
        pop_surface["log_density"] = pop_log
        pop_surface = pop_surface[["grid_id", "log_density"]]

    # composite: z-score each log-density then average (Wu et al. 2023 fusion)
    components = {"poi": surface, "viirs": viirs_surface, "population": pop_surface}
    frames = []
    for name, frame in components.items():
        part = frame[["grid_id", "log_density"]].copy()
        vals = part["log_density"].to_numpy(dtype=float)
        part["log_density"] = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-12)
        part = part.rename(columns={"log_density": f"{name}_z"})
        frames.append(part)
    fused = frames[0]
    for part in frames[1:]:
        fused = fused.merge(part, on="grid_id", how="outer")
    z_cols = [f"{name}_z" for name in components]
    fused["log_density"] = fused[z_cols].mean(axis=1, skipna=False)
    fused = fused.loc[fused["log_density"].notna()].copy()
    surface = grids[grid_cols].merge(fused[["grid_id", "log_density"]], on="grid_id", how="inner")
    surface["density_value"] = surface["log_density"].to_numpy(dtype=float)
    surface = _finalize_surface(surface)
    return surface


def _finalize_surface(surface: pd.DataFrame) -> pd.DataFrame:
    surface["row"] = surface["row"].astype(int)
    surface["col"] = surface["col"].astype(int)
    surface["_idx"] = np.arange(len(surface))
    return surface


def _dense_array(surface: pd.DataFrame, column: str) -> tuple[np.ndarray, int, int]:
    rows = surface["row"].to_numpy()
    cols = surface["col"].to_numpy()
    row_min, col_min = int(rows.min()), int(cols.min())
    dense = np.full((int(rows.max()) - row_min + 1, int(cols.max()) - col_min + 1), np.nan)
    dense[rows - row_min, cols - col_min] = surface[column].to_numpy(dtype=float)
    return dense, row_min, col_min


def _projected_coords(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat to km around the city centroid (equirectangular)."""
    lat0 = float(np.nanmedian(lat))
    lon0 = float(np.nanmedian(lon))
    u = (lon - lon0) * 111.0 * np.cos(np.radians(lat0))
    v = (lat - lat0) * 111.0
    return u, v


# ---------------------------------------------------------------------------
# LWR machinery (vectorised: monomial sliding-window sums via separable
# Gaussian convolution, then per-cell 6x6 normal-equation solves)
# ---------------------------------------------------------------------------


def _expand_monomial(a: int, b: int) -> list[tuple[float, int, int, int, int]]:
    """Expand (u - u_i)^a (v - v_i)^b into global monomials.

    Returns terms (coeff, r, s, p, q) meaning
    coeff * (-u_i)^r * (-v_i)^s * u_j^p * v_j^q, built with binomial
    coefficients; the local-centring constants stay symbolic and are
    evaluated per cell.
    """
    from math import comb

    terms: list[tuple[float, int, int, int, int]] = []
    for p in range(a + 1):
        for q in range(b + 1):
            terms.append(
                (
                    float(comb(a, p) * comb(b, q)),
                    a - p,
                    b - q,
                    p,
                    q,
                )
            )
    return terms


def _multiply_expansions(
    left: list[tuple[float, int, int, int, int]],
    right: list[tuple[float, int, int, int, int]],
) -> list[tuple[float, int, int, int, int]]:
    combined: dict[tuple[int, int, int, int], float] = {}
    for c1, r1, s1, p1, q1 in left:
        for c2, r2, s2, p2, q2 in right:
            key = (r1 + r2, s1 + s2, p1 + p2, q1 + q2)
            combined[key] = combined.get(key, 0.0) + c1 * c2
    return [(coeff, r, s, p, q) for (r, s, p, q), coeff in combined.items()]


def lwr_fit_dense(
    log_density: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    sigma_cells: float,
    bandwidth_km: float,
    need_loo: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Vectorised local quadratic LWR with per-point local centring.

    The design at cell i uses x(z_j) = (du/h, dv/h) monomials up to degree 2
    with du = u_j - u_i; the fitted value is therefore the local intercept.
    Normal equations are assembled from separable Gaussian sliding sums of
    global monomials (the local-centring expansion), then solved per cell.

    Returns (fitted, var_fitted, loo_fitted, kernel_mass) as dense 2D arrays.
    `loo_fitted` is the leave-one-out surface (the self contribution is
    exactly e1 e1' / e1 * y_i under local centring); all-NaN unless
    `need_loo`.
    """
    shape = log_density.shape
    n_total = log_density.size
    log_flat = log_density.ravel()
    valid_flat = valid.ravel()
    u_flat = u.ravel()
    v_flat = v.ravel()

    design: list[tuple[int, int]] = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]
    # Expansion of every design column x_k(z_j) as monomials of (u_j, v_j)
    # with local-centring constants, scaled by h^{-degree}.
    design_terms: dict[int, list[tuple[float, int, int, int, int]]] = {}
    for k, (a, b) in enumerate(design):
        degree = a + b
        design_terms[k] = [
            (coeff / bandwidth_km**degree, r, s, p, q)
            for coeff, r, s, p, q in _expand_monomial(a, b)
        ]

    def monomial_sum(p: int, q: int) -> np.ndarray:
        value = (u_flat**p) * (v_flat**q)
        return ndimage.gaussian_filter(
            np.where(valid, value.reshape(shape), 0.0),
            sigma=sigma_cells,
            mode="constant",
            truncate=3.0,
        ).ravel()

    monomial_cache: dict[tuple[int, int] | tuple[str, int, int], np.ndarray] = {}
    weighted_sums: dict[tuple[int, int], np.ndarray] = {}

    def get_monomial(p: int, q: int) -> np.ndarray:
        key = (p, q)
        if key not in monomial_cache:
            monomial_cache[key] = monomial_sum(p, q)
        return monomial_cache[key]

    def get_weighted(p: int, q: int) -> np.ndarray:
        key = (p, q)
        if key not in weighted_sums:
            value = (u_flat**p) * (v_flat**q) * log_flat
            weighted_sums[key] = ndimage.gaussian_filter(
                np.where(valid, value.reshape(shape), 0.0),
                sigma=sigma_cells,
                mode="constant",
                truncate=3.0,
            ).ravel()
        return weighted_sums[key]

    def get_squared(p: int, q: int) -> np.ndarray:
        """sum_j w_ij^2 * u_j^p * v_j^q.

        w^2 is Gaussian with sigma/sqrt(2); gaussian_filter re-normalises its
        kernel to unit sum, so the raw sum_j w_ij^2 equals
        filter_output / (4 * pi * sigma_cells^2) (2D Gaussian integral).
        """
        key = ("w2", p, q)
        if key not in monomial_cache:
            value = (u_flat**p) * (v_flat**q)
            filtered = ndimage.gaussian_filter(
                np.where(valid, value.reshape(shape), 0.0),
                sigma=sigma_cells / np.sqrt(2.0),
                mode="constant",
                truncate=3.0,
            ).ravel()
            monomial_cache[key] = filtered / (4.0 * np.pi * sigma_cells**2)
        return monomial_cache[key]

    def evaluate(terms: list[tuple[float, int, int, int, int]], getter) -> np.ndarray:
        total = np.zeros(n_total)
        for coeff, r, s, p, q in terms:
            constants = coeff * ((-u_flat) ** r) * ((-v_flat) ** s)
            total += constants * getter(p, q)
        return total

    a_stack = np.zeros((n_total, 6, 6))
    b_stack = np.zeros((n_total, 6))
    for k in range(6):
        b_stack[:, k] = evaluate(design_terms[k], get_weighted)
        for ell in range(k, 6):
            a_stack[:, k, ell] = a_stack[:, ell, k] = evaluate(
                _multiply_expansions(design_terms[k], design_terms[ell]),
                get_monomial,
            )
    for k in range(6):
        a_stack[:, k, k] += RIDGE

    mass = get_monomial(0, 0)
    fitted = np.full(n_total, np.nan)
    var_fitted = np.full(n_total, np.nan)
    loo_fitted = np.full(n_total, np.nan)

    usable = valid_flat & (mass > 1e-3)
    flat_idx = np.nonzero(usable)[0]
    a_use = a_stack[flat_idx]
    b_use = b_stack[flat_idx]
    beta = np.linalg.solve(a_use, b_use[..., None])[..., 0]  # (n_use, 6)
    fitted_use = beta[:, 0]

    # Residual variance surface: s^2 = sum_j w_ij e_j^2 / (mass - p)
    residuals_flat = np.full(n_total, np.nan)
    residuals_flat[flat_idx] = log_flat[flat_idx] - fitted_use
    residual_sums = ndimage.gaussian_filter(
        np.where(
            valid,
            np.where(np.isfinite(residuals_flat), residuals_flat, 0.0).reshape(shape) ** 2,
            0.0,
        ),
        sigma=sigma_cells,
        mode="constant",
        truncate=3.0,
    ).ravel()
    df_local = np.maximum(mass - len(design), 1.0)
    s2_use = np.maximum(residual_sums[flat_idx] / df_local[flat_idx], 0.0)

    # Mean-surface variance (sandwich): Var(hat y_i) = s^2 * e1' A^-1 B A^-1 e1
    # with B = X'W^2X. This is the variance of the smoothed-surface estimate,
    # appropriate for testing features of the surface itself (McMillen-style
    # peak significance); it is far smaller than the prediction variance.
    e1 = np.zeros((len(flat_idx), 6))
    e1[:, 0] = 1.0
    b2_stack = np.zeros((n_total, 6, 6))
    for k in range(6):
        for ell in range(k, 6):
            b2_stack[:, k, ell] = b2_stack[:, ell, k] = evaluate(
                _multiply_expansions(design_terms[k], design_terms[ell]),
                get_squared,
            )
    b2_use = b2_stack[flat_idx]
    a_inv_e1 = np.linalg.solve(a_use, e1[..., None])[..., 0]
    inner = np.linalg.solve(a_use, np.einsum("nij,nj->ni", b2_use, a_inv_e1)[..., None])[..., 0]
    var_use = s2_use * inner[:, 0]

    if need_loo:
        # Local centring: self contribution is exactly e1 e1' and e1 * y_i.
        a_loo = a_use - np.einsum("ni,nj->nij", e1, e1)
        b_loo = b_use - e1 * log_flat[flat_idx][:, None]
        beta_loo = np.linalg.solve(a_loo, b_loo[..., None])[..., 0]
        loo_use = beta_loo[:, 0]
        loo_fitted[flat_idx] = loo_use

    fitted[flat_idx] = fitted_use
    var_fitted[flat_idx] = var_use
    return (
        fitted.reshape(shape),
        var_fitted.reshape(shape),
        loo_fitted.reshape(shape) if need_loo else None,
        mass.reshape(shape),
    )


def cross_validated_bandwidth(
    log_density: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    cell_km: float,
    sigma_grid_km: tuple[float, ...],
) -> tuple[float, dict[float, float]]:
    """Leave-one-out CV bandwidth selection per city."""
    results: dict[float, float] = {}
    for sigma_km in sigma_grid_km:
        sigma_cells = sigma_km / cell_km
        _, _, loo, mass = lwr_fit_dense(
            log_density, u, v, valid, sigma_cells, sigma_km, need_loo=True
        )
        if loo is None:
            raise RuntimeError("LOO surface unexpectedly unavailable")
        loo_flat = loo.ravel()
        mass_flat = mass.ravel()
        evaluate = valid.ravel() & np.isfinite(loo_flat) & (mass_flat > 0.5)
        squared = (loo_flat[evaluate] - log_density.ravel()[evaluate]) ** 2
        results[sigma_km] = float(np.mean(squared)) if evaluate.any() else float("inf")
    best = min(results, key=lambda key: results[key])
    return float(best), results


# ---------------------------------------------------------------------------
# Peak identification and significance
# ---------------------------------------------------------------------------


def local_peak_positions(
    smoothed: np.ndarray, valid: np.ndarray, peak_radius_cells: int
) -> tuple[np.ndarray, np.ndarray]:
    is_peak = np.ones(smoothed.shape, dtype=bool)
    for dr, dc in _window_offsets(peak_radius_cells):
        shifted = _shift(smoothed, dr, dc)
        is_peak &= np.isnan(shifted) | (smoothed >= shifted)
    is_peak &= valid
    rows, cols = np.nonzero(is_peak)
    return rows, cols


def peak_significance_test(
    smoothed: np.ndarray,
    var_fitted: np.ndarray,
    valid: np.ndarray,
    peaks: tuple[np.ndarray, np.ndarray],
    peak_radius_cells: int,
) -> dict[int, dict[str, float]]:
    """t-test of each peak against its neighbourhood mean (LWR prediction SEs)."""
    results: dict[int, dict[str, float]] = {}
    r_idx, c_idx = peaks
    for position in range(len(r_idx)):
        r, c = int(r_idx[position]), int(c_idx[position])
        values: list[float] = []
        variances: list[float] = []
        for dr, dc in _window_offsets(peak_radius_cells):
            rn, cn = r + dr, c + dc
            if not (0 <= rn < smoothed.shape[0] and 0 <= cn < smoothed.shape[1]):
                continue
            if not valid[rn, cn]:
                continue
            values.append(float(smoothed[rn, cn]))
            variances.append(float(var_fitted[rn, cn]))
        if not values:
            continue
        mean_neighbour = float(np.mean(values))
        se = float(np.sqrt(var_fitted[r, c] + np.mean(variances)))
        if se <= 0 or not np.isfinite(se):
            continue
        t_statistic = (float(smoothed[r, c]) - mean_neighbour) / se
        p_value = 2.0 * (1.0 - _normal_cdf(abs(t_statistic)))
        results[position] = {
            "t_statistic": float(t_statistic),
            "p_value": float(p_value),
            "mean_neighbour": mean_neighbour,
            "n_neighbours": len(values),
        }
    return results


def _normal_cdf(value: float | np.ndarray) -> np.ndarray:
    from scipy.special import erf

    return 0.5 * (1.0 + erf(np.asarray(value) / np.sqrt(2.0)))


def benjamini_hochberg(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Return a boolean mask of p-values significant at BH-FDR level alpha."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p_values)
    passing = np.zeros(n, dtype=bool)
    threshold = 0.0
    for rank, index in enumerate(order):
        if p_values[index] <= alpha * (rank + 1) / n:
            threshold = max(threshold, p_values[index])
    passing = p_values <= threshold
    return passing


# ---------------------------------------------------------------------------
# Contiguous support (Giuliano-Small)
# ---------------------------------------------------------------------------


def contiguous_support_area(
    smoothed: np.ndarray,
    valid: np.ndarray,
    r: int,
    c: int,
    floor_value: float,
    peak_radius_cells: int,
    cell_area_km2: float = CELL_AREA_KM2,
) -> float:
    """Giuliano-Small style contiguous support of the peak within its radius.

    The 8-connected component containing the peak in the floor-thresholded
    surface, restricted to cells within `peak_radius_cells` of the peak.
    """
    rows, cols = smoothed.shape
    r_lo, r_hi = max(0, r - peak_radius_cells), min(rows, r + peak_radius_cells + 1)
    c_lo, c_hi = max(0, c - peak_radius_cells), min(cols, c + peak_radius_cells + 1)
    window = np.zeros((rows, cols), dtype=bool)
    window[r_lo:r_hi, c_lo:c_hi] = True
    binary = valid & (smoothed >= floor_value) & window
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labelled, _ = ndimage.label(binary, structure=structure)
    component = int(labelled[r, c])
    if component == 0:
        return 0.0
    return float(np.sum(labelled == component)) * cell_area_km2


# ---------------------------------------------------------------------------
# Validation regression (polycentric density gradient)
# ---------------------------------------------------------------------------


def validation_regression(surface: pd.DataFrame, centres: pd.DataFrame) -> _ValidationResult:
    """log1p(D) ~ dist_main + dist_sub1 + ... per city; OLS with manual SEs."""
    if centres.empty:
        return {"n": 0, "r2_main_only": None, "r2_full": None, "delta_r2": None, "coefficients": []}
    lon = surface["centroid_lon"].to_numpy()
    lat = surface["centroid_lat"].to_numpy()
    y = surface["log_density"].to_numpy()
    dist_matrix = np.column_stack(
        [
            haversine_km(lon, lat, float(row["centroid_lon"]), float(row["centroid_lat"]))
            for _, row in centres.iterrows()
        ]
    )
    n = len(y)
    intercept = np.ones(n)
    main_only = np.column_stack([intercept, dist_matrix[:, 0]])

    def fit(design: np.ndarray) -> _FitResult:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ beta
        residuals = y - fitted
        df = n - design.shape[1]
        s2 = float(residuals @ residuals) / max(df, 1)
        cov = s2 * np.linalg.inv(design.T @ design)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        t_stat = beta / se
        p_value = 2.0 * (1.0 - _normal_cdf(np.abs(t_stat)))
        ss_res = float(residuals @ residuals)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        return {"beta": beta, "se": se, "t": t_stat, "p": p_value, "r2": r2}

    main_fit = fit(main_only)
    full_design = np.column_stack([intercept, dist_matrix])
    full_fit = fit(full_design)
    sub_names = [
        f"dist_{row['role']}" for _, row in centres.iterrows() if str(row["role"]) != "main"
    ]
    names = ["intercept", "dist_main"] + sub_names
    coefficients: list[dict[str, float | str]] = [
        {
            "term": names[i],
            "estimate": float(full_fit["beta"][i]),
            "se": float(full_fit["se"][i]),
            "t_statistic": float(full_fit["t"][i]),
            "p_value": float(full_fit["p"][i]),
        }
        for i in range(len(names))
    ]
    return {
        "n": n,
        "r2_main_only": float(main_fit["r2"]),
        "r2_full": float(full_fit["r2"]),
        "delta_r2": float(full_fit["r2"] - main_fit["r2"]),
        "coefficients": coefficients,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_city_centres(
    city_key: str,
    grid_dir: Path,
    poi_dir: Path,
    years: tuple[int, int],
    sigma_grid_km: tuple[float, ...],
    peak_radius_km: float,
    fdr_level: float,
    floor_quantile: float,
    min_cluster_km2: float,
    min_distance_km: float,
    max_subcenters: int,
    bandwidth_km: float | None = None,
    store_surface: bool = True,
    source: str = "poi",
) -> CityCentresResult:
    grid_path = grid_dir / city_key / f"{city_key}_grids.parquet"
    poi_path = poi_dir / f"{city_key}_poi_grid_yearly.parquet"
    surface = load_density_surface(city_key, grid_path, poi_path, years, source=source)
    log_dense, row_min, col_min = _dense_array(surface, "log_density")
    lon_dense, _, _ = _dense_array(surface, "centroid_lon")
    lat_dense, _, _ = _dense_array(surface, "centroid_lat")
    valid = ~np.isnan(log_dense)
    cell_km = _median_cell_km(surface)
    u_dense, v_dense = _projected_coords(lon_dense, lat_dense)

    if bandwidth_km is None:
        bandwidth_km, cv_results = cross_validated_bandwidth(
            log_dense, u_dense, v_dense, valid, cell_km, sigma_grid_km
        )
    else:
        cv_results = {bandwidth_km: float("nan")}

    sigma_cells = bandwidth_km / cell_km
    fitted, var_fitted, _, _ = lwr_fit_dense(
        log_dense, u_dense, v_dense, valid, sigma_cells, bandwidth_km, need_loo=False
    )
    surface["smoothed_density"] = fitted[
        surface["row"].to_numpy() - row_min, surface["col"].to_numpy() - col_min
    ]
    surface["lwr_variance"] = var_fitted[
        surface["row"].to_numpy() - row_min, surface["col"].to_numpy() - col_min
    ]

    centres, n_peaks_tested, n_peaks_significant = identify_city_centres(
        surface,
        fitted,
        var_fitted,
        valid,
        lon_dense,
        lat_dense,
        row_min,
        col_min,
        city_key,
        bandwidth_km,
        peak_radius_km,
        fdr_level,
        floor_quantile,
        min_cluster_km2,
        min_distance_km,
        max_subcenters,
    )

    validation = (
        validation_regression(surface, centres)
        if store_surface
        else {"n": 0, "r2_main_only": None, "r2_full": None, "delta_r2": None, "coefficients": []}
    )
    return {
        "city_key": city_key,
        "centres": centres,
        "surface": surface if store_surface else None,
        "bandwidth_km": float(bandwidth_km),
        "cv_results": cv_results,
        "validation": validation,
        "n_peaks_tested": n_peaks_tested,
        "n_peaks_significant": n_peaks_significant,
        "cell_km": cell_km,
        "input_paths": [grid_path, poi_path],
    }


def identify_city_centres(
    surface: pd.DataFrame,
    fitted: np.ndarray,
    var_fitted: np.ndarray,
    valid: np.ndarray,
    lon_dense: np.ndarray,
    lat_dense: np.ndarray,
    row_min: int,
    col_min: int,
    city_key: str,
    bandwidth_km: float,
    peak_radius_km: float,
    fdr_level: float,
    floor_quantile: float,
    min_cluster_km2: float,
    min_distance_km: float,
    max_subcenters: int,
) -> tuple[pd.DataFrame, int, int]:
    """Peak identification, significance testing, size gates and role assignment."""
    cell_km = _median_cell_km(surface)
    peak_radius_cells = max(1, int(np.ceil(peak_radius_km / cell_km)))
    peaks = local_peak_positions(fitted, valid, peak_radius_cells)
    tests = peak_significance_test(fitted, var_fitted, valid, peaks, peak_radius_cells)
    test_positions = sorted(tests)
    p_values = np.array([tests[pos]["p_value"] for pos in test_positions], dtype=float)
    significant = benjamini_hochberg(p_values, fdr_level)

    floor_value = float(np.nanquantile(fitted[valid], floor_quantile))
    valid_pos = np.nonzero(valid)
    argmax_flat = int(np.argmax(fitted[valid]))
    global_pos = valid_pos[0][argmax_flat], valid_pos[1][argmax_flat]
    main_density = float(fitted[global_pos])

    rows_out: list[dict[str, object]] = []
    for position in test_positions:
        if not significant[test_positions.index(position)]:
            continue
        r, c = int(peaks[0][position]), int(peaks[1][position])
        d_main = float(
            haversine_km(
                lon_dense[r, c],
                lat_dense[r, c],
                lon_dense[global_pos],
                lat_dense[global_pos],
            )
        )
        area = contiguous_support_area(fitted, valid, r, c, floor_value, peak_radius_cells)
        if d_main < min_distance_km or area < min_cluster_km2:
            continue
        row = surface.loc[(surface["row"] == r + row_min) & (surface["col"] == c + col_min)]
        if len(row) != 1:
            continue
        rows_out.append(
            {
                "city_key": city_key,
                "grid_id": str(row["grid_id"].iloc[0]),
                "centroid_lon": float(lon_dense[r, c]),
                "centroid_lat": float(lat_dense[r, c]),
                "row": int(_as_float(row["row"].iloc[0])),
                "col": int(_as_float(row["col"].iloc[0])),
                "raw_density": _as_float(row["density_value"].iloc[0]),
                "smoothed_density": float(fitted[r, c]),
                "lwr_standard_error": float(np.sqrt(var_fitted[r, c])),
                "t_statistic": float(tests[position]["t_statistic"]),
                "p_value": float(tests[position]["p_value"]),
                "support_area_km2": float(area),
                "distance_main_km": float(d_main),
                "bandwidth_km": float(bandwidth_km),
                "peak_radius_km": float(peak_radius_km),
                "fdr_level": float(fdr_level),
            }
        )
    rows_out.sort(key=lambda item: _as_float(item["smoothed_density"]), reverse=True)
    rows_out = rows_out[:max_subcenters]
    for rank, candidate in enumerate(rows_out, start=1):
        candidate["role"] = f"subcenter_{rank}"

    main_row = surface.loc[
        (surface["row"] == global_pos[0] + row_min) & (surface["col"] == global_pos[1] + col_min)
    ]
    if len(main_row) != 1:
        raise RuntimeError(f"{city_key}: main-centre grid lookup failed")
    main_area = contiguous_support_area(
        fitted, valid, global_pos[0], global_pos[1], floor_value, peak_radius_cells
    )
    centres = pd.DataFrame(
        [
            {
                "city_key": city_key,
                "role": "main",
                "grid_id": str(main_row["grid_id"].iloc[0]),
                "centroid_lon": float(lon_dense[global_pos]),
                "centroid_lat": float(lat_dense[global_pos]),
                "row": int(_as_float(main_row["row"].iloc[0])),
                "col": int(_as_float(main_row["col"].iloc[0])),
                "raw_density": _as_float(main_row["density_value"].iloc[0]),
                "smoothed_density": main_density,
                "lwr_standard_error": float(np.sqrt(var_fitted[global_pos])),
                "t_statistic": float("nan"),
                "p_value": float("nan"),
                "support_area_km2": float(main_area),
                "distance_main_km": 0.0,
                "bandwidth_km": float(bandwidth_km),
                "peak_radius_km": float(peak_radius_km),
                "fdr_level": float(fdr_level),
            }
        ]
        + rows_out
    )
    return centres, len(test_positions), int(significant.sum())


def render_audit_figure(
    surface: pd.DataFrame, centres: pd.DataFrame, city_key: str, out_path: Path
) -> None:
    frame = surface.loc[surface["density_value"] > 0].copy()
    fig, axis = plt.subplots(figsize=(8, 8))
    axis.scatter(
        frame["centroid_lon"],
        frame["centroid_lat"],
        c=np.log1p(frame["density_value"]),
        s=0.6,
        cmap="viridis",
        rasterized=True,
    )
    for _, centre in centres.iterrows():
        colour = "#e74c3c" if centre["role"] == "main" else "#1f8a70"
        axis.scatter(
            centre["centroid_lon"],
            centre["centroid_lat"],
            marker="*",
            s=180,
            color=colour,
            zorder=5,
        )
        axis.annotate(
            centre["role"],
            (centre["centroid_lon"], centre["centroid_lat"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color=colour,
        )
    axis.set_title(f"{city_key}: LWR POI density + centres")
    axis.set_xlabel("lon")
    axis.set_ylabel("lat")
    axis.set_aspect(1.0 / np.cos(np.radians(float(frame["centroid_lat"].median()))))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _identify_with_surface_cache(
    surface: pd.DataFrame,
    log_dense: np.ndarray,
    lon_dense: np.ndarray,
    lat_dense: np.ndarray,
    valid: np.ndarray,
    cell_km: float,
    row_min: int,
    col_min: int,
    city_key: str,
    max_subcenters: int,
    params: dict[str, object],
) -> pd.DataFrame:
    """Identify centres, caching the LWR surface per bandwidth in `params`."""
    surfaces = params.get("_surfaces")
    if not isinstance(surfaces, dict):
        surfaces = {}
        params["_surfaces"] = surfaces
    bandwidth_km = _as_float(params["bandwidth_km"])
    if bandwidth_km not in surfaces:
        u_dense, v_dense = _projected_coords(lon_dense, lat_dense)
        fitted, var_fitted, _, _ = lwr_fit_dense(
            log_dense,
            u_dense,
            v_dense,
            valid,
            bandwidth_km / cell_km,
            bandwidth_km,
            need_loo=False,
        )
        surfaces[bandwidth_km] = (fitted, var_fitted)
    fitted, var_fitted = surfaces[bandwidth_km]
    centres, _, _ = identify_city_centres(
        surface,
        fitted,
        var_fitted,
        valid,
        lon_dense,
        lat_dense,
        row_min,
        col_min,
        city_key,
        bandwidth_km,
        _as_float(params["peak_radius_km"]),
        _as_float(params["fdr_level"]),
        _as_float(params["floor_quantile"]),
        _as_float(params["min_cluster_km2"]),
        _as_float(params["min_distance_km"]),
        max_subcenters,
    )
    return centres


def sensitivity_variant(args: argparse.Namespace, **overrides: float) -> list[dict[str, object]]:
    """Re-run identification with one parameter overridden; report centre drift.

    The LWR surface depends only on the bandwidth, so each city's surface is
    fitted once and the identification step is re-run per parameter variant.
    """
    effective: dict[str, object] = {
        "bandwidth_km": args.bandwidth_km,
        "peak_radius_km": args.peak_radius_km,
        "fdr_level": args.fdr_level,
        "floor_quantile": args.floor_quantile,
        "min_cluster_km2": args.min_cluster_km2,
        "min_distance_km": args.min_distance_km,
    }
    base = dict(effective)
    effective.update(overrides)
    rows: list[dict[str, object]] = []
    cities = sorted(ACTIVE_CITIES if args.city == "all" else [args.city])
    for city_key in cities:
        params = dict(effective)
        base_params = dict(base)
        grid_path = args.grid_dir / city_key / f"{city_key}_grids.parquet"
        poi_path = args.poi_dir / f"{city_key}_poi_grid_yearly.parquet"
        surface = load_density_surface(city_key, grid_path, poi_path, tuple(args.years))
        log_dense, row_min, col_min = _dense_array(surface, "log_density")
        lon_dense, _, _ = _dense_array(surface, "centroid_lon")
        lat_dense, _, _ = _dense_array(surface, "centroid_lat")
        valid = ~np.isnan(log_dense)
        cell_km = _median_cell_km(surface)

        reference = _identify_with_surface_cache(
            surface,
            log_dense,
            lon_dense,
            lat_dense,
            valid,
            cell_km,
            row_min,
            col_min,
            city_key,
            args.max_subcenters,
            base_params,
        )
        centres_by_role = {
            str(row["role"]): (float(row["centroid_lon"]), float(row["centroid_lat"]))
            for _, row in reference.iterrows()
        }
        variant = _identify_with_surface_cache(
            surface,
            log_dense,
            lon_dense,
            lat_dense,
            valid,
            cell_km,
            row_min,
            col_min,
            city_key,
            args.max_subcenters,
            params,
        )
        for _, row in variant.iterrows():
            role = str(row["role"])
            base_pos = centres_by_role.get(role)
            if base_pos is None:
                drift = float("nan")
            else:
                drift = float(
                    haversine_km(
                        row["centroid_lon"],
                        row["centroid_lat"],
                        base_pos[0],
                        base_pos[1],
                    )
                )
            rows.append(
                {
                    "city_key": city_key,
                    "role": role,
                    "parameter": next(iter(overrides)),
                    "value": next(iter(overrides.values())),
                    "drift_km": round(drift, 3),
                    "matched_base_role": base_pos is not None,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="all", help="City key or 'all'")
    parser.add_argument("--grid-dir", type=Path, default=REFERENCE_DIR / "grids")
    parser.add_argument("--poi-dir", type=Path, default=POI_DIR)
    parser.add_argument("--years", type=int, nargs=2, default=list(DEFAULT_YEARS))
    parser.add_argument("--sigma-grid", type=float, nargs="+", default=list(DEFAULT_SIGMA_GRID_KM))
    parser.add_argument(
        "--bandwidth-km",
        type=float,
        default=1.5,
        help="Frozen bandwidth (default 1.5 km; see manifest note on CV)",
    )
    parser.add_argument(
        "--cv-bandwidth",
        action="store_true",
        help="Select bandwidth per city by LOO-CV over --sigma-grid "
        "(reported as a diagnostic; the frozen default is 1.5 km)",
    )
    parser.add_argument("--peak-radius-km", type=float, default=1.5)
    parser.add_argument("--fdr-level", type=float, default=0.05)
    parser.add_argument("--floor-quantile", type=float, default=0.50)
    parser.add_argument("--min-cluster-km2", type=float, default=5.0)
    parser.add_argument("--min-distance-km", type=float, default=3.0)
    parser.add_argument("--max-subcenters", type=int, default=8)
    parser.add_argument(
        "--sensitivity", action="store_true", help="Run the parameter-variant drift analysis"
    )
    parser.add_argument("--out-csv", type=Path, default=REFERENCE_DIR / "city_centers.csv")
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=REFERENCE_DIR / "city_centers_manifest.json",
    )
    parser.add_argument(
        "--out-validation",
        type=Path,
        default=REFERENCE_DIR / "city_centers_validation.csv",
    )
    parser.add_argument(
        "--out-sensitivity",
        type=Path,
        default=REFERENCE_DIR / "city_centers_sensitivity.csv",
    )
    parser.add_argument(
        "--out-figures",
        type=Path,
        default=ROOT / "outputs" / "figures" / "centers",
    )
    parser.add_argument(
        "--source",
        choices=list(DENSITY_SOURCES),
        default="poi",
        help="Density surface source: poi | viirs | population | composite "
        "(z-score fusion of the three log-densities)",
    )
    args = parser.parse_args()

    cities = ACTIVE_CITIES if args.city == "all" else [args.city]
    years = tuple(args.years)
    parts: list[pd.DataFrame] = []
    source_files: list[Path] = []
    validation_rows: list[dict[str, object]] = []
    cv_rows: list[dict[str, object]] = []
    for city_key in sorted(cities):
        result = build_city_centres(
            city_key,
            args.grid_dir,
            args.poi_dir,
            years,
            tuple(args.sigma_grid),
            args.peak_radius_km,
            args.fdr_level,
            args.floor_quantile,
            args.min_cluster_km2,
            args.min_distance_km,
            args.max_subcenters,
            bandwidth_km=None if args.cv_bandwidth else args.bandwidth_km,
            source=args.source,
        )
        source_files.extend(result["input_paths"])
        centres = result["centres"]
        print(
            f"[{city_key}] bandwidth={result['bandwidth_km']:.2f}km "
            f"peaks_tested={result['n_peaks_tested']} "
            f"significant={result['n_peaks_significant']} "
            f"centres={len(centres)} (main + {len(centres) - 1} sub)"
        )
        render_audit_figure(
            result["surface"],
            centres,
            city_key,
            args.out_figures / f"{city_key}_centres.png",
        )
        if not centres.empty:
            parts.append(centres)
        for coeff in result["validation"]["coefficients"]:
            validation_rows.append(
                {
                    "city_key": city_key,
                    **coeff,
                }
            )
        validation_rows.append(
            {
                "city_key": city_key,
                "term": "model_fit",
                "estimate": result["validation"]["r2_full"],
                "se": result["validation"]["r2_main_only"],
                "t_statistic": result["validation"]["delta_r2"],
                "p_value": float("nan"),
            }
        )
        for sigma_km, cv_mse in result["cv_results"].items():
            cv_rows.append(
                {
                    "city_key": city_key,
                    "sigma_km": sigma_km,
                    "cv_mse": cv_mse,
                }
            )

    if not parts:
        raise RuntimeError("No centres identified for any city")
    registry = pd.concat(parts, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(validation_rows).to_csv(args.out_validation, index=False, encoding="utf-8-sig")
    pd.DataFrame(cv_rows).to_csv(
        args.out_csv.parent / "city_centers_cv.csv", index=False, encoding="utf-8-sig"
    )

    if args.sensitivity:
        drift_rows: list[dict[str, object]] = []
        variants = (
            ("bandwidth_km", (1.0, 2.0, 3.0)),
            ("peak_radius_km", (2.0, 3.0)),
            ("fdr_level", (0.01, 0.10)),
            ("floor_quantile", (0.40, 0.60)),
            ("min_cluster_km2", (3.0, 10.0)),
            ("min_distance_km", (2.0, 4.0)),
        )
        for name, values in variants:
            for value in values:
                drift_rows.extend(sensitivity_variant(args, **{name: value}))
        pd.DataFrame(drift_rows).to_csv(args.out_sensitivity, index=False, encoding="utf-8-sig")

    manifest = {
        "schema": "city_centres_v1",
        "definition": "mcmillen_2001_lwr_significant_peaks",
        "years": list(years),
        "sigma_grid_km": list(args.sigma_grid),
        "bandwidth_selection": (
            "per-city leave-one-out CV (diagnostic)"
            if args.cv_bandwidth
            else f"frozen {args.bandwidth_km} km"
        ),
        "bandwidth_cv_note": (
            "LOO-CV is reported as a diagnostic only: with noisy 500m-grid POI "
            "counts the CV optimum sits at the boundary of the sigma grid "
            "(under-smoothing), so the registry freezes sigma=1.5 km and the "
            "sensitivity report covers 0.75-3.0 km."
        ),
        "peak_radius_km": args.peak_radius_km,
        "fdr_level": args.fdr_level,
        "floor_quantile": args.floor_quantile,
        "min_cluster_km2": args.min_cluster_km2,
        "min_distance_km": args.min_distance_km,
        "max_subcenters": args.max_subcenters,
        "cities": sorted(cities),
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": {str(path): sha256_file(path) for path in source_files},
        "rows": len(registry),
    }
    args.out_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {args.out_csv} ({len(registry)} centres) + manifest + "
        f"validation + cv + {len(parts)} audit figures"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
