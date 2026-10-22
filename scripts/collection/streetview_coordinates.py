"""Convert actual Baidu panorama coordinates, preserving query coordinates.

The collector's sdata X/Y are BD09 Mercator multiplied by 100. Its existing
step5 conversion omitted that scale. Coefficients below reproduce the
collector's MC inverse; GCJ inverse is refined to avoid its one-step error.
No converted CSV or query location is used as the actual capture position.
"""
from __future__ import annotations

import numpy as np

PI = np.pi
A = 6378245.0
EE = 0.00669342162296594323
X_PI = PI * 3000.0 / 180.0
MCBAND = np.array([12890594.86, 8362377.87, 5591021, 3481989.83, 1678043.12, 0])
MC2LL = np.array([
    [1.410526172116255e-8, .00000898305509648872, -1.9939833816331, 200.9824383106796, -187.2403703815547, 91.6087516669843, -23.38765649603339, 2.57121317296198, -.03801003308653, 17337981.2],
    [-7.435856389565537e-9, .000008983055097726239, -.78625201886289, 96.32687599759846, -1.85204757529826, -59.36935905485877, 47.40033549296737, -16.50741931063887, 2.28786674699375, 10260144.86],
    [-3.030883460898826e-8, .00000898305509983578, .30071316287616, 59.74293618442277, 7.357984074871, -25.38371002664745, 13.45380521110908, -3.29883767235584, .32710905363475, 6856817.37],
    [-1.981981304930552e-8, .000008983055099779535, .03278182852591, 40.31678527705744, .65659298677277, -4.44255534477492, .85341911805263, .12923347998204, -.04625736007561, 4482777.06],
    [3.09191371068437e-9, .000008983055096812155, .00006995724062, 23.10934304144901, -.00023663490511, -.6321817810242, -.00663494467273, .03430082397953, -.00466043876332, 2555164.4],
    [2.890871144776878e-9, .000008983055095805407, -3.068298e-8, 7.47137025468032, -.00000353937994, -.02145144861037, -.00001234426596, .00010322952773, -.00000323890364, 826088.5],
])


def mc_to_bd(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    lon, lat = np.full_like(x, np.nan), np.full_like(y, np.nan)
    for i, lower in enumerate(MCBAND):
        mask = np.abs(y) >= lower
        if i:
            mask &= np.abs(y) < MCBAND[i - 1]
        c = MC2LL[i]
        z = np.abs(y[mask]) / c[9]
        lon[mask] = (c[0] + c[1] * np.abs(x[mask])) * np.sign(x[mask])
        lat[mask] = sum(c[j + 2] * z**j for j in range(7)) * np.sign(y[mask])
    return lon, lat


def bd_to_gcj(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = lon - .0065, lat - .006
    z = np.hypot(x, y) - .00002 * np.sin(y * X_PI)
    theta = np.arctan2(y, x) - .000003 * np.cos(x * X_PI)
    return z * np.cos(theta), z * np.sin(theta)


def wgs_to_gcj(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = lon - 105., lat - 35.
    dy = -100 + 2*x + 3*y + .2*y*y + .1*x*y + .2*np.sqrt(np.abs(x))
    dx = 300 + x + 2*y + .1*x*x + .1*x*y + .1*np.sqrt(np.abs(x))
    wave = (20*np.sin(6*x*PI) + 20*np.sin(2*x*PI)) * 2/3
    dy += wave + (20*np.sin(y*PI) + 40*np.sin(y/3*PI))*2/3
    dy += (160*np.sin(y/12*PI) + 320*np.sin(y*PI/30))*2/3
    dx += wave + (20*np.sin(x*PI) + 40*np.sin(x/3*PI))*2/3
    dx += (150*np.sin(x/12*PI) + 300*np.sin(x/30*PI))*2/3
    rad = lat / 180 * PI
    magic = 1 - EE * np.sin(rad)**2
    dy = dy * 180 / ((A*(1-EE)) / (magic*np.sqrt(magic)) * PI)
    dx = dx * 180 / (A / np.sqrt(magic) * np.cos(rad) * PI)
    return lon + dx, lat + dy


def gcj_to_wgs(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = lon.copy(), lat.copy()
    for _ in range(8):
        predicted_lon, predicted_lat = wgs_to_gcj(x, y)
        x -= predicted_lon - lon
        y -= predicted_lat - lat
    return x, y


def actual_pano_coordinates(
    ret_x: np.ndarray, ret_y: np.ndarray, calc_x: np.ndarray, calc_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve only supported units using the independent query MC values.

    Reject unknown scales rather than guessing a location from the query point.
    Both axes must agree within 5%, allowing normal nearest-panorama offsets.
    """
    ret_x, ret_y, calc_x, calc_y = [np.asarray(v, dtype=float) for v in (ret_x, ret_y, calc_x, calc_y)]
    scale = np.full_like(ret_x, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        for divisor in (1., 100.):
            valid = np.isfinite(ret_x + ret_y + calc_x + calc_y)
            valid &= (np.abs(ret_x / divisor / calc_x - 1) < .05)
            valid &= (np.abs(ret_y / divisor / calc_y - 1) < .05)
            scale[valid] = divisor
    lon, lat = mc_to_bd(ret_x / scale, ret_y / scale)
    lon, lat = bd_to_gcj(lon, lat)
    lon, lat = gcj_to_wgs(lon, lat)
    valid = (lon >= 70) & (lon <= 140) & (lat >= 15) & (lat <= 55)
    return np.where(valid, lon, np.nan), np.where(valid, lat, np.nan), scale
