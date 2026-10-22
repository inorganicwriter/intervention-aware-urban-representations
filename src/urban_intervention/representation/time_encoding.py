"""Time-gap encodings shared by visual pair construction and model inputs."""

from __future__ import annotations

import math


STREETVIEW_AGE_HALF_SCALE_MONTHS = 12.0


def compress_time_gap_months(
    months: int | float,
    *,
    half_scale_months: float = STREETVIEW_AGE_HALF_SCALE_MONTHS,
) -> float:
    """Map a nonnegative month gap to [0, 1) using ``d / (d + tau)``.

    The default ``tau=12`` makes a one-year gap equal to 0.5. The result is a
    temporal input feature, not a sample or loss weight.
    """
    gap = float(months)
    scale = float(half_scale_months)
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("months must be a finite, nonnegative time gap")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("half_scale_months must be finite and greater than zero")
    return gap / (gap + scale)
