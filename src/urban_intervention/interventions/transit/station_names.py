"""Compatibility import for canonical station-name normalization.

The implementation lives in :mod:`urban_intervention.text` so the configuration
layer does not depend on the transit domain package.
"""

from __future__ import annotations

from urban_intervention.text import normalize_station_name

__all__ = ["normalize_station_name"]
