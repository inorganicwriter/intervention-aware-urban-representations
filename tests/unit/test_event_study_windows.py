from __future__ import annotations

import pytest

from urban_intervention.causal.event_study import (
    CANONICAL_EVENT_WINDOWS,
    canonical_event_window,
)


def test_canonical_windows_cover_label_horizons_and_keep_five_leads() -> None:
    assert canonical_event_window("viirs") == (-24, 24)
    assert canonical_event_window("housing") == (-24, 24)
    assert canonical_event_window("poi") == (-6, 3)
    assert canonical_event_window("population") == (-6, 3)
    for family, (min_pre, max_post) in CANONICAL_EVENT_WINDOWS.items():
        assert min_pre <= -6, f"{family} must keep the five clean leads -6..-2"
        expected_max_post = 24 if family in {"viirs", "housing"} else 3
        assert max_post == expected_max_post, family


def test_canonical_window_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        canonical_event_window("unknown")
