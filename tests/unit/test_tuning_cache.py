from __future__ import annotations

from pathlib import Path

import pandas as pd

from urban_intervention.causal.gpu.contracts import FORMAL_IMPLEMENTATION_VERSION
from urban_intervention.causal.gpu.tuning_cache import (
    TUNING_CACHE_SCHEMA,
    load_tuning_cache,
    panel_tuning_signature,
    write_tuning_cache,
)


def _panel(target_shift: float = 0.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for role, grid, offset in (("donor", "d1", 0.0), ("treated", "target", target_shift)):
        for time_id in range(1, 5):
            rows.append(
                {
                    "city_key": "a",
                    "grid_id": grid,
                    "time_id": time_id,
                    "model_value": offset + time_id,
                    "role": role,
                    "D": int(role == "treated" and time_id >= 3),
                }
            )
    return pd.DataFrame(rows)


def test_gsc_signature_reuses_identical_control_panel_but_mc_does_not() -> None:
    contract = {"folds": 5, "seed": 7}
    first = _panel()
    changed_target = _panel(target_shift=10.0)
    assert panel_tuning_signature(
        first, "gsc", tuning_contract=contract
    ) == panel_tuning_signature(changed_target, "gsc", tuning_contract=contract)
    assert panel_tuning_signature(
        first, "mc", tuning_contract=contract
    ) != panel_tuning_signature(changed_target, "mc", tuning_contract=contract)


def test_tuning_cache_is_fail_closed_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    signature = "abc123"
    payload = {
        "schema": TUNING_CACHE_SCHEMA,
        "implementation_version": FORMAL_IMPLEMENTATION_VERSION,
        "estimator": "gsc",
        "panel_signature": signature,
        "selected_tuning": 2,
        "cv_min_mspe": 0.25,
    }
    write_tuning_cache(path, payload)
    assert load_tuning_cache(path, estimator="gsc", signature=signature) == payload
    assert load_tuning_cache(path, estimator="gsc", signature="different") is None
    assert load_tuning_cache(path, estimator="mc", signature=signature) is None

