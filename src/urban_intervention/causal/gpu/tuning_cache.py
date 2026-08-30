"""Content-addressed tuning cache for the formal Python estimators."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .contracts import FORMAL_IMPLEMENTATION_VERSION
from .provenance import estimator_code_fingerprint

Estimator = Literal["gsc", "mc"]
TUNING_CACHE_SCHEMA = "causal_python_tuning_cache"


def panel_tuning_signature(
    frame: pd.DataFrame,
    estimator: Estimator,
    *,
    tuning_contract: dict[str, Any],
) -> str:
    """Hash exactly the cells that determine an estimator's tuning choice.

    GSC rank CV is performed only on controls, so the signature intentionally
    excludes the treated unit.  MC CV includes the treated unit's pre-period
    cells, so its signature covers the complete panel and treatment mask.
    """
    required = {"city_key", "grid_id", "time_id", "model_value", "role", "D"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"tuning signature lacks panel columns: {sorted(missing)}")
    selected = frame.loc[frame["role"].eq("donor")].copy() if estimator == "gsc" else frame.copy()
    selected = selected.sort_values(
        ["city_key", "grid_id", "time_id"], kind="stable"
    )[["city_key", "grid_id", "time_id", "model_value", "D"]]
    selected["model_value"] = pd.to_numeric(selected["model_value"], errors="coerce")
    selected["D"] = pd.to_numeric(selected["D"], errors="raise").astype(np.int8)
    row_hashes = pd.util.hash_pandas_object(selected, index=False, categorize=True)
    digest = hashlib.sha256()
    digest.update(estimator.encode("ascii"))
    digest.update(FORMAL_IMPLEMENTATION_VERSION.encode("ascii"))
    digest.update(estimator_code_fingerprint(estimator).encode("ascii"))
    digest.update(
        json.dumps(tuning_contract, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    )
    digest.update(np.asarray(row_hashes, dtype=np.uint64).tobytes())
    return digest.hexdigest()


def tuning_cache_path(root: Path, estimator: Estimator, signature: str) -> Path:
    return root / "python_tuning_cache" / estimator / f"{signature}.json"


def load_tuning_cache(
    path: Path, *, estimator: Estimator, signature: str
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema") != TUNING_CACHE_SCHEMA
        or payload.get("implementation_version") != FORMAL_IMPLEMENTATION_VERSION
        or payload.get("code_fingerprint") != estimator_code_fingerprint(estimator)
        or payload.get("estimator") != estimator
        or payload.get("panel_signature") != signature
    ):
        return None
    try:
        selected = float(payload["selected_tuning"])
        cv_min = float(payload["cv_min_mspe"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(selected) or selected < 0 or not np.isfinite(cv_min) or cv_min < 0:
        return None
    if estimator == "gsc" and not selected.is_integer():
        return None
    return payload


def write_tuning_cache(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a deterministic tuning result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)
