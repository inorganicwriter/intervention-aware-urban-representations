"""Publish seven intervention fields and masks for the frozen treatment list.

Availability masks describe structurally usable reconstructed fields. They do
not certify historical line membership: that limitation is explicit metadata.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.data.paths import ACTIVE_DIR, TREATMENT_UNIT_LIST  # noqa: E402

ATTRIBUTES = ROOT / "outputs/causal_labels/station_attributes/station_attributes.parquet"
OUTPUT = ACTIVE_DIR / "intervention_grid_v1"
FIELDS = ["opening_year", "is_transfer_at_opening", "is_new_line_opening",
          "is_extension_opening", "is_terminal_at_opening", "n_lines_at_opening",
          "same_month_openings"]


def build_inputs(treatments: pd.DataFrame, attributes: pd.DataFrame) -> pd.DataFrame:
    attrs = attributes.loc[attributes.is_treated].copy()
    if treatments.treatment_order.duplicated().any() or attrs.treatment_order.duplicated().any():
        raise ValueError("Duplicate treatment_order in intervention sources")
    attrs["treatment_order"] = attrs.treatment_order.astype(int)
    merged = treatments.merge(attrs, on="treatment_order", how="left", validate="one_to_one",
                              suffixes=("", "_attribute"), indicator=True)
    if not merged["_merge"].eq("both").all() or len(attrs) != len(treatments):
        raise ValueError("Station attributes do not cover exactly the frozen treatment list")
    for key in ["city_key", "station_event_id"]:
        if not merged[key].eq(merged[f"{key}_attribute"]).all():
            raise ValueError(f"Station attribute identity mismatch: {key}")
    if not pd.to_datetime(merged.opening_date).eq(pd.to_datetime(merged.opening_date_attribute)).all():
        raise ValueError("Station attribute opening date mismatch")
    columns = ["treatment_order", "city_key", "grid_id", "station_event_id", "opening_date", "opening_month"]
    result = merged[columns].copy()
    result["opening_year"] = pd.to_datetime(merged.opening_date).dt.year.astype(float)
    for field in FIELDS[1:]:
        result[field] = pd.to_numeric(merged[field], errors="raise").astype(float)
    for field in FIELDS:
        valid = np.isfinite(result[field])
        if field.startswith("is_"):
            valid &= result[field].isin([0, 1])
        else:
            valid &= result[field].gt(0) & result[field].mod(1).eq(0)
        result.loc[~valid, field] = np.nan
    # With no operating line reconstructed, the related false flags have no
    # usable evidence either. Same-month city count and opening year stay valid.
    invalid_lines = result.n_lines_at_opening.isna()
    for field in ["is_transfer_at_opening", "is_new_line_opening", "is_extension_opening",
                  "is_terminal_at_opening"]:
        result.loc[invalid_lines, field] = np.nan
    for field in FIELDS:
        result[f"{field}_available"] = result[field].notna()
    result["terminal_evidence"] = merged.terminal_evidence
    result["line_reconstruction_valid"] = ~invalid_lines
    result["historical_source_review_status"] = "requires_source_review"
    result["attribute_provenance"] = "current_membership_and_adjacency_filtered_by_opening_dates"
    return result.sort_values("treatment_order").reset_index(drop=True)


def main() -> int:
    frame = build_inputs(pd.read_parquet(TREATMENT_UNIT_LIST), pd.read_parquet(ATTRIBUTES))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT / "intervention_inputs.parquet", index=False, compression="zstd")
    manifest = {
        "built_at_utc": datetime.now(UTC).isoformat(), "rows": len(frame),
        "cities": int(frame.city_key.nunique()), "fields": FIELDS,
        "mask_columns": [f"{field}_available" for field in FIELDS],
        "field_available_counts": {field: int(frame[f"{field}_available"].sum()) for field in FIELDS},
        "invalid_line_reconstruction_orders": frame.loc[~frame.line_reconstruction_valid, "treatment_order"].tolist(),
        "historical_source_review_status": "requires_source_review",
        "availability_mask_meaning": "structurally_usable_reconstruction_not_historical_certification",
        "sources": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [TREATMENT_UNIT_LIST, ATTRIBUTES]},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
