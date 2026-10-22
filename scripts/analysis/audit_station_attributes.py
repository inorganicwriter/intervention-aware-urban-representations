"""Audit existing station attributes against eligible representation inputs.

Writes only audit reports. Historical semantics require source review; a field
named *_at_opening or a structural pass is not historical verification.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FLAGS = (
    "is_transfer_at_opening",
    "is_new_line_opening",
    "is_extension_opening",
    "is_terminal_at_opening",
)
NUMERIC = ("n_lines_at_opening", "same_month_openings", "same_month_new_line_stations")


def _keys(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "treatment_order" not in frame:
        raise ValueError("Missing treatment_order")
    values = pd.to_numeric(frame["treatment_order"], errors="coerce")
    valid = values.notna() & np.isfinite(values) & (values >= 1) & (values == np.floor(values))
    invalid = int((frame["treatment_order"].notna() & ~valid).sum())
    result = frame.loc[valid].copy()
    result["treatment_order"] = values.loc[valid].astype("int64")
    return result, invalid


def audit_station_attributes(
    attributes: Path, dataset: Path, output_dir: Path
) -> dict[str, object]:
    if output_dir.resolve() in {dataset.resolve(), attributes.parent.resolve()}:
        raise ValueError("Use a separate audit output directory")
    attribute_bytes = attributes.read_bytes()
    raw = pd.read_parquet(io.BytesIO(attribute_bytes))
    attrs, invalid_orders = _keys(raw)
    duplicates = attrs["treatment_order"].duplicated(keep=False)
    duplicate_orders = sorted(attrs.loc[duplicates, "treatment_order"].unique().tolist())
    # Ambiguous keys remain uncovered rather than silently selecting one row.
    attrs = attrs.loc[~duplicates].set_index("treatment_order")
    index, invalid_index_orders = _keys(pd.read_parquet(dataset / "sample_index.parquet"))
    if invalid_index_orders or index["treatment_order"].duplicated().any():
        raise ValueError("Dataset sample_index has invalid or duplicate treatment_order")
    for column in ("split", "final_training_mask"):
        if column not in index:
            raise ValueError(f"Dataset sample_index lacks {column}")
    eligible = index.loc[index["final_training_mask"].fillna(False).astype(bool)].copy()
    if eligible.empty:
        raise ValueError("No eligible units")
    units_path = dataset / "unit_features.parquet"
    if units_path.is_file():
        available = pq.ParquetFile(units_path).schema_arrow.names
        wanted = [
            c
            for c in ("treatment_order", "city_key", "station_event_id", "opening_month")
            if c in available
        ]
        unit_frame, invalid_units = _keys(pd.read_parquet(units_path, columns=wanted))
        if invalid_units or unit_frame["treatment_order"].duplicated().any():
            raise ValueError("unit_features has invalid or duplicate treatment_order")
        for column in wanted:
            if column not in eligible and column != "treatment_order":
                eligible[column] = eligible["treatment_order"].map(
                    unit_frame.set_index("treatment_order")[column]
                )
    eligible = eligible.set_index("treatment_order")
    aligned = attrs.reindex(eligible.index)
    covered = pd.Series(eligible.index.isin(attrs.index), index=eligible.index)
    summary: dict[str, object] = {
        "dataset": str(dataset),
        "attributes": str(attributes),
        "attribute_sha256": hashlib.sha256(attribute_bytes).hexdigest(),
        "attribute_rows": len(raw),
        "invalid_attribute_orders": invalid_orders,
        "duplicate_attribute_orders": duplicate_orders,
        "eligible_units": len(eligible),
        "covered_units": int(covered.sum()),
        "uncovered_orders": eligible.index[~covered].tolist(),
        "missing_required_columns": [c for c in FLAGS if c not in raw],
        "flags": {},
        "numeric": {},
        "identity_checks": {},
        "coverage_groups": [],
        "temporal_status": "requires_source_review",
    }
    for column in FLAGS:
        if column not in aligned:
            continue
        values = pd.to_numeric(aligned[column], errors="coerce")
        bad = covered & aligned[column].notna() & (~values.isin([0, 1]))
        summary["flags"][column] = {
            "observed": int((covered & values.isin([0, 1])).sum()),
            "missing_on_covered_units": int((covered & aligned[column].isna()).sum()),
            "invalid": int(bad.sum()),
            "positive": int((covered & values.eq(1)).sum()),
        }
    for column in NUMERIC:
        if column not in aligned:
            continue
        values = pd.to_numeric(aligned[column], errors="coerce")
        valid = np.isfinite(values) & (values >= 0) & values.eq(np.floor(values))
        summary["numeric"][column] = {
            "missing_on_covered_units": int((covered & aligned[column].isna()).sum()),
            "invalid": int((covered & aligned[column].notna() & ~valid).sum()),
            "minimum": float(values[covered & valid].min()) if (covered & valid).any() else None,
            "maximum": float(values[covered & valid].max()) if (covered & valid).any() else None,
        }
    for column in ("city_key", "station_event_id"):
        if column in eligible and column in aligned:
            compare = covered & eligible[column].notna() & aligned[column].notna()
            summary["identity_checks"][column] = {
                "compared": int(compare.sum()),
                "mismatch": int(
                    (compare & eligible[column].astype(str).ne(aligned[column].astype(str))).sum()
                ),
            }
    if "opening_month" in eligible and "opening_date" in aligned:
        expected = pd.to_datetime(eligible["opening_month"], errors="coerce").dt.to_period("M")
        actual = pd.to_datetime(aligned["opening_date"], errors="coerce").dt.to_period("M")
        compare = covered & expected.notna() & actual.notna()
        summary["identity_checks"]["opening_month"] = {
            "compared": int(compare.sum()),
            "mismatch": int((compare & expected.ne(actual)).sum()),
            "missing_or_invalid": int((covered & ~compare).sum()),
        }
    group_columns = [c for c in ("split", "city_key") if c in eligible]
    groups = eligible.assign(_covered=covered).groupby(group_columns, dropna=False)
    for key, group in groups:
        keys = key if isinstance(key, tuple) else (key,)
        record = {c: str(v) for c, v in zip(group_columns, keys, strict=True)}
        record.update({"units": len(group), "covered": int(group["_covered"].sum())})
        record["flag_missing"] = {
            c: int(aligned.loc[group.index, c].isna().sum()) if c in aligned else len(group)
            for c in FLAGS
        }
        summary["coverage_groups"].append(record)
    if "terminal_evidence" in aligned:
        summary["terminal_evidence"] = (
            aligned.loc[covered, "terminal_evidence"].fillna("unknown").value_counts().to_dict()
        )

    review = eligible.join(aligned, rsuffix="_attribute")
    review["attribute_row_present"] = covered
    chosen = []
    for column in FLAGS:
        if column not in aligned:
            continue
        values = pd.to_numeric(aligned[column], errors="coerce")
        for value in (0, 1):
            candidates = review.loc[covered & values.eq(value)]
            chosen.extend(
                candidates.sample(n=min(5, len(candidates)), random_state=7).index.tolist()
            )
    unknown = review.loc[~covered | aligned.reindex(columns=FLAGS).isna().any(axis=1)]
    chosen.extend(unknown.head(10).index.tolist())
    if not chosen:
        chosen = review.head(10).index.tolist()
    examples = review.loc[list(dict.fromkeys(chosen))].reset_index()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "station_attribute_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    examples.to_csv(output_dir / "station_attribute_review.csv", index=False, encoding="utf-8-sig")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit_station_attributes(args.attributes, args.dataset, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
