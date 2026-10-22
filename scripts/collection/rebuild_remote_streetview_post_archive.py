"""Verify annual post-opening street views from the existing frozen-grid archive."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

try:
    from .rebuild_remote_streetview_index import select_verified_candidates, sha256
except ImportError:
    from rebuild_remote_streetview_index import select_verified_candidates, sha256


IDENTITY = ["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"]


def post_candidates(events: pd.DataFrame, index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matched = events[IDENTITY].merge(index, on=["city_key", "grid_id"], how="inner")
    matched["target_year"] = matched.capture_month.str[:4].astype(int)
    matched["event_time"] = matched.target_year - matched.opening_month.str[:4].astype(int)
    matched = matched.loc[matched.event_time.between(0, 3)].copy()
    matched["state_allowed"] = False
    matched["role"] = "post_intervention_archive_only"
    eligible = matched.loc[matched.capture_month > matched.opening_month.str[:7]].sort_values(
        ["treatment_order", "event_time", "capture_month", "panoid"], ascending=[True, True, False, True])
    return eligible, matched


def post_coverage(events: pd.DataFrame, selected: pd.DataFrame, eligible: pd.DataFrame,
                  all_year: pd.DataFrame, latest_complete_year: int = 2025) -> pd.DataFrame:
    coverage = events[IDENTITY].merge(pd.DataFrame({"event_time": [0, 1, 2, 3]}), how="cross")
    coverage["target_year"] = coverage.opening_month.str[:4].astype(int) + coverage.event_time
    coverage["state_allowed"] = False
    coverage["role"] = "post_intervention_archive_only"
    keys = ["treatment_order", "event_time"]
    for frame, name in [(selected, "selected_image_count"), (eligible, "postopening_candidate_count"),
                        (all_year, "year_metadata_count")]:
        counts = frame.groupby(keys).size().rename(name).reset_index()
        coverage = coverage.merge(counts, on=keys, how="left", validate="one_to_one")
        coverage[name] = coverage[name].fillna(0).astype(int)
    uncertain = all_year.loc[all_year.capture_month.eq(all_year.opening_month.str[:7])]
    counts = uncertain.groupby(keys).size().rename("opening_month_ambiguous_candidate_count").reset_index()
    coverage = coverage.merge(counts, on=keys, how="left", validate="one_to_one")
    coverage["opening_month_ambiguous_candidate_count"] = coverage.opening_month_ambiguous_candidate_count.fillna(0).astype(int)
    coverage["post_streetview_available"] = coverage.selected_image_count.gt(0)
    coverage["post_streetview_status"] = [
        "postopening_verified" if r.post_streetview_available else
        "incomplete_calendar_year" if r.target_year > latest_complete_year else
        "postopening_candidates_but_no_verified_image" if r.postopening_candidate_count else
        "opening_month_temporal_relation_unknown" if r.opening_month_ambiguous_candidate_count else
        "only_preopening_captures_in_opening_year" if r.year_metadata_count else
        "no_pano_metadata_in_target_year" for r in coverage.itertuples()]
    return coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-images", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.snapshot / "audit.json").read_text())
    contract = json.loads(args.contract.read_text())
    if not audit["run_finished"] or audit["cities"] != 44 or audit["events"] != 5048:
        raise ValueError("Completed 44-city source index required")
    if sha256(args.contract) != audit["contract_sha256"]:
        raise ValueError("Frozen contract mismatch")
    source = args.snapshot / "streetview_index.csv"
    if sha256(source) != audit["file_sha256"][source.name]:
        raise ValueError("Source metadata index changed")
    index = pd.read_csv(source, dtype={"panoid": str, "capture_month": str})
    events = pd.DataFrame(contract["events"])
    eligible, all_year = post_candidates(events, index)
    print(json.dumps({"post_candidates": len(eligible),
                      "annual_candidate_groups": eligible.groupby(["treatment_order", "event_time"]).ngroups}), flush=True)
    selected, quality = select_verified_candidates(eligible, contract, Path(audit["raw_root"]),
        args.max_images, args.workers, ["treatment_order", "event_time"])
    coverage = post_coverage(events, selected, eligible, all_year)
    checks = {"all_event_years_preserved": len(coverage) == 5048 * 4,
              "strictly_after_opening_month": bool((selected.capture_month > selected.opening_month.str[:7]).all()),
              "event_time_zero_to_three": bool(selected.event_time.between(0, 3).all()),
              "all_post_images_decoded": bool(selected.image_valid.all()),
              "post_images_never_state_inputs": bool((~selected.state_allowed).all() and (~coverage.state_allowed).all()),
              "source_index_row_count_matches": len(index) == audit["index_rows"]}
    if not all(checks.values()):
        raise ValueError(f"Post-opening audit failed: {checks}")
    selected.to_csv(args.output / "streetview_post_selected.csv", index=False)
    quality.to_csv(args.output / "post_image_quality.csv", index=False)
    coverage.to_csv(args.output / "post_event_coverage.csv", index=False)
    summary = {"schema": "streetview_postopening_annual_archive_v1", "run_finished": True,
               "created_at_utc": datetime.now(UTC).isoformat(), "events": len(events), "cities": 44,
               "event_times": [0, 1, 2, 3], "expected_event_years": len(coverage),
               "available_event_years": int(coverage.post_streetview_available.sum()),
               "available_events": int(selected.treatment_order.nunique()), "selected_rows": len(selected),
               "selected_unique_images": int(selected.asset_path.nunique()),
               "selected_unique_bytes": int(selected.drop_duplicates("asset_path").asset_bytes.sum()),
               "by_event_time": coverage.groupby("event_time").post_streetview_available.sum().to_dict(),
               "status_counts": coverage.post_streetview_status.value_counts().to_dict(),
               "raw_root": audit["raw_root"], "treatment_sha256": audit["treatment_sha256"],
               "contract_sha256": audit["contract_sha256"], "source_index_sha256": audit["file_sha256"][source.name],
               "capture_date_precision": "month", "temporal_rule": "capture_month > opening_month",
               "spatial_support": audit["spatial_support"], "max_images_per_event_year": args.max_images,
               "state_allowed": False, "checks": checks,
               "file_sha256": {name: sha256(args.output / name) for name in
                   ["streetview_post_selected.csv", "post_image_quality.csv", "post_event_coverage.csv"]}}
    (args.output / "post_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ["checks", "file_sha256"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
