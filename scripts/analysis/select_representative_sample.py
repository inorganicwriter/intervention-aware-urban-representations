"""Stratified representative sample of treated grids for the sample run.

Strata are city_key x opening_year.  Quotas use Hamilton's largest-remainder
method on 400 units with a floor of 1 per non-empty stratum; any shortfall
after the floor is reallocated to the largest strata, so every non-empty
stratum is represented while large strata keep depth.

Outputs (all under outputs/causal_labels/):

- representative_sample_400.csv: sampled treatment orders with stratum info
- representative_sample_400_diagnostics.json: coverage and composition report

Usage:
    python scripts/analysis/select_representative_sample.py [--n 400] [--seed mit-summer-2026]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TREATMENTS_PATH = ROOT / "data" / "active" / "causal" / "treatment_unit_list.parquet"
OUTPUT_DIR = ROOT / "outputs" / "causal_labels"


def allocate_quotas(layer_sizes: pd.Series, target: int) -> pd.Series:
    """Hamilton largest-remainder quotas with a floor of 1 per non-empty layer.

    After the floor is applied the total may exceed the target; the excess is
    then trimmed from the largest quotas (down to the floor).  Every non-empty
    stratum ends with quota >= 1 and the sum is exactly ``target``.
    """
    total = int(layer_sizes.sum())
    if total < target:
        raise ValueError(f"Population {total} smaller than target sample {target}")
    raw = layer_sizes * target / total
    quotas = raw.astype(int)
    remainder = raw - quotas
    to_fill = target - int(quotas.sum())
    if to_fill > 0:
        order = remainder.sort_values(ascending=False).index[:to_fill]
        quotas.loc[order] += 1
    floor = quotas.eq(0) & layer_sizes.gt(0)
    if floor.any():
        quotas.loc[floor] = 1
    excess = int(quotas.sum()) - target
    if excess > 0:
        candidates = quotas.loc[quotas.gt(1)].sort_values(ascending=False).index
        for layer in list(candidates):
            if excess <= 0:
                break
            quotas[layer] -= 1
            excess -= 1
    shortfall = target - int(quotas.sum())
    if shortfall > 0:
        candidates = quotas.loc[quotas.lt(layer_sizes)].sort_values(ascending=False).index
        for layer in list(candidates):
            if shortfall <= 0:
                break
            quotas[layer] += 1
            shortfall -= 1
    return quotas


def select_sample(
    treatments: pd.DataFrame, n: int, seed: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = treatments.copy()
    frame["opening_year"] = (
        pd.to_datetime(frame["opening_month"].astype(str) + "-01").dt.year.astype(int)
    )
    frame["stratum"] = frame["city_key"] + "::" + frame["opening_year"].astype(str)

    layer_sizes = frame.groupby("stratum").size()
    quotas = allocate_quotas(layer_sizes, n)
    rng = np.random.RandomState(seed)

    sampled_rows: list[pd.DataFrame] = []
    for stratum, quota in quotas.items():
        layer = frame.loc[frame["stratum"].eq(stratum)]
        chosen = layer.sample(n=int(min(quota, len(layer))), random_state=rng)
        chosen = chosen.copy()
        chosen["stratum_quota"] = int(quota)
        chosen["stratum_n"] = int(len(layer))
        sampled_rows.append(chosen)
    sample = pd.concat(sampled_rows, ignore_index=True)
    sample = sample.sort_values("treatment_order").reset_index(drop=True)

    coverage = pd.DataFrame(
        {
            "stratum": list(layer_sizes.index),
            "population_n": layer_sizes.values,
            "quota": quotas.values,
        }
    )
    diagnostics: dict[str, object] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "target_n": n,
        "seed": seed,
        "sampled_n": int(len(sample)),
        "non_empty_strata": int((layer_sizes > 0).sum()),
        "covered_strata": int((sample.groupby("stratum").size() > 0).sum()),
        "strata_with_floor_only": int(((coverage["quota"] == 1) & (coverage["population_n"] > 1)).sum()),
        "cities": int(sample["city_key"].nunique()),
        "opening_years": sorted(int(v) for v in sample["opening_year"].unique()),
        "by_year": sample["opening_year"].value_counts().sort_index().astype(int).to_dict(),
        "by_city": sample["city_key"].value_counts().sort_index().astype(int).to_dict(),
        "quota_correlation_with_population": round(
            float(np.corrcoef(coverage["population_n"], coverage["quota"])[0, 1]), 4
        ),
    }
    return sample, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    treatments = pd.read_parquet(TREATMENTS_PATH)
    if len(treatments) != 5_048 or treatments["treatment_order"].duplicated().any():
        raise ValueError("Treatment list is not the immutable 5,048-unit list")
    sample, diagnostics = select_sample(treatments, args.n, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_columns = [
        "treatment_order",
        "city_key",
        "grid_id",
        "station_event_id",
        "opening_month",
        "opening_year",
        "stratum",
        "stratum_quota",
        "stratum_n",
    ]
    sample[out_columns].to_csv(
        args.output_dir / "representative_sample_400.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "representative_sample_400_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Sampled {len(sample)} grids from {diagnostics['non_empty_strata']} non-empty strata")
    print(json.dumps(
        {k: v for k, v in diagnostics.items() if k not in {"created_utc", "by_year", "by_city"}},
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
