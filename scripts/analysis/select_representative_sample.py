"""Stratified representative sample of treated grids for the sample run.

Strata are city_key x opening_year.  Quotas use Hamilton's largest-remainder
method on 400 units with a floor of 1 per non-empty stratum; any shortfall
after the floor is reallocated to the largest strata, so every non-empty
stratum is represented while large strata keep depth.

Outputs (all under outputs/causal_labels/):

- representative_sample_400.csv: sampled treatment orders with stratum info
- representative_sample_400_diagnostics.json: coverage and composition report

The production sample defaults to the opening-month interval 2017-07 through
2022-12. With the current processed VIIRS cache (2014-01 through 2024-12),
this interval provides the full 42-month pre-treatment and 24-month
post-treatment window used by the monthly GSC specification.

Usage:
    python scripts/analysis/select_representative_sample.py [--n 400] [--seed 20260814]
    python scripts/analysis/select_representative_sample.py [--n 400] [--seed 20260814] [--min-opening-month YYYY-MM] [--max-opening-month YYYY-MM]
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
    if len(sample) != n or sample["treatment_order"].duplicated().any():
        raise ValueError(
            f"Representative sample cardinality/uniqueness failed: n={len(sample)}, target={n}"
        )

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
    parser.add_argument(
        "--min-opening-month",
        default="2017-07",
        help="Inclusive lower opening-month bound (default: 2017-07).",
    )
    parser.add_argument(
        "--max-opening-month",
        default="2022-12",
        help="Inclusive upper opening-month bound (default: 2022-12).",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    treatments = pd.read_parquet(TREATMENTS_PATH)
    if len(treatments) != 5_048 or treatments["treatment_order"].duplicated().any():
        raise ValueError("Treatment list is not the immutable 5,048-unit list")
    eligible = treatments.copy()
    cutoff = None
    ceiling = None
    if args.min_opening_month:
        try:
            cutoff = pd.Period(args.min_opening_month, freq="M")
        except Exception as exc:
            raise ValueError("--min-opening-month must use YYYY-MM format") from exc
    if args.max_opening_month:
        try:
            ceiling = pd.Period(args.max_opening_month, freq="M")
        except Exception as exc:
            raise ValueError("--max-opening-month must use YYYY-MM format") from exc
    if cutoff is not None and ceiling is not None and cutoff > ceiling:
        raise ValueError("--min-opening-month must not be later than --max-opening-month")
    if cutoff is not None or ceiling is not None:
        opening_period = pd.to_datetime(
            eligible["opening_month"].astype(str) + "-01", errors="raise"
        ).dt.to_period("M")
        keep = pd.Series(True, index=eligible.index)
        if cutoff is not None:
            keep &= opening_period >= cutoff
        if ceiling is not None:
            keep &= opening_period <= ceiling
        eligible = eligible.loc[keep].copy()
        if len(eligible) < args.n:
            raise ValueError(
                f"Eligible population {len(eligible)} is smaller than target sample {args.n}"
            )

    sample, diagnostics = select_sample(eligible, args.n, args.seed)
    diagnostics["source_n"] = int(len(treatments))
    diagnostics["eligible_n"] = int(len(eligible))
    diagnostics["excluded_n"] = int(len(treatments) - len(eligible))
    diagnostics["min_opening_month_filter"] = str(cutoff) if cutoff is not None else None
    diagnostics["max_opening_month_filter"] = str(ceiling) if ceiling is not None else None

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
    print(
        f"Sampled {len(sample)} grids from {diagnostics['eligible_n']} eligible units "
        f"({diagnostics['excluded_n']} excluded before the opening-month cutoff)"
    )
    print(json.dumps(
        {k: v for k, v in diagnostics.items() if k not in {"created_utc", "by_year", "by_city"}},
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
