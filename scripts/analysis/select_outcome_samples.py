#!/usr/bin/env python3
"""Select independent, data-complete presentation samples by outcome family.

The four outcome families are sampled separately from the 22-city Lianjia
study universe.  The output is an exact ``treatment_order x outcome_family``
task list for ``run_parallel_production.py --tasks-file``; it is intentionally
not a common treatment-order sample.

The selector is fail-closed:

* Phase 1 must be terminal for the full frozen treatment queue.
* every candidate must have the frozen pre-treatment support required by the
  corresponding family;
* annual outcomes must have five complete pre years and all three post years;
* monthly VIIRS must have the complete 36-month clean pre path and 24-month
  post path;
* monthly housing uses the same ``main`` price-measure resolution as formal
  production (hedonic when its derived city panel exists, otherwise median).
  Main-sample eligibility is pre-treatment only: at least 12 valid clean-pre
  months and at least one valid month in each 12-month pre block.  Post-period
  coverage is retained only as a horizon-level diagnostic.  The subset with
  every published post window complete is written separately for sensitivity
  analysis.

If a family has fewer selection-eligible tasks than ``--n-per-family``, all
eligible tasks are retained and the shortfall is recorded rather than filling
the sample with tasks that fail that family's treatment-path requirements.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.causal.gpu.panel_builder import (  # noqa: E402
    OUTCOMES,
    monthly_event_calendar,
    read_annual_outcome,
    read_monthly_housing,
    read_monthly_viirs,
)
from urban_intervention.data.paths import (  # noqa: E402
    CONTROL_DESIGN_QUEUE,
    FORMAL_TARGET_SUPPORT,
    OUTPUT_CAUSAL_LABELS_DIR,
    PROJECT_ROOT,
    TREATMENT_PREREND_AVAILABILITY,
    TREATMENT_UNIT_LIST,
)

FAMILIES = ("housing", "poi", "viirs", "population")
TERMINAL_CONTROL_STATUSES = frozenset({"matched", "gsc_pending"})
LIANJIA_CITIES = frozenset(
    {
        "beijing",
        "changzhou",
        "chongqing",
        "dongguan",
        "foshan",
        "guangzhou",
        "hangzhou",
        "jinan",
        "jinhua",
        "luoyang",
        "nanjing",
        "nantong",
        "ningbo",
        "qingdao",
        "shaoxing",
        "shenzhen",
        "suzhou",
        "taizhou",
        "wenzhou",
        "wuxi",
        "xuzhou",
        "zhengzhou",
    }
)
MONTHLY_HORIZONS = (1, 3, 6, 12, 18, 24)
ASSESSMENT_COLUMNS = [
    "treatment_order",
    "target_pre_valid",
    "target_pre_required",
    "target_post_valid",
    "target_post_required",
    "target_path_complete",
    "selection_eligible",
]


def _require_unique(frame: pd.DataFrame, name: str) -> None:
    if "treatment_order" not in frame.columns:
        raise ValueError(f"{name} lacks treatment_order")
    if frame["treatment_order"].duplicated().any():
        raise ValueError(f"{name} contains duplicate treatment_order values")


def validate_phase1(control: pd.DataFrame) -> None:
    """Require a complete Phase-1 queue before outcome sampling."""
    _require_unique(control, "control-design queue")
    if "status" not in control.columns:
        raise ValueError("control-design queue lacks status")
    status = control["status"].astype(str)
    nonterminal = ~status.isin(TERMINAL_CONTROL_STATUSES)
    if nonterminal.any():
        counts = status.loc[nonterminal].value_counts().sort_index().to_dict()
        raise ValueError(
            "Phase 1 is incomplete; outcome sampling would be status-selected: "
            f"{counts}"
        )


def build_candidate_frame(
    treatments: pd.DataFrame,
    support: pd.DataFrame,
    pretrend: pd.DataFrame,
    control: pd.DataFrame,
) -> pd.DataFrame:
    """Join the immutable treatment, support and terminal control records."""
    for name, frame in (
        ("treatments", treatments),
        ("formal target support", support),
        ("pretrend availability", pretrend),
        ("control-design queue", control),
    ):
        _require_unique(frame, name)
    validate_phase1(control)

    treatment_columns = [
        "treatment_order",
        "city_key",
        "grid_id",
        "station_event_id",
        "opening_month",
    ]
    missing = set(treatment_columns) - set(treatments.columns)
    if missing:
        raise ValueError(f"treatments lack columns: {sorted(missing)}")
    support_columns = [
        column
        for column in support.columns
        if column == "treatment_order"
        or column.endswith("_complete")
        or column.endswith("_gsc_ready")
    ]
    pretrend_columns = [
        column
        for column in pretrend.columns
        if column == "treatment_order" or column.endswith("_pre_periods")
    ]
    control_columns = [
        column
        for column in (
            "treatment_order",
            "status",
            "control_city_key",
            "control_grid_id",
        )
        if column in control.columns
    ]
    frame = treatments[treatment_columns].merge(
        support[support_columns], on="treatment_order", how="left", validate="one_to_one"
    )
    frame = frame.merge(
        pretrend[pretrend_columns],
        on="treatment_order",
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(
        control[control_columns],
        on="treatment_order",
        how="left",
        validate="one_to_one",
    )
    frame["treatment_order"] = pd.to_numeric(
        frame["treatment_order"], errors="raise"
    ).astype(int)
    frame["city_key"] = frame["city_key"].astype(str)
    frame["grid_id"] = frame["grid_id"].astype(str)
    frame["opening_year"] = frame["opening_month"].astype(str).str[:4].astype(int)
    frame["stratum"] = frame["city_key"] + "::" + frame["opening_year"].astype(str)
    frame["in_lianjia_22"] = frame["city_key"].isin(LIANJIA_CITIES)

    for family in FAMILIES:
        complete_column = f"{family}_complete"
        if complete_column not in frame.columns:
            raise ValueError(f"formal target support lacks {complete_column}")
        complete = frame[complete_column].fillna(False).astype(bool)
        periods_column = f"{family}_pre_periods"
        if periods_column not in frame.columns:
            raise ValueError(f"pretrend availability lacks {periods_column}")
        pre_ready = pd.to_numeric(
            frame[periods_column], errors="coerce"
        ).ge(5)
        frame[f"{family}_pre_ready"] = pre_ready
        ready_column = f"{family}_gsc_ready"
        if ready_column in frame.columns:
            ready = frame[ready_column].fillna(False).astype(bool)
        else:
            ready = complete & pre_ready
        frame[f"{family}_base_ready"] = complete & ready
    return frame


def _finite_key_set(frame: pd.DataFrame, outcomes: tuple[str, ...]) -> set[tuple[str, object]]:
    values = frame.loc[:, list(outcomes)].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values).all(axis=1)
    return set(
        zip(
            frame.loc[valid, "grid_id"].astype(str),
            frame.loc[valid, "period"],
            strict=False,
        )
    )


def assess_annual_targets(
    candidates: pd.DataFrame,
    family: str,
    *,
    root: Path,
) -> pd.DataFrame:
    """Check the exact five-pre/three-post annual target path."""
    if candidates.empty:
        return pd.DataFrame(columns=ASSESSMENT_COLUMNS)
    rows: list[dict[str, object]] = []
    for city, city_targets in candidates.groupby("city_key", sort=True):
        outcomes = read_annual_outcome(root, str(city), family).rename(
            columns={"year": "period"}
        )
        outcomes["period"] = pd.to_numeric(outcomes["period"], errors="raise").astype(int)
        available = _finite_key_set(outcomes, OUTCOMES[family])
        for target in city_targets.itertuples(index=False):
            grid = str(target.grid_id)
            opening_year = int(target.opening_year)
            required_pre = tuple(range(opening_year - 5, opening_year))
            required_post = tuple(range(opening_year + 1, opening_year + 4))
            pre_valid = sum((grid, period) in available for period in required_pre)
            post_valid = sum((grid, period) in available for period in required_post)
            rows.append(
                {
                    "treatment_order": int(target.treatment_order),
                    "target_pre_valid": pre_valid,
                    "target_pre_required": len(required_pre),
                    "target_post_valid": post_valid,
                    "target_post_required": len(required_post),
                    "target_path_complete": pre_valid == 5 and post_valid == 3,
                    "selection_eligible": pre_valid == 5 and post_valid == 3,
                }
            )
    return pd.DataFrame(rows)


def _published_months(post: pd.DatetimeIndex, window: int) -> pd.DatetimeIndex:
    leads = sorted(
        {
            lead
            for horizon in MONTHLY_HORIZONS
            for lead in range(max(1, horizon - window + 1), horizon + 1)
        }
    )
    return pd.DatetimeIndex([post[lead - 1] for lead in leads])


def _horizon_months(
    post: pd.DatetimeIndex, horizon: int, window: int
) -> pd.DatetimeIndex:
    start = max(1, horizon - window + 1)
    return pd.DatetimeIndex([post[lead - 1] for lead in range(start, horizon + 1)])


def resolve_housing_price_measure(root: Path, city: str, requested: str) -> str:
    """Resolve housing measure exactly as the formal ``main`` queue does."""
    if requested not in {"main", "median", "hedonic"}:
        raise ValueError(f"unsupported housing price measure: {requested}")
    if requested != "main":
        return requested
    hedonic_path = (
        root
        / OUTPUT_CAUSAL_LABELS_DIR.relative_to(PROJECT_ROOT)
        / "housing_hedonic"
        / f"{city}_monthly.parquet"
    )
    return "hedonic" if hedonic_path.exists() else "median"


def assess_monthly_targets(
    candidates: pd.DataFrame,
    family: str,
    *,
    root: Path,
    window: int,
    transaction_count_threshold: int,
    price_measure: str = "main",
) -> pd.DataFrame:
    """Check target-side monthly support used by production labels."""
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                *ASSESSMENT_COLUMNS,
                "published_post_valid",
                "published_post_required",
                "price_measure",
                "pre_lag1_valid",
                "pre_lag2_valid",
                "pre_lag3_valid",
                "pre_block_min_valid",
                "supported_horizon_count",
                "supported_horizons",
                *[
                    f"horizon_{horizon}_supported"
                    for horizon in MONTHLY_HORIZONS
                ],
            ]
        )
    rows: list[dict[str, object]] = []
    for city, city_targets in candidates.groupby("city_key", sort=True):
        calendars = {
            int(target.treatment_order): monthly_event_calendar(
                target.opening_month, anticipation_months=6
            )
            for target in city_targets.itertuples(index=False)
        }
        all_months = pd.DatetimeIndex(
            sorted(
                {
                    period
                    for calendar in calendars.values()
                    for period in calendar["times"]
                }
            )
        )
        if family == "housing":
            resolved_price_measure = resolve_housing_price_measure(
                root, str(city), price_measure
            )
            source = read_monthly_housing(root, str(city), resolved_price_measure)
            values = pd.to_numeric(source["housing_log_price"], errors="coerce")
            transactions = pd.to_numeric(source["transaction_count"], errors="coerce")
            valid = np.isfinite(values) & transactions.ge(transaction_count_threshold)
        elif family == "viirs":
            source = read_monthly_viirs(root, str(city), all_months)
            values = pd.to_numeric(source["viirs_avg_asinh"], errors="coerce")
            valid = np.isfinite(values)
        else:
            raise ValueError(f"unsupported monthly family: {family}")
        available = set(
            zip(
                source.loc[valid, "grid_id"].astype(str),
                pd.to_datetime(source.loc[valid, "period"]),
                strict=False,
            )
        )
        for target in city_targets.itertuples(index=False):
            order = int(target.treatment_order)
            calendar = calendars[order]
            grid = str(target.grid_id)
            pre = pd.DatetimeIndex(calendar["pre"])
            post = pd.DatetimeIndex(calendar["post"])
            published = _published_months(post, window)
            pre_valid = sum((grid, period) in available for period in pre)
            post_valid = sum((grid, period) in available for period in post)
            published_valid = sum((grid, period) in available for period in published)
            if family == "housing":
                pre_blocks = {
                    "pre_lag3_valid": sum(
                        (grid, period) in available for period in pre[:12]
                    ),
                    "pre_lag2_valid": sum(
                        (grid, period) in available for period in pre[12:24]
                    ),
                    "pre_lag1_valid": sum(
                        (grid, period) in available for period in pre[24:]
                    ),
                }
                pre_block_min_valid = min(pre_blocks.values())
                selection_eligible = pre_valid >= 12 and pre_block_min_valid >= 1
                horizon_support = {
                    horizon: all(
                        (grid, period) in available
                        for period in _horizon_months(post, horizon, window)
                    )
                    for horizon in MONTHLY_HORIZONS
                }
                supported_horizons = [
                    horizon
                    for horizon, supported in horizon_support.items()
                    if supported
                ]
                complete = selection_eligible and all(horizon_support.values())
            else:
                pre_blocks = {}
                pre_block_min_valid = pd.NA
                selection_eligible = pre_valid == len(pre) and post_valid == len(post)
                horizon_support = {}
                supported_horizons = []
                complete = pre_valid == len(pre) and post_valid == len(post)
            rows.append(
                {
                    "treatment_order": order,
                    "target_pre_valid": pre_valid,
                    "target_pre_required": len(pre),
                    "target_post_valid": post_valid,
                    "target_post_required": len(post),
                    "published_post_valid": published_valid,
                    "published_post_required": len(published),
                    "target_path_complete": complete,
                    "selection_eligible": selection_eligible,
                    "price_measure": (
                        resolved_price_measure if family == "housing" else pd.NA
                    ),
                    **pre_blocks,
                    "pre_block_min_valid": pre_block_min_valid,
                    "supported_horizon_count": (
                        len(supported_horizons) if family == "housing" else pd.NA
                    ),
                    "supported_horizons": (
                        ";".join(map(str, supported_horizons))
                        if family == "housing"
                        else pd.NA
                    ),
                    **{
                        f"horizon_{horizon}_supported": supported
                        for horizon, supported in horizon_support.items()
                    },
                }
            )
    return pd.DataFrame(rows)


def allocate_quotas(layer_sizes: pd.Series, target: int) -> pd.Series:
    """Allocate a fixed sample across strata by largest remainder.

    When the target is at least the number of non-empty strata, every stratum
    receives a floor of one before the remaining slots are apportioned.
    """
    if target < 1 or target > int(layer_sizes.sum()):
        raise ValueError("sample target must be within the eligible population")
    if target == int(layer_sizes.sum()):
        return layer_sizes.astype(int).copy()
    use_floor = target >= len(layer_sizes)
    quota = pd.Series(1 if use_floor else 0, index=layer_sizes.index, dtype=int)
    capacity = layer_sizes - quota
    remaining_target = target - int(quota.sum())
    if remaining_target and int(capacity.sum()):
        raw = capacity.astype(float) * remaining_target / int(capacity.sum())
        addition = np.floor(raw).astype(int)
        quota += addition
        remaining = target - int(quota.sum())
        remainders = raw - addition
    else:
        remaining = 0
        remainders = pd.Series(0.0, index=layer_sizes.index)
    if remaining:
        order = remainders.sort_values(ascending=False, kind="stable").index
        for stratum in order:
            if remaining == 0:
                break
            if quota.loc[stratum] < layer_sizes.loc[stratum]:
                quota.loc[stratum] += 1
                remaining -= 1
    if int(quota.sum()) != target or (quota > layer_sizes).any():
        raise ValueError("failed to allocate stratified sample quotas")
    return quota


def select_family_sample(
    eligible: pd.DataFrame,
    family: str,
    *,
    target: int,
    seed: int,
) -> pd.DataFrame:
    """Select up to ``target`` independently for one family."""
    if eligible.empty:
        empty = eligible.copy()
        empty["outcome_family"] = pd.Series(dtype="object")
        empty["stratum_n"] = pd.Series(dtype="int64")
        empty["stratum_quota"] = pd.Series(dtype="int64")
        empty["sampling_seed"] = pd.Series(dtype="int64")
        return empty
    sample_n = min(target, len(eligible))
    sizes = eligible.groupby("stratum", sort=True).size()
    quotas = allocate_quotas(sizes, sample_n)
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for stratum, quota in quotas.items():
        if quota == 0:
            continue
        layer = eligible.loc[eligible["stratum"].eq(stratum)].copy()
        layer["_tie"] = rng.random(len(layer))
        if family == "housing":
            # Housing selection is deliberately pre-treatment only.  Using
            # post-period transaction availability here would condition the
            # presentation sample on a quantity that treatment may affect.
            sort_columns = ["pre_block_min_valid", "target_pre_valid", "_tie"]
        else:
            sort_columns = ["target_pre_valid", "target_post_valid", "_tie"]
        layer = layer.sort_values(
            sort_columns,
            ascending=[False, False, True],
            kind="stable",
        ).head(int(quota))
        layer["stratum_n"] = int(sizes.loc[stratum])
        layer["stratum_quota"] = int(quota)
        parts.append(layer.drop(columns="_tie"))
    sample = pd.concat(parts, ignore_index=True)
    if len(sample) != sample_n or sample["treatment_order"].duplicated().any():
        raise ValueError(f"invalid {family} sample cardinality or duplicate order")
    sample["outcome_family"] = family
    sample["sampling_seed"] = seed
    return sample.sort_values("treatment_order", kind="stable").reset_index(drop=True)


def select_all_families(
    candidates: pd.DataFrame,
    *,
    root: Path,
    n_per_family: int,
    seed: int,
    window: int,
    transaction_count_threshold: int,
    price_measure: str = "main",
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    samples: list[pd.DataFrame] = []
    complete_case_samples: list[pd.DataFrame] = []
    diagnostics: dict[str, object] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "n_per_family_requested": n_per_family,
        "seed": seed,
        "housing_price_measure_requested": price_measure,
        "city_universe": sorted(LIANJIA_CITIES),
        "families": {},
    }
    for family_index, family in enumerate(FAMILIES):
        base_ready_column = (
            "housing_pre_ready" if family == "housing" else f"{family}_base_ready"
        )
        base = candidates.loc[
            candidates["in_lianjia_22"] & candidates[base_ready_column]
        ].copy()
        assessment = (
            assess_monthly_targets(
                base,
                family,
                root=root,
                window=window,
                transaction_count_threshold=transaction_count_threshold,
                price_measure=price_measure,
            )
            if family in {"housing", "viirs"}
            else assess_annual_targets(base, family, root=root)
        )
        audited = base.merge(
            assessment, on="treatment_order", how="inner", validate="one_to_one"
        )
        eligible = audited.loc[audited["selection_eligible"].fillna(False)].copy()
        strict_complete = audited.loc[
            audited["target_path_complete"].fillna(False)
        ].copy()
        family_seed = seed + family_index
        sample = select_family_sample(
            eligible,
            family,
            target=n_per_family,
            seed=family_seed,
        )
        samples.append(sample)
        if family == "housing":
            strict_complete["outcome_family"] = family
            strict_complete["sampling_seed"] = family_seed
            complete_case_samples.append(strict_complete)
        diagnostics["families"][family] = {
            "base_ready_in_22_cities": int(len(base)),
            "selection_eligible": int(len(eligible)),
            "strict_target_path_complete": int(len(strict_complete)),
            "sampled": int(len(sample)),
            "shortfall_from_requested": int(n_per_family - len(sample)),
            "cities_sampled": int(sample["city_key"].nunique()) if len(sample) else 0,
            "sampling_seed": family_seed,
            "candidate_pool_rule": (
                "pre_treatment_only"
                if family == "housing"
                else "formal_complete_path"
            ),
        }
        if family == "housing":
            diagnostics["families"][family][
                "formal_complete_base_ready_in_22_cities"
            ] = int(
                (
                    candidates["in_lianjia_22"]
                    & candidates["housing_base_ready"]
                ).sum()
            )
            sampled_measures = sample.get(
                "price_measure", pd.Series(dtype="object")
            ).dropna()
            diagnostics["families"][family]["price_measures_sampled"] = {
                str(measure): int(count)
                for measure, count in sampled_measures.value_counts().items()
            }
            diagnostics["families"][family][
                "post_information_used_for_sampling"
            ] = False
            diagnostics["families"][family]["supported_horizon_count_sampled"] = {
                str(int(count)): int(frequency)
                for count, frequency in sample["supported_horizon_count"]
                .value_counts()
                .sort_index()
                .items()
            }
            diagnostics["families"][family]["supported_tasks_by_horizon"] = {
                str(horizon): int(
                    sample[f"horizon_{horizon}_supported"].fillna(False).sum()
                )
                for horizon in MONTHLY_HORIZONS
            }
    tasks = pd.concat(samples, ignore_index=True)
    if tasks.duplicated(["treatment_order", "outcome_family"]).any():
        raise ValueError("combined sample contains duplicate treatment-family keys")
    diagnostics["total_tasks"] = int(len(tasks))
    diagnostics["unique_treatments"] = int(tasks["treatment_order"].nunique())
    complete_cases = (
        pd.concat(complete_case_samples, ignore_index=True)
        if complete_case_samples
        else pd.DataFrame()
    )
    diagnostics["housing_complete_case_sensitivity_tasks"] = int(len(complete_cases))
    return tasks, diagnostics, complete_cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-family", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--transaction-count-threshold", type=int, default=1)
    parser.add_argument(
        "--price-measure",
        choices=("main", "median", "hedonic"),
        default="main",
        help="Housing measure; main matches formal production city by city",
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    if args.n_per_family < 1:
        parser.error("--n-per-family must be positive")
    if not 1 <= args.window <= 6:
        parser.error("--window must be in 1..6")
    if args.transaction_count_threshold < 1:
        parser.error("--transaction-count-threshold must be positive")

    root = args.root.resolve()
    relative = lambda path: root / path.relative_to(PROJECT_ROOT)  # noqa: E731
    candidates = build_candidate_frame(
        pd.read_parquet(relative(TREATMENT_UNIT_LIST)),
        pd.read_parquet(relative(FORMAL_TARGET_SUPPORT)),
        pd.read_parquet(relative(TREATMENT_PREREND_AVAILABILITY)),
        pd.read_csv(relative(CONTROL_DESIGN_QUEUE)),
    )
    tasks, diagnostics, complete_cases = select_all_families(
        candidates,
        root=root,
        n_per_family=args.n_per_family,
        seed=args.seed,
        window=args.window,
        transaction_count_threshold=args.transaction_count_threshold,
        price_measure=args.price_measure,
    )
    output = args.output_directory or relative(OUTPUT_CAUSAL_LABELS_DIR)
    output.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "treatment_order",
        "outcome_family",
        "city_key",
        "grid_id",
        "station_event_id",
        "opening_month",
        "opening_year",
        "status",
        "control_city_key",
        "control_grid_id",
        "target_pre_valid",
        "target_pre_required",
        "target_post_valid",
        "target_post_required",
        "target_path_complete",
        "stratum",
        "stratum_n",
        "stratum_quota",
        "sampling_seed",
    ]
    optional = [
        "published_post_valid",
        "published_post_required",
        "price_measure",
        "selection_eligible",
        "pre_lag1_valid",
        "pre_lag2_valid",
        "pre_lag3_valid",
        "pre_block_min_valid",
        "supported_horizon_count",
        "supported_horizons",
        *[
            f"horizon_{horizon}_supported"
            for horizon in MONTHLY_HORIZONS
        ],
    ]
    output_columns.extend(column for column in optional if column in tasks.columns)
    tasks[output_columns].to_csv(
        output / "presentation_sample_tasks.csv", index=False, encoding="utf-8-sig"
    )
    for family in FAMILIES:
        tasks.loc[tasks["outcome_family"].eq(family), output_columns].to_csv(
            output / f"presentation_sample_{family}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    complete_case_columns = [
        column
        for column in output_columns
        if column in complete_cases.columns
        and column not in {"stratum_n", "stratum_quota"}
    ]
    complete_cases[complete_case_columns].to_csv(
        output / "presentation_sample_housing_complete_case.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output / "presentation_sample_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"Wrote {len(tasks)} exact treatment-family tasks to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
