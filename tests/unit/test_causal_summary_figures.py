from __future__ import annotations

import pandas as pd
from summarize_causal_labels import (
    build_distribution_table,
    build_generation_tables,
    primary_label_rows,
    render_summary_figures,
    scope_of,
)


def _artifact() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    outcomes = {
        "housing": "housing_log_price",
        "viirs": "viirs_avg_asinh",
        "poi": "poi_count_log",
        "population": "population_log",
    }
    for family, outcome in outcomes.items():
        for order in (1, 2):
            for event_time in (1, 2):
                available = not (family == "viirs" and order == 2)
                if family == "poi" and order == 2 and event_time == 2:
                    available = False
                rows.append(
                    {
                        "treatment_order": order,
                        "outcome_family": family,
                        "outcome": outcome,
                        "event_time": event_time,
                        "label_available": available,
                        "production_eligible": True,
                        "causal_response_label": order * 0.1 + event_time * 0.01,
                        "failure_reason": None if available else "missing_target_period",
                        "task_failure_reason": None,
                    }
                )
    rows.append(
        {
            "treatment_order": 1,
            "outcome_family": "poi",
            "outcome": "poi_category_entropy",
            "event_time": 1,
            "label_available": True,
            "production_eligible": True,
            "causal_response_label": 999.0,
            "failure_reason": None,
            "task_failure_reason": None,
        }
    )
    return pd.DataFrame(rows)


def test_generation_counts_tasks_once_and_separates_partial_paths() -> None:
    primary = primary_label_rows(_artifact())
    tasks, generation, failures = build_generation_tables(primary)

    assert len(tasks) == 8
    assert tasks.loc[
        tasks["outcome_family"].eq("viirs") & tasks["treatment_order"].eq(2),
        "generation_state",
    ].iloc[0] == "failed"
    assert tasks.loc[
        tasks["outcome_family"].eq("poi") & tasks["treatment_order"].eq(2),
        "generation_state",
    ].iloc[0] == "partial"
    assert generation["treated_grids"].sum() == 8
    assert failures["treated_grids"].sum() == 2


def test_distribution_uses_only_four_primary_outcome_scales(tmp_path) -> None:
    artifact = _artifact()
    primary = primary_label_rows(artifact)
    tasks, generation, failures = build_generation_tables(primary)
    distribution = build_distribution_table(primary)

    poi = distribution.loc[distribution["outcome_family"].eq("poi")].iloc[0]
    assert poi["max"] < 1.0
    written = render_summary_figures(
        artifact,
        generation,
        failures,
        distribution,
        tmp_path,
    )
    assert len(tasks) == 8
    assert len(written) == 12
    assert all(pd.notna(path) for path in written)


def test_scope_of_tolerates_nullable_string_values() -> None:
    assert scope_of(pd.Series({"donor_scope": pd.NA, "method": pd.NA})) == "unknown"
    assert (
        scope_of(
            pd.Series(
                {
                    "donor_scope": pd.NA,
                    "method": "xu_2017_gsynth_all_city_standardized",
                }
            )
        )
        == "cross_city"
    )
