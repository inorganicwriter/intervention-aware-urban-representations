"""Tests for causal queue recovery from QC snapshots and the release."""

from __future__ import annotations

import pandas as pd

from scripts.analysis.recover_causal_queues import apply_demotions, build_control_queue


def test_apply_demotions_marks_skipped() -> None:
    before = pd.DataFrame(
        {
            "treatment_order": [1, 2, 2],
            "outcome_family": ["poi", "poi", "housing"],
            "status": ["gsc_labelled", "mc_labelled", "mc_labelled"],
            "selected_method": ["gsc", "mc", "mc"],
            "failure_reason": [None, None, None],
        }
    )
    demotions = pd.DataFrame(
        {
            "treatment_order": [2],
            "outcome_family": ["poi"],
            "reason": ["mc_insufficient_pre_observation"],
        }
    )
    queue, count = apply_demotions(before, demotions)
    assert count == 1
    demoted = queue.loc[
        queue["treatment_order"].eq(2) & queue["outcome_family"].eq("poi")
    ].iloc[0]
    assert demoted["status"] == "skipped"
    assert pd.isna(demoted["selected_method"])
    assert demoted["failure_reason"] == "mc_insufficient_pre_observation"
    untouched = queue.loc[queue["treatment_order"].eq(2) & queue["outcome_family"].eq("housing")].iloc[0]
    assert untouched["status"] == "mc_labelled"
    assert untouched["selected_method"] == "mc"
    assert queue.loc[queue["treatment_order"].eq(1)].iloc[0]["status"] == "gsc_labelled"


def test_build_control_queue_filters_scope_and_dedups() -> None:
    artifact = pd.DataFrame(
        {
            "treatment_order": [1, 1, 2, 3, 4],
            "method": [
                "frozen_matched_change_x",
                "frozen_matched_change_x",
                "athey_2021_mc",
                "frozen_matched_change_x",
                "frozen_matched_change_x",
            ],
            "donor_scope": [
                "same_city",
                "same_city",
                "same_city",
                "all_city_standardized",
                "same_city",
            ],
            "control_unit_key": ["shanghai::g1", "shanghai::g1", "beijing::g9", "wuhan::g2", None],
        }
    )
    control = build_control_queue(artifact)
    assert control["treatment_order"].tolist() == [1]
    row = control.iloc[0]
    assert row["status"] == "matched"
    assert row["control_unit_key"] == "shanghai::g1"
    assert row["control_city_key"] == "shanghai"
    assert row["control_grid_id"] == "g1"
    assert row["donor_scope"] == "same_city"
