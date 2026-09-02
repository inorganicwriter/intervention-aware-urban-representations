from __future__ import annotations

import pandas as pd
import pytest

from scripts.analysis import select_outcome_samples as selector


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    treatments = pd.DataFrame(
        {
            "treatment_order": [1, 2, 3, 4],
            "city_key": ["beijing", "beijing", "hangzhou", "hangzhou"],
            "grid_id": ["g1", "g2", "g3", "g4"],
            "station_event_id": ["s1", "s2", "s3", "s4"],
            "opening_month": ["2020-01", "2020-01", "2021-01", "2021-01"],
        }
    )
    support = pd.DataFrame({"treatment_order": [1, 2, 3, 4]})
    pretrend = pd.DataFrame({"treatment_order": [1, 2, 3, 4]})
    for family in selector.FAMILIES:
        support[f"{family}_complete"] = True
        pretrend[f"{family}_pre_periods"] = 5
    control = treatments[["treatment_order"]].assign(
        status="matched",
        control_city_key="beijing",
        control_grid_id="c1",
    )
    return treatments, support, pretrend, control


def test_candidate_builder_rejects_incomplete_phase1() -> None:
    treatments, support, pretrend, control = _inputs()
    control.loc[0, "status"] = "running"
    with pytest.raises(ValueError, match="Phase 1 is incomplete"):
        selector.build_candidate_frame(treatments, support, pretrend, control)


def test_quota_floor_covers_each_stratum_when_target_allows() -> None:
    sizes = pd.Series({"large": 100, "small_a": 1, "small_b": 1})
    quota = selector.allocate_quotas(sizes, 10)
    assert int(quota.sum()) == 10
    assert (quota >= 1).all()


def test_family_samples_are_independent_and_allow_shortfall(monkeypatch) -> None:
    treatments, support, pretrend, control = _inputs()
    # Housing has only one pre-supported target; the other families have
    # distinct complete pools.  No cross-family intersection is applied.
    support.loc[support["treatment_order"].ne(1), "housing_complete"] = False
    pretrend.loc[pretrend["treatment_order"].ne(1), "housing_pre_periods"] = 4
    support.loc[support["treatment_order"].isin([1, 2]), "poi_complete"] = False
    support.loc[support["treatment_order"].isin([3, 4]), "viirs_complete"] = False
    frame = selector.build_candidate_frame(treatments, support, pretrend, control)

    def complete_assessment(candidates, family, **_kwargs):
        return pd.DataFrame(
            {
                "treatment_order": candidates["treatment_order"].astype(int),
                "target_pre_valid": 5,
                "target_pre_required": 5,
                "target_post_valid": 3,
                "target_post_required": 3,
                "target_path_complete": True,
                "selection_eligible": True,
                "pre_block_min_valid": 1,
                "supported_horizon_count": 6,
                **{
                    f"horizon_{horizon}_supported": True
                    for horizon in selector.MONTHLY_HORIZONS
                },
            }
        )

    monkeypatch.setattr(selector, "assess_annual_targets", complete_assessment)
    monkeypatch.setattr(selector, "assess_monthly_targets", complete_assessment)
    tasks, diagnostics, complete_cases = selector.select_all_families(
        frame,
        root=selector.ROOT,
        n_per_family=2,
        seed=10,
        window=3,
        transaction_count_threshold=1,
    )

    counts = tasks.groupby("outcome_family").size().to_dict()
    assert counts == {"housing": 1, "poi": 2, "population": 2, "viirs": 2}
    assert set(tasks.loc[tasks["outcome_family"].eq("poi"), "treatment_order"]) == {3, 4}
    assert set(tasks.loc[tasks["outcome_family"].eq("viirs"), "treatment_order"]) == {1, 2}
    assert diagnostics["families"]["housing"]["shortfall_from_requested"] == 1
    assert not tasks.duplicated(["treatment_order", "outcome_family"]).any()
    assert len(complete_cases) == 1


def test_housing_candidate_pool_does_not_require_post_complete_support() -> None:
    treatments, support, pretrend, control = _inputs()
    support["housing_complete"] = False
    frame = selector.build_candidate_frame(treatments, support, pretrend, control)

    assert frame["housing_pre_ready"].all()
    assert not frame["housing_base_ready"].any()


def test_monthly_housing_uses_post_windows_only_for_complete_case(monkeypatch) -> None:
    candidates = pd.DataFrame(
        {
            "treatment_order": [1, 2],
            "city_key": ["beijing", "beijing"],
            "grid_id": ["complete", "missing"],
            "opening_month": ["2020-01", "2020-01"],
        }
    )
    calendar = selector.monthly_event_calendar("2020-01", anticipation_months=6)
    published = selector._published_months(pd.DatetimeIndex(calendar["post"]), 3)
    required = pd.DatetimeIndex(calendar["pre"]).append(published)
    rows = []
    for grid in ("complete", "missing"):
        for period in required:
            if grid == "missing" and period == published[-1]:
                continue
            rows.append(
                {
                    "city_key": "beijing",
                    "grid_id": grid,
                    "period": period,
                    "housing_log_price": 1.0,
                    "transaction_count": 1,
                }
            )
    monkeypatch.setattr(
        selector,
        "read_monthly_housing",
        lambda *_args, **_kwargs: pd.DataFrame(rows),
    )
    result = selector.assess_monthly_targets(
        candidates,
        "housing",
        root=selector.ROOT,
        window=3,
        transaction_count_threshold=1,
    ).set_index("treatment_order")
    assert bool(result.loc[1, "selection_eligible"])
    assert bool(result.loc[2, "selection_eligible"])
    assert bool(result.loc[1, "target_path_complete"])
    assert not bool(result.loc[2, "target_path_complete"])
    assert int(result.loc[1, "supported_horizon_count"]) == 6
    assert int(result.loc[2, "supported_horizon_count"]) == 5


def test_monthly_housing_requires_support_in_each_pre_block(monkeypatch) -> None:
    candidates = pd.DataFrame(
        {
            "treatment_order": [1, 2],
            "city_key": ["beijing", "beijing"],
            "grid_id": ["balanced", "recent_only"],
            "opening_month": ["2020-01", "2020-01"],
        }
    )
    calendar = selector.monthly_event_calendar("2020-01", anticipation_months=6)
    pre = pd.DatetimeIndex(calendar["pre"])
    balanced = pre[:4].append(pre[12:16]).append(pre[24:28])
    rows = [
        {
            "city_key": "beijing",
            "grid_id": grid,
            "period": period,
            "housing_log_price": 1.0,
            "transaction_count": 1,
        }
        for grid, periods in (("balanced", balanced), ("recent_only", pre[-12:]))
        for period in periods
    ]
    monkeypatch.setattr(
        selector,
        "read_monthly_housing",
        lambda *_args, **_kwargs: pd.DataFrame(rows),
    )
    result = selector.assess_monthly_targets(
        candidates,
        "housing",
        root=selector.ROOT,
        window=3,
        transaction_count_threshold=1,
    ).set_index("treatment_order")

    assert bool(result.loc[1, "selection_eligible"])
    assert not bool(result.loc[2, "selection_eligible"])


def test_housing_selection_does_not_use_post_coverage() -> None:
    eligible = pd.DataFrame(
        {
            "treatment_order": [1, 2, 3, 4],
            "stratum": ["a::2020"] * 4,
            "pre_block_min_valid": [2, 4, 3, 1],
            "target_pre_valid": [20, 20, 20, 20],
            "target_post_valid": [24, 0, 12, 24],
        }
    )
    first = selector.select_family_sample(
        eligible, "housing", target=2, seed=10
    )
    changed = eligible.assign(target_post_valid=[0, 24, 24, 0])
    second = selector.select_family_sample(
        changed, "housing", target=2, seed=10
    )

    assert first["treatment_order"].tolist() == second["treatment_order"].tolist()


@pytest.mark.parametrize(
    ("hedonic_exists", "expected"),
    [(False, "median"), (True, "hedonic")],
)
def test_monthly_housing_main_matches_formal_measure_resolution(
    monkeypatch, tmp_path, hedonic_exists, expected
) -> None:
    candidates = pd.DataFrame(
        {
            "treatment_order": [1],
            "city_key": ["beijing"],
            "grid_id": ["g1"],
            "opening_month": ["2020-01"],
        }
    )
    if hedonic_exists:
        path = tmp_path / "outputs" / "causal_labels" / "housing_hedonic"
        path.mkdir(parents=True)
        (path / "beijing_monthly.parquet").touch()
    calls = []
    calendar = selector.monthly_event_calendar("2020-01", anticipation_months=6)
    periods = pd.DatetimeIndex(calendar["pre"]).append(
        pd.DatetimeIndex(calendar["post"])
    )

    def fake_reader(_root, _city, measure):
        calls.append(measure)
        return pd.DataFrame(
            {
                "city_key": "beijing",
                "grid_id": "g1",
                "period": periods,
                "housing_log_price": 1.0,
                "transaction_count": 1,
            }
        )

    monkeypatch.setattr(selector, "read_monthly_housing", fake_reader)
    result = selector.assess_monthly_targets(
        candidates,
        "housing",
        root=tmp_path,
        window=3,
        transaction_count_threshold=1,
        price_measure="main",
    )

    assert calls == [expected]
    assert result.loc[0, "price_measure"] == expected


def test_annual_target_requires_post_years(monkeypatch) -> None:
    candidates = pd.DataFrame(
        {
            "treatment_order": [1, 2],
            "city_key": ["beijing", "beijing"],
            "grid_id": ["complete", "missing"],
            "opening_year": [2020, 2020],
        }
    )
    rows = []
    for grid in ("complete", "missing"):
        for year in range(2015, 2024):
            if year == 2020 or (grid == "missing" and year == 2023):
                continue
            rows.append(
                {
                    "city_key": "beijing",
                    "grid_id": grid,
                    "year": year,
                    "population_log": 1.0,
                }
            )
    monkeypatch.setattr(
        selector,
        "read_annual_outcome",
        lambda *_args, **_kwargs: pd.DataFrame(rows),
    )
    result = selector.assess_annual_targets(
        candidates, "population", root=selector.ROOT
    ).set_index("treatment_order")
    assert bool(result.loc[1, "target_path_complete"])
    assert not bool(result.loc[2, "target_path_complete"])
