from __future__ import annotations

import numpy as np
import pandas as pd

from urban_intervention.causal.gpu.provenance import file_sha256
from urban_intervention.causal.pooled_event_study import (
    aggregate_effect_paths,
    attach_pretrend_metadata,
    pretrend_tests,
    run_pooled_path_event_study,
    select_pretrend_rows,
)


def _paths() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for order in range(1, 13):
        for event_time in (-9, -8, -7, 1, 2):
            rows.append(
                {
                    "treatment_order": order,
                    "city_key": f"city_{order % 4}",
                    "frequency": "monthly",
                    "outcome_family": "viirs",
                    "outcome": "viirs_avg_asinh",
                    "method": "xu_2017_gsynth",
                    "donor_scope": "same_city",
                    "event_time": event_time,
                    "causal_response_label": (
                        order * 0.001 if event_time < 0 else 0.2 * event_time + order * 0.001
                    ),
                    "label_available": True,
                    "standard_error": 0.05,
                    "specification_fingerprint": "main",
                    "run_id": f"run-{order}",
                }
            )
    return pd.DataFrame(rows)


def test_pooled_series_combines_within_and_between_variance() -> None:
    series = aggregate_effect_paths(_paths())
    row = series.loc[series["event_time"].eq(1)].iloc[0]
    assert row["n_grids"] == 12
    assert row["se_available"] == 12
    assert row["within_var"] > 0
    assert row["between_var"] > 0
    assert row["ci_lower"] < row["mean_label"] < row["ci_upper"]


def test_pretrend_metadata_is_diagnostic_not_an_admission_gate() -> None:
    paths = _paths()
    selected = select_pretrend_rows(paths, latest_n=2)
    assert set(selected["event_time"]) == {-8, -7}
    grid = pretrend_tests(paths, cluster="grid", latest_n=2)
    city = pretrend_tests(paths, cluster="city", latest_n=2)
    enriched = attach_pretrend_metadata(paths, grid, city)
    assert np.isfinite(enriched["grid_pretrend_p_value"]).all()
    assert enriched["pretrend_flag"].str.contains("cluster_").all()
    assert enriched["label_available"].all()


def test_pooled_path_runner_reads_manifest_and_writes_metadata(tmp_path) -> None:
    task = tmp_path / "staging" / "task"
    task.mkdir(parents=True)
    labels = _paths()
    labels.to_parquet(task / "causal_response_labels.parquet", index=False)
    manifest = {
        "run_mode": "production",
        "production_eligible": "TRUE",
        "estimator": "gsc",
        "frequency": "monthly",
        "donor_scope": "same_city",
        "specification_fingerprint": "main",
        "run_id": "qualification-run",
        "labels_sha256": file_sha256(task / "causal_response_labels.parquet"),
    }
    pd.DataFrame({"field": manifest.keys(), "value": manifest.values()}).to_csv(
        task / "manifest.csv", index=False
    )
    output = tmp_path / "result"
    result = run_pooled_path_event_study(
        tmp_path / "staging", output, figures=False
    )
    assert result.diagnostics["treatment_orders"] == 12
    assert (output / "effect_paths_with_pretrend.parquet").is_file()
    assert (output / "pretrend_city_cluster.csv").is_file()
