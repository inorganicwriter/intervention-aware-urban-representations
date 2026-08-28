# Representative sample run

Date: 2026-08-19

Execution state: deferred

This document defines a 400-grid production-like run for estimating execution
time, method routing and data support before the 5,048-grid run. It uses a
separate sample and produces non-production labels.

## Sample definition

`scripts/analysis/select_representative_sample.py` selects 400 grids using
fixed-seed city × opening-year Hamilton quotas with a per-stratum floor. The
sample selects opening months from `2017-07` through `2022-12`, which
fits the processed VIIRS cache and the required 42-month pre-treatment plus
24-month post-treatment window.

Output:

```text
outputs/causal_labels/representative_sample_400.csv
```

The file must contain treatment order, city, opening period, stratum and sample
weight.

## Outcome specifications

- Matching control design uses the unified median housing measure across all
  44 cities.
- Lianjia cities report hedonic-adjusted housing as the main price label and
  median housing as a companion specification.
- Other cities use median housing.
- VIIRS is monthly; POI and population are annual.
- Matching controls are selected once per treatment grid. GSC and MC are fitted
  separately for each outcome measure.

Housing observation windows are:

- `W=3`: main specification;
- `W=1` and `W=6`: sensitivity specifications.

The baseline remains the 12-month pre-opening block. Missing calendar months are
not imputed. Each window records the effective observed and counterfactual
sample counts.

## Scope and quality fields

- same-city and cross-city labels remain separate through `donor_scope`;
- `main_spec=1` only for same-city labels;
- quality order is `matched_same > gsc_same > mc_same > matched_cross >
  gsc_cross > mc_cross`;
- treatment-level station attributes are used for stratification and model
  conditioning, not treated-control matching.

## Execution

1. Generate the sample list.
2. Run a 10-grid dry-run and bounded test to measure task cost and memory.
3. Run the control-design queue for the selected orders.
4. Run the family-level label queue for the same orders.
5. Build a partial Response Artifact with `--allow-partial`.
6. Summarize routing, failures, label distributions and scope-specific results.

Example commands:

```bash
python scripts/analysis/select_representative_sample.py
python scripts/causal_r/run_grid_control_design_queue.py --orders <orders> --dry-run
python scripts/causal_python/run_causal_label_queue.py --orders <orders> --dry-run
```

Remove `--dry-run` after inspecting the bounded-test manifests and resource use.

## Outputs

- representative sample and stratum report;
- partial Response Artifact with `production_eligible = FALSE`;
- method-routing and failure summary;
- family-level label distributions;
- Matching, GSC and MC event-study outputs separated by method and donor scope;
- housing composition and observation-count diagnostics.

All outputs remain under `outputs/`. No sample-run artifact may overwrite
`data/active` or enter formal representation training.
