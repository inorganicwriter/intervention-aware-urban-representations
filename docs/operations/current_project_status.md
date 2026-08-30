# Current project state

Updated: 2026-08-30

This page contains the state required to resume production. Research definitions
are maintained in `docs/research/`; dataset locations and source identities are
maintained in `data/active/catalog/datasets.yaml`.

## Research frame

| Item | Current value |
|---|---|
| Cities | 44 |
| Spatial unit | fixed 500m × 500m grid |
| Treated grids | 5,048 |
| Initial spatially eligible donor rows | 3,771,800 |
| Main spatial exclusion | 1km from the treated station grid polygon |
| Outcome families | housing, VIIRS, POI, population |
| Routing | same-city Matching → GSC → MC → cross-city Matching → GSC → MC → skip |

The treatment list, donor universe, metadata and queues are under
`data/active/causal/`. These counts describe the spatial design. Successful
estimation and label availability are reported in the queue tables.

`data/active/` is the frozen input and queue baseline for this workspace. Review
and cleaning tasks use it as read-only input. A server production run uses an
isolated working copy and records the hashes of the frozen inputs.

## Data products

- Canonical housing month, quarter and year panels: `data/active/panels/`;
- POI grid-year panel, 2012-2024: `data/active/curated/`;
- annual VIIRS, 2012-2024: `data/active/curated/viirs_annual_aggregated/`;
- monthly VIIRS source: external path configured by `MIT_VIIRS_RAW`;
- population panel: `data/active/curated/population/`;
- Sentinel-2/Landsat panel, 2014-2024: `data/active/curated/sentinel2/`;
- location features: `data/active/curated/location_features/`;
- pre-treatment transit snapshots and accessibility features:
  `data/active/causal/transit_snapshots/` and
  `data/active/causal/accessibility_features/`.

Detailed source and coverage information belongs in `docs/data/`.

## Production entry points

| Task | Entry point |
|---|---|
| Grid control design | `scripts/causal_r/run_grid_control_design_queue.py` |
| Causal label queue | `scripts/causal_python/run_causal_label_queue.py` |
| Single Python/GPU estimator | `scripts/causal_python/run_formal_estimator.py` |
| Matching/GSC/MC event study | `scripts/causal_python/run_all_method_event_study.py` |
| GPU parity and qualification | `scripts/causal_gpu/` |
| Response Artifact | `scripts/causal_r/build_response_artifact.py` |
| Pretraining dataset | `scripts/causal_r/build_pretraining_dataset.py` |
| Representation training | `urban-train-representation` |

The default estimator backend is `python_gpu`. R `PanelMatch`, `Matching`,
`gsynth` and `fect` remain reference implementations and the explicit
`r_reference` backend. Python/GPU production stays stopped until the active
environment has a valid qualification receipt containing at least three
Matching, three GSC and three MC parity tasks.

The following parallel modular implementations remain experimental:

- `scripts/causal_python/run_causal_label_queue_modular.py`;
- `urban_intervention.config.project_modular`;
- `urban_intervention.representation.trainer_modular`.

The original files remain canonical until the modular candidates pass bounded
real-data validation on the production server. Cutover must replace the original
implementation bodies and remove the `_modular` entry points in the same change;
stable import and script paths may remain as delegation-only compatibility facades.

## Queue state

| Queue | Rows | State |
|---|---:|---|
| `control_design_queue.csv` | 5,048 | 4,648 `pending`; 218 `matched`; 182 `gsc_pending` |
| `outcome_family_work_queue.csv` | 20,192 | 20,192 `pending` |
| `counterfactual_work_queue.csv` | 5,048 | 5,048 `pending` |

Queue CSV files are single-writer resources. Parallel execution must use the
provided shard orchestrator rather than multiple independent writers.

## Work required before formal production

1. Run three representative GSC and three MC parity tasks on the RTX 4090
   server and issue the environment-bound qualification receipt.
2. Run a server dry-run and bounded test with the frozen inputs and production
   parameters.
3. Run the 5,048-row control queue and the 20,192 family-level label queue.
4. Aggregate Matching, GSC and MC event studies and review pre-trend metadata.
5. Publish the validated Response Artifact and pretraining dataset.
6. Train and evaluate the representation model on the formal release.

The code checks are in place: all queue entry points default to `python_gpu`,
startup prints the selected backend, and Response Artifact publication validates
the persistent control-record schema, backend, version and method. Production
launchers stop when the frozen formal spec disagrees with the current code
contract. These checks leave `data/active/` unchanged.

Before starting the full server run, two data-source and specification records
need to be aligned:

- all 218 currently `matched` control-queue rows map to task records carrying
  the historical `Match_M1`/old schema, while the current design is
  `M5 + static refine`;
- `data/active/causal/formal_matching_inputs/formal_matching_spec.dput` records
  `minimum_complete_families = 2`, while the current code contract is 1.

The frozen active file remains unchanged. Confirm the specification version and
decide whether old records are eligible for the server run. Rebuild in the
isolated server working copy when the records fail the current checks. The
release validator will reject records that fail these checks.

GSC production inference uses 200 parametric bootstrap replications. MC uses
fixed-lambda unit jackknife inference; its `nboots=200` field is compatibility
metadata.

## GPU estimator qualification status

The qualification workflow compares three Matching, three GSC and three MC
tasks with their R references. Production requires all point-estimate,
inference and quality gates to pass and requires a qualification receipt from
the same source tree and numerical environment. Qualification does not run the
5,048-row production queue. Production begins after qualification and the
server dry run both succeed.

## Representation output contract

Each training run writes:

- `best_model.pt`;
- `training_config.json` and `training_history.json`;
- `test_metrics.json` and `evaluation_report.json`;
- `runs.jsonl`.

Evaluation reports retrieval metrics, bootstrap intervals, response-label
permutation tests, raw-feature and appearance baselines, and transfer metrics.
Formal conclusions require the full production release; demo and partial
formal inference uses the full production release.

## Result acceptance

A label enters the formal dataset when its frozen design record, estimator
output, quality diagnostics, task identity, input hashes and run ID agree and
pass the Response Artifact checks. Partial, smoke, preview and expired outputs
carry `production_eligible = FALSE` and stay outside formal training.
