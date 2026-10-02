# Current project state

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

The local queue snapshot below is the frozen baseline. The server working copy
completed Phase 1 with 3,032 `matched` and 2,016 `gsc_pending` rows
(see [`server_runtime_profile.md`](server_runtime_profile.md)); do not infer
production progress from the local frozen queues.  Phase 2 is complete and the
strict causal release was published from the server copy; see the current-phase
section below.

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
| Four-family event-study final export | `scripts/causal_python/export_event_study_final.py` |
| GPU parity and qualification | `scripts/causal_gpu/` |
| Round-pool scheduler (optional) | `scripts/causal_python/run_round_pool.py` |
| Round-pool worker | `scripts/causal_python/run_round_pool_worker.py` |
| Round-pool status | `scripts/causal_python/round_pool_status.py` |
| Response Artifact | `scripts/causal_r/build_response_artifact.py` |
| Pretraining dataset | `scripts/causal_r/build_pretraining_dataset.py` |
| Representation training | `urban-train-representation` |

The analysis-stage closure sequence is: four-family event-study final release
→ VIIRS+POI label extraction → design-only weights → local verification →
frozen `training_ready` release → server training.  Server-side steps are
deferred and tracked as a checklist in
[`training_release_checklist.md`](training_release_checklist.md); the
event-study closure mechanics are in
[`event_study_final_runbook.md`](event_study_final_runbook.md).  Server work
does not start before the local release is frozen.

Phase 2 supports three scheduling modes: `global_rounds` (default),
`task_cascade`, and `round_pool`. The SQLite-backed `round_pool` mode uses
strict six-round barriers and dynamic task claiming; it requires `--phase 2`
with `--queue-phase all`, cannot be combined with `--dry-run`, resumes through
`--round-pool-run-id`, and exposes `--requeue-failed` for hard failures.

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

## Queue state (local frozen baseline)

| Queue | Rows | State |
|---|---:|---|
| `control_design_queue.csv` | 5,048 | 4,648 `pending`; 218 `matched`; 182 `gsc_pending` |
| `outcome_family_work_queue.csv` | 20,192 | 20,192 `pending` |
| `counterfactual_work_queue.csv` | 5,048 | 5,048 `pending` |

The server working copy completed Phase 2 (all 20,192
family-level tasks terminal; terminal queue census: 9,162 `skipped`, 7,553
`matched_labelled`, 2,322 `gsc_labelled`, 1,155 `mc_labelled`).  Before
publication, 226 MC tasks without a pre-treatment anchor were demoted to
`skipped` (housing 2 / POI 131 / population 93; audit JSON in
`outputs/causal_labels/qc/`), and the strict release
`data/active/causal/releases/production` (136,296 rows) was published
with attested control-design and qualification drift waivers.

Queue CSV files are single-writer resources. Parallel execution must use the
provided shard orchestrator (or the round-pool coordinator) rather than
multiple independent writers.

## Current phase (analysis closure)

Phase 1 and Phase 2 are complete and the strict causal release is published
(`data/active/causal/releases/production`, server working copy).  The terminal
queue is recovered from the QC pre-demotion backup plus the recorded MC
demotions, and code-package unzips always exclude
`data/active/causal/*queue*.csv` so server state is not rolled back
([`event_study_final_runbook.md`](event_study_final_runbook.md) §6).

The analysis stage is closed:

1. the four-family event-study suite has been re-run with the recovered queues,
   each family under both matching donor scopes; combinations without enough
   observations are recorded as `no_data` rather than dropped;
2. the frozen release is exported through `export_event_study_final.py`
   (canonical windows, availability, pre-trends, family selection, manifest)
   with validation passing;
3. pretraining datasets carry per-label reliability weights; the earlier joint
   build (`viirs-poi-design` / `viirs-poi-full`) is superseded by the
   per-family, same-city matched+GSC pipeline (`viirs-mg-design` /
   `poi-mg-design`; ablations `*-mg-w0` / `*-mg-full` / `*-3m-design`).

Next: per-family representation training (W-design main, then the ablation
matrix) and evaluation; the server checklist lives in
[`training_release_checklist.md`](training_release_checklist.md).

The representation stage is in diagnosis, not in the formal training matrix:

1. probe/evaluation fixes and the station-conditioning scale fix landed in
   `b945ed6`; the diagnostics and city-CV control scripts are
   [`representation_diagnostics.py`](../../scripts/analysis/representation_diagnostics.py)
   and [`representation_controls.py`](../../scripts/analysis/representation_controls.py);
2. several multi-seed checks show no stable increment over the raw-feature or
   appearance baselines; the label support, design-weight cancellation,
   float32, and embedding-geometry checks are recorded in
   [`representation_learning_audit.md`](../research/representation_learning_audit.md);
3. the next step is the minimal control (raw ridge / k-NN / mean baseline under
   fixed city-level cross-validation) to decide whether the bottleneck is the
   features or the objective; the formal multi-seed matrix starts only after
   that decision, and the per-family runbook gates retraining on it.

The code checks are in place: all queue entry points default to `python_gpu`,
startup prints the selected backend, and Response Artifact publication validates
the persistent control-record schema, backend, version and method. Production
launchers stop when the frozen formal spec disagrees with the current code
contract. These checks leave `data/active/` unchanged.

Two legacy records remain in the local frozen baseline:

- the 218 `matched` rows in the local queue map to historical `Match_M1`/old
  schema task records, while the current design is `M5 + static refine`;
- `data/active/causal/formal_matching_inputs/formal_matching_spec.dput` records
  `minimum_complete_families = 2`, while the current code contract is 1.

The local frozen file stays read-only. The server copy superseded these records
during the Phase 1 completion; on resume, `--repair-stale-controls`
recomputes only records that fail the current source checks, and the release
validator rejects any remaining stale records.

GSC production inference uses 200 parametric bootstrap replications. MC uses
fixed-lambda unit jackknife inference; its `nboots=200` field is compatibility
metadata. MC labels additionally require at least one observed clean
pre-treatment period: outcomes without one are failed by the label queue
(`mc_insufficient_pre_observation` / `python_mc_insufficient_pre_observation`)
so the routing falls through instead of publishing an unanchored
counterfactual, and the Response Artifact rejects any remaining such rows.

## GPU estimator qualification status

The qualification workflow compares three Matching, three GSC and three MC
tasks with their R references. Production requires all point-estimate,
inference and quality gates to pass and requires a qualification receipt from
the same source tree and numerical environment. Qualification does not run the
5,048-row production queue. Production begins after qualification and the
server dry run both succeed.

A reviewed, non-numerical change to qualification-bound sources (comments,
typing, error paths) can reuse an existing eligible receipt through an
attested drift waiver:

```bash
python scripts/causal_gpu/issue_qualification_waiver.py \
  --receipt outputs/causal_gpu/formal_qualification.json \
  --reason "<reviewed diff summary>" --confirm-no-numerical-change
```

Production passes it with `--qualification-waiver <path>` (or
`MIT_CAUSAL_QUALIFICATION_WAIVER`). The waiver binds the receipt digest, the
current estimator fingerprints and every bound source hash, and never
overrides parity, quality, convergence, environment or bound-input gates. Any
change that can affect estimator numerics still requires re-running the three
Matching, three GSC and three MC parity tasks.

## Representation output contract

Each training run writes:

- `best_model.pt`;
- `training_config.json` and `training_history.json`;
- `test_metrics.json` and `evaluation_report.json`;
- `runs.jsonl`.

Evaluation reports retrieval metrics, bootstrap intervals, response-label
permutation tests, raw-feature and appearance baselines, and transfer metrics.
Formal conclusions require the full production release; demo and partial
outputs carry `production_eligible = FALSE` and stay outside formal training.

## Result acceptance

A label enters the formal dataset when its frozen design record, estimator
output, quality diagnostics, task identity, input hashes and run ID agree and
pass the Response Artifact checks. Partial, smoke, preview and expired outputs
carry `production_eligible = FALSE` and stay outside formal training.
