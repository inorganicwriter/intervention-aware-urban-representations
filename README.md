# Learning How Cities Respond

**Intervention-Conditioned Urban Representations**: construct per-grid causal response labels from metro station opening events, and learn cross-city transferable representations from pre-treatment multimodal urban features.

The project ultimately aims to learn not "whether two places look similar" but whether two places **respond similarly to the same intervention**. This repository first solves the label problem: for each treated grid, find or construct a credible untreated counterfactual and compute

```text
response(i, h) = observed_outcome(i, h) - counterfactual_outcome(i, h)
```

## Current implementation status

<!-- Pre-GPU migration wording retained for provenance:
PanelMatch, Abadie–Imbens and Xu GSC were described here as R production estimators;
event-study aggregation and the resumable label queue were documented under scripts/causal_r/.
-->

| Module | Status | Notes |
|---|---|---|
| Station identity, city assignment and competing-transit-event resolution | Implemented | Manual resolutions are compiled by a traceable applier; the raw station table is never overwritten |
| 500m grid treatment list and 1km spatial donor exclusion | Implemented | Fixed 5,048 treated grids; spatial donor universe generated |
| Pre-treatment multivariate control matching | Implemented | **6-round routing**: same-city matching → same-city GSC → same-city MC → cross-city matching → cross-city GSC → cross-city MC → explicit skip; the selection stage never reads post-treatment outcomes |
| Published-estimator references and Python/GPU production | Implemented; server qualification pending for GSC/MC | R `PanelMatch`, `Matching`, `gsynth` and `fect` remain audited reference implementations. The default label backend is `python_gpu`: PyTorch Matching/GSC/MC under `src/urban_intervention/causal/gpu/`, gated by R/Python parity receipts. See DDR-003/004. |
| Event-study aggregation and parallel-trends validation | Implemented | `scripts/causal_python/run_all_method_event_study.py` is the R-free production entry for Matching, GSC and MC. R aggregation remains an explicit reference workflow. Annual/monthly and estimator/scope results remain separate; pre-trend tests are diagnostic metadata, not automatic label deletion. |
| Resumable production queues | Implemented | The canonical label orchestrator is `scripts/causal_python/run_causal_label_queue.py`; atomic updates, shard-level qualification validation, resume, failure reasons and task provenance are implemented. The old `causal_r` path is a compatibility wrapper. |
| Response Artifact | Implemented | Strict release, complete label skeleton, quality grades, training mask, input/code hashes |
| Pre-training multimodal dataset | Implemented | Read-only pre-treatment features; split by city; standardization parameters fit on training cities only |
| Formal full-scale computation on 5,048 grids | **Full run pending** | Queues reset (5,048 controls + 20,192 family-level tasks all pending); two-stage matching canary verified (orders 1–10 routed correctly to GSC; 3/10 same-city matches in orders 906–915, order 906 walked the full match→GSC→MC→skip chain across all families); canary artifacts cleaned; awaiting the formal server run |
| Multimodal representation model and trainer | Implemented | Intervention-conditioned representation model + trainer (`src/urban_intervention/representation/`, `urban-train-representation` entry point), end-to-end validated on a demo dataset (30 epochs converged, no NaN; training samples filtered by `final_training_mask`); every run outputs `evaluation_report.json` (full-pool retrieval + per-family `nn_corr@k`, bootstrap CI, response-shuffle permutation test, raw-feature baseline, linear-probe transfer metrics, **chance/appearance baselines** (random projection / PCA / frozen DINOv2 / appearance autoencoder), **per-city retrieval + few-shot probe curves + response-direction prediction AUC**), `runs.jsonl` experiment tracking (config hash + headline metrics), `--seeds` multi-seed aggregation; `urban-build-model-card` builds model cards, `urban-run-ablation` runs ablation grids (templates in `configs/representation/`), `urban-export-embeddings` exports embedding parquet, `urban-summarize-runs` aggregates comparison tables; optional `--conditioning opening_year` explicit intervention conditioning and `--image-pooling max/mean/meanmax` image pooling; algorithm options: `--se-shrinkage` (SE-aware similarity shrinkage, default on), `--queue-size` (MemoryBank-style negative queue), `--learnable-temperature`, `--uncertainty-weighted`; evaluation protocol in [docs/research/representation_learning.md](docs/research/representation_learning.md) |
| Matching covariates (new this round) | Implemented and wired into matching | Location (distance to primary/secondary city centres), pre-opening rail conditions (nearest-station distance / station count / line count / closeness), station attributes (transfer / terminal / new line / extension / concurrent opening) collected; **two-stage control selection**: first match M=5 candidates on pre-treatment outcome lags, then refine within candidates for the best location/rail covariate balance; common support and holdout/placebo gates still act on outcome-history features only; SMD diagnostics report both outcomes and static covariates. See `docs/research/transit_accessibility_method.md` |
| Station attributes (treatment-level, time-of-opening) | Implemented | `scripts/analysis/build_station_attributes.py` derives per-station `is_transfer_at_opening` / `is_new_line_opening` / `is_extension_opening` / `is_terminal_at_opening` / `same_month_openings` (batch heuristic for line first-opening dates; station-degree terminals with transfer-ambiguous and no-adjacency NA flags) into `outputs/causal_labels/station_attributes/`; used for event-study stratification, robustness subsamples and representation-model conditioning tokens (not for treated-control matching — controls have no station) |
| Hedonic housing prices (Lianjia 22 cities) | Implemented as the housing-family main price measure for its cities | `scripts/labels/build_housing_hedonic.py`: per-city transaction hedonic (area/age/bedrooms/floor/orientation/decoration/building-type/elevator + community FE + year×quarter FE) → grid-month median adjusted price + n_transactions → `outputs/causal_labels/housing_hedonic/`; **observation-window parameter W** (1/3/6 months; W=3 main, W=1/6 robustness) implemented in the fixed-control label path and R outcome readers (`price_measure`/`window` arguments); raw median stays the 44-city unified matching measure |
| Transaction-composition audit | Implemented | `scripts/analysis/audit_housing_composition.py` plots n_transactions / mean area / mean building age around openings; found a +9.7% mean-age jump post-opening (composition shift), motivating the hedonic main measure |
| Same-city-first quality grading | Implemented | Response Artifact grades rank same-city paths above cross-city paths (`matched_same > gsc_same > mc_same > matched_cross > gsc_cross > mc_cross`); labels carry `donor_scope` and `main_spec`; `--scope-view {all,same_city,cross_city}` supports explicit sensitivity views |
| Sample-run tooling | Implemented (run deferred) | `scripts/analysis/select_representative_sample.py` (400 grids, city×opening-year Hamilton quotas with per-stratum floor; the production sample defaults to the complete-cache interval `2017-07` through `2022-12`, covering the full monthly GSC window), `--orders` on both queue runners, `scripts/analysis/summarize_causal_labels.py` (success/failure reasons, label distributions, same-city/cross-city/merged views); plan in `docs/operations/sample_run_execution_plan.md` |
| No-data task pre-screen | Implemented | Label queue skips tasks whose grid has no observation in the family panel (`family_no_observed_support`) instead of a ~3-minute doomed GSC/MC run; 1,942 tasks affected (verified 1.4 s vs 209 s) |
| MC uncertainty (single-treated-unit design) | Implemented | The formal MC path uses fixed-lambda `fect` jackknife inference (`spec$mc$inference=jackknife`), which produces finite S.E./CI/p on the supported design; unit-level bootstrap is not the formal inference path, and the label queue requires the jackknife manifest |

The "counterfactual label and pre-training data production" code is therefore closed-loop; the end-to-end pipeline for the representation model, training, cross-city evaluation and chance/appearance baselines is implemented (see the [representation learning docs](docs/research/representation_learning.md)); formal training on the full 5,048 grids can start once the formal response labels are released.

## Frozen research scope

- Study scope: 44 Chinese mainland metro cities.
- Spatial unit: fixed `500m × 500m` grids; the primary treatment is the opening of a formal metro station within the grid.
- Treatment list: 5,048 treated grids confirmed by station resolution and uniqueness audits.
- Donors: start from all non-treated grids; the main specification excludes units within 1km of the treated station's grid polygon.
- Timing: station dates are kept to the day; monthly estimation defines the first complete natural month after opening as event period 1.
- Outcome families: housing prices, VIIRS, POI, population. Different counterfactuals or missing masks per family are allowed depending on data support.
- Control selection: only pre-treatment housing, VIIRS, POI, population and their availability may be used; post-treatment outcomes or post-treatment missingness must not be read.
- Coverage-priority boundary: at least one complete variable family is sufficient to enter physical matching; MC after GSC failure allows `min.T0=1` and retains the actual number of pre-treatment support periods in the artifact.
- Routing: when a single physical control fails the common-support, holdout or placebo gates, fall back to Xu GSC; run MC after GSC failure; skip explicitly only when all three paths fail.

Precise definitions are governed by the [causal response label design](docs/research/counterfactual_response_label_design.md) and the [frozen decisions](docs/research/decisions/README.md).

## Repository architecture

```text
src/urban_intervention/
  causal/                 spatial donors, response artifact, pre-training data publication
  representation/         intervention-conditioned representation model, trainer,
                          statistical evaluation, baselines and transfer evaluation
  interventions/transit/  station manual-resolution applier
  pipelines/housing/      housing import, source adaptation and canonical panels
  pipelines/poi/          POI reading, standardization and aggregation
  config/                 city and project configuration
  data/                   data paths and registry

scripts/
  causal_python/          default Python/GPU production queues and event-study entrypoints
  causal_gpu/             R/Python shadow parity, GPU qualification and receipt tooling
  causal_r/               audited R references plus compatibility/deployment wrappers
  collection/             external data collection and source ingestion
  data/                   causal inputs and fixed task-list construction
  data_management/        data layout, registry and snapshot utilities
  labels/                 observed outcome construction (housing etc.)
  analysis/               reproducible data-quality audits; contains no formal estimators

data/
  catalog/                sources, schemas, field mappings and manual quality resolutions
  reference/              fixed boundaries, grids and resolved station events
  raw/                    immutable raw materials
  staging/                rebuildable source-level intermediates
  curated/                standardized covariates
  labels/                 canonical outcome variables
  panels/                 analysis-ready canonical panels
  causal/                 fixed treatment list, donor universe, inputs and production queues

outputs/                  audits, estimator objects and staging results (rebuildable, not versioned)
tests/                    Python contract tests and R estimator gates
docs/                     research design, data contracts and operations
```

The script responsibility index is in [scripts/README.md](scripts/README.md); the data-layering contract is in [docs/architecture/data_layout.md](docs/architecture/data_layout.md).

## Environment

Python uses the project-specific `mit` conda environment; the locked R version and packages are in `scripts/causal_r/RUNTIME_LOCK.csv`.

```powershell
conda env update -n mit -f environment.yml --prune
conda run -n mit python -m pip install -e ".[dev]"
conda run -n mit python -c "import sys; assert sys.version_info[:2] == (3, 11)"
```

R can live on the system `PATH` or be configured through environment variables; no research rules in the code need to change:

```powershell
$env:MIT_RSCRIPT = (Get-Command Rscript).Source
$env:MIT_R_LIB = (Resolve-Path '.r-lib').Path
$env:MIT_VIIRS_RAW = '<external-monthly-VIIRS-directory>'
```

Secrets may only be provided through environment variables or the uncommitted `config.yaml`.

The `--use-images` training path downloads DINOv2
(`facebookresearch/dinov2`) via `torch.hub` on first run; the cache directory
can be set with the `TORCH_HUB` environment variable; torchvision must match
the installed torch build (see the comments in `requirements.txt`).

All formal tasks use the Python 3.11 `mit` environment. Before running,
execute `conda env update -n mit -f environment.yml --prune` to ensure
runtime and test dependencies, including `openpyxl`, are installed; the
repository requires Python 3.11 or newer.

## Standard workflow

### 1. Basic validation

```powershell
conda run -n mit python scripts/data_management/validate_registry.py
conda run -n mit python -m pytest -q
& $env:MIT_RSCRIPT tests/causal_r/test_complete_estimators.R
```

`tests/causal_r/verify_complete_implementation.R` depends on formal estimator artifacts and should be run after the control-design and label queues; it is outside the basic validation scope.

### 2. Rebuild formal causal inputs

These commands reset the production queues. Run them only after confirming the input snapshots and hashes:

```powershell
& $env:MIT_RSCRIPT scripts/causal_r/reset_counterfactual_queues.R
& $env:MIT_RSCRIPT scripts/causal_r/build_formal_matching_inputs.R
& $env:MIT_RSCRIPT scripts/causal_r/audit_formal_target_support.R
```

### 3. Run the control-design and label queues

Dry-run first, then run a limited canary; do not start the full 5,048-grid run before the canary is reviewed.

```powershell
conda run -n mit python scripts/causal_r/run_grid_control_design_queue.py --start-order 1 --max-units 10 --dry-run
conda run -n mit python scripts/causal_python/run_causal_label_queue.py --start-order 1 --max-tasks 4 --dry-run
```

<!-- Pre-GPU path retained for compatibility documentation:
python scripts/causal_r/run_causal_label_queue.py ...
The wrapper still works, but new automation must use causal_python/.
-->

Python/GPU qualification is described in
[the causal GPU guide](src/urban_intervention/causal/gpu/README.md); R reference
semantics and server-side execution remain documented in
[scripts/causal_r/README.md](scripts/causal_r/README.md).

### 4. Release labels and pre-training data

Formal releases can only be created after all outcome-family tasks reach their terminal state:

```powershell
conda run -n mit python scripts/causal_r/build_response_artifact.py --release-id production_YYYYMMDD
conda run -n mit python scripts/causal_r/build_pretraining_dataset.py --response-release data/active/causal/releases/production_YYYYMMDD --dataset-id production_YYYYMMDD
```

`--allow-partial` is only permitted for canaries and tests; its artifacts carry a non-production marker and cannot enter formal training.
Strict releases verify the treatment-file hash, per-treated-unit identity,
task-directory/manifest/label consistency and the code version. In source
bundles without `.git`, the code version uses a `tree-sha256` of the
executable source tree; `unknown` is never accepted.

## Documentation

- [Server deployment and production runs](docs/operations/server_deployment.md)
- [Complete project and production pipeline](docs/research/complete_project_pipeline.md)
- [Causal response label design](docs/research/counterfactual_response_label_design.md)
- [Intervention-conditioned representation learning](docs/research/representation_learning.md)
- [Sample-run execution plan](docs/operations/sample_run_execution_plan.md)
- [Core research architecture](docs/architecture/research_architecture.md)
- [Current run status](docs/operations/current_project_status.md)
- [Data contracts](docs/data/README.md)
- [Operations manual](docs/operations/README.md)
- [Related-work literature](docs/research/related_work_literature.md)
- [Estimator formulas](docs/research/estimator_formulas.md)

Use the frozen DDRs and current production code as the authority. Historical pooled DID, annual panels, prototype estimators, smoke runs and partial releases remain archival diagnostics, not formal response labels.
