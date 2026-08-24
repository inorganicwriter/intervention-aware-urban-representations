# Learning How Cities Respond

**Intervention-Conditioned Urban Representations**: construct per-grid causal response labels from metro station opening events, and learn cross-city transferable representations from pre-treatment multimodal urban features.

The representation target is similarity in response to the same intervention.
For each treated grid, the label layer constructs an untreated counterfactual and computes

```text
response(i, h) = observed_outcome(i, h) - counterfactual_outcome(i, h)
```

## Production boundary

The production label backend is `python_gpu`. Matching, GSC and MC run through
PyTorch implementations under `src/urban_intervention/causal/gpu/`; R
`PanelMatch`, `Matching`, `gsynth` and `fect` are reference implementations and
the explicit `r_reference` backend.

The production system includes:

- a fixed 44-city, 5,048-grid treatment design and 1km donor exclusion;
- pre-treatment-only six-round routing through same-city and cross-city
  Matching, GSC and MC;
- resumable queues with atomic updates, task provenance and structured failure
  reasons;
- strict Response Artifact and pretraining-dataset publishers;
- all-method event-study aggregation with pre-trend metadata;
- intervention-conditioned representation training, evaluation, baselines,
  transfer analysis and model-card generation.

Formal production remains contingent on three GSC and three MC parity tasks on
the RTX 4090 environment, issuance of the qualification receipt, execution of
the 5,048-grid queues, and publication of the strict response release. Current
queue counts and remaining work are maintained in
[`docs/operations/current_project_status.md`](docs/operations/current_project_status.md).

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
