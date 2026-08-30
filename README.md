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
- resumable queues with atomic updates, task source records and structured failure
  reasons;
- validated Response Artifact and pretraining-dataset publishers;
- all-method event-study aggregation with pre-trend metadata;
- intervention-conditioned representation training, evaluation, baselines,
  transfer analysis and model-card generation.

Formal production requires three Matching, three GSC and three MC parity tasks on
the four-card RTX 4090 environment, a qualification receipt, frozen-input source
records, completion of the 5,048-grid queues, and publication of a validated
response release. Current queue counts and
remaining work are maintained in
[`docs/operations/current_project_status.md`](docs/operations/current_project_status.md).

## Frozen research scope

- Study scope: 44 Chinese mainland metro cities.
- Spatial unit: fixed `500m × 500m` grids; the primary treatment is the opening of a formal metro station within the grid.
- Treatment list: 5,048 treated grids confirmed by station resolution and uniqueness audits.
- Donors: start from all non-treated grids; the main specification excludes units within 1km of the treated station's grid polygon.
- Timing: station dates are kept to the day; monthly estimation defines the first complete natural month after opening as event period 1.
- Outcome families: housing prices, VIIRS, POI, population. Different counterfactuals or missing masks per family are allowed depending on data support.
- Control selection uses pre-treatment housing, VIIRS, POI, population and their availability; post-treatment outcomes and missingness stay outside this stage.
- Coverage-priority boundary: at least one complete variable family is sufficient to enter physical matching; MC after GSC failure allows `min.T0=1` and retains the actual number of pre-treatment support periods in the artifact.
- Routing: a physical control that fails the common-support, holdout or placebo gates enters Xu GSC; MC follows a GSC failure; the queue records `skipped` after all three paths fail.

Precise econometric definitions are governed by the [econometric methods manual](docs/research/econometric_methods.md), the [causal response label design](docs/research/counterfactual_response_label_design.md), and the [frozen decisions](docs/research/decisions/README.md). Executable diagnostic outputs are documented in the [DID parallel-trend guide](docs/research/identification_and_diagnostics.md).

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

outputs/                  audits, estimator objects and staging results (rebuildable and unversioned)
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
execution and test dependencies, including `openpyxl`, are installed; the
repository requires Python 3.11 or newer.

## Workflow

1. Validate the registry and run the Python test suite in the local `mit` environment.
2. Rebuild inputs, run qualification, and execute the control and label queues in an isolated server working copy.
3. Publish the Response Artifact and pretraining dataset after every outcome-family task reaches a terminal state.
4. Train the representation model from the published pretraining dataset.

The complete server command sequence is in
[`docs/operations/server_deployment.md`](docs/operations/server_deployment.md).
GPU qualification and estimator contracts are in
[`src/urban_intervention/causal/gpu/README.md`](src/urban_intervention/causal/gpu/README.md).
R reference commands are in
[`scripts/causal_r/README.md`](scripts/causal_r/README.md).

Release validation checks the treatment-file hash, per-treated-unit identity,
task-directory/manifest/label consistency and code version. Bounded test
artifacts carry a non-production marker.

## Documentation

- [Project guide and document authority](docs/PROJECT_GUIDE.md)
- [Server deployment and production runs](docs/operations/server_deployment.md)
- [Causal response label design](docs/research/counterfactual_response_label_design.md)
- [Econometric methods and identification](docs/research/econometric_methods.md)
- [DID parallel trends and executable diagnostics](docs/research/identification_and_diagnostics.md)
- [Intervention-conditioned representation learning](docs/research/representation_learning.md)
- [Core research architecture](docs/architecture/research_architecture.md)
- [Current run status](docs/operations/current_project_status.md)
- [Data contracts](docs/data/README.md)
- [Operations manual](docs/operations/README.md)
- [Related-work literature](docs/research/related_work_literature.md)
- [Estimator formulas](docs/research/estimator_formulas.md)

Use the frozen DDRs and current production code as the authority. Formal response labels come from the current production route. Superseded notes are removed after their durable content is consolidated into the project guide and causal design documentation.
