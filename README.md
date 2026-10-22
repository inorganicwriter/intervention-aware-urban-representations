# Learning How Cities Respond

**Intervention-Conditioned Urban Representations**: construct per-grid causal response labels from metro station opening events, and learn cross-city transferable representations from pre-treatment multimodal urban features.

The representation target is similarity in response to the same intervention.
For each treated grid, the label layer constructs an untreated counterfactual and computes

```text
response(i, h) = observed_outcome(i, h) - counterfactual_outcome(i, h)
```

## Production boundary

The shared STC joint-training implementation added on 2026-10-22 uses one
`z128`, `t32`, direct `b96`, a 256-dimensional conditional predictor input,
and independent VIIRS6 / POI12 heads. Joint STC bundles automatically use this
architecture; `--training-mode viirs_only`, `poi_only`, and `joint` provide
same-architecture controls on the same bundle. Engineering smoke validation
is separate from formal-data performance evaluation. See the
[joint-training plan and acceptance contract](docs/operations/stc_joint_training_plan.md)
and [runbook](docs/operations/stc_joint_training_runbook.md).

The causal estimation layer (Phase 1 & Phase 2) has completed and the production causal response release
is frozen under `data/active/causal/releases/production` (136,296 rows).

The active system status:

- **Causal Counterfactual Layer**: Completed and frozen. 44 cities, 5,048 treated grids, 3.77M donors.
- **Empirical Supervision Boundary**: Following the four-family event-study audit (`outputs/event_study_final/family_selection.md`),
  downstream causal supervision is strictly restricted to **VIIRS (6 horizons) and POI (12 horizons: 4 outcomes × 3 years)
  same-city Matching and GSC**. Housing and population, as well as MC and cross-city estimations, are excluded
  due to pre-trend failures and data sparsity.
- **Current Representation Model (S/T/C)**: the production architecture is `STCResponseModel`:
  - State `S`: 15 pre-treatment POI/VIIRS lag features, all legal pre-opening street-view images with time gaps,
    and the independent satellite sequence for event years -6 through -1.
  - Intervention `T`: 7 opening/station fields plus availability masks.
  - Context `C`: full AlphaEarth64, population lags and city summaries (housing price fields are excluded).
  - State encoding produces the single embedding `z`; `T` and `C` condition the response predictor
    `tau_hat = g(z, T, C)`. The objective is **Soft Causal Anchoring (~85%)** plus weak response alignment (~15%).
  - The earlier VIGOR cross-view AlphaEarth/street-view model remains available as the legacy baseline,
    not as the current production path.

- **Prepared Inputs (2026-10-13)**: State uses POI, VIIRS, street view and independent satellite images;
  intervention uses station/opening attributes; context uses all 64 AlphaEarth dimensions,
  population and city statistics (housing price fields are excluded). The full bundle is published and bound in the server project,
  with 37 binding checks and 17 runtime path checks passed. Historical gaps retain masks;
  intervention source review remains pending. Temporary transfer/staging caches were cleaned,
  while formal inputs and audit evidence remain available. Next: model input integration and
  a small training/save/export validation run. See
  [`input publication and cleanup`](docs/operations/input_server_sync_2026-10-13.md).

Current development and queues are tracked in
[`docs/operations/current_project_status.md`](docs/operations/current_project_status.md) and
[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).

## Frozen research scope

- Study scope: 44 Chinese mainland metro cities.
- Spatial unit: fixed `500m × 500m` grids; the primary treatment is the opening of a formal metro station within the grid.
- Treatment list: 5,048 treated grids confirmed by station resolution and uniqueness audits.
- Donors: start from all non-treated grids; the main specification excludes units within 1km of the treated station's grid polygon.
- Timing: station dates are kept to the day; monthly estimation defines the first complete natural month after opening as event period 1.
- Visual & tabular pre-treatment barrier: all input visual features and covariates must strictly precede the opening month to prevent information leakage.
- Representation supervision targets: VIIRS (6 horizons) and POI (12 horizons) under same-city Matching and GSC with design reliability weights.

Precise econometric definitions are governed by the [econometric methods manual](docs/research/econometric_methods.md), the [causal response label design](docs/research/counterfactual_response_label_design.md), and the [representation learning specification](docs/research/representation_learning.md). Executable diagnostic outputs are documented in the [DID parallel-trend guide](docs/research/identification_and_diagnostics.md).

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
