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

The 2026-10-13 input refresh uses the frozen treatment/grid identities. New
AlphaEarth background embeddings are in `data/active/alphaearth_grid_v2/`
(3,297 available; 1,751 explicit historical gaps). Raw state/context/intervention
tables for all 5,048 events are in `data/active/multimodal_inputs_v1/`;
reconstructed intervention fields retain their source-review status. The 5,048
independent baseline satellite images passed per-image audit. Annual satellite
sequences are published in `data/active/satellite_event_sequences_v2/`
for event years -6 through +3 (50,480 rows: 48,465 available images, five
low-coverage frames and 2,010 incomplete-calendar-year placeholders). All 48,470
physical images passed per-frame audit with zero errors. The state sequence has
30,288 rows (30,283 available); only years -6 through -1 may enter the state encoder.
Street-view metadata and selected-image verification cover all 44 cities:
3,778 events have 7,543 legal pre-opening images; 1,270 events retain masks.
The full release is stored only in the server project at
`/home/nas/lsr/MIT_summer/data/active/streetview_grid_v2/`; its manifest confirms
final publication. Windows retains an earlier Beijing pilot, not the full image
release. The complete new AlphaEarth, annual satellite and S/T/C package is now
published in the same server project: all 269,738 transferred files passed SHA256
verification. Server input binding passed 37 checks and runtime path verification
passed 17 checks, including readable image/source samples in all 44 cities.
The server state table uses the full 3,778-event street-view availability masks;
the local pilot tables retain their original values. See
[`input_server_sync_2026-10-13.md`](input_server_sync_2026-10-13.md) for completion receipts.
The post-opening street-view archive retains all 20,192 event-year records for
years 0 through +3, with 5,489 verified available combinations and 10,854 images.
Its separate `post_manifest.json` confirms final publication; all post images
have `state_allowed=false`. Satellite opening-year and post-year archive frames
are already part of the full -6 through +3 local satellite sequence.
The independent S/T/C (State/Intervention/Context) multimodal representation model
architecture is implemented and hardened against review findings:
- State $S$ (pre-treatment POI/VIIRS lags, pre-opening streetview, pre-opening 6-year satellite reflectance sequences) encodes strictly into $z$;
- Intervention $T$ (7 fields + masks) encodes into $t$;
- Context $C$ (76 raw values: full AlphaEarth64 plus 12 historical population/city fields, with 13 masks; housing price fields are excluded) is projected into 64-dimensional $b$;
- The learned response-sensitive state embedding is `z = f_theta(S)`: POI/VIIRS history, all legal pre-opening streetviews, satellite image sequences and masks are encoded together. `T` and `C` are encoded separately as `t` and `b` and condition the response predictor: `tau_hat = g_phi(z, T, C)`, implemented as prediction from `concat(z, t, b)`. Causal response labels supervise prediction and response-similarity alignment of `z`; post-opening imagery is not a target.
- Review repairs (2026-10-16):
  1. Streetview consumption: `StateEncoder` directly consumes raw streetview images when precomputed feature cache is absent or zero;
  2. Identity binding: `publish_stc_dataset` and `RepresentationDataset` enforce 1-to-1 matching across `treatment_order`, `city_key`, `grid_id`, `station_event_id`, and `opening_month`;
  3. Temporal barrier: pre-treatment satellite frames strictly check `target_year == opening_year + event_time` and `target_year < opening_year`, rejecting future frames (e.g. 2099);
  4. Normalized convolution TIFF resampling: invalid/NaN/infinite pixels are zeroed prior to resampling, ensuring zero contamination of valid pixels;
  5. Checkpoint & export contract: `export_embeddings` strictly validates full S/T/C field ordering, relative years, and dimensions against checkpoint metadata.
- Local regression & smoke verification: The earlier local run on 2026-10-19 passed 60 related unit and integration tests with no skips or deselections, including real UMAP and an independent PCA fallback test. It used explicitly marked fixture labels. The latest server-candidate regression passed all 62 tests in the six required files; see the current acceptance section below.
- Server production acceptance: Completed on 2026-10-19 in the existing `mit` environment using the strict production Response Artifact, real W-design/Matching weights, and verified pre-opening images. The latest run also validates the current time-aware multi-streetview input path; see the acceptance section below and the consolidated receipt in `stc_code_readiness.md`.
See [the consolidated code readiness record](stc_code_readiness.md) and
[`stc_model_implementation_plan_2026-10-15.md`](stc_model_implementation_plan_2026-10-15.md).
Temporary transfer archives, duplicate extraction/street-view staging and local
diagnostic payloads were cleaned after verified publication. Formal data,
source manifests, original role backups and audit evidence remain available;
the current transfer receipt points to the persistent source manifest under
`outputs/input_sync_20261013_v1/`, rather than a removed `/tmp` directory.

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

## Current phase (causal closure complete; multimodal representation implementation)

Phase 1 and Phase 2 are complete and the strict causal release is published and frozen
(`data/active/causal/releases/production`, 136,296 rows). The terminal
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
3. per-family pretraining datasets carry per-label reliability weights; the earlier joint
   build (`viirs-poi-design` / `viirs-poi-full`) is superseded by the
   per-family, same-city matched+GSC pipeline (`viirs-mg-design` /
   `poi-mg-design`; ablations `*-mg-w0` / `*-mg-full` / `*-3m-design`).

### Representation Learning Upgrade: Multimodal Cross-View Architecture

The following architecture and component checklist describe the earlier
AlphaEarth/street-view implementation. The current S/T/C role definition and
asset coverage are stated above; this checklist does not confirm the new
independent satellite sequence and background branches or new training results.

Diagnosis of previous representation experiments
([`representation_learning_audit.md`](../research/representation_learning_audit.md))
identified that 27-d tabular features had low effective rank (8–12), causing
"dimensionality inversion" (expanding 27 to 128 embedding) which prevented the representation
from beating Raw Ridge baselines.

The project has upgraded to a **Multimodal Perceptual Representation Architecture**
([`representation_learning.md`](../research/representation_learning.md)):
1. **Inputs**:
   - **Ground Street-View Panoramas** ($360^\circ$ full panoramas residing on user server, DINOv2 visual features, strictly filtered to `capture_date < opening_month`).
   - **Satellite Embedding Field**: Google AlphaEarth Foundations (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, 64-d continuous dense field via Earth Engine `projects/macro-city-engine/assets/mit_grids_v2/{city}`, strictly `lag 1` pre-treatment).
   - **Tabular Priors & Station Tokens**: 27-d pre-treatment tabular covariates + 5-bit station attribute tokens.
2. **Cross-View Fusion**: VIGOR (CVPR 2021) cross-view synergy paradigm where ground perspective queries satellite field via cross-attention.
3. **Training Objective**: **Soft Causal Anchoring ($L_{\text{anchor}}$, ~85%)** using design-weighted Huber loss on causal response trajectories, regularized by weak contrastive alignment ($L_{\text{align}}$, ~15%).
4. **Supervision Boundary**: Strictly restricted to **VIIRS (6 horizons) and POI (12 horizons: 4 outcomes × 3 years) same-city Matching & GSC** (housing, population, MC, and cross-city are strictly excluded per `outputs/event_study_final/family_selection.md`).

### Implementation Checklist & Pending Items

| Component | Status | Verification & Artifacts |
|---|---|---|
| **Causal Counterfactual Layer** | ✅ Closed & Frozen | Immutable under `data/active/causal/releases/production` (136,296 rows; strictly restricted to VIIRS & POI same-city Matching & GSC). |
| **Google AlphaEarth Extraction** | ✅ Complete | 44 城 5,048 处理网格全量提取完成（`data/active/alphaearth/alphaearth_treated_embeddings.parquet`），100% 覆盖，通过 GEE 批量特征抽取。 |
| **Street-View Panorama Pipeline** | ✅ Complete | 全景图解析索引构建管线就绪（`scripts/collection/build_streetview_index.py`），严格实施开通前时序屏障（`capture_date < opening_month`）。 |
| **Representation Encoders** | ✅ Complete | 实现 `SatelliteAlphaEarthEncoder`、`StreetviewPanoramaEncoder`、`CrossViewSynergyModule`（VIGOR 交叉注意力）在 `encoder.py`，支持缺失模态动态退化。 |
| **Dataset Loader** | ✅ Complete | `dataset.py` 中 `GridSample`、`RepresentationDataset`、`collate_samples` 全面支持 AlphaEarth 64-d 嵌入向量、街景图像和站点拓扑标记。 |
| **Loss Function** | ✅ Complete | `loss.py` 实现设计权重惩罚 Huber 轨迹损失（Soft Causal Anchoring, ~85%）与弱对比正则（Soft-InfoNCE, ~15%）。 |
| **Evaluation Suite** | ✅ Complete | `evaluation.py` 实现 Level 1 轨迹预测（RMSE/MAE/Direction AUC）、Level 2 规划语义分类（三类机制 F1/Kappa）、Level 3 跨城泛化以及 Level 4 潜在流形流形可视化（UMAP/SVD PCA）。 |
| **CLI & Training Pipeline** | ✅ Complete | `cli.py`、`trainer.py`、`training/runner.py` 与 `build_pretraining_dataset.py` 均已接入 `--use-satellite` 与 `--use-cross-view-synergy` 标志，通过单元测试。 |

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

## S/T/C strict-production server acceptance: initial run (historical)

The real-data engineering acceptance completed in the existing `mit` environment on `222.29.102.59:2222`, under `/home/nas/lsr/MIT_summer/outputs/stc_server_acceptance/stc_acceptance_20261019T050410Z`. VIIRS and POI each used 64 events from the strict production Response Artifact, retained official city splits, read real streetview and satellite TIFF assets, completed two training epochs, restored checkpoints, and exported 64 unique event identities. VIIRS had 381 valid response cells across 6 dimensions; POI had 720 across 12. All 383 declared satellite TIFF frames and all 31 declared streetview images were read; legal missingness was retained and read failures were zero.

The six specified regression files passed all 60 tests with UMAP enabled; there were no failures, skips, deselections, or exclusions. Before/after SHA256 and byte counts matched for all 32 protected production inputs, response-release files, indexes, and queues. The complete receipt, machine-readable reports, model checkpoints, and exports are under the run directory. This short engineering run does not establish model performance.

The causal response predictor uses pre-opening S/T/C inputs and the post-opening causal response labels in the strict Response Artifact. A separate visual asset pairing index was produced as a historical coverage and temporal audit; it is not a training dataset, target, or model input. Its archived counts and SHA256 values remain documented in [`visual_state_transition_pair_contract_2026-10-19.md`](../research/visual_state_transition_pair_contract_2026-10-19.md).

## S/T/C input vector and latest server evidence

The latest real-data server run is `/home/nas/lsr/MIT_summer/outputs/stc_server_acceptance/stc_acceptance_20261019T114650Z`. VIIRS and POI each completed a strict-production two-epoch training, checkpoint restore, and 64-event export; the six required regression files passed all 62 tests (0 failures, skips, deselections, or exclusions). The before/after hash comparison for 18 protected inputs and queues is empty. Current reports, checkpoints, exports, and the receipt are in that run directory.

The model input follows the S/T/C contract: the state branch contains the 15 lagged POI/VIIRS fields, all verified pre-opening streetviews, and satellite image sequence for event years -6 through -1; the intervention branch contains seven event fields; and the context branch contains AlphaEarth64, population lags, and city summaries (housing price fields are excluded). The multimodal pretreatment state inputs jointly produce the single response-sensitive embedding `z=f_theta(S)`. Separate encoders produce `t` and `b`, which condition `tau_hat=g_phi(z,T,C)` through the response predictor. Response-similarity alignment uses `z`; causal response labels supervise both alignment and masked, reliability-weighted response prediction. This is representation learning in the sense described by `研究思路.docx`: learning an embedding shaped by causal response, not learning to predict post-opening imagery.

For the streetview part of S, the active implementation loads all verified images captured before the opening month, computes each image's calendar-month gap to opening, compresses it as `d / (d + 12)`, and uses learned time-conditioned attention to aggregate the set into one fixed-size streetview vector. This vector enters the state encoder with POI/VIIRS history and satellite sequence to form `z`; T and C condition response prediction separately. In the latest server run, both outcome-family samples contained the same 31 legal pre-opening images from 16 events; gaps were 1–30 calendar months. The updated path passed the six-file regression and strict-production training/save/restore/export acceptance. Post-opening streetview is neither a predictor input nor a training target.
