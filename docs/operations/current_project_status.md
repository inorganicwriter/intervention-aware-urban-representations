# Current Project Status

Updated: 2026-08-19

This page records only the active state needed to resume work. Detailed research
choices belong in the causal design and DDRs; dataset paths belong in
`data/active/catalog/datasets.yaml`.

## Fixed research assets

- The project contains 44 active cities and fixed 500m × 500m grids.
- Station identity and city-boundary issues have been reviewed and resolved.
- The immutable treatment list contains 5,048 station grids.
- The primary 1km spatial audit produced 3,771,800 initial non-treatment donor
  candidates before temporal and covariate gates.
- The fixed treatment list, donor universe, metadata, and queues live in
  `data/active/causal/`.

These counts define the spatial design only. They are not counts of successful
matches, GSC estimates, or final labels.

## Available covariate and outcome products

- POI grid-year products cover the 44 cities for 2012–2024.
- Population and annual VIIRS products are available under `data/active/curated/`.
  Annual VIIRS (`data/active/curated/viirs_annual_aggregated/`) is aggregated from
  the duplicate-free monthly VNP46A2 cache (2012–2024) by
  `scripts/analysis/build_viirs_annual_from_monthly.py`; the legacy per-city
  annual exports with duplicated geographic samples were removed.
- Monthly VIIRS source files are stored outside the repository at the directory
  configured by `MIT_VIIRS_RAW`; the causal pipeline materializes only required
  cache partitions.
- Canonical housing grid-month, grid-quarter, and grid-year panels exist under
  `data/active/panels/` and are documented in `docs/data/housing_price_panel.md`.
- Housing source provenance and admission rules are maintained under
  `data/active/catalog/` and `docs/data/`.
- Population panel (`data/active/curated/population/`) is rebuilt from clean sources:
  2010–2014 from GEE WorldPop (`WorldPop/GP/100m/pop`), 2015–2024 from WorldPop
  R2024B (Global 2) rasters aggregated to 500m grids, with a `source_version`
  column (see `scripts/collection/rebuild_population_panel.py`).  The 2014→2015
  transition is a product jump accepted by design.
- Sentinel-2 panel (`data/active/curated/sentinel2/`) is rebuilt from
  `reduceRegions` exports against the uploaded grid assets (one row per grid
  per year, duplicate-free), 2014–2024 (Landsat 8 for 2014–2017).
- Location features (`data/active/curated/location_features/`): per-grid distance to
  main centre / nearest subcentre / nearest centre, from the composite
  (POI + VIIRS + population) McMillen (2001) centre registry.
- Transit accessibility features (`data/active/causal/accessibility_features/`):
  per-treated-grid pre-treatment station distance, buffer counts, line
  diversity, network closeness (Wikidata P197 topology), and station
  attributes (transfer / terminal / new line / extension / same-month
  openings).  See `docs/research/transit_accessibility_method.md`.
- Transit snapshots (`data/active/causal/transit_snapshots/`): 482 per-opening-month
  pre-treatment transit feature files covering ALL grids (including donors)
  per city, so matching can align donor and target at the same pre-treatment
  time point (plan A on-demand snapshot cache).

## Active causal implementation

- Spatial donor construction is implemented in
  `src/urban_intervention/causal/spatial_donors.py`.
- Formal R estimators and the recoverable queue are under `scripts/causal_r/`.
- PanelMatch, Abadie–Imbens matching, and Xu GSC remain independent estimators.
- Control selection must use treatment-preceding information only.
- Matched-control failure routes an outcome task to GSC; GSC failure routes to
  MC, and only failure of all three paths produces an explicit skip reason.
- `control_design_queue.csv` has exactly one row per treated grid and freezes one
  physical control before any post-treatment outcome is read.
- Matching uses same-city donors first, then a pre-treatment-only standardized
  all-city fallback. Monthly housing and VIIRS use three 12-month blocks.
- GSC bootstrap uncertainty is normalized into one-grid label paths.
- The strict Response Artifact publisher now validates the complete expected
  label skeleton, task/control terminal states, label identities, provenance,
  hashes, and immutable release semantics.
- The pretraining-data publisher builds lagged pre-event multimodal features,
  modality masks, city-held-out splits, train-only normalization, and final
  training masks.
- Annual GSC now supports `annual_anticipation_years` for year-granularity
  anticipation sensitivity (default 0; set 1 to exclude opening_year-1).

## Event-study aggregation (parallel-trends validation)

GSC and MC estimator outputs keep negative-`event_time` rows (per-grid
pre-period counterfactual gaps) in `outputs/complete_estimators/staging/`;
these are the raw event-study coefficients. The label queue publishes only
post horizons, so parallel-trends validation is aggregated separately:

- `scripts/causal_r/run_event_study_aggregation.R` admits only production
  manifests (`run_mode=production`, `production_eligible=TRUE`) and writes
  `outputs/event_study/`. Annual and monthly results remain separate through
  the `frequency` field; the output includes per-frequency × family × outcome ×
  `event_time` series (mean/SD/SE/95% CI with per-period n), a grid-level joint
  zero-pre-trend test, a reading guide, and PNG/PDF/SVG event-study figures.
- Historical real-data check (2026-08-04, order 906 population MC task): mean pre-period
  label 0.0004 over 5 pre-period rows — consistent with the canary's
  near-zero pre-period claims; joint test deferred until ≥2 grids are labelled.
- The matched path contributes pre-trend evidence through the selection-stage
  holdout and donor-donor placebo q95 gates (see `grid_control_design_lib.R`),
  plus PanelMatch `placebo.test = TRUE`.

## Current queue status

Queue state after the deliberate reset for the 6-round routing (same-city
matching → same-city GSC → same-city MC → cross-city matching → cross-city
GSC → cross-city MC → skip):

- `control_design_queue.csv` (5,048 rows): 4,648 `pending`, 218 `matched`, 182 `gsc_pending`.
- `outcome_family_work_queue.csv` (20,192 rows): 20,192 `pending`.
- `counterfactual_work_queue.csv` (5,048 rows): 5,048 `pending`.

The demo and canary artifacts (releases, model inputs, representation
training outputs, demo figures) were removed from the tree; they were
non-production evidence and are not current queue rows.  Formal production
starts from the reset queues.

## Canary verification

Two-stage matching (outcome-history M=5 candidates + static location/transit
refinement) was wired into `grid_control_design_lib.R` on 2026-08-10; the
canary below reflects the **current** two-stage logic:

1. **Matched path**: queue canary on orders 906–915 (2026-08-10) matched 3/10
   same-city (hangzhou 910, chongqing 912, nanjing 915) with
   `control_selection_uses_post_outcome = FALSE`; the remaining 7 routed to
   `gsc_pending` (holdout/placebo gate rejections — expected with the new
   static covariates). `feature_balance.parquet` reports standardized gaps for
   the 9 static covariates (loc_*/transit_*) alongside the outcome lags.
   For orders 1–10 (Shanghai 2010 openings), the control queue records
   `gsc_pending` with reason `fewer_than_1_complete_pre_treatment_families`
   (VIIRS/POI start 2012); the corresponding family tasks skip directly with
   `no_complete_pre_treatment_families`.
2. **GSC path**: Xu (2017) gsynth smoke test on order 906 (Guangzhou 2015,
   population, annual, same-city) completed successfully — 49,329 donors,
   CV selected r=0, 20 parametric bootstrap replications, 8 label rows
   (5 pre-period + 3 post-period), all `label_available = TRUE` with finite
   standard errors. Pre-period labels near zero (-0.02 to +0.05); post-period
   population response +0.21 to +0.23 (log1p scale).

   A production-parameter queue canary was subsequently completed for the same
   order's population family. It used 49,329 pre-only donors, selected `r=0`
   by MSPE cross-validation, ran 200 parametric bootstrap replications, and
   produced three finite post-treatment labels. The Python task and R estimator
   manifests share one queue-generated `run_id`; both estimator-manifest and
   estimator-label hashes were revalidated.
3. **MC path**: matrix completion smoke execution selected a positive lambda by
   treatment-pre MSPE cross-validation and produced finite counterfactuals and
   jackknife uncertainty. End-to-end queue canary (2026-08-10, order 906 all
   families): matching failed the placebo gate → GSC → MC; poi/population/viirs
   reached `mc_labelled`, housing `skipped` (MC support failure) — the full
   6-round chain (match → GSC → MC → skip) ran atomically with task provenance.
4. **Skip path**: an outcome is skipped only after matching, same/all-city GSC,
   and same/all-city MC all fail their support or estimator gates.

Early-opening grids (2010–2014) mostly route to gsc_pending or skipped due to
insufficient pre-treatment data (VIIRS starts 2012, housing coverage varies by
city). Grids opening 2015+ have adequate support for both matching and GSC.

## Remaining steps before full production

The implementation, data layer, robustness checks and isolated estimator tests
are complete. Formal production execution is still outstanding:

1. Run the grid-control queue (5,048 rows, all `pending`) for same-city matching
   (6-round routing: same-city match -> GSC -> MC -> cross-city match -> GSC -> MC)
2. Run the family-level tasks (20,192 rows, all `pending`) for the same 5,048
   grids (matched labels + GSC + MC fallback)
3. Publish the strict Response Artifact and pretraining dataset, then train the
   representation model on the full production release


GSC in production mode uses 200 parametric bootstrap replications (vs 20 in
smoke test). MC uses fixed-lambda jackknife inference; its `nboots=200` field
is compatibility metadata, not 200 bootstrap replications. See
`scripts/causal_r/README.md` for commands. Every estimator invocation receives
a unique queue-generated `run_id`; stale, smoke-mode, non-production, or
incorrectly inferred artifacts are rejected before a task can be marked labelled.

## Representation-learning status

The intervention-conditioned representation model, loss, trainer, and
`urban-train-representation` entrypoint are implemented
(`src/urban_intervention/representation/`) and verified end-to-end on the
demo release (30 epochs on RTX 4060, no NaN, best val loss 0.188).

**Evaluation infrastructure** (`src/urban_intervention/representation/evaluation.py`):
every training run writes `evaluation_report.json` containing, for the
validation and test pools separately:

- `nn_corr@k` retrieval metrics overall and per outcome family, with the
  random-neighbour baseline (`baseline_corr` = mean pairwise response
  similarity);
- unit-resampling bootstrap CIs for `nn_corr@k`;
- a permutation test (response labels shuffled across units) giving a
  chance-level distribution and p-value;
- the same retrieval metrics for the raw z-scored features, i.e. the
  "no learned representation" baseline;
- a linear-probe transfer metric (per-cell ridge fitted on the train pool)
  comparing frozen embeddings against raw features on each target pool.

`urban-train-representation --seeds 1 2 3` runs one full training +
evaluation per seed into `seed_<n>/` subdirectories and writes a
`seed_summary.json` with mean/std across seeds. `urban-build-model-card`
renders a model card (`model_card.json`/`.md`) with architecture, training
config, history, per-family evaluation tables and explicit limitations;
`urban-run-ablation` runs a spec grid (e.g. `pred_weight` 0/0.5/1,
with/without images) and summarizes `ablation_summary.json`/`.md`. A short
CPU smoke on the demo release shows the report behaves as intended: validation `nn_corr@k` (0.065)
exceeds chance (0.020, p ≈ 0.02) but the raw-feature baseline (0.074) is not
yet beaten after 2 epochs, and the 5-unit test split has no measurable signal
(ratio 1.0) — the full production release is required for meaningful numbers.

Known methodological caveats for the full-production run:

- **Model selection**: the checkpoint is selected by validation total loss;
  retrieval gains should be reported with CI + permutation p-values, not raw
  point estimates.
- **Test set size**: a meaningful cross-city retrieval evaluation needs a
  much larger test pool than the demo's 5 units.
- **DINOv2**: image-path training requires torch.hub access to download
  `facebookresearch/dinov2` on first use (see README environment notes).

## Validity boundary

Outputs from earlier pooled DID, annual housing DID, prototype matching, disabled
runners, or pilot GSC runs are not final response labels. A result becomes usable
only after its design record, estimator object, quality diagnostics, and normalized
Response Artifact are all present and pass the frozen gates. Demo/partial releases
carry `production_eligible = FALSE` and cannot enter formal training.
