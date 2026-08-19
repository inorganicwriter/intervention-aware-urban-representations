# Sample-Run Execution Plan (v2, final)

Date: 2026-08-14
Status: approved for execution
Scope: code fixes, infrastructure, housing module, event-study extension, and a 400-grid representative sample run

## 1. Frozen specification decisions (from prior reviews)

### 1.1 Price measures and city blocks

- Matching (control design): **all 44 cities, unified `median` price measure, monthly blocks** — unchanged from current main spec. Cross-city comparability requires one price column for every treated/donor pair, and `median` is the only measure defined in all 44 cities.
- Housing-family labels:
  - Lianjia 22 cities (2,504 treated grids): **two label sets — hedonic-adjusted (main) and median (counterpart)**
  - Non-Lianjia 22 cities (2,544 treated grids): one set — **median**
- Other outcome families (VIIRS monthly, POI/population annual): unchanged, all 44 cities.
- Matching results (who is whose control) are computed **once** and shared by both price measures; GSC/MC are re-estimated per measure (weights depend on the outcome path).

### 1.2 Event-period observation window

- The monthly-vs-quarterly debate is reframed as an explicit **observation window width W**:
  - W=1: event period t = the single calendar month t (coverage ~10% of treated grids per period)
  - **W=3: main specification** — event period t = mean over months 3(t-1)+1 .. 3t (coverage ~25%)
  - W=6: robustness — mean over 6-month windows (coverage ~40%)
- Window applies only to **housing-family post-treatment observation**; baseline remains the 12-month pre-opening block; matching pre-treatment lags are unaffected.
- This is a formal change to the frozen DDR: "event period 1 = first complete natural month after opening" becomes "event period 1 = mean of the first W natural months after opening" for the housing family (W=3 main, W=1/6 robustness).
- Hedonic time FE: year×quarter for the main W=3 specification; year×month for the W=1 robustness.

### 1.3 Hedonic adjustment (sensitivity-to-main for Lianjia 22 cities)

- Transaction-level, per-city OLS: `log(unit_price_cny_m2) ~ area + area² + age + age²(missing-dummy) + bedrooms + floor-group + orientation + decoration + building-type + elevator + community FE + time FE`
- Quality-adjusted price per transaction = price / exp(Xβ̂); aggregated to grid×period as the median of adjusted prices, keeping `n_transactions`.
- Minimum-count rule: window with ≥1 transaction enters (per count-distribution review); reports also give ≥2/≥3 subsample sensitivity.
- Composition check (main report): event-study style paths of `n_transactions`, mean area, mean building age around opening.

### 1.4 Cross-city handling

- Cross-city labels are **kept and marked, never discarded**: `donor_scope` + `quality_grade` already tag them; new `main_spec` column (=1 iff `donor_scope == "same_city"`) added by the release builder.
- Training data keeps **all scopes by default** (unchanged behavior); `--scope-view {all,same_city,cross_city}` on the pre-training dataset builder enables explicit sensitivity/ablation views.
- Reports use three views: same-city main table, cross-city companion table, merged table.

### 1.5 Quality-grade ordering (B2)

- Reorder `response_artifact.py` grading so **any same-city path ranks above any cross-city path**: matched_same > gsc_same > mc_same > matched_cross > gsc_cross > mc_cross.

### 1.6 "Previously covered" grid condition (C1)

- Data shows only 22/5,048 treated grids (0.4%) had an existing station within 500 m at opening: no matching treatment, no stratification; one transparency row in the SMD diagnostics table only.

### 1.7 Station attributes (treatment-level)

- Grid-level covariates (location 9 + transit 6, already in matching): unchanged, now including a transparency row for C1.
- Treatment-level attributes (transfer / new-line / extension / terminal / same-month openings, from `outputs/causal_labels/station_attributes/`): cannot enter treated-control matching (no control counterpart); used for (a) event-study stratification, (b) robustness subsamples, (c) representation-model conditioning tokens.

## 2. Work packages

### Phase 1 — Code fixes (5)

| # | File | Change | Verification |
|---|---|---|---|
| F1 | `scripts/causal_r/run_event_study_matching.R:82` | Fix data.table self-comparison (`grid_id == grid_id` always TRUE): resolve the function argument instead of the column | R minimal repro passes |
| F2 | `scripts/causal_r/run_event_study_matching.R:222` | Sun-Abraham branch: drop the `!is.na(treatment_time)` filter that removed all never-treated controls | control rows > 0 after filter |
| F3 | `scripts/causal_r/fixed_control_label_lib.R:173-195` | Annual regression window: align to a calendar-year skeleton (mirror the monthly `calendar_frame` pattern) so missing years cannot shift positions | gap-scenario diagnostics no longer shift |
| F4 | `scripts/causal_r/run_cross_city_control_design.R:33` + `grid_control_design_lib.R:574` | Cross-city design must not overwrite the same-city durable `control_record.csv` on failure (write `cross_city_attempt.csv` instead) | simulated failure keeps same-city record |
| F5 | `src/urban_intervention/causal/response_artifact.py:574-602` | Same-city-first grade ordering (see 1.5) | unit tests updated and passing |

### Phase 2 — Infrastructure (4 + B1)

| # | File | Change |
|---|---|---|
| I1 | `scripts/causal_r/run_grid_control_design_queue.py` | `--orders` (comma-separated, mutually exclusive with start/end range) |
| I2 | `scripts/causal_r/run_causal_label_queue.py` | `--orders` (same) |
| I3 | new `scripts/analysis/select_representative_sample.py` | City×opening-year stratified, fixed-seed sample of **400** grids; writes `outputs/causal_labels/representative_sample_400.csv` (order + stratum weights + coverage report) |
| I4 | new `scripts/analysis/summarize_causal_labels.py` | (a) success/failure breakdown (research vs data-truncation vs code reasons); (b) label distributions per family (mean/median/quartiles/Tukey outliers, asinh/log note); (c) three scope views (1.4) |
| B1 | `build_response_artifact.py` / `build_pretraining_dataset.py` / I4 | `main_spec` marker column; `--scope-view` parameter (default all); three-view reports |

### Phase 3 — Housing module (4)

| # | File | Change |
|---|---|---|
| H1 | new `scripts/labels/build_housing_hedonic.py` | Per-city transaction hedonic → adjusted prices → windowed panel (W=1/3/6, median + n_transactions, count-distribution report) → `outputs/causal_labels/housing_hedonic/` |
| H2 | new `scripts/analysis/audit_housing_composition.py` | Composition event studies (`n_transactions`, mean area, mean age) around opening; pooled plots + jump detection; main report item |
| H3 | — | Minimum-count threshold set from H1 distribution (default ≥1 per window; ≥2/≥3 subsample sensitivity in reports) |
| H4 | `scripts/causal_r/complete_estimators_lib.R` + `run_fixed_control_labels.R` + GSC/MC runners | Housing outcome read supports `price_measure` (median/hedonic) and `window` (1/3/6); defaults unchanged (median monthly) |

### Phase 4 — Event-study extension (2)

| # | File | Change |
|---|---|---|
| E1 | `scripts/causal_r/run_event_study_matching.R` | (a) extend to VIIRS (monthly read layer exists); (b) attribute stratification from `outputs/causal_labels/station_attributes/` (transfer / new-line / terminal / same-month-opening groups) |
| E2 | `scripts/causal_r/run_event_study_aggregation.R` + `event_study_lib.R` | Same stratification parameters; per-stratum sample sizes annotated (small strata: trends only, power caveat) |

### Phase 5 — Sample run (400 grids)

1. I3 → sample list
2. Canary: 10 units (control design + label queue dry-run) to measure per-task cost, extrapolate duration
3. `run_grid_control_design_queue.py --orders <sample>` (workers 4-6, dry-run first)
4. `run_causal_label_queue.py --orders <sample>` (dry-run first, then phase=all, background + log monitoring)
5. `build_response_artifact.py --allow-partial` → sample release (non-production marker)
6. I4 summary → three deliverables: (a) event studies (GSC/MC 4 families + matching housing/VIIRS + attribute strata + composition checks H2), (b) success/failure reasons, (c) label distributions
7. Full report to `outputs/causal_labels/sample_report_YYYYMMDD/`

Estimated: ~1,600 family tasks, hours to a day with 4-8 workers.

### Phase 6 — Training-side fixes (after Phase 5, may run in parallel)

| # | File | Change |
|---|---|---|
| T1 | `src/urban_intervention/representation/baselines.py:163` | DINOv2 baseline: apply the same pixel normalization as `ImageEncoder.preprocess` |
| T2 | `trainer.py` (config_sha256 without timestamps/absolute paths; `_run_epoch` k from `eval_k`), `pretraining_dataset.py` (grade_rank covers GSC/minimal grades; explicit normalize fallback) | four small fixes + `--conditioning` extension to station-attribute tokens |

## 3. Execution order and dependencies

```
Phase 1 (fixes) → Phase 2 (infrastructure) → Phase 3 (housing) → Phase 4 (event study)
   ↓ all complete
Phase 5 sample run (canary first) ──┐
Phase 6 training fixes (parallel) ──┘
```

## 4. Deliverables

1. 400-grid representative sample list + stratification report
2. Sample-level Response Artifact (with `quality_grade`, `donor_scope`, `main_spec` tags)
3. Three deliverables: event-study figures (4 families × strata), success/failure reasons, label distributions
4. Housing module: hedonic panel (22 cities × W=1/3/6), composition-check figures, count-distribution report
5. All fixes verified by tests

## 5. Explicitly out of scope

- Any specification change beyond the two approved ones (hedonic main for Lianjia 22 cities; W=3 window main).
- Full 5,048-grid formal run (waits for sample review).
- Any write to `data/active` (read-only; all derived products go to `outputs/`; formal data-layer entry requires catalog registration).

## 6. Risks and boundaries

- Stratified event studies with small strata (e.g., ~46 transfer stations in a 400 sample): trends only, power caveat annotated.
- Early-opening Lianjia cities (pre-2012) have thinner pre-treatment lags: annotated in reports.
- Zero-coverage cities (e.g., Dongguan housing) produce NA naturally.
- Late openings (2023+) hit VIIRS/panel truncation: reported as a dedicated "data-truncation" failure class.
- All new products ship diagnostics (coverage, sample sizes, parameters) for auditability.

## 7. Implementation status (updated 2026-08-14)

### Done and verified

- **Phase 1 (F1-F5)**: all five fixes implemented and verified.
  - F1/F2: R repro tests pass; the Sun-Abraham branch was additionally found to
    have mixed calendar scales (absolute treat month vs relative event time),
    which made `sunab()` error out every time — the branch had never produced
    output; now uses `sunab(treatment_time, calendar_month)` with NA-treated
    never-treated controls retained.
  - F3: annual calendar-skeleton alignment verified (gap scenario yields NA,
    no silent shift).
  - F4: cross-city failure writes `cross_city_attempt.csv` and never touches
    the same-city `control_record.csv`; the Python queue reads the attempt
    file for failure reasons.
  - F5: same-city-first grade ordering; `grade_rank` in the pre-training
    dataset builder mirrors the new ordering.
- **Phase 2 (I1-I4, B1)**: `--orders` on both queue runners (mutually
  exclusive with ranges); representative 400-grid sample written
  (`outputs/causal_labels/representative_sample_400.csv`; 308 strata all
  covered, 44 cities x 16 opening years, quota-population corr 0.84);
  three-view summary script; `main_spec` marker column in the response
  artifact; `--scope-view` on the pre-training dataset builder (default
  `all` — cross-city labels are kept and marked, never discarded).
- **Phase 3 (H1-H4)**: hedonic panel for the 22 Lianjia cities
  (`outputs/causal_labels/housing_hedonic/`, grid-month median adjusted
  price + n_transactions; beijing R2 0.62, shenzhen 0.92); count
  distribution reported (grid-month median 1-3 transactions; >=2 in
  26-70% of populated grid-months); composition audit
  (`outputs/causal_labels/housing_composition/`) found a **+9.7% mean
  building-age jump** around openings — raw median prices understate effects
  and hedonic adjustment is warranted; R housing read layer supports
  `price_measure` (median/hedonic) and `window` (1/3/6) through
  fixed-control labels, GSC and MC runners, with defaults unchanged; the
  W=3 hedonic path was end-to-end verified (6/6 labels on a transaction-rich
  Beijing grid).
- **Phase 4 (E1/E2)**: matching event study extended to VIIRS (per-city
  partition window cache) and to treatment-attribute stratification
  (transfer / new-line / existing-line / terminal / batch size, per-stratum
  TWFE + Wald + sample-size annotation); aggregation script and library
  accept `--stratum-attribute/--stratum-values`.
- **Phase 6 (T1/T2)**: DINOv2 image baseline now applies the same
  ImageNet normalization as the trained encoder; `config_sha256` excludes
  created_utc/absolute path/device; `_run_epoch` uses `eval_k`;
  `normalize_from_train` emits NaN (not raw-scale values) for features with
  no training-city observations; conditioning extended to
  `station` / `opening_year_station` tokens (4-bit station attribute code),
  wired through dataset/trainer/model/export with a guard that refuses
  station-conditioned exports without `--station-attributes`.

### Self-review fixes (found while re-checking the modifications)

- `window_mean` used `(h - window + 1):h` without the lower bound clamp, so
  W=3 horizon 1 would read the pre-opening month; now `max(1, h-window+1):h`
  (verified: W=3 horizons map to 1 / 1-3 / 4-6 / 10-12 / 16-18 / 22-24).
- `read_viirs_window` built its month sequence from a float day offset,
  which turns the IDate sequence numeric and breaks `seq(by="month")`; now
  uses integral 31-day bounds.
- `export.py` did not pass `station_attributes_path` to the dataset, so
  station-conditioned exports would silently use zero tokens; now guarded
  and wired through the CLI.
- GSC/MC manifests now record `price_measure`.

### Remaining

- Phase 5: sample run (400 grids) — canary first, then control-design and
  label queues; deferred by decision.

### Follow-up fixes (2026-08-14, canary-driven)

- **Family support pre-screen**: `run_causal_label_queue.py` now caches each
  (city, family) panel's observed grid set and skips tasks whose grid has no
  observation anywhere in the family panel (`family_no_observed_support`)
  before invoking R — verified 1.4 s vs 209 s on a real no-data task.  A
  full-panel scan found 1,942 such tasks (1,941 housing + 1 poi), i.e. ~14
  core-hours at the old cost.  VIIRS is not pre-screened (partitions cover
  every grid).
- **MC uncertainty root cause and fix**: fect's unit-level bootstrap is
  structurally broken for the one-treated-unit grid design — every resample
  drops the treated unit with probability ~1/e, so `est.att` comes back with
  `count=1` and all-NA S.E. (verified on real data).  `parametric` is not
  available for method "mc"; **`jackknife` produces finite S.E./CI/p** and
  is now the MC inference method (`spec$mc$inference = "jackknife"`); the
  label queue's manifest check accepts both bootstrap and jackknife.
  Runtime cost rises (a 4-outcome annual family took >1 h at 8 cores);
  full-run budgets must account for it.
- **VIIRS cache ceiling**: `MIT_VIIRS_RAW` is unset and partitions end at
  2024-12; 2023+ openings will fail their viirs family as a data-truncation
  class unless the server provides 2025 raw monthly data and sets the
  variable (the `ensure_viirs` materialisation path already supports it).

### Verification round 2 (2026-08-15, real-run gaps closed)

- **Event-study script now runs end-to-end on real data** (13 matched
  Beijing pairs, VIIRS family): TWFE + city-clustered robustness + joint
  pre-trend Wald + Sun-Abraham + 6 attribute strata + spillover all write
  outputs.  Four latent bugs were found and fixed along the way:
  `treated_units` dropped the `treatment_order` column; three figure merges
  keyed `by="term"` on a column missing from the subset; the sunab
  coefficient parser handled neither the `:cohort::` suffix (multi-cohort)
  nor its absence (single-cohort); the attribute merge clashed with the
  panel's own `same_month_openings` column (renamed to
  `attr_same_month_openings`).  Sparse-panel degradation added: TWFE and
  the figure now degrade to notes instead of aborting when event dummies
  are collinear with unit FE or the clustered vcov is singular.
- **GSC hedonic path**: `run_complete_xu_gsc.R --price-measure hedonic`
  parses, reads the hedonic panel and calls gsynth identically to the
  median path (both fail identically on a data-sparse unit; parameter chain
  verified).
- **Station conditioning + export**: a synthetic 2-epoch training with
  `--conditioning opening_year_station` converges and checkpoints; export
  refuses station-conditioned checkpoints without `--station-attributes`
  and succeeds with it.
- **`--orders` truncation fixed**: an explicit order list is no longer cut
  by the default `--max-units/--max-tasks` of 1.
- Queues reset to pending afterwards; all verification artifacts cleaned.
