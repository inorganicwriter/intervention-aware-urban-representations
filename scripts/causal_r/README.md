# Complete causal estimators

> Production backend update: Matching control design, Matching labels, GSC and
> MC now run through the Python/PyTorch implementation by default. The R
> runners documented below remain the frozen reference implementation for
> parity and sensitivity audits. Select them explicitly with
> `--estimator-backend r_reference`.

Four-GPU production entry point:

```text
python scripts/causal_r/run_parallel_production.py --run-all --reset-queues \
  --estimator-backend python_gpu --gpu-ids 0,1,2,3 \
  --qualification-receipt outputs/causal_gpu/formal_qualification.json \
  --shard-count 4 --workers 4
```

Python production is fail-closed until the shadow queue has run with
`--formal-qualification`, generated passing Matching/GSC/MC point and inference
comparisons, and `audit_formal_qualification.py` has issued a receipt.
Matching qualification compares both control-design outputs and final
fixed-control label paths. The qualification queue samples three tasks per
estimator by default; use `--max-tasks-per-estimator` to change that count.
Preview runs and explicit `r_reference` runs do not require that receipt.
GSC/MC inference qualification uses unwindowed (`--window 1`) R reference
labels; formal Python moving windows are reconstructed from joint replicate
paths rather than marginal standard errors.
After the receipt is issued, the default production queue and pooled
event-study workflow are Python-only; R remains an explicit audit/reference
backend rather than a per-task runtime dependency.

The production estimators are specified in
`docs/research/decisions/DDR-003_complete_published_estimators.md`.
They are independent estimators; they are not components of a single hybrid
algorithm.

The causal-response label and routing contract is frozen in
`docs/research/decisions/DDR-004_causal_response_labels.md`. The opening month
is excluded, the next full month is event time 1, and the main anticipation
window is six months (0/12-month sensitivity specifications).

Runtime:

```powershell
$env:R_LIBS_USER = (Resolve-Path '.r-lib').Path
$rscript = (Get-Command Rscript).Source
```

Server runners may override local paths without editing code:

```text
MIT_RSCRIPT   Rscript executable
MIT_R_LIB     project R package library
MIT_VIIRS_RAW raw monthly VIIRS root
```

The immutable treatment list contains 5,048 station grids. Reset and rebuild
the read-only estimator inputs with:

```powershell
& $rscript scripts/causal_r/reset_counterfactual_queues.R
& $rscript scripts/causal_r/build_formal_matching_inputs.R
& $rscript scripts/causal_r/audit_formal_target_support.R
```

## Imai-Kim-Wang PanelMatch

This runner calls the official `PanelData()`, `PanelMatch()`,
`get_covariate_balance()`, `get_set_treatment_effects()`, and
`PanelEstimate()` workflow. Production settings are M=1 full-covariance
Mahalanobis refinement, placebo estimation, and 1,000 bootstrap iterations.

```powershell
& $rscript scripts/causal_r/run_complete_panelmatch.R xiamen 2019 population population+viirs annual
```

The production runner does not cap the same-city donor pool. A benchmark with
16,514 donors exceeded ten minutes on the current machine. The
`test_real_panelmatch_gate.R` 300-donor fixture is an integration test and is
excluded from formal estimates.

## Abadie-Imbens matching

This runner calls `Matching::Match()` with ATT, M=1, replacement,
Mahalanobis distance, bias adjustment, and `Var.calc=1`. The outcome is a
pre-specified pre/post change. Ordinary bootstrap is not used for estimator
inference.

```powershell
& $rscript scripts/causal_r/run_complete_abadie_imbens.R xiamen 2019 population 1 poi+population+viirs annual
```

The runner stops instead of silently accepting the package's downgrade when
the treated cohort is too small to identify the bias-adjustment regression.
Zero-variance covariates are removed and recorded because the Mahalanobis
covariance matrix is otherwise undefined.

Control identities are now selected with `Y=NULL` before any post-treatment
outcome is read. The nearest clean pre-period is held out, and a deterministic
donor-donor pseudo-treatment calibration supplies the q95 training-distance
and holdout-path gates. A failed gate routes the task to GSC; it is not written
as a credible matched label.

## Xu generalized synthetic control

This runner implements the Xu GSC specification through `fect::fect()` and
preserves a gsynth-compatible result class for downstream readers. It uses
two-way effects, MSPE cross-validation over `r=0:5`, `min.T0=5`, and 200
parametric bootstrap replications. Formal manifests record both the actual
`fect` backend/version and the `gsynth` compatibility version.

```powershell
& $rscript scripts/causal_r/run_complete_xu_gsc.R xiamen 2019 population population+viirs annual
```

Both housing and VIIRS can use monthly mode. VIIRS is loaded from persistent
city-month cache partitions, preserves finite negative radiance, and uses an
`asinh` transform. GSC donor admission is based only on complete clean
pre-treatment paths; normalized output is keyed by treatment order and event
time.

For the 3,771,800-unit all-city donor universe, Python GSC applies a
pre-outcome stable-hash sample (default 50,000, seed 20260823) before reading
outcome panels. The cap and seed are included in the specification fingerprint;
same-city GSC is uncapped, and cross-city cap sensitivity remains a required
robustness check.

The runner separates factor selection from inference: cross-validation may use
eight cores, while the 200-replication bootstrap is sequential to avoid copying
multi-gigabyte panels to every worker. A real-data smoke test must be explicitly
requested and is isolated from production outputs:

```powershell
& $rscript scripts/causal_r/run_complete_xu_gsc.R chengdu 2017 population auto annual 6 1415 same_city smoke_test
```

Smoke mode uses 20 bootstrap replications and writes a manifest with
`production_eligible=FALSE`. Omitting the final argument retains the formal 200
replications.

## Matrix completion fallback

When both same-city and all-city GSC fail, the queue enters the recoverable
`mc_pending` and `mc_running` states. The MC runner uses the official
`fect(method="mc")` implementation, two-way effects, MSPE cross-validation
over 20 lambda candidates, `min.T0=1`, `cv.nobs=1`, and zero CV donut/buffer,
using a pre-only CV selection stage followed by fixed-lambda jackknife
inference. The formal `nboots=200` setting is retained for run compatibility,
but it is not interpreted as 200 bootstrap replications under jackknife.
Treated post-period outcomes are
masked with a pre-treatment-only value before lambda construction, CV, and
  fitting. A successful production manifest must record `fitted_method=mc`, a
  finite non-negative `selected_lambda`, and finite CV MSPE. `lambda=0` is the
  unregularized endpoint of the frozen `fect` grid and is retained with an
  explicit `mc_regularized=FALSE` marker rather than being misclassified as a
  failed fit.

```powershell
& $rscript scripts/causal_r/run_complete_mc.R xiamen 2019 population auto annual 6 2346 same_city smoke_test
```

Smoke mode uses 20 compatibility repetitions with jackknife inference and isolated output signatures. A
multi-outcome family may retain successful outcomes while recording structured
failures for unavailable outcomes. MC returns a donor-based counterfactual
path, not one physical `control_grid_id`.

## Frozen grid-level control design

Control selection runs before the outcome-family queue and creates exactly one
design row for each of the 5,048 treated grids. Unlike the annual outcome
estimators, the control design uses monthly VIIRS in three twelve-month
pre-treatment blocks. Production matching therefore requires the complete
44-city, 2012-01–2024-12 monthly cache before the first grid is processed:

```powershell
$env:MIT_VIIRS_RAW = 'E:\Data\MIT_Summer_VIIRS'
conda run -n mit python scripts/causal_r/run_grid_control_design_queue.py --prepare-viirs-cache-only
```

The command is resumable and verifies all 6,864 Parquet+audit pairs. A missing
pair is a cache-contract failure; production does not convert it into family
unavailability or silently remove a city from the all-city donor pool.

After the cache contract passes:

```powershell
conda run -n mit python scripts/causal_r/run_grid_control_design_queue.py --start-order 1 --max-units 10 --dry-run
```

Remove `--dry-run` only for a bounded canary. The control-design queue records a
same-city attempt and, when it fails, an all-city pre-treatment-standardized
attempt; failed placebo/holdout gates are recorded as `gsc_pending`. Use bounded
batches (for example 25–50 rows) so durable per-grid records are synchronized
into the CSV queue. The current CSV orchestrator is single-writer, so run only
one control-design queue process against a given queue.

## Transactional production queue

The orchestrator calls the Python/GPU estimators by default, one treatment grid
and outcome family at a time. It updates the live CSV queue atomically, resumes
interrupted tasks, routes matching failures to Xu GSC and then MC, and binds
each accepted estimator manifest to a unique queue-generated `run_id`:

```powershell
conda run -n mit python scripts/causal_python/run_causal_label_queue.py --start-order 1 --max-tasks 1 --dry-run
```

<!-- GPU 迁移前入口 scripts/causal_r/run_causal_label_queue.py 仍由兼容包装器支持。 -->

The formal queues contain 5,048 unique treated grids and 20,192 family-level
tasks (`5,048 × 4` outcome families). One production GSC population canary for
order 906 is terminal; the larger task count does not represent additional
treatment units. Remove `--dry-run` only after the full gate suite passes.
Monthly VIIRS tasks
materialize just the required missing raw partitions with
`scripts/collection/ensure_viirs_monthly_cache.py`.

## Response Artifact and model inputs

Only after all four outcome-family rows for every treatment are terminal:

```powershell
conda run -n mit python scripts/causal_r/build_response_artifact.py --release-id production_YYYYMMDD
conda run -n mit python scripts/causal_r/build_pretraining_dataset.py --response-release data/active/causal/releases/production_YYYYMMDD --dataset-id production_YYYYMMDD
```

The first command refuses unfinished queues, smoke products, key violations,
post-treatment selection leakage, missing task products, or label-formula
violations. It writes an immutable Response Artifact and source hashes. The
second command uses only lagged pre-opening features, assigns whole cities to
one split, fits normalization on training cities only, and combines response
and feature masks.

`--allow-partial` is available on both commands for canary and unit-test
fixtures. Partial products are marked non-production and must not be read by
formal training.

## Event-study aggregation (parallel-trends validation)

GSC and MC task outputs keep negative `event_time` rows (pre-period
counterfactual gaps); these are the per-grid event-study coefficients. Their
monthly values use actual calendar offsets: under the main specification the
clean pre-period is `-42:-7`, while anticipation `-6:-1` is excluded. The
label queue publishes only post horizons, so the event-study aggregator reads
the estimator staging directly:

```powershell
& $rscript scripts/causal_r/run_event_study_aggregation.R
```

The R-free production equivalent covers all three selected methods:

```powershell
python scripts/causal_python/run_all_method_event_study.py
```

It writes method/scope-specific GSC and MC pooled paths plus Matching TWFE
results, grid- and city-clustered pre-trend diagnostics, robustly scaled PNG/PDF
figures, and `effect_paths_with_pretrend.parquet`. Pre-trend flags are
diagnostic metadata and are not an automatic sample-selection rule.

Only outputs with production manifests (`run_mode=production`,
`production_eligible=TRUE`) are admitted; smoke/canary artifacts are excluded.
It writes `outputs/event_study/`:

- `event_study_series.csv` — mean/SD/SE/95% CI per frequency × family × outcome ×
  `event_time` (pre and post), with per-period unit counts;
- `event_study_joint_tests.csv` — grid-level joint zero-pre-trend test on the
  latest five clean pre-periods per grid (one-sample t-test on per-grid mean
  pre-period labels; requires ≥2 grids);
- `event_study_report.md` — reading guide: near-zero pre-period means are
  consistent with, but do not establish, parallel trends; the joint test is
  read together with the design-stage placebo evidence;
- `figures/event_study_{frequency}__{family}__{outcome}.{png,pdf,svg}` —
  mean + 95% CI event-study plots, with family-level and pooled overview figures
  written alongside them.

## Verification

```powershell
& $rscript tests/causal_r/test_complete_estimators.R
& $rscript tests/causal_r/test_monthly_input_builder.R
& $rscript tests/causal_r/test_preonly_matching_contract.R
& $rscript tests/causal_r/test_preonly_placebo_quality.R
& $rscript tests/causal_r/test_monthly_viirs_reader.R
& $rscript tests/causal_r/test_causal_label_contract.R
& $rscript tests/causal_r/test_gsc_label_mapping.R
& $rscript tests/causal_r/test_real_panelmatch_gate.R
& $rscript tests/causal_r/test_event_study_aggregation.R
& $rscript tests/causal_r/verify_complete_implementation.R
conda run -n mit python -m pytest tests/unit/test_response_artifact.py tests/unit/test_pretraining_dataset.py -q
```

All estimator outputs go to `outputs/complete_estimators/staging`. None of the
three standalone estimator runners writes the formal treatment or outcome-family
queues. Superseded prototype queue runners have been removed; production routing
is owned only by `run_grid_control_design_queue.py` and
`scripts/causal_python/run_causal_label_queue.py`.
