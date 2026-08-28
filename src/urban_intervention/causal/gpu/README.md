# Python/GPU causal backend

This package is the default production implementation of the frozen Matching,
generalized synthetic control (GSC), and matrix-completion (MC) specifications.
The R estimators remain available as an explicit `r_reference` backend for
qualification and sensitivity checks; the normal label queue no longer invokes
R estimator processes.

## Estimator contracts

| Path | Python implementation | Formal uncertainty |
| --- | --- | --- |
| Matching control design | pre-treatment feature construction, robust city scaling, chunked GPU Mahalanobis search, static refinement and placebo/holdout gates | design-stage diagnostics |
| Matching ATT | `Matching::Match`-compatible ATT with replacement, bias correction and Abadie and Imbens analytic variance | analytic AI standard error |
| GSC | rolling rank CV, two-way interactive fixed effects, deterministic GPU SVD/EM, same-city and standardized all-city donors | 200-repetition two-stage parametric bootstrap (`auto`: empirical or AR errors) |
| MC | rolling lambda CV, two-way fixed effects and nuclear-norm completion | unit jackknife matching `fect::jackknifed` pseudo-values |

The Python panel builder also freezes the clean-pre, anticipation, excluded
opening-period and post-period calendars. Housing transaction support,
cross-city donor scope, convergence, tuning, inference counts and backend
version are carried into every label and manifest.

## Numerical and academic checks

- Formal work uses deterministic float64 PyTorch with TF32 disabled.
- Donor selection, city scaling, matching and tuning follow their documented
  pre-treatment information sets; treated post-period outcomes are masked.
- All-city GSC must pass a pre-only masked target-versus-20-donor placebo gate.
- The 3.77-million-unit all-city universe is reduced before any outcome is
  read by a fixed-seed stable-hash sample (default 50,000 donors). The cap and
  seed are part of the specification fingerprint; post-treatment outcomes
-  cannot affect selection. Report cap sensitivity for cross-city extensions.
- Production stops when fitting or formal inference is incomplete.
- R parity is used for software qualification. It does not select a second
  estimate after results are observed.
- Content-addressed tuning caches include the implementation version, tuning
  contract and determining panel cells. GSC rank CV can be reused when the
  complete control panel is identical. MC tuning is reused only when the full
  panel and treatment mask are identical.

## Execution model

The deployment design is one process per physical GPU. Four RTX 4090 cards run
four independent tasks concurrently; they are not exposed as one 96 GB device.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.is_available()"
nvidia-smi
```

Run a bounded formal estimator directly:

```bash
python scripts/causal_python/run_formal_estimator.py \
  --treatment-order 3301 --outcome-family population \
  --estimator gsc --donor-scope same_city --run-mode preview --device cuda:0
```

Run the production queues across four cards in an isolated server working copy
(do not reset the frozen active baseline):

```bash
export MIT_CAUSAL_GPU_IDS=0,1,2,3
python scripts/causal_r/run_parallel_production.py \
  --run-all --estimator-backend python_gpu \
  --qualification-receipt outputs/causal_gpu/formal_qualification.json \
  --gpu-ids 0,1,2,3 --shard-count 4 --workers 4
```

Generate that receipt only after the representative Matching/GSC/MC shadow
manifests have passed R/Python point-path and inference validation:

```bash
python scripts/causal_gpu/export_matching_qualification_set.py \
  --orders 507,509,530

python scripts/causal_gpu/run_shadow_queue.py \
  --formal-qualification --estimators matching,gsc,mc \
  --max-tasks-per-estimator 3 \
  --gpu-ids 0,1,2,3 --retry \
  --output-root outputs/complete_estimators/gpu_formal_qualification

python scripts/causal_gpu/audit_formal_qualification.py \
  --root outputs/complete_estimators/gpu_formal_qualification \
  --output outputs/causal_gpu/formal_qualification.json
```

The Matching references include both design-stage selections and final
fixed-control label paths. Production rejects a missing, ineligible, expired or
subsequently modified qualification source. Each shard performs the full
source audit once; estimator subprocesses verify the already-audited receipt
digest instead of re-hashing every qualification panel. The receipt records
the numerical environment that passed parity. The qualification run performs native GPU tuning,
200-repetition GSC bootstrap and MC unit-jackknife checks against R reference
labels; a manually changed manifest boolean does not satisfy the checks. Preview runs
do not require a receipt.

R is therefore retained only for the one-time reference/qualification stage.
Once an eligible receipt has been issued, the production Matching/GSC/MC queue
and the all-method event-study aggregation do not invoke R.

The R reference labels used for qualification use
`observation_window=1`. Legacy R moving-window standard errors combine
marginal errors without their covariance and are deliberately rejected as an
inference reference. Production Python windows are rebuilt from the complete
joint bootstrap/jackknife paths.

MC follows the full `fect` lambda grid, including the unregularized `lambda=0`
endpoint. Production requires a finite non-negative selected lambda and records
`mc_regularized` explicitly; a valid zero endpoint is not treated as a failed
fit.

Housing transaction support is configurable without changing the default:

```bash
python scripts/causal_r/run_parallel_production.py --run-all \
  --transaction-count-threshold 3 --run-mode preview
```

Use separate preview runs for threshold sensitivity before changing the main
threshold of one transaction. The threshold is applied to every housing
grid-month before Matching, GSC, or MC fitting, so low-support donor values
cannot enter the counterfactual model unnoticed.

Run the Python all-method event-study suite after labels exist:

```bash
python scripts/causal_python/run_all_method_event_study.py
```

Matching is estimated as a pooled matched-pair TWFE event study. GSC and MC
retain their estimator-specific effect paths and are pooled separately by
method and donor scope. Grid- and city-cluster pre-trend results are written as
diagnostic metadata. Label status follows the Response Artifact checks.

Phase 1 assigns control-design batches across `MIT_CAUSAL_GPU_IDS`; Phase 2
sets one `MIT_CAUSAL_DEVICE=cuda:N` for each shard. GSC inference dynamically
reduces its replicate batch size from the configured default according to free
device memory and rejects a panel that cannot fit even one replicate. The
launcher rejects more
Python label shards than GPUs because GSC bootstrap and MC jackknife jobs can
otherwise compete for the same 24 GB device.

Use `--estimator-backend r_reference` only for explicit reference runs.
`scripts/causal_gpu/run_shadow_queue.py` runs exported-contract parity tests;
it is not a production entry point.

## Validation boundary

Local unit, synthetic and representative real-panel checks establish that the
contracts run and reproduce audited fixtures. Before a full release, the
server run records CUDA device use, deterministic repeatability, representative
R/Python parity and wall-time/memory benchmarks. A bounded preview or test run
carries a non-production marker.
