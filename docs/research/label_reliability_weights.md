# label reliability weights (Rev3.3)

This document defines the label reliability weighting used when building
pretraining datasets from a published Response Artifact.  The scheme is
deliberately **ordinal and component-wise**: every component is stored so that
a later empirical calibration can replace one mapping without re-running any
causal estimation.

## Definition

For every label row \(i\):

\[
W_i = A_i \cdot \exp(-\gamma \, D_i) \cdot
      \left(\tilde p_i^{SE}\right)^{\alpha}
\]

where \(D_i\) is the mean of the **available** design-diagnostic percentiles
(see below), and after winsorization and normalization

\[
\mathrm{mean}_{\,i \in \text{eligible} \cap (m,o)} W_i = 1 .
\]

\(W_i\) is written to `response_weight`; components are written alongside it
(see **Stored fields**).

## Components

### 1. Eligibility \(A_i\) (hard boundaries only)

| Rule | Weight |
|---|---|
| label unavailable | 0 |
| zero pre-treatment anchors (`pre_observed_periods == 0`) | 0 |
| matched label with `max(holdout_rms_ratio, holdout_max_ratio) > 1` | 0 |
| otherwise | 1 |

These are identification or design-contract boundaries, not quality cutoffs.
No other rule may zero a label: poor quality is expressed through lower
weights, never through silent exclusion.

### 2. Design unreliability \(D_i\) (ordinal)

The pre-treatment design diagnostics are computed per method from data already
present in the Response Artifact:

| Component | Methods | Raw diagnostic |
|---|---|---|
| `p_trend` | GSC / MC | pre-treatment gap deviation \(\sqrt{\max(\texttt{pre\_rmspe}^2-\texttt{pre\_mean\_effect}^2,\,0)}\) |
| `p_holdout` | matched | `max(holdout_rms_ratio, holdout_max_ratio)` |
| `p_validation` | MC (when stored) | `mc_cv_mspe` (pre-treatment CV MSPE) |

Each raw diagnostic is converted to an empirical percentile computed on
**eligible labels only**, pooling upward until a stratum holds at least 30
labels: `(method, outcome) -> (method, family) -> method -> global` (a global
fallback below 30 labels uses the neutral 0.5).  \(D_i\) is the **row mean of
the available percentiles** (components that are not defined for a method are
simply absent, never imputed); rows without any diagnostic receive the neutral
percentile `0.5`.  The raw level percentile `p_level` is stored for QC only and
never enters the weight.  Ineligible labels are excluded from every reference
distribution (percentiles, precision normalization, winsorization cutoffs and
normalization means) and only receive \(W = 0\) at the end.

\(D_i\) is an **unreliability score**, not an estimated variance.  Calibrating
it to \(E[(\hat\tau^{pseudo})^2]\) is the W2/W3 roadmap below.

### 3. Tempered precision

For GSC/MC labels with a standard error, inverse sampling variance
\(1/SE^2\) is winsorized at the 1%/99% quantiles and normalized to mean one
within `(method, outcome)` over eligible model labels; the multiplier is
\((\tilde p^{SE})^{\alpha}\).  Matched labels have no sampling uncertainty and
receive exactly \(1\), so a missing uncertainty estimate never earns a bonus.

Separation of concerns: the representation loss already uses `response_se`
for pair-level statistical shrinkage (`response_alignment_loss`), so **training
builds the dataset with `alpha = 0`** (design-only weights).  The \(\alpha >
0\) tempering is retained for analysis, audit and the W-full ablation.

### 4. Aggregation and normalization

- raw weight: \(W^{raw}_i = A_i \exp(-\gamma D_i)(\tilde p_i^{SE})^\alpha\);
- winsorize at the 1%/99% percentiles of the eligible distribution within
  `(method, family)`; this is a numerical-robustness rule, so a poor-quality
  label keeps a strictly positive weight �?only eligibility failures are
  exactly zero;
- normalize to mean one within `(method, family)` over eligible labels;
- **training reliability is per label**: the loss sums
  \(\sum_{i,k} W_{i,k}^{design} L_{i,k}\) over tasks \(i\) and responses \(k\);
  a task-level aggregate is stored separately as `task_sampling_weight`
  (minimum over eligible labels) for sampling schemes only and must not be
  used as a substitute for the per-label reliability weight.

Default constants live in `pretraining_dataset.py`:
`DESIGN_GAMMA = 1.0`, `PRECISION_ALPHA = 0.5`,
`DESIGN_NEUTRAL_PERCENTILE = 0.5`, `WEIGHT_WINSOR_LOWER = 0.01`,
`WEIGHT_WINSOR_UPPER = 0.99`, `CALIBRATION_VERSION = "rev3.3-ordinal-2026-09"`.
`build_pretraining_dataset.py` exposes `--weight-gamma` and `--weight-alpha`
so the sensitivity variants (`alpha` = 0 / 0.5 / 1, `gamma` = 0 for the W0
equal-weight baseline) are exact rebuilds of the same pipeline.

## Stored fields

Per-label columns in `response_targets.parquet` (the raw release fields such
as `standard_error` remain available too):

| Field | Meaning |
|---|---|
| `eligible` | hard eligibility \(A_i\) |
| `p_trend` | trend-deviation percentile (GSC/MC) |
| `p_holdout` | matched holdout percentile |
| `p_validation` | MC CV-MSPE percentile |
| `p_level` | pre-treatment level percentile (QC only) |
| `design_unreliability` | \(D_i\), mean of available component percentiles |
| `design_weight` | \(\exp(-\gamma D_i)\) |
| `has_precision_weight` | label carries a usable SE |
| `precision_weight` | \((\tilde p^{SE})^\alpha\) or 1 |
| `weight_raw` | before winsorization/normalization |
| `response_weight` | final weight |
| `weight_stratum` | normalization stratum `(method, family)` |
| `calibration_version` | e.g. `rev3.3-ordinal-2026-09` |

Per-label: `response_targets.parquet:response_weight` is the training weight.
Per-treatment: `sample_index.parquet:task_sampling_weight` is an optional
aggregate for sampling only.  Release manifest: `response_weight` block with
the calibration version, parameters and eligible-label statistics.

## Training integration

The representation trainer consumes the per-label `response_weight` end to
end: `GridSample.response_weight` (27-dimensional, zero for ineligible or
missing labels) is stacked by `collate_samples`, carried by the labeled queue,
and passed to `total_loss`.  Without the column the loader falls back to unit
weights, so pre-Rev3 datasets stay usable.

- Response similarity uses pairwise design weights \(a_{ijk} = m_{ijk}\sqrt{W_{ik} W_{jk}}\):
  means, variances and covariance are weighted per family, and a pair is
  comparable only when at least two commonly observed cells have strictly
  positive pair weight and the weighted variance is non-degenerate.  Pairs
  failing that rule are removed from the InfoNCE denominator (an invalid pair
  is "no evidence", not "zero similarity").  The rule is invariant under
  rescaling all weights, and with unit weights it reduces to the historical
  unweighted correlation.
- The prediction head minimizes the per-family design-weighted MSE
  \(\sum m W \ell / \sum m W\); families whose observable cells all carry zero
  weight are skipped.  Sampling uncertainty is not re-used here.
- `response_se` keeps its pair-level statistical shrinkage in the alignment
  loss (`se_shrinkage`), so design quality and sampling precision enter through
  separate channels: design weights shape the targets, SE shrinks the
  statistical force of each pair.
- Training builds use `alpha = 0` (W-design); the W-full ablation runs the same
  code path with `alpha > 0` weights.  W0 (`gamma = 0, alpha = 0`) yields unit
  weights and therefore the historical unweighted comparison, apart from the
  prediction head now receiving design weights instead of `1/SE^2`.
- Retrieval metrics (`nn_corr@k` and the random baseline) intentionally remain
  unweighted so they stay comparable across weighting definitions; a
  `weighted_nn_corr` variant is a future add-on, not a current requirement.

## Versions and ablations

Naming is deliberately split so that a later change cannot be misattributed to
design weighting.  The repository has no "historical baseline" alias for W0.

| Version | Sample weight | Prediction head | Purpose |
|---|---|---|---|
| Historical | unit (no weight column) | `1/SE²` precision | pre-Rev3 trainer; naming reference only |
| W0 | unit (`gamma=0, alpha=0`) | design weights (all ones) | no design differentiation, new framework |
| **W-design** | design-only (`alpha=0`) | design weights | **main training version** |
| W-full | design × precision (`alpha=0.5`) | design weights | robustness / ablation |
| W2 | monotonic calibration of the design score on a stratified 300–600 task pseudo-treatment subsample | | robustness |
| W3 | full pseudo-treatment calibration \(V^{design} = E[(\hat\tau^{pseudo})^2 \mid X]\) with variance-additive combination \(1/(V^{design}+SE^2)\) | | paper method |

The representation alignment loss uses `response_se` pair shrinkage
(`se_shrinkage=True`, the default) in every version; the prediction head always
receives the design weights.  The only difference between Historical and W0 is
the prediction-head weighting, which is exactly why the two names must not be
used interchangeably in experiment tables.

W2/W3 require re-running the production estimators on pseudo interventions
(and, for the first-difference trend diagnostic, storing pre-treatment gap
paths in the task products).  Neither is required for W-design, whose stored
components are designed so the upgrade does not require re-running causal
estimation.

## Rationale

- Synthetic-control credibility rests on pre-treatment fit; the reliability of
  a counterfactual label is not captured by its sampling SE alone
  (Abadie, Diamond & Hainmueller 2010; Abadie 2021).
- Standardized-distribution conventions motivate using ordinal, scale-free
  comparisons rather than absolute cutoffs (Austin 2009; Rubin 2001).
- Pre-trend testing is diagnostic, not a gate (Roth 2022; Rambachan & Roth
  2023); accordingly no label is discarded for a weak pre-trend score.
- Inverse-variance combination with an added design-error term mirrors
  random-effects meta-analysis, but the multiplier form and the \(\alpha\)
  tempering keep sampling-error differences from dominating the loss.
