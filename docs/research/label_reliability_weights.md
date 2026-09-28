# Label reliability weights (Rev3)

This document defines the label reliability weighting used when building
pretraining datasets from a published Response Artifact.  The scheme is
deliberately **ordinal and component-wise**: every quantity is stored so that a
later empirical calibration can replace one mapping without re-running any
causal estimation.

## Definition

For every label row \(i\):

\[
W_i = A_i \cdot \exp(-\gamma \, P_i^{design}) \cdot
      \left(\tilde p_i^{SE}\right)^{\alpha}
\]

with, after winsorization and normalization,

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

### 2. Design unreliability \(P_i^{design}\) (ordinal)

The pre-treatment design diagnostic is estimated per method from data already
present in the Response Artifact:

- **matched** (`frozen_matched_change_*`): `max(holdout_rms_ratio, holdout_max_ratio)`;
- **GSC / MC**: the pre-treatment gap deviation
  \(\sqrt{\max(\texttt{pre\_rmspe}^2-\texttt{pre\_mean\_effect}^2,\,0)}\),
  i.e. the standard deviation of the pre-treatment gaps.

The raw distance is converted to an empirical percentile within
`(method, outcome)` (0 = most reliable).  Strata with fewer than two labels
fall back to `(method, family)`; labels with missing diagnostics receive the
neutral percentile `0.5` and are flagged by the stored components.

`P^{design}` is an **unreliability score**, not an estimated variance.
Calibrating it to \(E[(\hat\tau^{pseudo})^2]\) is the W2/W3 roadmap below.

### 3. Tempered precision

For GSC/MC labels with a standard error, inverse sampling variance
\(1/SE^2\) is winsorized at the 1%/99% quantiles and normalized to mean one
within `(method, outcome)`; the multiplier is
\((\tilde p^{SE})^{\alpha}\) with default \(\alpha = 0.5\).  Matched labels
have no sampling uncertainty and receive exactly \(1\), so a missing
uncertainty estimate never earns a bonus.  \(\alpha \in \{0, 0.5, 1\}\) is the
pre-specified sensitivity set (`0` = no precision adjustment, `1` = full
inverse variance).

### 4. Aggregation and normalization

- raw weight: \(W^{raw}_i = A_i \exp(-\gamma P_i^{design})(\tilde p_i^{SE})^\alpha\);
- winsorize at the 99th percentile within `(method, family)`;
- normalize to mean one within `(method, family)` over eligible labels;
- the per-treatment weight (`sample_index.parquet`) is the conservative
  minimum over the treatment's available labels.

Default constants live in `pretraining_dataset.py`:
`DESIGN_GAMMA = 1.0`, `PRECISION_ALPHA = 0.5`,
`DESIGN_NEUTRAL_PERCENTILE = 0.5`, `WEIGHT_WINSOR_UPPER = 0.99`,
`CALIBRATION_VERSION = "rev3-ordinal-2026-09"`.

## Stored fields

Per-label columns in `response_targets.parquet`:

| Field | Meaning |
|---|---|
| `eligible` | hard eligibility \(A_i\) |
| `design_unreliability` | percentile \(P^{design}_i\) |
| `design_weight` | \(\exp(-\gamma P^{design}_i)\) |
| `has_precision_weight` | label carries a usable SE |
| `precision_weight` | \((\tilde p^{SE})^\alpha\) or 1 |
| `weight_raw` | before winsorization/normalization |
| `response_weight` | final weight |
| `calibration_version` | e.g. `rev3-ordinal-2026-09` |

Per-treatment: `sample_index.parquet:response_weight`; release manifest:
`response_weight` block with the calibration version and parameters.

## Versions

| Version | Status | Method |
|---|---|---|
| W0 | baseline | admissible labels equally weighted |
| **W1 (Rev3)** | **current** | ordinal design percentile + tempered precision |
| W2 | robustness | monotonic calibration of design score on a stratified 300–600 task pseudo-treatment subsample, checking that design unreliability predicts \(\lvert\hat\tau^{pseudo}\rvert\) |
| W3 | paper | full pseudo-treatment calibration of expected label variance \(V^{design} = E[(\hat\tau^{pseudo})^2 \mid X]\) with variance-additive combination \(1/(V^{design}+SE^2)\) |

W2/W3 require re-running the production estimators on pseudo interventions
(and, for the first-difference trend diagnostic, storing pre-treatment gap
paths in the task products).  Neither is required for W1, and W1's stored
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
