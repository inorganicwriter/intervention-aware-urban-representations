# Research Documentation

Only the current research contract is indexed here. Superseded housing-DID designs,
prototype feasibility notes, and result-driven improvement plans are stored under
`../archive/` as historical records.

## Authoritative design

- [`econometric_methods.md`](econometric_methods.md): econometric objects, identification assumptions, estimators, inference, parallel-trend tests, robustness and reporting standards.
- [`counterfactual_response_label_design.md`](counterfactual_response_label_design.md): complete design for the 5,048 treated grids, pre-only matching, GSC fallback, response labels, quality gates, and production order.
- [`identification_and_diagnostics.md`](identification_and_diagnostics.md): executable event-study and pooled-path diagnostics, output files and interpretation flags.
- [`matching_and_gsc_methodology.md`](matching_and_gsc_methodology.md): matching and GSC algorithm details, literature mapping, and differences from original papers.
- [`../architecture/research_architecture.md`](../architecture/research_architecture.md): concise relationship between causal-label construction and representation learning.
- [`decisions/`](decisions/): frozen research-design decisions that protect the estimand and implementation boundary.

## Research context

- [`project_abstract.md`](project_abstract.md): project title and abstract.
- [`related_work_literature.md`](related_work_literature.md): literature review.
- [`robustness_plan.md`](robustness_plan.md): robustness specifications and reporting requirements.

The former `complete_project_pipeline.md` duplicated the root project guide and
the causal design. It is archived in `../archive/2026-08-document-consolidation/`;
use [`../PROJECT_GUIDE.md`](../PROJECT_GUIDE.md) for the end-to-end project map.

Dataset-specific documentation belongs in [`../data/`](../data/) and
[`../operations/`](../operations/). The active causal design keeps the research rules.
