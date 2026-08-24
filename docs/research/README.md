# Research Documentation

Only the current research contract is indexed here. Superseded housing-DID designs,
prototype feasibility notes, and result-driven improvement plans have been removed.

## Authoritative design

- [`complete_project_pipeline.md`](complete_project_pipeline.md): end-to-end project, production, Response Artifact, and leakage-safe pretraining-data contract.
- [`counterfactual_response_label_design.md`](counterfactual_response_label_design.md): complete design for the 5,048 treated grids, pre-only matching, GSC fallback, response labels, quality gates, and production order.
- [`matching_and_gsc_methodology.md`](matching_and_gsc_methodology.md): matching and GSC algorithm details, literature mapping, and differences from original papers.
- [`../architecture/research_architecture.md`](../architecture/research_architecture.md): concise relationship between causal-label construction and representation learning.
- [`decisions/`](decisions/): frozen research-design decisions that protect the estimand and implementation boundary.

## Research context

- [`project_abstract.md`](project_abstract.md): project title and abstract.
- [`related_work_literature.md`](related_work_literature.md): literature review.
- [`robustness_plan.md`](robustness_plan.md): robustness specifications and reporting requirements.

Dataset-specific documentation belongs in [`../data/`](../data/) or
[`../operations/`](../operations/), not in the active causal design.
