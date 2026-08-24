# Research Design Decisions

This directory records frozen or proposed economic and research-design choices.
Decision records are versioned separately from implementation code so that model
refactoring cannot silently change the estimand, risk set, or admissible response
supervision.

## Decisions

- [`DDR-001_spatial_treatment_and_donor_exclusion.md`](DDR-001_spatial_treatment_and_donor_exclusion.md): 500m treatment unit and 1km primary donor exclusion rule.
- [`DDR-003_complete_published_estimators.md`](DDR-003_complete_published_estimators.md): isolated implementations of PanelMatch, Abadie–Imbens matching, and Xu GSC.
- [`DDR-004_causal_response_labels.md`](DDR-004_causal_response_labels.md): response-label timing, information boundary, routing, and output contract.
- [`DDR-005_city_centres.md`](DDR-005_city_centres.md): McMillen-based city-centre and subcentre registry used by location covariates.

## Status values

- `proposed`: ready for discussion but not binding;
- `discussed`: reviewed but unresolved;
- `frozen`: binding for the named design version;
- `superseded`: replaced and normally removed from the active tree; Git history preserves provenance.
