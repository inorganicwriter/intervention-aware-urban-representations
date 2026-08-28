# Housing price acquisition

Updated: 2026-07-22

## Current decision

Acquisition discovery is unrestricted by city, year, or station cohort. Raw
records are preserved before downstream quality and causal-inclusion filters.

Wayback collection is complete under the exact-endpoint contract. The local
audit reports 202/202 targets complete, 4,253/4,253 captures with terminal
outcomes, 35,014 parsed rows, and complete exact-capture source records.
Run the audit at any time with:

```bash
python scripts/analysis/audit_housing_acquisition.py
```

Generated evidence is under `outputs/housing_acquisition/`.

## Live websites

Automated live Lianjia, Beike, and Anjuke collection is disabled unless the
platform supplies explicit permission or an authorized export. The retired
workflow used automation masking and interactive captcha handling; those paths
are no longer callable. A project member's authorization cannot substitute for
the platform's access permission.

The supported acquisition order is:

1. licensed or platform-provided bulk export;
2. platform user export where the agreement permits research use;
3. official open housing-transaction data;
4. permitted low-rate web collection with a documented authorization scope.

## Authorized import

Create a mapping based on
`data/active/catalog/schemas/housing_import_mapping_example.yaml`, then run:

```bash
python scripts/collection/import_housing_observations.py \
  --input /path/to/authorized_export.csv \
  --mapping /path/to/import_mapping.yaml
```

The importer accepts CSV, Parquet, XLS, and XLSX. It preserves every city, year,
and invalid row. It copies the immutable source and mapping to
`data/archive/raw/housing/platform_exports/authorized_imports/{batch_id}/`, writes canonical observations
to `data/archive/staging/housing/standardized/{batch_id}/`, and records hashes, row
counts, mappings, and quality flags in `import_manifest.json`.

Transaction, listing, platform-estimate, and index prices remain separate.
Anjuke community/listing data may improve the community registry and provide a
cross-sectional price observation. Lianjia transaction prices remain a separate
source layer.

## Offline parsers

`housing_price_fetcher.py` and `xiaoqu_fetcher.py` retain offline HTML parsers
for licensed page archives and regression tests. Their live network entrypoints
stop and direct the operator to the authorized importer.
