# Housing acquisition contract

Updated: 2026-07-22

## Objective

Collect every housing observation that is legally and technically available
without pre-filtering cities, years, station cohorts, or observed prices. Data
quality and causal support are evaluated only after immutable raw acquisition.

## Source layers

| Layer | Examples | Economic meaning | Default role |
|---|---|---|---|
| Transaction | licensed Lianjia/Beike transactions, Wayback chengjiao | realized sale | primary housing outcome |
| Listing | Anjuke/Lianjia community pages, Wayback xiaoqu | seller/platform asking reference | auxiliary outcome or calibration |
| Platform estimate | community reference price | modeled/current valuation | cross-sectional auxiliary signal |
| Index | NBS HPI | city-market change | city-month contextual control |

No layer may silently overwrite or fill another layer.

## Access modes

Allowed acquisition modes are licensed export, platform export, Wayback
archive, official open data, and web collection explicitly permitted by the
platform. Collection must stop when access controls, robots rules, or the
authorization scope prohibit automation. Captcha solving, browser-fingerprint
masking, proxy rotation, or similar access-control circumvention is not part of
the research pipeline.

## Storage

```text
data/archive/raw/housing/
  platform_exports/
    anjuke/cross_section/                    platform cross-section
    lianjia/purchased_transactions/          purchased source bundle
    authorized_imports/{batch_id}/           future authorized exports
  web_archives/wayback/                      archive inventories and outcomes
  open_data/{datasets,import_batches}/        repository downloads and imports
  spatial_support/{community_aoi,grid_price_2023_05}/

data/archive/staging/housing/
  standardized/{batch_id}/                   canonical observations + manifest
```

Every new batch has a unique identifier. Existing batches are never silently
overwritten. Corrections use a new batch identifier and record the superseded
batch in research documentation.

The complete naming policy is defined in
[`housing_raw_naming.md`](housing_raw_naming.md).

## Required provenance

- source platform and acquisition method;
- immutable input-file SHA-256;
- source record identifier or deterministic generated identifier;
- snapshot/acquisition date and economic observation date kept separately;
- source URL/page identifier when supplied;
- raw row number, mapping file, parser/importer version, and quality flags.

## Publication gates

1. `(source_platform, source_record_id)` is non-null and duplicate status is reported;
2. all input rows are represented in staging, including invalid rows;
3. transaction prices have `deal_date` and either unit price or total price plus area;
4. listing/estimate prices have an actual source snapshot date;
5. coordinates, city mappings, prices, and dates have explicit quality flags;
6. no personal data unrelated to the property observation is selected into the canonical table;
7. city/year coverage and structural missingness are published before model use.

The machine-readable schema is
`data/active/catalog/schemas/housing_observation.yaml`.
