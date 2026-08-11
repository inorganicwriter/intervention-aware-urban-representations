# Housing raw-data organization and naming

Updated: 2026-07-22

## Canonical layout

```text
data/archive/raw/housing/
  platform_exports/
    anjuke/cross_section/
    lianjia/purchased_transactions/
  web_archives/
    wayback/{inventories,manifests,parsed_pages,raw_html}/
  open_data/
    datasets/{provider_dataset_release}/
    import_batches/{batch_id}/
  spatial_support/
    community_aoi/{provider_or_batch}/
    grid_price_2023_05/{city_source_name}/
  README.md
```

The first directory identifies acquisition class. The next directory identifies
the platform, repository dataset, or spatial product. Download packages and
their extracted contents stay together under one dataset identifier.

## Naming rules

1. Project-owned directory and file names use lowercase ASCII `snake_case`.
2. Research city identifiers use the canonical lowercase pinyin `city_key`.
3. Dates use ISO order: `YYYY-MM-DD`; ranges use `YYYY-MM-DD--YYYY-MM-DD`.
4. Project-owned tabular files use
   `{city_key}__{source}__{unit}__{period}.{ext}` when all fields are known.
5. External dataset directories use their stable provider identifier and
   published release, for example `mendeley_pj2zff4p9m_v4`. Provider release
   labels are provenance, not project panel versions.
6. Source-owned basenames and archive-internal paths are never renamed. This
   includes purchased Chinese filenames, repository attachments, shapefile
   component names, and extracted software packages.
7. Derived, parsed, standardized, and quarantined artifacts belong in
   `data/archive/staging/housing/`, not beside immutable downloads.
8. Every raw file is recorded with relative path, byte size, classification,
   filename policy, and SHA-256 in
   `outputs/housing_acquisition/housing_raw_inventory.csv`.

## Change control

New raw batches are additive. Corrections receive a new stable batch identifier;
they do not overwrite an earlier download. A path migration must be recorded in
`housing_raw_path_migration.csv`, while content changes require a new checksum
and an explicit supersession record.
