# Canonical Data Layout

The July 2026 layout migration separates immutable source material from
staging outputs, curated features, research labels, and final panels.

```text
data/
├── active/              # Current frozen research assets and production queues
│   ├── catalog/         # Sources, dataset registry, schemas, resolutions
│   ├── reference/       # Boundaries, stable grids and resolved station events
│   ├── curated/         # Canonical covariates (VIIRS, POI, population, Sentinel-2)
│   ├── causal/          # Treatment list, donor universe, control inputs and queues
│   ├── labels/          # Outcomes separated by price type and source
│   └── panels/          # Versioned analysis-ready datasets
└── archive/             # Immutable source material and rebuildable staging
    ├── raw/             # Append-only source files and acquisition records
    └── staging/         # Parsed or cleaned intermediates awaiting publication checks
```

## Layer contract

- `active/`: published, frozen research assets; production queues live here and
  are single-writer resources.
- `archive/raw`: append-only; acquisition code adds files while the existing
  history remains unchanged.
- `archive/staging`: source-specific schemas are allowed and outputs are replaceable.
- `active/reference`: stable spatial definitions; canonical CRS is EPSG:4326.
- `active/curated`: published features; declared primary keys must be unique.
- `active/labels`: outcomes retain source and observation semantics.
- `active/panels`: versioned joins of reference, curated, treatment and label assets.
- `.runtime`: browser profiles and logs; these files belong to the execution environment.

Raw transit files use a source-first layout:
`data/archive/raw/transit/{amap,osm,wikipedia,wikidata,merged}/{city_key}/`.

The temporary compatibility views (`data/grids`, `data/processed`,
`data/raw_housing`, `data/external`, and `data/labels_canonical`) were removed
on 2026-07-14 after active code was migrated and the regression suite passed.
New code must import locations from `urban_intervention.data.paths`; the registry in
  `data/active/catalog/datasets.yaml` is the machine-readable dataset contract.

## Reproducibility

Create a snapshot before and after a migration:

```bash
python scripts/data_management/snapshot_data.py --name before_layout_v2
python scripts/data_management/migrate_data_layout.py          # dry run
python scripts/data_management/migrate_data_layout.py --execute
python scripts/data_management/snapshot_data.py --name after_layout_v2
```

For a full byte-level archive audit, add `--hash-mode all`. The default quick
mode records every file and hashes small or metadata-oriented assets.

## Canonical keys

- Grid reference: `city_key, grid_id`
- Grid-year facts: `city_key, grid_id, year`
- City HPI remains city-level and joins late; the panel keeps it as contextual information.

Housing transaction, listing and index outcomes must remain separate. The
required semantic vocabulary is defined in
`data/active/catalog/schemas/housing_observation.yaml`.
