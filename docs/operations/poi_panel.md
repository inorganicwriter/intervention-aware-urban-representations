# Amap POI Panel

Grid-year POI panels for 44 metro cities, 2012-2024.

## Output

Coverage: 44 cities × 13 years (2012-2024).

```text
data/active/curated/poi/{city}_poi_grid_yearly.parquet
data/active/curated/poi/{city}_poi_grid_yearly.provenance.json
```

The JSON sidecar records the producer, source format and source identifiers for
each year without adding non-feature columns to the Parquet panel.

28 category columns per grid-year row, including food, retail, life_service,
leisure, education_culture, healthcare, lodging, transport, finance, indoor,
road_facility, event, and more.  Also includes derived metrics: category
entropy, commercial/chain/community-commerce shares, and year-over-year
change proxies.

## Source Years

| Years | Format | Script | Categories |
|-------|--------|--------|------------|
| 2012-2017 | WGS84 city CSV | `poi_panel_builder.py` | ~8 key categories |
| 2018-2024 | Nationwide FileGDB | `poi_batch_panel_builder.py` | Full 22+ categories |

These are the only production routes. `poi_panel_builder.py` rejects years
after 2017, and `poi_batch_panel_builder.py` rejects years before 2018.

## Pipeline

```bash
# Pre-2018 CSV processing
python scripts/collection/poi_panel_builder.py --city all --years 2012-2017

# 2018+ GDB batch processing (with parquet cache)
python scripts/collection/poi_batch_panel_builder.py --city all --years 2018-2024 --workers 4

# Cache management
python scripts/collection/poi_batch_panel_builder.py --years 2023 --cache-status
python scripts/collection/poi_batch_panel_builder.py --city all --years 2023 --batch-index 3 --refresh-cache
```

FileGDB batch construction stops when any selected source fails, and the
year is not finalized or saved. `--refresh-cache` forces each selected cache
slice to be rebuilt during that run.

Generated features include:

- `poi_count`
- category counts such as `poi_food_count`, `poi_retail_count`,
  `poi_life_service_count`, `poi_leisure_count`
- `poi_commercial_count`
- `poi_chain_count`
- `poi_community_commerce_count`
- `poi_category_entropy`
- commercial, chain, and community-commerce shares
- net new / exit count proxies based on year-over-year count changes

## Notes

- The 2017→2018 break in total POI counts (e.g. Beijing 29k → 39k grids)
  reflects the source switch from filtered CSV to full-category GDB, not a
  data error.
- 2019 shows ~10-20% fewer POIs vs 2018 across most categories; this is a
  data vendor difference (confirmed by direct GDB comparison), not a pipeline
  artefact.  The 2019 food GDB has a nested structure (全国餐饮服务.gdb/
  全国2019餐饮服务.gdb) where the inner layer is a superset of the outer;
  the pipeline correctly selects the inner layer.
- Parquet cache at `data/archive/staging/poi/parquet_cache/` provides 50× speedup
  for subsequent runs.  The raw GDB archives (~220 GB) are no longer needed
  after processing.
