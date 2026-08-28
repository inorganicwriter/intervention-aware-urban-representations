# VIIRS monthly processing contract

Updated: 2026-07-22

## Raw batch

- Product: `NASA/VIIRS/002/VNP46A2`
- Band: `Gap_Filled_DNB_BRDF_Corrected_NTL`
- Coverage: 44 cities, 2012-01 through 2024-12
- Files: 6,864 CSV files under the external directory configured by `MIT_VIIRS_RAW`
- Raw size: 258,534,956,303 bytes
- Manifest status: complete; no duplicate city-month names, zero-byte files,
  or header variants

The external directory is immutable input. The project records its path and reads
the partitions through the cache contract.

## Why the exact grid assignment is used

The GEE exporter uses `image.sample(scale=500, projection="EPSG:4326")`.
Those source centers do not share the projected UTM lattice used by the
project's 500 m grids. In Beijing, the observed source-center spacing is about
382 m east-west and 499 m north-south. Multiple distinct VIIRS centers can
therefore fall inside one project grid.

Assigning every source center to the nearest target centroid within 500 m
represents distance rather than cell containment. On Beijing 2012-01 it admitted
854 centers from cells that had been clipped out of the reference grid and
assigned one additional center to a different neighboring grid.

## Canonical matching and aggregation

1. Validate city, month, product, coordinates, and valid-day range against the
   filename and source contract.
2. Within a city-month, identify a VIIRS source sample by its exact longitude
   and latitude. Repeated identical copies are collapsed. If the same
   coordinate has conflicting radiance or valid-day values, processing fails;
   conflicts are never silently averaged.
3. Recover the target grid's exact UTM origin from persisted `row`, `col`, and
   centroid coordinates. All 44 city grids reproduce a common 500 m lattice;
   the maximum observed origin residual is below 0.006 m.
4. Transform the source center to the city UTM CRS and assign it to the exact
   half-open target cell `[x0,x1) × [y0,y1)`. Centers in clipped-out cells are
   reported as outside the reference grid. Grid assignment uses the exact cell
   rule throughout.
5. Multiple *distinct* source centers inside one target grid form spatial
   support. The primary target-grid estimand is their
   unweighted mean radiance. Equal weighting is appropriate because centers
   come from one fixed-scale source lattice within a city. `valid_days` is a
   quality measure and leaves the radiance estimand unchanged.
6. Preserve `source_point_count`. Time-varying drops in this count are audited
   after processing because they can indicate changing within-grid support.

The center-in-cell mean is a zonal-mean approximation rather than an exact
source-pixel/target-polygon overlap calculation. A nearest-source-to-target-
centroid outcome may be produced later as a sensitivity analysis. The primary
grid-average outcome uses the center-in-cell mean.

## Compact publication

```text
data/active/curated/viirs/monthly/
  city_key={city}/year={YYYY}/month={MM}/part.parquet
```

Hive partition paths carry city and time. Each Zstandard-compressed Parquet
file stores only:

- `grid_id` (`string`)
- `avg_rad` (`float32`)
- `valid_days_mean` (`float32`)
- `source_point_count` (`uint16`)

Coordinates, product strings, filenames, matching diagnostics, and aggregation
descriptions are stored once per partition in
`outputs/viirs_monthly/partition_audits/` and referenced by every grid row.

## Publication checks

- unique `(city_key, grid_id, year, month)` after reading Hive partitions;
- no same-coordinate conflicts;
- no nearest-centroid assignments;
- all outside-grid and invalid-coordinate rows reported;
- distribution of `source_point_count` and valid days audited by city-month;
- negative finite radiance retained; modeling uses `asinh(avg_rad)`;
- complete city-month partitions or explicit structural-missingness records.
