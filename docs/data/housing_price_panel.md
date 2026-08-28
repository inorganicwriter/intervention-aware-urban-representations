# Housing price panel

Updated: 2026-07-22

## Outcome contract

The housing panel treats every located, dated price as an observation of the
local housing market. Listings are initial price observations and completed
transactions are final price observations. Dwelling characteristics remain
source fields and stay outside hedonic or composition adjustment.

Source and price-stage labels remain in the observation layer for source
records, lifecycle de-duplication, source balancing, and sensitivity analysis.
Panel inclusion follows the separate causal support checks.

## Published products

- `data/active/curated/housing/housing_observations/{city}.parquet`
- `data/active/panels/housing_grid_month/{city}_source.parquet`
- `data/active/panels/housing_grid_month/{city}.parquet`
- `data/active/panels/housing_grid_quarter/{city}_source.parquet`
- `data/active/panels/housing_grid_quarter/{city}.parquet`
- `data/active/panels/housing_grid_year/{city}_source.parquet`
- `data/active/panels/housing_grid_year/{city}.parquet`
- `outputs/housing_panel/`

The primary monthly outcome is `price_source_balanced_cny_m2`, calculated as
the exponentiated median of source-specific log-price medians within each
city-grid-month.  `price_raw_median_cny_m2` pools every canonical observation
and is retained as a sensitivity outcome.

Annual observations retain their annual period. Quarterly observations enter
the quarterly and annual panels, while monthly panels use monthly observations.
The Chengdu open-research transactions remain in the observation layer until
their coordinate CRS is resolved, after which the grid panel can use them.

## Current closure

- 44 cities
- 5,924,124 source-preserving price observations
- 4,059,519 monthly-eligible observations
- 1,187,043 unique 500 m grid-months
- 624,914 unique grid-quarters
- 317,105 unique grid-years

The formal audit records zero duplicate observation IDs, zero duplicate panel
keys, zero invalid primary prices, and zero annual/quarterly leakage into the
monthly panel.  See `outputs/housing_panel/row_closure.json`.

## Reproducible commands

```powershell
python scripts/labels/build_housing_price_panel.py
python scripts/analysis/audit_housing_panel.py
```
