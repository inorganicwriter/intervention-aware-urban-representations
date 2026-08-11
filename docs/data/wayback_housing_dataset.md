# Wayback Housing Price Dataset

**Version 1.1 — July 2026**

Historical housing price data for 42 Chinese metro cities (2012–2026), collected by archiving Wayback Machine snapshots of Lianjia, Beike, and Anjuke listing and transaction pages.

---

## 1. Provenance

All data was obtained from the **Internet Archive's Wayback Machine** (`web.archive.org`). No live websites were scraped; every record comes from historical page snapshots archived between 2012 and 2026.

| Source | Platform | Page Type | Content |
|--------|----------|-----------|---------|
| `chengjiao` | lianjia.com | `/chengjiao/` (transaction list) | Individual sale records: community name, unit price (¥/m²), total price (万¥), deal date |
| `xiaoqu` | lianjia.com | `/xiaoqu/` (community list) | Community listing snapshots: community name, average unit price, location description |
| `beike_chengjiao` | ke.com | `/chengjiao/` | Same structure as Lianjia chengjiao; Beike shares Lianjia's underlying database |
| `beike_xiaoqu` | ke.com | `/xiaoqu/` | Same as Lianjia xiaoqu; covers cities with no Lianjia subdomain |
| `anjuke` | anjuke.com | `/community/` | Historical community listing pages (2012–2018), filling early-year gaps |
| `detail` | lianjia.com | `/xiaoqu/{id}/` | Individual community detail pages; very low coverage (experimental) |

**Canonical exact-endpoint audit:** 202/202 city-platform-page targets have
inventories. All 4,253 inventory captures have terminal outcomes, 1,642
captures yielded parseable rows, and none remain retryable or missing. The
parsed target products contain 35,014 rows and every row traces to an exact
`timestamp + original URL` inventory key. Older counts in legacy CDX caches
used a broader, non-canonical discovery scope and are not the acceptance count.

The generated authoritative audit is
[`../../outputs/housing_acquisition/housing_acquisition_report.md`](../../outputs/housing_acquisition/housing_acquisition_report.md).

---

## 2. File Inventory

The canonical collector writes all files under
`data/archive/raw/housing/web_archives/wayback/`.
Each city × source combination produces one CSV.

```
data/archive/raw/housing/web_archives/wayback/
├── parsed_pages/
│   └── {city}_wayback_{source}.csv    # parsed rows, one file per city/source
├── inventories/
│   └── {source}_{city}_{page}.json    # exact CDX timestamp/original pairs
└── manifests/
    └── {source}_{city}_{page}.jsonl   # per-capture crawl outcomes and resume state
```

The inventory and manifest are part of the dataset provenance: a CSV row can
be traced back to the exact Wayback capture that produced it.

---

## 3. Schema

### 3.1 Chengjiao (Lianjia transactions)

| Column | Type | Description |
|--------|------|-------------|
| `city_key` | str | City identifier (e.g., `beijing`) |
| `community` | str | Community name in Chinese |
| `unit_price` | float | Unit price (¥/m²) |
| `total_price` | float | Total price (万¥, 10,000s ¥) |
| `deal_date` | str | Transaction date (e.g., `2020.09.01`) |
| `deal_year` | int | Transaction year |
| `layout` | str | Layout (e.g., `2室1厅`) |
| `area_m2` | float | Area (m²) |
| `snapshot_year` | int | Year of Wayback snapshot |
| `snapshot_date` | str | Wayback snapshot timestamp |

### 3.2 Xiaoqu (Lianjia community listings)

| Column | Type | Description |
|--------|------|-------------|
| `city_key` | str | City identifier |
| `community` | str | Community name in Chinese |
| `unit_price` | float | Average listing price (¥/m²) |
| `position` | str | Location description (district, road) |
| `house_info` | str | Community info (build year, building types) |
| `snapshot_year` | int | Year of Wayback snapshot |
| `snapshot_date` | str | Wayback snapshot timestamp |
| `detail_url` | str | Archived detail page URL (contains community ID) |
| `source_page` | int | Page number in listing (always 1) |

### 3.3 Beike (same structure as Lianjia, prefix `beike_`)

### 3.4 Anjuke (Wayback snapshots)

| Column | Type | Description |
|--------|------|-------------|
| `city_key` | str | City identifier |
| `community` | str | Community name in Chinese |
| `unit_price` | float | Unit price (¥/m²) |
| `position` | str | Location description |
| `house_info` | str | Additional info |
| `snapshot_year` | int | Year of Wayback snapshot |
| `snapshot_date` | str | Wayback snapshot timestamp |

---

## 4. City Coverage

| City | chengjiao | xiaoqu | beike_cj | beike_xq | anjuke | Total |
|------|-----------|--------|----------|----------|--------|-------|
| Beijing | 2,010 | 251 | 2,460 | 472 | 454 | 5,673 |
| Nanjing | 990 | 184 | 0 | 180 | 32 | 1,386 |
| Dalian | 930 | 197 | 0 | 150 | 103 | 1,381 |
| Chongqing | 750 | 246 | 0 | 120 | 46 | 1,162 |
| Guangzhou | 507 | 377 | 0 | 151 | 83 | 1,119 |
| Shanghai | 690 | 120 | 0 | 120 | 153 | 1,083 |
| Shenzhen | 480 | 247 | 0 | 120 | 134 | 981 |
| Jinan | 300 | 224 | 0 | 205 | 153 | 882 |
| Qingdao | 390 | 150 | 0 | 181 | 90 | 811 |
| Foshan | 30 | 287 | 0 | 180 | 286 | 783 |
| ... | ... | ... | ... | ... | ... | ... |

**43 cities have parsed canonical target data.** Changzhou has inventories but
no parsed rows. Target-level counts are generated in
`outputs/housing_acquisition/wayback_target_audit.csv`; the manually maintained
examples above are descriptive and not the acceptance source.

---

## 5. Year Coverage

| Period | Rows | Primary Source |
|--------|------|----------------|
| 2012–2015 | ~3,900 | Anjuke Wayback |
| 2016–2020 | ~14,000 | Lianjia chengjiao + xiaoqu |
| 2021–2026 | ~5,000 | Lianjia xiaoqu + Beike |

**Peak years:** 2019–2021 (2,600–3,300 rows/year). Early years (2012–2015) are sparse because only Anjuke pages were archived.

---

## 6. Limitations

1. **No coordinates.** Wayback archives are HTML pages; latitude/longitude are not embedded. Community names must be geocoded externally (via Amap/Baidu API or cross-referenced with coordinate databases).

2. **Spatial density.** Most cities have 50–400 communities represented. Each community appeared only on page 1 of the listing (30 items/page). Wayback did not archive paginated pages (pg2+). This is the fundamental constraint of page-level web archiving.

3. **Chengjiao is spotty.** Transaction pages (chengjiao) show 30 recent sales per snapshot, typically from a single time window. Most communities appear in only 1–2 snapshot years.

4. **No duplicate removal across sources.** A community may appear in Lianjia, Beike, and Anjuke snapshots with slightly different names. Cross-source deduplication is the responsibility of downstream processing.

5. **Price type varies.** Chengjiao prices are actual transaction values; xiaoqu prices are listing reference prices (挂牌均价). Beike and Anjuke prices follow the same conventions as their source platforms.

---

## 7. Collection Pipeline

```
1. CDX API query:
   https://web.archive.org/cdx/search/cdx?url={sub}.lianjia.com/xiaoqu/&matchType=exact
   → Returns captures of the exact list endpoint only; the original URL and
     exact 14-digit timestamp are persisted in the inventory.

2. Snapshot selection:
   Fetch every in-scope exact-endpoint capture (2012–2026 by default). CDX
   resumption keys prevent silent truncation at the per-response limit.

3. Page fetch:
   https://web.archive.org/web/{timestamp}id_/{original_url}
   → Replays the exact CDX capture, rather than a nearest capture for a
     different URL on the same date. The collector prefers HTTPS and falls
     back to HTTP only when the configured proxy breaks Wayback TLS.

4. HTML parsing:
   BeautifulSoup extracts community name + price from structured elements

5. Incremental save:
   Results and a snapshot-level JSONL manifest are updated every 20 captures;
   failed requests are retryable and completed captures are not re-fetched.
```

**Canonical collection script:** `scripts/collection/wayback_research_scraper.py`

---

## 8. Usage

```python
import pandas as pd

# Load all Beijing data
base = "data/archive/raw/housing/web_archives/wayback/parsed_pages"
beijing = pd.read_csv(f"{base}/beijing_wayback_chengjiao.csv")
beijing_xq = pd.read_csv(f"{base}/beijing_wayback_xiaoqu.csv")
beijing_bk = pd.read_csv(f"{base}/beijing_wayback_beike_chengjiao.csv")

# Each CSV has a different schema — see Section 3.
# Join by community name for cross-source validation.
```

For grid-level panel construction, see `scripts/labels/build_wayback_label.py`.

---

## 9. License & Attribution

Data sourced from publicly archived web pages via the Internet Archive. The underlying listing/transaction data originates from Lianjia (lianjia.com), Beike (ke.com), and Anjuke (anjuke.com). This dataset is intended for academic research use only.

**Contact:** MIT Summer Research Project, July 2026
