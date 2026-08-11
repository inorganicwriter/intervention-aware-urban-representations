/**
 * GEE Code Editor Script: VIIRS Black Marble VNP46A2 500m Nighttime Lights
 * Extracts annual mean radiance for 44 Chinese metro cities.
 *
 * Product: NASA/VIIRS/002/VNP46A2 (gap-filled, BRDF-corrected).
 * Band:   Gap_Filled_DNB_BRDF_Corrected_NTL  (nW·cm^-2·sr^-1)
 *
 * ────────────────────────────────────────────────────────────────────
 *  SAMPLING STRATEGY (root-cause fix)
 * ────────────────────────────────────────────────────────────────────
 *  Previous version used `Image.sample()` which generates pixel-centroid
 *  points with GEE-internal origins that do NOT align with the Python-side
 *  `grid_id` system in `grid_builder.py`.  The two layers had no shared
 *  key, forcing a lossy spatial join downstream.
 *
 *  This version uses `Image.sampleRegions()` against a per-city grid
 *  FeatureCollection uploaded as a GEE Asset.  Each uploaded feature
 *  carries its `grid_id`, which is automatically propagated into the
 *  exported CSV — enabling an exact `grid_id` join with the Python panel.
 *
 *  Prerequisite (one-time per city):
 *    1. Run `python scripts/collection/grid_builder.py` (produces
 *       `data/reference/grids/{city}/{city}_grids.geojson`).
 *    2. Upload each GeoJSON to GEE as an Asset:
 *         Asset ID:  users/<YOUR_USER>/mit_grids/<city>
 *       (Use the Code Editor "Assets" tab → "Upload" → "Shapefile/GeoJSON".
 *        For GeoJSON, zip it first or use the EarthEngine CLI `upload table`.)
 *    3. Set `GRID_ASSET_PREFIX` below to your GEE username path.
 *
 *  Fallback: set `USE_SAMPLE_REGIONS = false` to revert to `Image.sample()`
 *  with explicit latitude/longitude attributes.  This still exports
 *  coordinates (so the Python side can do a spatial join), but does NOT
 *  carry `grid_id` — use only when you cannot upload assets.
 *
 * ────────────────────────────────────────────────────────────────────
 *  BBOX SOURCE
 * ────────────────────────────────────────────────────────────────────
 *  These bboxes are the hardcoded center ±0.6° fallback values from
 *  pipeline_config.py.  The Python grid pipeline prefers admin-boundary-
 *  derived bboxes.  Because sampleRegions() uses the uploaded grid
 *  features directly (not the bbox), bbox mismatches between GEE and
 *  Python only affect the VIIRS ImageCollection filterBounds() pre-filter
 *  — the actual sampled footprint is always the uploaded grid.
 *
 *  Paste into https://code.earthengine.google.com/ → Run → Export to Drive
 */

// === USER CONFIG =================================================
// Set to your GEE asset path prefix.  Per-city assets are expected at:
//   {GRID_ASSET_PREFIX}/{cityKey}
// e.g. users/mit_researcher/mit_grids/beijing
var GRID_ASSET_PREFIX = 'projects/macro-city-engine/assets/mit_grids_v2';
// Set to false to fall back to Image.sample() (no grid_id; coords only).
var USE_SAMPLE_REGIONS = true;
// =================================================================


// ── City bboxes (synced with pipeline_config.py — center ±0.6°) ─
var CITIES = {
  beijing: ee.Geometry.Rectangle([115.3,39.35,117.63,41.15]),
  changchun: ee.Geometry.Rectangle([123.88,43.07,127.21,45.34]),
  changsha: ee.Geometry.Rectangle([111.79,27.76,114.36,28.75]),
  changzhou: ee.Geometry.Rectangle([119.03,31.06,120.31,32.15]),
  chengdu: ee.Geometry.Rectangle([102.88,30.0,105.0,31.53]),
  chongqing: ee.Geometry.Rectangle([105.18,28.07,110.3,32.29]),
  dalian: ee.Geometry.Rectangle([120.97,38.63,123.63,40.29]),
  dongguan: ee.Geometry.Rectangle([113.42,22.57,114.35,23.23]),
  foshan: ee.Geometry.Rectangle([112.29,22.56,113.49,23.67]),
  fuzhou: ee.Geometry.Rectangle([118.28,25.01,120.82,26.73]),
  guangzhou: ee.Geometry.Rectangle([112.85,22.47,114.15,24.03]),
  guiyang: ee.Geometry.Rectangle([106.02,26.1,107.38,27.45]),
  hangzhou: ee.Geometry.Rectangle([118.24,29.1,120.83,30.65]),
  harbin: ee.Geometry.Rectangle([125.55,43.97,130.36,46.76]),
  hefei: ee.Geometry.Rectangle([116.58,30.86,118.07,32.63]),
  hohhot: ee.Geometry.Rectangle([110.39,39.5,112.42,41.47]),
  jinan: ee.Geometry.Rectangle([116.11,35.9,118.09,37.63]),
  jinhua: ee.Geometry.Rectangle([119.11,28.43,120.88,29.77]),
  kunming: ee.Geometry.Rectangle([102.07,24.3,103.77,26.64]),
  lanzhou: ee.Geometry.Rectangle([102.58,35.48,104.69,37.13]),
  luoyang: ee.Geometry.Rectangle([111.02,33.47,113.08,35.16]),
  nanchang: ee.Geometry.Rectangle([115.33,28.07,116.66,29.23]),
  nanjing: ee.Geometry.Rectangle([118.25,31.14,119.35,32.71]),
  nanning: ee.Geometry.Rectangle([107.23,22.12,109.72,24.12]),
  nantong: ee.Geometry.Rectangle([120.09,31.54,122.49,32.95]),
  ningbo: ee.Geometry.Rectangle([120.79,28.67,122.97,30.54]),
  qingdao: ee.Geometry.Rectangle([119.4,35.36,121.68,37.24]),
  shanghai: ee.Geometry.Rectangle([120.75,30.58,123.33,31.96]),
  shaoxing: ee.Geometry.Rectangle([119.78,29.14,121.33,30.39]),
  shenyang: ee.Geometry.Rectangle([122.3,41.11,123.93,43.13]),
  shenzhen: ee.Geometry.Rectangle([113.58,21.73,114.89,22.95]),
  shijiazhuang: ee.Geometry.Rectangle([113.4,37.35,115.59,38.85]),
  suzhou: ee.Geometry.Rectangle([119.81,30.67,121.49,32.14]),
  taiyuan: ee.Geometry.Rectangle([111.39,37.35,113.27,38.51]),
  taizhou: ee.Geometry.Rectangle([120.18,27.88,122.55,29.44]),
  tianjin: ee.Geometry.Rectangle([116.59,38.46,118.18,40.34]),
  urumqi: ee.Geometry.Rectangle([86.67,42.83,89.1,45.09]),
  wenzhou: ee.Geometry.Rectangle([119.52,26.95,121.94,28.71]),
  wuhan: ee.Geometry.Rectangle([113.59,29.88,115.18,31.45]),
  wuxi: ee.Geometry.Rectangle([119.41,31.01,120.71,32.08]),
  xiamen: ee.Geometry.Rectangle([117.78,24.29,118.55,25.0]),
  xian: ee.Geometry.Rectangle([107.55,33.61,109.93,34.83]),
  xuzhou: ee.Geometry.Rectangle([116.25,33.62,118.78,35.07]),
  zhengzhou: ee.Geometry.Rectangle([112.6,34.17,114.31,35.08]),
};;

// ── VIIRS annual extraction ────────────────────────────────────
function extractVIIRS(bbox, cityKey, year) {
  var startDate = ee.Date.fromYMD(year, 1, 1);
  var endDate = ee.Date.fromYMD(year + 1, 1, 1);

  // VNP46A2 is a daily, gap-filled & BRDF-corrected product.
  // Mandatory_Quality_Flag semantics (per NASA Black Marble team):
  //   0 = High-quality
  //   1 = Low-quality  (cloud-contaminated etc.)
  //   2 = Gap-filled   (imputed by the Black Marble algorithm — this is
  //                     the whole point of using VNP46A2 over VNP46A1,
  //                     so we MUST keep it)
  var viirs = ee.ImageCollection('NASA/VIIRS/002/VNP46A2')
    .filterDate(startDate, endDate)
    .filterBounds(bbox)
    .select(['Gap_Filled_DNB_BRDF_Corrected_NTL', 'Mandatory_Quality_Flag']);

  // VNP46A2.002 official quality semantics:
  //   0 = persistent high-quality, 1 = ephemeral high-quality,
  //   2 = poor quality / possible cloud contamination.
  function applyQualityMask(img) {
    var qf = img.select('Mandatory_Quality_Flag');
    var snow = img.select('Snow_Flag');
    return img.updateMask(qf.lte(1).and(snow.eq(0)));
  }
  viirs = viirs.map(applyQualityMask);

  var annual = viirs.select('Gap_Filled_DNB_BRDF_Corrected_NTL')
                    .mean()
                    .rename('avg_rad')
                    .clip(bbox);

  // ── Sampling ────────────────────────────────────────────────
  var fc;
  if (USE_SAMPLE_REGIONS) {
    // Preferred path: sample at uploaded grid cells so grid_id is carried
    // into the CSV.  tileScale=4 avoids "User memory limit exceeded" on
    // large bboxes (Beijing ~50k grids).
    var gridAsset = ee.FeatureCollection(GRID_ASSET_PREFIX + '/' + cityKey);
    fc = annual.reduceRegions({
      collection: gridAsset,
      reducer: ee.Reducer.mean(),
      scale: 500,
      tileScale: 4
    });
    // Tag each feature with city/year metadata (grid_id already present).
    fc = fc.map(function(f) {
      return f.set({city: cityKey, year: year});
    });
  } else {
    // Fallback: Image.sample() — no grid_id, but explicit coords are set
    // so the Python side can do a spatial join.
    fc = ee.FeatureCollection(annual.sample({
      region: bbox,
      scale: 500,
      projection: 'EPSG:4326',
      geometries: true,
      tileScale: 4
    }));
    fc = fc.map(function(f) {
      var coords = f.geometry().coordinates();
      return f.set({
        city: cityKey,
        year: year,
        longitude: coords.get(0),
        latitude:  coords.get(1)
      });
    });
  }
  return fc;
}

// ── Export for a single city-year ──────────────────────────────
function exportCityYear(cityKey, year) {
  var bbox = CITIES[cityKey];
  var fc = extractVIIRS(bbox, cityKey, year);

  // Selectors: include grid_id (sampleRegions mode) or lat/lon (fallback).
  var selectors = USE_SAMPLE_REGIONS
    ? ['city', 'year', 'grid_id', 'avg_rad']
    : ['city', 'year', 'avg_rad', 'latitude', 'longitude'];

  Export.table.toDrive({
    collection: fc,
    description: 'viirs_' + cityKey + '_' + year,
    folder: 'MIT_Summer_VIIRS',
    fileFormat: 'CSV',
    selectors: selectors
  });
  print('Export started: ' + cityKey + ' ' + year +
        (USE_SAMPLE_REGIONS ? ' [sampleRegions]' : ' [sample fallback]'));
}

// ── Run ────────────────────────────────────────────────────────
// Export all 44 cities × 2012-2024 (one at a time to avoid rate limit).
// Note: VIIRS starts 2012-01-17, so 2012 is a partial year.
var cities = Object.keys(CITIES);
var years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024];

// Run one first to test:
exportCityYear('beijing', 2023);

// Then uncomment to batch:
// cities.forEach(function(cityKey) {
//   years.forEach(function(year) {
//     exportCityYear(cityKey, year);
//   });
// });

print('Done queuing tasks. Check Tasks tab to run exports.');
