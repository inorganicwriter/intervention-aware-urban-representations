/**
 * GEE Code Editor Script: VIIRS Black Marble VNP46A2 MONTHLY Nighttime Lights
 * Extracts MONTHLY mean radiance for 44 Chinese metro cities.
 *
 * Same methodology as gee_viirs_export.js but aggregates by month instead of year.
 * Each export produces one city-month CSV with year, month, grid_id, avg_rad.
 *
 * Product:  NASA/VIIRS/002/VNP46A2 (daily, gap-filled, BRDF-corrected)
 * Band:    Gap_Filled_DNB_BRDF_Corrected_NTL  (nW·cm^-2·sr^-1)
 * Period:  2012-01 through 2024-12
 *
 * Prerequisite:
 *   Upload grid GeoJSON assets per city to GEE:
 *     Asset ID: projects/macro-city-engine/assets/mit_grids_v2/{city}
 *   Run: python scripts/collection/grid_builder.py first
 *
 * Usage in GEE Code Editor:
 *   1. Paste this entire script
 *   2. Run to queue exports (Run button)
 *   3. Click "Run" in Tasks tab for each export
 *   4. Download CSVs from Google Drive
 *   5. Process with: python scripts/collection/process_viirs_monthly.py
 *
 * NOTE: This queues 44 × 12 = 528 exports per year. To avoid overwhelming
 * GEE, export one city at a time by uncommenting specific city.
 */

var GRID_ASSET_PREFIX = 'projects/macro-city-engine/assets/mit_grids_v2';

var CITIES = {
  beijing: [116.3, 39.9],
  changchun: [125.32, 43.82],
  changsha: [112.97, 28.23],
  changzhou: [119.97, 31.77],
  chengdu: [104.07, 30.57],
  chongqing: [106.55, 29.56],
  dalian: [121.62, 38.91],
  dongguan: [113.75, 23.02],
  foshan: [113.12, 23.02],
  fuzhou: [119.3, 26.07],
  guangzhou: [113.26, 23.13],
  guiyang: [106.71, 26.65],
  hangzhou: [120.21, 30.27],
  harbin: [126.63, 45.75],
  hefei: [117.23, 31.82],
  hohhot: [111.67, 40.82],
  jinan: [117.0, 36.65],
  jinhua: [119.65, 29.08],
  kunming: [102.73, 25.04],
  lanzhou: [103.73, 36.03],
  luoyang: [112.45, 34.62],
  nanchang: [115.86, 28.68],
  nanjing: [118.8, 32.06],
  nanning: [108.32, 22.82],
  nantong: [120.9, 31.98],
  ningbo: [121.54, 29.87],
  qingdao: [120.38, 36.07],
  shanghai: [121.47, 31.23],
  shaoxing: [120.58, 30.03],
  shenyang: [123.43, 41.8],
  shenzhen: [114.06, 22.54],
  shijiazhuang: [114.51, 38.04],
  suzhou: [120.59, 31.3],
  taiyuan: [112.55, 37.87],
  taizhou: [121.42, 28.66],
  tianjin: [117.2, 39.13],
  urumqi: [87.62, 43.79],
  wenzhou: [120.7, 28.0],
  wuhan: [114.3, 30.59],
  wuxi: [120.3, 31.57],
  xiamen: [118.09, 24.48],
  xian: [108.94, 34.26],
  xuzhou: [117.18, 34.26],
  zhengzhou: [113.65, 34.76],
};

/**
 * Extract monthly mean VIIRS radiance for one city-month.
 */
function extractVIIRSMonthly(cityKey, centerLon, centerLat, year, month) {
  var startDate = ee.Date.fromYMD(year, month, 1);
  var endDate = startDate.advance(1, 'month');
  
  // Bbox: center ±0.6° for filterBounds pre-filter
  var bbox = ee.Geometry.Rectangle([
    centerLon - 0.6, centerLat - 0.6,
    centerLon + 0.6, centerLat + 0.6
  ]);
  
  var viirs = ee.ImageCollection('NASA/VIIRS/002/VNP46A2')
    .filterDate(startDate, endDate)
    .filterBounds(bbox)
    .select(['Gap_Filled_DNB_BRDF_Corrected_NTL', 'Mandatory_Quality_Flag']);
  
  // Quality mask: keep high-quality (0) and ephemeral high-quality (1)
  // Drop poor quality (2) and snow-contaminated pixels
  function applyQualityMask(img) {
    var qf = img.select('Mandatory_Quality_Flag');
    return img.updateMask(qf.lte(1));
  }
  viirs = viirs.map(applyQualityMask);
  
  var monthly = viirs.select('Gap_Filled_DNB_BRDF_Corrected_NTL')
                     .mean()
                     .rename('avg_rad')
                     .clip(bbox);
  
  var gridAsset = ee.FeatureCollection(GRID_ASSET_PREFIX + '/' + cityKey);
  var fc = monthly.reduceRegions({
    collection: gridAsset,
    reducer: ee.Reducer.mean(),
    scale: 500,
    tileScale: 4
  });
  fc = fc.map(function(f) {
    return f.set({city: cityKey, year: year, month: month});
  });
  return fc;
}

/**
 * Export one city-month to Google Drive.
 */
function exportCityMonth(cityKey, centerLon, centerLat, year, month) {
  var fc = extractVIIRSMonthly(cityKey, centerLon, centerLat, year, month);
  
  Export.table.toDrive({
    collection: fc,
    description: 'viirs_' + cityKey + '_' + year + '_' + String(month).padStart(2,'0'),
    folder: 'MIT_Summer_VIIRS_Monthly',
    fileFormat: 'CSV',
    selectors: ['city', 'year', 'month', 'grid_id', 'avg_rad']
  });
}

// ── RUN ────────────────────────────────────────────────────
// Export one city-month first to test:
exportCityMonth('beijing', 116.3, 39.9, 2023, 1);

// To batch export all cities for one year, replace the above with:
// var YEAR = 2014;
// var cityKeys = Object.keys(CITIES);
// cityKeys.forEach(function(ck) {
//   var center = CITIES[ck];
//   for (var m = 1; m <= 12; m++) {
//     exportCityMonth(ck, center[0], center[1], YEAR, m);
//   }
// });

print('Queued. Click Run in Tasks tab to start exports.');
