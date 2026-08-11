/**
 * GEE Code Editor Script: Sentinel-2 NDVI / NDBI 500m Grid Extraction
 * Annual median composite for 44 Chinese metro cities (2018-2024).
 *
 * ────────────────────────────────────────────────────────────────────
 *  SAMPLING STRATEGY (root-cause fix)
 * ────────────────────────────────────────────────────────────────────
 *  Uses `Image.sampleRegions()` against per-city grid assets uploaded to
 *  GEE so that `grid_id` is carried into the exported CSV and can be
 *  joined exactly with the Python panel.  See gee_viirs_export.js for the
 *  full upload instructions.
 *
 *  Fallback (`USE_SAMPLE_REGIONS = false`): `Image.sample()` with explicit
 *  latitude/longitude attributes — no grid_id, requires spatial join.
 *
 * ────────────────────────────────────────────────────────────────────
 *  CLOUD / SNOW MASK
 * ────────────────────────────────────────────────────────────────────
 *  CLOUDY_PIXEL_PERCENTAGE threshold = 20 (was 30 — aligned with
 *  pipeline_config.py:376 and docs/data_inventory_and_acquisition_plan.md).
 *  SCL mask now excludes snow/ice (SCL=11) in addition to cloud shadow
 *  (3), medium/high-probability cloud (8/9), and thin cirrus (10) —
 *  critical for northern cities (Harbin, Urumqi, etc.) where winter snow
 *  severely distorts NDVI.
 *
 *  Band selection includes B12 (per pipeline_config.py:375 and docs).
 *
 *  Paste into https://code.earthengine.google.com/ → Run → Export to Drive
 */

// === USER CONFIG =================================================
var GRID_ASSET_PREFIX = 'users/inorganicwriter/mit_grids';
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

function extractS2(bbox, cityKey, year) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = ee.Date.fromYMD(year + 1, 1, 1);

  // Band selection per pipeline_config.py:375 (B2,B3,B4,B8,B11,B12 + SCL).
  // CLOUDY_PIXEL_PERCENTAGE < 20 — aligned with pipeline_config.py:376 and
  // docs/data_inventory_and_acquisition_plan.md.
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(start, end)
    .filterBounds(bbox)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'SCL']);

  // SCL mask: drop cloud shadow (3), medium/high-prob cloud (8/9), thin
  // cirrus (10), AND snow/ice (11).  Snow coverage severely distorts NDVI
  // in northern cities (Harbin, Changchun, Shenyang, Urumqi, Hohhot).
  function maskS2(img) {
    var scl = img.select('SCL');
    var mask = scl.neq(3)
      .and(scl.neq(8))
      .and(scl.neq(9))
      .and(scl.neq(10))
      .and(scl.neq(11));
    return img.updateMask(mask);
  }
  s2 = s2.map(maskS2);

  // NDVI = (B8 - B4) / (B8 + B4)
  // NDBI = (B11 - B8) / (B11 + B8)  (built-up index)
  function addIndices(img) {
    var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
    var ndbi = img.normalizedDifference(['B11', 'B8']).rename('NDBI');
    return img.addBands([ndvi, ndbi]);
  }

  s2 = s2.map(addIndices).select(['NDVI', 'NDBI']);
  // Median composite.  clip(bbox) is redundant with sampleRegions but kept
  // for clarity when fallback sample() is used.
  var annual = s2.median().clip(bbox);

  // ── Sampling ────────────────────────────────────────────────
  var fc;
  if (USE_SAMPLE_REGIONS) {
    var gridAsset = ee.FeatureCollection(GRID_ASSET_PREFIX + '/' + cityKey);
    fc = annual.sampleRegions({
      collection: gridAsset,
      scale: 500,
      projection: 'EPSG:4326',
      tileScale: 8  // S2 is heavier than VIIRS; higher tileScale needed.
    });
    fc = fc.map(function(f) {
      return f.set({city: cityKey, year: year});
    });
  } else {
    fc = ee.FeatureCollection(annual.sample({
      region: bbox,
      scale: 500,
      projection: 'EPSG:4326',
      geometries: true,
      tileScale: 8
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

function exportCityYear(cityKey, year) {
  var fc = extractS2(CITIES[cityKey], cityKey, year);
  var selectors = USE_SAMPLE_REGIONS
    ? ['city', 'year', 'grid_id', 'NDVI', 'NDBI']
    : ['city', 'year', 'NDVI', 'NDBI', 'latitude', 'longitude'];
  Export.table.toDrive({
    collection: fc,
    description: 's2_' + cityKey + '_' + year,
    folder: 'MIT_Summer_S2',
    fileFormat: 'CSV',
    selectors: selectors
  });
  print('Queued: ' + cityKey + ' ' + year +
        (USE_SAMPLE_REGIONS ? ' [sampleRegions]' : ' [sample fallback]'));
}

// ── Run ────────────────────────────────────────────────────────
// 2014-2017 → Landsat 8 (extractS2 branches on year < 2018);
// 2018-2024 → Sentinel-2 (S2_SR_HARMONIZED starts ~2018-03-28, 2018 partial).
var cities = Object.keys(CITIES);
var years  = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024];

// Test first:
exportCityYear('beijing', 2023);

// Then batch:
// cities.forEach(function(ck) { years.forEach(function(y) { exportCityYear(ck, y); }); });

print('Done. Check Tasks tab → Run exports.');
