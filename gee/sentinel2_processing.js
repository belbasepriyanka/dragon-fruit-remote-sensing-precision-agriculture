// Sentinel-2 precision-agriculture workflow template
// Author: Priyanka Belbase
// Replace the demonstration AOI with an actual field boundary before operational use.

var aoi = ee.Geometry.Rectangle([-80.55, 25.45, -80.25, 25.75]);

function maskS2Clouds(image) {
  var scl = image.select('SCL');
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  return image.updateMask(mask).divide(10000).copyProperties(image, ['system:time_start']);
}

function addIndices(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndre = image.normalizedDifference(['B8', 'B5']).rename('NDRE');
  var gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI');
  var ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI');
  return image.addBands([ndvi, ndre, gndvi, ndmi]);
}

var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate('2025-01-01', '2025-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .map(maskS2Clouds)
  .map(addIndices);

var median = collection.median().clip(aoi);
Map.centerObject(aoi, 11);
Map.addLayer(median.select('NDVI'), {min: 0.2, max: 0.9, palette: ['brown', 'yellow', 'green']}, 'Median NDVI');
Map.addLayer(median.select('NDRE'), {min: 0.05, max: 0.6, palette: ['purple', 'white', 'green']}, 'Median NDRE');

var series = ui.Chart.image.series({imageCollection: collection.select(['NDVI', 'NDRE']), region: aoi, reducer: ee.Reducer.mean(), scale: 20}).setOptions({title: 'Sentinel-2 vegetation-index time series', hAxis: {title: 'Date'}, vAxis: {title: 'Index value'}});
print(series);
