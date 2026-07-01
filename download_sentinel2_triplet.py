import ee
import numpy as np
import requests
import zipfile
import json
import io
from pathlib import Path
from datetime import datetime, timezone

BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
SCALE_METERS = 10
UTM_CRS = 'EPSG:32643'
MAX_CLOUD_PERCENT = 5
START_DATE = '2024-01-01'
END_DATE = '2024-03-30'
REFLECTANCE_SCALE = 10000.0
OUTPUT_DIR = Path('sentinel2_triplet')
FRAME_LABELS = ['frame_a', 'frame_b', 'frame_c']
REQUEST_TIMEOUT = 600

def initialize_earth_engine():
    try:
        ee.Initialize(project='isrobah-500712')
    except Exception:
        ee.Authenticate()
        ee.Initialize(project='isrobah-500712')

def create_punjab_roi():
    return ee.Geometry.Rectangle([75.82, 30.88, 75.87, 30.93])

def build_filtered_collection(roi):
    return (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_PERCENT))
        .select(BANDS)
        .sort('system:time_start')
    )

def retrieve_chronological_triplet(collection):
    image_list = collection.toList(collection.size())
    available_count = image_list.size().getInfo()
    if available_count < 3:
        raise RuntimeError(
            f"Insufficient cloud-free images: {available_count} found, "
            f"3 required. Expand date range or increase MAX_CLOUD_PERCENT."
        )
    print(f"  Found {available_count} cloud-free images in collection")
    images = []
    dates = []
    cloud_pcts = []
    for idx in range(3):
        image = ee.Image(image_list.get(idx))
        props = image.getInfo()['properties']
        timestamp_ms = props['system:time_start']
        cloud_pct = props.get('CLOUDY_PIXEL_PERCENTAGE', -1)
        acq_date = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        images.append(image)
        dates.append(acq_date)
        cloud_pcts.append(cloud_pct)
    return images, dates, cloud_pcts

def compute_utm_grid(roi):
    bounds = roi.transform(UTM_CRS, 1).bounds(1, UTM_CRS).getInfo()['coordinates'][0]
    xs = [p[0] for p in bounds]
    ys = [p[1] for p in bounds]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = int(np.ceil((max_x - min_x) / SCALE_METERS))
    height = int(np.ceil((max_y - min_y) / SCALE_METERS))
    
    return {
        'dimensions': {'width': width, 'height': height},
        'affineTransform': {
            'scaleX': SCALE_METERS, 'shearX': 0, 'translateX': min_x,
            'shearY': 0, 'scaleY': -SCALE_METERS, 'translateY': max_y,
        },
        'crsCode': UTM_CRS,
    }

def download_geotiff(image, roi, output_path):
    url = image.getDownloadURL({
        'bands': BANDS,
        'region': roi.getInfo(),
        'scale': SCALE_METERS,
        'crs': UTM_CRS,
        'format': 'GEO_TIFF',
    })
    response = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
    response.raise_for_status()
    raw_bytes = response.content
    if raw_bytes[:4] == b'PK\x03\x04':
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            tif_entries = [n for n in zf.namelist() if n.lower().endswith('.tif')]
            if not tif_entries:
                raise RuntimeError("ZIP archive contains no .tif files")
            tif_data = zf.read(tif_entries[0])
            output_path.write_bytes(tif_data)
    else:
        output_path.write_bytes(raw_bytes)
    return output_path

def image_to_numpy(image, grid):
    structured = ee.data.computePixels({
        'expression': image,
        'fileFormat': 'NUMPY_NDARRAY',
        'grid': grid,
    })
    band_names = list(structured.dtype.names)
    channels = [structured[name] for name in band_names]
    return np.stack(channels, axis=-1).astype(np.float32)

def normalize_reflectance(array):
    return np.clip(array / REFLECTANCE_SCALE, 0.0, 1.0)

def save_triplet_metadata(dates, cloud_pcts, arrays, output_dir):
    metadata = {
        'collection': 'COPERNICUS/S2_SR_HARMONIZED',
        'roi': 'Punjab, India [75.82-75.87E, 30.88-30.93N]',
        'crs': UTM_CRS,
        'scale_meters': SCALE_METERS,
        'bands': BANDS,
        'normalization': f'reflectance / {REFLECTANCE_SCALE}',
        'frames': {},
    }
    for label, date, cloud, arr in zip(FRAME_LABELS, dates, cloud_pcts, arrays):
        metadata['frames'][label] = {
            'acquisition_date': date.strftime('%Y-%m-%d'),
            'cloud_cover_pct': round(cloud, 2),
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
            'min': round(float(arr.min()), 6),
            'max': round(float(arr.max()), 6),
            'mean': round(float(arr.mean()), 6),
        }
    gap_ab = (dates[1] - dates[0]).days
    gap_bc = (dates[2] - dates[1]).days
    metadata['temporal_gaps_days'] = {'A_to_B': gap_ab, 'B_to_C': gap_bc, 'A_to_C': gap_ab + gap_bc}
    meta_path = output_dir / 'triplet_metadata.json'
    meta_path.write_text(json.dumps(metadata, indent=2))
    return meta_path

def main():
    print("Initializing Earth Engine...")
    initialize_earth_engine()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    roi = create_punjab_roi()
    
    collection = build_filtered_collection(roi)
    images, dates, cloud_pcts = retrieve_chronological_triplet(collection)
    
    grid = compute_utm_grid(roi)
    print(f"Grid dimensions: {grid['dimensions']['width']}x{grid['dimensions']['height']}")
    
    triplet_arrays = []
    for image, date, cloud, label in zip(images, dates, cloud_pcts, FRAME_LABELS):
        tif_path = OUTPUT_DIR / f'{label}.tif'
        npy_path = OUTPUT_DIR / f'{label}.npy'
        
        download_geotiff(image, roi, tif_path)
        array = image_to_numpy(image, grid)
        array = normalize_reflectance(array)
        np.save(npy_path, array)
        triplet_arrays.append(array)
        
        print(f"Saved {label} | Shape: {array.shape}")
        
    save_triplet_metadata(dates, cloud_pcts, triplet_arrays, OUTPUT_DIR)
    print("Extraction complete.")
    return triplet_arrays

if __name__ == '__main__':
    triplet = main()