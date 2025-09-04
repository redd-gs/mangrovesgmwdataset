from pathlib import Path
from typing import Iterable, List, Tuple
import concurrent.futures as cf
import os
import numpy as np
from PIL import Image, ImageFilter
from sentinelhub import (
    BBox, CRS, MimeType, SentinelHubRequest,
    DataCollection, bbox_to_dimensions
)
import sentinelhub as sh
from config.settings import settings
from pipeline.src.config.context import get_sh_config
import rasterio
from rasterio.transform import from_bounds

# Individual band evalscripts for downloading separate TIFF files
BAND_EVALSCRIPTS = {
    'B02': """//VERSION=3
function setup(){
    return {
        input: [{bands:["B02"], units:"REFLECTANCE"}],
        output: [{id:"default", bands:1, sampleType:"FLOAT32"}]
    };
}
function evaluatePixel(s){
    return [s.B02];
}
""",
    'B03': """//VERSION=3
function setup(){
    return {
        input: [{bands:["B03"], units:"REFLECTANCE"}],
        output: [{id:"default", bands:1, sampleType:"FLOAT32"}]
    };
}
function evaluatePixel(s){
    return [s.B03];
}
""",
    'B04': """//VERSION=3
function setup(){
    return {
        input: [{bands:["B04"], units:"REFLECTANCE"}],
        output: [{id:"default", bands:1, sampleType:"FLOAT32"}]
    };
}
function evaluatePixel(s){
    return [s.B04];
}
"""
}

TRUE_COLOR_EVALSCRIPT = """//VERSION=3
function setup(){
    return {
        input: [{bands:["B02","B03","B04","dataMask"], units:"REFLECTANCE"}],
        output: [{id:"default", bands:3, sampleType:"FLOAT32"}]
    };
}
function evaluatePixel(s){
    return [s.B04, s.B03, s.B02];
}
"""

ENHANCED_COLOR_EVALSCRIPT = """//VERSION=3
function setup(){
    return {
        input: [{bands:["B02","B03","B04","dataMask"], units:"REFLECTANCE"}],
        output: [{id:"default", bands:3, sampleType:"FLOAT32"}]
    };
}
function evaluatePixel(s){
    // Légère accentuation canal rouge et vert pour distinguer végétation
    return [s.B04*1.2, s.B03*1.1, s.B02];
}
"""

DEBUG = os.getenv("DEBUG_SH", "1") not in ("0", "false", "False")  # mettre DEBUG_SH=0 pour désactiver

DEBUG_EVALSCRIPT = """//VERSION=3
function setup(){
    return {
        input: [{bands:["B02","B03","B04","dataMask"], units:"REFLECTANCE"}],
        output: [
            {id:"rgb", bands:3, sampleType:"FLOAT32"},
            {id:"mask", bands:1, sampleType:"UINT8"}
        ]
    };
}
function evaluatePixel(s){
    return {
        rgb: [s.B04, s.B03, s.B02],
        mask: [s.dataMask]
    };
}
"""

def download_single(bbox: BBox,
                    output_path: Path,
                    enhanced: bool = False) -> bool:
    cfg = settings()
    sh_cfg = get_sh_config()

    # Récupère l'intervalle de temps centralisé (YYYY-MM-DD, YYYY-MM-DD)
    start_date, end_date = cfg.time_interval_tuple

    # Create temp directory for individual bands
    temp_dir = cfg.TEMP_DIR / output_path.stem
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Download individual bands
    bands_data = {}
    for band_name, evalscript in BAND_EVALSCRIPTS.items():
        print(f"[INFO] Downloading band {band_name}...")
        req = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(start_date, end_date),
                    mosaicking_order="leastCC"
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=bbox_to_dimensions(bbox, resolution=cfg.IMAGE_RESOLUTION),
            config=sh_cfg
        )

        try:
            data = req.get_data()
            if data and isinstance(data[0], np.ndarray):
                bands_data[band_name] = data[0]
                print(f"[SUCCESS] Downloaded band {band_name}")
            else:
                print(f"[ERROR] Failed to download band {band_name}")
                return False
        except Exception as e:
            print(f"[ERROR] Exception downloading band {band_name}: {e}")
            return False

    # Save individual bands as TIFF files
    transform = from_bounds(*bbox, bands_data['B02'].shape[1], bands_data['B02'].shape[0])
    for band_name, band_data in bands_data.items():
        band_file = temp_dir / f"{band_name}.tif"
        # Save without CRS to avoid PROJ issues
        with rasterio.open(
            band_file,
            'w',
            driver='GTiff',
            height=band_data.shape[0],
            width=band_data.shape[1],
            count=1,
            dtype=band_data.dtype,
            transform=transform
        ) as dst:
            dst.write(band_data, 1)
        print(f"[INFO] Saved {band_file}")

    # Create RGB image from bands
    try:
        create_rgb_from_bands(temp_dir, output_path)
        print(f"[SUCCESS] Created RGB image: {output_path}")
        return True
    except ValueError as e:
        print(f"[WARNING] Skipping RGB creation for {output_path.stem}: {e}")
        # Clean up temp directory if RGB creation fails
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"[INFO] Cleaned up temp directory: {temp_dir}")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to create RGB image: {e}")
        # Clean up temp directory if RGB creation fails
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"[INFO] Cleaned up temp directory: {temp_dir}")
        return False

def run_download(bboxes: Iterable[BBox],
                 prefix: str = "patch",
                 enhanced: bool = True,
                 workers: int = 1) -> List[Path]:
    cfg = settings()
    results: List[Path] = []

    def task(item):
        idx, bb = item
        out = cfg.OUTPUT_DIR / f"{prefix}_{idx}.png"
        ok = download_single(bb, out, enhanced=enhanced)
        return out if ok else None

    items = list(enumerate(bboxes, start=1))
    if workers > 1:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for p in ex.map(task, items):
                if p:
                    results.append(p)
    else:
        for it in items:
            p = task(it)
            if p:
                results.append(p)
    return results

if __name__ == "__main__":
    # Test manuel optionnel (désactivé par défaut pour éviter erreurs lors d'import)
    DEBUG_RUN = False

def scale(image, a=0.0000275, b=0.2):
    # Convertit les valeurs brutes du capteur en réflectance de surface (valeurs physiques)
    # Les Digital Numbers (DN) représentent les valeurs numériques brutes enregistrées par le capteur Landsat pour chaque pixel.
    # Ces valeurs sont codées sur 16 bits (uint16), allant généralement de 0 à 65535, et doivent être converties en valeurs physiques (réflectance) pour l'analyse.
    # Chaque valeur numérique brute (Digital Number, DN) enregistrée par le capteur pour chaque pixel représente l'intensité du signal réfléchi par la surface terrestre dans une bande spectrale donnée, telle que mesurée par le satellite.
    # 'a' et 'b' sont les coefficients de calibration fournis dans la documentation Landsat pour convertir les DN en réflectance de surface (valeurs physiques comprises entre 0 et 1).
    # Ces valeurs spécifiques de 'a' et 'b' proviennent des métadonnées du produit Landsat (voir le fichier MTL.txt associé à chaque image) et peuvent varier selon la scène ou le capteur.
    image = image.astype(np.float32)
    image = (image * a) + b
    return image

def crop(image, xmin, xmax, ymin, ymax):
    # Découpe l'image pour ne garder qu'une zone d'intérêt (sous-image)
    # xmin/xmax/ymin/ymax définissent la fenêtre spatiale à extraire
    return image[ymin:ymax, xmin:xmax]

def process(band, xmin, xmax, ymin, ymax):
    # Combine les deux étapes précédentes :
    # 1. Transpose la bande pour avoir les dimensions (hauteur, largeur, canaux)
    # 2. Découpe la zone d'intérêt
    # 3. Applique la conversion en réflectance
    return scale(crop(band.transpose((1, 2, 0)), xmin, xmax, ymin, ymax))

def create_rgb_from_bands(temp_dir: Path, output_path: Path):
    """Create RGB image from individual band TIFF files."""
    # Mapping for RGB bands
    sentinel2_mapping = {
        'red': 'B04',
        'green': 'B03',
        'blue': 'B02',
        "nir": "B5",
        "swir1": "B6",
        "swir2": "B7",
    }

    data = {}
    for band_name, band_code in sentinel2_mapping.items():
        band_file = temp_dir / f'{band_code}.tif'
        if not band_file.exists():
            raise FileNotFoundError(f"Band file not found: {band_file}")

        with rasterio.open(band_file) as src:
            band_data = src.read(1)  # Read first band

            # Check if band data is valid (not empty and has reasonable size)
            if band_data.size == 0 or band_data.shape[0] == 0 or band_data.shape[1] == 0:
                raise ValueError(f"Band {band_code} has invalid dimensions: {band_data.shape}")

            # Check if band data has meaningful values (not all zeros or NaN)
            if np.all(band_data == 0) or np.all(np.isnan(band_data)):
                raise ValueError(f"Band {band_code} contains no valid data (all zeros or NaN)")

            # Normalize to 0-1 range (assuming reflectance values)
            if band_data.max() > 1.0:
                band_data = band_data / 10000.0  # Sentinel-2 reflectance scaling

            data[band_name] = band_data

    # Stack bands into RGB
    rgb = np.dstack((data['red'], data['green'], data['blue']))

    # Check if RGB has valid data
    if rgb.size == 0 or np.all(rgb == 0):
        raise ValueError("RGB composite contains no valid data")

    # Apply normalization like plt.imshow((rgb - rgb.min()) / (rgb.max() - rgb.min()))
    rgb_min = rgb.min()
    rgb_max = rgb.max()

    if rgb_max > rgb_min:
        rgb_normalized = (rgb - rgb_min) / (rgb_max - rgb_min)
    else:
        # If all values are the same, create a gray image
        rgb_normalized = np.full_like(rgb, 0.5, dtype=np.float32)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the normalized RGB image directly using matplotlib
    import matplotlib.pyplot as plt
    plt.imsave(output_path, rgb_normalized)
    print(f"[INFO] Saved normalized RGB image: {output_path}")

