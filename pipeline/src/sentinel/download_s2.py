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
from config.settings_s2 import settings_s2
from config.context import get_sh_config
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
    cfg = settings_s2()
    sh_cfg = get_sh_config()

    # Calculer la catégorie de couverture de mangroves
    try:
        # Import local pour éviter les imports circulaires
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dataset.mangrove_dataset import calculate_mangrove_coverage, get_coverage_category
        
        gmw_table = f"{cfg.PG_SCHEMA}.{cfg.PG_TABLE}"
        coverage = calculate_mangrove_coverage(bbox, gmw_table)
        category = get_coverage_category(coverage)
        print(f"[INFO] Couverture de mangroves calculée: {coverage:.2f}% - Catégorie: {category}")
    except Exception as e:
        print(f"[WARNING] Impossible de calculer la couverture de mangroves: {e}")
        print("[WARNING] Utilisation de la catégorie par défaut: no_mangroves")
        category = "no_mangroves"
        coverage = 0

    # Modifier les chemins de sortie pour inclure la catégorie
    categorized_bands_dir = cfg.BANDS_DIR / category / output_path.stem
    categorized_output_path = cfg.OUTPUT_DIR / category / output_path.name
    
    # Créer les répertoires si nécessaire
    categorized_bands_dir.mkdir(parents=True, exist_ok=True)
    categorized_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Récupère l'intervalle de temps centralisé (YYYY-MM-DD, YYYY-MM-DD)
    start_date, end_date = cfg.time_interval_tuple

    # Create temp directory for individual bands
    bands_dir = categorized_bands_dir
    bands_dir.mkdir(parents=True, exist_ok=True)

    # Calculate image size
    image_size = bbox_to_dimensions(bbox, resolution=cfg.IMAGE_RESOLUTION)
    
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
            size=image_size,
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
        band_file = bands_dir / f"{band_name}.tif"
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
        create_rgb_from_bands(bands_dir, categorized_output_path, image_size)
        print(f"[SUCCESS] Created RGB image: {categorized_output_path}")
        print(f"[INFO] Image classée dans la catégorie '{category}' ({coverage:.2f}% de couverture)")
        return True
    except ValueError as e:
        print(f"[WARNING] Skipping RGB creation for {categorized_output_path.stem}: {e}")
        # Clean up bands directory if RGB creation fails
        import shutil
        if bands_dir.exists():
            shutil.rmtree(bands_dir)
            print(f"[INFO] Cleaned up bands directory: {bands_dir}")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to create RGB image: {e}")
        # Clean up bands directory if RGB creation fails
        import shutil
        if bands_dir.exists():
            shutil.rmtree(bands_dir)
            print(f"[INFO] Cleaned up bands directory: {bands_dir}")
        return False

def run_download(bboxes: Iterable[BBox],
                 prefix: str = "patch",
                 enhanced: bool = True,
                 workers: int = 1) -> List[Path]:
    cfg = settings_s2()
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

def create_rgb_from_bands(bands_dir: Path, output_path: Path, size: Tuple[int, int] = None):
    """Crée une image RGB à partir des bandes B04, B03, B02 téléchargées."""
    try:
        # Chemins vers les bandes téléchargées
        b4_path = bands_dir / 'B04.tif'
        b3_path = bands_dir / 'B03.tif'
        b2_path = bands_dir / 'B02.tif'

        # Vérifier que tous les fichiers existent
        for band_path in [b4_path, b3_path, b2_path]:
            if not band_path.exists():
                raise FileNotFoundError(f"Le fichier de bande n'a pas été trouvé : {band_path}")

        # Lire les données de chaque bande
        with rasterio.open(b4_path) as src:
            r = src.read(1)
        with rasterio.open(b3_path) as src:
            g = src.read(1)
        with rasterio.open(b2_path) as src:
            b = src.read(1)

        # Normaliser les bandes pour l'affichage (les données sont en FLOAT32)
        def normalize(band):
            band_min, band_max = band.min(), band.max()
            if band_max > band_min:
                return ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
            return np.zeros_like(band, dtype=np.uint8)

        r_norm = normalize(r)
        g_norm = normalize(g)
        b_norm = normalize(b)

        # Empiler les bandes pour former une image RGB
        rgb_image = np.stack([r_norm, g_norm, b_norm], axis=-1)

        # Sauvegarder l'image
        img = Image.fromarray(rgb_image)
        img.save(output_path, 'PNG')
        print(f"[SUCCESS] Image RGB créée et sauvegardée dans {output_path}")

    except Exception as e:
        print(f"[ERROR] Échec de la création de l'image RGB : {e}")
        raise

