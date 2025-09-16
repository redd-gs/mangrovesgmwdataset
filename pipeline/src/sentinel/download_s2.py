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


def scale(image, scale_factor=10000.0):
    image = image.astype(np.float32)
    image = image / scale_factor
    # Clamp les valeurs pour éviter les valeurs aberrantes (parfois dues aux nuages ou erreurs de traitement)
    image = np.clip(image, 0.0, 1.0)
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

# Normaliser les bandes pour l'affichage (les données sont en FLOAT32)
def normalize(band):
    band_min, band_max = band.min(), band.max()
    if band_max > band_min:
            return ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
    return np.zeros_like(band, dtype=np.uint8)


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



def validate_image_quality(image_data: np.ndarray, cfg) -> Tuple[bool, str]:
    """
    Valide la qualité d'une image téléchargée pour éviter les images noires/inutilisables.
    
    Args:
        image_data: Données de l'image sous forme de array numpy
        cfg: Configuration avec les seuils de qualité
        
    Returns:
        Tuple (is_valid, reason) - True si l'image est valide, False sinon avec la raison
    """
    try:
        if image_data is None or image_data.size == 0:
            return False, "Image vide ou non trouvée"
        
        # Vérifier que l'image n'est pas entièrement noire
        if np.all(image_data == 0):
            return False, "Image entièrement noire"
            
        # Calculer les statistiques de l'image
        valid_pixels = image_data[image_data > 0]
        
        if len(valid_pixels) == 0:
            return False, "Aucun pixel valide trouvé"
            
        # Ratio de pixels valides
        valid_ratio = len(valid_pixels) / image_data.size
        if valid_ratio < cfg.MIN_VALID_PIXELS_RATIO:
            return False, f"Trop peu de pixels valides ({valid_ratio:.2%} < {cfg.MIN_VALID_PIXELS_RATIO:.2%})"
        
        # Vérifier la luminosité moyenne
        mean_brightness = np.mean(valid_pixels)
        if mean_brightness < cfg.MIN_BRIGHTNESS_THRESHOLD:
            return False, f"Image trop sombre (luminosité moyenne: {mean_brightness:.3f})"
            
        if mean_brightness > cfg.MAX_BRIGHTNESS_THRESHOLD:
            return False, f"Image trop claire/saturée (luminosité moyenne: {mean_brightness:.3f})"
        
        # Vérifier le contraste (écart-type normalisé)
        if len(valid_pixels) > 1:
            contrast = np.std(valid_pixels) / (mean_brightness + 1e-6)  # Éviter division par zéro
            if contrast < cfg.MIN_CONTRAST_RATIO:
                return False, f"Contraste insuffisant ({contrast:.3f} < {cfg.MIN_CONTRAST_RATIO})"
        
        return True, "Image valide"
        
    except Exception as e:
        return False, f"Erreur lors de la validation: {str(e)}"


def download_single_with_retry(bbox: BBox,
                              output_path: Path,
                              enhanced: bool = False,
                              max_retries: int = None) -> bool:
    """
    Version améliorée de download_single avec retry en cas d'image de mauvaise qualité.
    
    Args:
        bbox: Bounding box à télécharger
        output_path: Chemin de sortie
        enhanced: Si True, applique des améliorations
        max_retries: Nombre max de tentatives (utilise config si None)
        
    Returns:
        True si succès, False sinon
    """
    cfg = settings_s2()
    retries = max_retries if max_retries is not None else cfg.RETRY_COUNT
    
    for attempt in range(retries):
        print(f"[INFO] Tentative {attempt + 1}/{retries} pour {output_path.stem}")
        
        # Essayer avec des fenêtres de temps légèrement décalées pour éviter les images nuageuses
        if attempt > 0:
            # Décaler la fenêtre temporelle pour essayer d'autres acquisitions
            start_date, end_date = cfg.time_interval_tuple
            from datetime import datetime, timedelta
            import random
            
            try:
                base_date = datetime.strptime(start_date, "%Y-%m-%d")
                # Décaler de -30 à +30 jours aléatoirement
                offset_days = random.randint(-30, 30)
                new_start = base_date + timedelta(days=offset_days)
                new_end = new_start + timedelta(days=60)  # Fenêtre de 60 jours
                
                modified_time_interval = (new_start.strftime("%Y-%m-%d"), new_end.strftime("%Y-%m-%d"))
                print(f"[INFO] Tentative avec fenêtre temporelle: {modified_time_interval}")
                
                # Temporairement modifier la config pour cette tentative
                original_interval = cfg.TIME_INTERVAL
                cfg.TIME_INTERVAL = f"{modified_time_interval[0]}/{modified_time_interval[1]}"
                
            except Exception as e:
                print(f"[WARNING] Impossible de modifier la fenêtre temporelle: {e}")
        
        # Essayer le téléchargement standard
        success = download_single(bbox, output_path, enhanced)
        
        # Restaurer la config originale si elle a été modifiée
        if attempt > 0 and 'original_interval' in locals():
            cfg.TIME_INTERVAL = original_interval
        
        if not success:
            print(f"[WARNING] Échec du téléchargement, tentative {attempt + 1}")
            continue
            
        # Vérifier la qualité de l'image téléchargée
        try:
            # Déterminer la catégorie pour trouver les bons chemins
            from utils.optimized_coverage import calculate_mangrove_coverage_optimized
            from utils.optimized_download import get_coverage_category
            
            try:
                gmw_table = f"{cfg.PG_SCHEMA}.{cfg.PG_TABLE}"
                coverage = calculate_mangrove_coverage_optimized(bbox, gmw_table)
                category = get_coverage_category(coverage)
            except:
                category = "0%"
            
            # Lire l'image pour validation
            categorized_output_path = get_categorized_output_path(output_path, cfg, category)
            if categorized_output_path.exists():
                # Lire une bande pour validation (on peut aussi lire l'image RGB)
                bands_dir = get_categorized_bands_dir(output_path, cfg, category)
                b04_path = bands_dir / 'B04.tif'
                
                if b04_path.exists():
                    with rasterio.open(b04_path) as src:
                        image_data = src.read(1)
                    
                    is_valid, reason = validate_image_quality(image_data, cfg)
                    
                    if is_valid:
                        print(f"[SUCCESS] Image valide: {reason}")
                        return True
                    else:
                        print(f"[WARNING] Image invalide: {reason}")
                        # Supprimer l'image invalide
                        cleanup_invalid_image(output_path, cfg, category)
                        continue
                        
        except Exception as e:
            print(f"[WARNING] Erreur lors de la validation: {e}")
            continue
    
    print(f"[ERROR] Échec après {retries} tentatives pour {output_path.stem}")
    return False


def get_categorized_output_path(output_path: Path, cfg, category: str = None) -> Path:
    """Obtient le chemin de sortie catégorisé."""
    if category is None:
        # Essayer de déterminer la catégorie depuis le chemin de sortie
        # ou utiliser une catégorie par défaut
        category = "0%"
    return cfg.OUTPUT_DIR / category / output_path.name


def get_categorized_bands_dir(output_path: Path, cfg, category: str = None) -> Path:
    """Obtient le répertoire des bandes catégorisées."""
    if category is None:
        category = "0%"
    return cfg.BANDS_DIR / category / output_path.stem


def cleanup_invalid_image(output_path: Path, cfg, category: str = None):
    """Nettoie les fichiers d'une image invalide."""
    try:
        import shutil
        
        # Supprimer l'image RGB
        categorized_output = get_categorized_output_path(output_path, cfg, category)
        if categorized_output.exists():
            categorized_output.unlink()
            
        # Supprimer le dossier des bandes
        bands_dir = get_categorized_bands_dir(output_path, cfg, category)
        if bands_dir.exists():
            shutil.rmtree(bands_dir)
            
        print(f"[INFO] Fichiers invalides supprimés pour {output_path.stem}")
        
    except Exception as e:
        print(f"[WARNING] Erreur lors du nettoyage: {e}")


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
        from utils.optimized_coverage import calculate_mangrove_coverage_optimized
        from utils.optimized_download import get_coverage_category
        
        gmw_table = f"{cfg.PG_SCHEMA}.{cfg.PG_TABLE}"
        coverage = calculate_mangrove_coverage_optimized(bbox, gmw_table)
        category = get_coverage_category(coverage)
        print(f"[INFO] Couverture de mangroves calculée: {coverage:.2f}% - Catégorie: {category}")
    except Exception as e:
        print(f"[WARNING] Impossible de calculer la couverture de mangroves: {e}")
        print("[WARNING] Utilisation de la catégorie par défaut: 0%")
        category = "0%"
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
    
    # Download individual bands with cloud filtering
    bands_data = {}
    for band_name, evalscript in BAND_EVALSCRIPTS.items():
        print(f"[INFO] Downloading band {band_name} with cloud cover < {cfg.MAX_CLOUD_COVER}%...")
        req = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(start_date, end_date),
                    mosaicking_order="leastCC",
                    maxcc=cfg.MAX_CLOUD_COVER / 100.0  # Conversion en ratio (0-1)
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
                # Valider la qualité de la bande téléchargée
                is_valid, reason = validate_image_quality(data[0], cfg)
                if not is_valid:
                    print(f"[WARNING] Bande {band_name} invalide: {reason}")
                    return False
                    
                bands_data[band_name] = data[0]
                print(f"[SUCCESS] Downloaded and validated band {band_name}")
            else:
                print(f"[ERROR] Failed to download band {band_name}")
                return False
        except Exception as e:
            print(f"[ERROR] Exception downloading band {band_name}: {e}")
            return False
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
        # Utiliser la version avec retry pour une meilleure qualité
        ok = download_single_with_retry(bb, out, enhanced=enhanced)
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

