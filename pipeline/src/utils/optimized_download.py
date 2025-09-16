"""
Version optimisée du téléchargement Sentinel-2 avec calcul de couverture en batch.
"""

import sys
import os
import numpy as np
import concurrent.futures as cf
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentinelhub import BBox, CRS
from sentinel.download_s2 import download_single
from utils.optimized_coverage import calculate_mangrove_coverage_batch, extract_bbox_coords, create_spatial_index_if_not_exists
import time
from pathlib import Path
from typing import List, Dict, Any
from config.settings_s2 import get_config as settings_s2
from sentinel.auth import get_sentinel_config as get_sh_config
from sentinelhub import SentinelHubRequest, DataCollection, MimeType, BBox
from processing.enhancements import create_rgb_from_bands


def get_coverage_category(coverage: float) -> str:
    """
    Détermine la catégorie de couverture basée sur le pourcentage de mangroves.
    
    Args:
        coverage: Pourcentage de couverture (0-100)
        
    Returns:
        Nom de la catégorie basé sur les tranches de pourcentage
    """
    if coverage >= 80:
        return "80-100%"
    elif coverage >= 60:
        return "60-80%"
    elif coverage >= 40:
        return "40-60%"
    elif coverage >= 20:
        return "20-40%"
    elif coverage > 0:
        return "1-20%"
    else:
        return "0%"


def download_with_batch_coverage(bboxes: List[Any], 
                                prefix: str = "gmw", 
                                enhanced: bool = True, 
                                workers: int = 4,
                                gmw_table: str = "public.gmw_v3_2020_vec") -> List[Path]:
    """
    Version optimisée qui calcule toutes les couvertures en batch avant de télécharger.
    
    Args:
        bboxes: Liste des BBox à traiter
        prefix: Préfixe pour les noms de fichiers
        enhanced: Si True, applique des améliorations d'image
        workers: Nombre de workers pour parallélisation des téléchargements
        gmw_table: Table contenant les données de mangroves
        
    Returns:
        Liste des chemins des images créées
    """
    print(f"[INFO] Téléchargement {len(bboxes)} tuiles avec calcul de couverture optimisé...")
    
    # Import pour timing détaillé
    
    # Étape 1: Créer l'index spatial si nécessaire
    index_start = time.time()
    create_spatial_index_if_not_exists()
    index_time = time.time() - index_start
    print(f"[SUCCESS] Index spatial créé en {index_time:.2f}s")

    # Étape 2: Calculer toutes les couvertures en batch
    print(f"[INFO] Calcul des couvertures en batch pour {len(bboxes)} BBox...")
    coverage_start = time.time()
    
    try:
        coverages = calculate_mangrove_coverage_batch(bboxes, gmw_table)
        coverage_time = time.time() - coverage_start
        print(f"[SUCCESS] Couvertures calculées en {coverage_time:.2f}s ({coverage_time/len(bboxes)*1000:.1f}ms par BBox)")
    except Exception as e:
        print(f"[WARNING] Erreur dans le calcul batch: {e}")
        print("[WARNING] Basculement vers le calcul individuel...")
        coverages = [0.0] * len(bboxes)
    
    # Étape 3: Préparer les métadonnées pour chaque BBox
    bbox_metadata = []
    for i, (bbox, coverage) in enumerate(zip(bboxes, coverages)):
        category = get_coverage_category(coverage)
        
        bbox_metadata.append({
            'bbox': bbox,
            'coverage': coverage,
            'category': category,
            'index': i,
            'output_name': f"{prefix}_{i+1}"
        })
    
    # Afficher un résumé des catégories
    category_counts = {}
    for meta in bbox_metadata:
        cat = meta['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"[INFO] Répartition des catégories:")
    for category, count in category_counts.items():
        if category is not None:
            print(f"  {category}: {count} images ({count/len(bboxes)*100:.1f}%)")
        else:
            print(f"  [Catégorie inconnue]: {count} images ({count/len(bboxes)*100:.1f}%)")
    
    # Étape 4: Télécharger les images avec les métadonnées pré-calculées
    print(f"[INFO] Début du téléchargement des images avec {workers} workers...")
    
    downloaded_paths = []
    download_start = time.time()

    def download_single_task(meta_with_index):
        """Fonction pour télécharger une seule image (pour parallélisation)"""
        i, meta = meta_with_index
        bbox = meta['bbox']
        coverage = meta['coverage']
        category = meta['category']
        output_name = meta['output_name']
        
        print(f"[INFO] Traitement {i+1}/{len(bboxes)}: {output_name} (couverture: {coverage:.2f}%, catégorie: {category})")
        
        try:
            # Mesurer le temps pour chaque image
            single_start = time.time()
            
            # Utiliser la fonction de téléchargement optimisée
            path = download_single_with_precomputed_coverage(
                bbox=bbox,
                output_path=Path(output_name + ".png"),
                coverage=coverage,
                category=category,
                enhanced=enhanced
            )
            
            single_time = time.time() - single_start
            
            if path:
                print(f"[SUCCESS] Image téléchargée: {path} (temps: {single_time:.1f}s)")
                return path
            else:
                print(f"[WARNING] Échec du téléchargement pour {output_name}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Erreur lors du téléchargement de {output_name}: {e}")
            return None
    
    # Préparer les tâches avec indices
    tasks = list(enumerate(bbox_metadata))
    
    # Téléchargement en parallèle ou séquentiel selon le nombre de workers
    if workers > 1:
        with cf.ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(download_single_task, tasks))
        downloaded_paths = [path for path in results if path is not None]
    else:
        # Mode séquentiel pour workers=1
        for task in tasks:
            path = download_single_task(task)
            if path:
                downloaded_paths.append(path)
    
    total_download_time = time.time() - download_start
    print(f"[SUCCESS] Téléchargement terminé en {total_download_time:.2f}s")
    print(f"[SUCCESS] {len(downloaded_paths)}/{len(bboxes)} images téléchargées avec succès")
    
    # Statistiques détaillées
    if len(downloaded_paths) > 0:
        avg_time_per_image = total_download_time / len(downloaded_paths)
        print(f"[TIMING] Temps moyen par image: {avg_time_per_image:.2f}s")
        print(f"[TIMING] Vitesse: {len(downloaded_paths)/(total_download_time/60):.1f} images/minute")
    
    return downloaded_paths

def download_single_with_precomputed_coverage(bbox: Any,
                                            output_path: Path,
                                            coverage: float,
                                            category: str,
                                            enhanced: bool = False) -> Path:
    """
    Version modifiée de download_single qui utilise une couverture pré-calculée.
    """
    try:
        # Import des modules nécessaires        
        cfg = settings_s2()
        sh_cfg = get_sh_config()
        
        print(f"[INFO] Image classée dans la catégorie '{category}' ({coverage:.2f}% de couverture)")
        
        # Organiser les fichiers par catégorie dans les dossiers existants
        categorized_bands_dir = cfg.BANDS_DIR / category / output_path.stem
        categorized_output_path = cfg.OUTPUT_DIR / category / output_path.name
        
        # Créer les répertoires si nécessaire
        categorized_bands_dir.mkdir(parents=True, exist_ok=True)
        categorized_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Récupère l'intervalle de temps
        time_interval = cfg.time_interval_tuple
        
        # Télécharger les bandes nécessaires
        bands_to_download = ['B02', 'B03', 'B04']  # RGB
        band_paths = {}
        
        for band in bands_to_download:
            print(f"[INFO] Downloading band {band}...")
            
            evalscript = f"""//VERSION=3
function setup() {{
    return {{
        input: [{{bands:["{band}"], units:"REFLECTANCE"}}],
        output: [{{id:"default", bands:1, sampleType:"FLOAT32"}}]
    }};
}}
function evaluatePixel(sample) {{
    return [sample.{band}];
}}"""
            
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=time_interval,
                        mosaicking_order='leastCC'
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF)
                ],
                bbox=bbox,
                size=(cfg.PATCH_SIZE_PX, cfg.PATCH_SIZE_PX),
                config=sh_cfg
            )
            
            data = request.get_data()
            
            if data and len(data) > 0 and isinstance(data[0], np.ndarray):
                # Sauvegarder avec rasterio au lieu d'écrire les bytes bruts
                band_path = categorized_bands_dir / f"{band}.tif"
                band_data = data[0]
                
                # Utiliser rasterio pour sauvegarder correctement
                import rasterio
                from rasterio.transform import from_bounds
                
                transform = from_bounds(*bbox, band_data.shape[1], band_data.shape[0])
                
                with rasterio.open(
                    band_path,
                    'w',
                    driver='GTiff',
                    height=band_data.shape[0],
                    width=band_data.shape[1],
                    count=1,
                    dtype=band_data.dtype,
                    transform=transform
                ) as dst:
                    dst.write(band_data, 1)
                
                band_paths[band] = band_path
                print(f"[INFO] Saved {band_path}")
                print(f"[SUCCESS] Downloaded band {band}")
            else:
                print(f"[ERROR] Pas de données pour la bande {band}")
                return None
        
        # Créer l'image RGB
        if len(band_paths) == 3:
            rgb_success = create_rgb_from_bands(
                band_paths['B04'],  # Red
                band_paths['B03'],  # Green  
                band_paths['B02'],  # Blue
                str(categorized_output_path),
                enhanced=enhanced
            )
            
            if rgb_success:
                print(f"[SUCCESS] Created RGB image: {categorized_output_path}")
                return categorized_output_path
            else:
                print(f"[ERROR] Échec de la création de l'image RGB")
                return None
        else:
            print(f"[ERROR] Bandes manquantes pour créer l'image RGB")
            return None
            
    except Exception as e:
        print(f"[ERROR] Erreur lors du téléchargement: {e}")
        return None

def download_with_predefined_categories(bbox_with_categories: List[tuple], 
                                       prefix: str = "gmw", 
                                       enhanced: bool = True, 
                                       workers: int = 4) -> List[Path]:
    """
    Télécharge des images avec des catégories pré-définies.
    
    Args:
        bbox_with_categories: Liste de tuples (bbox, category)
        prefix: Préfixe pour les noms de fichiers
        enhanced: Si True, applique des améliorations d'image
        workers: Nombre de workers pour parallélisation
        
    Returns:
        Liste des chemins des images créées
    """
    print(f"[INFO] Téléchargement {len(bbox_with_categories)} tuiles avec catégories pré-définies...")
    
    downloaded_paths = []
    download_start = time.time()
    
    # Compter par catégorie pour suivi
    from collections import defaultdict
    category_counters = defaultdict(int)

    def download_single_task(item_with_index):
        """Fonction pour télécharger une seule image avec catégorie pré-définie"""
        i, (bbox, target_category) = item_with_index
        
        # Incrémenter le compteur pour cette catégorie
        category_counters[target_category] += 1
        current_count = category_counters[target_category]
        
        output_name = f"{prefix}_{target_category.replace('%', 'pct').replace('-', '_')}_{current_count:03d}"
        
        print(f"[INFO] Traitement {i+1}/{len(bbox_with_categories)}: {output_name} (catégorie: {target_category})")
        
        try:
            single_start = time.time()
            
            # Télécharger directement dans la catégorie cible
            path = download_single_with_fixed_category(
                bbox=bbox,
                output_path=Path(output_name + ".png"),
                target_category=target_category,
                enhanced=enhanced
            )
            
            single_time = time.time() - single_start
            
            if path:
                print(f"[SUCCESS] Image téléchargée: {path} (temps: {single_time:.1f}s)")
                return path
            else:
                print(f"[WARNING] Échec du téléchargement pour {output_name}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Erreur lors du téléchargement de {output_name}: {e}")
            return None
    
    # Préparer les tâches avec indices
    tasks = list(enumerate(bbox_with_categories))
    
    # Téléchargement en parallèle ou séquentiel
    if workers > 1:
        with cf.ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(download_single_task, tasks))
        downloaded_paths = [path for path in results if path is not None]
    else:
        for task in tasks:
            path = download_single_task(task)
            if path:
                downloaded_paths.append(path)
    
    total_download_time = time.time() - download_start
    print(f"[SUCCESS] Téléchargement terminé en {total_download_time:.2f}s")
    print(f"[SUCCESS] {len(downloaded_paths)}/{len(bbox_with_categories)} images téléchargées avec succès")
    
    return downloaded_paths


def download_single_with_fixed_category(bbox: Any,
                                       output_path: Path,
                                       target_category: str,
                                       enhanced: bool = False) -> Path:
    """
    Télécharge une image en utilisant une catégorie fixe pré-définie.
    """
    try:
        cfg = settings_s2()
        sh_cfg = get_sh_config()
        
        # Organiser les fichiers directement dans la catégorie cible
        categorized_bands_dir = cfg.BANDS_DIR / target_category / output_path.stem
        categorized_output_path = cfg.OUTPUT_DIR / target_category / output_path.name
        
        # Créer les répertoires si nécessaire
        categorized_bands_dir.mkdir(parents=True, exist_ok=True)
        categorized_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Récupère l'intervalle de temps
        time_interval = cfg.time_interval_tuple
        
        # Télécharger les bandes nécessaires
        bands_to_download = ['B02', 'B03', 'B04']  # RGB
        band_paths = {}
        
        for band in bands_to_download:
            evalscript = f"""//VERSION=3
function setup() {{
    return {{
        input: [{{bands:["{band}"], units:"REFLECTANCE"}}],
        output: [{{id:"default", bands:1, sampleType:"FLOAT32"}}]
    }};
}}
function evaluatePixel(sample) {{
    return [sample.{band}];
}}"""
            
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=time_interval,
                        mosaicking_order='leastCC'
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF)
                ],
                bbox=bbox,
                size=(cfg.PATCH_SIZE_PX, cfg.PATCH_SIZE_PX),
                config=sh_cfg
            )
            
            data = request.get_data()
            
            if data and len(data) > 0 and isinstance(data[0], np.ndarray):
                band_path = categorized_bands_dir / f"{band}.tif"
                band_data = data[0]
                
                import rasterio
                from rasterio.transform import from_bounds
                
                transform = from_bounds(*bbox, band_data.shape[1], band_data.shape[0])
                
                with rasterio.open(
                    band_path,
                    'w',
                    driver='GTiff',
                    height=band_data.shape[0],
                    width=band_data.shape[1],
                    count=1,
                    dtype=band_data.dtype,
                    transform=transform
                ) as dst:
                    dst.write(band_data, 1)
                
                band_paths[band] = band_path
            else:
                print(f"[ERROR] Pas de données pour la bande {band}")
                return None
        
        # Créer l'image RGB
        if len(band_paths) == 3:
            rgb_success = create_rgb_from_bands(
                band_paths['B04'],  # Red
                band_paths['B03'],  # Green  
                band_paths['B02'],  # Blue
                str(categorized_output_path),
                enhanced=enhanced
            )
            
            if rgb_success:
                return categorized_output_path
            else:
                return None
        else:
            return None
            
    except Exception as e:
        print(f"[ERROR] Erreur lors du téléchargement: {e}")
        return None
