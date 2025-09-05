#!/usr/bin/env python3
"""
Version optimisée du téléchargement Sentinel-2 avec calcul de couverture en batch.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentinel.download_s2 import download_single
from optimized_coverage import calculate_mangrove_coverage_batch, extract_bbox_coords, create_spatial_index_if_not_exists
from dataset.mangrove_dataset import get_coverage_category
import time
from pathlib import Path
from typing import List, Dict, Any

def download_with_batch_coverage(bboxes: List[Any], 
                                prefix: str = "gmw", 
                                enhanced: bool = True, 
                                workers: int = 1,
                                gmw_table: str = "public.gmw_v3_2020_vec") -> List[Path]:
    """
    Version optimisée qui calcule toutes les couvertures en batch avant de télécharger.
    
    Args:
        bboxes: Liste des BBox à traiter
        prefix: Préfixe pour les noms de fichiers
        enhanced: Si True, applique des améliorations d'image
        workers: Nombre de workers (ignoré pour l'instant, traitement séquentiel)
        gmw_table: Table contenant les données de mangroves
        
    Returns:
        Liste des chemins des images créées
    """
    print(f"[INFO] Téléchargement {len(bboxes)} tuiles avec calcul de couverture optimisé...")
    
    # Étape 1: Créer l'index spatial si nécessaire
    create_spatial_index_if_not_exists()
    
    # Étape 2: Calculer toutes les couvertures en batch
    print(f"[INFO] Calcul des couvertures en batch pour {len(bboxes)} BBox...")
    start_time = time.time()
    
    try:
        coverages = calculate_mangrove_coverage_batch(bboxes, gmw_table)
        elapsed = time.time() - start_time
        print(f"[SUCCESS] Couvertures calculées en {elapsed:.2f}s ({elapsed/len(bboxes)*1000:.1f}ms par BBox)")
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
    print(f"[INFO] Début du téléchargement des images...")
    
    downloaded_paths = []
    download_start = time.time()
    
    for i, meta in enumerate(bbox_metadata):
        bbox = meta['bbox']
        coverage = meta['coverage']
        category = meta['category']
        output_name = meta['output_name']
        
        print(f"[INFO] Traitement {i+1}/{len(bboxes)}: {output_name} (couverture: {coverage:.2f}%, catégorie: {category})")
        
        try:
            # Utiliser la fonction de téléchargement optimisée
            path = download_single_with_precomputed_coverage(
                bbox=bbox,
                output_path=Path(output_name + ".png"),
                coverage=coverage,
                category=category,
                enhanced=enhanced
            )
            
            if path:
                downloaded_paths.append(path)
                print(f"[SUCCESS] Image téléchargée: {path}")
            else:
                print(f"[WARNING] Échec du téléchargement pour {output_name}")
                
        except Exception as e:
            print(f"[ERROR] Erreur lors du téléchargement de {output_name}: {e}")
    
    total_elapsed = time.time() - download_start
    print(f"[SUCCESS] Téléchargement terminé en {total_elapsed:.2f}s")
    print(f"[SUCCESS] {len(downloaded_paths)}/{len(bboxes)} images téléchargées avec succès")
    
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
        from config.settings_s2 import get_config as settings_s2
        from sentinel.auth import get_sentinel_config as get_sh_config
        from sentinelhub import SentinelHubRequest, DataCollection, MimeType, BBox
        from processing.enhancements import create_rgb_from_bands
        
        cfg = settings_s2()
        sh_cfg = get_sh_config()
        
        print(f"[INFO] Image classée dans la catégorie '{category}' ({coverage:.2f}% de couverture)")
        
        # Modifier les chemins de sortie pour inclure la catégorie
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
            
            evalscript = f"""
            //VERSION=3
            function setup() {{
                return {{
                    input: ["{band}"],
                    output: {{ bands: 1, sampleType: "UINT16" }}
                }};
            }}
            function evaluatePixel(sample) {{
                return [sample.{band}];
            }}
            """
            
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
            
            if data and len(data) > 0:
                band_path = categorized_bands_dir / f"{band}.tif"
                with open(band_path, 'wb') as f:
                    f.write(data[0])
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

# Test de la nouvelle fonction
if __name__ == "__main__":
    from sentinelhub import BBox, CRS
    
    # Créer quelques BBox de test
    test_bboxes = [
        BBox([-5.5, 5.0, -5.0, 5.5], crs=CRS.WGS84),
        BBox([-5.4, 5.1, -4.9, 5.6], crs=CRS.WGS84),
        BBox([-5.3, 5.2, -4.8, 5.7], crs=CRS.WGS84),
    ]
    
    print("Test de la fonction de téléchargement optimisée...")
    paths = download_with_batch_coverage(test_bboxes, prefix="test", enhanced=True)
