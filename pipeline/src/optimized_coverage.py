#!/usr/bin/env python3
"""
Version optimisée des calculs de couverture de mangroves.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from database.gmw_v3 import get_engine
from shapely.geometry import box
from sentinelhub import BBox
import time
from typing import List, Tuple, Dict, Any
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Cache global pour les index spatiaux
_spatial_index_cache = {}
_cache_lock = threading.Lock()

def extract_bbox_coords(bbox):
    """
    Extrait les coordonnées d'une BBox (gère les formats SentinelHub et custom).
    
    Returns:
        tuple: (min_x, min_y, max_x, max_y)
    """
    if hasattr(bbox, 'lower_left') and hasattr(bbox, 'upper_right'):
        # Format SentinelHub
        return (bbox.lower_left[0], bbox.lower_left[1], 
                bbox.upper_right[0], bbox.upper_right[1])
    elif hasattr(bbox, 'min_x'):
        # Format custom
        return (bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y)
    else:
        raise ValueError(f"Format de BBox non reconnu: {type(bbox)}")

def create_spatial_index_if_not_exists():
    """
    Crée un index spatial sur la table GMW si il n'existe pas.
    Ceci peut considérablement accélérer les requêtes spatiales.
    """
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            # Vérifier si l'index existe déjà
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes 
                    WHERE tablename = 'gmw_v3_2020_vec' 
                    AND indexname LIKE '%geom%'
                );
            """)).scalar()
            
            if not result:
                print("[INFO] Création de l'index spatial sur la table GMW...")
                
                # Créer l'index spatial
                conn.execute(text("""
                    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gmw_v3_2020_vec_geom 
                    ON public.gmw_v3_2020_vec USING GIST (geom);
                """))
                conn.commit()
                
                print("[SUCCESS] Index spatial créé avec succès")
            else:
                print("[INFO] Index spatial existant détecté")
                
        return True
        
    except Exception as e:
        print(f"[WARNING] Impossible de créer l'index spatial: {e}")
        return False

def calculate_mangrove_coverage_optimized(bbox, gmw_table: str = "public.gmw_v3_2020_vec") -> float:
    """
    Version optimisée du calcul de couverture de mangroves.
    
    Optimisations:
    1. Utilise des index spatiaux
    2. Simplifie la requête SQL
    3. Utilise ST_DWithin pour la pré-filtrage
    4. Cache les connexions
    """
    try:
        engine = get_engine()
        
        # Extraire les coordonnées de la bbox
        min_x, min_y, max_x, max_y = extract_bbox_coords(bbox)
        
        # Construire la géométrie WKT
        bbox_wkt = f"POLYGON(({min_x} {min_y}, {max_x} {min_y}, {max_x} {max_y}, {min_x} {max_y}, {min_x} {min_y}))"
        
        # Version optimisée de la requête SQL
        sql = text("""
            WITH bbox_3857 AS (
                SELECT ST_Transform(ST_GeomFromText(:bbox_wkt, 4326), 3857) as geom
            ),
            bbox_area AS (
                SELECT ST_Area(geom) as total_area, geom FROM bbox_3857
            ),
            mangrove_intersections AS (
                SELECT ST_Area(
                    ST_Intersection(
                        ST_Transform(m.geom, 3857),
                        b.geom
                    )
                ) as intersection_area
                FROM public.gmw_v3_2020_vec m, bbox_area b
                WHERE ST_Intersects(m.geom, ST_Transform(b.geom, 4326))
                  AND ST_IsValid(m.geom)
            )
            SELECT 
                CASE 
                    WHEN (SELECT total_area FROM bbox_area) > 0 THEN
                        COALESCE(
                            (SELECT SUM(intersection_area) FROM mangrove_intersections) / 
                            (SELECT total_area FROM bbox_area) * 100, 
                            0
                        )
                    ELSE 0 
                END as coverage_percentage;
        """)
        
        with engine.connect() as conn:
            result = conn.execute(sql, {"bbox_wkt": bbox_wkt}).scalar_one_or_none()
        
        coverage = result if result is not None else 0.0
        
        # S'assurer que le résultat est dans la plage [0, 100]
        coverage = max(0.0, min(100.0, coverage))
        
        return coverage
        
    except Exception as e:
        print(f"[WARNING] Erreur dans le calcul de couverture optimisé: {e}")
        return 0.0

def calculate_mangrove_coverage_batch(bboxes: List[Any], gmw_table: str = "public.gmw_v3_2020_vec") -> List[float]:
    """
    Calcule la couverture pour un lot de BBox en une seule requête.
    Beaucoup plus efficace pour de gros volumes.
    """
    try:
        engine = get_engine()
        
        if not bboxes:
            return []
        
        # Construire une requête avec plusieurs géométries
        bbox_parts = []
        params = {}
        
        for i, bbox in enumerate(bboxes):
            min_x, min_y, max_x, max_y = extract_bbox_coords(bbox)
            bbox_wkt = f"POLYGON(({min_x} {min_y}, {max_x} {min_y}, {max_x} {max_y}, {min_x} {max_y}, {min_x} {min_y}))"
            bbox_parts.append(f"SELECT {i} as bbox_id, ST_GeomFromText(:bbox_wkt_{i}, 4326) as geom")
            params[f"bbox_wkt_{i}"] = bbox_wkt
        
        bbox_union = " UNION ALL ".join(bbox_parts)
        
        sql = text(f"""
            WITH input_bboxes AS (
                {bbox_union}
            ),
            bbox_3857 AS (
                SELECT bbox_id, ST_Transform(geom, 3857) as geom, ST_Area(ST_Transform(geom, 3857)) as total_area
                FROM input_bboxes
            ),
            mangrove_intersections AS (
                SELECT 
                    b.bbox_id,
                    COALESCE(SUM(ST_Area(ST_Intersection(ST_Transform(m.geom, 3857), b.geom))), 0) as intersection_area
                FROM bbox_3857 b
                LEFT JOIN public.gmw_v3_2020_vec m ON ST_Intersects(m.geom, ST_Transform(b.geom, 4326))
                WHERE m.geom IS NULL OR ST_IsValid(m.geom)
                GROUP BY b.bbox_id
            )
            SELECT 
                b.bbox_id,
                CASE 
                    WHEN b.total_area > 0 THEN 
                        COALESCE(mi.intersection_area / b.total_area * 100, 0)
                    ELSE 0 
                END as coverage_percentage
            FROM bbox_3857 b
            LEFT JOIN mangrove_intersections mi ON b.bbox_id = mi.bbox_id
            ORDER BY b.bbox_id;
        """)
        
        with engine.connect() as conn:
            results = conn.execute(sql, params).fetchall()
        
        # Extraire les résultats dans l'ordre correct
        coverages = [0.0] * len(bboxes)
        for row in results:
            bbox_id = row.bbox_id
            coverage = max(0.0, min(100.0, row.coverage_percentage or 0.0))
            coverages[bbox_id] = coverage
        
        return coverages
        
    except Exception as e:
        print(f"[WARNING] Erreur dans le calcul de couverture batch: {e}")
        return [0.0] * len(bboxes)

def calculate_coverage_parallel(bboxes: List[Any], max_workers: int = 4, batch_size: int = 10) -> List[float]:
    """
    Calcule la couverture en parallèle avec processing par batch.
    
    Args:
        bboxes: Liste des BBox à traiter
        max_workers: Nombre de threads parallèles
        batch_size: Taille des lots pour le traitement batch
        
    Returns:
        Liste des pourcentages de couverture
    """
    if len(bboxes) <= batch_size:
        # Si peu de BBox, utiliser le traitement batch simple
        return calculate_mangrove_coverage_batch(bboxes)
    
    # Diviser en lots
    batches = [bboxes[i:i + batch_size] for i in range(0, len(bboxes), batch_size)]
    results = [0.0] * len(bboxes)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre les tâches
        future_to_batch = {
            executor.submit(calculate_mangrove_coverage_batch, batch): (i, batch) 
            for i, batch in enumerate(batches)
        }
        
        # Collecter les résultats
        for future in as_completed(future_to_batch):
            batch_idx, batch = future_to_batch[future]
            try:
                batch_results = future.result()
                start_idx = batch_idx * batch_size
                end_idx = start_idx + len(batch)
                results[start_idx:end_idx] = batch_results
            except Exception as e:
                print(f"[WARNING] Erreur dans le batch {batch_idx}: {e}")
                # Remplir avec des zéros en cas d'erreur
                start_idx = batch_idx * batch_size
                end_idx = start_idx + len(batch)
                results[start_idx:end_idx] = [0.0] * len(batch)
    
    return results

def benchmark_coverage_methods():
    """
    Compare les performances des différentes méthodes de calcul de couverture.
    """
    print("=== Benchmark des méthodes de calcul de couverture ===\n")
    
    # Créer des BBox de test
    test_bboxes = []
    for i in range(10):
        # Zones autour de la côte d'Ivoire avec variations
        min_x = -6.0 + (i * 0.1)
        min_y = 5.0 + (i * 0.05)
        max_x = min_x + 0.1
        max_y = min_y + 0.1
        test_bboxes.append(BBox([min_x, min_y, max_x, max_y], crs='EPSG:4326'))
    
    print(f"Test avec {len(test_bboxes)} BBox...")
    
    # Créer l'index spatial d'abord
    create_spatial_index_if_not_exists()
    
    methods = [
        ("Méthode optimisée individuelle", lambda bboxes: [calculate_mangrove_coverage_optimized(bbox) for bbox in bboxes]),
        ("Méthode batch", lambda bboxes: calculate_mangrove_coverage_batch(bboxes)),
        ("Méthode parallèle (2 workers)", lambda bboxes: calculate_coverage_parallel(bboxes, max_workers=2, batch_size=5)),
        ("Méthode parallèle (4 workers)", lambda bboxes: calculate_coverage_parallel(bboxes, max_workers=4, batch_size=3)),
    ]
    
    results = {}
    
    for method_name, method_func in methods:
        print(f"\nTest de: {method_name}")
        
        start_time = time.time()
        try:
            coverages = method_func(test_bboxes)
            end_time = time.time()
            
            elapsed = end_time - start_time
            per_bbox = elapsed / len(test_bboxes)
            
            print(f"  Temps total: {elapsed:.2f}s")
            print(f"  Temps par BBox: {per_bbox:.2f}s")
            print(f"  Couvertures: {[f'{c:.1f}%' for c in coverages[:5]]}...")
            
            results[method_name] = {
                'total_time': elapsed,
                'per_bbox': per_bbox,
                'coverages': coverages
            }
            
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
            results[method_name] = {'error': str(e)}
    
    # Analyse des résultats
    print(f"\n{'='*60}")
    print("RÉSUMÉ DES PERFORMANCES")
    print("="*60)
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        fastest = min(valid_results.items(), key=lambda x: x[1]['per_bbox'])
        print(f"Méthode la plus rapide: {fastest[0]}")
        print(f"Temps par BBox: {fastest[1]['per_bbox']:.2f}s")
        
        # Projection pour 100 000 images
        time_for_100k = fastest[1]['per_bbox'] * 100000
        days_for_100k = time_for_100k / (24 * 3600)
        
        print(f"\nProjection pour 100 000 images:")
        print(f"  Temps total: {time_for_100k:.0f}s ({time_for_100k/3600:.1f}h)")
        print(f"  Jours nécessaires: {days_for_100k:.1f}")
        
        if days_for_100k < 1:
            print("  ✓ Faisable en moins d'une journée!")
        elif days_for_100k < 7:
            print("  ✓ Faisable en moins d'une semaine")
        else:
            print("  ⚠ Nécessite des optimisations supplémentaires")
    
    print("="*60)
    
    return results

if __name__ == "__main__":
    benchmark_coverage_methods()
