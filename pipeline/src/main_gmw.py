from pathlib import Path
from typing import List
import os
import subprocess
import time
from shapely import wkb
from sqlalchemy import text
from sentinelhub import BBox, CRS
import random

from config.settings_s2 import settings_s2
from config.context import get_engine
from sentinel.download_s2 import run_download
from processing.bbox import create_valid_bbox  # Utiliser une bbox de taille fixe autour du centroïde
from utils.optimized_download import download_with_predefined_categories


def fetch_geometries_by_category(limit_per_category: int) -> List:
    """Retourne 'limit_per_category' géométries aléatoires pour chaque catégorie de mangrove.
    """
    cfg = settings_s2()
    full_table = f'"{cfg.PG_SCHEMA}"."{cfg.PG_TABLE}"'
    
    # Récupérer les catégories disponibles
    categories_sql = text(
        f"""
        SELECT DISTINCT pxlval
        FROM {full_table}
        WHERE geom IS NOT NULL AND NOT ST_IsEmpty(geom) AND pxlval IS NOT NULL
        ORDER BY pxlval
        """
    )
    
    rows: List = []
    with get_engine().connect() as conn:
        # Obtenir toutes les catégories
        categories = [row[0] for row in conn.execute(categories_sql)]
        print(f"[INFO] Catégories trouvées: {categories}")
        
        # Pour chaque catégorie, récupérer limit_per_category géométries
        for category in categories:
            print(f"[INFO] Récupération de {limit_per_category} géométries pour la catégorie {category}")
            
            sql = text(
                f"""
                SELECT ST_AsBinary(ST_PointOnSurface(geom))
                FROM {full_table}
                WHERE geom IS NOT NULL AND NOT ST_IsEmpty(geom) AND pxlval = :category
                ORDER BY random()
                LIMIT :lim
                """
            )
            
            for (gbytes,) in conn.execute(sql, {"category": category, "lim": limit_per_category}):
                if not gbytes:
                    continue
                try:
                    geom = wkb.loads(bytes(gbytes))
                    if geom.is_empty:
                        continue
                    rows.append(geom)
                except Exception:
                    continue
    
    # Shuffle pour mélanger les catégories
    random.shuffle(rows)
    print(f"[INFO] Total de {len(rows)} géométries récupérées")
    return rows


def generate_bboxes_from_gmw_v3_by_category_fixed(max_patches: int, patch_size_m: int = 512) -> List[tuple]:
    """
    Génère des bboxes équitablement réparties entre les catégories de mangrove.
    Retourne une liste de tuples (bbox, target_category) pour garantir la distribution.
    """
    cfg = settings_s2()
    
    # Définir toutes les catégories possibles
    all_categories = ["80-100%", "60-80%", "40-60%", "20-40%", "1-20%", "0%"]
    patches_per_category = max_patches // len(all_categories)
    
    print(f"[INFO] Distribution exacte: {patches_per_category} patches par catégorie ({len(all_categories)} catégories)")
    print(f"[INFO] Total: {patches_per_category * len(all_categories)} patches")
    
    bbox_with_categories = []
    
    # Import pour le calcul de couverture
    from utils.optimized_coverage import calculate_mangrove_coverage_batch, create_spatial_index_if_not_exists
    
    # Créer l'index spatial si nécessaire
    create_spatial_index_if_not_exists()
    
    with get_engine().connect() as conn:
        full_table = f'"{cfg.PG_SCHEMA}"."{cfg.PG_TABLE}"'
        
        # ÉTAPE 1: Générer tous les candidats d'un coup
        print(f"[INFO] Génération de candidats pour toutes les catégories...")
        
        # Générer plus de candidats de mangroves
        sql = text(
            f"""
            SELECT ST_AsBinary(geom), ST_Area(ST_Transform(geom, 3857)) as area_m2
            FROM {full_table}
            WHERE geom IS NOT NULL AND NOT ST_IsEmpty(geom)
            ORDER BY area_m2 DESC, random()
            LIMIT :lim
            """
        )
        
        candidates_bboxes = []
        print(f"[INFO] Récupération de {max_patches * 3} candidats depuis la base...")
        
        for (gbytes, area_m2) in conn.execute(sql, {"lim": max_patches * 3}):
            if len(candidates_bboxes) >= max_patches * 2:  # Limiter pour éviter l'explosion mémoire
                break
                
            if not gbytes:
                continue
                
            try:
                geom = wkb.loads(bytes(gbytes))
                if geom.is_empty:
                    continue
                
                bbox = create_valid_bbox(geom, patch_size_m)
                if bbox:
                    candidates_bboxes.append(bbox)
                    
            except Exception:
                continue
        
        print(f"[INFO] {len(candidates_bboxes)} candidats de mangroves générés")
        
        # ÉTAPE 2: Calculer toutes les couvertures en une fois
        if candidates_bboxes:
            print(f"[INFO] Calcul des couvertures pour tous les candidats...")
            try:
                coverages = calculate_mangrove_coverage_batch(candidates_bboxes)
                print(f"[INFO] Couvertures calculées avec succès")
                
                # ÉTAPE 3: Distribuer les candidats par catégorie
                candidates_by_category = {cat: [] for cat in all_categories}
                
                for bbox, coverage in zip(candidates_bboxes, coverages):
                    category = get_coverage_category_from_coverage(coverage)
                    candidates_by_category[category].append((bbox, category, coverage))
                
                # Afficher la répartition trouvée
                print(f"[INFO] Répartition des candidats trouvés:")
                for cat in all_categories:
                    count = len(candidates_by_category[cat])
                    print(f"  {cat}: {count} candidats")
                
                # ÉTAPE 4: Sélectionner le nombre voulu par catégorie
                for category in all_categories:
                    available = candidates_by_category[category]
                    needed = patches_per_category
                    
                    if len(available) >= needed:
                        # Prendre les premiers (déjà mélangés par random())
                        selected = available[:needed]
                        bbox_with_categories.extend([(bbox, cat) for bbox, cat, cov in selected])
                        print(f"[SUCCESS] {needed} patches sélectionnés pour {category}")
                    else:
                        # Prendre tous ceux disponibles
                        bbox_with_categories.extend([(bbox, cat) for bbox, cat, cov in available])
                        print(f"[WARNING] Seulement {len(available)} patches trouvés pour {category} (voulu: {needed})")
                
            except Exception as e:
                print(f"[ERROR] Erreur lors du calcul des couvertures: {e}")
                print(f"[INFO] Fallback: génération sans vérification de couverture...")
                
                # Fallback: prendre les candidats sans vérification stricte
                for i, bbox in enumerate(candidates_bboxes[:max_patches]):
                    category = all_categories[i % len(all_categories)]  # Distribution round-robin
                    bbox_with_categories.append((bbox, category))
        
        # ÉTAPE 5: Compléter avec des zones "0%" si nécessaire
        current_total = len(bbox_with_categories)
        target_total = patches_per_category * len(all_categories)
        
        if current_total < target_total:
            missing = target_total - current_total
            print(f"[INFO] Génération de {missing} patches supplémentaires (zones 0%)...")
            
            # Générer des points aléatoirement pour compléter
            for i in range(missing):
                lat = random.uniform(-60, 60)
                lon = random.uniform(-180, 180)
                
                bbox = create_valid_bbox_from_point(lat, lon, patch_size_m)
                if bbox:
                    bbox_with_categories.append((bbox, "0%"))
    
    print(f"[INFO] Total de {len(bbox_with_categories)} bboxes avec catégories générées")
    
    # Mélanger pour éviter les groupes
    random.shuffle(bbox_with_categories)
    
    # Vérification finale de la distribution
    from collections import Counter
    final_counts = Counter([cat for bbox, cat in bbox_with_categories])
    print(f"[INFO] Distribution finale:")
    for cat in all_categories:
        count = final_counts.get(cat, 0)
        print(f"  {cat}: {count} patches")
    
    return bbox_with_categories


def get_coverage_category_from_coverage(coverage: float) -> str:
    """Retourne la catégorie basée sur le pourcentage exact avec tolérance."""
    # Ajouter une petite tolérance pour éviter les problèmes de précision
    if coverage >= 79.9:  # Au lieu de 80 exact
        return "80-100%"
    elif coverage >= 59.9:  # Au lieu de 60 exact
        return "60-80%"
    elif coverage >= 39.9:  # Au lieu de 40 exact
        return "40-60%"
    elif coverage >= 19.9:  # Au lieu de 20 exact
        return "20-40%"
    elif coverage >= 0.1:   # Au lieu de > 0 exact
        return "1-20%"
    else:
        return "0%"


def create_valid_bbox_from_point(lat: float, lon: float, patch_size_m: int) -> BBox:
    """Crée une bbox de taille fixe autour d'un point."""
    from processing.bbox import create_valid_bbox
    from shapely.geometry import Point
    
    point = Point(lon, lat)
    return create_valid_bbox(point, patch_size_m)


def geom_to_bbox(geom) -> BBox:
    """Ancienne méthode : renvoie la bbox brute (peut être dégénérée si le geom est un point).

    Conservée pour référence mais on privilégie désormais create_valid_bbox qui garantit une
    surface minimale en mètres, évitant des images 1×1 toute noires/blanches.
    """
    minx, miny, maxx, maxy = geom.bounds
    return BBox([minx, miny, maxx, maxy], crs=CRS.WGS84)


def clear_outputs():
    """Nettoie automatiquement les dossiers de sortie avant chaque exécution."""
    try:
        base_dir = Path(__file__).resolve().parents[2]
        script_path = base_dir / 'pipeline' / 'scripts' / 'clearoutputs.ps1'

        print("[INFO] Nettoyage automatique des outputs...")

        # Exécute le script PowerShell
        result = subprocess.run(
            ['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', str(script_path)],
            capture_output=True,
            text=True,
            cwd=script_path.parent
        )

        if result.returncode == 0:
            print("[SUCCESS] Nettoyage terminé.")
            if result.stdout:
                print(f"[SCRIPT OUTPUT] {result.stdout.strip()}")
        else:
            print(f"[WARNING] Problème lors du nettoyage: {result.stderr.strip()}")

    except Exception as e:
        print(f"[ERROR] Impossible d'exécuter le script de nettoyage: {e}")


def main():
    """Main entry point."""
    # Timing détaillé
    start_total = time.time()
    
    cfg = settings_s2()
    
    print(f"[INFO] Démarrage du téléchargement de {cfg.MAX_PATCHES} images...")
    print(f"[INFO] Heure de début: {time.strftime('%H:%M:%S')}")
    
    cleanup_start = time.time()
    clear_outputs()
    cleanup_time = time.time() - cleanup_start
    print(f"[TIMING] Nettoyage terminé en {cleanup_time:.2f}s")

    print(f"[INFO] DB: postgresql://{cfg.PG_USER}:{cfg.PG_PASSWORD}@{cfg.PG_HOST}:{cfg.PG_PORT}/{cfg.PG_DB}")
    print(f"[INFO] OUTPUT_DIR: {cfg.OUTPUT_DIR}")
    
    bbox_start = time.time()
    print("[INFO] Mode gmw_v3 activé: génération de tuiles pré-filtrées par catégorie")
    bbox_with_categories = generate_bboxes_from_gmw_v3_by_category_fixed(
        max_patches=cfg.MAX_PATCHES, 
        patch_size_m=cfg.PATCH_SIZE_M
    )
    
    bbox_time = time.time() - bbox_start
    print(f"[TIMING] Génération des BBox terminée en {bbox_time:.2f}s")

    if not bbox_with_categories:
        print("[ERREUR] Aucune bbox valide générée → arrêt.")
        return
    
    # Séparer les bboxes et les catégories pour l'ancien système
    bboxes = [item[0] for item in bbox_with_categories]
    target_categories = [item[1] for item in bbox_with_categories]
    
    print(f"[INFO] Téléchargement {len(bboxes)} tuiles pré-classées...")
    
    # Compter par catégorie pour vérification
    from collections import Counter
    category_counts = Counter(target_categories)
    print("[INFO] Répartition finale:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} images")
    
    # Utiliser la version optimisée avec les catégories pré-définies
    download_start = time.time()
    try:
        paths = download_with_predefined_categories(bbox_with_categories, prefix="gmw", enhanced=True, workers=4)
    except ImportError:
        print("[WARNING] Module optimisé non disponible, utilisation de la méthode standard...")
        paths = run_download(bboxes, prefix="gmw", enhanced=True, workers=1)
    
    download_time = time.time() - download_start
    total_time = time.time() - start_total
    
    # Statistiques détaillées
    print("\n=== STATISTIQUES DE PERFORMANCE ===")
    print(f"[TIMING] Nettoyage: {cleanup_time:.2f}s")
    print(f"[TIMING] Génération BBox: {bbox_time:.2f}s") 
    print(f"[TIMING] Téléchargement: {download_time:.2f}s")
    print(f"[TIMING] Temps total: {total_time:.2f}s")
    if paths:
        print(f"[TIMING] Temps par image: {download_time/len(paths):.2f}s")
        print(f"[TIMING] Images par minute: {len(paths)/(download_time/60):.1f}")
        print(f"[TIMING] Temps pour 100000 images: {len(paths)/(download_time*100000/3600):.1f}")


if __name__ == "__main__":
    main()