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
from database.gmw_v3 import generate_bboxes_from_gmw_v3
from dataset.mangrove_dataset import generate_dataset
from optimized_download import download_with_batch_coverage


def fetch_geometries(limit: int) -> List:
    """Retourne 'limit' géométries aléatoires (centroïdes) de zones de mangrove.

    On utilise ORDER BY random() pour varier les emprises à chaque exécution.
    Si la table est très grande, on pourrait optimiser (TABLESAMPLE) mais
    pour 10 items cela reste acceptable.
    """
    cfg = settings_s2()
    full_table = f'"{cfg.PG_SCHEMA}"."{cfg.PG_TABLE}"'
    # On extrait un point robuste intérieur (PointOnSurface) puis son centroid (équivalent ici) pour garantir qu'il tombe dans le polygone.
    sql = text(
        f"""
        SELECT ST_AsBinary(ST_PointOnSurface(geom))
        FROM {full_table}
        WHERE geom IS NOT NULL AND NOT ST_IsEmpty(geom)
        ORDER BY random()
        LIMIT :lim
        """
    )
    rows: List = []
    with get_engine().connect() as conn:
        for (gbytes,) in conn.execute(sql, {"lim": limit}):
            if not gbytes:
                continue
            try:
                geom = wkb.loads(bytes(gbytes))
                if geom.is_empty:
                    continue
                rows.append(geom)
            except Exception:
                continue
    # Shuffle supplémentaire (probablement superflu mais assure variété si cache / planification DB)
    random.shuffle(rows)
    return rows


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
        
        if not script_path.exists():
            print(f"[WARNING] Le script de nettoyage n'a pas été trouvé à l'emplacement: {script_path}")
            return

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
    # Configuration pour 40 images
    TARGET_IMAGES = 40
    
    # Timing détaillé
    start_total = time.time()
    
    print(f"[INFO] Démarrage du téléchargement de {TARGET_IMAGES} images...")
    print(f"[INFO] Heure de début: {time.strftime('%H:%M:%S')}")
    
    cleanup_start = time.time()
    clear_outputs()
    cleanup_time = time.time() - cleanup_start
    print(f"[TIMING] Nettoyage terminé en {cleanup_time:.2f}s")

    cfg = settings_s2()
    print(f"[INFO] DB: postgresql://{cfg.PG_USER}:{cfg.PG_PASSWORD}@{cfg.PG_HOST}:{cfg.PG_PORT}/{cfg.PG_DB}")
    print(f"[INFO] OUTPUT_DIR: {cfg.OUTPUT_DIR}")
    
    use_gmw_v3 = True  # Mode recommandé
    
    bbox_start = time.time()
    if use_gmw_v3:
        print("[INFO] Mode gmw_v3 activé: génération de tuiles à partir des polygones")
        bboxes = generate_bboxes_from_gmw_v3(limit_polygons=TARGET_IMAGES*2, max_patches=TARGET_IMAGES, patch_size_m=cfg.PATCH_SIZE_M)
    else:
        geoms = fetch_geometries(limit=TARGET_IMAGES)
        if not geoms:
            print("[AVERTISSEMENT] Aucune géométrie trouvée.")
            return
        bboxes = []
        for g in geoms:
            bb = create_valid_bbox(g, cfg.PATCH_SIZE_M)
            if bb:
                bboxes.append(bb)
            else:
                print("[AVERTISSEMENT] BBox ignorée (géométrie vide).")
    
    bbox_time = time.time() - bbox_start
    print(f"[TIMING] Génération des BBox terminée en {bbox_time:.2f}s")

    if not bboxes:
        print("[ERREUR] Aucune bbox valide générée → arrêt.")
        return
    print(f"[INFO] Téléchargement {len(bboxes)} tuiles (taille cible {cfg.PATCH_SIZE_M} m)...")
    
    # Utiliser la version optimisée avec calcul de couverture en batch
    download_start = time.time()
    try:
        paths = download_with_batch_coverage(bboxes, prefix="gmw", enhanced=True, workers=1)
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
    print(f"[TIMING] Temps par image: {download_time/len(paths):.2f}s")
    print(f"[TIMING] Images par minute: {len(paths)/(download_time/60):.1f}")
    print(f"[TIMING] Estimation pour 100,000 images: {(download_time/len(paths)*100000/3600):.1f}h")


if __name__ == "__main__":
    main()