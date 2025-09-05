from pathlib import Path
from typing import List
import os
import subprocess
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
    # Nettoie automatiquement les outputs avant chaque exécution
    clear_outputs()

    cfg = settings_s2()
    print(f"[INFO] DB: {cfg.pg_dsn}")
    print(f"[INFO] OUTPUT_DIR: {cfg.OUTPUT_DIR}")
    print(f"[DEBUG] PATCH_SIZE_M from config: {cfg.PATCH_SIZE_M}")
    print(f"[DEBUG] PATCH_SIZE_M from env: {os.getenv('PATCH_SIZE_M', 'NOT_SET')}")

    # Nouvelle option: si variable d'env USE_GMW_V3=1 on lit directement gmw_v3 et on tuiles les polygones
    use_gmw_v3 = os.getenv("USE_GMW_V3", "0") in ("1","true","True")
    generate_cls_dataset = os.getenv("GENERATE_DATASET", "0") in ("1","true","True")

    if generate_cls_dataset:
        print("[INFO] Lancement de la génération du jeu de données de classification...")
        
        # Paramètres pour le dataset, modifiables via variables d'environnement si besoin
        images_per_category = int(os.getenv("IMAGES_PER_CATEGORY", "50")) # 50 images par catégorie par défaut
        patch_size_px = 256 # Taille d'image fixe
        patch_size_m = int(os.getenv("PATCH_SIZE_M", "2560")) # 256px * 10m/px = 2560m
        
        gmw_table = f"{cfg.PG_SCHEMA}.{cfg.PG_TABLE}"

        generate_dataset(
            output_dir=cfg.OUTPUT_DIR,
            images_per_category=images_per_category,
            patch_size_px=patch_size_px,
            patch_size_m=patch_size_m,
            gmw_table=gmw_table
        )
        
        print("[SUCCESS] Génération du jeu de données terminée.")
        return # On arrête l'exécution ici

    if use_gmw_v3:
        print("[INFO] Mode gmw_v3 activé: génération de tuiles à partir des polygones")
        bboxes = generate_bboxes_from_gmw_v3(limit_polygons=cfg.MAX_PATCHES*2, max_patches=cfg.MAX_PATCHES, patch_size_m=cfg.PATCH_SIZE_M)
    else:
        geoms = fetch_geometries(limit=cfg.MAX_PATCHES)
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

    if not bboxes:
        print("[ERREUR] Aucune bbox valide générée → arrêt.")
        return
    print(f"[INFO] Téléchargement {len(bboxes)} tuiles (taille cible {cfg.PATCH_SIZE_M} m)...")
    
    # Utiliser la version optimisée avec calcul de couverture en batch
    try:
        from optimized_download import download_with_batch_coverage
        paths = download_with_batch_coverage(bboxes, prefix="gmw", enhanced=True, workers=1)
    except ImportError:
        print("[WARNING] Module optimisé non disponible, utilisation de la méthode standard...")
        paths = run_download(bboxes, prefix="gmw", enhanced=True, workers=1)
    
    print(f"[INFO] Succès: {len(paths)} fichiers.")
    for p in paths:
        print(f" - {p}")


if __name__ == "__main__":
    main()