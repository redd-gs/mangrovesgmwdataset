from typing import List
import os
from shapely import wkb
from sqlalchemy import text
from sentinelhub import BBox, CRS
import random

from config.settings import settings
from core.context import get_engine
from sentinel.download import run_download
from processing.bbox import create_valid_bbox  # Utiliser une bbox de taille fixe autour du centroïde
from db.gmw_v3 import generate_bboxes_from_gmw_v3


def fetch_geometries(limit: int) -> List:
    """Retourne 'limit' géométries aléatoires (centroïdes) de zones de mangrove.

    On utilise ORDER BY random() pour varier les emprises à chaque exécution.
    Si la table est très grande, on pourrait optimiser (TABLESAMPLE) mais
    pour 10 items cela reste acceptable.
    """
    cfg = settings()
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


def main():
    cfg = settings()
    print(f"[INFO] DB: {cfg.pg_dsn}")
    print(f"[INFO] OUTPUT_DIR: {cfg.OUTPUT_DIR}")

    # Nouvelle option: si variable d'env USE_GMW_V3=1 on lit directement gmw_v3 et on tuiles les polygones
    use_gmw_v3 = os.getenv("USE_GMW_V3", "0") in ("1","true","True")
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
    paths = run_download(bboxes, prefix="gmw", enhanced=True, workers=1)
    print(f"[INFO] Succès: {len(paths)} fichiers.")
    for p in paths:
        print(f" - {p}")


if __name__ == "__main__":
    main()