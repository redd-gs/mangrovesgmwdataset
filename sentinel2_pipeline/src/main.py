from typing import List
from shapely import wkb
from sqlalchemy import text
from sentinelhub import BBox, CRS
import random

from config.settings import settings
from core.context import get_engine
from sentinel.download import run_download
from processing.bbox import create_valid_bbox  # Utiliser une bbox de taille fixe autour du centroïde


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

    geoms = fetch_geometries(limit=cfg.MAX_PATCHES)
    if not geoms:
        print("[AVERTISSEMENT] Aucune géométrie trouvée.")
        return

    # Ancien comportement (souvent bbox nulle si géométrie = point) :
    # bboxes = [geom_to_bbox(g) for g in geoms]
    # Nouveau : créer une bbox carrée autour du centroïde avec la taille PATCH_SIZE_M
    bboxes: List[BBox] = []
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