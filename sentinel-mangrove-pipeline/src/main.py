from typing import List
from shapely import wkb
from sqlalchemy import text
from sentinelhub import BBox, CRS
from sentinelhub.time_utils import parse_time_interval

from config.settings import settings
from core.context import get_engine
from sentinel.download import run_download


def fetch_geometries(limit: int) -> List:
    """
    Récupère des géométries WGS84.
    Si ton SRID n'est pas 4326, adapte en ajoutant ST_Transform.
    """
    cfg = settings()
    full_table = f'"{cfg.PG_SCHEMA}"."{cfg.PG_TABLE}"'
    sql = text(f"SELECT ST_AsBinary(geom) FROM {full_table} LIMIT :lim")
    rows = []
    with get_engine().connect() as conn:
        for (gbytes,) in conn.execute(sql, {"lim": limit}):
            if gbytes:
                geom = wkb.loads(bytes(gbytes))
                if not geom.is_empty:
                    rows.append(geom)
    return rows


def geom_to_bbox(geom) -> BBox:
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

    bboxes = [geom_to_bbox(g) for g in geoms]
    start_dt, end_dt = parse_time_interval(cfg.TIME_INTERVAL)
    time_interval = (start_dt.isoformat(), end_dt.isoformat())

    print(f"[INFO] Téléchargement {len(bboxes)} tuiles...")
    paths = run_download(bboxes, time_interval, prefix="gmw", enhanced=True, workers=1)
    print(f"[INFO] Succès: {len(paths)} fichiers.")
    for p in paths:
        print(f" - {p}")


if __name__ == "__main__":
    main()