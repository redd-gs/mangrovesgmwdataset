from __future__ import annotations
"""Utilities to extract mangrove geometries/tiles from the gmw_v3 PostgreSQL database."""
from typing import List, Iterable
from shapely import wkb
from shapely.geometry import Polygon, box
import geopandas as gpd
from sqlalchemy import text
from sentinelhub import BBox, CRS
from config.settings import settings
from pipeline.src.config.context import get_engine


def get_estuarine_settings():
    return settings("estuarine")


def fetch_random_polygons(limit: int, min_area_m2: float = 0.0) -> List[Polygon]:
    """Fetch random mangrove polygons (or their representative surface points) from gmw_v3.

    Uses ST_PointOnSurface to guarantee representative interior points when later tiling.
    """
    cfg = get_estuarine_settings()
    full_table = f'"{cfg.PG_SCHEMA}"."{cfg.PG_TABLE}"'
    sql = text(
        f"""
        SELECT ST_AsBinary(geom)
        FROM {full_table}
        WHERE geom IS NOT NULL
          AND NOT ST_IsEmpty(geom)
          { 'AND ST_Area(ST_Transform(geom,3857)) >= :min_area' if min_area_m2>0 else ''}
        ORDER BY random()
        LIMIT :lim
        """
    )
    geoms: List[Polygon] = []
    params = {"lim": limit}
    if min_area_m2>0:
        params["min_area"] = min_area_m2
    with get_engine().connect() as conn:
        for (gbytes,) in conn.execute(sql, params):
            try:
                geom = wkb.loads(bytes(gbytes))
                if geom.is_empty:
                    continue
                geoms.append(geom)
            except Exception:
                continue
    return geoms


def polygon_to_square_tiles(geom, patch_size_m: int) -> List[BBox]:
    """Generate square tiles (BBox) covering the polygon extent at approximate patch_size_m (in 3857)."""
    if geom.is_empty:
        return []
    g3857 = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(3857)
    minx, miny, maxx, maxy = g3857.total_bounds
    size = patch_size_m
    tiles: List[BBox] = []
    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            tile_poly = box(x, y, x+size, y+size)
            # keep tile if intersects polygon
            if tile_poly.intersects(g3857.geometry.iloc[0]):
                tile_wgs84 = gpd.GeoSeries([tile_poly], crs=3857).to_crs(4326).geometry.iloc[0]
                minx2, miny2, maxx2, maxy2 = tile_wgs84.bounds
                tiles.append(BBox([minx2, miny2, maxx2, maxy2], crs=CRS.WGS84))
            x += size
        y += size
    return tiles


def generate_bboxes_from_gmw_v3(limit_polygons: int, max_patches: int, patch_size_m: int) -> List[BBox]:
    polys = fetch_random_polygons(limit_polygons)
    bboxes: List[BBox] = []
    for p in polys:
        tiles = polygon_to_square_tiles(p, patch_size_m)
        bboxes.extend(tiles)
        if len(bboxes) >= max_patches:
            break
    return bboxes[:max_patches]
