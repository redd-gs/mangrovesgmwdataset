# -*- coding: utf-8 -*-
import geopandas as gpd
from shapely.geometry import box
from sentinelhub import BBox, CRS

def create_valid_bbox(geometry, size_m):
    try:
        if geometry.is_empty:
            print("Geometrie vide")
            return None

        centroid = geometry.centroid
        gdf = gpd.GeoDataFrame(geometry=[centroid], crs="EPSG:4326").to_crs(epsg=3857)
        x, y = gdf.geometry.iloc[0].x, gdf.geometry.iloc[0].y

        half_size = max(size_m / 2, 10)
        bbox_3857 = box(x - half_size, y - half_size, x + half_size, y + half_size)
        bbox_wgs84 = gpd.GeoDataFrame(geometry=[bbox_3857], crs="EPSG:3857").to_crs(epsg=4326)
        minx, miny, maxx, maxy = bbox_wgs84.total_bounds

        # Removed emoji to avoid Windows console / source encoding issues.
        print(f"BBox generee: [{minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f}]")
        return BBox([minx, miny, maxx, maxy], crs=CRS.WGS84)

    except Exception as e:
        print(f"Erreur creation BBox: {str(e)}")
        return None

def optimize_bbox_creation(geometries, size_m):
    bboxes = []
    for geometry in geometries:
        bbox = create_valid_bbox(geometry, size_m)
        if bbox:
            bboxes.append(bbox)
    return bboxes

def batch_process_bboxes(geometries, size_m):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as executor:
        bboxes = list(executor.map(lambda geom: create_valid_bbox(geom, size_m), geometries))
    return [bbox for bbox in bboxes if bbox is not None]