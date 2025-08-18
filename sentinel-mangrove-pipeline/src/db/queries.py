from datetime import datetime, timedelta, timezone
import numpy as np
from tkinter import Image
from shapely.geometry import box
from numpy import shape
from sqlalchemy import text
from config.settings import Config
import geopandas as gpd
from sentinelhub import BBox, CRS, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions
from sentinel.download import TRUE_COLOR_EVALSCRIPT
from config.settings import settings


def get_sample_geometries(engine, limit=1):
    with engine.begin() as conn:
        result = conn.execute(text(f"""
            SELECT gid, ST_AsGeoJSON(geom)::json AS geometry
            FROM {Config.PG_SCHEMA}.{Config.PG_TABLE}
            WHERE geom IS NOT NULL
            LIMIT {limit}
        """))
        return [(row[0], shape(row[1])) for row in result]

def create_valid_bbox(geometry, size_m):
    try:
        if geometry.is_empty:
            print("⚠️ Géométrie vide!")
            return None

        centroid = geometry.centroid
        gdf = gpd.GeoDataFrame(geometry=[centroid], crs="EPSG:4326").to_crs(epsg=3857)
        x, y = gdf.geometry.iloc[0].x, gdf.geometry.iloc[0].y

        half_size = max(size_m / 2, 10)
        bbox_3857 = box(x - half_size, y - half_size, x + half_size, y + half_size)
        bbox_wgs84 = gpd.GeoDataFrame(geometry=[bbox_3857], crs="EPSG:3857").to_crs(epsg=4326)
        minx, miny, maxx, maxy = bbox_wgs84.total_bounds

        print(f"🌍 BBox générée: [{minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f}]")
        return BBox([minx, miny, maxx, maxy], crs=CRS.WGS84)

    except Exception as e:
        print(f"❌ Erreur création BBox: {str(e)}")
        return None

def _interval_iso8601():
    """Construit un intervalle ISO 8601 (startZ, endZ) à partir du settings centralisé."""
    cfg = settings()
    start_str, end_str = cfg.time_interval_tuple
    # On inclut toute la journée de fin (23:59:59) pour les requêtes STAC si besoin
    start = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1) - timedelta(seconds=1)
    return (start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ"))

def search_images(catalog, bbox, time_interval, max_cc):
    try:
        print(f"🔍 Recherche d'images pour:")
        print(f"   - BBox: {bbox}")
        print(f"   - Période: {time_interval}")
        print(f"   - Couverture nuageuse max: {max_cc}%")

        # Filtre au format CQL2
        cloud_filter = f"eo:cloud_cover < {max_cc}"

        results = list(catalog.search(
            collection=DataCollection.SENTINEL2_L2A,
            bbox=bbox,
            time=time_interval,
            filter=cloud_filter,
            limit=5
        ))

        print(f"📊 Nombre d'images trouvées: {len(results)}")
        for i, res in enumerate(results, 1):
            print(f"   {i}. ID: {res['id']}")
            print(f"      Date: {res['properties']['datetime']}")
            print(f"      Couverture nuageuse: {res['properties']['eo:cloud_cover']}%")

        return results

    except Exception as e:
        print(f"❌ Erreur recherche: {str(e)}")
        return []

def download_image(cfg, bbox, time_interval, output_path):
    try:
        print(f"📥 Téléchargement de l'image vers {output_path}")

        request = SentinelHubRequest(
            evalscript=TRUE_COLOR_EVALSCRIPT,
            input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
            )],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=bbox,
            size=bbox_to_dimensions(bbox, resolution=10),
            config=cfg
        )

        image = request.get_data()[0]
        k = 2.2
        image = np.clip(image * k, 0, 1)

        gamma = 0.9
        image = np.power(image, gamma)

        image8 = (image * 255 + 0.5).astype(np.uint8)
        Image.fromarray(image8).save(output_path)
        print(f"✅ Image sauvegardée: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Erreur téléchargement: {str(e)}")
        return False