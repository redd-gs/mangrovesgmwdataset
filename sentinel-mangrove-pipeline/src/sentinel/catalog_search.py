import os
from sentinelhub import SentinelHubCatalog, BBox, DataCollection
from config import settings
from utils.time import format_time_interval

class CatalogSearch:
    def __init__(self, config):
        self.catalog = SentinelHubCatalog(config=config)

    def search_images(self, bbox: BBox, time_interval: str, max_cloud_cover: int = 20):
        try:
            print(f"🔍 Searching for images:")
            print(f"   - BBox: {bbox}")
            print(f"   - Time Interval: {time_interval}")
            print(f"   - Max Cloud Cover: {max_cloud_cover}%")

            cloud_filter = f"eo:cloud_cover < {max_cloud_cover}"

            results = list(self.catalog.search(
                collection=DataCollection.SENTINEL2_L2A,
                bbox=bbox,
                time=time_interval,
                filter=cloud_filter,
                limit=5
            ))

            print(f"📊 Number of images found: {len(results)}")
            for i, res in enumerate(results, 1):
                print(f"   {i}. ID: {res['id']}")
                print(f"      Date: {res['properties']['datetime']}")
                print(f"      Cloud Cover: {res['properties']['eo:cloud_cover']}%")

            return results

        except Exception as e:
            print(f"❌ Error during search: {str(e)}")
            return []

    def search_images_with_enhancements(self, bbox: BBox, time_interval: str, max_cloud_cover: int = 20):
        images = self.search_images(bbox, time_interval, max_cloud_cover)
        # Additional processing or enhancements can be added here
        return images

def build_search_time_interval():
    cfg = settings()
    start, end = format_time_interval(cfg.TIME_INTERVAL)
    return (start, end)