import os

from db.queries import format_time_interval, get_sample_geometries
if "PYTHONPATH" not in os.environ:
    import sys
    from pathlib import Path
    sys.path.append(str(Path.cwd() / "src"))

from config import settings
cfg = settings()
cfg.OUTPUT_DIR

from pathlib import Path
from datetime import datetime, timedelta, timezone
from sentinelhub import SHConfig, BBox, CRS, MimeType, SentinelHubRequest, SentinelHubCatalog, DataCollection, bbox_to_dimensions
import geopandas as gpd
import numpy as np
from shapely.geometry import shape
from sqlalchemy import create_engine, text
from config.settings import Config
from db.connection import create_db_connection
from sentinel.auth import authenticate
from sentinel.catalog_search import search_images
from sentinel.download import download_image
from processing.bbox import create_valid_bbox
from processing.enhancements import enhance_image
from io.writer import save_image

def main():
    cfg = settings()
    print("DB:", cfg.pg_dsn)
    print("Sentinel client id present:", bool(cfg.SH_CLIENT_ID))
    print("🚀 Starting Sentinel-2 Mangrove Pipeline")
    
    # Setup database connection
    engine = create_db_connection()
    
    # Authenticate with Sentinel Hub
    cfg = authenticate()
    
    # Get sample geometries from the database
    samples = get_sample_geometries(engine, Config.MAX_PATCHES)
    
    if not samples:
        print("❌ No geometries found!")
        return
    
    for gid, geom in samples:
        print(f"\n🔍 Processing area {gid}")
        
        # Create bounding box
        bbox = create_valid_bbox(geom, Config.PATCH_SIZE_M)
        if not bbox:
            continue
        
        # Format time interval
        time_interval = format_time_interval(Config.TIME_INTERVAL)
        print(f"🕒 Formatted time interval: {time_interval}")
        
        # Search for images
        images = search_images(cfg, bbox, time_interval, Config.MAX_CLOUD_COVER)
        if not images:
            print("\n⚠️ No images available for this area/date.")
            continue
        
        # Download and enhance images
        for image in images:
            output_file = Config.OUTPUT_DIR / f"mangrove_{gid}_{image['properties']['datetime']}.png"
            success = download_image(cfg, bbox, time_interval, output_file)
            if success:
                enhanced_image = enhance_image(output_file)
                save_image(enhanced_image, output_file)
                print(f"✅ Image saved: {output_file}")
            else:
                print("❌ Download failed.")
    
if __name__ == "__main__":
    main()