import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from sentinelhub import (
    SHConfig, BBox, CRS, MimeType, SentinelHubRequest, SentinelHubSession,
    DataCollection, bbox_to_dimensions, SentinelHubCatalog
)
from PIL import Image
import numpy as np
from sqlalchemy import create_engine, text
from src.config.settings import Config

TRUE_COLOR_EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{bands: ["B02","B03","B04"], units: "REFLECTANCE"}],
        output: { bands: 3 }
    };
}
function evaluatePixel(s) {
  return [s.B04, s.B03, s.B02];
}
"""

ENHANCED_COLOR_EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{bands: ["B02","B03","B04"], units: "REFLECTANCE"}],
        output: { bands: 3 }
    };
}
function evaluatePixel(s) {
  return [s.B04 * 1.2, s.B03 * 1.1, s.B02]; // Simple enhancement
}
"""

def sh_config() -> SHConfig:
    cfg = SHConfig()
    cfg.sh_client_id = Config.SH_CLIENT_ID
    cfg.sh_client_secret = Config.SH_CLIENT_SECRET
    cfg.sh_base_url = "https://services.sentinel-hub.com"
    cfg.sh_token_url = "https://services.sentinel-hub.com/oauth/token"
    cfg.instance_id = "516edccb-ad8c-40e2-b406-82bd4405c1da"
    return cfg

def download_image(cfg, bbox, time_interval, output_path, enhanced=False):
    try:
        evalscript = ENHANCED_COLOR_EVALSCRIPT if enhanced else TRUE_COLOR_EVALSCRIPT
        request = SentinelHubRequest(
            evalscript=evalscript,
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
        image = np.clip(image, 0, 1)

        # Apply gamma correction for better visual quality
        gamma = 0.9
        image = np.power(image, gamma)

        image8 = (image * 255 + 0.5).astype(np.uint8)
        Image.fromarray(image8).save(output_path)
        return True

    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        return False

def main():
    cfg = sh_config()
    bbox = BBox([minx, miny, maxx, maxy], crs=CRS.WGS84)  # Define your bbox here
    time_interval = ("2024-06-01T00:00:00Z", "2024-06-10T23:59:59Z")  # Define your time interval here
    output_file = Path("output_image.png")  # Define your output path here

    success = download_image(cfg, bbox, time_interval, output_file, enhanced=True)
    if success:
        print("Image downloaded successfully.")
    else:
        print("Failed to download image.")

if __name__ == "__main__":
    main()