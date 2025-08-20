import os
from datetime import datetime, timedelta
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, BBox, CRS
from pathlib import Path
import geopandas as gpd
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../sentinel2-mangrove-pipeline/src/config'))
from config.settings import settings

# --- CONFIGURATION ---
cfg = settings()


# BBox de Pulau Ubin (Singapour) en WGS84
# minx, miny, maxx, maxy
bbox = BBox([103.958, 1.396, 104.03, 1.45], crs=CRS.WGS84)

# Paramètres temporels
start_date = datetime.strptime("2024-12-01", "%Y-%m-%d")
period_days = 14
n_images = 10
dates = [(start_date + timedelta(days=i*period_days)).strftime("%Y-%m-%d") for i in range(n_images)]

# Authentification Sentinel Hub
sh_config = SHConfig()
sh_config.sh_client_id = cfg.SH_CLIENT_ID
sh_config.sh_client_secret = cfg.SH_CLIENT_SECRET
if hasattr(cfg, "SH_INSTANCE_ID"):
    sh_config.instance_id = cfg.SH_INSTANCE_ID

# Créer le dossier de sortie
output_dir = Path(cfg.OUTPUT_DIR) / "sentinel1_time_series"
output_dir.mkdir(parents=True, exist_ok=True)

# Générer les requêtes et télécharger les images
for i, date in enumerate(dates):
    time_interval = (date, (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"))
    request = SentinelHubRequest(
        data_folder=str(output_dir),
        evalscript="""
        //VERSION=3
        function setup() {
            return {
                input: ["VV", "VH"],
                output: { bands: 2, sampleType: "FLOAT32" }
            };
        }
        function evaluatePixel(sample) {
            return [sample.VV, sample.VH];
        }
        """,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL1_IW,
                time_interval=time_interval,
                mosaicking_order="mostRecent"
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", MimeType.TIFF)
        ],
        bbox=bbox,
        size=(cfg.PATCH_SIZE_M, cfg.PATCH_SIZE_M),
        config=sh_config
    )
    img = request.get_data(save_data=True)
    print(f"Image {i+1}/10 téléchargée pour la date {date}.")

print("Série temporelle Sentinel-1 générée dans:", output_dir)
