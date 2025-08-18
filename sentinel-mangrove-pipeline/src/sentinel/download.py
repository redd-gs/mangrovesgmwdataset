from pathlib import Path
from typing import Iterable, List, Tuple
import concurrent.futures as cf
import numpy as np
from PIL import Image
from pyproj import CRS
from sentinelhub import (
    BBox, MimeType, SentinelHubRequest,
    DataCollection, bbox_to_dimensions
)
from config.settings import settings
from core.context import get_sh_config

TRUE_COLOR_EVALSCRIPT = """//VERSION=3
function setup(){return{input:[{bands:["B02","B03","B04"],units:"REFLECTANCE"}],output:{bands:3}};}
function evaluatePixel(s){return [s.B04,s.B03,s.B02];}
"""

ENHANCED_COLOR_EVALSCRIPT = """//VERSION=3
function setup(){return{input:[{bands:["B02","B03","B04"],units:"REFLECTANCE"}],output:{bands:3}};}
function evaluatePixel(s){return [s.B04*1.15,s.B03*1.08,s.B02];}
"""

DEBUG = True  # passe à False quand OK

DEBUG_EVALSCRIPT = """//VERSION=3
function setup(){
  return {
    input: [{bands:["B02","B03","B04","dataMask"], units:"REFLECTANCE"}],
    output: [
      {id:"rgb", bands:3},
      {id:"mask", bands:1}
    ]
  };
}
function evaluatePixel(s){
  return {
    rgb: [s.B04, s.B03, s.B02],
    mask: [s.dataMask]
  };
}
"""

def download_single(bbox, time_interval, output_path, enhanced=False):
    cfg = settings()
    sh_cfg = get_sh_config()
    evalscript = DEBUG_EVALSCRIPT if DEBUG else (ENHANCED_COLOR_EVALSCRIPT if enhanced else TRUE_COLOR_EVALSCRIPT)

    req = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=time_interval,
            mosaicking_order="leastCC"
        )],
        responses=[
            SentinelHubRequest.output_response("rgb", MimeType.TIFF),
            SentinelHubRequest.output_response("mask", MimeType.TIFF)
        ] if DEBUG else [SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=bbox_to_dimensions(bbox, resolution=cfg.IMAGE_RESOLUTION),
        config=sh_cfg
    )
    try:
        data = req.get_data()
        if DEBUG:
            rgb, mask = data
            import numpy as np
            mratio = mask.mean()
            print(f"[DEBUG] bbox={bbox} mask_mean={mratio:.3f} rgb_minmax=({rgb.min():.3f},{rgb.max():.3f})")
            if mratio < 0.05:
                print("[DEBUG] Presque aucune donnée valide (dataMask≈0) → vérifier bbox ou date.")
            rgb = np.clip(rgb,0,1)
            if cfg.ENHANCEMENT_METHOD == "gamma":
                rgb = np.power(rgb, cfg.GAMMA_VALUE)
            out_png = output_path.with_suffix(".png")
            from PIL import Image
            Image.fromarray((rgb*255+0.5).astype("uint8")).save(out_png)
            Image.fromarray((mask*255).astype("uint8")).save(output_path.with_name(output_path.stem+"_mask.png"))
            return True
        else:
            img = data[0]
            # ... traitement existant ...
            return True
    except Exception as e:
        print(f"[ERREUR] {e}")
        return False

def run_download(bboxes: Iterable[BBox],
                 time_interval: Tuple[str, str],
                 prefix="patch",
                 enhanced=True,
                 workers: int = 1) -> List[Path]:
    cfg = settings()
    results: List[Path] = []

    def task(item):
        idx, bb = item
        out = cfg.OUTPUT_DIR / f"{prefix}_{idx}.png"
        ok = download_single(bb, time_interval, out, enhanced=enhanced)
        return out if ok else None

    items = list(enumerate(bboxes, start=1))
    if workers > 1:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for path in ex.map(task, items):
                if path:
                    results.append(path)
    else:
        for it in items:
            path = task(it)
            if path:
                results.append(path)
    return results

test_bbox = BBox([2.27,48.84,2.30,48.86], crs=CRS.WGS84)
time_interval = ("2023-01-01", "2023-01-31")  # Remplacez par les dates souhaitées
paths = run_download([test_bbox], time_interval, prefix="test", enhanced=True, workers=1)

cfg = settings()
print(f"[DEBUG] TIME_INTERVAL={cfg.TIME_INTERVAL}")
print(f"[DEBUG] SH_CLIENT_ID set? {'oui' if cfg.SH_CLIENT_ID else 'non'}")
# Example: define geoms as a list of bounding boxes or geometries
geoms = [test_bbox]  # Replace with your actual geometries if needed
for g in geoms:
        print("[DEBUG] geom bounds:", g.bounds)
                
