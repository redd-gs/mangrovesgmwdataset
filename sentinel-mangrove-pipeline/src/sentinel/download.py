from pathlib import Path
from typing import Iterable, List, Tuple
import concurrent.futures as cf
import numpy as np
from PIL import Image
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

def download_single(bbox: BBox, time_interval: Tuple[str, str], output_path: Path, enhanced=False) -> bool:
    cfg = settings()
    sh_cfg = get_sh_config()
    evalscript = ENHANCED_COLOR_EVALSCRIPT if enhanced else TRUE_COLOR_EVALSCRIPT

    req = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
                mosaicking_order="mostRecent"
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=bbox_to_dimensions(bbox, resolution=cfg.IMAGE_RESOLUTION),
        config=sh_cfg
    )
    try:
        data = req.get_data()[0]  # ndarray float32 in [0,1]
        data = np.clip(data, 0, 1)
        if cfg.ENHANCEMENT_METHOD == "gamma":
            data = np.power(data, cfg.GAMMA_VALUE)
        img8 = (data * 255 + 0.5).astype(np.uint8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img8).save(output_path)
        return True
    except Exception as e:
        print(f"[ERREUR] BBox {bbox} : {e}")
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
