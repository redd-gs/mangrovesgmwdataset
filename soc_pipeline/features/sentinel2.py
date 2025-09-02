from __future__ import annotations
import rasterio
import numpy as np
from pathlib import Path
from typing import Dict, List

S2_BANDS = ["B02","B03","B04","B08","B11","B12"]


def compute_indices(band_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    b = band_data
    def safe_ratio(a,b):
        return np.where((a+b)==0, 0, (a-b)/(a+b))
    ndvi = safe_ratio(b["B08"], b["B04"])
    ndmi = safe_ratio(b["B08"], b["B11"])
    mndwi = safe_ratio(b["B03"], b["B11"])
    nirv = b["B08"] * ndvi
    return {"NDVI": ndvi, "NDMI": ndmi, "MNDWI": mndwi, "NIRv": nirv}
