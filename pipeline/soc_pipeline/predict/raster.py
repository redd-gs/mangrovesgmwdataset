from __future__ import annotations
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
import pandas as pd
from typing import Dict, Any


def _read_rasters(raster_config: str):
    cfg = pd.read_csv(raster_config)
    rasters = {}
    ref_profile = None
    for _, row in cfg.iterrows():
        feat = row["feature"]
        path = row["path"]
        ds = rasterio.open(path)
        if ref_profile is None:
            ref_profile = ds.profile
        else:
            if ds.width != ref_profile["width"] or ds.height != ref_profile["height"]:
                raise ValueError("Raster dimension mismatch")
        rasters[feat] = ds
    return rasters, ref_profile


def _stack_window(rasters: Dict[str, rasterio.io.DatasetReader], w: Window):
    arrs = []
    for feat, ds in rasters.items():
        data = ds.read(1, window=w)
        arrs.append(data)
    stack = np.stack(arrs, axis=0)  # (F, h, w)
    return stack


def raster_predict(bundle: Dict[str, Any], raster_config: str, out_dir: str, transform, mask_path=None, tile_size=512, ood_action="none", ood_threshold=0.99):
    rasters, profile = _read_rasters(raster_config)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    feats = bundle["features"]
    # Ensure config covers needed features
    missing = [f for f in feats if f not in rasters]
    if missing:
        raise ValueError(f"Missing rasters for features: {missing}")

    mask_ds = rasterio.open(mask_path) if mask_path else None
    model = None
    ensemble = bundle.get("ensemble", False)

    width = profile["width"]; height = profile["height"]
    profile_out = profile.copy()
    profile_out.update(dtype="float32", count=7, compress="deflate")  # bands: median, mean, std, q05, q95, p90width, iqr
    out_main = rasterio.open(str(Path(out_dir)/"soc_distribution.tif"), "w", **profile_out)

    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            h = min(tile_size, height - y0)
            w = min(tile_size, width - x0)
            win = Window(x0, y0, w, h)
            stack = _stack_window({f: rasters[f] for f in feats}, win)  # (F,h,w)
            X = stack.reshape(len(feats), -1).T  # (N,F)
            # OOD detection
            if "ood" in bundle and ood_action != "none":
                from ..utils.ood import ood_scores
                scores = ood_scores(bundle["ood"], X)
                # Combine scores by rank-normalized max
                combined = np.zeros(X.shape[0])
                for s in scores.values():
                    r = np.argsort(np.argsort(s)) / (len(s)-1)
                    combined = np.maximum(combined, r)
                ood_mask = combined > ood_threshold
            else:
                ood_mask = np.zeros(X.shape[0], dtype=bool)

            if ensemble:
                preds = []
                for m in bundle["models"]:
                    d = m.pred_dist(X)
                    preds.append({"mean": d.mean(), "std": d.std(), "q05": d.ppf(0.05), "q50": d.ppf(0.5), "q95": d.ppf(0.95)})
                # Average moments/quantiles (approx)
                mean = np.mean([p["mean"] for p in preds], axis=0)
                std = np.mean([p["std"] for p in preds], axis=0)
                q05 = np.mean([p["q05"] for p in preds], axis=0)
                q50 = np.mean([p["q50"] for p in preds], axis=0)
                q95 = np.mean([p["q95"] for p in preds], axis=0)
            else:
                model = bundle["model"]
                d = model.pred_dist(X)
                mean = d.mean(); std = d.std(); q05 = d.ppf(0.05); q50 = d.ppf(0.5); q95 = d.ppf(0.95)

            # Inverse transform if needed
            if transform:
                # If model in log-space for SOC: distribution is on transformed; we approximated stats in transformed space.
                # For LogNormal, we should recompute from parameters but here we approximate via exponentiation.
                mean = transform.inverse(mean)
                q05 = transform.inverse(q05)
                q50 = transform.inverse(q50)
                q95 = transform.inverse(q95)
                # std not directly invertible; approximate via delta method using local derivative exp(mu)
                # Using q95 and q05 spread as proxy
                std = (q95 - q05) / 3.29  # approx for 90% interval of lognormal ~3.29*sd

            # Reshape back
            mean_r = mean.reshape(h, w).astype("float32")
            std_r = std.reshape(h, w).astype("float32")
            q05_r = q05.reshape(h, w).astype("float32")
            q50_r = q50.reshape(h, w).astype("float32")
            q95_r = q95.reshape(h, w).astype("float32")

            # Derived uncertainty layers
            p90width_r = (q95_r - q05_r).astype("float32")
            iqr_r = (q95_r - q05_r) * 0.588  # crude since we lack q25/q75; placeholder

            if mask_ds is not None:
                mask = mask_ds.read(1, window=win) == 0
                for arr in (mean_r, std_r, q05_r, q50_r, q95_r, p90width_r, iqr_r):
                    arr[mask] = np.nan
            if ood_action in ("mask","nan"):
                ood_mask_img = ood_mask.reshape(h, w)
                for arr in (mean_r, std_r, q05_r, q50_r, q95_r, p90width_r, iqr_r):
                    arr[ood_mask_img] = np.nan

            out_main.write(q50_r, 1)      # median
            out_main.write(mean_r, 2)     # mean
            out_main.write(std_r, 3)      # std approx
            out_main.write(q05_r, 4)      # q05
            out_main.write(q95_r, 5)      # q95
            out_main.write(p90width_r,6)  # width Q95-Q05
            out_main.write(iqr_r,7)       # approx IQR placeholder
    out_main.close()
    # Close rasters
    for ds in rasters.values():
        ds.close()
    if mask_ds: mask_ds.close()
