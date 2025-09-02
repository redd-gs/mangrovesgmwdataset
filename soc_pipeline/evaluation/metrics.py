from __future__ import annotations
import numpy as np
from typing import Dict
try:
    from properscoring import crps_gaussian
except ImportError:  # fallback
    crps_gaussian = None


def negative_log_likelihood(ngb_model, X, y) -> float:
    dists = ngb_model.pred_dist(X)
    return -dists.logpdf(y).mean()


def crps(ngb_model, X, y) -> float:
    d = ngb_model.pred_dist(X)
    # If Normal, use closed form CRPS
    if hasattr(d, "scale") and crps_gaussian is not None:
        mu = d.loc
        sigma = d.scale
        return float(np.mean(crps_gaussian(y, mu, sigma)))
    # else Monte Carlo approximation
    draws = np.asarray(d.sample(400))
    y2 = y.reshape(1, -1)
    crps_vals = np.mean(np.abs(draws - y2), axis=0) - 0.5 * np.mean(
        np.abs(draws[:, :, None] - draws[:, None, :]), axis=(0, 1)
    )
    return float(crps_vals.mean())


def mae_rmse(ngb_model, X, y) -> Dict[str, float]:
    d = ngb_model.pred_dist(X)
    try:
        mean_pred = d.mean()
    except Exception:
        mean_pred = d.loc
    resid = y - mean_pred
    return {"mae": float(np.mean(np.abs(resid))), "rmse": float(np.sqrt(np.mean(resid**2)))}


def prediction_interval_coverage(ngb_model, X, y, alpha: float = 0.9) -> Dict[str, float]:
    dists = ngb_model.pred_dist(X)
    lower = dists.ppf((1 - alpha) / 2)
    upper = dists.ppf(1 - (1 - alpha) / 2)
    inside = (y >= lower) & (y <= upper)
    return {"alpha": alpha, "empirical_coverage": inside.mean()}
