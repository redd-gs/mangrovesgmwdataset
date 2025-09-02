from __future__ import annotations
import numpy as np
import pandas as pd


def pit_values(ngb_model, X, y) -> np.ndarray:
    d = ngb_model.pred_dist(X)
    return d.cdf(y)


def pit_histogram(pit: np.ndarray, bins: int = 20) -> dict:
    hist, edges = np.histogram(pit, bins=bins, range=(0,1), density=True)
    return {"hist": hist.tolist(), "edges": edges.tolist()}


def interval_coverages(ngb_model, X, y, alphas=(0.5,0.8,0.95)):
    d = ngb_model.pred_dist(X)
    results = []
    for a in alphas:
        lower = d.ppf((1-a)/2)
        upper = d.ppf(1-(1-a)/2)
        cov = ((y>=lower)&(y<=upper)).mean()
        results.append({"alpha": a, "coverage": float(cov)})
    return results


def residuals_summary(ngb_model, X, y, log_space=False):
    d = ngb_model.pred_dist(X)
    mean_pred = d.loc  # mean parameter for Normal; for others use expectation()
    try:
        mean_pred = d.mean()
    except Exception:
        pass
    resid = y - mean_pred
    return {
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "bias": float(np.mean(resid)),
    }
