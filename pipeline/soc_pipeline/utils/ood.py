from __future__ import annotations
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from typing import Dict, List


def fit_ood_models(X: np.ndarray, methods: List[str]):
    out = {}
    if "mahalanobis" in methods:
        cov = EmpiricalCovariance().fit(X)
        out["mahalanobis"] = cov
    if "isolation_forest" in methods:
        iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        iso.fit(X)
        out["isolation_forest"] = iso
    return out


def ood_scores(models: Dict[str, object], X: np.ndarray) -> Dict[str, np.ndarray]:
    scores = {}
    if "mahalanobis" in models:
        cov = models["mahalanobis"]
        scores["mahalanobis"] = cov.mahalanobis(X)
    if "isolation_forest" in models:
        iso = models["isolation_forest"]
        # IsolationForest: lower score more abnormal -> convert to positive anomaly dist
        s = -iso.score_samples(X)
        scores["isolation_forest"] = s
    return scores
