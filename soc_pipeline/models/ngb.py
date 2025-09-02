from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
from ngboost import NGBRegressor
from ngboost.distns import LogNormal, Gamma, Normal
from sklearn.tree import DecisionTreeRegressor
from ngboost.scores import CRPScore

SUPPORTED = {"lognormal": LogNormal, "gamma": Gamma, "normal": Normal}

@dataclass
class NGBoostConfig:
    distribution: Literal["lognormal", "gamma", "normal"] = "lognormal"
    n_estimators: int = 800
    learning_rate: float = 0.01
    col_sample: float = 0.8
    subsample: float = 1.0  # row subsample per iteration (minibatch_frac)
    max_depth: int = 4
    min_samples_leaf: int = 50
    random_state: int = 42


def make_ngb(cfg: NGBoostConfig) -> NGBRegressor:
    Dist = SUPPORTED[cfg.distribution]
    base = DecisionTreeRegressor(max_depth=cfg.max_depth, min_samples_leaf=cfg.min_samples_leaf, random_state=cfg.random_state)
    return NGBRegressor(
        Dist=Dist,
        Score=CRPScore,
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        natural_gradient=True,
        col_sample=cfg.col_sample,
        minibatch_frac=cfg.subsample,
        Base=base,
        verbose=False,
        random_state=cfg.random_state,
    )
