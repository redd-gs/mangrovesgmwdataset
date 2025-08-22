from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class LogTransform:
    eps: float = 1e-6

    def forward(self, y: np.ndarray) -> np.ndarray:
        return np.log(y + self.eps)

    def inverse(self, y_log: np.ndarray) -> np.ndarray:
        return np.exp(y_log) - self.eps

@dataclass
class IdentityTransform:
    def forward(self, y: np.ndarray) -> np.ndarray:
        return y
    def inverse(self, y: np.ndarray) -> np.ndarray:
        return y
