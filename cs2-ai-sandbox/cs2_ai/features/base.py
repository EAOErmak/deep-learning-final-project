from __future__ import annotations

from typing import Protocol

import numpy as np


class FeatureExtractor(Protocol):
    def extract(self, *args, **kwargs) -> np.ndarray:
        ...

    def feature_dim(self) -> int:
        ...
