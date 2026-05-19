from __future__ import annotations

import random

import numpy as np

try:
    import torch
except Exception:
    torch = None


def torch_available() -> bool:
    return torch is not None


def get_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
