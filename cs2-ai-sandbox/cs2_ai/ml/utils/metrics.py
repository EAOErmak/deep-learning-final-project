from __future__ import annotations

import numpy as np


def mean_absolute_error(pred: np.ndarray, target: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32)
    return float(np.mean(np.abs(pred_arr - target_arr)))


def binary_accuracy(logits: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    logits_arr = np.asarray(logits, dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32)
    predicted = (logits_arr >= threshold).astype(np.float32)
    return float(np.mean(predicted == target_arr))
