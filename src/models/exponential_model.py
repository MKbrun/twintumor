"""Log-linear exponential growth fit/predict helpers.

Lifted out of `app.py` so every part of the framework can share it.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def fit_exponential(times: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
    """
    Fit V(t) = V0 * exp(k*t) by linear regression on log(V).
    Returns (k, log_V0).
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)

    if np.any(values <= 0):
        raise ValueError("Exponential fit requires strictly positive values")

    log_vals = np.log(values)
    k, log_v0 = np.polyfit(times, log_vals, 1)
    return float(k), float(log_v0)


def predict_exponential(times: np.ndarray, k: float, log_v0: float) -> np.ndarray:
    return np.exp(log_v0 + k * np.asarray(times, dtype=float))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))
