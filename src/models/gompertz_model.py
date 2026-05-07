"""
Gompertz growth model for tumor trajectory forecasting.

The Gompertz model is one of the standard growth curves in
tumor biology. It captures an early near-exponential growth
phase followed by saturation toward a carrying capacity:

    V(t) = V_inf * exp(-b * exp(-c * t))

Parameters
----------
V_inf : float
    Asymptotic (carrying-capacity) volume / signal.
b : float
    Determines the initial value V(0) = V_inf * exp(-b).
c : float
    Growth deceleration rate (larger c means faster saturation).

The curve fit is done on the observed window (typically
baseline + FU1 + FU2) and then used to forecast the remaining
future timepoints.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit


def gompertz(t, V_inf: float, b: float, c: float):
    """Gompertz function V(t) = V_inf * exp(-b * exp(-c * t))."""
    t = np.asarray(t, dtype=float)
    return V_inf * np.exp(-b * np.exp(-c * t))


def fit_gompertz(
    times: np.ndarray,
    values: np.ndarray,
    v_inf_upper_multiplier: float = 10.0,
) -> Tuple[float, float, float]:
    """
    Fit Gompertz parameters (V_inf, b, c) to observed (times, values).

    The fit uses bounded non-linear least squares via scipy.curve_fit.
    Raises RuntimeError if the optimiser fails to converge.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(times) != len(values):
        raise ValueError("times and values must have the same length")

    if len(times) < 3:
        raise ValueError("Gompertz fit requires at least 3 observed points")

    if np.any(values <= 0):
        raise ValueError("Gompertz fit requires strictly positive values")

    # Gompertz needs some dynamic range; refuse a fit on flat / near-flat
    # trajectories to avoid OptimizeWarning spam and meaningless parameters.
    spread = float(np.max(values) - np.min(values))
    if spread < 1e-6 * max(abs(float(np.mean(values))), 1.0):
        raise ValueError("Trajectory is flat (no dynamic range) — Gompertz fit skipped")

    v0 = float(values[0])
    v_peak = float(np.max(values))

    # Initial parameter guesses
    v_inf_guess = max(v_peak * 1.5, v0 * 1.5)
    b_guess = np.log(max(v_inf_guess / v0, 1.01))
    c_guess = 0.3

    # Bounds chosen to keep the optimiser numerically stable.
    lower = [v0 * 1.001, 1e-4, 1e-4]
    upper = [v_peak * v_inf_upper_multiplier + 1e3, 20.0, 5.0]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                gompertz,
                times,
                values,
                p0=[v_inf_guess, b_guess, c_guess],
                bounds=(lower, upper),
                maxfev=10000,
            )
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Gompertz fit failed: {exc}") from exc

    return float(popt[0]), float(popt[1]), float(popt[2])


def predict_gompertz(
    times: np.ndarray,
    V_inf: float,
    b: float,
    c: float,
) -> np.ndarray:
    """Predict Gompertz values at the given times using fitted parameters."""
    return gompertz(times, V_inf, b, c)


def fit_and_predict_gompertz(
    observed_times: np.ndarray,
    observed_values: np.ndarray,
    forecast_times: np.ndarray,
) -> np.ndarray:
    """
    Convenience function: fit on the observed window, predict at the forecast times.
    """
    V_inf, b, c = fit_gompertz(observed_times, observed_values)
    return predict_gompertz(forecast_times, V_inf, b, c)
