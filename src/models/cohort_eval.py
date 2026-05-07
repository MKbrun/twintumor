"""
Cohort-level forecasting helpers shared between the Streamlit demo and any
batch evaluation script. Lifted out of `app.py`.

`forecast_all_models` takes one observed window (Baseline, FU1, FU2) and
returns the Exponential / Gompertz forecasts for FU3..FU5.

`build_cohort_predictions` runs every (subject, scenario) row in the cohort
through Exponential, Gompertz and the trained ML predictor and returns a
long-form dataframe of actuals vs predictions.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.data.loaders import get_trajectory
from src.models.exponential_model import (
    fit_exponential,
    predict_exponential,
)
from src.models.gompertz_model import fit_gompertz, predict_gompertz
from src.models.ml_predictor import predict_future


OBSERVED_TIMES = np.array([0, 1, 2])
ALL_TIMES = np.arange(6)


def forecast_all_models(observed_values: list[float]) -> Dict[str, dict]:
    """Fit Exponential and Gompertz to the observed window."""
    observed = np.asarray(observed_values, dtype=float)
    results: Dict[str, dict] = {}

    try:
        k, log_v0 = fit_exponential(OBSERVED_TIMES, observed)
        full = predict_exponential(ALL_TIMES, k, log_v0)
        results["Exponential"] = {"full": full, "future": full[3:], "ok": True, "error": None}
    except Exception as exc:
        results["Exponential"] = {"full": None, "future": None, "ok": False, "error": str(exc)}

    try:
        v_inf, b, c = fit_gompertz(OBSERVED_TIMES, observed)
        full = predict_gompertz(ALL_TIMES, v_inf, b, c)
        results["Gompertz"] = {
            "full": full,
            "future": full[3:],
            "ok": True,
            "error": None,
            "params": {"V_inf": v_inf, "b": b, "c": c},
        }
    except Exception as exc:
        results["Gompertz"] = {"full": None, "future": None, "ok": False, "error": str(exc)}

    return results


def build_cohort_predictions(df: pd.DataFrame, ml_model) -> pd.DataFrame:
    """Fit every model on every (subject, scenario) row's observed window."""
    records: list[dict] = []
    for _, row in df.iterrows():
        for scenario in ("progression", "remission"):
            traj = get_trajectory(row, scenario)
            observed = traj[:3]
            actual = traj[3:]

            forecasts = forecast_all_models(observed)
            exp_future = forecasts["Exponential"]["future"]
            gom_future = forecasts["Gompertz"]["future"]

            ml_pred = predict_future(ml_model, observed[0], observed[1], observed[2])
            ml_future = np.array([ml_pred["FU3"], ml_pred["FU4"], ml_pred["FU5"]])

            rec = {
                "subject": row["subject"],
                "scenario": scenario,
                "baseline": observed[0],
                "FU1": observed[1],
                "FU2": observed[2],
                "actual_FU3": actual[0],
                "actual_FU4": actual[1],
                "actual_FU5": actual[2],
                "exp_FU3": float(exp_future[0]) if exp_future is not None else np.nan,
                "exp_FU4": float(exp_future[1]) if exp_future is not None else np.nan,
                "exp_FU5": float(exp_future[2]) if exp_future is not None else np.nan,
                "gom_FU3": float(gom_future[0]) if gom_future is not None else np.nan,
                "gom_FU4": float(gom_future[1]) if gom_future is not None else np.nan,
                "gom_FU5": float(gom_future[2]) if gom_future is not None else np.nan,
                "ml_FU3":  float(ml_future[0]),
                "ml_FU4":  float(ml_future[1]),
                "ml_FU5":  float(ml_future[2]),
            }
            records.append(rec)
    return pd.DataFrame(records)
