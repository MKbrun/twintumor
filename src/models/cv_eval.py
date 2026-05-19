"""
Leave-one-patient-out cross-validation for the model zoo.

Why patient-level CV
--------------------
Each patient contributes two rows (progression + remission) that share the
same baseline volume. A row-level shuffle split (as used by
`multi_model_eval.evaluate_models`) can therefore leak baseline information
from a patient's training row into their test row, inflating every model's
apparent accuracy. LOPO-CV holds out *all rows for one patient at a time*
so train and test never share a subject.

Why this matters for the thesis
-------------------------------
A digital twin claims to forecast for a *new* patient. Numbers reported
without patient-level CV describe how well the model fits its training
data, which is not the question the framework is supposed to answer.
LOPO-CV gives an honest expected error on an unseen patient.

API
---
    leave_one_patient_out(df, progress=None) -> (summary, per_patient)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.loaders import get_trajectory
from src.models.exponential_model import fit_exponential, predict_exponential
from src.models.gompertz_model import fit_gompertz, predict_gompertz
from src.models.multi_model_eval import (
    FEATURE_COLS,
    TARGET_COLS,
    _build_models,
    _build_relative_samples,
)


ProgressCallback = Callable[[int, int, str], None]


def _growth_predictions_for_row(observed: np.ndarray) -> Dict[str, np.ndarray]:
    times = np.array([0, 1, 2])
    all_times = np.arange(6)
    out: Dict[str, np.ndarray] = {}
    try:
        k, log_v0 = fit_exponential(times, observed)
        out["Exponential"] = predict_exponential(all_times, k, log_v0)[3:]
    except Exception:
        out["Exponential"] = np.full(3, np.nan)
    try:
        v_inf, b, c = fit_gompertz(times, observed)
        out["Gompertz"] = predict_gompertz(all_times, v_inf, b, c)[3:]
    except Exception:
        out["Gompertz"] = np.full(3, np.nan)
    return out


def leave_one_patient_out(
    df: pd.DataFrame,
    progress: Optional[ProgressCallback] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run patient-level CV across the cohort and return:

      summary     — model × timepoint MAE (mean over folds, original units)
      per_patient — long-form per-fold predictions for every model
    """
    samples = _build_relative_samples(df)
    if samples.empty:
        return pd.DataFrame(), pd.DataFrame()

    subjects = sorted(samples["subject"].unique())
    fu_labels = ["FU3", "FU4", "FU5"]
    df_indexed = df.set_index("subject")

    per_rows: List[dict] = []

    for fold_i, held_out in enumerate(subjects, start=1):
        if progress is not None:
            progress(fold_i, len(subjects), held_out)

        train_mask = samples["subject"] != held_out
        train, test = samples[train_mask], samples[~train_mask]
        if train.empty or test.empty:
            continue

        X_train = train[FEATURE_COLS].to_numpy()
        Y_train = train[TARGET_COLS].to_numpy()
        X_test  = test[FEATURE_COLS].to_numpy()
        Y_test  = test[TARGET_COLS].to_numpy()
        base_test = test["baseline"].to_numpy().reshape(-1, 1)

        # ML models: fit on the 104 training patients, predict on the held-out one
        ml_predictions: Dict[str, np.ndarray] = {}
        for name, est in _build_models().items():
            est.fit(X_train, Y_train)
            rel_pred = est.predict(X_test)
            ml_predictions[name] = rel_pred * base_test

        actual = Y_test * base_test  # original units

        # Per-patient growth-curve fits (independent of training set)
        growth_predictions: Dict[str, np.ndarray] = {"Exponential": [], "Gompertz": []}
        for _, s in test.iterrows():
            traj = get_trajectory(df_indexed.loc[s["subject"]], s["scenario"])
            curves = _growth_predictions_for_row(np.array(traj[:3], dtype=float))
            growth_predictions["Exponential"].append(curves["Exponential"])
            growth_predictions["Gompertz"].append(curves["Gompertz"])
        for k_ in growth_predictions:
            growth_predictions[k_] = np.array(growth_predictions[k_])

        # Per-row records
        for i, (_, s) in enumerate(test.iterrows()):
            rec = {
                "subject":  s["subject"],
                "scenario": s["scenario"],
                "baseline": float(base_test[i, 0]),
            }
            for j, fu in enumerate(fu_labels):
                rec[f"actual_{fu}"] = float(actual[i, j])
            for name, pred in {**growth_predictions, **ml_predictions}.items():
                for j, fu in enumerate(fu_labels):
                    rec[f"{name}_{fu}"] = float(pred[i, j]) if not np.isnan(pred[i, j]) else np.nan
            per_rows.append(rec)

    per_patient = pd.DataFrame(per_rows)
    if per_patient.empty:
        return pd.DataFrame(), per_patient

    # Summary — mean MAE per (model, timepoint) across the whole cohort
    model_names: List[str] = []
    seen = set()
    for col in per_patient.columns:
        for fu in fu_labels:
            suffix = f"_{fu}"
            if col.endswith(suffix) and not col.startswith("actual"):
                name = col[: -len(suffix)]
                if name not in seen:
                    seen.add(name); model_names.append(name)

    summary_rows: List[dict] = []
    actual_arr = per_patient[[f"actual_{fu}" for fu in fu_labels]].to_numpy()
    for name in model_names:
        pred_arr = per_patient[[f"{name}_{fu}" for fu in fu_labels]].to_numpy()
        for j, fu in enumerate(fu_labels):
            err = np.abs(actual_arr[:, j] - pred_arr[:, j])
            summary_rows.append({
                "Model": name, "Timepoint": fu,
                "MAE": float(np.nanmean(err)),
                "n_test": int(np.sum(~np.isnan(err))),
            })
        all_err = np.abs(actual_arr - pred_arr)
        summary_rows.append({
            "Model": name, "Timepoint": "FU3-FU5 mean",
            "MAE": float(np.nanmean(all_err)),
            "n_test": int(np.sum(~np.isnan(all_err.mean(axis=1)))),
        })

    summary = pd.DataFrame(summary_rows)
    return summary, per_patient
