from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.models.future_growth_predictor import (
    train_future_predictor,
    predict_future_for_case,
)

st.set_page_config(page_title="TwinTumor Digital Twin Demo", layout="wide")
st.title("TwinTumor – Digital Twin Growth Prediction")

CSV_PATH = Path("data/analysis/consistent_tumor_analysis.csv")


@st.cache_data
def load_consistent_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    return pd.read_csv(CSV_PATH)


@st.cache_resource
def load_model():
    model, samples = train_future_predictor(CSV_PATH)
    return model, samples


def fit_exponential(times: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """
    Fit exponential model:
        y = exp(log_v0 + k*t)
    using the observed values.
    """
    values = np.asarray(values, dtype=float)

    if np.any(values <= 0):
        raise ValueError("Exponential model requires all values > 0")

    log_vals = np.log(values)
    k, log_v0 = np.polyfit(times, log_vals, 1)
    return float(k), float(log_v0)


def predict_exponential(times: np.ndarray, k: float, log_v0: float) -> np.ndarray:
    return np.exp(log_v0 + k * times)


def mean_absolute_error_list(y_true: list[float], y_pred: list[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


df = load_consistent_data()
model, samples = load_model()

patient_ids = sorted(df["subject"].unique().tolist())

left_sidebar, right_sidebar = st.columns([1, 1])
with left_sidebar:
    selected_patient = st.selectbox("Choose patient", patient_ids)

with right_sidebar:
    selected_scenario = st.selectbox("Choose scenario", ["progression", "remission"])

row = df[df["subject"] == selected_patient].iloc[0]

baseline = float(row["baseline_percent"])
fu1 = float(row[f"{selected_scenario}_FU1"])
fu2 = float(row[f"{selected_scenario}_FU2"])
fu3 = float(row[f"{selected_scenario}_FU3"])
fu4 = float(row[f"{selected_scenario}_FU4"])
fu5 = float(row[f"{selected_scenario}_FU5"])

observed_history = [baseline, fu1, fu2]
actual_future = [fu3, fu4, fu5]
actual_full = observed_history + actual_future

# ML prediction
ml_prediction = predict_future_for_case(model, baseline, fu1, fu2)
ml_future = [
    ml_prediction["FU3"],
    ml_prediction["FU4"],
    ml_prediction["FU5"],
]
ml_full = observed_history + ml_future

# Exponential baseline model
exp_future = [None, None, None]
exp_error = None

try:
    k, log_v0 = fit_exponential(
        np.array([0, 1, 2], dtype=float),
        np.array(observed_history, dtype=float),
    )
    exp_pred_all = predict_exponential(np.array([0, 1, 2, 3, 4, 5], dtype=float), k, log_v0)
    exp_future = exp_pred_all[3:].tolist()
    exp_full = exp_pred_all.tolist()
    exp_error = mean_absolute_error_list(actual_future, exp_future)
except ValueError:
    exp_full = [None] * 6

ml_error = mean_absolute_error_list(actual_future, ml_future)

plot_col, info_col = st.columns([2, 1])

with plot_col:
    st.subheader("Observed Growth vs Predicted Future")

    times = np.arange(6)
    labels = ["Baseline", "FU1", "FU2", "FU3", "FU4", "FU5"]

    fig, ax = plt.subplots(figsize=(11, 6))

    # Actual full trajectory
    ax.plot(times, actual_full, marker="o", linewidth=2.5, label="Actual trajectory")

    # ML full predicted trajectory
    ax.plot(times, ml_full, marker="o", linestyle="--", linewidth=2, label="ML prediction")

    # Exponential full predicted trajectory
    if exp_full[0] is not None:
        ax.plot(times, exp_full, marker="o", linestyle=":", linewidth=2, label="Exponential baseline")

    # Visual split between observed and future
    ax.axvline(x=2, linestyle="--", alpha=0.6)
    ax.text(0.2, max(actual_full) * 1.02, "Observed", fontsize=10)
    ax.text(3.05, max(actual_full) * 1.02, "Predicted future", fontsize=10)

    ax.set_xticks(times)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Tumor value / signal")
    ax.set_title(f"{selected_patient} – {selected_scenario}")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

with info_col:
    st.subheader("Summary")
    st.write(f"**Patient:** {selected_patient}")
    st.write(f"**Scenario:** {selected_scenario}")
    st.write("**Observed input:** Baseline, FU1, FU2")
    st.write("**Prediction target:** FU3, FU4, FU5")
    st.write(f"**ML mean absolute error:** {ml_error:.3f}")

    if exp_error is not None:
        st.write(f"**Exponential mean absolute error:** {exp_error:.3f}")
    else:
        st.write("**Exponential mean absolute error:** Not available")

    if exp_error is not None:
        if ml_error < exp_error:
            st.success("ML performs better than the exponential baseline on this case.")
        elif ml_error > exp_error:
            st.warning("Exponential baseline performs better than ML on this case.")
        else:
            st.info("ML and exponential baseline perform equally on this case.")

st.subheader("Trajectory values")

table_df = pd.DataFrame({
    "timepoint": ["Baseline", "FU1", "FU2", "FU3", "FU4", "FU5"],
    "actual": actual_full,
    "ml_prediction": [None, None, None] + ml_future,
    "exponential_prediction": [None, None, None] + (
        exp_future if exp_future[0] is not None else [None, None, None]
    ),
})

st.dataframe(table_df, use_container_width=True)

st.subheader("Interpretation")

st.markdown(
    """
This view represents a simple digital twin style workflow:

- **Observed history**: the patient trajectory known so far
- **ML prediction**: predicted future growth from the trained regression model
- **Exponential baseline**: simple mathematical growth baseline
- **Actual future**: ground truth for comparison

The goal is to show how a model can estimate future tumor development from early timepoints.
"""
)