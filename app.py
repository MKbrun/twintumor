from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.models.future_growth_predictor import (
    train_future_predictor,
    predict_future_for_case,
)

st.set_page_config(page_title="TwinTumor Prediction Demo", layout="wide")
st.title("TwinTumor Prediction Demo")

CSV_PATH = Path("data/analysis/consistent_tumor_analysis.csv")


@st.cache_data
def load_consistent_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    return pd.read_csv(CSV_PATH)


@st.cache_resource
def load_model():
    model, samples = train_future_predictor()
    return model, samples


df = load_consistent_data()
model, samples = load_model()

patient_ids = sorted(df["subject"].unique().tolist())
selected_patient = st.selectbox("Choose patient", patient_ids)
selected_scenario = st.selectbox("Choose scenario", ["progression", "remission"])

row = df[df["subject"] == selected_patient].iloc[0]

baseline = float(row["baseline_percent"])
fu1 = float(row[f"{selected_scenario}_FU1"])
fu2 = float(row[f"{selected_scenario}_FU2"])
fu3 = float(row[f"{selected_scenario}_FU3"])
fu4 = float(row[f"{selected_scenario}_FU4"])
fu5 = float(row[f"{selected_scenario}_FU5"])

prediction = predict_future_for_case(model, baseline, fu1, fu2)

observed_x = ["Baseline", "FU1", "FU2"]
observed_y = [baseline, fu1, fu2]

pred_x = ["FU3", "FU4", "FU5"]
pred_y = [prediction["FU3"], prediction["FU4"], prediction["FU5"]]

actual_future_x = ["FU3", "FU4", "FU5"]
actual_future_y = [fu3, fu4, fu5]

left, right = st.columns([2, 1])

with left:
    st.subheader("Observed vs Predicted vs Actual Future")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(observed_x, observed_y, marker="o", label="Observed input")
    ax.plot(pred_x, pred_y, marker="o", linestyle="--", label="Predicted future")
    ax.plot(actual_future_x, actual_future_y, marker="o", linestyle=":", label="Actual future")

    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Tumor signal / value")
    ax.set_title(f"{selected_patient} - {selected_scenario}")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

with right:
    st.subheader("Summary")
    st.write(f"**Patient:** {selected_patient}")
    st.write(f"**Scenario:** {selected_scenario}")
    st.write(f"**Observed input:** Baseline, FU1, FU2")
    st.write(f"**Predicted target:** FU3, FU4, FU5")

    mae = (
        abs(prediction["FU3"] - fu3)
        + abs(prediction["FU4"] - fu4)
        + abs(prediction["FU5"] - fu5)
    ) / 3.0

    st.write(f"**Mean absolute error (FU3-FU5):** {mae:.3f}")

st.subheader("Values")

table_df = pd.DataFrame({
    "timepoint": ["Baseline", "FU1", "FU2", "FU3", "FU4", "FU5"],
    "observed_or_actual": [baseline, fu1, fu2, fu3, fu4, fu5],
    "predicted": [None, None, None, prediction["FU3"], prediction["FU4"], prediction["FU5"]],
})

st.dataframe(table_df, use_container_width=True)