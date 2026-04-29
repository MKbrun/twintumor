"""
TwinTumor Digital Twin Demo UI

A Streamlit app that demonstrates a rule-based + growth-curve + ML pipeline
for forecasting brain metastasis trajectories from longitudinal MRI data.

The demo uses the first three observed timepoints (baseline, FU1, FU2) to
forecast the remaining three (FU3, FU4, FU5) with three different models
and visualises them at the single-patient and cohort level.

Tabs
----
- Single-patient twin : per-patient forecast with Exponential / Gompertz / ML
- Cohort view        : overlay the selected patient on the population
- Heatmaps           : cohort trajectory heatmap + prediction-error heatmap
- Model comparison   : MAE distribution per model across the full dataset
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.models.future_growth_predictor import (
    train_future_predictor,
    predict_future_for_case,
)
from src.models.gompertz_model import fit_gompertz, predict_gompertz


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="TwinTumor Digital Twin",
    page_icon="🧠",
    layout="wide",
)

CSV_PATH = Path("data/analysis/consistent_tumor_analysis.csv")

TIME_LABELS = ["Baseline", "FU1", "FU2", "FU3", "FU4", "FU5"]
OBSERVED_IDX = [0, 1, 2]
FUTURE_IDX = [3, 4, 5]


# ----------------------------------------------------------------------
# Data & model loading
# ----------------------------------------------------------------------

@st.cache_data
def load_consistent_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    return pd.read_csv(CSV_PATH)


@st.cache_resource
def load_ml_model():
    model, samples = train_future_predictor(CSV_PATH)
    return model, samples


# ----------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------

def get_trajectory(row: pd.Series, scenario: str) -> list[float]:
    return [
        float(row["baseline_percent"]),
        float(row[f"{scenario}_FU1"]),
        float(row[f"{scenario}_FU2"]),
        float(row[f"{scenario}_FU3"]),
        float(row[f"{scenario}_FU4"]),
        float(row[f"{scenario}_FU5"]),
    ]


def fit_exponential(times: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0):
        raise ValueError("Exponential requires positive values")
    log_vals = np.log(values)
    k, log_v0 = np.polyfit(times, log_vals, 1)
    return float(k), float(log_v0)


def predict_exponential(times: np.ndarray, k: float, log_v0: float) -> np.ndarray:
    return np.exp(log_v0 + k * np.asarray(times, dtype=float))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def ml_prediction_with_uncertainty(
    model, baseline: float, fu1: float, fu2: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    For the MultiOutputRegressor(RandomForestRegressor) model,
    query the individual trees to build an ensemble mean and
    standard deviation for each of FU3, FU4, FU5.
    """
    X = pd.DataFrame([{"baseline": baseline, "FU1": fu1, "FU2": fu2}])
    means: list[float] = []
    stds: list[float] = []
    for rf in model.estimators_:
        tree_preds = np.array([tree.predict(X.values)[0] for tree in rf.estimators_])
        means.append(float(tree_preds.mean()))
        stds.append(float(tree_preds.std()))
    return np.array(means), np.array(stds)


def forecast_all_models(observed_values: list[float]):
    """
    Fit Exponential and Gompertz on the observed window and
    return (full_curve_over_6_points, future_part, mae_or_none)
    for each model. Returns a dict keyed by model name.
    """
    observed = np.asarray(observed_values, dtype=float)
    times_all = np.arange(6)

    results: dict[str, dict] = {}

    # Exponential
    try:
        k, log_v0 = fit_exponential(np.array([0, 1, 2]), observed)
        exp_full = predict_exponential(times_all, k, log_v0)
        results["Exponential"] = {
            "full": exp_full,
            "future": exp_full[3:],
            "ok": True,
            "error": None,
        }
    except Exception as e:  # pragma: no cover
        results["Exponential"] = {"full": None, "future": None, "ok": False, "error": str(e)}

    # Gompertz
    try:
        V_inf, b, c = fit_gompertz(np.array([0, 1, 2]), observed)
        gom_full = predict_gompertz(times_all, V_inf, b, c)
        results["Gompertz"] = {
            "full": gom_full,
            "future": gom_full[3:],
            "params": {"V_inf": V_inf, "b": b, "c": c},
            "ok": True,
            "error": None,
        }
    except Exception as e:
        results["Gompertz"] = {"full": None, "future": None, "ok": False, "error": str(e)}

    return results


# ----------------------------------------------------------------------
# Cohort-level predictions (cached)
# ----------------------------------------------------------------------

@st.cache_data
def build_cohort_predictions(_df: pd.DataFrame, _ml_model) -> pd.DataFrame:
    """
    For every (subject, scenario), fit exponential and Gompertz on the first
    three observed points and compare with the trained ML model. Returns a
    long-form dataframe with actual and predicted FU3/FU4/FU5 values.
    """
    records: list[dict] = []
    for _, row in _df.iterrows():
        subject = row["subject"]
        for scenario in ["progression", "remission"]:
            traj = get_trajectory(row, scenario)
            observed = traj[:3]
            actual = traj[3:]

            forecasts = forecast_all_models(observed)

            exp_future = forecasts["Exponential"]["future"]
            gom_future = forecasts["Gompertz"]["future"]

            pred = predict_future_for_case(_ml_model, observed[0], observed[1], observed[2])
            ml_future = np.array([pred["FU3"], pred["FU4"], pred["FU5"]])

            rec = {
                "subject": subject,
                "scenario": scenario,
                "baseline": observed[0],
                "FU1": observed[1],
                "FU2": observed[2],
                "actual_FU3": actual[0],
                "actual_FU4": actual[1],
                "actual_FU5": actual[2],
            }
            rec.update({
                "exp_FU3": float(exp_future[0]) if exp_future is not None else np.nan,
                "exp_FU4": float(exp_future[1]) if exp_future is not None else np.nan,
                "exp_FU5": float(exp_future[2]) if exp_future is not None else np.nan,
            })
            rec.update({
                "gom_FU3": float(gom_future[0]) if gom_future is not None else np.nan,
                "gom_FU4": float(gom_future[1]) if gom_future is not None else np.nan,
                "gom_FU5": float(gom_future[2]) if gom_future is not None else np.nan,
            })
            rec.update({
                "ml_FU3": float(ml_future[0]),
                "ml_FU4": float(ml_future[1]),
                "ml_FU5": float(ml_future[2]),
            })
            records.append(rec)

    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# UI entry point
# ----------------------------------------------------------------------

st.title("TwinTumor – Digital Twin Growth Prediction")
st.caption(
    "Forecasting brain-metastasis trajectories from the first three longitudinal "
    "MRI timepoints using an exponential baseline, a Gompertz growth model and "
    "a Random Forest digital twin."
)

if not CSV_PATH.exists():
    st.error(
        f"Data file not found at `{CSV_PATH}`.\n\n"
        "Please place `consistent_tumor_analysis.csv` at that path and reload."
    )
    st.stop()

df = load_consistent_data()
ml_model, _samples = load_ml_model()

# Sidebar selection
with st.sidebar:
    st.header("Patient selection")
    patient_ids = sorted(df["subject"].unique().tolist())
    selected_patient = st.selectbox("Patient", patient_ids)
    selected_scenario = st.selectbox("Scenario", ["progression", "remission"])
    st.markdown("---")
    st.markdown("### About the demo")
    st.write(
        "Observed window: Baseline, FU1, FU2.\n\n"
        "Forecast window: FU3, FU4, FU5.\n\n"
        f"Cohort size: **{len(df)} patients × 2 scenarios = {2*len(df)} trajectories**."
    )

row = df[df["subject"] == selected_patient].iloc[0]
trajectory = get_trajectory(row, selected_scenario)
observed = trajectory[:3]
actual_future = trajectory[3:]
actual_full = trajectory

tab_single, tab_cohort, tab_heatmap, tab_compare = st.tabs([
    "Single-patient twin",
    "Cohort view",
    "Heatmaps",
    "Model comparison",
])


# ======================================================================
# TAB 1: Single-patient twin
# ======================================================================

with tab_single:
    st.subheader("Single-patient digital twin")
    st.write(
        "The observed values (Baseline, FU1, FU2) drive three forecasts: "
        "a log-linear **exponential** fit, a **Gompertz** saturating-growth fit, "
        "and a **Random Forest** ML model trained on the full cohort. "
        "The shaded band around the ML line is the ensemble ±1σ the spread of "
        "the 200 individual regression trees which gives a sense of the twin's "
        "forecast uncertainty."
    )

    forecasts = forecast_all_models(observed)

    # ML prediction with uncertainty
    ml_mean, ml_std = ml_prediction_with_uncertainty(ml_model, observed[0], observed[1], observed[2])
    ml_full = np.concatenate([observed, ml_mean])
    ml_err = mae(actual_future, ml_mean)

    times_all = np.arange(6)

    col_plot, col_info = st.columns([2, 1])

    with col_plot:
        fig, ax = plt.subplots(figsize=(11, 6))

        # Actual trajectory
        ax.plot(times_all, actual_full, marker="o", linewidth=2.5,
                color="black", label="Actual", zorder=5)

        # Exponential
        if forecasts["Exponential"]["ok"]:
            exp_full = forecasts["Exponential"]["full"]
            exp_err = mae(actual_future, forecasts["Exponential"]["future"])
            ax.plot(times_all, exp_full, marker="s", linestyle=":", linewidth=2,
                    color="tab:orange",
                    label=f"Exponential (MAE = {exp_err:.2f})")
        else:
            exp_err = None

        # Gompertz
        if forecasts["Gompertz"]["ok"]:
            gom_full = forecasts["Gompertz"]["full"]
            gom_err = mae(actual_future, forecasts["Gompertz"]["future"])
            ax.plot(times_all, gom_full, marker="^", linestyle="-.", linewidth=2,
                    color="tab:green",
                    label=f"Gompertz (MAE = {gom_err:.2f})")
        else:
            gom_err = None

        # ML + uncertainty band
        ax.plot(times_all, ml_full, marker="d", linestyle="--", linewidth=2,
                color="tab:blue",
                label=f"ML (MAE = {ml_err:.2f})")
        future_times = times_all[3:]
        ax.fill_between(
            future_times,
            ml_mean - ml_std,
            ml_mean + ml_std,
            color="tab:blue", alpha=0.18,
            label="ML ensemble ±1σ",
        )

        # Separator between observed and forecast
        ax.axvline(x=2, linestyle="--", alpha=0.4, color="gray")
        ymin, ymax = ax.get_ylim()
        ax.text(0.1, ymax * 0.97, "Observed", fontsize=10,
                color="gray", ha="left", va="top")
        ax.text(3.1, ymax * 0.97, "Forecast", fontsize=10,
                color="gray", ha="left", va="top")

        ax.set_xticks(times_all)
        ax.set_xticklabels(TIME_LABELS)
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Tumor signal (%)")
        ax.set_title(f"{selected_patient} – {selected_scenario}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        st.pyplot(fig)

    with col_info:
        st.markdown("### Forecast summary")
        err_rows = []
        if forecasts["Exponential"]["ok"]:
            err_rows.append(("Exponential", exp_err))
        if forecasts["Gompertz"]["ok"]:
            err_rows.append(("Gompertz", gom_err))
        err_rows.append(("ML (Random Forest)", ml_err))

        err_df = pd.DataFrame(err_rows, columns=["Model", "MAE (FU3–FU5)"])
        err_df["MAE (FU3–FU5)"] = err_df["MAE (FU3–FU5)"].round(3)
        st.dataframe(err_df, hide_index=True, width="stretch")

        if err_rows:
            best = min(err_rows, key=lambda r: r[1])
            st.success(f"Best on this case: **{best[0]}** (MAE = {best[1]:.2f})")

        if forecasts["Gompertz"]["ok"]:
            p = forecasts["Gompertz"]["params"]
            st.markdown("### Gompertz parameters")
            st.write(
                f"V∞ = {p['V_inf']:.2f}  \n"
                f"b = {p['b']:.3f}  \n"
                f"c = {p['c']:.3f}"
            )
        elif not forecasts["Gompertz"]["ok"]:
            st.warning(f"Gompertz fit failed: {forecasts['Gompertz']['error']}")

    # Trajectory values table
    st.markdown("#### Trajectory values")
    table = {"timepoint": TIME_LABELS, "actual": actual_full}
    if forecasts["Exponential"]["ok"]:
        table["exponential"] = np.round(forecasts["Exponential"]["full"], 3).tolist()
    if forecasts["Gompertz"]["ok"]:
        table["gompertz"] = np.round(forecasts["Gompertz"]["full"], 3).tolist()
    table["ml"] = np.round(ml_full, 3).tolist()
    table["ml ±1σ"] = [None, None, None] + [f"±{s:.2f}" for s in ml_std]
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")


# ======================================================================
# TAB 2: Cohort view
# ======================================================================

with tab_cohort:
    st.subheader("Cohort overlay")
    st.write(
        "Where does this patient sit in the cohort? The thin grey lines are "
        "individual patient trajectories for the selected scenario, the blue "
        "band is the cohort **mean ±1 standard deviation**, and the bold black "
        "line highlights the selected patient."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        n_overlay = st.slider(
            "How many random patients to overlay", 5, len(df), 40,
            help="All cohort statistics are computed from the full dataset; "
                 "this only controls how many thin lines are drawn.",
        )
    with col_b:
        same_scenario = st.checkbox(
            "Use the currently selected scenario", value=True,
            help="Uncheck to compare progression and remission side-by-side.",
        )

    # Full-cohort trajectory matrix per scenario
    traj_by_scenario: dict[str, np.ndarray] = {}
    for scen in ["progression", "remission"]:
        rows = []
        for _, r in df.iterrows():
            rows.append(get_trajectory(r, scen))
        traj_by_scenario[scen] = np.array(rows)

    cohort_ids = df["subject"].tolist()
    rng = np.random.default_rng(42)
    overlay_ids = rng.choice(cohort_ids, size=min(n_overlay, len(cohort_ids)), replace=False).tolist()
    if selected_patient not in overlay_ids:
        overlay_ids[0] = selected_patient

    if same_scenario:
        fig, ax = plt.subplots(figsize=(11, 6))
        traj_matrix = traj_by_scenario[selected_scenario]
        for pid in overlay_ids:
            idx = cohort_ids.index(pid)
            ax.plot(range(6), traj_matrix[idx], color="gray", alpha=0.25, linewidth=1)
        mean_traj = traj_matrix.mean(axis=0)
        std_traj = traj_matrix.std(axis=0)
        ax.fill_between(range(6), mean_traj - std_traj, mean_traj + std_traj,
                        color="steelblue", alpha=0.2, label="Cohort ±1σ")
        ax.plot(range(6), mean_traj, color="steelblue", linewidth=2, label="Cohort mean")
        ax.plot(range(6), trajectory, color="black", linewidth=3, marker="o",
                label=f"Selected: {selected_patient}")
        ax.axvline(x=2, linestyle="--", alpha=0.4, color="gray")
        ax.set_xticks(range(6))
        ax.set_xticklabels(TIME_LABELS)
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Tumor signal (%)")
        ax.set_title(f"Cohort overlay – {selected_scenario}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, scen in zip(axes, ["progression", "remission"]):
            traj_matrix = traj_by_scenario[scen]
            for pid in overlay_ids:
                idx = cohort_ids.index(pid)
                ax.plot(range(6), traj_matrix[idx], color="gray", alpha=0.25, linewidth=1)
            mean_traj = traj_matrix.mean(axis=0)
            std_traj = traj_matrix.std(axis=0)
            ax.fill_between(range(6), mean_traj - std_traj, mean_traj + std_traj,
                            color="steelblue", alpha=0.2, label="Cohort ±1σ")
            ax.plot(range(6), mean_traj, color="steelblue", linewidth=2, label="Cohort mean")
            ax.plot(range(6), get_trajectory(row, scen),
                    color="black", linewidth=3, marker="o",
                    label=f"Selected: {selected_patient}")
            ax.axvline(x=2, linestyle="--", alpha=0.4, color="gray")
            ax.set_xticks(range(6))
            ax.set_xticklabels(TIME_LABELS)
            ax.set_title(scen)
            ax.set_xlabel("Timepoint")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
        axes[0].set_ylabel("Tumor signal (%)")
        fig.suptitle("Cohort overlay — progression vs remission", y=1.02)
        st.pyplot(fig)

    # Nearest-neighbor view
    st.subheader("Nearest neighbours (case-based twin)")
    st.write(
        "The five patients whose observed window (Baseline, FU1, FU2) is closest "
        "to the selected patient in Euclidean distance. This is the simplest "
        "expression of a digital twin: find similar past cases and use them "
        "as a reference for the future."
    )

    k_neighbors = st.slider("Number of neighbours", 3, 15, 5)

    observed_selected = np.array(observed)
    distances = []
    for _, r in df.iterrows():
        other_obs = np.array(get_trajectory(r, selected_scenario)[:3])
        distances.append((r["subject"], float(np.linalg.norm(other_obs - observed_selected))))
    distances.sort(key=lambda x: x[1])
    neighbours = [d for d in distances if d[0] != selected_patient][:k_neighbors]

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.tab10
    for i, (pid, dist) in enumerate(neighbours):
        r_n = df[df["subject"] == pid].iloc[0]
        traj_n = get_trajectory(r_n, selected_scenario)
        ax2.plot(range(6), traj_n, alpha=0.85, linewidth=1.8,
                 color=cmap(i % 10), marker="o",
                 label=f"{pid} (d={dist:.2f})")
    ax2.plot(range(6), trajectory, color="black", linewidth=3, marker="s",
             label=f"Selected: {selected_patient}")
    ax2.axvline(x=2, linestyle="--", alpha=0.4, color="gray")
    ax2.set_xticks(range(6))
    ax2.set_xticklabels(TIME_LABELS)
    ax2.set_xlabel("Timepoint")
    ax2.set_ylabel("Tumor signal (%)")
    ax2.set_title(f"{k_neighbors} most similar patients — {selected_scenario}")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)
    st.pyplot(fig2)


# ======================================================================
# TAB 3: Heatmaps
# ======================================================================

with tab_heatmap:
    st.subheader("Cohort trajectory heatmap")
    st.write(
        "One row per patient, one column per timepoint, colour encoding the "
        "tumor-signal percentage. Sorting the rows by baseline value or by "
        "total change exposes cohort-wide growth patterns: consistent bright "
        "bands indicate sustained high signal, vertical gradients indicate "
        "growing or shrinking tumors."
    )

    sort_mode = st.radio(
        "Sort patients by",
        ["Baseline", "Total change (FU5 − baseline)", "Subject ID"],
        horizontal=True,
        key="heatmap_sort",
    )

    traj_matrix = np.array([get_trajectory(r, selected_scenario) for _, r in df.iterrows()])
    ids = df["subject"].values

    if sort_mode == "Baseline":
        order = np.argsort(traj_matrix[:, 0])
    elif sort_mode == "Total change (FU5 − baseline)":
        order = np.argsort(traj_matrix[:, -1] - traj_matrix[:, 0])
    else:
        order = np.argsort(ids)

    sorted_matrix = traj_matrix[order]
    sorted_ids = ids[order]

    fig, ax = plt.subplots(figsize=(9, max(6, len(sorted_ids) * 0.12)))
    im = ax.imshow(sorted_matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(6))
    ax.set_xticklabels(TIME_LABELS)
    ax.set_yticks(range(len(sorted_ids)))
    ax.set_yticklabels(sorted_ids, fontsize=6)
    sel_pos = np.where(sorted_ids == selected_patient)[0]
    if len(sel_pos) > 0:
        ax.axhline(sel_pos[0], color="red", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Timepoint")
    ax.set_title(f"Cohort heatmap — {selected_scenario}")
    fig.colorbar(im, ax=ax, label="Tumor signal (%)")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Prediction-error heatmap")
    st.write(
        "Absolute forecast error `|actual − predicted|` for every patient and "
        "future timepoint. Darker cells mean better predictions, brighter cells "
        "mean the model struggled. Useful for spotting outlier cases and "
        "systematic per-timepoint errors."
    )

    model_choice = st.selectbox(
        "Model",
        ["Exponential", "Gompertz", "ML (Random Forest)"],
        key="error_heatmap_model",
    )

    with st.spinner("Building cohort predictions…"):
        pred_table = build_cohort_predictions(df, ml_model)

    scenario_rows = pred_table[pred_table["scenario"] == selected_scenario].copy()
    actual = scenario_rows[["actual_FU3", "actual_FU4", "actual_FU5"]].values

    if model_choice == "Exponential":
        pred = scenario_rows[["exp_FU3", "exp_FU4", "exp_FU5"]].values
    elif model_choice == "Gompertz":
        pred = scenario_rows[["gom_FU3", "gom_FU4", "gom_FU5"]].values
    else:
        pred = scenario_rows[["ml_FU3", "ml_FU4", "ml_FU5"]].values

    err = np.abs(actual - pred)
    mean_err = np.nanmean(err, axis=1)
    err_order = np.argsort(mean_err)
    err_sorted = err[err_order]
    ids_sorted = scenario_rows["subject"].values[err_order]

    fig2, ax2 = plt.subplots(figsize=(7, max(6, len(ids_sorted) * 0.12)))
    im2 = ax2.imshow(err_sorted, aspect="auto", cmap="magma")
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["FU3", "FU4", "FU5"])
    ax2.set_yticks(range(len(ids_sorted)))
    ax2.set_yticklabels(ids_sorted, fontsize=6)
    sel_pos = np.where(ids_sorted == selected_patient)[0]
    if len(sel_pos) > 0:
        ax2.axhline(sel_pos[0], color="cyan", linewidth=1.2, alpha=0.85)
    ax2.set_xlabel("Future timepoint")
    ax2.set_title(f"{model_choice}: |actual − predicted|  ({selected_scenario})")
    fig2.colorbar(im2, ax=ax2, label="Absolute error")
    st.pyplot(fig2)


# ======================================================================
# TAB 4: Model comparison
# ======================================================================

with tab_compare:
    st.subheader("Model comparison across the full cohort")
    st.write(
        "MAE on FU3/FU4/FU5 computed per patient and per scenario for the "
        "Exponential, Gompertz and Random Forest models. Box plots show the "
        "full distribution; the bar chart contrasts the mean MAE by scenario; "
        "the per-timepoint table reveals which horizon is hardest to forecast."
    )

    with st.spinner("Building cohort predictions…"):
        pred_table = build_cohort_predictions(df, ml_model)

    actual_cols = ["actual_FU3", "actual_FU4", "actual_FU5"]
    model_cols = {
        "Exponential": ["exp_FU3", "exp_FU4", "exp_FU5"],
        "Gompertz":    ["gom_FU3", "gom_FU4", "gom_FU5"],
        "ML (RF)":     ["ml_FU3",  "ml_FU4",  "ml_FU5"],
    }

    mae_records: list[dict] = []
    for model_name, pred_cols in model_cols.items():
        err = np.abs(pred_table[actual_cols].values - pred_table[pred_cols].values)
        per_case_mae = np.nanmean(err, axis=1)
        for m_val, scen in zip(per_case_mae, pred_table["scenario"]):
            if not np.isnan(m_val):
                mae_records.append({"Model": model_name, "Scenario": scen, "MAE": m_val})
    mae_df = pd.DataFrame(mae_records)

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [mae_df[mae_df["Model"] == m]["MAE"].values for m in model_cols.keys()]
        bp = ax.boxplot(data, labels=list(model_cols.keys()), patch_artist=True)
        colors = ["tab:orange", "tab:green", "tab:blue"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_ylabel("MAE (FU3–FU5)")
        ax.set_title("Per-patient forecast MAE distribution")
        ax.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(8, 5))
        summary = mae_df.groupby(["Model", "Scenario"])["MAE"].mean().unstack()
        summary = summary.reindex(list(model_cols.keys()))
        summary.plot(kind="bar", ax=ax, color=["tab:cyan", "tab:red"])
        ax.set_ylabel("Mean MAE")
        ax.set_title("Mean MAE by scenario")
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=0)
        st.pyplot(fig)

    # Summary table
    st.markdown("### Summary statistics")
    summary_tbl = (
        mae_df.groupby(["Model", "Scenario"])["MAE"]
        .agg(["mean", "median", "std", "count"])
        .round(3)
    )
    st.dataframe(summary_tbl)

    # Per-timepoint MAE
    st.markdown("### Per-timepoint MAE (both scenarios combined)")
    per_tp_records = []
    for model_name, pred_cols in model_cols.items():
        for i, tp in enumerate(["FU3", "FU4", "FU5"]):
            err_col = np.abs(pred_table[actual_cols[i]].values - pred_table[pred_cols[i]].values)
            per_tp_records.append({
                "Model": model_name,
                "Timepoint": tp,
                "MAE": float(np.nanmean(err_col)),
            })
    per_tp_df = pd.DataFrame(per_tp_records)
    pivot = per_tp_df.pivot(index="Model", columns="Timepoint", values="MAE").round(3)
    pivot = pivot.reindex(list(model_cols.keys()))
    st.dataframe(pivot, width="stretch")

    # Download raw predictions
    st.markdown("### Export")
    csv_bytes = pred_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cohort prediction table (CSV)",
        data=csv_bytes,
        file_name="cohort_predictions.csv",
        mime="text/csv",
    )
