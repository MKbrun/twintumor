"""
TwinTumor — Digital Twin Demo (Streamlit).

Pipeline used by this app:
    1. The user points the app at a folder of Mets_* patient series.
    2. `build_volumes_csv` walks every patient and computes tumor volume in
       mm^3 from each timepoint's seg.nii (real MRI-derived volumes).
    3. The wide CSV (data/processed/tumor_volumes.csv) is the input to the
       Exponential / Gompertz / Random-Forest forecasters.
    4. The Random Forest is persisted to disk (data/models/rf_predictor.joblib)
       so users do NOT retrain on every launch and so a clone-and-run user
       with only a handful of patients can still load a model trained on the
       full cohort.
    5. The RANO rule agent classifies each timepoint and the labels are shown
       alongside the forecasts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.agent.treatment_agent import TreatmentAgent
from src.data.loaders import (
    FUTURE_IDX,
    OBSERVED_IDX,
    TIME_LABELS,
    flat_trajectory_summary,
    get_trajectory,
    is_trajectory_flat,
    load_dataset,
)
from src.data.paths import (
    DEFAULT_ML_MODEL_PATH,
    DEMO_DATASET_CSV,
    RAW_DATA_DIR,
    TUMOR_VOLUMES_CSV,
)
from src.models.cohort_eval import build_cohort_predictions, forecast_all_models
from src.models.cv_eval import leave_one_patient_out
from src.models.multi_model_eval import evaluate_models as eval_extra_models
from src.pipelines.single_patient import (
    discover_scenarios,
    forecast_for_patient,
    read_patient_volumes,
)
from src.pipelines.fu_signal_masks import compute_fu_signal_masks
from src.pipelines.otsu_signal_extractor import detect_root_layout
from src.io.mri_loader import load_mri_volume
from src.io.nifti_loader import load_mask
from src.viz.growth_heatmap import (
    GrowthLayer,
    grow_mask_2d_to_area,
    predicted_growth_layers_2d,
    render_growth_view,
    render_side_by_side,
    synthetic_disk_mask,
)
from src.models.exponential_model import mae
from src.models.ml_predictor import (
    DEFAULT_INCREMENTAL_TREES,
    get_cached_lopo_mae,
    get_or_train,
    predict_future,
    predict_future_with_uncertainty,
    reset_model,
    train_fresh_from,
    train_incremental,
)
from src.pipelines.build_volumes_csv import build_and_save


# ---------------------------------------------------------------- page setup

st.set_page_config(page_title="TwinTumor Digital Twin", page_icon="🧠", layout="wide")


# ---------------------------------------------------------------- caching

@st.cache_data(show_spinner=False)
def _load_dataset(csv_path: str, mtime: float) -> pd.DataFrame:
    """Cache key includes file mtime so a rebuild invalidates automatically."""
    return load_dataset(csv_path)


@st.cache_resource(show_spinner=False)
def _load_or_train_model(model_path: str, model_mtime: float):
    """Cache-keyed on the model file's mtime so retraining auto-invalidates."""
    return get_or_train(model_path)


# ---------------------------------------------------------------- data sources

DATA_SOURCE_DEMO = "Bundled demo dataset (105 patients)"
DATA_SOURCE_MRI  = "Built from my MRI folder (real volumes)"


def _csv_for_source(source: str) -> Path:
    return DEMO_DATASET_CSV if source == DATA_SOURCE_DEMO else TUMOR_VOLUMES_CSV


def _model_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


# ---------------------------------------------------------------- helpers

def _pick_folder_dialog() -> str | None:
    """Open a native folder picker. Falls back gracefully if tkinter is absent."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    chosen = filedialog.askdirectory(title="Select folder containing Mets_* patients")
    root.destroy()
    return chosen or None


def _ml_with_uncertainty(model, baseline: float, fu1: float, fu2: float):
    """Mean and ±1σ across all forest trees, denormalised to the patient's units."""
    return predict_future_with_uncertainty(model, baseline, fu1, fu2)


# ---------------------------------------------------------------- styling

# Make the "Reset" button visually destructive (red). The selector targets
# any element with class .reset-btn placed inside an st.button.
st.markdown(
    """
    <style>
    /* Red destructive button via wrapper class */
    div.reset-btn button {
        background-color: #b71c1c !important;
        color: #ffffff !important;
        border: 1px solid #7f0000 !important;
    }
    div.reset-btn button:hover {
        background-color: #d32f2f !important;
        border-color: #b71c1c !important;
    }
    /* Make st.metric in the sidebar a touch tighter */
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- sidebar: load data + model first

st.sidebar.title("TwinTumor")
st.sidebar.caption("Digital twin for brain-metastasis volume trajectories.")

# We need the cohort + model loaded before patient selection can show the
# patient list. Default source = demo cohort. The MRI folder is opt-in below.
model_path = DEFAULT_ML_MODEL_PATH

# One explicit radio drives the active source. Whatever is selected here is
# what every tab reads, and what every train button trains on.
if "active_source" not in st.session_state:
    st.session_state["active_source"] = DATA_SOURCE_DEMO

# Provisional value (the actual radio is rendered below in the sidebar
# header). Falls back to demo if the MRI CSV doesn't exist.
_chosen = st.session_state["active_source"]
if _chosen == DATA_SOURCE_MRI and not TUMOR_VOLUMES_CSV.exists():
    _chosen = DATA_SOURCE_DEMO
active_source = _chosen
csv_path = _csv_for_source(active_source)


# ---------------------------------------------------------------- main page guard

st.title("TwinTumor – Digital Twin Growth Prediction")
st.caption(
    "Forecasting brain-metastasis volume trajectories from the first three "
    "MRI timepoints using an exponential baseline, a Gompertz growth model, "
    "and a persisted Random-Forest digital twin."
)

if not csv_path.exists():
    st.error(
        f"Active dataset missing at `{csv_path}`. "
        "Restore the bundled demo CSV or build a volume CSV from your MRI folder."
    )
    st.stop()

csv_mtime = csv_path.stat().st_mtime
df = _load_dataset(str(csv_path), csv_mtime)
ml_model, training_log, was_trained = _load_or_train_model(
    str(model_path), _model_mtime(model_path)
)


# ============================================================ SIDEBAR — patient

with st.sidebar:
    st.header("🩻 Patient")
    patient_ids = sorted(df["subject"].unique().tolist())
    selected_patient = st.selectbox("Patient", patient_ids, label_visibility="collapsed")
    selected_scenario = st.selectbox(
        "Scenario", ["progression", "remission"], label_visibility="visible",
    )
    enable_pseudo = st.checkbox(
        "RANO pseudoprogression handling",
        value=True,
        help=("When ON, an apparent first-time progression is flagged as "
              "*Provisional Progression*. The next scan either confirms it as "
              "*Progression* or retroactively reclassifies it as "
              "*Pseudoprogression*. Affects the RANO labels in the "
              "trajectory table below."),
    )

# Compute trajectory + RANO labels for the rest of the page
row = df[df["subject"] == selected_patient].iloc[0]
trajectory = get_trajectory(row, selected_scenario)
observed = trajectory[:3]
actual_future = trajectory[3:]

agent = TreatmentAgent(initial_volume=trajectory[0], enable_pseudoprogression=enable_pseudo)
for v in trajectory[1:]:
    agent.evaluate(v)
rano_statuses = agent.final_statuses()


# ============================================================ SIDEBAR — data source

with st.sidebar:
    st.divider()
    st.header("📚 Data source")

    # ── ONE explicit radio drives everything ──
    options = [DATA_SOURCE_DEMO]
    if TUMOR_VOLUMES_CSV.exists():
        options.append(DATA_SOURCE_MRI)
    saved = st.session_state.get("active_source", DATA_SOURCE_DEMO)
    if saved not in options:
        saved = DATA_SOURCE_DEMO
    new_choice = st.radio(
        "Active dataset",
        options,
        index=options.index(saved),
        help=(
            "**This single choice drives the whole demo**: which patient list "
            "you see, which dataset every tab visualises, what data the ML "
            "model trains on, what data it predicts on. Switching here is the "
            "only way to change which dataset is active."
        ),
        key="active_source_radio",
    )
    if new_choice != active_source:
        st.session_state["active_source"] = new_choice
        st.rerun()

    if active_source == DATA_SOURCE_DEMO:
        st.info(
            "Reading **AIMI's `consistent_tumor_analysis.csv`** "
            "(105 patients, bundled with the repository). No MRI files needed. "
            "The persisted ML model is auto-trained on this dataset on the "
            "first launch (200 trees) — see *ML model* below.",
            icon="ℹ️",
        )
    else:
        st.caption(
            "**Your MRI cohort** — built locally from your `Mets_*` MRI folder "
            "via the AIMI Otsu-within-ROI pipeline. Stays on your machine."
        )


# ── MRI folder settings (visible only when the MRI source is active or
#    the CSV has never been built — i.e. the user has something to do here)
if active_source == DATA_SOURCE_MRI or not TUMOR_VOLUMES_CSV.exists():
    with st.sidebar.expander(
        "⚙ MRI folder settings",
        expanded=(active_source == DATA_SOURCE_MRI) or not TUMOR_VOLUMES_CSV.exists(),
    ):
        st.caption(
            "Point the framework at a folder of `Mets_*` patient series. "
            "It computes tumor signal values from each timepoint's "
            "`t1_gd.nii` (Otsu inside the baseline `seg.nii` ROI)."
        )
        if "data_root" not in st.session_state:
            st.session_state["data_root"] = str(RAW_DATA_DIR)
        st.session_state["data_root"] = st.text_input(
            "MRI folder",
            value=st.session_state["data_root"],
            label_visibility="collapsed",
            placeholder="e.g. E:/mri_series/series",
        )
        if st.button("Browse…", key="mri_browse", width="stretch"):
            chosen = _pick_folder_dialog()
            if chosen:
                st.session_state["data_root"] = chosen
                st.rerun()

        try:
            _layout = detect_root_layout(Path(st.session_state["data_root"]))
        except Exception:
            _layout = "unknown"
        if _layout == "aimi":
            st.success("Detected **AIMI layout** — Otsu within baseline ROI.")
        elif _layout == "legacy":
            st.warning(
                "Detected **legacy layout** — duplicate FU `seg.nii` placeholders. "
                "Trajectories will be flat. Use `E:/mri_series/series` instead."
            )
        else:
            st.caption("Layout: not detected yet (check the path).")

        build_clicked = st.button(
            "🔨 Build volume CSV from this folder",
            type="primary",
            width="stretch",
            disabled=st.session_state.get("_building", False),
        )
        if build_clicked and not st.session_state.get("_building", False):
            st.session_state["_building"] = True
            progress_bar = st.progress(0.0, text="Starting…")
            try:
                def _cb(i: int, total: int, pid: str) -> None:
                    progress_bar.progress(i / max(total, 1), text=f"{i}/{total}  {pid}")
                out_path = build_and_save(
                    st.session_state["data_root"], TUMOR_VOLUMES_CSV,
                    progress=_cb, verbose=False,
                )
                progress_bar.empty()
                st.success(
                    f"Built `{out_path.name}` ✓ — pick *Your MRI cohort* "
                    "in the radio above to use it."
                )
                _load_dataset.clear()
                _load_or_train_model.clear()
            except Exception as exc:
                progress_bar.empty()
                st.error(f"Build failed: {exc}")
            finally:
                st.session_state["_building"] = False

# ── Data-quality summary (only when MRI cohort is active)
if active_source == DATA_SOURCE_MRI and TUMOR_VOLUMES_CSV.exists():
    with st.sidebar.expander("🔬 Data-quality summary", expanded=False):
        try:
            summary = flat_trajectory_summary(df)
            total = summary["n_total"]
            prog = summary["flat_counts"]["progression"]
            rem  = summary["flat_counts"]["remission"]
            st.caption(
                "*Flat trajectory* = every timepoint has the same volume (a "
                "data-quality red flag in the source MRI files)."
            )
            col1, col2 = st.columns(2)
            col1.metric("Flat progression", f"{prog} / {total}")
            col2.metric("Flat remission",   f"{rem} / {total}")
            if prog > 0:
                with st.expander(f"Patients with flat progression ({prog})", expanded=False):
                    st.write(", ".join(summary["flat_subjects"]["progression"]))
            if rem > 0:
                with st.expander(f"Patients with flat remission ({rem})", expanded=False):
                    st.write(", ".join(summary["flat_subjects"]["remission"]))
        except Exception as exc:
            st.error(f"Could not compute summary: {exc}")


# ============================================================ SIDEBAR — ML model

with st.sidebar:
    st.divider()
    st.header("🧠 ML model")

    # Status: what data was the model trained on, how big, expected accuracy
    n_rounds = len(training_log)
    n_trees = len(ml_model.estimators_) if hasattr(ml_model, "estimators_") else 0
    last_round_label = training_log[-1].source_label if training_log else "—"
    if n_rounds <= 1:
        st.success(f"✓ Trained on **{last_round_label}** · {n_trees} trees")
    else:
        st.success(
            f"✓ Trained on **{n_rounds} rounds** (last: {last_round_label}) · "
            f"{n_trees} trees"
        )

    # Cohort-level expected accuracy on an unseen patient
    cached_lopo = get_cached_lopo_mae(model_path)
    if cached_lopo is not None:
        st.caption(
            f"📊 Expected MAE on a *new* patient: **~{cached_lopo:.2f}** "
            "(LOPO-CV, cached at training time)"
        )

    # ── Heads-up when the model's training source ≠ what the user is viewing.
    # Predictions still work, but the model wasn't calibrated to this view's
    # distribution.
    last_label = (training_log[-1].source_label if training_log else "").lower()
    trained_on_demo = "demo" in last_label
    trained_on_mri  = "mri" in last_label or "your mri" in last_label
    mismatch = (
        (active_source == DATA_SOURCE_DEMO and trained_on_mri) or
        (active_source == DATA_SOURCE_MRI and trained_on_demo)
    )
    if mismatch:
        st.warning(
            f"You're viewing **{active_source}** but the model was trained on "
            "a *different* dataset. Predictions will work, but they may be "
            "miscalibrated for this distribution. To recalibrate, train fresh "
            "on the currently active dataset (button below).",
            icon="🔁",
        )

    # ── Training controls — only shown when MRI is active. In demo mode the
    #    model is already trained on the demo CSV at startup, so these buttons
    #    are meaningless there.
    if active_source == DATA_SOURCE_MRI:
        if "n_new_trees" not in st.session_state:
            st.session_state["n_new_trees"] = DEFAULT_INCREMENTAL_TREES

        if st.button(
            f"➕ Add this MRI cohort to the model  "
            f"(+{st.session_state['n_new_trees']} trees)",
            type="primary",
            width="stretch",
            help=("Trains additional trees on your MRI cohort and saves them. "
                  "The model's prior knowledge from earlier training rounds "
                  "is preserved."),
        ):
            try:
                with st.spinner("Adding new trees to the model…"):
                    train_incremental(
                        csv_path=csv_path,
                        source_label=f"{active_source} · {csv_path.name}",
                        model_path=model_path,
                        n_new_trees=int(st.session_state["n_new_trees"]),
                    )
                _load_or_train_model.clear()
                st.success("Model updated ✓")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")

        if st.button(
            "🔄 Train fresh on this MRI cohort only",
            width="stretch",
            help=("Wipes the model and trains a new one using ONLY this MRI "
                  "cohort. Recommended after switching to your MRI cohort, "
                  "to recalibrate the model from demo-data ratios to your "
                  "data's ratios."),
        ):
            try:
                with st.spinner("Resetting and training fresh on this MRI cohort…"):
                    train_fresh_from(
                        csv_path=csv_path,
                        source_label=active_source,
                        model_path=model_path,
                    )
                _load_or_train_model.clear()
                st.success("Model retrained on the MRI cohort ✓")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")
    else:
        # Demo mode: nothing to do here — the model is already trained on
        # the demo CSV at first launch.
        st.caption(
            "Switch to *Your MRI cohort* in *Data source* above to train on "
            "your own data. While in demo mode there is nothing to retrain — "
            "the model is auto-fitted to the demo CSV at first launch."
        )

    with st.expander("⚙ Advanced", expanded=False):
        if active_source == DATA_SOURCE_MRI:
            st.session_state["n_new_trees"] = st.slider(
                "Strength of new training (trees added per click)",
                min_value=10, max_value=500,
                value=int(st.session_state.get("n_new_trees", DEFAULT_INCREMENTAL_TREES)),
                step=10,
                help=("Each click of *Add this MRI cohort to the model* trains "
                      "this many new decision trees. Random Forests saturate "
                      "around 100–500 trees; more trees ≠ better. 100 is a "
                      "sensible default."),
            )

        st.markdown("##### Training history")
        if training_log:
            log_df = pd.DataFrame([vars(r) for r in training_log])
            st.dataframe(log_df, hide_index=True, width="stretch")
        else:
            st.caption("(empty)")

        st.markdown("##### Danger zone")
        st.caption(
            "Deletes the saved model. Next page-load auto-retrains a fresh "
            "one from the demo cohort (200 trees)."
        )
        st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
        if st.button("⚠ Reset model to demo-only", width="stretch"):
            if reset_model(model_path):
                _load_or_train_model.clear()
                st.success("Model deleted. Reload the page to retrain from demo.")
                st.rerun()
            else:
                st.info("No saved model to delete.")
        st.markdown('</div>', unsafe_allow_html=True)

# Inform user of cold-start retrain (after sidebar is laid out)
if was_trained:
    st.sidebar.info("First run — trained an initial model from the demo cohort.")

row = df[df["subject"] == selected_patient].iloc[0]
trajectory = get_trajectory(row, selected_scenario)
observed = trajectory[:3]
actual_future = trajectory[3:]

# RANO statuses -------------------------------------------------------------
agent = TreatmentAgent(initial_volume=trajectory[0], enable_pseudoprogression=enable_pseudo)
for v in trajectory[1:]:
    agent.evaluate(v)
rano_statuses = agent.final_statuses()

# Tabs ---------------------------------------------------------------------
tab_single, tab_cohort, tab_heatmap, tab_compare, tab_one_patient = st.tabs([
    "Single-patient twin", "Cohort view", "Heatmaps", "Model comparison",
    "One-patient inference",
])


# ============================================================ TAB 1: single
with tab_single:
    st.subheader("Single-patient digital twin")
    st.write(
        "The observed window (Baseline, FU1, FU2) drives three forecasts: a "
        "log-linear **exponential** fit, a **Gompertz** saturating-growth fit, "
        "and a **Random Forest** trained on the cohort. The shaded band on the "
        "ML line is the per-tree ensemble ±1σ."
    )

    # Data-quality banner — flat trajectory means duplicate segmentations
    # in the source MRI cohort. This is a source-data issue, not a code bug.
    if active_source == DATA_SOURCE_MRI and is_trajectory_flat(trajectory):
        other_scenario = "remission" if selected_scenario == "progression" else "progression"
        other_traj = get_trajectory(row, other_scenario)
        other_is_flat = is_trajectory_flat(other_traj)
        other_hint = (
            f"The **{other_scenario}** branch for this patient also looks flat — "
            "the source data appears to lack real per-timepoint segmentations "
            "for both branches."
            if other_is_flat else
            f"Try the **{other_scenario}** branch for this patient — those "
            "segmentations vary across timepoints and produce a meaningful trajectory."
        )
        st.warning(
            f"**Data quality notice — `{selected_patient}` / {selected_scenario}**\n\n"
            f"All six timepoints compute to the same volume (~{trajectory[0]:.0f} mm³). "
            "This is **not a forecasting bug**; it means the `seg.nii` files at "
            f"`{selected_patient}/{selected_scenario}/FU*/seg.nii` are byte-identical "
            "copies of the baseline segmentation in the source MRI cohort. "
            "The framework is correctly reporting the volumes those files contain.\n\n"
            f"{other_hint} "
            "Switch the **Data source** to *demo cohort* in the sidebar to see "
            "predictions on synthetic, non-flat trajectories.",
            icon="⚠️",
        )

    forecasts = forecast_all_models(observed)
    ml_mean, ml_std = _ml_with_uncertainty(ml_model, *observed)
    ml_full = np.concatenate([observed, ml_mean])
    ml_err = mae(actual_future, ml_mean)
    times_all = np.arange(6)

    col_plot, col_info = st.columns([2, 1])

    with col_plot:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(times_all, trajectory, marker="o", linewidth=2.5, color="black", label="Actual", zorder=5)

        if forecasts["Exponential"]["ok"]:
            full = forecasts["Exponential"]["full"]
            exp_err = mae(actual_future, forecasts["Exponential"]["future"])
            ax.plot(times_all, full, marker="s", linestyle=":", linewidth=2,
                    color="tab:orange", label=f"Exponential (MAE = {exp_err:.1f})")
        else:
            exp_err = None

        if forecasts["Gompertz"]["ok"]:
            full = forecasts["Gompertz"]["full"]
            gom_err = mae(actual_future, forecasts["Gompertz"]["future"])
            ax.plot(times_all, full, marker="^", linestyle="-.", linewidth=2,
                    color="tab:green", label=f"Gompertz (MAE = {gom_err:.1f})")
        else:
            gom_err = None

        ax.plot(times_all, ml_full, marker="d", linestyle="--", linewidth=2,
                color="tab:blue", label=f"ML (MAE = {ml_err:.1f})")
        ax.fill_between(times_all[3:], ml_mean - ml_std, ml_mean + ml_std,
                        color="tab:blue", alpha=0.18, label="ML ensemble ±1σ")

        ax.axvline(x=2, linestyle="--", alpha=0.4, color="gray")
        ax.set_xticks(times_all)
        ax.set_xticklabels(TIME_LABELS)
        ax.set_xlabel("Timepoint"); ax.set_ylabel("Tumor volume (mm³)")
        ax.set_title(f"{selected_patient} – {selected_scenario}")
        ax.grid(True, alpha=0.3); ax.legend(loc="best")
        st.pyplot(fig)

    with col_info:
        st.markdown("### Forecast MAE (mm³)")
        rows = []
        if forecasts["Exponential"]["ok"]:
            rows.append(("Exponential", exp_err))
        if forecasts["Gompertz"]["ok"]:
            rows.append(("Gompertz", gom_err))
        rows.append(("ML (Random Forest)", ml_err))
        err_df = pd.DataFrame(rows, columns=["Model", "MAE"])
        err_df["MAE"] = err_df["MAE"].round(2)
        st.dataframe(err_df, hide_index=True, width="stretch")
        if rows:
            best = min(rows, key=lambda r: r[1])
            st.success(f"Best on this case: **{best[0]}** (MAE = {best[1]:.1f} mm³)")

        # Honest framing: this MAE is in-sample if the patient was in the
        # ML training set (which it always is for the active CSV today).
        cached_lopo = get_cached_lopo_mae(model_path)
        st.caption(
            "ⓘ The ML MAE above is **in-sample** — `{p}` was in the model's "
            "training set, so the model already 'knows' this patient. The "
            "honest benchmark (model evaluated on a patient it has *never* seen) "
            "is the cohort LOPO-CV: {lopo}.".format(
                p=selected_patient,
                lopo=f"**~{cached_lopo:.2f} mm³**" if cached_lopo is not None
                     else "*not cached* (retrain to populate)",
            )
        )
        if forecasts["Gompertz"]["ok"]:
            p = forecasts["Gompertz"]["params"]
            st.markdown("### Gompertz parameters")
            st.write(f"V∞ = {p['V_inf']:.1f}  \nb = {p['b']:.3f}  \nc = {p['c']:.3f}")
        elif not forecasts["Gompertz"]["ok"]:
            st.warning(f"Gompertz fit failed: {forecasts['Gompertz']['error']}")

    # Trajectory + RANO labels
    st.markdown("#### Trajectory values and RANO classification")
    table = {
        "timepoint": TIME_LABELS,
        "actual_mm3": np.round(trajectory, 2),
        "RANO": rano_statuses,
    }
    if forecasts["Exponential"]["ok"]:
        table["exponential"] = np.round(forecasts["Exponential"]["full"], 2).tolist()
    if forecasts["Gompertz"]["ok"]:
        table["gompertz"] = np.round(forecasts["Gompertz"]["full"], 2).tolist()
    table["ml"] = np.round(ml_full, 2).tolist()
    table["ml ±1σ"] = [None, None, None] + [f"±{s:.1f}" for s in ml_std]
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")


# ============================================================ TAB 2: cohort
with tab_cohort:
    st.subheader("Cohort overlay")
    st.write(
        "Where does this patient sit in the cohort? Thin grey lines are "
        "individual patients, the blue band is the cohort mean ±1σ, the bold "
        "black line is the selected patient."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        n_overlay = st.slider("Random patients to overlay", 5, len(df), min(40, len(df)))
    with col_b:
        same_scenario = st.checkbox("Use selected scenario", value=True)

    traj_by_scenario: dict[str, np.ndarray] = {
        scen: np.array([get_trajectory(r, scen) for _, r in df.iterrows()])
        for scen in ("progression", "remission")
    }
    cohort_ids = df["subject"].tolist()
    rng = np.random.default_rng(42)
    overlay_ids = rng.choice(
        cohort_ids, size=min(n_overlay, len(cohort_ids)), replace=False
    ).tolist()
    if selected_patient not in overlay_ids:
        overlay_ids[0] = selected_patient

    def _overlay(ax, scen):
        traj_matrix = traj_by_scenario[scen]
        for pid in overlay_ids:
            idx = cohort_ids.index(pid)
            ax.plot(range(6), traj_matrix[idx], color="gray", alpha=0.25, linewidth=1)
        m, s = traj_matrix.mean(axis=0), traj_matrix.std(axis=0)
        ax.fill_between(range(6), m - s, m + s, color="steelblue", alpha=0.2, label="Cohort ±1σ")
        ax.plot(range(6), m, color="steelblue", linewidth=2, label="Cohort mean")
        ax.plot(range(6), get_trajectory(row, scen), color="black", linewidth=3,
                marker="o", label=f"Selected: {selected_patient}")
        ax.axvline(x=2, linestyle="--", alpha=0.4, color="gray")
        ax.set_xticks(range(6)); ax.set_xticklabels(TIME_LABELS)
        ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)

    if same_scenario:
        fig, ax = plt.subplots(figsize=(11, 6))
        _overlay(ax, selected_scenario)
        ax.set_xlabel("Timepoint"); ax.set_ylabel("Tumor volume (mm³)")
        ax.set_title(f"Cohort overlay — {selected_scenario}")
        st.pyplot(fig)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, scen in zip(axes, ("progression", "remission")):
            _overlay(ax, scen); ax.set_title(scen); ax.set_xlabel("Timepoint")
        axes[0].set_ylabel("Tumor volume (mm³)")
        st.pyplot(fig)

    # Nearest neighbours
    st.subheader("Nearest neighbours (case-based twin)")
    k = st.slider("Number of neighbours", 3, 15, 5)
    obs = np.array(observed)
    distances = []
    for _, r in df.iterrows():
        other_obs = np.array(get_trajectory(r, selected_scenario)[:3])
        distances.append((r["subject"], float(np.linalg.norm(other_obs - obs))))
    distances.sort(key=lambda x: x[1])
    neighbours = [d for d in distances if d[0] != selected_patient][:k]

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.tab10
    for i, (pid, dist) in enumerate(neighbours):
        r_n = df[df["subject"] == pid].iloc[0]
        ax2.plot(range(6), get_trajectory(r_n, selected_scenario),
                 alpha=0.85, linewidth=1.8, color=cmap(i % 10), marker="o",
                 label=f"{pid} (d={dist:.1f})")
    ax2.plot(range(6), trajectory, color="black", linewidth=3, marker="s",
             label=f"Selected: {selected_patient}")
    ax2.axvline(x=2, linestyle="--", alpha=0.4, color="gray")
    ax2.set_xticks(range(6)); ax2.set_xticklabels(TIME_LABELS)
    ax2.set_xlabel("Timepoint"); ax2.set_ylabel("Tumor volume (mm³)")
    ax2.set_title(f"{k} most similar patients — {selected_scenario}")
    ax2.grid(True, alpha=0.3); ax2.legend(loc="best", fontsize=8)
    st.pyplot(fig2)


# ============================================================ TAB 3: heatmaps
with tab_heatmap:
    heat_mode = st.radio(
        "View",
        ["Tumor slice + growth forecast", "Cohort trajectory matrix"],
        horizontal=True, key="heat_mode",
    )

    # ─────────────── 2D growth-forecast view ───────────────
    if heat_mode == "Tumor slice + growth forecast":
        st.subheader("Tumor slice + predicted growth")
        st.caption(
            "Red fill = current tumor (baseline). Yellow fill = pixels predicted "
            "to *become* tumor by FU5 (growth). Cyan fill = pixels predicted to "
            "*stop* being tumor by FU5 (shrinkage). Dashed contours mark each "
            "intermediate timepoint (FU3, FU4, FU5). The spatial extent assumes "
            "*isotropic uniform growth* — the 2D area scales as "
            "(V_future / V_current)^(2/3). The model predicts volumes, not future "
            "masks, so this view illustrates the volume change in image space."
        )

        # Real MRI is only used when the user explicitly chose the MRI source.
        # Otherwise we always show the synthetic illustration so demo mode is
        # genuinely demo mode regardless of whatever data_root might be cached.
        use_real_mri = (active_source == DATA_SOURCE_MRI)

        seg_path: Optional[Path] = None
        mri_path: Optional[Path] = None
        if use_real_mri:
            mri_root = Path(st.session_state.get("data_root", str(RAW_DATA_DIR)))
            if mri_root.exists():
                seg_path = mri_root / selected_patient / "baseline" / "seg.nii"
                mri_path = mri_root / selected_patient / "baseline" / "t1_gd.nii"
        have_real_mri = use_real_mri and (seg_path is not None and seg_path.exists())

        # Forecast volumes from the persisted RF
        from src.models.ml_predictor import predict_future
        ml_pred = predict_future(ml_model, baseline=trajectory[0], fu1=trajectory[1], fu2=trajectory[2])
        predicted_volumes = [ml_pred["FU3"], ml_pred["FU4"], ml_pred["FU5"]]
        future_labels = ["FU3", "FU4", "FU5"]

        # Layout controls
        col_l, col_r = st.columns(2)
        with col_l:
            view_mode = st.radio(
                "Layout",
                ["Single panel (overlay)",
                 "Side-by-side timepoints",
                 "Predicted vs actual (real Otsu masks)"],
                horizontal=False,
                key="growth_view_mode",
                help=(
                    "**Single panel** — predicted overlay on one axial slice.\n\n"
                    "**Side-by-side** — one panel per timepoint, all predicted.\n\n"
                    "**Predicted vs actual** — only available with the AIMI MRI "
                    "layout. Loads each FU's `t1_gd.nii`, applies Otsu within "
                    "the baseline ROI to get the *real* observed signal mask, "
                    "and shows it next to the framework's predicted overlay."
                ),
            )
        with col_r:
            crop = st.checkbox("Crop to tumor region", value=True, key="growth_crop")

        if have_real_mri:
            st.caption(f"📁 Reading real MRI: `{seg_path}`")
            try:
                seg, voxel_dims = load_mask(seg_path)
                mri_data = None
                if mri_path is not None and mri_path.exists():
                    mri_data, _ = load_mri_volume(mri_path)
                slice_idx, current_2d, layers = predicted_growth_layers_2d(
                    seg_3d=seg, voxel_dims=voxel_dims,
                    current_volume_mm3=trajectory[0],
                    predicted_volumes_mm3=predicted_volumes,
                    timepoint_labels=future_labels,
                )
                bg = (mri_data[:, :, slice_idx]
                      if mri_data is not None and 0 <= slice_idx < mri_data.shape[2]
                      else np.zeros_like(current_2d, dtype=float))

                if view_mode == "Single panel (overlay)":
                    fig, ax = plt.subplots(figsize=(8, 8))
                    render_growth_view(ax, bg, current_2d, layers,
                                       current_volume_mm3=trajectory[0],
                                       crop_to_tumor=crop)
                    fig.suptitle(
                        f"{selected_patient} · slice {slice_idx} · {selected_scenario}",
                        fontsize=12,
                    )
                    st.pyplot(fig)
                elif view_mode == "Side-by-side timepoints":
                    fig = plt.figure(figsize=(14, 4))
                    render_side_by_side(fig, bg, current_2d, layers,
                                        current_volume_mm3=trajectory[0],
                                        crop_to_tumor=crop)
                    fig.suptitle(
                        f"{selected_patient} · slice {slice_idx} · {selected_scenario}",
                        fontsize=12,
                    )
                    st.pyplot(fig)
                else:
                    # Predicted vs actual — requires AIMI layout (FU t1_gd files).
                    layout = detect_root_layout(mri_root) if mri_root.exists() else "unknown"
                    if layout != "aimi":
                        st.warning(
                            "This view needs the AIMI layout (per-FU `t1_gd.nii`). "
                            f"Detected layout under `{mri_root}`: **{layout}**. "
                            "Point the *Use my own MRI folder* path at "
                            "`E:/mri_series/series` and rebuild."
                        )
                    else:
                        try:
                            with st.spinner("Computing real Otsu masks at each FU…"):
                                roi3d, base_t1, vox, fu_masks = compute_fu_signal_masks(
                                    patient_dir=mri_root / selected_patient,
                                    scenario=selected_scenario,
                                    timepoints=future_labels,
                                )
                            # Use the slice with the largest baseline ROI cross-section
                            from src.viz.growth_heatmap import find_largest_tumor_slice
                            try:
                                z = find_largest_tumor_slice(roi3d.astype(np.uint8))
                            except Exception:
                                z = roi3d.shape[2] // 2

                            roi_2d = roi3d[:, :, z]
                            base_2d = base_t1[:, :, z]

                            n = 1 + len(future_labels)
                            fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 7.5))
                            # Crop helper
                            from src.viz.growth_heatmap import _bbox_with_margin
                            actual_layers_2d = [fu_masks[tp].signal_mask_3d[:, :, z]
                                                if tp in fu_masks else np.zeros_like(roi_2d)
                                                for tp in future_labels]
                            predicted_layers_2d = [layers[i].mask_2d for i in range(len(layers))]
                            bbox = _bbox_with_margin(
                                [roi_2d, current_2d, *actual_layers_2d, *predicted_layers_2d]
                            ) if crop else None
                            if bbox is not None:
                                r0, r1, c0, c1 = bbox
                                base_2d = base_2d[r0:r1+1, c0:c1+1]
                                roi_2d = roi_2d[r0:r1+1, c0:c1+1]
                                current_2d_c = current_2d[r0:r1+1, c0:c1+1]
                                actual_layers_2d = [m[r0:r1+1, c0:c1+1] for m in actual_layers_2d]
                                predicted_layers_2d = [m[r0:r1+1, c0:c1+1] for m in predicted_layers_2d]
                            else:
                                current_2d_c = current_2d

                            # Window the t1_gd image
                            p1, p99 = np.percentile(base_2d, [1, 99]) if base_2d.size else (0, 1)
                            base_disp = np.clip(base_2d, p1, p99) if p99 > p1 else base_2d

                            # ─── Top row: PREDICTED ───
                            ax = axes[0, 0]
                            ax.imshow(base_disp, cmap="gray")
                            if current_2d_c.any():
                                fill = np.zeros((*current_2d_c.shape, 4))
                                fill[current_2d_c] = [1.0, 0.20, 0.20, 0.55]
                                ax.imshow(fill)
                                ax.contour(current_2d_c, levels=[0.5], colors="red", linewidths=2)
                            ax.set_title(f"Baseline ROI\n{trajectory[0]:.1f}", fontsize=10)
                            ax.set_xticks([]); ax.set_yticks([])
                            for i, (tp, pmask) in enumerate(zip(future_labels, predicted_layers_2d), start=1):
                                ax = axes[0, i]
                                ax.imshow(base_disp, cmap="gray")
                                if current_2d_c.any():
                                    ax.contour(current_2d_c, levels=[0.5], colors="red",
                                               linewidths=1.0, alpha=0.7)
                                if pmask.any():
                                    fill = np.zeros((*pmask.shape, 4))
                                    fill[pmask] = [1.0, 0.85, 0.10, 0.5]
                                    ax.imshow(fill)
                                    ax.contour(pmask, levels=[0.5], colors="orange", linewidths=1.5)
                                ax.set_title(f"{tp} predicted\n{layers[i-1].volume_mm3:.1f}", fontsize=10)
                                ax.set_xticks([]); ax.set_yticks([])

                            # ─── Bottom row: ACTUAL OTSU MASKS ───
                            ax = axes[1, 0]
                            ax.imshow(base_disp, cmap="gray")
                            if roi_2d.any():
                                ax.contour(roi_2d, levels=[0.5], colors="red", linewidths=2)
                            ax.set_title("Baseline ROI", fontsize=10)
                            ax.set_xticks([]); ax.set_yticks([])
                            for i, (tp, amask) in enumerate(zip(future_labels, actual_layers_2d), start=1):
                                ax = axes[1, i]
                                # Use this FU's t1_gd as background for the actual panel
                                if tp in fu_masks:
                                    fu_bg_3d = fu_masks[tp].t1_gd_3d
                                    fu_bg = fu_bg_3d[:, :, z]
                                    if bbox is not None:
                                        fu_bg = fu_bg[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
                                    p1f, p99f = np.percentile(fu_bg, [1, 99]) if fu_bg.size else (0, 1)
                                    ax.imshow(np.clip(fu_bg, p1f, p99f) if p99f > p1f else fu_bg, cmap="gray")
                                    pct = fu_masks[tp].percent
                                    title = f"{tp} actual (Otsu)\n{pct:.1f}%"
                                else:
                                    ax.imshow(base_disp, cmap="gray")
                                    title = f"{tp} (no t1_gd)"
                                if roi_2d.any():
                                    ax.contour(roi_2d, levels=[0.5], colors="red",
                                               linewidths=1.0, alpha=0.7)
                                if amask.any():
                                    fill = np.zeros((*amask.shape, 4))
                                    fill[amask] = [0.20, 0.85, 0.20, 0.55]
                                    ax.imshow(fill)
                                    ax.contour(amask, levels=[0.5], colors="lime", linewidths=1.5)
                                ax.set_title(title, fontsize=10)
                                ax.set_xticks([]); ax.set_yticks([])

                            fig.suptitle(
                                f"{selected_patient} · slice {z} · {selected_scenario} — "
                                f"top: predicted (isotropic dilation),  bottom: actual (Otsu within ROI)",
                                fontsize=11,
                            )
                            fig.tight_layout()
                            st.pyplot(fig)
                        except Exception as exc:
                            st.error(f"Could not render predicted-vs-actual view: {exc}")
            except Exception as exc:
                st.error(f"Could not render growth heatmap: {exc}")
        else:
            # Synthetic illustration: forced for demo source, or when MRI is missing
            if active_source == DATA_SOURCE_DEMO:
                st.info(
                    "Demo source has no MRI files on disk — showing a "
                    "**synthetic illustration** of the same growth principle. "
                    "Switch to *Use my own MRI folder* in the sidebar to "
                    "render the real MRI slice for a patient with `seg.nii`."
                )
            else:
                st.warning(
                    f"No `seg.nii` found at `{seg_path}` — showing a synthetic "
                    "illustration. Build a volume CSV from the right MRI folder first."
                )

            disk = synthetic_disk_mask(side=240, radius=22)
            current_area = float(disk.sum())
            if trajectory[0] > 0:
                ratios = [v / trajectory[0] for v in predicted_volumes]
            else:
                ratios = [1.0, 1.0, 1.0]
            layers = []
            for label, ratio in zip(future_labels, ratios):
                target = int(round(current_area * (max(ratio, 0) ** (2 / 3))))
                layers.append(GrowthLayer(
                    label=label, target_area_px=target,
                    mask_2d=grow_mask_2d_to_area(disk, target),
                    volume_mm3=trajectory[0] * ratio,
                ))

            if view_mode == "Single panel (overlay)":
                fig, ax = plt.subplots(figsize=(7, 7))
                render_growth_view(ax, np.zeros_like(disk, dtype=float), disk, layers,
                                   current_volume_mm3=trajectory[0],
                                   crop_to_tumor=crop)
                fig.suptitle(
                    f"Synthetic illustration · {selected_patient} ({selected_scenario})",
                    fontsize=12,
                )
            else:
                fig = plt.figure(figsize=(14, 4))
                render_side_by_side(fig, np.zeros_like(disk, dtype=float), disk, layers,
                                    current_volume_mm3=trajectory[0],
                                    crop_to_tumor=crop)
                fig.suptitle(
                    f"Synthetic illustration · {selected_patient} ({selected_scenario})",
                    fontsize=12,
                )
            st.pyplot(fig)

        st.markdown("---")
        st.caption(
            "*Why isotropic*: the framework's ML model predicts a future volume, "
            "not a future segmentation. A learned spatial growth model "
            "(reaction–diffusion PDE or generative segmentation predictor) "
            "would be a master-thesis-scope extension — see *Future work* in the report."
        )

    # ─────────────── original cohort trajectory matrix view ───────────────
    else:
        st.subheader("Cohort trajectory heatmap")
        sort_mode = st.radio(
            "Sort patients by",
            ["Baseline", "Total change (FU5 − baseline)", "Subject ID"],
            horizontal=True, key="heatmap_sort",
        )

        traj_matrix = np.array([get_trajectory(r, selected_scenario) for _, r in df.iterrows()])
        ids = df["subject"].values
        if sort_mode == "Baseline":
            order = np.argsort(traj_matrix[:, 0])
        elif sort_mode == "Total change (FU5 − baseline)":
            order = np.argsort(traj_matrix[:, -1] - traj_matrix[:, 0])
        else:
            order = np.argsort(ids)

        sorted_matrix = traj_matrix[order]; sorted_ids = ids[order]
        fig, ax = plt.subplots(figsize=(9, max(6, len(sorted_ids) * 0.12)))
        im = ax.imshow(sorted_matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(6)); ax.set_xticklabels(TIME_LABELS)
        ax.set_yticks(range(len(sorted_ids))); ax.set_yticklabels(sorted_ids, fontsize=6)
        sel_pos = np.where(sorted_ids == selected_patient)[0]
        if len(sel_pos): ax.axhline(sel_pos[0], color="red", linewidth=1.2, alpha=0.85)
        ax.set_xlabel("Timepoint"); ax.set_title(f"Cohort heatmap — {selected_scenario}")
        fig.colorbar(im, ax=ax, label="Tumor volume (mm³)")
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Prediction-error heatmap")
        model_choice = st.selectbox("Model", ["Exponential", "Gompertz", "ML (Random Forest)"], key="err_heatmap")

        with st.spinner("Building cohort predictions…"):
            pred_table = build_cohort_predictions(df, ml_model)

        scen_rows = pred_table[pred_table["scenario"] == selected_scenario].copy()
        actual = scen_rows[["actual_FU3", "actual_FU4", "actual_FU5"]].values
        if model_choice == "Exponential":
            pred = scen_rows[["exp_FU3", "exp_FU4", "exp_FU5"]].values
        elif model_choice == "Gompertz":
            pred = scen_rows[["gom_FU3", "gom_FU4", "gom_FU5"]].values
        else:
            pred = scen_rows[["ml_FU3", "ml_FU4", "ml_FU5"]].values

        err = np.abs(actual - pred)
        err_order = np.argsort(np.nanmean(err, axis=1))
        err_sorted = err[err_order]
        ids_sorted = scen_rows["subject"].values[err_order]
        fig2, ax2 = plt.subplots(figsize=(7, max(6, len(ids_sorted) * 0.12)))
        im2 = ax2.imshow(err_sorted, aspect="auto", cmap="magma")
        ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels(["FU3", "FU4", "FU5"])
        ax2.set_yticks(range(len(ids_sorted))); ax2.set_yticklabels(ids_sorted, fontsize=6)
        sel_pos = np.where(ids_sorted == selected_patient)[0]
        if len(sel_pos): ax2.axhline(sel_pos[0], color="cyan", linewidth=1.2, alpha=0.85)
        ax2.set_xlabel("Future timepoint"); ax2.set_title(f"{model_choice}: |actual − predicted| ({selected_scenario})")
        fig2.colorbar(im2, ax=ax2, label="Absolute error (mm³)")
        st.pyplot(fig2)


# ============================================================ TAB 4: compare
with tab_compare:
    st.subheader("Model comparison across the full cohort")
    with st.spinner("Building cohort predictions…"):
        pred_table = build_cohort_predictions(df, ml_model)

    actual_cols = ["actual_FU3", "actual_FU4", "actual_FU5"]
    model_cols = {
        "Exponential": ["exp_FU3", "exp_FU4", "exp_FU5"],
        "Gompertz":    ["gom_FU3", "gom_FU4", "gom_FU5"],
        "ML (RF)":     ["ml_FU3",  "ml_FU4",  "ml_FU5"],
    }

    mae_records = []
    for name, cols in model_cols.items():
        err = np.abs(pred_table[actual_cols].values - pred_table[cols].values)
        per_case = np.nanmean(err, axis=1)
        for m_val, scen in zip(per_case, pred_table["scenario"]):
            if not np.isnan(m_val):
                mae_records.append({"Model": name, "Scenario": scen, "MAE": m_val})
    mae_df = pd.DataFrame(mae_records)

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [mae_df[mae_df["Model"] == m]["MAE"].values for m in model_cols.keys()]
        bp = ax.boxplot(data, labels=list(model_cols.keys()), patch_artist=True)
        for patch, c in zip(bp["boxes"], ["tab:orange", "tab:green", "tab:blue"]):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_ylabel("MAE (FU3–FU5, mm³)"); ax.set_title("Per-patient forecast MAE distribution")
        ax.grid(True, axis="y", alpha=0.3); st.pyplot(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(8, 5))
        summary = mae_df.groupby(["Model", "Scenario"])["MAE"].mean().unstack()
        summary = summary.reindex(list(model_cols.keys()))
        summary.plot(kind="bar", ax=ax, color=["tab:cyan", "tab:red"])
        ax.set_ylabel("Mean MAE (mm³)"); ax.set_title("Mean MAE by scenario")
        ax.grid(True, axis="y", alpha=0.3); plt.xticks(rotation=0)
        st.pyplot(fig)

    st.markdown("### Summary statistics")
    st.dataframe(
        mae_df.groupby(["Model", "Scenario"])["MAE"].agg(["mean", "median", "std", "count"]).round(2)
    )

    st.markdown("### Per-timepoint MAE (both scenarios combined)")
    per_tp = []
    for name, cols in model_cols.items():
        for i, tp in enumerate(["FU3", "FU4", "FU5"]):
            err_col = np.abs(pred_table[actual_cols[i]].values - pred_table[cols[i]].values)
            per_tp.append({"Model": name, "Timepoint": tp, "MAE": float(np.nanmean(err_col))})
    pivot = pd.DataFrame(per_tp).pivot(index="Model", columns="Timepoint", values="MAE").round(2)
    st.dataframe(pivot.reindex(list(model_cols.keys())), width="stretch")

    st.markdown("### Export")
    st.download_button(
        "Download cohort prediction table (CSV)",
        data=pred_table.to_csv(index=False).encode("utf-8"),
        file_name="cohort_predictions.csv",
        mime="text/csv",
    )

    # ---------------------------------------------------------------- extras
    st.markdown("---")
    st.subheader("ML model showdown (held-out test split)")
    st.write(
        "The numbers above use the persisted continual-training Random Forest "
        "evaluated on the same patients it was trained on (in-sample). "
        "Below, four ML models — Linear Regression, Random Forest, Gradient "
        "Boosting and a small MLP neural network — are trained on a randomly "
        "drawn 80% of the cohort and scored on the remaining 20%, alongside "
        "per-patient Exponential and Gompertz fits. These MAEs are honest "
        "generalisation error."
    )

    col_x, col_y = st.columns(2)
    with col_x:
        test_size = st.slider("Test split size", 0.1, 0.5, 0.2, 0.05)
    with col_y:
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    if st.button("Run ML model showdown"):
        with st.spinner("Training Linear / RF / GB / MLP and evaluating per-patient growth fits…"):
            summary, per_subject = eval_extra_models(df, test_size=test_size, random_state=int(seed))

        if summary.empty:
            st.warning("No usable rows in the dataset.")
        else:
            mean_only = summary[summary["Timepoint"] == "FU3-FU5 mean"].copy()
            mean_only = mean_only.sort_values("MAE")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(mean_only["Model"], mean_only["MAE"], color="tab:blue", alpha=0.7)
            ax.set_xlabel("MAE on held-out test split (original units)")
            ax.set_title(f"Held-out MAE (test_size={test_size}, seed={int(seed)})")
            ax.grid(True, axis="x", alpha=0.3)
            st.pyplot(fig)

            st.markdown("##### Per-timepoint MAE")
            pivot = summary.pivot(index="Model", columns="Timepoint", values="MAE").round(2)
            col_order = ["FU3", "FU4", "FU5", "FU3-FU5 mean"]
            st.dataframe(pivot[col_order], width="stretch")

            st.download_button(
                "Download per-subject test-split predictions (CSV)",
                data=per_subject.to_csv(index=False).encode("utf-8"),
                file_name="multi_model_test_predictions.csv",
                mime="text/csv",
            )

    # ---------------------------------------------------------------- LOPO
    st.markdown("---")
    st.subheader("Leave-one-patient-out cross-validation")
    st.write(
        "For every patient *p* in the cohort, train every ML model on the "
        "**other** patients and forecast for *p*. Repeat across all patients "
        "and average the errors. Each patient contributes both their "
        "progression and remission rows together, so train and test never "
        "share a subject — eliminating baseline-volume leakage between scenarios. "
        "These MAEs describe how the framework would perform on a "
        "*new* patient it has never seen, which is the question the "
        "digital-twin idea is supposed to answer."
    )

    if st.button("Run LOPO-CV (slow — trains one fold per patient)"):
        progress_bar = st.progress(0.0, text="Starting…")
        def _cb(i: int, total: int, pid: str) -> None:
            progress_bar.progress(i / max(total, 1), text=f"Fold {i}/{total}  · holdout: {pid}")

        with st.spinner("Running leave-one-patient-out CV…"):
            lopo_summary, lopo_per_patient = leave_one_patient_out(df, progress=_cb)
        progress_bar.empty()

        if lopo_summary.empty:
            st.warning("Could not run LOPO-CV (no usable rows).")
        else:
            mean_only = lopo_summary[lopo_summary["Timepoint"] == "FU3-FU5 mean"].copy()
            mean_only = mean_only.sort_values("MAE")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(mean_only["Model"], mean_only["MAE"], color="tab:purple", alpha=0.7)
            ax.set_xlabel("LOPO-CV MAE on held-out patient (original units)")
            ax.set_title(f"Leave-one-patient-out MAE (n={len(df)} folds)")
            ax.grid(True, axis="x", alpha=0.3)
            st.pyplot(fig)

            st.markdown("##### Per-timepoint LOPO MAE")
            pivot = lopo_summary.pivot(index="Model", columns="Timepoint", values="MAE").round(2)
            st.dataframe(pivot[["FU3", "FU4", "FU5", "FU3-FU5 mean"]], width="stretch")

            st.download_button(
                "Download per-patient LOPO predictions (CSV)",
                data=lopo_per_patient.to_csv(index=False).encode("utf-8"),
                file_name="lopo_predictions.csv",
                mime="text/csv",
            )


# ============================================================ TAB 5: one patient
with tab_one_patient:
    st.subheader("One-patient inference")
    st.write(
        "Hand the framework a single patient folder and forecast their future "
        "timepoints against the persisted shared model. Bypasses the cohort-CSV "
        "build step entirely. Useful for clone-and-run users who only have a "
        "few patients of their own."
    )

    if "single_patient_dir" not in st.session_state:
        st.session_state["single_patient_dir"] = ""

    col_path, col_browse = st.columns([4, 1])
    with col_path:
        patient_dir_input = st.text_input(
            "Patient folder (containing baseline/, progression/ or remission/)",
            value=st.session_state["single_patient_dir"],
            placeholder="e.g. E:/series/Mets_005",
        )
        st.session_state["single_patient_dir"] = patient_dir_input
    with col_browse:
        st.markdown("&nbsp;")
        if st.button("Browse…", key="single_browse"):
            chosen = _pick_folder_dialog()
            if chosen:
                st.session_state["single_patient_dir"] = chosen
                st.rerun()

    if not patient_dir_input:
        st.info("Pick a patient folder to begin.")
    else:
        pdir = Path(patient_dir_input)
        if not pdir.exists():
            st.error(f"Folder does not exist: {pdir}")
        elif not (pdir / "baseline" / "seg.nii").exists():
            st.error(f"Missing {pdir / 'baseline' / 'seg.nii'}")
        else:
            scenarios_available = discover_scenarios(pdir)
            if not scenarios_available:
                st.warning("No `progression/` or `remission/` subfolders found.")
            else:
                chosen_scenario = st.radio(
                    "Scenario branch", scenarios_available, horizontal=True,
                )
                enable_pseudo_one = st.checkbox(
                    "RANO pseudoprogression handling",
                    value=True, key="one_patient_pseudo",
                )

                scan = None
                with st.spinner(f"Reading {chosen_scenario}/seg.nii files…"):
                    try:
                        scan = read_patient_volumes(pdir, chosen_scenario)
                    except Exception as exc:
                        st.error(f"Could not read patient: {exc}")

                result = None
                if scan is None:
                    pass  # error already shown above
                elif len(scan.trajectory) < 3:
                    st.warning(
                        f"Only {len(scan.trajectory)} timepoint(s) available; need "
                        "baseline + FU1 + FU2 to forecast."
                    )
                else:
                    try:
                        result = forecast_for_patient(scan, ml_model)
                    except Exception as exc:
                        st.error(f"Forecast failed: {exc}")

                if scan is not None and result is not None:
                    observed = result["observed"]
                    actual_future = result["actual_future"]
                    forecasts = result["forecasts"]
                    ml_mean, ml_std = result["ml_mean"], result["ml_std"]

                    # RANO labels for whatever timepoints exist
                    agent = TreatmentAgent(
                        initial_volume=scan.baseline,
                        enable_pseudoprogression=enable_pseudo_one,
                    )
                    for v in scan.trajectory[1:]:
                        agent.evaluate(v)
                    rano_labels = agent.final_statuses()

                    # Plot
                    n_obs = len(scan.trajectory)
                    times_all = np.arange(6)
                    fig, ax = plt.subplots(figsize=(11, 6))

                    ax.plot(range(n_obs), scan.trajectory, marker="o", linewidth=2.5,
                            color="black", label=f"Observed ({n_obs} pts)", zorder=5)

                    if forecasts["Exponential"]["ok"]:
                        ax.plot(times_all, forecasts["Exponential"]["full"], marker="s",
                                linestyle=":", color="tab:orange", label="Exponential")
                    if forecasts["Gompertz"]["ok"]:
                        ax.plot(times_all, forecasts["Gompertz"]["full"], marker="^",
                                linestyle="-.", color="tab:green", label="Gompertz")
                    ml_full = np.concatenate([observed, ml_mean])
                    ax.plot(times_all, ml_full, marker="d", linestyle="--",
                            color="tab:blue", label="ML (continual RF)")
                    ax.fill_between(times_all[3:], ml_mean - ml_std, ml_mean + ml_std,
                                    color="tab:blue", alpha=0.18, label="ML ±1σ")

                    ax.axvline(x=2, linestyle="--", alpha=0.4, color="gray")
                    ax.set_xticks(times_all); ax.set_xticklabels(TIME_LABELS)
                    ax.set_xlabel("Timepoint"); ax.set_ylabel("Tumor volume (mm³)")
                    ax.set_title(f"{pdir.name} — {chosen_scenario}")
                    ax.grid(True, alpha=0.3); ax.legend(loc="best")
                    st.pyplot(fig)

                    # Forecast vs actual MAE if any future timepoints exist
                    if actual_future is not None and len(actual_future) > 0:
                        n_fut = len(actual_future)
                        rows = [{"Model": "ML (continual RF)",
                                 "MAE": float(mae(actual_future, ml_mean[:n_fut]))}]
                        if forecasts["Exponential"]["ok"]:
                            rows.append({"Model": "Exponential",
                                         "MAE": float(mae(
                                             actual_future,
                                             forecasts["Exponential"]["future"][:n_fut]))})
                        if forecasts["Gompertz"]["ok"]:
                            rows.append({"Model": "Gompertz",
                                         "MAE": float(mae(
                                             actual_future,
                                             forecasts["Gompertz"]["future"][:n_fut]))})
                        st.markdown(f"##### MAE on {n_fut} actual future point(s)")
                        st.dataframe(pd.DataFrame(rows).round(2),
                                     hide_index=True, width="stretch")

                    # Volumes + RANO
                    st.markdown("##### Volumes + RANO classification")
                    rano_table = {
                        "timepoint": TIME_LABELS[:n_obs],
                        "volume_mm3": np.round(scan.trajectory, 2),
                        "RANO": rano_labels,
                    }
                    st.dataframe(pd.DataFrame(rano_table), hide_index=True, width="stretch")

                    # Future predictions table
                    st.markdown("##### Forecast (FU3..FU5) in mm³")
                    pred_rows = {"timepoint": ["FU3", "FU4", "FU5"]}
                    if forecasts["Exponential"]["ok"]:
                        pred_rows["exponential"] = np.round(forecasts["Exponential"]["future"], 2).tolist()
                    if forecasts["Gompertz"]["ok"]:
                        pred_rows["gompertz"] = np.round(forecasts["Gompertz"]["future"], 2).tolist()
                    pred_rows["ml mean"] = np.round(ml_mean, 2).tolist()
                    pred_rows["ml ±1σ"] = [f"±{s:.1f}" for s in ml_std]
                    st.dataframe(pd.DataFrame(pred_rows), hide_index=True, width="stretch")
