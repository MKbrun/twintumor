import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

# Some graphs and plots to help visualize the data
# This is just a test to try out different plots and check the data

# Load data
df = pd.read_csv("src/data/consistent_tumor_analysis.csv")

# Basic counts (FU1 vs FU5)

prog_over = (df["progression_FU5"] > df["progression_FU1"]).sum()
prog_equal = np.isclose(df["progression_FU5"], df["progression_FU1"]).sum()
prog_under = (df["progression_FU5"] < df["progression_FU1"]).sum()

rem_over = (df["remission_FU5"] > df["remission_FU1"]).sum()
rem_equal = np.isclose(df["remission_FU5"], df["remission_FU1"]).sum()
rem_under = (df["remission_FU5"] < df["remission_FU1"]).sum()

print("\n--- Progression (FU5 vs FU1) ---")
print("Over FU1: ", prog_over)
print("Equal to FU1:", prog_equal)
print("Under FU1:", prog_under)

print("\n--- Remission (FU5 vs FU1) ---")
print("Over FU1: ", rem_over)
print("Equal to FU1:", rem_equal)
print("Under FU1:", rem_under)

# Statistical signal

df["prog_delta"] = df["progression_FU5"] - df["progression_FU1"]
df["rem_delta"] = df["remission_FU5"] - df["remission_FU1"]

t_stat, t_p = stats.ttest_ind(df["prog_delta"], df["rem_delta"], equal_var=False)
u_stat, u_p = stats.mannwhitneyu(df["prog_delta"], df["rem_delta"])

print("\n--- Statistical Tests ---")
print(f"T-test p-value: {t_p}")
print(f"Mann-Whitney U p-value: {u_p}")

# AUC 
y_true = np.concatenate([np.ones(len(df)), np.zeros(len(df))])
scores = np.concatenate([df["prog_delta"], df["rem_delta"]])
auc = roc_auc_score(y_true, scores)

print(f"ROC AUC (delta only): {auc}")

# FEATURE ENGINEERING

def extract_features(prefix):
    data = df[["baseline_percent"] + [f"{prefix}_FU{i}" for i in range(1, 6)]].values

    features = {}

    # total change (baseline - FU5)
    features["delta_total"] = data[:, -1] - data[:, 0]

    # FU1 - FU5 (keep for comparison)
    features["delta_1_5"] = data[:, -1] - data[:, 1]

    # percent change from baseline
    features["pct_change"] = (data[:, -1] - data[:, 0]) / (data[:, 0] + 1e-6)

    # slope across ALL timepoints
    x = np.arange(6)
    features["slope"] = [np.polyfit(x, row, 1)[0] for row in data]

    # early (baseline - FU2)
    features["early"] = data[:, 2] - data[:, 0]

    # late (FU3 - FU5)
    features["late"] = data[:, -1] - data[:, 3]

    # stats
    features["max"] = data.max(axis=1)
    features["min"] = data.min(axis=1)
    features["std"] = data.std(axis=1)

    # directionality
    diffs = np.diff(data, axis=1)
    features["num_up"] = (diffs > 0).sum(axis=1)
    features["num_down"] = (diffs < 0).sum(axis=1)

    return pd.DataFrame(features)

prog_feat = extract_features("progression")
rem_feat = extract_features("remission")

# VISUALIZATIONS

# 1. Full trajectories (baseline included)
plt.figure(figsize=(10, 6))

for _, row in df.iterrows():
    times = [0, 1, 2, 3, 4, 5]

    plt.plot(times, [
        row["baseline_percent"],
        row["progression_FU1"],
        row["progression_FU2"],
        row["progression_FU3"],
        row["progression_FU4"],
        row["progression_FU5"]
    ], alpha=0.2)

    plt.plot(times, [
        row["baseline_percent"],
        row["remission_FU1"],
        row["remission_FU2"],
        row["remission_FU3"],
        row["remission_FU4"],
        row["remission_FU5"]
    ], alpha=0.2)

plt.title(f"Full Trajectories (Baseline → FU5) | AUC ≈ {auc:.2f}")
plt.xlabel("Time (Baseline → FU5)")
plt.ylabel("Tumor %")
plt.grid(True)
plt.show()

#  2. Mean trajectory + variability 
plt.figure()

time = np.arange(6)

prog_data = df[["baseline_percent"] + [f"progression_FU{i}" for i in range(1, 6)]].values
rem_data = df[["baseline_percent"] + [f"remission_FU{i}" for i in range(1, 6)]].values

plt.plot(time, prog_data.mean(axis=0), label="Progression")
plt.fill_between(time,
                 prog_data.mean(axis=0) - prog_data.std(axis=0),
                 prog_data.mean(axis=0) + prog_data.std(axis=0),
                 alpha=0.2)

plt.plot(time, rem_data.mean(axis=0), label="Remission")
plt.fill_between(time,
                 rem_data.mean(axis=0) - rem_data.std(axis=0),
                 rem_data.mean(axis=0) + rem_data.std(axis=0),
                 alpha=0.2)

plt.title("Mean Trajectories with Variability")
plt.xlabel("Time")
plt.ylabel("Tumor %")
plt.legend()
plt.grid(True)
plt.show()

# 3. Histogram (baseline - FU5 change) 
plt.figure()

plt.hist(prog_feat["delta_total"], bins=20, alpha=0.5, label="Progression")
plt.hist(rem_feat["delta_total"], bins=20, alpha=0.5, label="Remission")

plt.title("Baseline → FU5 Change Distribution")
plt.legend()
plt.show()

# 4. Slope vs total change 
plt.figure()

plt.scatter(prog_feat["slope"], prog_feat["delta_total"], alpha=0.6, label="Progression")
plt.scatter(rem_feat["slope"], rem_feat["delta_total"], alpha=0.6, label="Remission")

plt.xlabel("Slope (Baseline → FU5)")
plt.ylabel("Total Change")
plt.title("Slope vs Total Change")
plt.legend()
plt.show()

# 5. Early vs late dynamics 
plt.figure()

plt.scatter(prog_feat["early"], prog_feat["late"], alpha=0.6, label="Progression")
plt.scatter(rem_feat["early"], rem_feat["late"], alpha=0.6, label="Remission")

plt.xlabel("Early change (Baseline → FU2)")
plt.ylabel("Late change (FU3 → FU5)")
plt.title("Early vs Late Behavior")
plt.legend()
plt.show()

# 6. Variability 
plt.figure()

plt.hist(prog_feat["std"], bins=20, alpha=0.5, label="Progression")
plt.hist(rem_feat["std"], bins=20, alpha=0.5, label="Remission")

plt.title("Trajectory Variability")
plt.legend()
plt.show()