import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# LOAD DATA

CSV_PATH = Path("data/analysis/consistent_tumor_analysis.csv")
df = pd.read_csv(CSV_PATH)

# RULE-BASED AGENT

def rule_based_classification(baseline, current, smallest):
    pct_change_baseline = ((current - baseline) / baseline) * 100
    pct_change_smallest = ((current - smallest) / smallest) * 100

    if current == 0:
        return "Complete Remission"
    elif pct_change_baseline <= -50:
        return "Partial Remission"
    elif pct_change_smallest >= 25:
        return "Progression"
    else:
        return "Stable"

# Example usage on one patient
row = df.iloc[0]
baseline = row["baseline_percent"]
smallest = baseline

print("\n--- Rule-Based Example ---")
for t in range(1, 6):
    current = row[f"progression_FU{t}"]
    smallest = min(smallest, current)
    label = rule_based_classification(baseline, current, smallest)
    print(f"FU{t}: {label}")

# EXPONENTIAL MODEL

def fit_exponential(times, values):
    log_vals = np.log(values)
    k, log_V0 = np.polyfit(times, log_vals, 1)
    return k, log_V0

def predict_exponential(times, k, log_V0):
    return np.exp(log_V0 + k * times)

# BUILD DATASET FOR ML

X = []
y = []

for _, row in df.iterrows():
    features = [
        row["baseline_percent"],
        row["progression_FU1"],
        row["progression_FU2"]
    ]

    target = [
        row["progression_FU3"],
        row["progression_FU4"],
        row["progression_FU5"]
    ]

    X.append(features)
    y.append(target)

X = np.array(X)
y = np.array(y)

# TRAIN ML MODEL

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ml_model = LinearRegression()
ml_model.fit(X_train, y_train)

y_pred = ml_model.predict(X_test)

ml_error = mean_absolute_error(y_test, y_pred)
print("\nML Model Error:", ml_error)

# EVALUATE EXPONENTIAL MODEL

exp_errors = []

for _, row in df.iterrows():
    values = np.array([
        row["baseline_percent"],
        row["progression_FU1"],
        row["progression_FU2"],
        row["progression_FU3"],
        row["progression_FU4"],
        row["progression_FU5"]
    ])

    times_known = np.array([0, 1, 2])
    values_known = values[:3]

    k, log_V0 = fit_exponential(times_known, values_known)

    times_all = np.arange(6)
    predicted = predict_exponential(times_all, k, log_V0)

    error = np.mean(np.abs(values[3:] - predicted[3:]))
    exp_errors.append(error)

print("Exponential Model Error:", np.mean(exp_errors))

# VISUALIZATION 1 — MULTIPLE SUBJECTS

print("\nShowing multiple patient predictions")

for i in range(3):  # show 3 subjects

    input_vals = X_test[i]
    true_future = y_test[i]
    pred_future = y_pred[i]

    true_full = np.concatenate([input_vals, true_future])
    pred_full_ml = np.concatenate([input_vals, pred_future])

    k, log_V0 = fit_exponential(
        np.array([0, 1, 2]),
        input_vals
    )

    times = np.arange(6)
    pred_full_exp = predict_exponential(times, k, log_V0)

    plt.figure()

    plt.plot(times, true_full, label="True", linewidth=2)
    plt.plot(times, pred_full_ml, label="ML", linestyle="--")
    plt.plot(times, pred_full_exp, label="Exponential", linestyle=":")

    plt.title(f"Subject {i}")
    plt.xlabel("Time")
    plt.ylabel("Tumor %")
    plt.legend()
    plt.grid(True)
    plt.show()

# VISUALIZATION 2 — SINGLE CLEAN DEMO

i = 0

input_vals = X_test[i]
true_future = y_test[i]
pred_future = y_pred[i]

true_full = np.concatenate([input_vals, true_future])
pred_full_ml = np.concatenate([input_vals, pred_future])

k, log_V0 = fit_exponential(
    np.array([0, 1, 2]),
    input_vals
)

times = np.arange(6)
pred_full_exp = predict_exponential(times, k, log_V0)

plt.figure(figsize=(8, 5))

plt.plot(times, true_full, label="True", linewidth=2)
plt.plot(times, pred_full_ml, label="ML Prediction", linestyle="--")
plt.plot(times, pred_full_exp, label="Exponential Prediction", linestyle=":")

plt.xlabel("Time (Baseline → FU5)")
plt.ylabel("Tumor %")
plt.title("Digital Twin: Prediction Comparison")
plt.legend()
plt.grid(True)
plt.show()

# VISUALIZATION 3 — AVERAGE PERFORMANCE

mean_true = np.mean(y_test, axis=0)
mean_pred = np.mean(y_pred, axis=0)

plt.figure()

plt.plot([3, 4, 5], mean_true, label="True Mean", linewidth=2)
plt.plot([3, 4, 5], mean_pred, label="Predicted Mean", linestyle="--")

plt.xlabel("Future Timepoints (FU3–FU5)")
plt.ylabel("Tumor %")
plt.title("Average Prediction Performance")
plt.legend()
plt.grid(True)
plt.show()