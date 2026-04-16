import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# LOAD DATA

df = pd.read_csv("src/data/consistent_tumor_analysis.csv")

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

# Example usage
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

# BUILD DATASET

X, y = [], []

for _, row in df.iterrows():
    X.append([
        row["baseline_percent"],
        row["progression_FU1"],
        row["progression_FU2"]
    ])

    y.append([
        row["progression_FU3"],
        row["progression_FU4"],
        row["progression_FU5"]
    ])

X = np.array(X)
y = np.array(y)

# SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SCALING (for MLP)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ML MODELS

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
ml_error = mean_absolute_error(y_test, y_pred_linear)

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_error = mean_absolute_error(y_test, y_pred_rf)

# Gradient Boosting
gb_model = MultiOutputRegressor(
    GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
gb_error = mean_absolute_error(y_test, y_pred_gb)

# MLP (Neural Network)
mlp_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42
)

mlp_model.fit(X_train_scaled, y_train)
y_pred_mlp = mlp_model.predict(X_test_scaled)
mlp_error = mean_absolute_error(y_test, y_pred_mlp)

print("\n=== Overall Model Performance ===")
print("Linear Regression MAE:", ml_error)
print("Random Forest MAE:", rf_error)
print("Gradient Boosting MAE:", gb_error)
print("MLP MAE:", mlp_error)

# EXPONENTIAL MODEL

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

    k, log_V0 = fit_exponential(np.array([0, 1, 2]), values[:3])
    predicted = predict_exponential(np.arange(6), k, log_V0)

    error = np.mean(np.abs(values[3:] - predicted[3:]))
    exp_errors.append(error)

exp_error = np.mean(exp_errors)
print("Exponential Model MAE:", exp_error)

# VISUALIZATION — MULTIPLE SUBJECTS

print("\nShowing patient predictions")

num_subjects = 4

for i in range(num_subjects):

    input_vals = X_test[i]
    true_future = y_test[i]

    pred_future_linear = y_pred_linear[i]
    pred_future_rf = y_pred_rf[i]
    pred_future_gb = y_pred_gb[i]
    pred_future_mlp = y_pred_mlp[i]

    # patient-specific MAE
    patient_error_linear = np.mean(np.abs(true_future - pred_future_linear))
    patient_error_rf = np.mean(np.abs(true_future - pred_future_rf))
    patient_error_gb = np.mean(np.abs(true_future - pred_future_gb))
    patient_error_mlp = np.mean(np.abs(true_future - pred_future_mlp))

    true_full = np.concatenate([input_vals, true_future])
    pred_full_linear = np.concatenate([input_vals, pred_future_linear])
    pred_full_rf = np.concatenate([input_vals, pred_future_rf])
    pred_full_gb = np.concatenate([input_vals, pred_future_gb])
    pred_full_mlp = np.concatenate([input_vals, pred_future_mlp])

    k, log_V0 = fit_exponential(np.array([0, 1, 2]), input_vals)
    times = np.arange(6)
    pred_full_exp = predict_exponential(times, k, log_V0)

    plt.figure(figsize=(7, 4))

    plt.plot(times, true_full, label="True Tumor Progression", linewidth=2)

    plt.plot(times, pred_full_linear,
             label=f"Linear Regression (Subject MAE={patient_error_linear:.2f})",
             linestyle="--")

    plt.plot(times, pred_full_rf,
             label=f"Random Forest (Subject MAE={patient_error_rf:.2f})",
             linestyle="-.")

    plt.plot(times, pred_full_gb,
             label=f"Gradient Boosting (Subject MAE={patient_error_gb:.2f})",
             linestyle="-")

    plt.plot(times, pred_full_mlp,
             label=f"MLP (Subject MAE={patient_error_mlp:.2f})",
             linestyle="-.")

    plt.plot(times, pred_full_exp,
             label="Exponential Model",
             linestyle=":")

    plt.title(f"Model Comparison - Random Subject {i}")
    plt.xlabel("Time (Baseline -> FU5)")
    plt.ylabel("Tumor %")
    plt.legend()
    plt.grid(True)

    plt.show()

# VISUALIZATION — AVERAGE PERFORMANCE

mean_true = np.mean(y_test, axis=0)
mean_pred_linear = np.mean(y_pred_linear, axis=0)
mean_pred_rf = np.mean(y_pred_rf, axis=0)
mean_pred_gb = np.mean(y_pred_gb, axis=0)
mean_pred_mlp = np.mean(y_pred_mlp, axis=0)

plt.figure()

plt.plot([3, 4, 5], mean_true, label="True Mean", linewidth=2)

plt.plot([3, 4, 5], mean_pred_linear,
         label=f"Linear (MAE={ml_error:.2f})", linestyle="--")

plt.plot([3, 4, 5], mean_pred_rf,
         label=f"RF (MAE={rf_error:.2f})", linestyle="-.")

plt.plot([3, 4, 5], mean_pred_gb,
         label=f"GB (MAE={gb_error:.2f})", linestyle="-")

plt.plot([3, 4, 5], mean_pred_mlp,
         label=f"MLP (MAE={mlp_error:.2f})", linestyle="-.")

plt.xlabel("Future Timepoints (FU3-FU5)")
plt.ylabel("Tumor %")
plt.title("Average Prediction Performance")

plt.legend()
plt.grid(True)
plt.show()