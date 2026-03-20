import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Machine learning models to classify each patient trajectory
# Does not predict future growth
# This is just to try out machine learning

# Load dataset
df = pd.read_csv("src/data/consistent_tumor_analysis.csv")


# Split patients 
patients = df["subject"].unique()

train_patients, test_patients = train_test_split(
    patients, test_size=0.2, random_state=42
)

X_train, y_train = [], []
X_test, y_test = [], []


# Create features
for _, row in df.iterrows():

    baseline = row["baseline_percent"]
    patient = row["subject"]

    # progression trajectory
    prog = [
        row["progression_FU1"],
        row["progression_FU2"],
        row["progression_FU3"],
        row["progression_FU4"],
        row["progression_FU5"]
    ]

    final = prog[-1]
    change = final - baseline

    # step-wise changes 
    d1 = prog[0] - baseline
    d5 = prog[4] - prog[3]

    features = [baseline, final, change, d1, d5]

    if patient in train_patients:
        X_train.append(features)
        y_train.append(1)  # progression
    else:
        X_test.append(features)
        y_test.append(1)


    # remission trajectory
    rem = [
        row["remission_FU1"],
        row["remission_FU2"],
        row["remission_FU3"],
        row["remission_FU4"],
        row["remission_FU5"]
    ]

    final = rem[-1]
    change = final - baseline

    # step-wise changes 
    d1 = rem[0] - baseline
    d5 = rem[4] - rem[3]

    features = [baseline, final, change, d1, d5]

    if patient in train_patients:
        X_train.append(features)
        y_train.append(0)  # remission
    else:
        X_test.append(features)
        y_test.append(0)


# Scale features 
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Logistic Regression
log_model = LogisticRegression(max_iter=1000, C=0.5)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)

print("Logistic Regression accuracy:", acc_log)


# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=3,
    random_state=42
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("Random Forest accuracy:", acc_rf)


# Support Vector Machine
svm_model = SVC(kernel="rbf", C=1.0)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)

print("SVM accuracy:", acc_svm)


# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)

print("KNN accuracy:", acc_knn)