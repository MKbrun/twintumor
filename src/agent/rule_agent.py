import pandas as pd
from collections import Counter
from src.agent.treatment_agent import TreatmentAgent


# Rule-based agent
#
# This class represents a simple rule-based decision model that
# evaluate tumor changes over time.
#
# The agent looks at a sequence of tumor measurements from several
# MRI scans and classifies the tumor response as:
#
# -Progression
# -Partial Remission
# -Stable Disease
#
# The rules compare the current tumor measurement with:
# 1. The baseline measurement
# 2. The smallest tumor value seen so far
#
# In our dataset we use the
# percentage of tumor signal detected in MRI scans.



# Load the dataset containing tumor trajectories
#
# Each patient in the dataset has two simulated futures:
# one progression scenario and one remission scenario.
#
# Both scenarios start from the same baseline scan but
# develop differently over the follow-up scans.
#
# Example:
# Mets_005 progression trajectory
# Mets_005 remission trajectory
#
# This means each patient contributes two samples.
# With 105 patients we therefore get 210 tumor trajectories.

df = pd.read_csv("src/data/consistent_tumor_analysis.csv")


# Run the rule-based agent on every trajectory

results = []

for _, row in df.iterrows():

    baseline = row["baseline_percent"]

    # Progression trajectory
    #
    # Each trajectory contains five follow-up scans.
    # These values represent the percentage of tumor signal
    # detected in the MRI region of interest.

    prog_values = [
        row["progression_FU1"],
        row["progression_FU2"],
        row["progression_FU3"],
        row["progression_FU4"],
        row["progression_FU5"]
    ]

    agent = TreatmentAgent(baseline)

    for v in prog_values:
        evaluation = agent.evaluate(v)
        status = evaluation["status"]

    results.append((row["subject"], "progression", status))

    # Remission trajectory
    #
    # This represents the simulated future where the
    # tumor responds to treatment and decreases over time.

    rem_values = [
        row["remission_FU1"],
        row["remission_FU2"],
        row["remission_FU3"],
        row["remission_FU4"],
        row["remission_FU5"]
    ]

    agent = TreatmentAgent(baseline)

    for v in rem_values:
        evaluation = agent.evaluate(v)
        status = evaluation["status"]

    results.append((row["subject"], "remission", status))

# Evaluate predictions
#
# The rule agent returns three possible outputs:
# Progression, Partial Remission, or Stable Disease.
#
# For evaluation we simplify this into two classes:
# progression and remission.
#
# Stable Disease and Partial Remission are both treated
# as remission for this classification.

correct = 0
predictions = []

for patient, true, pred in results:

    if pred == "Progression":
        pred_label = "progression"
    else:
        pred_label = "remission"

    predictions.append(pred_label)

    if pred_label == true:
        correct += 1
    else:
        print("Disagreement on", patient, "true:", true, "predicted:", pred_label)


accuracy = correct / len(results)

print("Prediction distribution:", Counter(predictions))
print("Rule-based model accuracy:", accuracy)


# Interpretation
#
# If the accuracy is close to 0.5, the rule-based model is
# performing roughly like random guessing.
#
# One reason for this is that the clinical rules were originally
# designed for tumor volume measurements, while our dataset uses
# MRI signal percentages instead.
#
# This rule-based model therefore serves mainly as a baseline
# that we can later compare with more advanced models such as
# machine learning approaches.