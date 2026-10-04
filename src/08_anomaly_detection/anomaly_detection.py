# ==============================================================
# ANOMALY DETECTION
# Placement Prediction Project
# ==============================================================
#
# Algorithms:
# 1. Isolation Forest
# 2. Local Outlier Factor (LOF)
# 3. One-Class SVM
#
# OUTPUT:
# PNG images only + TXT report
# NO CSV FILES
# ==============================================================


import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


# ==============================================================
# PROJECT PATHS
# ==============================================================

# anomaly_detection.py is inside:
#
# PlacementPredict_EDA/
# └── src/
#     └── anomaly/
#         └── anomaly_detection.py
#
# parents[0] = anomaly
# parents[1] = src
# parents[2] = PlacementPredict_EDA

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "placement_data.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "anomaly"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==============================================================
# DISPLAY
# ==============================================================

print("=" * 65)
print("                 ANOMALY DETECTION")
print("=" * 65)

print("\nProject directory:")
print(BASE_DIR)

print("\nDataset path:")
print(DATA_PATH)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ==============================================================
# LOAD DATASET
# ==============================================================

print("\n" + "=" * 65)
print("LOADING DATASET")
print("=" * 65)

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}"
    )


df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")

print("Dataset shape:", df.shape)


# ==============================================================
# SELECT NUMERICAL FEATURES
# ==============================================================

print("\n" + "=" * 65)
print("FEATURE SELECTION")
print("=" * 65)


numeric_df = df.select_dtypes(
    include=[np.number]
).copy()


# --------------------------------------------------------------
# Remove columns that should NOT be used for anomaly detection
# --------------------------------------------------------------

exclude_columns = [
    "StudentID",
    "PlacementStatus",
    "IsAnomaly"
]


for column in exclude_columns:

    if column in numeric_df.columns:

        numeric_df.drop(
            columns=column,
            inplace=True
        )


print("\nFeatures used:")

for column in numeric_df.columns:

    print("-", column)


print(
    "\nNumber of features:",
    numeric_df.shape[1]
)


# ==============================================================
# CHECK FEATURES
# ==============================================================

if numeric_df.shape[1] == 0:

    raise ValueError(
        "No numerical features available "
        "for anomaly detection."
    )


# ==============================================================
# HANDLE MISSING VALUES
# ==============================================================

print("\n" + "=" * 65)
print("MISSING VALUE HANDLING")
print("=" * 65)


missing_before = int(
    numeric_df.isnull().sum().sum()
)


print(
    "Missing values before:",
    missing_before
)


if missing_before > 0:

    numeric_df = numeric_df.fillna(
        numeric_df.median()
    )


missing_after = int(
    numeric_df.isnull().sum().sum()
)


print(
    "Missing values after:",
    missing_after
)


# ==============================================================
# HANDLE INFINITE VALUES
# ==============================================================

numeric_df = numeric_df.replace(
    [np.inf, -np.inf],
    np.nan
)


if numeric_df.isnull().sum().sum() > 0:

    numeric_df = numeric_df.fillna(
        numeric_df.median()
    )


# ==============================================================
# STANDARDIZATION
# ==============================================================

print("\n" + "=" * 65)
print("STANDARDIZATION")
print("=" * 65)


scaler = StandardScaler()

X = scaler.fit_transform(
    numeric_df
)


print(
    "Numeric features standardized successfully."
)


# ==============================================================
# 1. ISOLATION FOREST
# ==============================================================

print("\n" + "=" * 65)
print("1. ISOLATION FOREST")
print("=" * 65)


isolation_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)


isolation_labels = (
    isolation_forest.fit_predict(X)
)


isolation_scores = (
    isolation_forest.decision_function(X)
)


# -1 = anomaly
#  1 = normal

isolation_anomalies = int(
    np.sum(isolation_labels == -1)
)

isolation_normal = int(
    np.sum(isolation_labels == 1)
)


print(
    "\nNormal records:",
    isolation_normal
)

print(
    "Anomalous records:",
    isolation_anomalies
)


# ==============================================================
# ISOLATION FOREST GRAPH
# ==============================================================

plt.figure(
    figsize=(10, 6)
)


plt.hist(
    isolation_scores,
    bins=35,
    edgecolor="black"
)


plt.axvline(
    0,
    linestyle="--",
    linewidth=2,
    label="Anomaly Boundary"
)


plt.title(
    "Isolation Forest - Anomaly Score Distribution",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Anomaly Score"
)

plt.ylabel(
    "Number of Students"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()


isolation_plot = (
    OUTPUT_DIR /
    "isolation_forest_scores.png"
)


plt.savefig(
    isolation_plot,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nSaved:",
    isolation_plot
)


# ==============================================================
# 2. LOCAL OUTLIER FACTOR
# ==============================================================

print("\n" + "=" * 65)
print("2. LOCAL OUTLIER FACTOR")
print("=" * 65)


n_neighbors = min(
    20,
    max(2, len(X) - 1)
)


lof = LocalOutlierFactor(
    n_neighbors=n_neighbors,
    contamination=0.05
)


lof_labels = lof.fit_predict(X)

lof_scores = (
    lof.negative_outlier_factor_
)


# -1 = anomaly
#  1 = normal

lof_anomalies = int(
    np.sum(lof_labels == -1)
)

lof_normal = int(
    np.sum(lof_labels == 1)
)


print(
    "\nNormal records:",
    lof_normal
)

print(
    "Anomalous records:",
    lof_anomalies
)


# ==============================================================
# LOF GRAPH
# ==============================================================

plt.figure(
    figsize=(10, 6)
)


plt.hist(
    lof_scores,
    bins=35,
    edgecolor="black"
)


plt.axvline(
    -1,
    linestyle="--",
    linewidth=2,
    label="Reference Level"
)


plt.title(
    "Local Outlier Factor - Score Distribution",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "LOF Negative Outlier Factor"
)

plt.ylabel(
    "Number of Students"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()


lof_plot = (
    OUTPUT_DIR /
    "lof_scores.png"
)


plt.savefig(
    lof_plot,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nSaved:",
    lof_plot
)


# ==============================================================
# 3. ONE-CLASS SVM
# ==============================================================

print("\n" + "=" * 65)
print("3. ONE-CLASS SVM")
print("=" * 65)


one_class_svm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.05
)


svm_labels = (
    one_class_svm.fit_predict(X)
)


svm_scores = (
    one_class_svm.decision_function(X)
)


# -1 = anomaly
#  1 = normal

svm_anomalies = int(
    np.sum(svm_labels == -1)
)

svm_normal = int(
    np.sum(svm_labels == 1)
)


print(
    "\nNormal records:",
    svm_normal
)

print(
    "Anomalous records:",
    svm_anomalies
)


# ==============================================================
# ONE-CLASS SVM GRAPH
# ==============================================================

plt.figure(
    figsize=(10, 6)
)


plt.hist(
    svm_scores,
    bins=35,
    edgecolor="black"
)


plt.axvline(
    0,
    linestyle="--",
    linewidth=2,
    label="Anomaly Boundary"
)


plt.title(
    "One-Class SVM - Anomaly Score Distribution",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Decision Function Score"
)

plt.ylabel(
    "Number of Students"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()


svm_plot = (
    OUTPUT_DIR /
    "one_class_svm_scores.png"
)


plt.savefig(
    svm_plot,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nSaved:",
    svm_plot
)


# ==============================================================
# 4. METHOD COMPARISON
# ==============================================================

print("\n" + "=" * 65)
print("ANOMALY DETECTION COMPARISON")
print("=" * 65)


methods = [
    "Isolation Forest",
    "Local Outlier Factor",
    "One-Class SVM"
]


normal_counts = [
    isolation_normal,
    lof_normal,
    svm_normal
]


anomaly_counts = [
    isolation_anomalies,
    lof_anomalies,
    svm_anomalies
]


print("\nMethod Results:")

for i in range(len(methods)):

    print(
        f"{methods[i]} -> "
        f"Normal: {normal_counts[i]}, "
        f"Anomaly: {anomaly_counts[i]}"
    )


# ==============================================================
# COMPARISON GRAPH
# ==============================================================

x = np.arange(
    len(methods)
)

width = 0.35


plt.figure(
    figsize=(11, 6)
)


plt.bar(
    x - width / 2,
    normal_counts,
    width,
    label="Normal Records",
    edgecolor="black"
)


plt.bar(
    x + width / 2,
    anomaly_counts,
    width,
    label="Anomalous Records",
    edgecolor="black"
)


plt.xticks(
    x,
    [
        "Isolation Forest",
        "LOF",
        "One-Class SVM"
    ]
)


plt.ylabel(
    "Number of Students"
)

plt.xlabel(
    "Anomaly Detection Method"
)


plt.title(
    "Comparison of Anomaly Detection Methods",
    fontsize=16,
    fontweight="bold"
)


plt.legend()

plt.grid(
    axis="y",
    alpha=0.25
)


plt.tight_layout()


comparison_plot = (
    OUTPUT_DIR /
    "anomaly_method_comparison.png"
)


plt.savefig(
    comparison_plot,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nSaved:",
    comparison_plot
)


# ==============================================================
# 5. ANOMALY AGREEMENT
# ==============================================================

print("\n" + "=" * 65)
print("ANOMALY AGREEMENT")
print("=" * 65)


# Count how many algorithms detected each record
# as an anomaly.

anomaly_count = (
    (isolation_labels == -1).astype(int)
    +
    (lof_labels == -1).astype(int)
    +
    (svm_labels == -1).astype(int)
)


# --------------------------------------------------------------
# Final anomaly:
#
# At least 2 of 3 algorithms must agree.
# --------------------------------------------------------------

final_anomaly = np.where(
    anomaly_count >= 2,
    -1,
    1
)


final_anomalies = int(
    np.sum(final_anomaly == -1)
)

final_normal = int(
    np.sum(final_anomaly == 1)
)


print(
    "\nFinal normal records:",
    final_normal
)

print(
    "Final anomalous records:",
    final_anomalies
)


# ==============================================================
# AGREEMENT COUNTS
# ==============================================================

agreement_0 = int(
    np.sum(anomaly_count == 0)
)

agreement_1 = int(
    np.sum(anomaly_count == 1)
)

agreement_2 = int(
    np.sum(anomaly_count == 2)
)

agreement_3 = int(
    np.sum(anomaly_count == 3)
)


# ==============================================================
# AGREEMENT GRAPH
# ==============================================================

agreement_labels = [
    "0 Methods",
    "1 Method",
    "2 Methods",
    "3 Methods"
]


agreement_values = [
    agreement_0,
    agreement_1,
    agreement_2,
    agreement_3
]


plt.figure(
    figsize=(9, 6)
)


bars = plt.bar(
    agreement_labels,
    agreement_values,
    edgecolor="black"
)


plt.title(
    "Anomaly Detection Agreement",
    fontsize=16,
    fontweight="bold"
)


plt.xlabel(
    "Number of Methods Detecting Anomaly"
)

plt.ylabel(
    "Number of Students"
)


plt.grid(
    axis="y",
    alpha=0.25
)


# Add values above bars

for bar, value in zip(
    bars,
    agreement_values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        str(value),
        ha="center",
        va="bottom",
        fontweight="bold"
    )


plt.tight_layout()


agreement_plot = (
    OUTPUT_DIR /
    "anomaly_agreement.png"
)


plt.savefig(
    agreement_plot,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nSaved:",
    agreement_plot
)


# ==============================================================
# 6. FINAL ANOMALY DISTRIBUTION
# ==============================================================

plt.figure(
    figsize=(8, 6)
)


labels = [
    "Normal Students",
    "Anomalous Students"
]


values = [
    final_normal,
    final_anomalies
]


bars = plt.bar(
    labels,
    values,
    edgecolor="black"
)


plt.title(
    "Final Anomaly Detection Result",
    fontsize=16,
    fontweight="bold"
)


plt.ylabel(
    "Number of Students"
)


plt.grid(
    axis="y",
    alpha=0.25
)


for bar, value in zip(
    bars,
    values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        str(value),
        ha="center",
        va="bottom",
        fontweight="bold"
    )


plt.tight_layout()


final_plot = (
    OUTPUT_DIR /
    "final_anomaly_distribution.png"
)


plt.savefig(
    final_plot,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved:",
    final_plot
)


# ==============================================================
# 7. CREATE REPORT
# ==============================================================

report_path = (
    OUTPUT_DIR /
    "anomaly_detection_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "====================================================\n"
    )

    file.write(
        "ANOMALY DETECTION REPORT\n"
    )

    file.write(
        "====================================================\n\n"
    )

    file.write(
        f"Dataset: {DATA_PATH}\n"
    )

    file.write(
        f"Number of records: {len(df)}\n"
    )

    file.write(
        f"Number of features used: "
        f"{numeric_df.shape[1]}\n\n"
    )


    file.write(
        "ALGORITHMS USED\n"
    )

    file.write(
        "----------------------------------------------------\n"
    )

    file.write(
        "1. Isolation Forest\n"
    )

    file.write(
        "2. Local Outlier Factor (LOF)\n"
    )

    file.write(
        "3. One-Class SVM\n\n"
    )


    file.write(
        "METHOD RESULTS\n"
    )

    file.write(
        "----------------------------------------------------\n"
    )

    file.write(
        f"Isolation Forest anomalies: "
        f"{isolation_anomalies}\n"
    )

    file.write(
        f"Isolation Forest normal: "
        f"{isolation_normal}\n\n"
    )

    file.write(
        f"LOF anomalies: "
        f"{lof_anomalies}\n"
    )

    file.write(
        f"LOF normal: "
        f"{lof_normal}\n\n"
    )

    file.write(
        f"One-Class SVM anomalies: "
        f"{svm_anomalies}\n"
    )

    file.write(
        f"One-Class SVM normal: "
        f"{svm_normal}\n\n"
    )


    file.write(
        "FINAL RESULT\n"
    )

    file.write(
        "----------------------------------------------------\n"
    )

    file.write(
        "Final anomaly rule: "
        "At least 2 of 3 methods must agree.\n\n"
    )

    file.write(
        f"Final anomalous records: "
        f"{final_anomalies}\n"
    )

    file.write(
        f"Final normal records: "
        f"{final_normal}\n\n"
    )


    file.write(
        "AGREEMENT\n"
    )

    file.write(
        "----------------------------------------------------\n"
    )

    file.write(
        f"0 methods: {agreement_0}\n"
    )

    file.write(
        f"1 method : {agreement_1}\n"
    )

    file.write(
        f"2 methods: {agreement_2}\n"
    )

    file.write(
        f"3 methods: {agreement_3}\n"
    )


print(
    "\nSaved:",
    report_path
)


# ==============================================================
# FINAL OUTPUT
# ==============================================================

print("\n" + "=" * 65)
print("ANOMALY DETECTION COMPLETED SUCCESSFULLY")
print("=" * 65)


print("\nOutput folder:")
print(OUTPUT_DIR)


print("\nGenerated files:")

for file in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        "-",
        file.name
    )


print("\n")


# ==============================================================
# END
# ==============================================================