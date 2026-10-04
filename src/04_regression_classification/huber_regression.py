import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "placement_data.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "huber_regression"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("HUBER REGRESSION")
print("=" * 65)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(
    "Dataset shape:",
    df.shape
)

print("\nColumns:")
print(
    df.columns.tolist()
)


# ============================================================
# 2. TARGET COLUMN
# ============================================================

target_candidates = [
    "Salary Package",
    "Salary",
    "salary",
    "Package",
    "package",
    "CTC",
    "ctc"
]

target = None

for column in target_candidates:

    if column in df.columns:

        target = column

        break


if target is None:

    raise ValueError(
        "\nNo continuous salary/package target was found.\n"
        "Huber Regression requires a continuous target."
    )


print(
    "\nTarget column:",
    target
)


# ============================================================
# 3. NUMERICAL FEATURES
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


print("\nNumerical columns:")
print(
    numeric_columns
)


# Remove target
features = [
    column
    for column in numeric_columns
    if column != target
]


# Remove ID / anomaly columns
columns_to_remove = [
    "StudentID",
    "IsAnomaly",
    "PlacementStatus"
]

features = [
    column
    for column in features
    if column not in columns_to_remove
]


print("\nFeatures used:")

for feature in features:

    print(
        "-",
        feature
    )


# ============================================================
# 4. CREATE X AND y
# ============================================================

X = df[features].copy()

y = df[target].copy()


# ============================================================
# 5. HANDLE INVALID VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

y = y.replace(
    [np.inf, -np.inf],
    np.nan
)


# Fill feature missing values
X = X.fillna(
    X.median()
)


# Remove rows where target is missing
valid_rows = y.notna()

X = X.loc[
    valid_rows
]

y = y.loc[
    valid_rows
]


print(
    "\nRows after cleaning:",
    len(X)
)


# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)


print("\n" + "=" * 65)
print("TRAIN / TEST SPLIT")
print("=" * 65)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)


# ============================================================
# 7. HUBER REGRESSION PIPELINE
# ============================================================

pipeline = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "huber",
        HuberRegressor(

            epsilon=1.35,

            alpha=0.0001,

            max_iter=1000
        )
    )
])


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print(
    "\nTraining Huber Regression..."
)

pipeline.fit(
    X_train,
    y_train
)

print(
    "Training completed."
)


# ============================================================
# 9. PREDICTIONS
# ============================================================

y_pred = pipeline.predict(
    X_test
)


# ============================================================
# 10. EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 65)
print("MODEL EVALUATION")
print("=" * 65)

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# 11. ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(10, 7)
)

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5,
    s=20
)

min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=2
)

plt.xlabel(
    "Actual Salary Package"
)

plt.ylabel(
    "Predicted Salary Package"
)

plt.title(
    "Huber Regression - Actual vs Predicted",
    fontsize=16
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


actual_predicted_path = os.path.join(
    OUTPUT_DIR,
    "huber_actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "\nSaved:",
    actual_predicted_path
)


# ============================================================
# 12. RESIDUAL PLOT
# ============================================================

residuals = (
    y_test.values -
    y_pred
)


plt.figure(
    figsize=(10, 7)
)

plt.scatter(
    y_pred,
    residuals,
    alpha=0.5,
    s=20
)

plt.axhline(
    y=0,
    linestyle="--",
    linewidth=2
)

plt.xlabel(
    "Predicted Salary Package"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Huber Regression - Residual Plot",
    fontsize=16
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


residual_path = os.path.join(
    OUTPUT_DIR,
    "huber_residual_plot.png"
)

plt.savefig(
    residual_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    residual_path
)


# ============================================================
# 13. FEATURE COEFFICIENTS
# ============================================================

model = pipeline.named_steps[
    "huber"
]


coefficients = pd.DataFrame({

    "Feature":
        features,

    "Coefficient":
        model.coef_

})


coefficients = coefficients.sort_values(
    by="Coefficient"
)


plt.figure(
    figsize=(12, 8)
)

plt.barh(
    coefficients["Feature"],
    coefficients["Coefficient"]
)

plt.axvline(
    x=0,
    linewidth=1
)

plt.xlabel(
    "Huber Regression Coefficient"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Huber Regression - Feature Coefficients",
    fontsize=16
)

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()


coefficient_path = os.path.join(
    OUTPUT_DIR,
    "huber_feature_coefficients.png"
)

plt.savefig(
    coefficient_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    coefficient_path
)


# ============================================================
# 14. PREDICTION ERROR DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 7)
)

plt.hist(
    residuals,
    bins=30,
    alpha=0.75
)

plt.axvline(
    x=0,
    linestyle="--",
    linewidth=2
)

plt.xlabel(
    "Prediction Error / Residual"
)

plt.ylabel(
    "Number of Students"
)

plt.title(
    "Huber Regression - Residual Distribution",
    fontsize=16
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


residual_distribution_path = os.path.join(
    OUTPUT_DIR,
    "huber_residual_distribution.png"
)

plt.savefig(
    residual_distribution_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    residual_distribution_path
)


# ============================================================
# 15. MODEL SUMMARY IMAGE
# ============================================================

plt.figure(
    figsize=(11, 7)
)

plt.axis(
    "off"
)


summary_text = (
    "HUBER REGRESSION SUMMARY\n"
    "\n"
    f"Dataset Records : {len(df):,}\n"
    f"Features Used   : {len(features)}\n"
    f"Training Records: {len(X_train):,}\n"
    f"Testing Records : {len(X_test):,}\n"
    "\n"
    f"Target          : {target}\n"
    "\n"
    f"MAE             : {mae:.4f}\n"
    f"RMSE            : {rmse:.4f}\n"
    f"R² Score        : {r2:.4f}\n"
    "\n"
    "Huber Parameters\n"
    "epsilon = 1.35\n"
    "alpha = 0.0001\n"
    "max_iter = 1000"
)


plt.text(
    0.5,
    0.5,
    summary_text,
    ha="center",
    va="center",
    fontsize=14
)

plt.title(
    "Huber Regression Model Summary",
    fontsize=18,
    pad=20
)

plt.tight_layout()


summary_path = os.path.join(
    OUTPUT_DIR,
    "huber_model_summary.png"
)

plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    summary_path
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("HUBER REGRESSION COMPLETED")
print("=" * 65)

print("\nOutput folder:")
print(
    OUTPUT_DIR
)

print("\nPNG files:")

print(
    "- huber_actual_vs_predicted.png"
)

print(
    "- huber_residual_plot.png"
)

print(
    "- huber_feature_coefficients.png"
)

print(
    "- huber_residual_distribution.png"
)

print(
    "- huber_model_summary.png"
)