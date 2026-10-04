import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "placement_data.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "linear_regression"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("LINEAR REGRESSION")
print("=" * 65)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# TARGET
# ============================================================

TARGET = "Salary Package"

if TARGET not in df.columns:

    raise ValueError(
        f"\nTarget column '{TARGET}' was not found.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


# Convert target to numeric
df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)


# Remove rows where target is missing
df = df.dropna(
    subset=[TARGET]
).copy()


print(
    f"\nValid records for regression: {len(df)}"
)


# ============================================================
# REMOVE IRRELEVANT / TARGET COLUMNS
# ============================================================

columns_to_remove = [
    "StudentID",
    "PlacementStatus",
    "IsAnomaly",
    "Salary Package"
]

columns_to_remove = [
    col
    for col in columns_to_remove
    if col in df.columns
]


X = df.drop(
    columns=columns_to_remove
)

y = df[TARGET]


# ============================================================
# IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


print("\nNumerical Features:")
for feature in numeric_features:
    print("-", feature)


print("\nCategorical Features:")
for feature in categorical_features:
    print("-", feature)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n================================")
print("DATA SPLIT")
print("================================")

print(
    f"Training Records: {len(X_train)}"
)

print(
    f"Testing Records : {len(X_test)}"
)


# ============================================================
# NUMERICAL PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ============================================================
# PREPROCESSOR
# ============================================================

transformers = []


if numeric_features:

    transformers.append(
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        )
    )


if categorical_features:

    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    )


preprocessor = ColumnTransformer(
    transformers=transformers
)


# ============================================================
# LINEAR REGRESSION MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n================================")
print("TRAINING LINEAR REGRESSION")
print("================================")

model.fit(
    X_train,
    y_train
)

print(
    "Model trained successfully."
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# MODEL EVALUATION
# ============================================================

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(
    f"MSE  : {mse:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# IMAGE 1
# ACTUAL VS PREDICTED
# ============================================================

print(
    "\nCreating actual vs predicted graph..."
)

plt.figure(
    figsize=(10, 7)
)

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5,
    s=20
)

# Perfect prediction line
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
    linewidth=2
)

plt.xlabel(
    "Actual Salary Package"
)

plt.ylabel(
    "Predicted Salary Package"
)

plt.title(
    "Linear Regression - Actual vs Predicted Salary"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()


actual_predicted_path = os.path.join(
    OUTPUT_DIR,
    "actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {actual_predicted_path}"
)


# ============================================================
# IMAGE 2
# RESIDUAL PLOT
# ============================================================

print(
    "Creating residual plot..."
)

residuals = y_test - y_pred

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
    linewidth=2
)

plt.xlabel(
    "Predicted Salary Package"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Linear Regression - Residual Plot"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()


residual_path = os.path.join(
    OUTPUT_DIR,
    "residual_plot.png"
)

plt.savefig(
    residual_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {residual_path}"
)


# ============================================================
# IMAGE 3
# RESIDUAL DISTRIBUTION
# ============================================================

print(
    "Creating residual distribution..."
)

plt.figure(
    figsize=(10, 7)
)

plt.hist(
    residuals,
    bins=30,
    edgecolor="black",
    alpha=0.8
)

plt.axvline(
    x=0,
    linewidth=2
)

plt.xlabel(
    "Residual"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Linear Regression - Residual Distribution"
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()


residual_histogram_path = os.path.join(
    OUTPUT_DIR,
    "residual_histogram.png"
)

plt.savefig(
    residual_histogram_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {residual_histogram_path}"
)


# ============================================================
# IMAGE 4
# PREDICTION ERROR
# ============================================================

print(
    "Creating prediction error graph..."
)

absolute_errors = np.abs(
    y_test - y_pred
)

plt.figure(
    figsize=(10, 7)
)

plt.hist(
    absolute_errors,
    bins=30,
    edgecolor="black",
    alpha=0.8
)

plt.xlabel(
    "Absolute Prediction Error"
)

plt.ylabel(
    "Number of Students"
)

plt.title(
    "Linear Regression - Prediction Error Distribution"
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()


error_path = os.path.join(
    OUTPUT_DIR,
    "prediction_error_distribution.png"
)

plt.savefig(
    error_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {error_path}"
)


# ============================================================
# IMAGE 5
# MODEL PERFORMANCE
# ============================================================

print(
    "Creating model performance graph..."
)

metrics_names = [
    "RMSE",
    "MAE"
]

metrics_values = [
    rmse,
    mae
]

plt.figure(
    figsize=(9, 6)
)

bars = plt.bar(
    metrics_names,
    metrics_values
)

plt.ylabel(
    "Error"
)

plt.title(
    "Linear Regression - Error Metrics"
)

plt.grid(
    axis="y",
    alpha=0.25
)

for bar, value in zip(
    bars,
    metrics_values
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()


metrics_path = os.path.join(
    OUTPUT_DIR,
    "error_metrics.png"
)

plt.savefig(
    metrics_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {metrics_path}"
)


# ============================================================
# IMAGE 6
# TARGET DISTRIBUTION
# ============================================================

print(
    "Creating salary distribution graph..."
)

plt.figure(
    figsize=(10, 7)
)

plt.hist(
    y,
    bins=30,
    edgecolor="black",
    alpha=0.8
)

plt.xlabel(
    "Salary Package"
)

plt.ylabel(
    "Number of Students"
)

plt.title(
    "Salary Package Distribution"
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()


salary_distribution_path = os.path.join(
    OUTPUT_DIR,
    "salary_distribution.png"
)

plt.savefig(
    salary_distribution_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {salary_distribution_path}"
)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "linear_regression_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "LINEAR REGRESSION REPORT\n"
    )

    file.write(
        "========================\n\n"
    )

    file.write(
        f"Dataset Records: {len(df)}\n"
    )

    file.write(
        f"Training Records: {len(X_train)}\n"
    )

    file.write(
        f"Testing Records: {len(X_test)}\n\n"
    )

    file.write(
        "Target: Salary Package\n\n"
    )

    file.write(
        "MODEL PERFORMANCE\n"
    )

    file.write(
        "-----------------\n"
    )

    file.write(
        f"MSE  : {mse:.4f}\n"
    )

    file.write(
        f"RMSE : {rmse:.4f}\n"
    )

    file.write(
        f"MAE  : {mae:.4f}\n"
    )

    file.write(
        f"R2   : {r2:.4f}\n"
    )

print(
    f"\nReport saved: {report_path}"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("LINEAR REGRESSION COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nPNG OUTPUTS:")

print("1. actual_vs_predicted.png")
print("2. residual_plot.png")
print("3. residual_histogram.png")
print("4. prediction_error_distribution.png")
print("5. error_metrics.png")
print("6. salary_distribution.png")

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)

print("=" * 65)