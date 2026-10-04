import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
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

import config


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "logistic_regression"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def train_logistic_model():

    print("=" * 60)
    print("LOGISTIC REGRESSION")
    print("=" * 60)

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print("\nLoading dataset...")

    df = pd.read_csv(
        config.RAW_DATA_PATH
    )

    print(
        f"Dataset Shape: {df.shape}"
    )

    # ========================================================
    # TARGET
    # ========================================================

    target = "PlacementStatus"

    if target not in df.columns:

        raise ValueError(
            f"Target column '{target}' not found."
        )

    # Convert target to numeric
    df[target] = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    # Remove invalid target rows
    df = df.dropna(
        subset=[target]
    ).copy()

    df[target] = df[target].astype(int)

    # ========================================================
    # SELECT NUMERICAL FEATURES
    # ========================================================

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Remove target
    if target in numeric_columns:
        numeric_columns.remove(target)

    # Remove ID-like columns
    columns_to_remove = [
        "StudentID",
        "IsAnomaly"
    ]

    numeric_columns = [
        col
        for col in numeric_columns
        if col not in columns_to_remove
    ]

    print("\nFeatures Used:")

    for feature in numeric_columns:
        print("-", feature)

    # ========================================================
    # CREATE X AND y
    # ========================================================

    X = df[numeric_columns].copy()

    y = df[target].copy()

    # ========================================================
    # HANDLE MISSING VALUES
    # ========================================================

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median()
    )

    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    print("\n" + "=" * 40)
    print("DATA SPLIT")
    print("=" * 40)

    print(
        f"Training Records: {len(X_train)}"
    )

    print(
        f"Testing Records : {len(X_test)}"
    )

    # ========================================================
    # STANDARDIZATION
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ========================================================
    # TRAIN LOGISTIC REGRESSION
    # ========================================================

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    y_pred = model.predict(
        X_test_scaled
    )

    y_probability = model.predict_proba(
        X_test_scaled
    )[:, 1]

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\n" + "=" * 40)
    print("MODEL PERFORMANCE")
    print("=" * 40)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Not Placed",
                "Placed"
            ]
        )
    )

    # ========================================================
    # 1. CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "Logistic Regression - Confusion Matrix",
        fontsize=16,
        pad=15
    )

    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["Not Placed", "Placed"]
    )

    plt.yticks(
        [0, 1],
        ["Not Placed", "Placed"]
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=16
            )

    plt.tight_layout()

    confusion_path = os.path.join(
        OUTPUT_DIR,
        "logistic_confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nSaved:",
        confusion_path
    )

    # ========================================================
    # 2. FEATURE COEFFICIENTS
    # ========================================================

    coefficients = pd.DataFrame({

        "Feature": numeric_columns,

        "Coefficient":
            model.coef_[0]

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

    plt.xlabel(
        "Coefficient"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Logistic Regression - Feature Coefficients",
        fontsize=16,
        pad=15
    )

    plt.axvline(
        x=0,
        linewidth=1
    )

    plt.tight_layout()

    coefficient_path = os.path.join(
        OUTPUT_DIR,
        "logistic_feature_coefficients.png"
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

    # ========================================================
    # 3. ROC CURVE
    # ========================================================

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_probability
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(
        figsize=(9, 7)
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"Logistic Regression (AUC = {roc_auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Random Classifier"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Logistic Regression - ROC Curve",
        fontsize=16,
        pad=15
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    roc_path = os.path.join(
        OUTPUT_DIR,
        "logistic_roc_curve.png"
    )

    plt.savefig(
        roc_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        roc_path
    )

    # ========================================================
    # 4. PREDICTION PROBABILITY DISTRIBUTION
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        y_probability[y_test.values == 0],
        bins=20,
        alpha=0.6,
        label="Not Placed"
    )

    plt.hist(
        y_probability[y_test.values == 1],
        bins=20,
        alpha=0.6,
        label="Placed"
    )

    plt.xlabel(
        "Predicted Probability of Placement"
    )

    plt.ylabel(
        "Number of Students"
    )

    plt.title(
        "Logistic Regression - Prediction Probability Distribution",
        fontsize=15,
        pad=15
    )

    plt.legend()

    plt.tight_layout()

    probability_path = os.path.join(
        OUTPUT_DIR,
        "logistic_probability_distribution.png"
    )

    plt.savefig(
        probability_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        probability_path
    )

    # ========================================================
    # 5. MODEL SUMMARY IMAGE
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.axis(
        "off"
    )

    summary_text = (
        "LOGISTIC REGRESSION MODEL\n"
        "\n"
        f"Dataset Records : {len(df):,}\n"
        f"Features Used   : {len(numeric_columns)}\n"
        f"Training Records: {len(X_train):,}\n"
        f"Testing Records : {len(X_test):,}\n"
        "\n"
        f"Accuracy : {accuracy:.4f}\n"
        f"Accuracy : {accuracy * 100:.2f}%\n"
        f"ROC-AUC  : {roc_auc:.4f}\n"
        "\n"
        "Target: PlacementStatus\n"
        "Classes: Not Placed / Placed"
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
        "Logistic Regression Summary",
        fontsize=18,
        pad=20
    )

    plt.tight_layout()

    summary_path = os.path.join(
        OUTPUT_DIR,
        "logistic_model_summary.png"
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

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION COMPLETED")
    print("=" * 60)

    print("\nOutput folder:")
    print(OUTPUT_DIR)

    print("\nPNG files created:")

    print(
        "- logistic_confusion_matrix.png"
    )

    print(
        "- logistic_feature_coefficients.png"
    )

    print(
        "- logistic_roc_curve.png"
    )

    print(
        "- logistic_probability_distribution.png"
    )

    print(
        "- logistic_model_summary.png"
    )

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "features": numeric_columns
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    train_logistic_model()