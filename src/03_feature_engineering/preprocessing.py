import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler
)


# =========================================================
# PROJECT PATH
# =========================================================

# preprocession.py
#       ↓
# 03_feature_engineering
#       ↓
# src
#       ↓
# PlacementPredict_EDA

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = BASE_DIR / "data" / "placement_data.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "preprocessing"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=================================================")
print("PREPROCESSING")
print("=================================================")

print("\nPROJECT ROOT:")
print(BASE_DIR)

print("\nDATA PATH:")
print(DATA_PATH)

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)


# =========================================================
# NUMERICAL FEATURES
# =========================================================

NUMERIC_COLUMNS = [

    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore"
]


# =========================================================
# RUN PREPROCESSING
# =========================================================

def run_preprocessing():

    # -----------------------------------------------------
    # 1. LOAD DATASET
    # -----------------------------------------------------

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"\nDataset not found!\n"
            f"Expected location:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully!")

    print(
        "Dataset Shape:",
        df.shape
    )


    # -----------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # -----------------------------------------------------

    required_columns = (
        NUMERIC_COLUMNS
        + [
            "PlacementStatus",
            "Salary Package"
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(missing_columns)
        )


    # -----------------------------------------------------
    # 2. TRAIN / TEST SPLIT
    # -----------------------------------------------------

    train_df, test_df = train_test_split(

        df,

        test_size=0.20,

        random_state=42,

        stratify=df["PlacementStatus"]
    )


    print("\n=================================================")
    print("TRAIN / TEST SPLIT")
    print("=================================================")

    print(
        "Training Data:",
        train_df.shape
    )

    print(
        "Testing Data:",
        test_df.shape
    )


    # -----------------------------------------------------
    # 3. MIN-MAX SCALING
    # -----------------------------------------------------

    print("\nCreating Min-Max scaled data...")

    minmax_scaler = MinMaxScaler()

    train_minmax = train_df.copy()

    test_minmax = test_df.copy()

    train_minmax[NUMERIC_COLUMNS] = (
        minmax_scaler.fit_transform(
            train_df[NUMERIC_COLUMNS]
        )
    )

    test_minmax[NUMERIC_COLUMNS] = (
        minmax_scaler.transform(
            test_df[NUMERIC_COLUMNS]
        )
    )


    # -----------------------------------------------------
    # 4. OUTLIER HANDLING USING IQR
    # -----------------------------------------------------

    print("\nApplying IQR outlier handling...")


    q1 = train_df[
        "Salary Package"
    ].quantile(0.25)


    q3 = train_df[
        "Salary Package"
    ].quantile(0.75)


    iqr = q3 - q1


    lower_limit = q1 - (
        1.5 * iqr
    )


    upper_limit = q3 + (
        1.5 * iqr
    )


    train_outlier = train_df.copy()

    test_outlier = test_df.copy()


    train_outlier[
        "Salary Package"
    ] = train_outlier[
        "Salary Package"
    ].clip(
        lower=lower_limit,
        upper=upper_limit
    )


    test_outlier[
        "Salary Package"
    ] = test_outlier[
        "Salary Package"
    ].clip(
        lower=lower_limit,
        upper=upper_limit
    )


    max_salary_before = float(
        train_df[
            "Salary Package"
        ].max()
    )


    max_salary_after = float(
        train_outlier[
            "Salary Package"
        ].max()
    )


    # -----------------------------------------------------
    # COUNT OUTLIERS
    # -----------------------------------------------------

    outlier_mask = (

        (train_df["Salary Package"] < lower_limit)
        |
        (train_df["Salary Package"] > upper_limit)

    )

    outlier_count = int(
        outlier_mask.sum()
    )


    print(
        "Q1:",
        round(float(q1), 2)
    )

    print(
        "Q3:",
        round(float(q3), 2)
    )

    print(
        "IQR:",
        round(float(iqr), 2)
    )

    print(
        "Upper Limit:",
        round(float(upper_limit), 2)
    )

    print(
        "Detected Outliers:",
        outlier_count
    )


    # -----------------------------------------------------
    # 5. STANDARDIZATION
    # -----------------------------------------------------

    print("\nCreating StandardScaler data...")

    standard_scaler = StandardScaler()

    train_standard = train_outlier.copy()

    test_standard = test_outlier.copy()


    train_standard[NUMERIC_COLUMNS] = (
        standard_scaler.fit_transform(
            train_outlier[
                NUMERIC_COLUMNS
            ]
        )
    )


    test_standard[NUMERIC_COLUMNS] = (
        standard_scaler.transform(
            test_outlier[
                NUMERIC_COLUMNS
            ]
        )
    )


    # -----------------------------------------------------
    # 6. ROBUST SCALING
    # -----------------------------------------------------

    print("\nCreating RobustScaler data...")

    robust_scaler = RobustScaler()

    train_robust = train_outlier.copy()

    test_robust = test_outlier.copy()


    train_robust[NUMERIC_COLUMNS] = (
        robust_scaler.fit_transform(
            train_outlier[
                NUMERIC_COLUMNS
            ]
        )
    )


    test_robust[NUMERIC_COLUMNS] = (
        robust_scaler.transform(
            test_outlier[
                NUMERIC_COLUMNS
            ]
        )
    )


    # =====================================================
    # SAVE CSV FILES
    # =====================================================

    print("\n=================================================")
    print("SAVING CSV OUTPUTS")
    print("=================================================")


    train_minmax.to_csv(
        OUTPUT_DIR /
        "train_minmax.csv",
        index=False
    )


    test_minmax.to_csv(
        OUTPUT_DIR /
        "test_minmax.csv",
        index=False
    )


    train_outlier.to_csv(
        OUTPUT_DIR /
        "train_iqr_outlier_handled.csv",
        index=False
    )


    test_outlier.to_csv(
        OUTPUT_DIR /
        "test_iqr_outlier_handled.csv",
        index=False
    )


    train_standard.to_csv(
        OUTPUT_DIR /
        "train_standardized.csv",
        index=False
    )


    test_standard.to_csv(
        OUTPUT_DIR /
        "test_standardized.csv",
        index=False
    )


    train_robust.to_csv(
        OUTPUT_DIR /
        "train_robust_scaled.csv",
        index=False
    )


    test_robust.to_csv(
        OUTPUT_DIR /
        "test_robust_scaled.csv",
        index=False
    )


    # =====================================================
    # GRAPH 1 - SALARY BEFORE / AFTER OUTLIER HANDLING
    # =====================================================

    print("\nCreating salary outlier comparison graph...")


    plt.figure(
        figsize=(10, 6)
    )

    plt.boxplot(
        [
            train_df["Salary Package"],
            train_outlier["Salary Package"]
        ],
        tick_labels=[
            "Before IQR",
            "After IQR"
        ]
    )


    plt.title(
        "Salary Package Before and After IQR Outlier Handling"
    )

    plt.ylabel(
        "Salary Package"
    )

    plt.tight_layout()


    salary_path = (
        OUTPUT_DIR /
        "salary_iqr_comparison.png"
    )


    plt.savefig(
        salary_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # =====================================================
    # GRAPH 2 - SCALING COMPARISON
    # =====================================================

    print(
        "Creating scaling comparison graph..."
    )


    sample_features = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]


    original_means = [
        train_df[
            feature
        ].mean()
        for feature in sample_features
    ]


    minmax_means = [
        train_minmax[
            feature
        ].mean()
        for feature in sample_features
    ]


    standard_means = [
        train_standard[
            feature
        ].mean()
        for feature in sample_features
    ]


    robust_means = [
        train_robust[
            feature
        ].mean()
        for feature in sample_features
    ]


    plt.figure(
        figsize=(13, 7)
    )


    x = range(
        len(sample_features)
    )


    plt.plot(
        x,
        original_means,
        marker="o",
        label="Original"
    )


    plt.plot(
        x,
        minmax_means,
        marker="o",
        label="Min-Max"
    )


    plt.plot(
        x,
        standard_means,
        marker="o",
        label="Standardized"
    )


    plt.plot(
        x,
        robust_means,
        marker="o",
        label="Robust"
    )


    plt.xticks(
        x,
        sample_features,
        rotation=30
    )


    plt.xlabel(
        "Features"
    )


    plt.ylabel(
        "Mean Value"
    )


    plt.title(
        "Comparison of Scaling Techniques"
    )


    plt.legend()


    plt.tight_layout()


    scaling_path = (
        OUTPUT_DIR /
        "scaling_methods_comparison.png"
    )


    plt.savefig(
        scaling_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # =====================================================
    # GRAPH 3 - CGPA SCALING
    # =====================================================

    print(
        "Creating CGPA scaling graph..."
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.hist(
        train_df[
            "CGPA"
        ],
        bins=30,
        alpha=0.5,
        label="Original"
    )


    plt.hist(
        train_minmax[
            "CGPA"
        ],
        bins=30,
        alpha=0.5,
        label="Min-Max"
    )


    plt.xlabel(
        "CGPA"
    )


    plt.ylabel(
        "Number of Students"
    )


    plt.title(
        "CGPA Distribution Before and After Min-Max Scaling"
    )


    plt.legend()


    plt.tight_layout()


    cgpa_path = (
        OUTPUT_DIR /
        "cgpa_scaling_distribution.png"
    )


    plt.savefig(
        cgpa_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # =====================================================
    # GRAPH 4 - SALARY DISTRIBUTION
    # =====================================================

    print(
        "Creating salary distribution graph..."
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.hist(
        train_df[
            "Salary Package"
        ],
        bins=30,
        alpha=0.6
    )


    plt.axvline(
        upper_limit,
        linestyle="--",
        linewidth=2,
        label="Upper IQR Limit"
    )


    plt.xlabel(
        "Salary Package"
    )


    plt.ylabel(
        "Number of Students"
    )


    plt.title(
        "Salary Package Distribution and IQR Upper Limit"
    )


    plt.legend()


    plt.tight_layout()


    salary_distribution_path = (
        OUTPUT_DIR /
        "salary_distribution_iqr.png"
    )


    plt.savefig(
        salary_distribution_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # =====================================================
    # DASHBOARD RESULTS
    # =====================================================

    results = {

        "total_records":
            len(df),

        "total_features":
            len(df.columns),

        "train_records":
            len(train_df),

        "test_records":
            len(test_df),

        "numeric_features":
            len(NUMERIC_COLUMNS),

        "q1":
            round(float(q1), 2),

        "q3":
            round(float(q3), 2),

        "iqr":
            round(float(iqr), 2),

        "lower_limit":
            round(float(lower_limit), 2),

        "upper_limit":
            round(float(upper_limit), 2),

        "outlier_count":
            outlier_count,

        "max_salary_before":
            round(max_salary_before, 2),

        "max_salary_after":
            round(max_salary_after, 2),

        "minmax_sample":
            train_minmax[
                NUMERIC_COLUMNS
            ].head(3).round(3).to_dict(
                orient="records"
            ),

        "standard_sample":
            train_standard[
                NUMERIC_COLUMNS
            ].head(3).round(3).to_dict(
                orient="records"
            ),

        "robust_sample":
            train_robust[
                NUMERIC_COLUMNS
            ].head(3).round(3).to_dict(
                orient="records"
            )
    }


    # =====================================================
    # REPORT
    # =====================================================

    report_path = (
        OUTPUT_DIR /
        "preprocessing_report.txt"
    )


    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PREPROCESSING REPORT\n"
        )

        file.write(
            "====================\n\n"
        )

        file.write(
            f"Dataset Records: {len(df)}\n"
        )

        file.write(
            f"Dataset Features: {len(df.columns)}\n"
        )

        file.write(
            f"Training Records: {len(train_df)}\n"
        )

        file.write(
            f"Testing Records: {len(test_df)}\n"
        )

        file.write(
            f"Numerical Features: {len(NUMERIC_COLUMNS)}\n\n"
        )

        file.write(
            "IQR OUTLIER HANDLING\n"
        )

        file.write(
            "--------------------\n"
        )

        file.write(
            f"Q1: {q1:.2f}\n"
        )

        file.write(
            f"Q3: {q3:.2f}\n"
        )

        file.write(
            f"IQR: {iqr:.2f}\n"
        )

        file.write(
            f"Lower Limit: {lower_limit:.2f}\n"
        )

        file.write(
            f"Upper Limit: {upper_limit:.2f}\n"
        )

        file.write(
            f"Detected Outliers: {outlier_count}\n"
        )

        file.write(
            f"Maximum Salary Before: {max_salary_before:.2f}\n"
        )

        file.write(
            f"Maximum Salary After: {max_salary_after:.2f}\n\n"
        )

        file.write(
            "SCALING METHODS\n"
        )

        file.write(
            "---------------\n"
        )

        file.write(
            "1. Min-Max Scaling\n"
        )

        file.write(
            "2. Standardization\n"
        )

        file.write(
            "3. Robust Scaling\n"
        )


    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print("\n=================================================")
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=================================================")

    print("\nOutput Directory:")
    print(OUTPUT_DIR)

    print("\nCSV FILES:")
    print("- train_minmax.csv")
    print("- test_minmax.csv")
    print("- train_iqr_outlier_handled.csv")
    print("- test_iqr_outlier_handled.csv")
    print("- train_standardized.csv")
    print("- test_standardized.csv")
    print("- train_robust_scaled.csv")
    print("- test_robust_scaled.csv")

    print("\nIMAGE FILES:")
    print("- salary_iqr_comparison.png")
    print("- scaling_methods_comparison.png")
    print("- cgpa_scaling_distribution.png")
    print("- salary_distribution_iqr.png")

    print("\nREPORT:")
    print("- preprocessing_report.txt")


    return results


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    results = run_preprocessing()

    print("\nPreprocessing finished successfully!")