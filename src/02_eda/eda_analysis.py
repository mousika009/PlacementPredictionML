import os
import sys

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import seaborn as sns

# =========================================================
# PROJECT ROOT
# =========================================================

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


# =========================================================
# OUTPUT DIRECTORIES
# =========================================================

PLOTS_DIR = config.PLOTS_DIR
REPORTS_DIR = config.REPORTS_DIR

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


print("=" * 60)
print("EDA ANALYSIS")
print("=" * 60)

print("\nProject Root:")
print(PROJECT_ROOT)

print("\nEDA Plot Directory:")
print(PLOTS_DIR)

print("\nEDA Report Directory:")
print(REPORTS_DIR)


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading dataset...")

if os.path.exists(config.CLEANED_DATA_PATH):

    df = pd.read_csv(
        config.CLEANED_DATA_PATH
    )

    print("Loaded cleaned dataset.")

else:

    df = pd.read_csv(
        config.RAW_DATA_PATH
    )

    print("Cleaned dataset not found.")
    print("Loaded raw dataset.")


print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# =========================================================
# BASIC INFORMATION
# =========================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\nNumeric Columns:")
print(numeric_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# =========================================================
# HELPER FUNCTION
# =========================================================

def save_plot(filename):

    path = os.path.join(
        PLOTS_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", path)


# =========================================================
# 1. PLACEMENT STATUS DISTRIBUTION
# =========================================================

if "PlacementStatus" in df.columns:

    plt.figure(figsize=(8, 6))

    sns.countplot(
        data=df,
        x="PlacementStatus"
    )

    plt.title(
        "Placement Status Distribution",
        fontsize=16
    )

    plt.xlabel("Placement Status")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    save_plot(
        "placement_status_distribution.png"
    )


# =========================================================
# 2. CGPA DISTRIBUTION
# =========================================================

if "CGPA" in df.columns:

    plt.figure(figsize=(9, 6))

    sns.histplot(
        data=df,
        x="CGPA",
        bins=30,
        kde=True
    )

    plt.title(
        "CGPA Distribution",
        fontsize=16
    )

    plt.xlabel("CGPA")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    save_plot(
        "cgpa_distribution.png"
    )


# =========================================================
# 3. ATTENDANCE DISTRIBUTION
# =========================================================

if "AttendancePercent" in df.columns:

    plt.figure(figsize=(9, 6))

    sns.histplot(
        data=df,
        x="AttendancePercent",
        bins=30,
        kde=True
    )

    plt.title(
        "Attendance Percentage Distribution",
        fontsize=16
    )

    plt.xlabel("Attendance Percentage")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    save_plot(
        "attendance_distribution.png"
    )


# =========================================================
# 4. CGPA VS PLACEMENT
# =========================================================

if (
    "CGPA" in df.columns
    and "PlacementStatus" in df.columns
):

    plt.figure(figsize=(9, 6))

    sns.boxplot(
        data=df,
        x="PlacementStatus",
        y="CGPA"
    )

    plt.title(
        "CGPA vs Placement Status",
        fontsize=16
    )

    plt.xlabel("Placement Status")
    plt.ylabel("CGPA")

    plt.tight_layout()

    save_plot(
        "cgpa_vs_placement.png"
    )


# =========================================================
# 5. ATTENDANCE VS PLACEMENT
# =========================================================

if (
    "AttendancePercent" in df.columns
    and "PlacementStatus" in df.columns
):

    plt.figure(figsize=(9, 6))

    sns.boxplot(
        data=df,
        x="PlacementStatus",
        y="AttendancePercent"
    )

    plt.title(
        "Attendance vs Placement Status",
        fontsize=16
    )

    plt.xlabel("Placement Status")
    plt.ylabel("Attendance Percentage")

    plt.tight_layout()

    save_plot(
        "attendance_vs_placement.png"
    )


# =========================================================
# 6. INTERNSHIPS VS PLACEMENT
# =========================================================

if (
    "Internships" in df.columns
    and "PlacementStatus" in df.columns
):

    plt.figure(figsize=(9, 6))

    sns.countplot(
        data=df,
        x="Internships",
        hue="PlacementStatus"
    )

    plt.title(
        "Internships vs Placement Status",
        fontsize=16
    )

    plt.xlabel("Number of Internships")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    save_plot(
        "internships_vs_placement.png"
    )


# =========================================================
# 7. PROJECTS VS PLACEMENT
# =========================================================

if (
    "Projects" in df.columns
    and "PlacementStatus" in df.columns
):

    plt.figure(figsize=(9, 6))

    sns.countplot(
        data=df,
        x="Projects",
        hue="PlacementStatus"
    )

    plt.title(
        "Projects vs Placement Status",
        fontsize=16
    )

    plt.xlabel("Number of Projects")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    save_plot(
        "projects_vs_placement.png"
    )


# =========================================================
# 8. APTITUDE SCORE DISTRIBUTION
# =========================================================

if "AptitudeTestScore" in df.columns:

    plt.figure(figsize=(9, 6))

    sns.histplot(
        data=df,
        x="AptitudeTestScore",
        bins=30,
        kde=True
    )

    plt.title(
        "Aptitude Test Score Distribution",
        fontsize=16
    )

    plt.xlabel("Aptitude Test Score")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    save_plot(
        "aptitude_score_distribution.png"
    )


# =========================================================
# 9. CODING SCORE DISTRIBUTION
# =========================================================

if "CodingTestScore" in df.columns:

    plt.figure(figsize=(9, 6))

    sns.histplot(
        data=df,
        x="CodingTestScore",
        bins=30,
        kde=True
    )

    plt.title(
        "Coding Test Score Distribution",
        fontsize=16
    )

    plt.xlabel("Coding Test Score")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    save_plot(
        "coding_score_distribution.png"
    )


# =========================================================
# 10. SKILLS VS PLACEMENT
# =========================================================

skill_columns = [
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore"
]

available_skills = [
    col
    for col in skill_columns
    if col in df.columns
]

if (
    len(available_skills) > 0
    and "PlacementStatus" in df.columns
):

    placement_means = (
        df.groupby("PlacementStatus")[
            available_skills
        ].mean()
    )

    placement_means.T.plot(
        kind="bar",
        figsize=(11, 7)
    )

    plt.title(
        "Average Skill Scores by Placement Status",
        fontsize=16
    )

    plt.xlabel("Skill")
    plt.ylabel("Average Score")

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.legend(
        title="Placement Status"
    )

    plt.tight_layout()

    save_plot(
        "skills_vs_placement.png"
    )


# =========================================================
# 11. CORRELATION HEATMAP
# =========================================================

if len(numeric_columns) >= 2:

    correlation = df[
        numeric_columns
    ].corr()

    plt.figure(
        figsize=(16, 12)
    )

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        linewidths=0.3
    )

    plt.title(
        "Correlation Heatmap",
        fontsize=18
    )

    plt.tight_layout()

    save_plot(
        "correlation_heatmap.png"
    )


# =========================================================
# 12. TOP NUMERIC FEATURES
# =========================================================

if (
    "PlacementStatus" in df.columns
    and len(numeric_columns) > 1
):

    correlations = (
        df[numeric_columns]
        .corr()["PlacementStatus"]
        .drop("PlacementStatus", errors="ignore")
        .abs()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(
        figsize=(10, 7)
    )

    correlations.sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Top Features Correlated with Placement",
        fontsize=16
    )

    plt.xlabel(
        "Absolute Correlation"
    )

    plt.ylabel(
        "Feature"
    )

    plt.tight_layout()

    save_plot(
        "top_placement_correlations.png"
    )


# =========================================================
# REPORT
# =========================================================

report_path = config.EDA_REPORT_PATH

print("\nCreating EDA report...")

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PLACEMENT PREDICTION - EDA REPORT\n"
    )

    f.write(
        "=" * 50 + "\n\n"
    )

    f.write(
        f"Dataset Rows: {df.shape[0]}\n"
    )

    f.write(
        f"Dataset Columns: {df.shape[1]}\n\n"
    )

    f.write(
        "NUMERIC COLUMNS\n"
    )

    f.write(
        "-" * 30 + "\n"
    )

    for column in numeric_columns:

        f.write(
            f"{column}\n"
        )

    f.write(
        "\nCATEGORICAL COLUMNS\n"
    )

    f.write(
        "-" * 30 + "\n"
    )

    for column in categorical_columns:

        f.write(
            f"{column}\n"
        )

    f.write(
        "\nMISSING VALUES\n"
    )

    f.write(
        "-" * 30 + "\n"
    )

    missing_values = df.isnull().sum()

    for column, count in missing_values.items():

        if count > 0:

            f.write(
                f"{column}: {count}\n"
            )

    f.write(
        "\nDUPLICATE ROWS\n"
    )

    f.write(
        "-" * 30 + "\n"
    )

    f.write(
        f"{df.duplicated().sum()}\n"
    )

    f.write(
        "\nDESCRIPTIVE STATISTICS\n"
    )

    f.write(
        "-" * 30 + "\n"
    )

    f.write(
        df.describe(
            include="all"
        ).to_string()
    )

    f.write(
        "\n\nGENERATED PLOTS\n"
    )

    f.write(
        "-" * 30 + "\n"
    )

    if os.path.exists(PLOTS_DIR):

        for filename in sorted(
            os.listdir(PLOTS_DIR)
        ):

            if filename.lower().endswith(".png"):

                f.write(
                    f"{filename}\n"
                )


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n" + "=" * 60)
print("EDA ANALYSIS COMPLETED")
print("=" * 60)

print("\nPlots saved in:")
print(PLOTS_DIR)

print("\nReport saved in:")
print(report_path)

print("\nGenerated plot files:")

for filename in sorted(
    os.listdir(PLOTS_DIR)
):

    if filename.lower().endswith(".png"):

        print(
            "-",
            filename
        )

print("\nDone!")