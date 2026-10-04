import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = BASE_DIR / "data" / "placement_data.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "feature_scaling"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("==============================================")
print("FEATURE SCALING - MIN MAX SCALING")
print("==============================================")

print("\nPROJECT ROOT:")
print(BASE_DIR)

print("\nDATA PATH:")
print(DATA_PATH)

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)


# =========================================================
# 1. CHECK DATASET
# =========================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\nDataset not found!\n"
        f"Expected location:\n{DATA_PATH}"
    )


# =========================================================
# 2. LOAD DATASET
# =========================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

print(
    "Dataset Shape:",
    df.shape
)


# =========================================================
# 3. NUMERICAL FEATURES
# =========================================================

numeric_features = [

    "SGPA_Sem1",
    "SGPA_Sem2",
    "SGPA_Sem3",
    "SGPA_Sem4",
    "SGPA_Sem5",
    "SGPA_Sem6",
    "SGPA_Sem7",
    "SGPA_Sem8",

    "CGPA",

    "AttendancePercent",

    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "Publications",

    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore",

    "ExtraCurricular"
]


# =========================================================
# 4. CHECK FEATURES
# =========================================================

missing_features = [
    column
    for column in numeric_features
    if column not in df.columns
]

if missing_features:

    raise ValueError(
        "\nThe following required features "
        "were not found:\n"
        + "\n".join(missing_features)
    )


# =========================================================
# 5. HANDLE MISSING VALUES
# =========================================================

X = df[numeric_features].copy()

X = X.fillna(
    X.median()
)


# =========================================================
# 6. TARGET
# =========================================================

y = df["PlacementStatus"].copy()


print("\n================================")
print("FEATURE / TARGET")
print("================================")

print(
    "Features Shape:",
    X.shape
)

print(
    "Target Shape:",
    y.shape
)


# =========================================================
# 7. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n================================")
print("TRAIN TEST SPLIT")
print("================================")

print(
    "Training Data:",
    X_train.shape
)

print(
    "Testing Data:",
    X_test.shape
)


# =========================================================
# 8. SAVE ORIGINAL DATA FOR VISUALIZATION
# =========================================================

X_train_original = X_train.copy()


# =========================================================
# 9. CREATE MIN MAX SCALER
# =========================================================

scaler = MinMaxScaler()


# =========================================================
# 10. FIT ONLY ON TRAINING DATA
# =========================================================

X_train_scaled = scaler.fit_transform(
    X_train
)


# =========================================================
# 11. TRANSFORM TEST DATA
# =========================================================

X_test_scaled = scaler.transform(
    X_test
)


# =========================================================
# 12. CONVERT TO DATAFRAMES
# =========================================================

X_train_scaled = pd.DataFrame(

    X_train_scaled,

    columns=numeric_features,

    index=X_train.index
)


X_test_scaled = pd.DataFrame(

    X_test_scaled,

    columns=numeric_features,

    index=X_test.index
)


# =========================================================
# 13. DISPLAY SCALED TRAINING DATA
# =========================================================

print("\n================================")
print("SCALED TRAINING DATA")
print("================================")

print(
    X_train_scaled.head()
)


# =========================================================
# 14. DISPLAY SCALED TEST DATA
# =========================================================

print("\n================================")
print("SCALED TEST DATA")
print("================================")

print(
    X_test_scaled.head()
)


# =========================================================
# 15. CHECK MINIMUM VALUES
# =========================================================

print("\n================================")
print("TRAINING DATA MINIMUM")
print("================================")

print(
    X_train_scaled.min().round(2)
)


# =========================================================
# 16. CHECK MAXIMUM VALUES
# =========================================================

print("\n================================")
print("TRAINING DATA MAXIMUM")
print("================================")

print(
    X_train_scaled.max().round(2)
)


# =========================================================
# 17. CGPA COMPARISON
# =========================================================

comparison = pd.DataFrame({

    "Original_CGPA":
        X_train_original["CGPA"].head(10).values,

    "Scaled_CGPA":
        X_train_scaled["CGPA"].head(10).values

})


print("\n================================")
print("CGPA BEFORE AND AFTER SCALING")
print("================================")

print(
    comparison.round(4)
)


# =========================================================
# 18. SAVE CSV FILES
# =========================================================

train_output = (
    OUTPUT_DIR /
    "scaled_training_data.csv"
)

X_train_scaled.to_csv(
    train_output,
    index=False
)


test_output = (
    OUTPUT_DIR /
    "scaled_testing_data.csv"
)

X_test_scaled.to_csv(
    test_output,
    index=False
)


comparison_output = (
    OUTPUT_DIR /
    "cgpa_scaling_comparison.csv"
)

comparison.to_csv(
    comparison_output,
    index=False
)


print("\nCSV files saved successfully.")


# =========================================================
# 19. GRAPH 1
# FEATURE DISTRIBUTION BEFORE SCALING
# =========================================================

print("\nCreating feature distribution before scaling...")


plt.figure(
    figsize=(14, 7)
)

X_train_original.boxplot(
    rot=90
)

plt.title(
    "Feature Distribution Before Min-Max Scaling",
    fontsize=16
)

plt.xlabel(
    "Features"
)

plt.ylabel(
    "Original Values"
)

plt.tight_layout()


before_path = (
    OUTPUT_DIR /
    "feature_distribution_before.png"
)

plt.savefig(
    before_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 20. GRAPH 2
# FEATURE DISTRIBUTION AFTER SCALING
# =========================================================

print(
    "Creating feature distribution after scaling..."
)


plt.figure(
    figsize=(14, 7)
)

X_train_scaled.boxplot(
    rot=90
)

plt.title(
    "Feature Distribution After Min-Max Scaling",
    fontsize=16
)

plt.xlabel(
    "Features"
)

plt.ylabel(
    "Scaled Values (0 - 1)"
)

plt.tight_layout()


after_path = (
    OUTPUT_DIR /
    "feature_distribution_after.png"
)

plt.savefig(
    after_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 21. GRAPH 3
# CGPA BEFORE VS AFTER
# =========================================================

print(
    "Creating CGPA scaling comparison..."
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    range(1, 11),
    comparison["Original_CGPA"],
    marker="o",
    label="Original CGPA"
)

plt.plot(
    range(1, 11),
    comparison["Scaled_CGPA"],
    marker="o",
    label="Scaled CGPA"
)

plt.xlabel(
    "Student Sample"
)

plt.ylabel(
    "CGPA Value"
)

plt.title(
    "CGPA Before and After Min-Max Scaling"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


comparison_path = (
    OUTPUT_DIR /
    "scaling_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 22. GRAPH 4
# FEATURE MINIMUM / MAXIMUM AFTER SCALING
# =========================================================

print(
    "Creating feature range graph..."
)


feature_min = X_train_scaled.min()

feature_max = X_train_scaled.max()


plt.figure(
    figsize=(15, 7)
)

plt.plot(
    range(len(numeric_features)),
    feature_min.values,
    marker="o",
    label="Minimum"
)

plt.plot(
    range(len(numeric_features)),
    feature_max.values,
    marker="o",
    label="Maximum"
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.axhline(
    1,
    linestyle="--",
    linewidth=1
)

plt.xticks(
    range(len(numeric_features)),
    numeric_features,
    rotation=90
)

plt.xlabel(
    "Features"
)

plt.ylabel(
    "Scaled Value"
)

plt.title(
    "Feature Range After Min-Max Scaling"
)

plt.legend()

plt.tight_layout()


range_path = (
    OUTPUT_DIR /
    "feature_range.png"
)

plt.savefig(
    range_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 23. SAVE SCALING REPORT
# =========================================================

report_path = (
    OUTPUT_DIR /
    "scaling_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "FEATURE SCALING REPORT\n"
    )

    file.write(
        "======================\n\n"
    )

    file.write(
        f"Dataset Records: {len(df)}\n"
    )

    file.write(
        f"Training Records: {len(X_train)}\n"
    )

    file.write(
        f"Testing Records: {len(X_test)}\n"
    )

    file.write(
        f"Features Scaled: {len(numeric_features)}\n"
    )

    file.write(
        "Scaling Method: Min-Max Scaling\n\n"
    )

    file.write(
        "Formula concept:\n"
    )

    file.write(
        "Values are transformed to a range between 0 and 1.\n\n"
    )

    file.write(
        "Features:\n"
    )

    for feature in numeric_features:

        file.write(
            f"- {feature}\n"
        )


# =========================================================
# 24. FINAL OUTPUT
# =========================================================

print("\n==============================================")
print("FEATURE SCALING COMPLETED SUCCESSFULLY")
print("==============================================")


print("\nOutput Directory:")

print(
    OUTPUT_DIR
)


print("\nCSV FILES:")

print(
    "1. scaled_training_data.csv"
)

print(
    "2. scaled_testing_data.csv"
)

print(
    "3. cgpa_scaling_comparison.csv"
)


print("\nIMAGE FILES:")

print(
    "1. feature_distribution_before.png"
)

print(
    "2. feature_distribution_after.png"
)

print(
    "3. scaling_comparison.png"
)

print(
    "4. feature_range.png"
)


print("\nREPORT:")

print(
    "scaling_report.txt"
)


print("\n==============================================")
print("ALL FEATURE SCALING OUTPUTS GENERATED")
print("==============================================")