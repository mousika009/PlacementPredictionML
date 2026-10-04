import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# DECISION TREE CLASSIFICATION
# ============================================================

print("=" * 65)
print("DECISION TREE CLASSIFICATION")
print("=" * 65)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "placement_data.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "decision_tree"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("\nProject Directory:")
print(BASE_DIR)

print("\nOutput Directory:")
print(OUTPUT_DIR)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 65)
print("LOADING DATASET")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 3. TARGET CLEANING
# ============================================================

target = "PlacementStatus"

print("\nOriginal PlacementStatus values:")
print(
    df[target].value_counts(
        dropna=False
    )
)


df[target] = pd.to_numeric(
    df[target],
    errors="coerce"
)


invalid_rows = df[target].isna().sum()

print(
    f"\nInvalid target rows: {invalid_rows}"
)


df = df.dropna(
    subset=[target]
).copy()


df[target] = df[target].astype(int)


print("\nClean PlacementStatus:")
print(
    df[target].value_counts()
)


# ============================================================
# 4. FEATURES
# ============================================================

features = [
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


print("\n" + "=" * 65)
print("FEATURES USED")
print("=" * 65)

for feature in features:
    print("-", feature)


# ============================================================
# 5. CREATE X AND y
# ============================================================

X = df[features].copy()

y = df[target].copy()


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

X = X.fillna(
    X.median(numeric_only=True)
)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 65)
print("DATA SPLIT")
print("=" * 65)

print(
    f"Training Records : {len(X_train)}"
)

print(
    f"Testing Records  : {len(X_test)}"
)


# ============================================================
# 8. CREATE DECISION TREE
# ============================================================

model = DecisionTreeClassifier(

    criterion="gini",

    max_depth=5,

    min_samples_split=10,

    min_samples_leaf=5,

    random_state=42
)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\n" + "=" * 65)
print("TRAINING DECISION TREE")
print("=" * 65)

model.fit(
    X_train,
    y_train
)

print(
    "Training completed successfully."
)


# ============================================================
# 10. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 11. MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 65)
print("MODEL PERFORMANCE")
print("=" * 65)

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


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({

    "Feature": features,

    "Importance":
        model.feature_importances_

})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


print("\n" + "=" * 65)
print("FEATURE IMPORTANCE")
print("=" * 65)

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# 13. FEATURE IMPORTANCE IMAGE
# ============================================================

plt.figure(
    figsize=(11, 7)
)

plt.barh(
    importance_df["Feature"][::-1],
    importance_df["Importance"][::-1]
)

plt.xlabel(
    "Importance",
    fontsize=12
)

plt.ylabel(
    "Features",
    fontsize=12
)

plt.title(
    "Decision Tree - Feature Importance",
    fontsize=16,
    fontweight="bold"
)

plt.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()


importance_png = os.path.join(
    OUTPUT_DIR,
    "feature_importance.png"
)


plt.savefig(
    importance_png,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()


print(
    "\nFeature importance image saved:"
)

print(
    importance_png
)


# ============================================================
# 14. DECISION TREE STRUCTURE IMAGE
# ============================================================

print("\n" + "=" * 65)
print("CREATING DECISION TREE IMAGE")
print("=" * 65)


# IMPORTANT:
# Use a wider figure and high DPI.
# max_depth=5 keeps the tree readable.

fig, ax = plt.subplots(
    figsize=(32, 18)
)


plot_tree(

    model,

    feature_names=features,

    class_names=[
        "Not Placed",
        "Placed"
    ],

    filled=True,

    rounded=True,

    impurity=True,

    proportion=False,

    precision=2,

    fontsize=10,

    ax=ax

)


ax.set_title(
    "Decision Tree - Student Placement Prediction",
    fontsize=24,
    fontweight="bold",
    pad=25
)


plt.tight_layout()


tree_png = os.path.join(
    OUTPUT_DIR,
    "decision_tree_structure.png"
)


plt.savefig(
    tree_png,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)


plt.close()


print(
    "\nDecision Tree image saved:"
)

print(
    tree_png
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


fig, ax = plt.subplots(
    figsize=(8, 7)
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Not Placed",
        "Placed"
    ]
)


disp.plot(
    ax=ax,
    values_format="d"
)


ax.set_title(
    "Decision Tree - Confusion Matrix",
    fontsize=16,
    fontweight="bold"
)


plt.tight_layout()


confusion_png = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)


plt.savefig(
    confusion_png,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)


plt.close()


print(
    "\nConfusion matrix saved:"
)

print(
    confusion_png
)


# ============================================================
# 16. ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(10, 6)
)


sample_size = min(
    100,
    len(y_test)
)


plt.plot(
    range(sample_size),
    y_test.iloc[:sample_size].values,
    marker="o",
    linestyle="-",
    label="Actual"
)


plt.plot(
    range(sample_size),
    y_pred[:sample_size],
    marker="x",
    linestyle="--",
    label="Predicted"
)


plt.xlabel(
    "Test Student Index",
    fontsize=12
)

plt.ylabel(
    "Placement Status",
    fontsize=12
)

plt.yticks(
    [0, 1],
    ["Not Placed", "Placed"]
)

plt.title(
    "Decision Tree - Actual vs Predicted",
    fontsize=16,
    fontweight="bold"
)

plt.legend()

plt.grid(
    alpha=0.25
)


plt.tight_layout()


actual_predicted_png = os.path.join(
    OUTPUT_DIR,
    "actual_vs_predicted.png"
)


plt.savefig(
    actual_predicted_png,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)


plt.close()


print(
    "\nActual vs Predicted image saved:"
)

print(
    actual_predicted_png
)


# ============================================================
# 17. TREE INFORMATION
# ============================================================

print("\n" + "=" * 65)
print("TREE INFORMATION")
print("=" * 65)

print(
    f"Criterion      : {model.criterion}"
)

print(
    f"Maximum Depth  : 5"
)

print(
    f"Actual Depth   : {model.get_depth()}"
)

print(
    f"Leaf Nodes     : {model.get_n_leaves()}"
)

print(
    f"Total Nodes    : {model.tree_.node_count}"
)


# ============================================================
# 18. SAVE TEXT REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "decision_tree_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DECISION TREE CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        f"Dataset Shape: {df.shape}\n"
    )

    file.write(
        f"Training Records: {len(X_train)}\n"
    )

    file.write(
        f"Testing Records: {len(X_test)}\n\n"
    )

    file.write(
        "Features Used:\n"
    )

    for feature in features:

        file.write(
            f"- {feature}\n"
        )

    file.write(
        "\nModel Parameters:\n"
    )

    file.write(
        f"Criterion: {model.criterion}\n"
    )

    file.write(
        "Maximum Depth: 5\n"
    )

    file.write(
        "Minimum Samples Split: 10\n"
    )

    file.write(
        "Minimum Samples Leaf: 5\n"
    )

    file.write(
        f"\nActual Tree Depth: {model.get_depth()}\n"
    )

    file.write(
        f"Leaf Nodes: {model.get_n_leaves()}\n"
    )

    file.write(
        f"Total Nodes: {model.tree_.node_count}\n"
    )

    file.write(
        f"\nAccuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Accuracy Percentage: {accuracy * 100:.2f}%\n"
    )


# ============================================================
# 19. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("DECISION TREE COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nImages saved in:")

print(
    OUTPUT_DIR
)

print("\nPNG files:")

print(
    "1. decision_tree_structure.png"
)

print(
    "2. feature_importance.png"
)

print(
    "3. confusion_matrix.png"
)

print(
    "4. actual_vs_predicted.png"
)

print(
    "\nReport:"
)

print(
    "decision_tree_report.txt"
)

print("\nDone!")