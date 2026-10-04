import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# PCA - PRINCIPAL COMPONENT ANALYSIS
# Placement Prediction 50K Dataset
# ============================================================

print("=" * 60)
print("          PRINCIPAL COMPONENT ANALYSIS")
print("=" * 60)


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
    "pca"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)


# ============================================================
# 3. SELECT NUMERICAL FEATURES
# ============================================================

exclude_columns = [
    "StudentID",
    "PlacementStatus",
    "IsAnomaly",
    "Salary Package"
]

feature_cols = [
    column
    for column in df.select_dtypes(
        include=np.number
    ).columns
    if column not in exclude_columns
]


if len(feature_cols) == 0:

    raise ValueError(
        "No numerical features available for PCA."
    )


print("\n================================")
print("FEATURES USED FOR PCA")
print("================================")

for feature in feature_cols:

    print("-", feature)

print(
    "\nNumber of features:",
    len(feature_cols)
)


# ============================================================
# 4. CREATE FEATURE MATRIX
# ============================================================

X = df[
    feature_cols
].copy()


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

print("\n================================")
print("MISSING VALUE HANDLING")
print("================================")

missing_before = int(
    X.isna().sum().sum()
)

print(
    "Missing values before:",
    missing_before
)


imputer = SimpleImputer(
    strategy="median"
)

X_imputed = imputer.fit_transform(
    X
)


missing_after = int(
    np.isnan(X_imputed).sum()
)

print(
    "Missing values after:",
    missing_after
)


# ============================================================
# 6. STANDARDIZATION
# ============================================================

print("\n================================")
print("FEATURE STANDARDIZATION")
print("================================")


scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_imputed
)

print(
    "Standardization completed."
)


# ============================================================
# 7. PCA
# ============================================================

print("\n================================")
print("PCA ANALYSIS")
print("================================")


pca = PCA()

X_pca = pca.fit_transform(
    X_scaled
)


explained_variance = (
    pca.explained_variance_ratio_
)

cumulative_variance = np.cumsum(
    explained_variance
)


# ============================================================
# 8. DISPLAY VARIANCE
# ============================================================

print("\nVariance explained:")

for i, variance in enumerate(
    explained_variance
):

    print(
        f"PC{i + 1}: "
        f"{variance * 100:.2f}% "
        f"(Cumulative: "
        f"{cumulative_variance[i] * 100:.2f}%)"
    )


# ============================================================
# 9. VARIANCE THRESHOLDS
# ============================================================

print("\n================================")
print("VARIANCE THRESHOLDS")
print("================================")


thresholds = [
    0.80,
    0.90,
    0.95
]

threshold_results = {}


for threshold in thresholds:

    components_required = (
        np.searchsorted(
            cumulative_variance,
            threshold
        ) + 1
    )

    threshold_results[
        threshold
    ] = components_required

    print(
        f"{int(threshold * 100)}% variance "
        f"requires "
        f"{components_required} components"
    )


# ============================================================
# 10. SCREE PLOT
# ============================================================

print("\nCreating scree plot...")


components = range(
    1,
    len(explained_variance) + 1
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    components,
    explained_variance * 100,
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Principal Component"
)

plt.ylabel(
    "Variance Explained (%)"
)

plt.title(
    "PCA Scree Plot"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


scree_path = os.path.join(
    OUTPUT_DIR,
    "pca_scree_plot.png"
)


plt.savefig(
    scree_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    scree_path
)


# ============================================================
# 11. CUMULATIVE VARIANCE PLOT
# ============================================================

print(
    "\nCreating cumulative variance plot..."
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    components,
    cumulative_variance * 100,
    marker="o",
    linewidth=2
)


plt.axhline(
    80,
    linestyle="--",
    label="80%"
)

plt.axhline(
    90,
    linestyle="--",
    label="90%"
)

plt.axhline(
    95,
    linestyle="--",
    label="95%"
)


plt.xlabel(
    "Number of Components"
)

plt.ylabel(
    "Cumulative Variance Explained (%)"
)

plt.title(
    "PCA Cumulative Variance Explained"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


cumulative_path = os.path.join(
    OUTPUT_DIR,
    "pca_cumulative_variance.png"
)


plt.savefig(
    cumulative_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    cumulative_path
)


# ============================================================
# 12. PCA 2D PROJECTION
# ============================================================

print(
    "\nCreating PCA 2D projection..."
)


X_2d = X_pca[:, :2]


plt.figure(
    figsize=(11, 8)
)


if "PlacementStatus" in df.columns:

    placement = pd.to_numeric(
        df["PlacementStatus"],
        errors="coerce"
    )

    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=placement,
        cmap="coolwarm",
        s=10,
        alpha=0.55
    )

    plt.colorbar(
        scatter,
        label="Placement Status"
    )

else:

    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        s=10,
        alpha=0.55
    )


plt.xlabel(
    f"PC1 "
    f"({explained_variance[0] * 100:.2f}% variance)"
)

plt.ylabel(
    f"PC2 "
    f"({explained_variance[1] * 100:.2f}% variance)"
)

plt.title(
    "PCA 2D Projection - Student Placement Dataset"
)

plt.grid(
    alpha=0.2
)

plt.tight_layout()


pca_2d_path = os.path.join(
    OUTPUT_DIR,
    "pca_2d_projection.png"
)


plt.savefig(
    pca_2d_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    pca_2d_path
)


# ============================================================
# 13. PCA LOADINGS IMAGE
# ============================================================

print("\nCreating PCA loadings plot...")


loadings = pd.DataFrame(
    pca.components_.T,
    columns=[
        f"PC{i + 1}"
        for i in range(
            len(feature_cols)
        )
    ],
    index=feature_cols
)


# Top features for PC1
pc1_top = (
    loadings["PC1"]
    .abs()
    .sort_values(
        ascending=False
    )
    .head(10)
)


# Top features for PC2
pc2_top = (
    loadings["PC2"]
    .abs()
    .sort_values(
        ascending=False
    )
    .head(10)
)


# ------------------------------------------------------------
# PC1 LOADINGS
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 7)
)


pc1_values = loadings.loc[
    pc1_top.index,
    "PC1"
].sort_values()


plt.barh(
    pc1_values.index,
    pc1_values.values
)

plt.xlabel(
    "Loading"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "PCA - Top Feature Loadings for PC1"
)

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()


pc1_loadings_path = os.path.join(
    OUTPUT_DIR,
    "pca_pc1_loadings.png"
)


plt.savefig(
    pc1_loadings_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# PC2 LOADINGS
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 7)
)


pc2_values = loadings.loc[
    pc2_top.index,
    "PC2"
].sort_values()


plt.barh(
    pc2_values.index,
    pc2_values.values
)

plt.xlabel(
    "Loading"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "PCA - Top Feature Loadings for PC2"
)

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()


pc2_loadings_path = os.path.join(
    OUTPUT_DIR,
    "pca_pc2_loadings.png"
)


plt.savefig(
    pc2_loadings_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved PC1 loadings:"
)

print(
    pc1_loadings_path
)

print(
    "Saved PC2 loadings:"
)

print(
    pc2_loadings_path
)


# ============================================================
# 14. PCA REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "pca_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PRINCIPAL COMPONENT ANALYSIS REPORT\n"
    )

    file.write(
        "=" * 55 + "\n\n"
    )

    file.write(
        f"Dataset Shape: {df.shape}\n"
    )

    file.write(
        f"Number of Input Features: "
        f"{len(feature_cols)}\n\n"
    )


    file.write(
        "Features Used:\n"
    )

    for feature in feature_cols:

        file.write(
            f"- {feature}\n"
        )


    file.write(
        "\nVariance Explained:\n"
    )

    for i, variance in enumerate(
        explained_variance
    ):

        file.write(
            f"PC{i + 1}: "
            f"{variance * 100:.2f}% "
            f"(Cumulative: "
            f"{cumulative_variance[i] * 100:.2f}%)\n"
        )


    file.write(
        "\nVariance Thresholds:\n"
    )

    for threshold, count in (
        threshold_results.items()
    ):

        file.write(
            f"{int(threshold * 100)}%: "
            f"{count} components\n"
        )


    file.write(
        "\nPC1 + PC2 Variance: "
        f"{cumulative_variance[1] * 100:.2f}%\n"
    )


    file.write(
        "\nTop PC1 Features:\n"
    )

    for feature in pc1_top.index:

        file.write(
            f"{feature}: "
            f"{loadings.loc[feature, 'PC1']:.4f}\n"
        )


    file.write(
        "\nTop PC2 Features:\n"
    )

    for feature in pc2_top.index:

        file.write(
            f"{feature}: "
            f"{loadings.loc[feature, 'PC2']:.4f}\n"
        )


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("PCA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nPNG files generated:")

print(
    "- pca_scree_plot.png"
)

print(
    "- pca_cumulative_variance.png"
)

print(
    "- pca_2d_projection.png"
)

print(
    "- pca_pc1_loadings.png"
)

print(
    "- pca_pc2_loadings.png"
)

print(
    "\nReport:"
)

print(
    "- pca_report.txt"
)

print("\n" + "=" * 60)