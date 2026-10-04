# ============================================================
# t-SNE DIMENSIONALITY REDUCTION
# Placement Prediction Dataset
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
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
    "tsne"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

MAX_SAMPLES = 5000

PERPLEXITY = 30

N_COMPONENTS = 2

N_ITER = 1000


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("                 t-SNE DIMENSIONALITY REDUCTION")
print("=" * 65)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset Shape: {df.shape}"
)


# ============================================================
# SAMPLE DATA
# ============================================================

if len(df) > MAX_SAMPLES:

    print(
        f"\nDataset contains {len(df)} records."
    )

    print(
        f"Using {MAX_SAMPLES} records for t-SNE."
    )

    df_sample = df.sample(
        n=MAX_SAMPLES,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

else:

    df_sample = df.copy().reset_index(
        drop=True
    )


print(
    f"t-SNE Input Shape: {df_sample.shape}"
)


# ============================================================
# REMOVE COMPLETELY EMPTY COLUMNS
# ============================================================

df_sample = df_sample.dropna(
    axis=1,
    how="all"
)


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

numeric_columns = (
    df_sample
    .select_dtypes(include=np.number)
    .columns
    .tolist()
)

categorical_columns = (
    df_sample
    .select_dtypes(exclude=np.number)
    .columns
    .tolist()
)


# Do not use identifier / target columns
exclude_columns = [
    "StudentID",
    "PlacementStatus",
    "IsAnomaly",
    "Salary Package"
]


numeric_columns = [
    column
    for column in numeric_columns
    if column not in exclude_columns
]

categorical_columns = [
    column
    for column in categorical_columns
    if column not in exclude_columns
]


print("\nNumerical Features:")
print(len(numeric_columns))

print("\nCategorical Features:")
print(len(categorical_columns))


# ============================================================
# NUMERICAL FEATURES
# ============================================================

if numeric_columns:

    X_numeric = df_sample[
        numeric_columns
    ].copy()

    imputer = SimpleImputer(
        strategy="median"
    )

    X_numeric = imputer.fit_transform(
        X_numeric
    )

    scaler = StandardScaler()

    X_numeric = scaler.fit_transform(
        X_numeric
    )

else:

    X_numeric = np.empty(
        (len(df_sample), 0)
    )


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

if categorical_columns:

    X_categorical = pd.get_dummies(
        df_sample[
            categorical_columns
        ],
        drop_first=False,
        dtype=float
    )

    X_categorical = X_categorical.fillna(
        0
    )

    X_categorical = X_categorical.values

else:

    X_categorical = np.empty(
        (len(df_sample), 0)
    )


# ============================================================
# COMBINE FEATURES
# ============================================================

if (
    X_numeric.shape[1] > 0
    and
    X_categorical.shape[1] > 0
):

    X = np.hstack(
        [
            X_numeric,
            X_categorical
        ]
    )

elif X_numeric.shape[1] > 0:

    X = X_numeric

elif X_categorical.shape[1] > 0:

    X = X_categorical

else:

    raise ValueError(
        "No usable features found."
    )


print(
    f"\nFeature Matrix Shape: {X.shape}"
)


# ============================================================
# PCA PREPROCESSING
# ============================================================

if X.shape[1] > 20:

    print(
        "\nApplying PCA preprocessing..."
    )

    n_pca_components = min(
        20,
        X.shape[1],
        X.shape[0] - 1
    )

    pca = PCA(
        n_components=n_pca_components,
        random_state=RANDOM_STATE
    )

    X_reduced = pca.fit_transform(
        X
    )

    explained_variance = (
        pca.explained_variance_ratio_.sum()
        * 100
    )

    print(
        f"PCA Components: {n_pca_components}"
    )

    print(
        f"Variance Retained: "
        f"{explained_variance:.2f}%"
    )

else:

    X_reduced = X

    explained_variance = 100


# ============================================================
# RUN t-SNE
# ============================================================

print("\n" + "=" * 65)
print("Running t-SNE...")
print("=" * 65)

tsne = TSNE(
    n_components=N_COMPONENTS,
    perplexity=PERPLEXITY,
    init="pca",
    random_state=RANDOM_STATE,
    learning_rate="auto",
    max_iter=N_ITER
)

embedding = tsne.fit_transform(
    X_reduced
)


print(
    "\nt-SNE completed successfully."
)


# ============================================================
# CREATE RESULT DATAFRAME
# ============================================================

tsne_df = pd.DataFrame(
    embedding,
    columns=[
        "TSNE1",
        "TSNE2"
    ]
)


# Add PlacementStatus ONLY FOR VISUALIZATION
if "PlacementStatus" in df_sample.columns:

    tsne_df[
        "PlacementStatus"
    ] = pd.to_numeric(
        df_sample[
            "PlacementStatus"
        ],
        errors="coerce"
    ).values


# ============================================================
# IMAGE 1
# BASIC t-SNE PLOT
# ============================================================

print(
    "\nCreating t-SNE visualization..."
)

plt.figure(
    figsize=(12, 8)
)

plt.scatter(
    tsne_df["TSNE1"],
    tsne_df["TSNE2"],
    s=25,
    alpha=0.65
)

plt.title(
    "t-SNE 2D Visualization",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "t-SNE Component 1",
    fontsize=12
)

plt.ylabel(
    "t-SNE Component 2",
    fontsize=12
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()


tsne_plot = os.path.join(
    OUTPUT_DIR,
    "tsne_2d_projection.png"
)

plt.savefig(
    tsne_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:"
)

print(
    tsne_plot
)


# ============================================================
# IMAGE 2
# t-SNE BY PLACEMENT STATUS
# ============================================================

if "PlacementStatus" in tsne_df.columns:

    print(
        "\nCreating placement status visualization..."
    )

    plt.figure(
        figsize=(12, 8)
    )

    placed = (
        tsne_df[
            tsne_df["PlacementStatus"] == 1
        ]
    )

    not_placed = (
        tsne_df[
            tsne_df["PlacementStatus"] == 0
        ]
    )

    plt.scatter(
        not_placed["TSNE1"],
        not_placed["TSNE2"],
        s=25,
        alpha=0.55,
        label="Not Placed"
    )

    plt.scatter(
        placed["TSNE1"],
        placed["TSNE2"],
        s=25,
        alpha=0.55,
        label="Placed"
    )

    plt.title(
        "t-SNE Visualization by Placement Status",
        fontsize=18,
        fontweight="bold"
    )

    plt.xlabel(
        "t-SNE Component 1",
        fontsize=12
    )

    plt.ylabel(
        "t-SNE Component 2",
        fontsize=12
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()


    placement_plot = os.path.join(
        OUTPUT_DIR,
        "tsne_placement_status.png"
    )

    plt.savefig(
        placement_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        placement_plot
    )


# ============================================================
# IMAGE 3
# t-SNE SETTINGS / INFORMATION
# ============================================================

print(
    "\nCreating t-SNE information image..."
)

fig = plt.figure(
    figsize=(10, 7)
)

plt.axis("off")

information = f"""
t-SNE ANALYSIS

Dataset Records       : {len(df):,}
Samples Used          : {len(df_sample):,}

Original Features     : {X.shape[1]}
PCA Features          : {X_reduced.shape[1]}

Perplexity             : {PERPLEXITY}
Iterations             : {N_ITER}
Components             : {N_COMPONENTS}
Random State           : {RANDOM_STATE}

PCA Variance Retained  : {explained_variance:.2f}%

Purpose:
t-SNE reduces high-dimensional student
data into a 2D representation.

Similar student profiles tend to appear
closer together in the visualization.
"""

plt.text(
    0.05,
    0.95,
    information,
    transform=fig.transFigure,
    fontsize=14,
    verticalalignment="top",
    family="DejaVu Sans"
)

plt.tight_layout()


info_plot = os.path.join(
    OUTPUT_DIR,
    "tsne_analysis_information.png"
)

plt.savefig(
    info_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    info_plot
)


# ============================================================
# FINAL TERMINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("t-SNE ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nPNG FILES GENERATED:")

print(
    "1. tsne_2d_projection.png"
)

if "PlacementStatus" in tsne_df.columns:

    print(
        "2. tsne_placement_status.png"
    )

print(
    "3. tsne_analysis_information.png"
)

print(
    "\nOutput Folder:"
)

print(
    OUTPUT_DIR
)

print("\nNo CSV files were generated.")