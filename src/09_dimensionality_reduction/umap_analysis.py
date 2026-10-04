# ============================================================
# UMAP DIMENSIONALITY REDUCTION
# Placement Prediction Dataset
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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
    "umap"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

MAX_SAMPLES = 10000

N_NEIGHBORS = 15

MIN_DIST = 0.1

N_COMPONENTS = 2


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("UMAP DIMENSIONALITY REDUCTION")
print("=" * 65)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)


# ============================================================
# SAMPLE DATA
# ============================================================

if len(df) > MAX_SAMPLES:

    print(
        f"\nDataset contains {len(df)} records."
    )

    print(
        f"Using {MAX_SAMPLES} records for UMAP."
    )

    df_sample = df.sample(
        n=MAX_SAMPLES,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

else:

    df_sample = df.copy().reset_index(drop=True)


print(
    "UMAP Input Shape:",
    df_sample.shape
)


# ============================================================
# REMOVE COMPLETELY EMPTY COLUMNS
# ============================================================

df_sample = df_sample.dropna(
    axis=1,
    how="all"
)


# ============================================================
# SELECT FEATURES
# ============================================================

# Do not use identifiers or target-related columns
# for dimensionality reduction.

exclude_columns = [
    "StudentID",
    "PlacementStatus",
    "IsAnomaly",
    "Salary Package"
]


feature_df = df_sample.drop(
    columns=[
        column
        for column in exclude_columns
        if column in df_sample.columns
    ],
    errors="ignore"
)


numeric_columns = feature_df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = feature_df.select_dtypes(
    exclude=np.number
).columns.tolist()


print(
    "\nNumerical Features:",
    len(numeric_columns)
)

print(
    "Categorical Features:",
    len(categorical_columns)
)


# ============================================================
# NUMERICAL FEATURES
# ============================================================

if len(numeric_columns) > 0:

    X_numeric = feature_df[
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
        (
            len(df_sample),
            0
        )
    )


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

if len(categorical_columns) > 0:

    X_categorical = pd.get_dummies(
        feature_df[
            categorical_columns
        ],
        drop_first=False,
        dtype=float
    )

    X_categorical = X_categorical.fillna(0)

    X_categorical = X_categorical.values

else:

    X_categorical = np.empty(
        (
            len(df_sample),
            0
        )
    )


# ============================================================
# COMBINE FEATURES
# ============================================================

if (
    X_numeric.shape[1] > 0
    and X_categorical.shape[1] > 0
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
        "No usable features found for UMAP."
    )


print(
    "\nFeature Matrix Shape:",
    X.shape
)


# ============================================================
# RUN UMAP
# ============================================================

print("\n" + "=" * 65)
print("RUNNING UMAP")
print("=" * 65)

reducer = umap.UMAP(
    n_neighbors=N_NEIGHBORS,
    min_dist=MIN_DIST,
    n_components=N_COMPONENTS,
    random_state=RANDOM_STATE
)


embedding = reducer.fit_transform(
    X
)


print(
    "\nUMAP completed successfully."
)


# ============================================================
# CREATE UMAP DATAFRAME
# ============================================================

umap_df = pd.DataFrame(
    embedding,
    columns=[
        "UMAP1",
        "UMAP2"
    ]
)


# ============================================================
# PLOT 1 — BASIC UMAP
# ============================================================

print(
    "\nCreating UMAP 2D visualization..."
)

plt.figure(
    figsize=(12, 8)
)

sns.scatterplot(
    data=umap_df,
    x="UMAP1",
    y="UMAP2",
    s=35,
    alpha=0.7,
    edgecolor=None
)

plt.title(
    "UMAP 2D Visualization",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "UMAP Component 1",
    fontsize=12
)

plt.ylabel(
    "UMAP Component 2",
    fontsize=12
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()


umap_plot = os.path.join(
    OUTPUT_DIR,
    "umap_2d_projection.png"
)

plt.savefig(
    umap_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    umap_plot
)


# ============================================================
# PLOT 2 — UMAP WITH PLACEMENT STATUS
# ============================================================

if "PlacementStatus" in df_sample.columns:

    print(
        "\nCreating placement-based UMAP..."
    )

    plot_df = umap_df.copy()

    placement_values = pd.to_numeric(
        df_sample["PlacementStatus"],
        errors="coerce"
    )

    plot_df["PlacementStatus"] = (
        placement_values.values
    )

    plot_df = plot_df.dropna(
        subset=["PlacementStatus"]
    )

    plt.figure(
        figsize=(12, 8)
    )

    sns.scatterplot(
        data=plot_df,
        x="UMAP1",
        y="UMAP2",
        hue="PlacementStatus",
        palette="viridis",
        s=40,
        alpha=0.75
    )

    plt.title(
        "UMAP Visualization by Placement Status",
        fontsize=18,
        fontweight="bold"
    )

    plt.xlabel(
        "UMAP Component 1"
    )

    plt.ylabel(
        "UMAP Component 2"
    )

    plt.legend(
        title="Placement Status"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    placement_plot = os.path.join(
        OUTPUT_DIR,
        "umap_placement_status.png"
    )

    plt.savefig(
        placement_plot,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        placement_plot
    )


# ============================================================
# PLOT 3 — DIFFERENT n_neighbors
# ============================================================

print(
    "\nCreating n_neighbors comparison..."
)

neighbor_values = [
    5,
    15,
    50
]

figures_created = []

for neighbors in neighbor_values:

    reducer_temp = umap.UMAP(
        n_neighbors=neighbors,
        min_dist=MIN_DIST,
        n_components=2,
        random_state=RANDOM_STATE
    )

    embedding_temp = reducer_temp.fit_transform(
        X
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.scatter(
        embedding_temp[:, 0],
        embedding_temp[:, 1],
        s=20,
        alpha=0.65
    )

    plt.title(
        f"UMAP - n_neighbors = {neighbors}",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "UMAP Component 1"
    )

    plt.ylabel(
        "UMAP Component 2"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        f"umap_neighbors_{neighbors}.png"
    )

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    figures_created.append(path)

    print(
        "Saved:",
        path
    )


# ============================================================
# PLOT 4 — DIFFERENT MIN_DIST
# ============================================================

print(
    "\nCreating min_dist comparison..."
)

min_dist_values = [
    0.0,
    0.1,
    0.5
]

for min_dist_value in min_dist_values:

    reducer_temp = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=min_dist_value,
        n_components=2,
        random_state=RANDOM_STATE
    )

    embedding_temp = reducer_temp.fit_transform(
        X
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.scatter(
        embedding_temp[:, 0],
        embedding_temp[:, 1],
        s=20,
        alpha=0.65
    )

    plt.title(
        f"UMAP - min_dist = {min_dist_value}",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "UMAP Component 1"
    )

    plt.ylabel(
        "UMAP Component 2"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    filename = (
        "umap_mindist_"
        + str(min_dist_value).replace(".", "_")
        + ".png"
    )

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        path
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("UMAP COMPLETED SUCCESSFULLY")
print("=" * 65)

print(
    "\nOutput folder:"
)

print(
    OUTPUT_DIR
)

print(
    "\nPNG files generated:"
)

for file in sorted(
    os.listdir(OUTPUT_DIR)
):

    if file.lower().endswith(".png"):

        print(
            "-",
            file
        )

print(
    "\nNo CSV files were generated."
)

print(
    "\nSelected UMAP parameters:"
)

print(
    "n_neighbors =",
    N_NEIGHBORS
)

print(
    "min_dist =",
    MIN_DIST
)

print(
    "n_components =",
    N_COMPONENTS
)

print(
    "\nDone!"
)