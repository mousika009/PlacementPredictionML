import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


# ============================================================
# DBSCAN CLUSTERING
# Placement Prediction 50K Dataset
# ============================================================

print("=" * 60)
print("                 DBSCAN CLUSTERING")
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
    "dbscan"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

SUBSAMPLE_SIZE = 5000
MIN_SAMPLES = 40
RANDOM_STATE = 42


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df_full = pd.read_csv(DATA_PATH)

print(
    "Dataset Shape:",
    df_full.shape
)


# ============================================================
# 4. SELECT NUMERICAL FEATURES
# ============================================================

DROP_COLUMNS = [
    "StudentID",
    "PlacementStatus",
    "Salary Package",
    "IsAnomaly"
]


drop_existing = [
    column
    for column in DROP_COLUMNS
    if column in df_full.columns
]


df_features = df_full.drop(
    columns=drop_existing
)


numeric_columns = (
    df_features
    .select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)


if not numeric_columns:

    raise ValueError(
        "No numerical features found for DBSCAN."
    )


print("\n" + "=" * 60)
print("FEATURES USED FOR DBSCAN")
print("=" * 60)

for feature in numeric_columns:
    print("-", feature)

print(
    "\nNumber of features:",
    len(numeric_columns)
)


# ============================================================
# 5. CREATE FEATURE MATRIX
# ============================================================

X = df_full[
    numeric_columns
].copy()


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE HANDLING")
print("=" * 60)

missing_before = X.isnull().sum().sum()

print(
    "Missing values before:",
    missing_before
)


imputer = SimpleImputer(
    strategy="median"
)


X_imputed = imputer.fit_transform(X)


print(
    "Missing values after:",
    np.isnan(X_imputed).sum()
)


# ============================================================
# 7. STANDARDIZATION
# ============================================================

print("\n" + "=" * 60)
print("STANDARDIZATION")
print("=" * 60)


scaler = StandardScaler()


X_scaled_full = scaler.fit_transform(
    X_imputed
)


print(
    "Standardization completed."
)


# ============================================================
# 8. CREATE SAMPLE
# ============================================================

rng = np.random.RandomState(
    RANDOM_STATE
)


sample_size = min(
    SUBSAMPLE_SIZE,
    len(X_scaled_full)
)


sample_idx = rng.choice(
    len(X_scaled_full),
    size=sample_size,
    replace=False
)


X_scaled = X_scaled_full[
    sample_idx
]


df = df_full.iloc[
    sample_idx
].reset_index(drop=True)


print("\n" + "=" * 60)
print("DATA SAMPLE")
print("=" * 60)

print(
    "Full dataset:",
    len(df_full)
)

print(
    "DBSCAN sample:",
    len(df)
)


# ============================================================
# 9. K-DISTANCE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("K-DISTANCE ANALYSIS")
print("=" * 60)


neighbors = NearestNeighbors(
    n_neighbors=MIN_SAMPLES
)


neighbors.fit(
    X_scaled
)


distances, indices = neighbors.kneighbors(
    X_scaled
)


k_distances = np.sort(
    distances[:, -1]
)


# ============================================================
# 10. K-DISTANCE PLOT
# ============================================================

plt.figure(
    figsize=(11, 6)
)


plt.plot(
    k_distances,
    linewidth=2
)


plt.xlabel(
    "Points Sorted by K-Distance",
    fontsize=12
)

plt.ylabel(
    f"Distance to {MIN_SAMPLES}th Nearest Neighbor",
    fontsize=12
)


plt.title(
    "DBSCAN K-Distance Plot",
    fontsize=16,
    fontweight="bold"
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


k_distance_path = os.path.join(
    OUTPUT_DIR,
    "k_distance_plot.png"
)


plt.savefig(
    k_distance_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nK-distance plot saved:"
)

print(
    k_distance_path
)


# ============================================================
# 11. AUTOMATIC EPS SELECTION
# ============================================================

n_points = len(
    k_distances
)


x_norm = np.linspace(
    0,
    1,
    n_points
)


distance_range = (
    k_distances.max()
    -
    k_distances.min()
)


if distance_range == 0:

    EPS = 0.5

else:

    y_norm = (
        k_distances
        -
        k_distances.min()
    ) / distance_range


    numerator = np.abs(
        (
            y_norm[-1]
            -
            y_norm[0]
        ) * x_norm
        -
        (
            x_norm[-1]
            -
            x_norm[0]
        ) * y_norm
        +
        x_norm[-1]
        * y_norm[0]
        -
        y_norm[-1]
        * x_norm[0]
    )


    denominator = np.sqrt(
        (
            y_norm[-1]
            -
            y_norm[0]
        ) ** 2
        +
        (
            x_norm[-1]
            -
            x_norm[0]
        ) ** 2
    )


    if denominator == 0:

        EPS = 0.5

    else:

        perpendicular_distance = (
            numerator
            /
            denominator
        )


        elbow_idx = int(
            np.argmax(
                perpendicular_distance
            )
        )


        EPS = round(
            float(
                k_distances[
                    elbow_idx
                ]
            ),
            2
        )


print(
    "\nAutomatically selected EPS:",
    EPS
)


# ============================================================
# 12. RUN DBSCAN
# ============================================================

print("\n" + "=" * 60)
print("RUNNING DBSCAN")
print("=" * 60)


dbscan = DBSCAN(
    eps=EPS,
    min_samples=MIN_SAMPLES,
    algorithm="ball_tree",
    n_jobs=1
)


cluster_labels = dbscan.fit_predict(
    X_scaled
)


# ============================================================
# 13. DBSCAN RESULTS
# ============================================================

n_clusters = (
    len(
        set(cluster_labels)
    )
    -
    (
        1
        if -1 in cluster_labels
        else 0
    )
)


n_noise = int(
    (
        cluster_labels == -1
    ).sum()
)


n_core = len(
    dbscan.core_sample_indices_
)


noise_percentage = (
    n_noise
    /
    len(df)
    *
    100
)


print("\n" + "=" * 60)
print("DBSCAN RESULTS")
print("=" * 60)

print(
    "Clusters found:",
    n_clusters
)

print(
    "Core points:",
    n_core
)

print(
    "Noise points:",
    n_noise
)

print(
    "Noise percentage:",
    round(
        noise_percentage,
        2
    ),
    "%"
)


# ============================================================
# 14. CLUSTER SIZES
# ============================================================

print("\n" + "=" * 60)
print("CLUSTER SIZES")
print("=" * 60)


cluster_sizes = (
    pd.Series(
        cluster_labels
    )
    .value_counts()
    .sort_index()
)


print(
    cluster_sizes
)


# ============================================================
# 15. PCA 2D VISUALIZATION
# ============================================================

print("\n" + "=" * 60)
print("CREATING PCA VISUALIZATION")
print("=" * 60)


pca = PCA(
    n_components=2
)


X_2D = pca.fit_transform(
    X_scaled
)


# ============================================================
# 16. CREATE DBSCAN CLUSTER GRAPH
# ============================================================

plt.figure(
    figsize=(12, 8)
)


noise_mask = (
    cluster_labels == -1
)


cluster_mask = (
    cluster_labels != -1
)


# Normal clusters
if cluster_mask.any():

    scatter = plt.scatter(
        X_2D[
            cluster_mask,
            0
        ],

        X_2D[
            cluster_mask,
            1
        ],

        c=cluster_labels[
            cluster_mask
        ],

        cmap="tab10",

        s=18,

        alpha=0.65,

        edgecolors="none",

        label="Clusters"
    )


# Noise points
if noise_mask.any():

    plt.scatter(
        X_2D[
            noise_mask,
            0
        ],

        X_2D[
            noise_mask,
            1
        ],

        c="black",

        marker="x",

        s=35,

        alpha=0.8,

        label="Noise / Outliers"
    )


plt.xlabel(
    "Principal Component 1",
    fontsize=12
)


plt.ylabel(
    "Principal Component 2",
    fontsize=12
)


plt.title(
    f"DBSCAN Student Clusters\n"
    f"EPS = {EPS}, Min Samples = {MIN_SAMPLES}",
    fontsize=16,
    fontweight="bold"
)


plt.legend(
    loc="best"
)


plt.grid(
    alpha=0.25
)


plt.tight_layout()


pca_path = os.path.join(
    OUTPUT_DIR,
    "dbscan_clusters_pca_2d.png"
)


plt.savefig(
    pca_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nDBSCAN PCA visualization saved:"
)

print(
    pca_path
)


# ============================================================
# 17. CLUSTER SIZE GRAPH
# ============================================================

print("\nCreating cluster size graph...")


cluster_names = []

cluster_values = []


for cluster, size in cluster_sizes.items():

    if cluster == -1:
        cluster_names.append(
            "Noise"
        )

    else:
        cluster_names.append(
            f"Cluster {cluster}"
        )

    cluster_values.append(
        size
    )


plt.figure(
    figsize=(11, 6)
)


bars = plt.bar(
    cluster_names,
    cluster_values
)


plt.xlabel(
    "DBSCAN Group",
    fontsize=12
)


plt.ylabel(
    "Number of Students",
    fontsize=12
)


plt.title(
    "DBSCAN Cluster Distribution",
    fontsize=16,
    fontweight="bold"
)


plt.xticks(
    rotation=30,
    ha="right"
)


# Add values above bars
for bar, value in zip(
    bars,
    cluster_values
):

    plt.text(
        bar.get_x()
        +
        bar.get_width() / 2,

        bar.get_height(),

        str(value),

        ha="center",

        va="bottom",

        fontsize=10
    )


plt.grid(
    axis="y",
    alpha=0.25
)


plt.tight_layout()


cluster_size_path = os.path.join(
    OUTPUT_DIR,
    "dbscan_cluster_distribution.png"
)


plt.savefig(
    cluster_size_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Cluster distribution graph saved:"
)

print(
    cluster_size_path
)


# ============================================================
# 18. NOISE / OUTLIER GRAPH
# ============================================================

print("\nCreating noise analysis graph...")


normal_points = len(df) - n_noise


plt.figure(
    figsize=(8, 6)
)


plt.bar(
    [
        "Normal Points",
        "Noise / Outliers"
    ],

    [
        normal_points,
        n_noise
    ]
)


plt.ylabel(
    "Number of Students"
)


plt.title(
    "DBSCAN Noise / Outlier Detection",
    fontsize=16,
    fontweight="bold"
)


plt.grid(
    axis="y",
    alpha=0.25
)


plt.tight_layout()


noise_path = os.path.join(
    OUTPUT_DIR,
    "dbscan_noise_analysis.png"
)


plt.savefig(
    noise_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Noise analysis graph saved:"
)

print(
    noise_path
)


# ============================================================
# 19. PLACEMENT ANALYSIS
# ============================================================

if "PlacementStatus" in df.columns:

    print("\n" + "=" * 60)
    print("PLACEMENT ANALYSIS")
    print("=" * 60)


    placement_rate = (
        df.assign(
            DBSCAN_Cluster=cluster_labels
        )
        .groupby(
            "DBSCAN_Cluster"
        )[
            "PlacementStatus"
        ]
        .mean()
    )


    for cluster, rate in placement_rate.items():

        if cluster == -1:

            name = "Noise"

        else:

            name = (
                f"Cluster {cluster}"
            )


        print(
            f"{name}: "
            f"{rate * 100:.2f}% placed"
        )


    # --------------------------------------------------------
    # Placement Rate Graph
    # --------------------------------------------------------

    names = []

    rates = []


    for cluster, rate in placement_rate.items():

        if cluster == -1:

            names.append(
                "Noise"
            )

        else:

            names.append(
                f"Cluster {cluster}"
            )


        rates.append(
            rate * 100
        )


    plt.figure(
        figsize=(11, 6)
    )


    bars = plt.bar(
        names,
        rates
    )


    plt.xlabel(
        "DBSCAN Group"
    )


    plt.ylabel(
        "Placement Rate (%)"
    )


    plt.title(
        "Placement Rate by DBSCAN Cluster",
        fontsize=16,
        fontweight="bold"
    )


    plt.xticks(
        rotation=30,
        ha="right"
    )


    for bar, rate in zip(
        bars,
        rates
    ):

        plt.text(
            bar.get_x()
            +
            bar.get_width() / 2,

            bar.get_height(),

            f"{rate:.1f}%",

            ha="center",

            va="bottom"
        )


    plt.grid(
        axis="y",
        alpha=0.25
    )


    plt.tight_layout()


    placement_path = os.path.join(
        OUTPUT_DIR,
        "dbscan_placement_rate.png"
    )


    plt.savefig(
        placement_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "\nPlacement rate graph saved:"
    )

    print(
        placement_path
    )


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("DBSCAN COMPLETED SUCCESSFULLY")
print("=" * 60)


print("\nPNG FILES GENERATED:")

print(
    "1. k_distance_plot.png"
)

print(
    "2. dbscan_clusters_pca_2d.png"
)

print(
    "3. dbscan_cluster_distribution.png"
)

print(
    "4. dbscan_noise_analysis.png"
)

if "PlacementStatus" in df.columns:

    print(
        "5. dbscan_placement_rate.png"
    )


print("\nOutput folder:")

print(
    OUTPUT_DIR
)