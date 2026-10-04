import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ============================================================
# K-MEANS CLUSTERING
# Placement Prediction 50K Dataset
# ============================================================

print("=" * 65)
print("                    K-MEANS CLUSTERING")
print("=" * 65)


# ============================================================
# 1. PATHS
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
    "kmeans"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

RANDOM_STATE = 42


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

print(
    "\nDataset Shape:",
    df.shape
)


# ============================================================
# 3. SELECT NUMERICAL FEATURES
# ============================================================

exclude_cols = [
    "PlacementStatus",
    "StudentID",
    "IsAnomaly",
    "Salary Package"
]


feature_cols = [
    column
    for column in df.select_dtypes(
        include=np.number
    ).columns
    if column not in exclude_cols
]


if len(feature_cols) == 0:

    raise ValueError(
        "No numeric features were found for clustering."
    )


print("\n" + "=" * 65)
print("FEATURES USED FOR CLUSTERING")
print("=" * 65)

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

print("\n" + "=" * 65)
print("MISSING VALUE HANDLING")
print("=" * 65)

missing_before = int(
    X.isnull().sum().sum()
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

print("\n" + "=" * 65)
print("FEATURE STANDARDIZATION")
print("=" * 65)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_imputed
)

print(
    "Standardization completed."
)


# ============================================================
# 7. FIND BEST K
# ============================================================

print("\n" + "=" * 65)
print("FINDING BEST K USING SILHOUETTE SCORE")
print("=" * 65)


k_values = range(
    2,
    11
)

silhouette_scores = []


for k in k_values:

    model = KMeans(

        n_clusters=k,

        init="k-means++",

        n_init=10,

        random_state=RANDOM_STATE
    )

    labels = model.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(
        score
    )

    print(
        f"K = {k} | "
        f"Silhouette Score = {score:.4f}"
    )


# ============================================================
# 8. SELECT BEST K
# ============================================================

best_index = np.argmax(
    silhouette_scores
)

best_k = list(
    k_values
)[best_index]

best_score = silhouette_scores[
    best_index
]


print(
    "\nBest K:",
    best_k
)

print(
    "Best Silhouette Score:",
    round(
        best_score,
        4
    )
)


# ============================================================
# 9. SILHOUETTE SCORE GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o",
    linewidth=2
)

plt.axvline(
    best_k,
    linestyle="--",
    linewidth=1.5,
    label=f"Best K = {best_k}"
)

plt.scatter(
    [best_k],
    [best_score],
    s=100,
    zorder=5
)

plt.xlabel(
    "Number of Clusters (K)",
    fontsize=12
)

plt.ylabel(
    "Silhouette Score",
    fontsize=12
)

plt.title(
    "K-Means - Silhouette Score Analysis",
    fontsize=16,
    fontweight="bold"
)

plt.xticks(
    list(k_values)
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()


silhouette_path = os.path.join(
    OUTPUT_DIR,
    "kmeans_silhouette_scores.png"
)


plt.savefig(
    silhouette_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()


print(
    "\nSilhouette plot saved:"
)

print(
    silhouette_path
)


# ============================================================
# 10. FINAL K-MEANS MODEL
# ============================================================

print("\n" + "=" * 65)
print("FINAL K-MEANS MODEL")
print("=" * 65)


kmeans = KMeans(

    n_clusters=best_k,

    init="k-means++",

    n_init=10,

    random_state=RANDOM_STATE
)


clusters = kmeans.fit_predict(
    X_scaled
)


df["Cluster"] = clusters


print(
    "Final K-Means model trained."
)


# ============================================================
# 11. CLUSTER SIZES
# ============================================================

print("\n" + "=" * 65)
print("CLUSTER SIZES")
print("=" * 65)


cluster_sizes = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)


for cluster, size in cluster_sizes.items():

    print(
        f"Cluster {cluster}: "
        f"{size} students"
    )


# ============================================================
# 12. CLUSTER SIZE GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


clusters_list = (
    cluster_sizes.index
    .tolist()
)

sizes_list = (
    cluster_sizes.values
    .tolist()
)


bars = plt.bar(
    clusters_list,
    sizes_list
)


plt.xlabel(
    "Cluster",
    fontsize=12
)

plt.ylabel(
    "Number of Students",
    fontsize=12
)

plt.title(
    "K-Means - Students in Each Cluster",
    fontsize=16,
    fontweight="bold"
)

plt.xticks(
    clusters_list
)


# Display values above bars

for bar, value in zip(
    bars,
    sizes_list
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height(),

        f"{value:,}",

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
    "kmeans_cluster_sizes.png"
)


plt.savefig(
    cluster_size_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()


print(
    "\nCluster size graph saved:"
)

print(
    cluster_size_path
)


# ============================================================
# 13. PLACEMENT RATE BY CLUSTER
# ============================================================

placement_rate = None


if "PlacementStatus" in df.columns:

    print("\n" + "=" * 65)
    print("PLACEMENT RATE BY CLUSTER")
    print("=" * 65)

    placement_rate = (
        df.groupby(
            "Cluster"
        )["PlacementStatus"]
        .mean()
    )

    for cluster, rate in placement_rate.items():

        print(
            f"Cluster {cluster}: "
            f"{rate * 100:.2f}% placed"
        )


# ============================================================
# 14. PLACEMENT RATE GRAPH
# ============================================================

if placement_rate is not None:

    plt.figure(
        figsize=(10, 6)
    )

    rates = (
        placement_rate
        * 100
    )

    bars = plt.bar(
        rates.index,
        rates.values
    )

    plt.xlabel(
        "Cluster",
        fontsize=12
    )

    plt.ylabel(
        "Placement Rate (%)",
        fontsize=12
    )

    plt.title(
        "K-Means - Placement Rate by Cluster",
        fontsize=16,
        fontweight="bold"
    )

    plt.xticks(
        rates.index
    )

    plt.ylim(
        0,
        100
    )


    for bar, value in zip(
        bars,
        rates.values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height(),

            f"{value:.1f}%",

            ha="center",

            va="bottom",

            fontsize=10
        )


    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()


    placement_path = os.path.join(
        OUTPUT_DIR,
        "kmeans_placement_rate.png"
    )


    plt.savefig(
        placement_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()


    print(
        "\nPlacement rate graph saved:"
    )

    print(
        placement_path
    )


# ============================================================
# 15. PCA 2D VISUALIZATION OF K-MEANS CLUSTERS
# ============================================================

print("\n" + "=" * 65)
print("CREATING 2D CLUSTER VISUALIZATION")
print("=" * 65)


pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)


plt.figure(
    figsize=(10, 7)
)


scatter = plt.scatter(

    X_pca[:, 0],

    X_pca[:, 1],

    c=clusters,

    cmap="viridis",

    s=8,

    alpha=0.55
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
    "K-Means Clusters - PCA 2D Projection",
    fontsize=16,
    fontweight="bold"
)


plt.colorbar(
    scatter,
    label="Cluster"
)


plt.grid(
    alpha=0.2
)

plt.tight_layout()


cluster_visualization_path = os.path.join(
    OUTPUT_DIR,
    "kmeans_clusters_pca_2d.png"
)


plt.savefig(
    cluster_visualization_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()


print(
    "\nCluster visualization saved:"
)

print(
    cluster_visualization_path
)


# ============================================================
# 16. SAVE REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "kmeans_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "K-MEANS CLUSTERING REPORT\n"
    )

    file.write(
        "=" * 55 + "\n\n"
    )

    file.write(
        f"Dataset Shape: {df.shape}\n"
    )

    file.write(
        f"Number of Features Used: "
        f"{len(feature_cols)}\n"
    )

    file.write(
        f"Best K: {best_k}\n"
    )

    file.write(
        f"Best Silhouette Score: "
        f"{best_score:.4f}\n\n"
    )


    file.write(
        "Features Used:\n"
    )

    for feature in feature_cols:

        file.write(
            f"- {feature}\n"
        )


    file.write(
        "\nSilhouette Scores:\n"
    )

    for k, score in zip(
        k_values,
        silhouette_scores
    ):

        file.write(
            f"K={k}: {score:.4f}\n"
        )


    file.write(
        "\nCluster Sizes:\n"
    )

    for cluster, size in (
        cluster_sizes.items()
    ):

        file.write(
            f"Cluster {cluster}: "
            f"{size} students\n"
        )


    if placement_rate is not None:

        file.write(
            "\nPlacement Rate by Cluster:\n"
        )

        for cluster, rate in (
            placement_rate.items()
        ):

            file.write(
                f"Cluster {cluster}: "
                f"{rate * 100:.2f}% placed\n"
            )


    file.write(
        "\nPCA Visualization:\n"
    )

    file.write(
        "The K-Means clusters were projected "
        "onto two principal components "
        "for visualization.\n"
    )


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("K-MEANS COMPLETED SUCCESSFULLY")
print("=" * 65)


print("\nPNG files created:")

print(
    "1. kmeans_silhouette_scores.png"
)

print(
    "2. kmeans_cluster_sizes.png"
)

if placement_rate is not None:

    print(
        "3. kmeans_placement_rate.png"
    )

print(
    "4. kmeans_clusters_pca_2d.png"
)


print("\nReport:")

print(
    "kmeans_report.txt"
)


print("\nOutput directory:")

print(
    OUTPUT_DIR
)

print("\nDone!")