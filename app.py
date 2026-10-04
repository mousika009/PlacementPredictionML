# ============================================================
# PLACEMENT PREDICTION DASHBOARD
# Flask Application
# ============================================================

import os
import glob
import math

import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory
)

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    classification_report,
    silhouette_score
)


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)


# ============================================================
# DATA PATH
# ============================================================

DATA_PATH = os.path.join(
    DATA_DIR,
    "placement_data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    try:

        if not os.path.exists(DATA_PATH):
            print(
                "Dataset not found:",
                DATA_PATH
            )

            return pd.DataFrame()

        df = pd.read_csv(
            DATA_PATH
        )

        return df

    except Exception as e:

        print(
            "Error loading dataset:",
            e
        )

        return pd.DataFrame()


# ============================================================
# NUMBER FORMAT
# ============================================================

def format_number(value):

    try:

        if isinstance(value, float):

            if value.is_integer():
                return f"{int(value):,}"

            return f"{value:,.2f}"

        return f"{int(value):,}"

    except:

        return str(value)


# ============================================================
# FIND PNG IMAGES
# ============================================================

def get_images(folder_name=None):

    """
    Automatically finds PNG files.

    No CSV files are used here.

    Example:

    outputs/
        eda/
            image1.png
            image2.png

    """

    images = []

    if folder_name:

        folder_path = os.path.join(
            OUTPUT_DIR,
            folder_name
        )

        search_paths = [
            folder_path
        ]

    else:

        search_paths = [
            OUTPUT_DIR
        ]

    for folder in search_paths:

        if not os.path.exists(folder):
            continue

        for root, dirs, files in os.walk(folder):

            for filename in files:

                if filename.lower().endswith(".png"):

                    full_path = os.path.join(
                        root,
                        filename
                    )

                    relative_path = os.path.relpath(
                        full_path,
                        OUTPUT_DIR
                    )

                    relative_path = relative_path.replace(
                        "\\",
                        "/"
                    )

                    image_url = url_for(
                        "output_image",
                        filepath=relative_path
                    )

                    display_name = os.path.splitext(
                        filename
                    )[0]

                    display_name = display_name.replace(
                        "_",
                        " "
                    ).replace(
                        "-",
                        " "
                    )

                    display_name = display_name.title()

                    images.append({
                        "name": display_name,
                        "url": image_url
                    })

    return images


# ============================================================
# SERVE PNG FILES
# ============================================================

@app.route(
    "/outputs/<path:filepath>"
)
def output_image(filepath):

    return send_from_directory(
        OUTPUT_DIR,
        filepath
    )


# ============================================================
# DATASET STATISTICS
# ============================================================

def dataset_statistics():

    df = load_data()

    if df.empty:

        return {
            "records": 0,
            "features": 0,
            "plots": 0
        }

    png_files = glob.glob(
        os.path.join(
            OUTPUT_DIR,
            "**",
            "*.png"
        ),
        recursive=True
    )

    return {

        "records": len(df),

        "features": len(df.columns),

        "plots": len(png_files)

    }


# ============================================================
# GENERIC MODULE PAGE
# ============================================================

def module_page(
    title,
    description,
    theory,
    purpose,
    formula,
    result,
    metrics=None,
    images=None,
    interpretation=""
):

    return render_template(

        "module.html",

        title=title,

        description=description,

        theory=theory,

        purpose=purpose,

        formula=formula,

        result=result,

        metrics=metrics or [],

        images=images or [],

        interpretation=interpretation

    )


# ============================================================
# HOME / DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    stats = dataset_statistics()

    df = load_data()

    placed = 0
    not_placed = 0

    if not df.empty and "PlacementStatus" in df.columns:

        values = pd.to_numeric(
            df["PlacementStatus"],
            errors="coerce"
        )

        placed = int(
            (values == 1).sum()
        )

        not_placed = int(
            (values == 0).sum()
        )

    placement_rate = 0

    if len(df) > 0:

        placement_rate = (
            placed / len(df)
        ) * 100

    return render_template(

        "dashboard.html",

        stats=stats,

        placed=placed,

        not_placed=not_placed,

        placement_rate=round(
            placement_rate,
            2
        )

    )


# ============================================================
# DATASET SUMMARY
# ============================================================

@app.route("/load")
def dataset_summary():

    df = load_data()

    if df.empty:

        return module_page(

            "Dataset Summary",

            "Overview of the placement dataset.",

            "The dataset contains student academic, technical and placement-related information.",

            "Dataset inspection is the first step before performing exploratory data analysis and machine learning.",

            "Rows = Number of student records\nColumns = Number of features",

            "Dataset could not be loaded.",

            images=get_images("dataset_summary")

        )

    numeric_count = len(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    categorical_count = len(
        df.select_dtypes(
            exclude=np.number
        ).columns
    )

    metrics = [

        {
            "label": "Records",
            "value": f"{len(df):,}"
        },

        {
            "label": "Features",
            "value": f"{len(df.columns):,}"
        },

        {
            "label": "Numerical",
            "value": f"{numeric_count}"
        },

        {
            "label": "Categorical",
            "value": f"{categorical_count}"
        }

    ]

    return module_page(

        "Dataset Summary",

        "Overview of the Placement Prediction Dataset.",

        (
            "Dataset summary provides a basic understanding "
            "of the size, structure and feature types of the "
            "placement dataset."
        ),

        (
            "It helps us understand how many students and "
            "features are available before preprocessing "
            "and model development."
        ),

        (
            "Dataset Size = Rows × Columns"
        ),

        (
            f"The dataset contains {len(df):,} student records "
            f"and {len(df.columns)} features."
        ),

        metrics=metrics,

        images=get_images("dataset_summary"),

        interpretation=(
            "The dataset contains both numerical and "
            "categorical information that can be used "
            "during exploratory analysis and machine learning."
        )

    )


# ============================================================
# EDA
# ============================================================

@app.route("/eda")
def eda():

    df = load_data()

    images = get_images("eda")

    metrics = []

    if not df.empty:

        metrics = [

            {
                "label": "Records",
                "value": f"{len(df):,}"
            },

            {
                "label": "Features",
                "value": f"{len(df.columns)}"
            },

            {
                "label": "Missing Cells",
                "value": f"{int(df.isna().sum().sum()):,}"
            },

            {
                "label": "Duplicate Rows",
                "value": f"{int(df.duplicated().sum()):,}"
            }

        ]

    return module_page(

        "EDA Overview",

        "Exploratory Data Analysis of student placement data.",

        (
            "Exploratory Data Analysis, or EDA, is used to "
            "understand patterns, distributions, relationships "
            "and unusual observations in a dataset."
        ),

        (
            "EDA helps identify important placement-related "
            "patterns before applying machine learning models."
        ),

        (
            "EDA = Distribution + Relationships + Patterns + Outliers"
        ),

        (
            f"Generated {len(images)} visualization(s) "
            "for the dataset."
        ),

        metrics=metrics,

        images=images,

        interpretation=(
            "The visualizations help identify relationships "
            "between academic performance, technical skills "
            "and placement outcomes."
        )

    )


# ============================================================
# DATA QUALITY
# ============================================================

@app.route("/data-quality")
def data_quality():

    df = load_data()

    if df.empty:

        return module_page(

            "Data Quality",

            "Dataset quality analysis.",

            "Data quality analysis checks whether the dataset is complete, consistent and reliable.",

            "Clean data is important because missing and duplicate records can affect machine learning results.",

            "Missing Cells = Total empty cells",

            "Dataset could not be loaded.",

            images=get_images("data_quality")

        )

    missing = int(
        df.isna().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )

    complete_rows = int(
        df.dropna().shape[0]
    )

    metrics = [

        {
            "label": "Missing Cells",
            "value": f"{missing:,}"
        },

        {
            "label": "Duplicate Rows",
            "value": f"{duplicates:,}"
        },

        {
            "label": "Complete Rows",
            "value": f"{complete_rows:,}"
        },

        {
            "label": "Records",
            "value": f"{len(df):,}"
        }

    ]

    return module_page(

        "Data Quality",

        "Checking missing values, duplicates and dataset consistency.",

        (
            "Data quality analysis checks the dataset for "
            "missing values, duplicate records and incomplete "
            "observations."
        ),

        (
            "This step ensures that the data is suitable for "
            "preprocessing and machine learning."
        ),

        (
            "Missing Cells = Count of empty cells\n\n"
            "Duplicate Rows = Number of identical records"
        ),

        (
            f"The dataset contains {missing:,} missing cells "
            f"and {duplicates:,} duplicate rows."
        ),

        metrics=metrics,

        images=get_images("data_quality"),

        interpretation=(
            "The quality results indicate how much cleaning "
            "is required before model training."
        )

    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

@app.route("/feature-engineering")
def feature_engineering():

    df = load_data()

    numeric_count = 0

    if not df.empty:

        numeric_count = len(
            df.select_dtypes(
                include=np.number
            ).columns
        )

    return module_page(

        "Feature Engineering",

        "Preparing useful features for machine learning.",

        (
            "Feature engineering transforms raw dataset "
            "variables into useful representations that "
            "can improve machine learning performance."
        ),

        (
            "In placement prediction, academic, attendance, "
            "internship, project, aptitude and technical "
            "attributes can be prepared for model training."
        ),

        (
            "Feature = Original Information → "
            "Machine Learning Representation"
        ),

        (
            f"{numeric_count} numerical feature(s) "
            "are available for modelling."
        ),

        metrics=[

            {
                "label": "Numerical Features",
                "value": str(numeric_count)
            }

        ],

        images=get_images(
            "feature_engineering"
        ),

        interpretation=(
            "Good feature engineering can make patterns "
            "easier for machine learning models to learn."
        )

    )


# ============================================================
# FEATURE ENCODING
# ============================================================

@app.route("/feature-encoding")
def feature_encoding():

    df = load_data()

    if df.empty:

        return module_page(

            "Feature Encoding",

            "Encoding categorical features.",

            "Feature encoding converts categorical information into numerical representations.",

            "Machine learning algorithms generally require numerical input.",

            "Category → Numerical Representation",

            "Dataset could not be loaded.",

            images=get_images("feature_encoding")

        )

    categorical = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    metrics = [

        {
            "label": "Categorical",
            "value": str(len(categorical))
        },

        {
            "label": "Numerical",
            "value": str(len(numerical))
        },

        {
            "label": "Total Features",
            "value": str(len(df.columns))
        },

        {
            "label": "Records",
            "value": f"{len(df):,}"
        }

    ]

    return module_page(

        "Feature Encoding",

        "Converting categorical variables into machine-readable values.",

        (
            "Feature encoding transforms categorical variables "
            "into numerical representations that machine "
            "learning algorithms can process."
        ),

        (
            "Encoding is useful when the dataset contains "
            "categorical information such as gender, degree, "
            "specialization or other non-numerical attributes."
        ),

        (
            "One-Hot Encoding:\n"
            "Category → 0 / 1 indicator columns"
        ),

        (
            f"The dataset contains {len(categorical)} "
            f"categorical and {len(numerical)} numerical features."
        ),

        metrics=metrics,

        images=get_images(
            "feature_encoding"
        ),

        interpretation=(
            "After encoding, categorical information can be "
            "represented numerically and supplied to machine "
            "learning algorithms."
        )

    )


# ============================================================
# LINEAR REGRESSION
# ============================================================

@app.route("/linear-regression")
def linear_regression():

    df = load_data()

    if df.empty:

        return module_page(
            "Linear Regression",
            "Continuous-value prediction.",
            "Linear Regression predicts a continuous target using a linear relationship between input features and the target.",
            "It can be used to estimate a continuous value such as salary package.",
            "y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ",
            "Dataset could not be loaded.",
            images=get_images("linear_regression")
        )

    target = None

    for candidate in [
        "Salary Package",
        "Salary",
        "salary",
        "Package",
        "package"
    ]:

        if candidate in df.columns:
            target = candidate
            break

    images = get_images(
        "linear_regression"
    )

    if target is None:

        return module_page(

            "Linear Regression",

            "Continuous-value prediction.",

            "Linear Regression predicts a continuous target using a linear relationship between input variables and the target.",

            "It can be used to estimate a continuous value such as salary package.",

            "y = β₀ + β₁x₁ + ... + βₙxₙ",

            "No continuous salary target was found.",

            images=images

        )

    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    features = [
        c for c in numeric
        if c != target
    ]

    if not features:

        return module_page(
            "Linear Regression",
            "Continuous-value prediction.",
            "Linear Regression predicts a continuous target.",
            "It estimates a continuous value.",
            "y = β₀ + β₁x₁ + ... + βₙxₙ",
            "No numerical features were available.",
            images=images
        )

    X = df[features].copy()

    y = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    valid = y.notna()

    X = X.loc[valid]

    y = y.loc[valid]

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median()
    )

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = LinearRegression()

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        rmse = math.sqrt(
            mean_squared_error(
                y_test,
                prediction
            )
        )

        mae = mean_absolute_error(
            y_test,
            prediction
        )

        r2 = r2_score(
            y_test,
            prediction
        )

        metrics = [

            {
                "label": "RMSE",
                "value": f"{rmse:.4f}"
            },

            {
                "label": "MAE",
                "value": f"{mae:.4f}"
            },

            {
                "label": "R² Score",
                "value": f"{r2:.4f}"
            }

        ]

        result = (
            f"Linear Regression achieved an R² score of "
            f"{r2:.4f} on the test data."
        )

    except Exception as e:

        metrics = []

        result = (
            f"Model calculation could not be completed: {e}"
        )

    return module_page(

        "Linear Regression",

        "Predicting a continuous placement-related value.",

        (
            "Linear Regression is a supervised learning "
            "algorithm used to predict a continuous numerical "
            "target."
        ),

        (
            "It can be used in this project to estimate "
            "continuous outcomes such as salary package."
        ),

        "y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ",

        result,

        metrics=metrics,

        images=images,

        interpretation=(
            "A higher R² generally indicates that the model "
            "explains a larger proportion of the variation "
            "in the target."
        )

    )


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

@app.route("/logistic-regression")
def logistic_regression():

    df = load_data()

    images = get_images(
        "logistic_regression"
    )

    if df.empty:

        return module_page(
            "Logistic Regression",
            "Binary placement classification.",
            "Logistic Regression is a supervised classification algorithm used to estimate the probability of a class.",
            "It is suitable for predicting whether a student is placed or not placed.",
            "P(y=1) = 1 / (1 + e⁻ᶻ)",
            "Dataset could not be loaded.",
            images=images
        )

    if "PlacementStatus" not in df.columns:

        return module_page(
            "Logistic Regression",
            "Binary placement classification.",
            "Logistic Regression predicts a categorical outcome.",
            "It is useful for placement classification.",
            "P(y=1) = 1 / (1 + e⁻ᶻ)",
            "PlacementStatus column was not found.",
            images=images
        )

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

    features = [
        c for c in features
        if c in df.columns
    ]

    X = df[features].copy()

    y = pd.to_numeric(
        df["PlacementStatus"],
        errors="coerce"
    )

    valid = y.notna()

    X = X.loc[valid]

    y = y.loc[valid]

    X = X.fillna(
        X.median()
    )

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y.astype(int),
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        model = LogisticRegression(
            max_iter=1000
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        prediction = model.predict(
            X_test_scaled
        )

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        metrics = [

            {
                "label": "Accuracy",
                "value": f"{accuracy * 100:.2f}%"
            },

            {
                "label": "Training Records",
                "value": f"{len(X_train):,}"
            },

            {
                "label": "Testing Records",
                "value": f"{len(X_test):,}"
            }

        ]

        result = (
            f"Logistic Regression classified placement "
            f"status with an accuracy of "
            f"{accuracy * 100:.2f}%."
        )

    except Exception as e:

        metrics = []

        result = (
            f"Model calculation could not be completed: {e}"
        )

    return module_page(

        "Logistic Regression",

        "Predicting whether a student is placed or not placed.",

        (
            "Logistic Regression is a supervised classification "
            "algorithm that estimates the probability of a "
            "binary outcome."
        ),

        (
            "It is directly suitable for the placement problem "
            "because PlacementStatus represents placed and "
            "not-placed outcomes."
        ),

        "P(y=1) = 1 / (1 + e⁻ᶻ)",

        result,

        metrics=metrics,

        images=images,

        interpretation=(
            "The model uses student attributes such as CGPA, "
            "attendance, internships and test scores to "
            "classify placement status."
        )

    )


# ============================================================
# DECISION TREE
# ============================================================

@app.route("/decision-tree")
def decision_tree():

    df = load_data()

    images = get_images(
        "decision_tree"
    )

    if df.empty:

        return module_page(
            "Decision Tree",
            "Tree-based placement classification.",
            "A Decision Tree makes predictions using a sequence of feature-based decisions.",
            "It can classify students into placed and not-placed categories.",
            "Gini = 1 − Σpᵢ²",
            "Dataset could not be loaded.",
            images=images
        )

    if "PlacementStatus" not in df.columns:

        return module_page(
            "Decision Tree",
            "Tree-based placement classification.",
            "Decision Trees split data using feature conditions.",
            "Useful for placement classification.",
            "Gini = 1 − Σpᵢ²",
            "PlacementStatus was not found.",
            images=images
        )

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

    features = [
        c for c in features
        if c in df.columns
    ]

    X = df[features].copy()

    y = pd.to_numeric(
        df["PlacementStatus"],
        errors="coerce"
    )

    valid = y.notna()

    X = X.loc[valid]

    y = y.loc[valid].astype(int)

    X = X.fillna(
        X.median()
    )

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        metrics = [

            {
                "label": "Accuracy",
                "value": f"{accuracy * 100:.2f}%"
            },

            {
                "label": "Tree Depth",
                "value": str(model.get_depth())
            },

            {
                "label": "Leaf Nodes",
                "value": str(model.get_n_leaves())
            }

        ]

        result = (
            f"Decision Tree achieved an accuracy of "
            f"{accuracy * 100:.2f}%."
        )

    except Exception as e:

        metrics = []

        result = (
            f"Model calculation failed: {e}"
        )

    return module_page(

        "Decision Tree",

        "Tree-based classification for student placement prediction.",

        (
            "A Decision Tree recursively divides data into "
            "smaller groups using feature-based conditions."
        ),

        (
            "It is useful for placement prediction because "
            "the resulting decision rules are relatively easy "
            "to understand."
        ),

        "Gini = 1 − Σpᵢ²",

        result,

        metrics=metrics,

        images=images,

        interpretation=(
            "The tree structure and feature importance "
            "visualizations help explain which student "
            "attributes contribute to placement prediction."
        )

    )


# ============================================================
# K-MEANS
# ============================================================

@app.route("/kmeans")
def kmeans():

    df = load_data()

    images = get_images(
        "kmeans"
    )

    if df.empty:

        return module_page(
            "K-Means Clustering",
            "Unsupervised student grouping.",
            "K-Means is an unsupervised learning algorithm that divides observations into K clusters.",
            "It can identify groups of students with similar characteristics.",
            "Minimize Σ ||xᵢ − μₖ||²",
            "Dataset could not be loaded.",
            images=images
        )

    exclude = [
        "PlacementStatus",
        "StudentID",
        "IsAnomaly",
        "Salary Package"
    ]

    features = [
        c for c in df.select_dtypes(
            include=np.number
        ).columns
        if c not in exclude
    ]

    X = df[features].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median()
    )

    try:

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            X
        )

        best_k = 3

        model = KMeans(
            n_clusters=best_k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(
            X_scaled
        )

        score = silhouette_score(
            X_scaled,
            labels
        )

        metrics = [

            {
                "label": "Clusters",
                "value": str(best_k)
            },

            {
                "label": "Silhouette Score",
                "value": f"{score:.4f}"
            },

            {
                "label": "Students",
                "value": f"{len(df):,}"
            }

        ]

        result = (
            f"K-Means grouped the students into "
            f"{best_k} clusters with a silhouette score "
            f"of {score:.4f}."
        )

    except Exception as e:

        metrics = []

        result = (
            f"K-Means calculation failed: {e}"
        )

    return module_page(

        "K-Means Clustering",

        "Grouping students based on similar characteristics.",

        (
            "K-Means is an unsupervised clustering algorithm "
            "that assigns observations to K groups based on "
            "distance from cluster centroids."
        ),

        (
            "It helps identify natural student groups based "
            "on academic and technical characteristics."
        ),

        "Minimize Σ ||xᵢ − μₖ||²",

        result,

        metrics=metrics,

        images=images,

        interpretation=(
            "Clusters can reveal groups of students with "
            "similar academic and skill profiles."
        )

    )


# ============================================================
# DBSCAN
# ============================================================

@app.route("/dbscan")
def dbscan():

    df = load_data()

    images = get_images(
        "dbscan"
    )

    if df.empty:

        return module_page(
            "DBSCAN Clustering",
            "Density-based student clustering.",
            "DBSCAN groups points based on density and can identify noise points.",
            "It is useful for discovering unusual student groups and dense regions.",
            "Core Point: number of neighbours ≥ MinPts",
            "Dataset could not be loaded.",
            images=images
        )

    exclude = [
        "PlacementStatus",
        "StudentID",
        "IsAnomaly",
        "Salary Package"
    ]

    features = [
        c for c in df.select_dtypes(
            include=np.number
        ).columns
        if c not in exclude
    ]

    X = df[features].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median()
    )

    try:

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            X
        )

        model = DBSCAN(
            eps=0.8,
            min_samples=5
        )

        labels = model.fit_predict(
            X_scaled
        )

        unique_labels = set(labels)

        cluster_count = len(
            unique_labels - {-1}
        )

        noise_count = int(
            (labels == -1).sum()
        )

        metrics = [

            {
                "label": "Clusters",
                "value": str(cluster_count)
            },

            {
                "label": "Noise Points",
                "value": f"{noise_count:,}"
            },

            {
                "label": "Records",
                "value": f"{len(df):,}"
            }

        ]

        result = (
            f"DBSCAN identified {cluster_count} clusters "
            f"and {noise_count:,} noise points."
        )

    except Exception as e:

        metrics = []

        result = (
            f"DBSCAN calculation failed: {e}"
        )

    return module_page(

        "DBSCAN Clustering",

        "Density-based clustering and noise detection.",

        (
            "DBSCAN groups observations based on the density "
            "of neighbouring points. Points that do not belong "
            "to dense regions can be marked as noise."
        ),

        (
            "It can identify natural student groups while "
            "also detecting unusual observations."
        ),

        (
            "Core Point: Neighbours ≥ MinPts"
        ),

        result,

        metrics=metrics,

        images=images,

        interpretation=(
            "Noise points identified by DBSCAN may represent "
            "unusual student profiles or observations that "
            "do not belong to a dense group."
        )

    )


# ============================================================
# PCA
# ============================================================

@app.route("/pca")
def pca():

    df = load_data()

    images = get_images(
        "pca"
    )

    if df.empty:

        return module_page(
            "PCA",
            "Principal Component Analysis.",
            "PCA reduces the dimensionality of numerical data while preserving as much variance as possible.",
            "It helps visualize high-dimensional student data in a smaller number of components.",
            "X → Principal Components",
            "Dataset could not be loaded.",
            images=images
        )

    exclude = [
        "PlacementStatus",
        "StudentID",
        "IsAnomaly",
        "Salary Package"
    ]

    features = [
        c for c in df.select_dtypes(
            include=np.number
        ).columns
        if c not in exclude
    ]

    X = df[features].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median()
    )

    try:

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            X
        )

        n_components = min(
            2,
            X_scaled.shape[1]
        )

        model = PCA(
            n_components=n_components
        )

        transformed = model.fit_transform(
            X_scaled
        )

        explained = (
            model.explained_variance_ratio_.sum()
            * 100
        )

        metrics = [

            {
                "label": "Components",
                "value": str(n_components)
            },

            {
                "label": "Variance Retained",
                "value": f"{explained:.2f}%"
            },

            {
                "label": "Features",
                "value": str(len(features))
            }

        ]

        result = (
            f"The selected PCA components retain "
            f"{explained:.2f}% of the original variance."
        )

    except Exception as e:

        metrics = []

        result = (
            f"PCA calculation failed: {e}"
        )

    return module_page(

        "PCA",

        "Principal Component Analysis for dimensionality reduction.",

        (
            "Principal Component Analysis transforms correlated "
            "features into a smaller set of principal components."
        ),

        (
            "PCA helps reduce dimensionality and makes "
            "high-dimensional student data easier to visualize."
        ),

        "Principal Components maximize explained variance.",

        result,

        metrics=metrics,

        images=images,

        interpretation=(
            "The PCA visualization shows how student records "
            "are distributed in a reduced feature space."
        )

    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

@app.route("/anomaly")
def anomaly():

    df = load_data()

    images = get_images(
        "anomaly"
    )

    if df.empty:

        return module_page(
            "Anomaly Detection",
            "Finding unusual student records.",
            "Anomaly detection identifies observations that differ significantly from normal patterns.",
            "It can help identify unusual student profiles in the placement dataset.",
            "Isolation Forest isolates unusual observations using random partitioning.",
            "Dataset could not be loaded.",
            images=images
        )

    exclude = [
        "PlacementStatus",
        "StudentID",
        "Salary Package"
    ]

    features = [
        c for c in df.select_dtypes(
            include=np.number
        ).columns
        if c not in exclude
    ]

    X = df[features].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median()
    )

    try:

        model = IsolationForest(
            contamination=0.02,
            random_state=42
        )

        labels = model.fit_predict(
            X
        )

        anomaly_count = int(
            (labels == -1).sum()
        )

        normal_count = int(
            (labels == 1).sum()
        )

        percentage = (
            anomaly_count / len(X)
        ) * 100

        metrics = [

            {
                "label": "Anomalies",
                "value": f"{anomaly_count:,}"
            },

            {
                "label": "Normal Records",
                "value": f"{normal_count:,}"
            },

            {
                "label": "Anomaly Rate",
                "value": f"{percentage:.2f}%"
            }

        ]

        result = (
            f"Isolation Forest identified "
            f"{anomaly_count:,} potentially unusual records."
        )

    except Exception as e:

        metrics = []

        result = (
            f"Anomaly detection failed: {e}"
        )

    return module_page(

        "Anomaly Detection",

        "Identifying unusual student profiles.",

        (
            "Anomaly detection identifies records that differ "
            "substantially from the normal pattern of the data."
        ),

        (
            "It helps identify unusual combinations of "
            "academic and technical characteristics."
        ),

        (
            "Isolation Forest isolates observations using "
            "random feature-based partitions."
        ),

        result,

        metrics=metrics,

        images=images,

        interpretation=(
            "Anomalies should be investigated carefully because "
            "they may represent unusual but valid students "
            "or potential data-quality problems."
        )

    )


# ============================================================
# t-SNE
# ============================================================

@app.route("/tsne")
def tsne():

    images = get_images(
        "tsne"
    )

    return module_page(

        "t-SNE",

        "Nonlinear dimensionality reduction and visualization.",

        (
            "t-SNE is a nonlinear dimensionality reduction "
            "technique used to visualize high-dimensional "
            "data in two dimensions."
        ),

        (
            "It helps visualize local similarities between "
            "student profiles."
        ),

        (
            "t-SNE minimizes the difference between "
            "high-dimensional and low-dimensional similarity distributions."
        ),

        (
            "The generated t-SNE visualization represents "
            "student profiles in a two-dimensional space."
        ),

        images=images,

        interpretation=(
            "Students appearing close together in the "
            "visualization have similar feature patterns."
        )

    )


# ============================================================
# UMAP
# ============================================================

@app.route("/umap")
def umap():

    images = get_images(
        "umap"
    )

    return module_page(

        "UMAP",

        "Nonlinear dimensionality reduction and visualization.",

        (
            "UMAP is a nonlinear dimensionality reduction "
            "technique that preserves important local and "
            "global structure in high-dimensional data."
        ),

        (
            "It can visualize similarities and patterns "
            "among student profiles."
        ),

        (
            "n_neighbors controls local/global structure.\n"
            "min_dist controls how tightly points are packed."
        ),

        (
            "The UMAP visualization provides a two-dimensional "
            "representation of the student dataset."
        ),

        images=images,

        interpretation=(
            "Clusters or groups visible in the UMAP plot "
            "may indicate similar student profiles."
        )

    )


# ============================================================
# MODEL TRAINING
# ============================================================

@app.route("/model-training")
def model_training():

    images = get_images(
        "model_training"
    )

    return module_page(

        "Model Training",

        "Training machine learning models for placement prediction.",

        (
            "Model training is the process of learning patterns "
            "from historical student data."
        ),

        (
            "The trained models can use student characteristics "
            "to predict placement outcomes."
        ),

        (
            "Training Data → Algorithm → Learned Model"
        ),

        (
            "Machine learning models are trained using "
            "the available placement dataset."
        ),

        images=images,

        interpretation=(
            "Training performance should be evaluated on "
            "unseen test data to estimate generalization."
        )

    )


# ============================================================
# MODEL EVALUATION
# ============================================================

@app.route("/model-evaluation")
def model_evaluation():

    images = get_images(
        "model_evaluation"
    )

    return module_page(

        "Model Evaluation",

        "Comparing machine learning model performance.",

        (
            "Model evaluation measures how accurately a "
            "machine learning model performs on unseen data."
        ),

        (
            "Evaluation helps compare different algorithms "
            "and select an appropriate model."
        ),

        (
            "Classification: Accuracy, Precision, Recall, F1\n"
            "Regression: MAE, RMSE, R²"
        ),

        (
            "Model performance can be compared using the "
            "generated evaluation visualizations."
        ),

        images=images,

        interpretation=(
            "The best model should be selected based on "
            "appropriate evaluation metrics rather than "
            "training performance alone."
        )

    )


# ============================================================
# PREDICT STUDENT
# ============================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    prediction = None

    probability = None

    if request.method == "POST":

        try:

            values = {

                "CGPA":
                    float(
                        request.form.get(
                            "CGPA",
                            0
                        )
                    ),

                "AttendancePercent":
                    float(
                        request.form.get(
                            "AttendancePercent",
                            0
                        )
                    ),

                "Internships":
                    float(
                        request.form.get(
                            "Internships",
                            0
                        )
                    ),

                "Projects":
                    float(
                        request.form.get(
                            "Projects",
                            0
                        )
                    ),

                "Workshops":
                    float(
                        request.form.get(
                            "Workshops",
                            0
                        )
                    ),

                "Certifications":
                    float(
                        request.form.get(
                            "Certifications",
                            0
                        )
                    ),

                "AptitudeTestScore":
                    float(
                        request.form.get(
                            "AptitudeTestScore",
                            0
                        )
                    ),

                "SoftSkillsRating":
                    float(
                        request.form.get(
                            "SoftSkillsRating",
                            0
                        )
                    ),

                "CodingTestScore":
                    float(
                        request.form.get(
                            "CodingTestScore",
                            0
                        )
                    ),

                "MockInterviewScore":
                    float(
                        request.form.get(
                            "MockInterviewScore",
                            0
                        )
                    )

            }

            prediction = (
                "Placed"
                if (
                    values["CGPA"] >= 7
                    and
                    values["CodingTestScore"] >= 50
                )
                else
                "Not Placed"
            )

        except Exception as e:

            prediction = (
                f"Prediction error: {e}"
            )

    return render_template(

        "predict.html",

        prediction=prediction,

        probability=probability

    )


# ============================================================
# STUDENT PROFILES
# ============================================================

@app.route("/students")
def students():

    df = load_data()

    if df.empty:

        return render_template(
            "students.html",
            students=[],
            columns=[]
        )

    display_df = df.head(
        100
    ).copy()

    display_df = display_df.fillna(
        "-"
    )

    columns = display_df.columns.tolist()

    students_data = (
        display_df.to_dict(
            orient="records"
        )
    )

    return render_template(

        "students.html",

        students=students_data,

        columns=columns

    )


# ============================================================
# UPLOAD DATASET
# ============================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
def upload():

    message = None

    if request.method == "POST":

        file = request.files.get(
            "file"
        )

        if file:

            filename = file.filename.lower()

            if filename.endswith(".csv"):

                save_path = os.path.join(
                    DATA_DIR,
                    "placement_data.csv"
                )

                file.save(
                    save_path
                )

                message = (
                    "Dataset uploaded successfully."
                )

            else:

                message = (
                    "Please upload a CSV dataset."
                )

    return render_template(
        "upload.html",
        message=message
    )


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# REPORT
# ============================================================

@app.route("/report")
def report():

    reports = []

    for root, dirs, files in os.walk(
        OUTPUT_DIR
    ):

        for filename in files:

            if filename.lower().endswith(
                ".txt"
            ):

                path = os.path.join(
                    root,
                    filename
                )

                try:

                    with open(
                        path,
                        "r",
                        encoding="utf-8"
                    ) as f:

                        content = f.read()

                    reports.append({

                        "name":
                            filename,

                        "content":
                            content

                    })

                except:

                    pass

    return render_template(

        "report.html",

        reports=reports

    )


# ============================================================
# ERROR HANDLER
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(

        "module.html",

        title="Page Not Found",

        description="The requested page does not exist.",

        theory="The requested dashboard module could not be found.",

        purpose="Please use the navigation menu to open a valid module.",

        formula="",

        result="404 - Page Not Found",

        metrics=[],

        images=[],

        interpretation="Please check the URL."

    ), 404


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("PLACEMENT PREDICTION DASHBOARD")
    print("=" * 65)

    print()
    print(
        "Dataset:",
        DATA_PATH
    )

    print(
        "Outputs:",
        OUTPUT_DIR
    )

    print()
    print(
        "Starting Flask server..."
    )

    print()
    print(
        "Open: http://127.0.0.1:5000"
    )

    print()
    print("=" * 65)

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )