import os
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# =========================================================
# PROJECT PATH
# =========================================================

# featureencoding.py
#     ↓
# 03_feature_engineering
#     ↓
# src
#     ↓
# PlacementPredict_EDA  ← PROJECT ROOT

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = BASE_DIR / "data" / "placement_data.csv"


print("==============================================")
print("FEATURE ENCODING")
print("==============================================")

print("PROJECT ROOT:")
print(BASE_DIR)

print("\nDATA PATH:")
print(DATA_PATH)


# =========================================================
# CATEGORICAL FEATURES
# =========================================================

NOMINAL_COLUMNS = [
    "Gender",
    "City",
    "Stream",
    "Specialisation",
    "Hostel",
    "HistoryOfBacklogs"
]


# =========================================================
# ORDINAL FEATURES
# =========================================================

ORDINAL_COLUMNS = [
    "CollegeTier",
    "CGPA_Tier"
]


# =========================================================
# TARGET CONVERSION
# =========================================================

def convert_target(series):

    if series.dtype == "object":

        mapping = {
            "Yes": 1,
            "No": 0,
            "Placed": 1,
            "Not Placed": 0
        }

        return series.map(mapping)

    return series


# =========================================================
# SMOOTHED TARGET ENCODING
# =========================================================

def smooth_target_encode(
    train,
    column,
    target,
    m=50
):

    global_mean = target.mean()

    temp = train[[column]].copy()

    temp["_target"] = target.values

    grouped = temp.groupby(
        column
    )["_target"].agg(
        ["mean", "count"]
    )

    smoothed = (
        grouped["count"] * grouped["mean"]
        + m * global_mean
    ) / (
        grouped["count"] + m
    )

    return smoothed


# =========================================================
# MAIN FEATURE ENCODING FUNCTION
# =========================================================

def run_feature_encoding():

    # =====================================================
    # CHECK DATASET
    # =====================================================

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"\nDataset not found!\n"
            f"Expected location:\n{DATA_PATH}\n"
        )

    # =====================================================
    # LOAD DATASET
    # =====================================================

    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully!")

    print(
        "Dataset Shape:",
        df.shape
    )

    # =====================================================
    # TRAIN / TEST SPLIT
    # =====================================================

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["PlacementStatus"]
    )

    # =====================================================
    # 1. ONE-HOT ENCODING
    # =====================================================

    onehot_encoder = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    # Fit only on training data

    train_onehot = onehot_encoder.fit_transform(
        train_df[NOMINAL_COLUMNS]
    )

    # Transform test data

    test_onehot = onehot_encoder.transform(
        test_df[NOMINAL_COLUMNS]
    )

    # Get encoded column names

    onehot_columns = (
        onehot_encoder.get_feature_names_out(
            NOMINAL_COLUMNS
        )
    )

    # Convert to DataFrames

    train_onehot_df = pd.DataFrame(
        train_onehot,
        columns=onehot_columns,
        index=train_df.index
    )

    test_onehot_df = pd.DataFrame(
        test_onehot,
        columns=onehot_columns,
        index=test_df.index
    )

    # =====================================================
    # 2. ORDINAL ENCODING
    # =====================================================

    # CollegeTier:
    # Tier3 -> 0
    # Tier2 -> 1
    # Tier1 -> 2

    college_tier_order = [
        ["Tier3", "Tier2", "Tier1"]
    ]

    # CGPA_Tier:
    # Low -> 0
    # Medium -> 1
    # High -> 2

    cgpa_tier_order = [
        ["Low", "Medium", "High"]
    ]

    ordinal_encoder = OrdinalEncoder(
        categories=[
            college_tier_order[0],
            cgpa_tier_order[0]
        ],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    train_ordinal = ordinal_encoder.fit_transform(
        train_df[ORDINAL_COLUMNS]
    )

    test_ordinal = ordinal_encoder.transform(
        test_df[ORDINAL_COLUMNS]
    )

    train_ordinal_df = pd.DataFrame(
        train_ordinal,
        columns=[
            "CollegeTier_encoded",
            "CGPA_Tier_encoded"
        ],
        index=train_df.index
    )

    test_ordinal_df = pd.DataFrame(
        test_ordinal,
        columns=[
            "CollegeTier_encoded",
            "CGPA_Tier_encoded"
        ],
        index=test_df.index
    )

    # =====================================================
    # 3. TARGET CONVERSION
    # =====================================================

    train_target = convert_target(
        train_df["PlacementStatus"]
    )

    test_target = convert_target(
        test_df["PlacementStatus"]
    )

    # =====================================================
    # 4. NAIVE TARGET ENCODING
    # =====================================================

    city_means = train_df.copy()

    city_means["Target"] = train_target

    city_mapping = city_means.groupby(
        "City"
    )["Target"].mean()

    train_city_target = train_df[
        "City"
    ].map(
        city_mapping
    )

    test_city_target = test_df[
        "City"
    ].map(
        city_mapping
    )

    # Global training mean for unseen cities

    global_mean = train_target.mean()

    train_city_target = (
        train_city_target.fillna(
            global_mean
        )
    )

    test_city_target = (
        test_city_target.fillna(
            global_mean
        )
    )

    # =====================================================
    # 5. SMOOTHED TARGET ENCODING
    # =====================================================

    city_smooth_mapping = smooth_target_encode(
        train_df,
        "City",
        train_target,
        m=50
    )

    train_city_smooth = train_df[
        "City"
    ].map(
        city_smooth_mapping
    )

    test_city_smooth = test_df[
        "City"
    ].map(
        city_smooth_mapping
    )

    train_city_smooth = (
        train_city_smooth.fillna(
            global_mean
        )
    )

    test_city_smooth = (
        test_city_smooth.fillna(
            global_mean
        )
    )

    # =====================================================
    # 6. K-FOLD TARGET ENCODING
    # =====================================================

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_kfold_target = pd.Series(
        global_mean,
        index=train_df.index,
        dtype=float
    )

    for train_index, validation_index in kf.split(
        train_df
    ):

        fold_train = train_df.iloc[
            train_index
        ]

        fold_validation = train_df.iloc[
            validation_index
        ]

        fold_target = train_target.iloc[
            train_index
        ]

        fold_mapping = smooth_target_encode(
            fold_train,
            "City",
            fold_target,
            m=50
        )

        encoded_values = fold_validation[
            "City"
        ].map(
            fold_mapping
        )

        encoded_values = encoded_values.fillna(
            fold_target.mean()
        )

        train_kfold_target.iloc[
            validation_index
        ] = encoded_values.values

    # Mapping learned from complete training data

    final_city_mapping = smooth_target_encode(
        train_df,
        "City",
        train_target,
        m=50
    )

    test_kfold_target = test_df[
        "City"
    ].map(
        final_city_mapping
    )

    test_kfold_target = (
        test_kfold_target.fillna(
            global_mean
        )
    )

    # =====================================================
    # 7. FINAL ENCODED DATA
    # =====================================================

    train_encoded = train_df.copy()

    test_encoded = test_df.copy()

    # Add One-Hot features

    train_encoded = train_encoded.join(
        train_onehot_df
    )

    test_encoded = test_encoded.join(
        test_onehot_df
    )

    # Add Ordinal features

    train_encoded = train_encoded.join(
        train_ordinal_df
    )

    test_encoded = test_encoded.join(
        test_ordinal_df
    )

    # Add K-Fold Target encoded City

    train_encoded[
        "City_Target_Encoded"
    ] = train_kfold_target

    test_encoded[
        "City_Target_Encoded"
    ] = test_kfold_target

    # =====================================================
    # 8. DASHBOARD SAMPLE DATA
    # =====================================================

    # Original data - first 10 rows

    original_sample = train_df[
        NOMINAL_COLUMNS + ORDINAL_COLUMNS
    ].head(10)

    # One-Hot Encoding - first 10 rows

    onehot_sample = train_onehot_df.head(10)

    # Ordinal Encoding - first 10 rows

    ordinal_sample = train_ordinal_df.head(10)

    # Target Encoding - first 10 rows

    target_sample = pd.DataFrame({

        "City":
            train_df["City"].head(10).values,

        "Target Encoding":
            train_city_target.head(10).values,

        "Smoothed Encoding":
            train_city_smooth.head(10).values,

        "K-Fold Encoding":
            train_kfold_target.head(10).values
    })

    # =====================================================
    # 9. RETURN EVERYTHING NEEDED BY FLASK
    # =====================================================

    return {

        # Basic information

        "total_records":
            len(df),

        "original_features":
            len(df.columns),

        "train_records":
            len(train_df),

        "test_records":
            len(test_df),

        # Feature counts

        "nominal_features":
            len(NOMINAL_COLUMNS),

        "ordinal_features":
            len(ORDINAL_COLUMNS),

        "onehot_features":
            len(onehot_columns),

        "onehot_count":
            len(onehot_columns),

        "final_features":
            train_encoded.shape[1],

        # Feature names

        "nominal_columns":
            NOMINAL_COLUMNS,

        "ordinal_columns":
            ORDINAL_COLUMNS,

        "onehot_columns":
            list(onehot_columns),

        # Original sample

        "original_sample":
            original_sample.to_dict(
                orient="records"
            ),

        # One-Hot sample

        "onehot_sample":
            onehot_sample.round(
                3
            ).to_dict(
                orient="records"
            ),

        # Ordinal sample

        "ordinal_sample":
            ordinal_sample.round(
                3
            ).to_dict(
                orient="records"
            ),

        # Target sample

        "target_sample":
            target_sample.round(
                4
            ).to_dict(
                orient="records"
            ),

        # HTML table columns

        "ordinal_sample_columns":
            list(
                ordinal_sample.columns
            ),

        "target_sample_columns":
            list(
                target_sample.columns
            ),

        # Final DataFrames

        "train_encoded":
            train_encoded,

        "test_encoded":
            test_encoded
    }


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    result = run_feature_encoding()

    print("\n================================")
    print("TRAIN / TEST SPLIT")
    print("================================")

    print(
        "Training data:",
        (
            result["train_records"],
            result["original_features"]
        )
    )

    print(
        "Testing data :",
        (
            result["test_records"],
            result["original_features"]
        )
    )

    print("\n================================")
    print("1. ONE-HOT ENCODING")
    print("================================")

    print("\nOne-Hot Encoded Training Data:")

    print(
        pd.DataFrame(
            result["onehot_sample"]
        )
    )

    print("\nOne-Hot Encoded Columns:")

    print(
        result["onehot_columns"]
    )

    print("\n================================")
    print("2. ORDINAL ENCODING")
    print("================================")

    print("\nOrdinal Encoded Training Data:")

    print(
        pd.DataFrame(
            result["ordinal_sample"]
        )
    )

    print("\n================================")
    print("3. TARGET ENCODING")
    print("================================")

    print("\nTarget Encoding - 10 Rows:")

    print(
        pd.DataFrame(
            result["target_sample"]
        )
    )

    print("\n================================")
    print("FINAL ENCODED DATA")
    print("================================")

    print(
        "Final Training Data Shape:",
        result["train_encoded"].shape
    )

    print(
        "Final Testing Data Shape:",
        result["test_encoded"].shape
    )

    print("\n================================")
    print("ENCODING SUMMARY")
    print("================================")

    print(
        "Original categorical features:",
        result["nominal_columns"]
    )

    print(
        "Ordinal features:",
        result["ordinal_columns"]
    )

    print(
        "Number of One-Hot features:",
        result["onehot_features"]
    )

    print(
        "Target encoded feature:",
        "City_Target_Encoded"
    )

    print("\nFeature encoding completed successfully!")