import os
import pandas as pd
import config


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_raw():

    return pd.read_csv(config.RAW_DATA_PATH)


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df=None, save=True):

    if df is None:
        df = load_raw()

    df = df.copy()

    missing_cols = [
        "Workshops",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    for col in missing_cols:

        if col in df.columns:

            df[col] = df[col].fillna(df[col].median())

    # Remove duplicate Student IDs
    before = len(df)

    df = df.drop_duplicates(
        subset=[config.ID_COL]
    )

    duplicates_removed = before - len(df)

    # Save cleaned dataset
    if save:

        processed_folder = os.path.dirname(
            config.CLEANED_DATA_PATH
        )

        os.makedirs(
            processed_folder,
            exist_ok=True
        )

        df.to_csv(
            config.CLEANED_DATA_PATH,
            index=False
        )

    return df, duplicates_removed


# ============================================================
# LOAD CLEANED DATA
# ============================================================

def load_cleaned():

    if os.path.exists(config.CLEANED_DATA_PATH):

        return pd.read_csv(
            config.CLEANED_DATA_PATH
        )

    df, _ = clean_data()

    return df


# ============================================================
# NUMERIC COLUMNS
# ============================================================

def numeric_columns(df):

    return df.select_dtypes(
        include=["number"]
    ).columns.tolist()


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

def categorical_columns(df):

    return df.select_dtypes(
        include=["object"]
    ).columns.tolist()


# ============================================================
# DATASET SUMMARY
# ============================================================

def dataset_summary(df):

    return {

        "rows": int(df.shape[0]),

        "columns": int(df.shape[1]),

        "missing": int(
            df.isnull().sum().sum()
        ),

        "duplicates": int(
            df.duplicated().sum()
        )
    }