import os
import sys
import importlib
from pathlib import Path

import config


# =========================================================
# IMPORT DATA FOUNDATION
# =========================================================

# Get the src folder
SRC_DIR = Path(__file__).resolve().parents[1]

# Add src to Python path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import data_utils from 01_data_foundation
data_utils = importlib.import_module(
    "01_data_foundation.data_utils"
)

load_raw = data_utils.load_raw
load_cleaned = data_utils.load_cleaned


# =========================================================
# LOAD DATA
# =========================================================

raw_df = load_raw()
clean_df = load_cleaned()


# =========================================================
# DASHBOARD STATISTICS
# =========================================================

def get_dashboard_stats():

    stats = {
        "records": len(raw_df),

        "features": len(raw_df.columns),

        # IMPORTANT:
        # Missing values are calculated from RAW dataset
        "missing": int(
            raw_df.isnull().sum().sum()
        ),

        "duplicates": int(
            raw_df.duplicated().sum()
        ),

        "plots": 11,

        "placed": (
            int(raw_df["PlacementStatus"].sum())
            if "PlacementStatus" in raw_df.columns
            else 0
        ),

        "not_placed": (
            int(
                (raw_df["PlacementStatus"] == 0).sum()
            )
            if "PlacementStatus" in raw_df.columns
            else 0
        )
    }

    return stats


# =========================================================
# DATASET SUMMARY
# =========================================================

def get_dataset_summary():

    # Use RAW data here so that missing values are visible
    df = raw_df

    missing_by_column = df.isnull().sum()

    columns = []

    for column in df.columns:

        columns.append({

            "name": column,

            "dtype": str(
                df[column].dtype
            ),

            "missing": int(
                missing_by_column[column]
            )

        })

    return {

        "rows": int(
            df.shape[0]
        ),

        "columns": int(
            df.shape[1]
        ),

        "missing": int(
            df.isnull().sum().sum()
        ),

        "duplicates": int(
            df.duplicated().sum()
        ),

        "columns_info": columns,

        "describe": df.describe(
            include="all"
        ).round(2)

    }


# =========================================================
# REPORT
# =========================================================

def get_report():

    if os.path.exists(
        config.EDA_REPORT_PATH
    ):

        with open(
            config.EDA_REPORT_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            return file.read()

    return "EDA Report not found."


# =========================================================
# PLOTS
# =========================================================

def get_plots():

    plot_dir = config.PLOTS_DIR

    if not os.path.exists(
        plot_dir
    ):

        return []

    images = []

    for file in sorted(
        os.listdir(plot_dir)
    ):

        if file.lower().endswith(".png"):

            images.append(file)

    return images


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EDA MODULE")
    print("=" * 60)

    print("\nDataset loaded successfully.")

    print(
        "Rows:",
        raw_df.shape[0]
    )

    print(
        "Columns:",
        raw_df.shape[1]
    )

    print(
        "Missing values:",
        raw_df.isnull().sum().sum()
    )

    print(
        "Duplicates:",
        raw_df.duplicated().sum()
    )

    print("\nDashboard Statistics:")

    print(
        get_dashboard_stats()
    )

    print("\nEDA module completed successfully.")