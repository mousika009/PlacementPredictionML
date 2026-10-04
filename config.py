import os

# =========================================================
# PROJECT ROOT
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# =========================================================
# DATA
# =========================================================

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

RAW_DATA_PATH = os.path.join(
    DATA_DIR,
    "placement_data.csv"
)

CLEANED_DATA_PATH = os.path.join(
    DATA_DIR,
    "cleaned_placement_data.csv"
)

ID_COL = "StudentID"

PLACEMENT_TARGET = "PlacementStatus"

SALARY_TARGET = "Salary Package"

# =========================================================
# OUTPUTS
# =========================================================

OUTPUTS_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

# =========================================================
# EDA
# =========================================================

EDA_DIR = os.path.join(
    OUTPUTS_DIR,
    "eda"
)

PLOTS_DIR = os.path.join(
    EDA_DIR,
    "plots"
)

REPORTS_DIR = os.path.join(
    EDA_DIR,
    "reports"
)

EDA_REPORT_PATH = os.path.join(
    REPORTS_DIR,
    "eda_report.txt"
)

# =========================================================
# CREATE DIRECTORIES
# =========================================================

os.makedirs(
    OUTPUTS_DIR,
    exist_ok=True
)

os.makedirs(
    EDA_DIR,
    exist_ok=True
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)

os.makedirs(
    REPORTS_DIR,
    exist_ok=True
)