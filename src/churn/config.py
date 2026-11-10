"""Configuration centrale du projet (chemins, graine aléatoire, colonnes)."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"
SQL_DIR = ROOT_DIR / "sql"

RAW_FILE = RAW_DIR / "customers_raw.csv"
CLEAN_FILE = PROCESSED_DIR / "customers_clean.csv"
DB_FILE = PROCESSED_DIR / "churn.db"

RANDOM_STATE = 42
N_CUSTOMERS = 5000
TEST_SIZE = 0.2

TARGET = "churn"
ID_COL = "customer_id"

NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls",
    "avg_charge_per_month",
]
CATEGORICAL_FEATURES = [
    "gender",
    "region",
    "contract",
    "internet_service",
    "payment_method",
    "paperless_billing",
    "tenure_group",
]


def ensure_dirs() -> None:
    """Crée les dossiers de sortie s'ils n'existent pas."""
    for d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
