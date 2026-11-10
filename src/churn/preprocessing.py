"""Nettoyage des données brutes et ingénierie de variables."""
import numpy as np
import pandas as pd

from . import config

CANONICAL = {
    "contract": {"mensuel": "Mensuel", "annuel": "Annuel", "biannuel": "Biannuel"},
    "internet_service": {"fibre": "Fibre", "adsl": "ADSL", "aucun": "Aucun"},
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Applique toutes les règles de nettoyage et renvoie un nouveau DataFrame."""
    df = df.copy()

    # Doublons
    df = df.drop_duplicates(subset=config.ID_COL, keep="first")

    # Normalisation des catégories (espaces + casse)
    for col, mapping in CANONICAL.items():
        df[col] = df[col].astype(str).str.strip().str.lower().map(mapping)

    # Conversions numériques
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Valeurs aberrantes : charges mensuelles > Q3 + 3*IQR -> remplacées par la médiane
    q1, q3 = df["monthly_charges"].quantile([0.25, 0.75])
    upper = q3 + 3 * (q3 - q1)
    median_charge = df.loc[df["monthly_charges"] <= upper, "monthly_charges"].median()
    df.loc[df["monthly_charges"] > upper, "monthly_charges"] = median_charge

    # Imputation
    missing_total = df["total_charges"].isna()
    df.loc[missing_total, "total_charges"] = (
        df.loc[missing_total, "monthly_charges"] * df.loc[missing_total, "tenure_months"]
    ).round(2)
    df["age"] = df["age"].fillna(df["age"].median()).astype(int)

    df[config.TARGET] = (df[config.TARGET].str.strip() == "Oui").astype(int)
    return add_features(df).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_charge_per_month"] = (df["total_charges"] / df["tenure_months"].clip(lower=1)).round(2)
    df["tenure_group"] = pd.cut(
        df["tenure_months"],
        bins=[0, 6, 12, 24, 48, np.inf],
        labels=["0-6 mois", "7-12 mois", "13-24 mois", "25-48 mois", "49+ mois"],
    ).astype(str)
    return df


def quality_report(raw: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """Résumé chiffré des corrections appliquées."""
    return {
        "lignes_brutes": int(len(raw)),
        "lignes_nettoyees": int(len(clean_df)),
        "doublons_supprimes": int(raw.duplicated(subset=config.ID_COL).sum()),
        "total_charges_manquants": int(pd.to_numeric(raw["total_charges"], errors="coerce").isna().sum()),
        "age_manquants": int(raw["age"].isna().sum()),
        "valeurs_manquantes_restantes": int(clean_df.isna().sum().sum()),
    }


def main() -> None:
    config.ensure_dirs()
    raw = pd.read_csv(config.RAW_FILE)
    df = clean(raw)
    df.to_csv(config.CLEAN_FILE, index=False)
    print(f"[clean] {quality_report(raw, df)}")


if __name__ == "__main__":
    main()
