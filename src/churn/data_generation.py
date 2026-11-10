"""Génération d'un jeu de données synthétique et réaliste de clients télécom.

Le jeu contient volontairement des défauts de qualité (valeurs manquantes,
doublons, casse incohérente, valeurs aberrantes) pour illustrer l'étape de
nettoyage.
"""
import numpy as np
import pandas as pd

from . import config


def generate_customers(n: int = config.N_CUSTOMERS, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Femme", "Homme"], size=n)
    age = rng.integers(18, 80, size=n)
    region = rng.choice(
        ["Île-de-France", "Auvergne-Rhône-Alpes", "Occitanie", "Hauts-de-France", "Grand Est", "Bretagne"],
        size=n,
        p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1],
    )
    contract = rng.choice(["Mensuel", "Annuel", "Biannuel"], size=n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["Fibre", "ADSL", "Aucun"], size=n, p=[0.45, 0.35, 0.20])
    payment = rng.choice(
        ["Prélèvement", "Carte bancaire", "Virement", "Chèque"], size=n, p=[0.35, 0.3, 0.2, 0.15]
    )
    paperless = rng.choice(["Oui", "Non"], size=n, p=[0.6, 0.4])

    # L'ancienneté dépend du type de contrat
    tenure_scale = np.select([contract == "Mensuel", contract == "Annuel"], [14, 30], default=45)
    tenure = np.clip(rng.exponential(tenure_scale), 1, 72).astype(int)

    base_charge = np.select([internet == "Fibre", internet == "ADSL"], [65, 45], default=20)
    monthly = np.round(base_charge + rng.normal(10, 12, size=n).clip(-15, 50), 2)
    total = np.round(monthly * tenure * rng.uniform(0.95, 1.05, size=n), 2)
    support_calls = rng.poisson(lam=np.where(internet == "Fibre", 1.8, 1.0))

    # Probabilité de churn : modèle logistique avec des facteurs métier crédibles
    logit = (
        -1.6
        + 1.3 * (contract == "Mensuel")
        - 0.9 * (contract == "Biannuel")
        - 0.04 * tenure
        + 0.022 * (monthly - 50)
        + 0.35 * support_calls
        + 0.45 * (payment == "Chèque")
        + 0.25 * (paperless == "Oui")
        + 0.3 * (internet == "Fibre")
        + 0.2 * (age > 65)
    )
    proba = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, proba)

    df = pd.DataFrame(
        {
            config.ID_COL: [f"C{i:05d}" for i in range(1, n + 1)],
            "gender": gender,
            "age": age.astype(float),
            "region": region,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "paperless_billing": paperless,
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total.astype(object),
            "support_calls": support_calls,
            config.TARGET: np.where(churn == 1, "Oui", "Non"),
        }
    )
    return _inject_quality_issues(df, rng)


def _inject_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # 1. Valeurs manquantes / vides dans total_charges et age
    idx = rng.choice(n, size=int(0.02 * n), replace=False)
    df.loc[idx, "total_charges"] = " "
    idx = rng.choice(n, size=int(0.015 * n), replace=False)
    df.loc[idx, "age"] = np.nan

    # 2. Casse et espaces incohérents
    idx = rng.choice(n, size=int(0.03 * n), replace=False)
    df.loc[idx, "contract"] = df.loc[idx, "contract"].str.upper() + " "
    idx = rng.choice(n, size=int(0.02 * n), replace=False)
    df.loc[idx, "internet_service"] = df.loc[idx, "internet_service"].str.lower()

    # 3. Valeurs aberrantes (erreurs de saisie)
    idx = rng.choice(n, size=10, replace=False)
    df.loc[idx, "monthly_charges"] = df.loc[idx, "monthly_charges"] * 100

    # 4. Doublons
    dup = df.sample(n=int(0.01 * n), random_state=1)
    df = pd.concat([df, dup], ignore_index=True)
    return df.sample(frac=1, random_state=2).reset_index(drop=True)


def main() -> None:
    config.ensure_dirs()
    df = generate_customers()
    df.to_csv(config.RAW_FILE, index=False)
    print(f"[data] {len(df)} lignes écrites dans {config.RAW_FILE}")


if __name__ == "__main__":
    main()
