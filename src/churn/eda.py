"""Analyse exploratoire : génération des graphiques dans reports/figures."""
import matplotlib

matplotlib.use("Agg")  # rendu sans écran (CI, serveurs)
import matplotlib.pyplot as plt
import pandas as pd

from . import config

COLOR_STAY = "#4C78A8"
COLOR_CHURN = "#E45756"


def _save(fig, name: str) -> str:
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path.relative_to(config.REPORTS_DIR)).replace("\\", "/")


def plot_churn_distribution(df: pd.DataFrame) -> str:
    counts = df[config.TARGET].map({0: "Reste", 1: "Part"}).value_counts().reindex(["Reste", "Part"], fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=[COLOR_STAY, COLOR_CHURN])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Répartition du churn")
    ax.set_ylabel("Nombre de clients")
    return _save(fig, "01_churn_distribution.png")


def plot_churn_rate_by(df: pd.DataFrame, col: str, filename: str, order=None) -> str:
    rate = df.groupby(col)[config.TARGET].mean().mul(100)
    rate = rate.reindex(order) if order else rate.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rate.index.astype(str), rate.values, color=COLOR_CHURN)
    ax.axhline(df[config.TARGET].mean() * 100, ls="--", color="grey", label="Moyenne")
    ax.set_title(f"Taux de churn par {col}")
    ax.set_ylabel("Taux de churn (%)")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, filename)


def plot_numeric_distributions(df: pd.DataFrame) -> str:
    cols = ["tenure_months", "monthly_charges", "support_calls", "age"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col in zip(axes.ravel(), cols):
        for val, color, label in [(0, COLOR_STAY, "Reste"), (1, COLOR_CHURN, "Part")]:
            ax.hist(df.loc[df[config.TARGET] == val, col], bins=25, alpha=0.6, color=color, label=label, density=True)
        ax.set_title(col)
        ax.legend()
    fig.suptitle("Distributions des variables numériques selon le churn")
    return _save(fig, "05_numeric_distributions.png")


def plot_correlations(df: pd.DataFrame) -> str:
    corr = df[config.NUMERIC_FEATURES + [config.TARGET]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Matrice de corrélation")
    return _save(fig, "06_correlations.png")


def run_eda(df: pd.DataFrame) -> list:
    config.ensure_dirs()
    return [
        plot_churn_distribution(df),
        plot_churn_rate_by(df, "contract", "02_churn_by_contract.png", ["Mensuel", "Annuel", "Biannuel"]),
        plot_churn_rate_by(
            df,
            "tenure_group",
            "03_churn_by_tenure.png",
            ["0-6 mois", "7-12 mois", "13-24 mois", "25-48 mois", "49+ mois"],
        ),
        plot_churn_rate_by(df, "payment_method", "04_churn_by_payment.png"),
        plot_numeric_distributions(df),
        plot_correlations(df),
    ]
