"""Pipeline de bout en bout : données -> nettoyage -> SQL -> EDA -> modèle -> rapport."""
import time

import pandas as pd

from . import config, data_generation, eda, model, preprocessing, sql_analysis


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Conversion DataFrame -> tableau Markdown, sans dépendance externe."""
    def fmt(v):
        return floatfmt.format(v) if isinstance(v, float) else str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def write_report(quality: dict, kpis: dict, figures: list, results: dict) -> None:
    m = results["metrics"]
    g = kpis["global_kpis"].iloc[0]
    lines = [
        "# Rapport d'analyse – Churn clients télécom",
        "",
        "_Rapport généré automatiquement par `python run_pipeline.py`._",
        "",
        "## 1. Synthèse",
        "",
        f"- **{int(g['nb_clients'])} clients** analysés, taux de churn de **{g['taux_churn_pct']} %**.",
        f"- Revenu mensuel perdu à cause du churn : **{g['revenu_mensuel_perdu']:,.0f} €**.",
        f"- Meilleur modèle : **{m['modele']}** – ROC AUC = **{m['roc_auc']}**, "
        f"rappel = **{m['recall']:.0%}** (part des clients partants détectés).",
        "",
        "## 2. Qualité des données",
        "",
        to_markdown(pd.DataFrame([quality])),
        "",
        "## 3. Indicateurs clés (requêtes SQL)",
        "",
    ]
    for name, table in kpis.items():
        lines += [f"### {name}", "", to_markdown(table, "{:.2f}"), ""]

    lines += ["## 4. Analyse exploratoire", ""]
    lines += [f"![{f}]({f})" for f in figures]
    lines += [
        "",
        "## 5. Modélisation",
        "",
        "Comparaison des modèles (validation croisée 5 plis, ROC AUC) :",
        "",
        to_markdown(results["comparison"]),
        "",
        "Performance du modèle retenu sur le jeu de test :",
        "",
        to_markdown(pd.DataFrame([m])),
        "",
        "![Évaluation](figures/07_model_evaluation.png)",
        "![Importance](figures/08_feature_importance.png)",
        "",
        "## 6. Recommandations métier",
        "",
        "1. **Inciter au passage en contrat annuel/biannuel** : le contrat mensuel est le premier facteur de churn.",
        "2. **Soigner les 12 premiers mois** : programme d'accueil et suivi proactif des nouveaux clients.",
        "3. **Traiter les appels au support répétés** : au-delà de 3 appels, déclencher une action de rétention.",
        "4. **Encourager le prélèvement automatique** : les payeurs par chèque partent nettement plus.",
        "5. **Cibler les campagnes** avec le score du modèle (`models/churn_model.joblib`).",
        "",
    ]
    (config.REPORTS_DIR / "rapport.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    t0 = time.time()
    config.ensure_dirs()

    print("1/5 Génération des données brutes…")
    raw = data_generation.generate_customers()
    raw.to_csv(config.RAW_FILE, index=False)

    print("2/5 Nettoyage et ingénierie de variables…")
    raw = pd.read_csv(config.RAW_FILE)
    df = preprocessing.clean(raw)
    df.to_csv(config.CLEAN_FILE, index=False)
    quality = preprocessing.quality_report(raw, df)

    print("3/5 Chargement SQLite et requêtes KPI…")
    sql_analysis.load_to_sqlite(df, config.DB_FILE)
    kpis = sql_analysis.run_kpi_queries(config.DB_FILE)

    print("4/5 Analyse exploratoire…")
    figures = eda.run_eda(df)

    print("5/5 Entraînement et évaluation des modèles…")
    results = model.train_and_evaluate(df)

    write_report(quality, kpis, figures, results)
    print(f"\nTerminé en {time.time() - t0:.1f}s – métriques : {results['metrics']}")
    print(f"Rapport : {config.REPORTS_DIR / 'rapport.md'}")
    return results
