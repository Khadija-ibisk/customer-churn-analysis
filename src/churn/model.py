"""Modélisation : comparaison de modèles, évaluation et sauvegarde."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ]
    )


def candidate_models() -> dict:
    rs = config.RANDOM_STATE
    return {
        "Régression logistique": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=rs, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=rs),
    }


def split(df: pd.DataFrame):
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = df[config.TARGET]
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)


def compare_models(X_train, y_train) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    rows = []
    for name, clf in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        rows.append({"modele": name, "roc_auc_moyen": scores.mean(), "roc_auc_std": scores.std()})
    return pd.DataFrame(rows).sort_values("roc_auc_moyen", ascending=False).reset_index(drop=True)


def best_threshold(y_true, proba) -> float:
    """Seuil maximisant le F1 (utile car les classes sont déséquilibrées)."""
    thresholds = np.linspace(0.1, 0.9, 81)
    f1s = [f1_score(y_true, (proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(f1s))])


def feature_importance(pipe: Pipeline, top: int = 15) -> pd.DataFrame:
    names = pipe.named_steps["prep"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    values = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else clf.feature_importances_
    imp = pd.DataFrame({"variable": names, "importance": values})
    imp["variable"] = imp["variable"].str.replace(r"^(num|cat)__", "", regex=True)
    return imp.sort_values("importance", ascending=False).head(top).reset_index(drop=True)


def train_and_evaluate(df: pd.DataFrame) -> dict:
    config.ensure_dirs()
    X_train, X_test, y_train, y_test = split(df)

    comparison = compare_models(X_train, y_train)
    best_name = comparison.loc[0, "modele"]
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", candidate_models()[best_name])])
    pipe.fit(X_train, y_train)

    # Seuil choisi sur le jeu d'entraînement (pas de fuite vers le test)
    threshold = best_threshold(y_train, pipe.predict_proba(X_train)[:, 1])
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    metrics = {
        "modele": best_name,
        "seuil": round(threshold, 3),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "taille_test": int(len(y_test)),
    }

    # Figures d'évaluation
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[0], name=best_name)
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set_title("Courbe ROC (jeu de test)")
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Reste", "Part"], ax=axes[1], cmap="Blues", colorbar=False
    )
    axes[1].set_title(f"Matrice de confusion (seuil = {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "07_model_evaluation.png", dpi=120)
    plt.close(fig)

    importance = feature_importance(pipe)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importance["variable"][::-1], importance["importance"][::-1], color="#4C78A8")
    ax.set_title(f"Variables les plus influentes – {best_name}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "08_feature_importance.png", dpi=120)
    plt.close(fig)

    joblib.dump({"pipeline": pipe, "threshold": threshold}, config.MODELS_DIR / "churn_model.joblib")
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"metrics": metrics, "comparison": comparison, "importance": importance}


def predict(df: pd.DataFrame, model_path=None) -> pd.DataFrame:
    """Score de risque de churn pour de nouveaux clients (données déjà nettoyées)."""
    bundle = joblib.load(model_path or config.MODELS_DIR / "churn_model.joblib")
    proba = bundle["pipeline"].predict_proba(df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES])[:, 1]
    out = df[[config.ID_COL]].copy()
    out["proba_churn"] = proba.round(4)
    out["a_risque"] = (proba >= bundle["threshold"]).astype(int)
    return out.sort_values("proba_churn", ascending=False)
