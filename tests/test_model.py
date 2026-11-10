from churn import config, eda, model


def test_model_training_and_prediction(clean_df, tmp_project):
    results = model.train_and_evaluate(clean_df)
    m = results["metrics"]

    assert m["roc_auc"] > 0.65
    assert 0 < m["recall"] <= 1
    assert len(results["comparison"]) == 3
    assert (config.MODELS_DIR / "churn_model.joblib").exists()
    assert (config.REPORTS_DIR / "metrics.json").exists()

    scores = model.predict(clean_df.head(50))
    assert len(scores) == 50
    assert scores["proba_churn"].between(0, 1).all()
    assert set(scores["a_risque"].unique()) <= {0, 1}


def test_eda_creates_figures(clean_df, tmp_project):
    figures = eda.run_eda(clean_df)
    assert len(figures) == 6
    for f in figures:
        assert (config.REPORTS_DIR / f).exists()
