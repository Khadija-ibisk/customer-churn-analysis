import pandas as pd

from churn import config, data_generation, preprocessing


def test_generation_is_reproducible():
    a = data_generation.generate_customers(n=300, seed=7)
    b = data_generation.generate_customers(n=300, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_raw_data_contains_quality_issues(raw_df):
    assert raw_df[config.ID_COL].duplicated().any()
    assert raw_df["age"].isna().any()
    assert (raw_df["total_charges"] == " ").any()
    assert not raw_df["contract"].isin(["Mensuel", "Annuel", "Biannuel"]).all()


def test_clean_removes_duplicates_and_missing(clean_df):
    assert not clean_df[config.ID_COL].duplicated().any()
    assert clean_df.isna().sum().sum() == 0


def test_clean_normalises_categories(clean_df):
    assert set(clean_df["contract"]) == {"Mensuel", "Annuel", "Biannuel"}
    assert set(clean_df["internet_service"]) == {"Fibre", "ADSL", "Aucun"}


def test_clean_handles_outliers(raw_df, clean_df):
    assert raw_df["monthly_charges"].max() > 1000
    assert clean_df["monthly_charges"].max() < 200


def test_target_is_binary(clean_df):
    assert set(clean_df[config.TARGET].unique()) <= {0, 1}
    assert 0.1 < clean_df[config.TARGET].mean() < 0.6


def test_features_are_created(clean_df):
    for col in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES:
        assert col in clean_df.columns
    assert (clean_df["avg_charge_per_month"] > 0).all()


def test_quality_report(raw_df, clean_df):
    report = preprocessing.quality_report(raw_df, clean_df)
    assert report["lignes_brutes"] - report["doublons_supprimes"] == report["lignes_nettoyees"]
    assert report["valeurs_manquantes_restantes"] == 0
