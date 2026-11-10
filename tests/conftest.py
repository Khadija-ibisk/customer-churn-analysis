import pytest

from churn import config, data_generation, preprocessing


@pytest.fixture(scope="session")
def raw_df():
    return data_generation.generate_customers(n=1500, seed=0)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return preprocessing.clean(raw_df)


@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """Redirige toutes les sorties du projet vers un dossier temporaire."""
    reports = tmp_path / "reports"
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    monkeypatch.setattr(config, "FIGURES_DIR", reports / "figures")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    config.ensure_dirs()
    return tmp_path
