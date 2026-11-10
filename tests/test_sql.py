from churn import sql_analysis


def test_split_queries():
    text = "-- name: a\nSELECT 1;\n\n-- name: b\nSELECT 2\nFROM t;\n"
    queries = sql_analysis.split_queries(text)
    assert list(queries) == ["a", "b"]
    assert queries["b"] == "SELECT 2\nFROM t"


def test_kpi_queries_run(clean_df, tmp_path):
    db = tmp_path / "test.db"
    sql_analysis.load_to_sqlite(clean_df, db)
    results = sql_analysis.run_kpi_queries(db)

    assert {"global_kpis", "churn_by_contract", "churn_by_tenure_group", "top_risk_regions"} <= set(results)
    g = results["global_kpis"].iloc[0]
    assert g["nb_clients"] == len(clean_df)
    assert abs(g["taux_churn_pct"] - 100 * clean_df["churn"].mean()) < 0.01

    by_contract = results["churn_by_contract"].set_index("contract")["taux_churn_pct"]
    assert by_contract["Mensuel"] > by_contract["Biannuel"]
