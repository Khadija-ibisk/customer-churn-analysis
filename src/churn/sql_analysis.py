"""Chargement des données nettoyées dans SQLite et exécution des requêtes KPI."""
import sqlite3

import pandas as pd

from . import config


def load_to_sqlite(df: pd.DataFrame, db_path=config.DB_FILE) -> None:
    with sqlite3.connect(db_path) as conn:
        df.to_sql("customers", conn, if_exists="replace", index=False)


def split_queries(sql_text: str) -> dict:
    """Découpe un fichier SQL en requêtes nommées via des commentaires `-- name: xxx`."""
    queries, name, buf = {}, None, []
    for line in sql_text.splitlines():
        if line.strip().lower().startswith("-- name:"):
            if name and "".join(buf).strip():
                queries[name] = "\n".join(buf).strip().rstrip(";")
            name, buf = line.split(":", 1)[1].strip(), []
        elif name:
            buf.append(line)
    if name and "".join(buf).strip():
        queries[name] = "\n".join(buf).strip().rstrip(";")
    return queries


def run_kpi_queries(db_path=config.DB_FILE) -> dict:
    sql_text = (config.SQL_DIR / "kpis.sql").read_text(encoding="utf-8")
    results = {}
    with sqlite3.connect(db_path) as conn:
        for name, query in split_queries(sql_text).items():
            results[name] = pd.read_sql_query(query, conn)
    return results
