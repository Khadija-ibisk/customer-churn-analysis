-- Requêtes KPI exécutées automatiquement par le pipeline (SQLite).
-- Chaque requête est identifiée par un commentaire "-- name: <nom>".

-- name: global_kpis
SELECT
    COUNT(*)                                AS nb_clients,
    ROUND(100.0 * AVG(churn), 2)            AS taux_churn_pct,
    ROUND(AVG(monthly_charges), 2)          AS panier_mensuel_moyen,
    ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END), 2) AS revenu_mensuel_perdu
FROM customers;

-- name: churn_by_contract
SELECT
    contract,
    COUNT(*)                         AS nb_clients,
    ROUND(100.0 * AVG(churn), 2)     AS taux_churn_pct,
    ROUND(AVG(tenure_months), 1)     AS anciennete_moyenne
FROM customers
GROUP BY contract
ORDER BY taux_churn_pct DESC;

-- name: churn_by_tenure_group
SELECT
    tenure_group,
    COUNT(*)                         AS nb_clients,
    ROUND(100.0 * AVG(churn), 2)     AS taux_churn_pct
FROM customers
GROUP BY tenure_group
ORDER BY MIN(tenure_months);

-- name: churn_by_payment
SELECT
    payment_method,
    COUNT(*)                         AS nb_clients,
    ROUND(100.0 * AVG(churn), 2)     AS taux_churn_pct
FROM customers
GROUP BY payment_method
ORDER BY taux_churn_pct DESC;

-- name: top_risk_regions
WITH region_stats AS (
    SELECT
        region,
        COUNT(*)                      AS nb_clients,
        AVG(churn)                    AS churn_rate,
        SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END) AS revenu_perdu
    FROM customers
    GROUP BY region
)
SELECT
    region,
    nb_clients,
    ROUND(100.0 * churn_rate, 2)      AS taux_churn_pct,
    ROUND(revenu_perdu, 2)            AS revenu_mensuel_perdu,
    RANK() OVER (ORDER BY revenu_perdu DESC) AS rang_impact
FROM region_stats
ORDER BY rang_impact;
