-- checking number and names of tables present --
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public';
-- 70 tables present -- 

-- Tables with records present --
SELECT COUNT(*) FROM account_balance;-- 10 records present --
SELECT COUNT(*) FROM assets; -- 4 records present --
SELECT COUNT(*) FROM "plans"; -- 15 records present --
SELECT COUNT(*) FROM investment_plans; -- 10 records present --
SELECT COUNT(*) FROM categories; -- 5 records present --
SELECT COUNT(*) FROM transactions; -- 153 records present --
SELECT COUNT(*) FROM users; --10 records present --

-- investigating --
SELECT *
FROM users;

SELECT * 
FROM account_balance;

SELECT * 
FROM assets;

SELECT * 
FROM "plans";

SELECT * 
FROM categories;

SELECT *
FROM transactions;

SELECT *
FROM investment_plans;

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'account_balance';

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'assets';

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'plans';

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'investment_plans';

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'categories';

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'transactions';

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'users';



-- Extracting columns from fields --
WITH first_transactions AS (
    SELECT 
        COALESCE(p.user_id, ip.user_id) AS user_id,
        MIN(t.created_at) AS first_transaction_date
    FROM transactions t
    LEFT JOIN investment_plans ip ON ip.id = t.investment_plan_id
    LEFT JOIN plans p ON p.id = t.plan_id
    WHERE t.status = 'success'
    GROUP BY COALESCE(p.user_id, ip.user_id)
)

SELECT 
    t.id AS transaction_id,
    CASE 
        WHEN t.metadata::text ILIKE '%Monthly maintenance fee deduction%' THEN 'maintenance_fee'
        ELSE t.tx_type
    END AS transaction_type,
	
    u.id AS customer_id,
    u.first_name || ' ' || u.last_name AS customer_name,
    ft.first_transaction_date,
    TO_CHAR(t.created_at, 'YYYY-MM') AS transaction_cohort,
    TO_CHAR(u.created_at, 'YYYY-MM') AS sign_up_cohort,
    TO_CHAR(u.updated_at, 'YYYY-MM') AS kyc_cohort,
    EXTRACT(WEEK FROM t.created_at) AS week_number,
    
    t.created_at AS transaction_date,
    t.amount AS ghs_amount,
    t.usd_amount,
    t.exchange_rate,

    u.created_at AS sign_up_date,
    u.updated_at AS kyc_completed_date,

    CASE 
        WHEN t.investment_plan_id IS NOT NULL THEN ip.plan_option
        WHEN t.plan_id IS NOT NULL THEN p.plan_option
        ELSE NULL
    END AS investment_type,
    
    a.name AS asset_type,
    a.maturity_date AS assets_maturity_date,
    ip.maturity_date AS investment_maturity_date

FROM transactions t
LEFT JOIN investment_plans ip ON ip.id = t.investment_plan_id
LEFT JOIN plans p ON p.id = t.plan_id
LEFT JOIN users u ON u.id = COALESCE(p.user_id, ip.user_id)
LEFT JOIN first_transactions ft ON ft.user_id = u.id
LEFT JOIN assets a ON a.id = ip.asset_id
WHERE t.status = 'success';


