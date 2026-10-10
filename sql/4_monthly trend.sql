-- Average daily peak per month, with month-over-month change and the change since the first month.
-- The first (Dec 2017) and last (Aug 2020) months are partial.
DROP TABLE IF EXISTS insight_monthly;
CREATE TABLE insight_monthly AS
WITH m AS (
    SELECT game, month, AVG(peak_players) AS avg_peak
    FROM daily
    GROUP BY game, month
)
SELECT game, month,
       ROUND(avg_peak, 0) AS avg_peak,
       ROUND(100.0 * (avg_peak / LAG(avg_peak) OVER (PARTITION BY game ORDER BY month) - 1), 1) AS mom_pct,
       ROUND(100.0 * (avg_peak / FIRST_VALUE(avg_peak) OVER (PARTITION BY game ORDER BY month) - 1), 1)
           AS vs_first_month_pct
FROM m
ORDER BY game, month;