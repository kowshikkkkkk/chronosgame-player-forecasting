-- Does a sale move the daily peak? For every sale start, compare the average daily peak over the
-- first 3 days of the sale with the average over the 7 days before it.
-- Only sales with a full 7 days before and 3 days after inside the price data (Apr 2019 - Aug 2020).
-- LEFT JOIN keeps every calendar day, so the 7-day and 3-day windows are real calendar windows even
-- where the price file has a missing day (it lacks 2020-07-18). A day with no price row is never a sale start.
DROP TABLE IF EXISTS insight_sale_effect;
CREATE TABLE insight_sale_effect AS
WITH j AS (
    SELECT d.game, d.date, d.peak_players, p.discount,
           LAG(p.discount) OVER (PARTITION BY d.game ORDER BY d.date) AS prev_discount,
           AVG(d.peak_players) OVER (PARTITION BY d.game ORDER BY d.date
                                     ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS prior7,
           COUNT(d.peak_players) OVER (PARTITION BY d.game ORDER BY d.date
                                       ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS n_prior,
           AVG(d.peak_players) OVER (PARTITION BY d.game ORDER BY d.date
                                     ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING) AS first3,
           COUNT(d.peak_players) OVER (PARTITION BY d.game ORDER BY d.date
                                       ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING) AS n_first
    FROM daily d
    LEFT JOIN prices p ON p.appid = d.appid AND p.date = d.date
)
SELECT game, date AS sale_start, discount,
       ROUND(prior7, 0) AS prior7_peak, ROUND(first3, 0) AS first3_peak,
       ROUND(100.0 * (first3 / prior7 - 1), 1) AS lift_pct
FROM j
WHERE discount > 0 AND prev_discount = 0 AND n_prior = 7 AND n_first = 3
ORDER BY game, sale_start;

DROP TABLE IF EXISTS insight_sale_summary;
CREATE TABLE insight_sale_summary AS
SELECT game,
       COUNT(*) AS sales,
       ROUND(AVG(lift_pct), 1) AS avg_lift_pct,
       ROUND(100.0 * SUM(CASE WHEN lift_pct > 0 THEN 1 ELSE 0 END) / COUNT(*), 0) AS share_positive_pct,
       MIN(lift_pct) AS worst_pct,
       MAX(lift_pct) AS best_pct
FROM insight_sale_effect
GROUP BY game
ORDER BY game;