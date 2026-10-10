-- How much higher is the average daily peak on weekends (Sat, Sun) than on weekdays?
-- dow: 0 = Sunday ... 6 = Saturday (set when the table is loaded).
DROP TABLE IF EXISTS insight_weekend;
CREATE TABLE insight_weekend AS
WITH p AS (
    SELECT game,
           AVG(CASE WHEN dow IN (0, 6) THEN peak_players END) AS weekend_peak,
           AVG(CASE WHEN dow NOT IN (0, 6) THEN peak_players END) AS weekday_peak
    FROM daily
    GROUP BY game
)
SELECT game,
       ROUND(weekday_peak, 0) AS avg_weekday_peak,
       ROUND(weekend_peak, 0) AS avg_weekend_peak,
       ROUND(100.0 * (weekend_peak / weekday_peak - 1), 1) AS weekend_lift_pct
FROM p
ORDER BY weekend_lift_pct DESC;