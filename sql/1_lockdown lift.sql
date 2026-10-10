-- How much higher was each game's average daily peak in Mar-Jun 2020 than in Mar-Jun 2019?
-- (Same months a year earlier, so the weekly/seasonal shape is comparable. It still includes each
-- game's own underlying trend, e.g. PUBG was declining all along.)
DROP TABLE IF EXISTS insight_lockdown;
CREATE TABLE insight_lockdown AS
WITH p AS (
    SELECT game,
           AVG(CASE WHEN date >= '2019-03-01' AND date < '2019-07-01' THEN peak_players END) AS peak_2019,
           AVG(CASE WHEN date >= '2020-03-01' AND date < '2020-07-01' THEN peak_players END) AS peak_2020
    FROM daily
    GROUP BY game
)
SELECT game,
       ROUND(peak_2019, 0) AS avg_peak_mar_jun_2019,
       ROUND(peak_2020, 0) AS avg_peak_mar_jun_2020,
       ROUND(100.0 * (peak_2020 / peak_2019 - 1), 1) AS lift_pct
FROM p
ORDER BY lift_pct DESC;