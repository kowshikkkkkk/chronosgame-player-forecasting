-- 7-day rolling average of the daily peak, and week-over-week change (vs the same weekday last week).
DROP TABLE IF EXISTS insight_rolling;
CREATE TABLE insight_rolling AS
SELECT game, date, peak_players,
       CASE WHEN COUNT(peak_players) OVER (PARTITION BY game ORDER BY date
                                           ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) = 7
            THEN ROUND(AVG(peak_players) OVER (PARTITION BY game ORDER BY date
                                               ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 0)
       END AS roll7,
       ROUND(100.0 * (peak_players / LAG(peak_players, 7) OVER (PARTITION BY game ORDER BY date) - 1), 1)
           AS wow_pct
FROM daily
ORDER BY game, date;