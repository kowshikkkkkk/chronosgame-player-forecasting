"""Clean the hourly Steam player-count data and build daily series.

Run from the project root (e.g. /e/chronosgame):
    python src/clean_steam.py

Reads   steam_sample/steam_hourly.csv
Writes  data/processed/hourly_clean.csv   one row per game per hour, no gaps
        data/processed/daily_clean.csv    one row per game per day

Cleaning rules (applied per game):
  * mean is BAD when n_obs == 0 (no readings) or min == 0 (a failed reading stored as 0
    drags the hour's average down, e.g. 12 readings with one 0 -> mean ~8% too low)
  * max is BAD only when n_obs == 0 or max == 0 (a single 0 does not affect the max)
  * bad stretches of <= 6 hours are filled by straight-line interpolation
  * longer stretches are filled with the average of the same hour one week earlier and
    one week later (keeps the daily/weekly shape), falling back to interpolation
  * every filled hour is flagged, so models and charts can show/discount it
  * separately, hours far below normal (< 60% of the same hour-of-week in the 3 weeks
    before/after) get a `dip` flag. Values are NOT changed: some dips are shared
    collector/Steam outages (a handful of hours), others (PUBG, mostly Wednesday early
    morning) look like real weekly server maintenance.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = {252490: "Rust", 252950: "Rocket League", 271590: "GTA V",
         289070: "Civilization VI", 578080: "PUBG"}
SHORT_GAP = 6      # hours: interpolate gaps up to this long
WEEK = 24 * 7
DIP_RATIO = 0.6   # flag hours below this share of the normal level


def read_hourly(path):
    d = pd.read_csv(path)
    try:  # normal file written by summarize_steam.py
        d["hour"] = pd.to_datetime(d["hour"], format="%Y-%m-%d %H:%M:%S")
    except ValueError:  # file was re-saved by Excel (DD-MM-YYYY HH:MM)
        d["hour"] = pd.to_datetime(d["hour"], format="%d-%m-%Y %H:%M")
    return d


def fill(s, bad):
    """Return (filled series, method label per hour) for series s with bad hours masked."""
    s = s.where(~bad)
    lin = s.interpolate(method="time", limit_area="inside")
    run_id = (s.isna() != s.isna().shift()).cumsum()
    run_len = s.isna().groupby(run_id).transform("sum")           # length of the NaN run each hour sits in
    seasonal = pd.concat([s.shift(WEEK), s.shift(-WEEK)], axis=1).mean(axis=1)

    out = s.copy()
    method = pd.Series("", index=s.index)
    na = s.isna()
    short = na & (run_len <= SHORT_GAP)
    long_ = na & (run_len > SHORT_GAP)
    out[short] = lin[short]
    method[short] = "interpolated"
    out[long_] = seasonal[long_].fillna(lin[long_])
    method[long_] = "seasonal"
    out = out.interpolate(method="time").ffill().bfill()            # safety net for series edges
    return out, method


def clean_game(g, end=None):
    """Clean one game's hourly rows. With `end` (an hour), the series is cut off there: hours with no
    row up to `end` count as missing, and nothing after `end` is ever looked at. The backtest uses this
    to clean the data as it looked on each forecast date."""
    g = g.set_index("hour").sort_index().asfreq("h")
    if end is not None:
        g = g.reindex(pd.date_range(g.index[0], end, freq="h"))
    n_obs = g["n_obs"].fillna(0)
    mean_bad = g["mean"].isna() | (g["min"] <= 0) | (n_obs == 0)
    max_bad = g["max"].isna() | (g["max"] <= 0) | (n_obs == 0)
    players, how = fill(g["mean"], mean_bad)
    peak, _ = fill(g["max"], max_bad)
    normal = pd.concat([players.shift(k * WEEK) for k in (-3, -2, -1, 1, 2, 3)], axis=1).median(axis=1)
    dip = (players < DIP_RATIO * normal) & ~mean_bad
    return pd.DataFrame({"players": players.round(1), "peak": peak.round(1),
                         "imputed": mean_bad, "method": how, "dip": dip})


def main(src="steam_sample/steam_hourly.csv", out_dir="data/processed"):
    raw = read_hourly(src)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hourly = []
    for appid, g in raw.groupby("appid"):
        c = clean_game(g).reset_index()
        c.insert(0, "appid", appid)
        c.insert(1, "game", NAMES.get(appid, str(appid)))
        hourly.append(c)
        n = len(c)
        m = c.method.value_counts()
        print(f"{NAMES.get(appid, appid):16s} hours={n:,}  filled={int(c.imputed.sum()):4d} "
              f"({c.imputed.mean():.1%})  interpolated={m.get('interpolated', 0):4d}  seasonal={m.get('seasonal', 0):3d}  "
              f"dip_hours={int(c.dip.sum())}")
    hourly = pd.concat(hourly, ignore_index=True)
    hourly.to_csv(out / "hourly_clean.csv", index=False)

    daily = (hourly.set_index("hour").groupby(["appid", "game"])
             .resample("D").agg(avg_players=("players", "mean"), peak_players=("peak", "max"),
                                hours_filled=("imputed", "sum"), dip_hours=("dip", "sum"))
             .reset_index().rename(columns={"hour": "date"}))
    daily["avg_players"] = daily.avg_players.round(1)
    daily.to_csv(out / "daily_clean.csv", index=False)

    print(f"\nwrote {out/'hourly_clean.csv'}  ({len(hourly):,} rows)")
    print(f"wrote {out/'daily_clean.csv'}   ({len(daily):,} rows)")
    print(f"days with >6 filled hours: {int((daily.hours_filled > 6).sum())}  "
          f"| NaN left in players/peak: {int(hourly[['players','peak']].isna().sum().sum())}")


if __name__ == "__main__":
    main(*sys.argv[1:3])