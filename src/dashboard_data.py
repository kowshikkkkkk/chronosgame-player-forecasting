"""Data preparation for the Streamlit dashboard.

Pure pandas (no Streamlit/Altair imports) so it can be tested on its own.
Reads the files produced by the earlier steps in data/processed/:
    daily_clean.csv, backtest_<target>.csv, backtest_<target>_by_day.csv,
    backtest_<target>_intervals.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from intervals_and_charts import read_dates, summarise

TARGET = "peak_players"
LOCKDOWN_START = pd.Timestamp("2020-03-01")
MODEL_ORDER = ["seasonal_naive", "ets", "sarimax", "ensemble"]
MODEL_LABELS = {"seasonal_naive": "Seasonal naive", "ets": "ETS", "sarimax": "SARIMAX",
                "ensemble": "Ensemble"}


def load(proc, target=TARGET):
    """Read every table the dashboard needs. Raises FileNotFoundError with a helpful message."""
    proc = Path(proc)
    names = {"daily": "daily_clean.csv",
             "backtest": f"backtest_{target}.csv",
             "by_day": f"backtest_{target}_by_day.csv",
             "intervals": f"backtest_{target}_intervals.csv"}
    out = {}
    for key, fname in names.items():
        path = proc / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run clean_steam.py, backtest_baselines.py and "
                                    f"intervals_and_charts.py first.")
        out[key] = pd.read_csv(path)
    out["daily"]["date"] = read_dates(out["daily"], "date")
    out["backtest"]["cutoff"] = read_dates(out["backtest"], "cutoff")
    for key in ("by_day", "intervals"):
        out[key]["cutoff"] = read_dates(out[key], "cutoff")
        out[key]["date"] = read_dates(out[key], "date")
    return out


INSIGHT_FILES = {"lockdown": "insight_lockdown.csv", "sale_effect": "insight_sale_effect.csv",
                 "sale_summary": "insight_sale_summary.csv", "weekend": "insight_weekend.csv",
                 "monthly": "insight_monthly.csv", "rolling": "insight_rolling.csv",
                 "dips": "insight_dips.csv"}
DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]     # dow 0 = Sunday, as in the SQL


def load_insights(proc):
    """Tables written by run_sql.py, or None if it has not been run yet."""
    proc = Path(proc)
    out = {}
    for key, fname in INSIGHT_FILES.items():
        path = proc / fname
        if not path.exists():
            return None
        out[key] = pd.read_csv(path)
    out["sale_effect"]["sale_start"] = pd.to_datetime(out["sale_effect"]["sale_start"])
    out["monthly"]["month"] = pd.to_datetime(out["monthly"]["month"] + "-01")
    out["rolling"]["date"] = read_dates(out["rolling"], "date")
    return out


def dips_grid(dips, game):
    """Full 7 x 24 grid (weekday by hour) of flagged dip hours for one game, zeros filled in."""
    g = dips[dips.game == game].set_index(["dow", "hod"]).dip_hours
    full = pd.MultiIndex.from_product([range(7), range(24)], names=["dow", "hod"])
    out = g.reindex(full, fill_value=0).reset_index()
    out["day"] = out.dow.map(lambda i: DAY_NAMES[i])
    return out


def game_list(intervals):
    return sorted(intervals.game.unique())


def cutoffs_for(intervals, game):
    return sorted(pd.Timestamp(c) for c in intervals.loc[intervals.game == game, "cutoff"].unique())


def nearest(options, target_date):
    return min(options, key=lambda d: abs(d - pd.Timestamp(target_date)))


def forecast_window(intervals, daily, game, cutoff, history_days=42, target=TARGET):
    """History before the cutoff, plus the 14 forecast rows (forecast, ranges, actual)."""
    cutoff = pd.Timestamp(cutoff)
    f = intervals[(intervals.game == game) & (intervals.cutoff == cutoff)].sort_values("date")
    series = daily[daily.game == game].set_index("date")[target]
    hist = series[(series.index >= cutoff - pd.Timedelta(days=history_days)) & (series.index < cutoff)]
    hist = hist.rename("players").rename_axis("date").reset_index()
    return hist, f.reset_index(drop=True)


def window_kpis(f):
    """Headline numbers for one 14-day forecast window."""
    forecast_peak = float(f.forecast.max())
    provision = float(f.hi95.max())          # one capacity level that covers the whole fortnight
    actual_peak = float(f.actual.max())
    return dict(forecast_peak=forecast_peak, provision=provision, actual_peak=actual_peak,
                buffer=provision / forecast_peak - 1,
                enough=actual_peak <= provision,
                shortfall=max(0.0, actual_peak / provision - 1),
                days_over=int((f.actual > f.hi95).sum()),
                days_outside80=int(((f.actual < f.lo80) | (f.actual > f.hi80)).sum()),
                n_days=len(f))


def line_frame(hist, f):
    """Long-format frame for the forecast chart (one row per point, a 'series' column)."""
    parts = [hist.rename(columns={"players": "value"}).assign(series="Earlier actual"),
             f[["date", "actual"]].rename(columns={"actual": "value"}).assign(series="Actual"),
             f[["date", "forecast"]].rename(columns={"forecast": "value"}).assign(series="Forecast"),
             f[["date", "hi95"]].rename(columns={"hi95": "value"}).assign(series="95% upper bound")]
    return pd.concat(parts, ignore_index=True)


def _ordered(cols):
    return [m for m in MODEL_ORDER if m in cols]


def accuracy_table(backtest, metric, since=None):
    """Mean MASE (or sMAPE as a percentage) per game and model, plus the best model."""
    b = backtest if since is None else backtest[backtest.cutoff >= pd.Timestamp(since)]
    t = b.pivot_table(index="game", columns="model", values=metric)
    t = t[_ordered(t.columns)]
    if metric == "smape":
        t = t * 100
    best = t.idxmin(axis=1).map(MODEL_LABELS)
    t.columns = [MODEL_LABELS[c] for c in t.columns]
    t = t.round(3 if metric == "mase" else 1)
    t["Best"] = best
    return t.reset_index().rename(columns={"game": "Game"})


def error_by_day(by_day, since=None):
    b = by_day if since is None else by_day[by_day.cutoff >= pd.Timestamp(since)]
    m = b.groupby(["model", "day"]).scaled_err.mean().reset_index()
    m["Model"] = m.model.map(MODEL_LABELS)
    return m.rename(columns={"day": "Days ahead", "scaled_err": "Error"})[["Model", "Days ahead", "Error"]]


def capacity_table(intervals):
    """Calibration and capacity buffer by game and forecast window (percent)."""
    t, all_ = summarise(intervals)
    t = t.reset_index()
    all_ = all_.reset_index().assign(game="All five games")
    t = pd.concat([t, all_], ignore_index=True)
    return pd.DataFrame({"Game": t.game, "Window": t.window,
                         "Inside 80% range (%)": (t.inside80 * 100).round(0),
                         "Above 95% bound (%)": (t.over95 * 100).round(1),
                         "Buffer above forecast (%)": (t.headroom * 100).round(0)})


def data_quality(daily):
    """Per game: how much of the history was rebuilt or flagged during cleaning."""
    g = daily.groupby("game").agg(days=("date", "count"), first=("date", "min"), last=("date", "max"),
                                  filled=("hours_filled", "sum"), dips=("dip_hours", "sum"))
    g["Hours rebuilt (%)"] = (g.filled / (g.days * 24) * 100).round(1)
    g["Unusual-dip hours"] = g.dips.astype(int)
    g["From"] = g["first"].dt.strftime("%d %b %Y")
    g["To"] = g["last"].dt.strftime("%d %b %Y")
    g["Days"] = g.days
    return g[["Days", "From", "To", "Hours rebuilt (%)", "Unusual-dip hours"]].reset_index().rename(
        columns={"game": "Game"})