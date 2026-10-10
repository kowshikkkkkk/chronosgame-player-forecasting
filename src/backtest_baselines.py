"""Rolling-origin backtest of baseline forecasts on the cleaned daily player series.

Run from the project root (e.g. /e/chronosgame):
    pip install statsmodels
    python src/backtest_baselines.py                  # target = peak_players
    python src/backtest_baselines.py avg_players      # or the daily average

Models: seasonal naive (repeat the last 7 days), ETS (Holt-Winters, damped trend, weekly
seasonality, on log scale), SARIMAX ((1,1,1)(1,0,1,7), on log scale), and an ensemble that
is simply the average of the ETS and SARIMAX forecasts (no extra fitting).

How the test works: for each game, train on all days up to a cutoff, forecast the next 14
days, score against what really happened, then move the cutoff forward 14 days and repeat
(first cutoff after 365 days of history). Time order is always respected.

Metrics: MASE (error relative to a seasonal-naive forecast made inside the training data;
below 1 is good, comparable across games), sMAPE (percent error), RMSE (players).

Outputs: data/processed/backtest_<target>.csv            one row per game/model/cutoff
         data/processed/backtest_<target>_by_day.csv     scaled error for forecast day 1..14
         data/processed/backtest_<target>_forecasts.csv  every forecast next to the actual value
                                                         (used by intervals_and_charts.py)
"""
import sys
import warnings

import numpy as np
import pandas as pd

H = 14          # forecast horizon, days
STEP = 14       # days between cutoffs
INITIAL = 365   # days of history before the first cutoff
M = 7           # weekly seasonality
LOCKDOWN_START = pd.Timestamp("2020-03-01")
BUCKETS = [("days 1-3", 1, 3), ("days 4-7", 4, 7), ("days 8-14", 8, 14)]


# ---------- models: each takes (train: np.ndarray, h) and returns h forecasts ----------
def seasonal_naive(train, h):
    return np.resize(train[-M:], h)


def ets(train, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    fit = ExponentialSmoothing(np.log(train), trend="add", damped_trend=True, seasonal="add",
                               seasonal_periods=M, initialization_method="estimated").fit()
    return np.exp(np.asarray(fit.forecast(h)))


def sarimax(train, h):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    fit = SARIMAX(np.log(train), order=(1, 1, 1), seasonal_order=(1, 0, 1, M),
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return np.exp(np.asarray(fit.forecast(h)))


MODELS = {"seasonal_naive": seasonal_naive, "ets": ets, "sarimax": sarimax}


# ---------- metrics ----------
def scale_of(train):
    return float(np.mean(np.abs(train[M:] - train[:-M])))


def smape(y, f):
    return float(np.mean(2 * np.abs(y - f) / (np.abs(y) + np.abs(f))))


def rmse(y, f):
    return float(np.sqrt(np.mean((y - f) ** 2)))


def read_daily(path):
    d = pd.read_csv(path)
    try:
        d["date"] = pd.to_datetime(d["date"], format="%Y-%m-%d")
    except ValueError:  # re-saved by Excel
        d["date"] = pd.to_datetime(d["date"], format="%d-%m-%Y")
    return d.sort_values(["game", "date"])


def forecast_all(train, skipped, game, cutoff):
    """Forecasts from every available model for one cutoff, plus the ensemble."""
    fcs = {}
    for name, fn in MODELS.items():
        if name in skipped:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fcs[name] = np.asarray(fn(train, H), float)
        except ImportError:
            print(f"[skip] {name}: install statsmodels (pip install statsmodels)")
            skipped.add(name)
        except Exception as e:  # one bad fit should not stop the run
            print(f"[warn] {game} {name} cutoff {cutoff.date()}: {e}")
    if "ets" in fcs and "sarimax" in fcs:
        fcs["ensemble"] = (fcs["ets"] + fcs["sarimax"]) / 2
    return fcs


def main(target="peak_players", path="data/processed/daily_clean.csv"):
    daily = read_daily(path)
    rows, by_day = [], []
    skipped = set()
    for game, g in daily.groupby("game"):
        y = g[target].to_numpy(float)
        dates = g["date"].to_numpy()
        for origin in range(INITIAL, len(y) - H + 1, STEP):
            train, actual = y[:origin], y[origin:origin + H]
            cutoff = pd.Timestamp(dates[origin])
            scale = scale_of(train)
            for name, f in forecast_all(train, skipped, game, cutoff).items():
                abs_err = np.abs(actual - f)
                rows.append(dict(game=game, model=name, cutoff=cutoff,
                                 mase=float(abs_err.mean() / scale),
                                 smape=smape(actual, f), rmse=rmse(actual, f)))
                for k in range(H):
                    by_day.append(dict(game=game, model=name, cutoff=cutoff, day=k + 1,
                                       date=cutoff + pd.Timedelta(days=k),
                                       forecast=float(f[k]), actual=float(actual[k]),
                                       scaled_err=float(abs_err[k] / scale)))
        print(f"done {game}")

    res, day = pd.DataFrame(rows), pd.DataFrame(by_day)
    res.to_csv(f"data/processed/backtest_{target}.csv", index=False)
    day.to_csv(f"data/processed/backtest_{target}_by_day.csv", index=False)
    day[["game", "model", "cutoff", "day", "date", "forecast", "actual"]].to_csv(
        f"data/processed/backtest_{target}_forecasts.csv", index=False)

    pd.set_option("display.width", 140)
    order = [m for m in ["seasonal_naive", "ets", "sarimax", "ensemble"] if m in set(res.model)]
    n_folds = res[res.model == "seasonal_naive"].groupby("game").size().iloc[0]
    print(f"\nTarget: {target} | {n_folds} cutoffs per game, {H}-day horizon")
    print("\nMean MASE (lower is better):")
    print(res.pivot_table(index="game", columns="model", values="mase")[order].round(3).to_string())
    print("\nMean sMAPE (percent error):")
    print((res.pivot_table(index="game", columns="model", values="smape")[order] * 100).round(1).to_string())

    # --- does the ensemble help? ---
    if "ensemble" in order:
        mm = res.pivot_table(index="game", columns="model", values="mase")
        best_single = mm[["ets", "sarimax"]].min(axis=1)
        print("\nEnsemble vs the better of ETS / SARIMAX (MASE; negative = ensemble better):")
        print(((mm["ensemble"] / best_single - 1) * 100).round(1).rename("% difference").to_string())
        f = res.pivot_table(index=["game", "cutoff"], columns="model", values="mase")
        print("\nShare of cutoffs where the ensemble beats BOTH ETS and SARIMAX:")
        print(((f["ensemble"] < f[["ets", "sarimax"]].min(axis=1)).groupby("game").mean()).round(2).to_string())

    # --- accuracy by forecast day ---
    print("\nMean scaled error by how far ahead (average over the five games; lower is better):")
    tab = {}
    for label, lo, hi in BUCKETS:
        sub = day[(day.day >= lo) & (day.day <= hi)]
        tab[label] = sub.groupby(["game", "model"]).scaled_err.mean().groupby("model").mean()
    print(pd.DataFrame(tab).loc[order].round(3).to_string())
    print("\nSame, per game, days 8-14 only (the hardest stretch):")
    sub = day[day.day >= 8]
    print(sub.pivot_table(index="game", columns="model", values="scaled_err")[order].round(3).to_string())

    late = res[res.cutoff >= LOCKDOWN_START]
    if len(late):
        print(f"\nMean MASE for cutoffs from {LOCKDOWN_START.date()} (lockdown period):")
        print(late.pivot_table(index="game", columns="model", values="mase")[order].round(3).to_string())
    print(f"\nwrote data/processed/backtest_{target}.csv, _by_day.csv and _forecasts.csv")


if __name__ == "__main__":
    main(*sys.argv[1:3])