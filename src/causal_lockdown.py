"""Causal analysis 1: how much did the March 2020 lockdown change the daily peak? (counterfactual forecast)

Run from the project root (needs statsmodels, like the backtest):
    python src/causal_lockdown.py

Question
    By how much did the daily PEAK of concurrent players differ, after 11 March 2020, from what the
    game's own pre-lockdown pattern would have produced? Reported as an ESTIMATED effect with a range.

Method
  1. Pre-period = 14 Dec 2017 to the day before the intervention. Fit the same ETS as the backtest
     (damped trend, weekly seasonality, on the log scale) on the pre-period only.
  2. Simulate N_PATHS possible futures ("counterfactual paths") from the fitted model for every day
     until 12 Aug 2020. Each path is one answer to "what could the peak have been without lockdown".
  3. For each path: lift = average over the window of (actual / counterfactual - 1).
     Report the median and the 10-90% range over paths, for the first 30 days, for 11 Mar - 30 Jun
     (the months the SQL insight looks at) and for the whole post-period.
  4. Placebo test: repeat steps 1-3 at fake intervention dates inside the pre-period (every 14 days,
     at least 365 days of history, same post-period length as the real one). No lockdown happened
     there, so those "effects" show how big a miss of this method is by pure chance/trend change.
     The real effect is judged against that spread, not against zero.
  5. Sensitivity: repeat step 1-3 for two other plausible dates (1 Mar and 23 Mar).
  Other games are NOT used as controls: all five were hit by the lockdown at the same time.

No leakage: the pre-period series comes from backtest_baselines.train_series, which cleans only the raw
readings from before the cutoff (the cleaning rules look a week ahead, so this matters). The actual
values after the cutoff are used only to score, never to fit. tests/test_causal_no_leakage.py checks it.

Assumptions (stated again in the README)
  * nothing else changed at the same time for these games (patches, price cuts, other releases);
  * without lockdown the pre-period pattern (level, damped trend, weekly shape) would have continued.
  The simulated paths include random day-to-day noise but NOT uncertainty in the fitted parameters, so
  the 10-90% range is probably too narrow; the placebo spread is the honest yardstick.

Outputs (reports/):
    causal_lockdown_effect.csv           real effect per game, date, window
    causal_lockdown_placebo.csv          effect at every fake date
    causal_lockdown_placebo_summary.csv  placebo spread vs the real effect
    causal_lockdown_vs_sql.csv           this method next to the SQL insight numbers
    figures/causal_lockdown_counterfactual.png, figures/causal_lockdown_placebo.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_baselines import INITIAL, M, STEP, read_daily, train_series  # noqa: E402
from clean_steam import NAMES, read_hourly  # noqa: E402
from intervals_and_charts import BLUE, GRID, INK, INK2, MUTED, ORANGE, SURFACE, save_fig, style_axes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TARGET = "peak_players"
INTERVENTION = pd.Timestamp("2020-03-11")                     # parameter
SENSITIVITY = [pd.Timestamp("2020-03-01"), pd.Timestamp("2020-03-23")]
END = pd.Timestamp("2020-08-12")                               # last day of data
JUNE_END = pd.Timestamp("2020-06-30")
EARLY = 30                                                     # "first 30 days" window
N_PATHS = 1000
SEED = 2020
Q_LO, Q_HI = 10, 90


# ---------- the counterfactual model ----------
def ets_simulate(train, h, n_paths, seed):
    """Fit ETS (damped trend, weekly seasonality, log scale) on `train` and simulate n_paths futures.
    Returns (point forecast of length h, paths array of shape (h, n_paths)), both in players."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    fit = ExponentialSmoothing(np.log(train), trend="add", damped_trend=True, seasonal="add",
                               seasonal_periods=M, initialization_method="estimated").fit()
    point = np.exp(np.asarray(fit.forecast(h)))
    sim = np.asarray(fit.simulate(h, anchor="end", repetitions=n_paths, random_state=seed))
    if sim.shape != (h, n_paths):
        sim = sim.reshape(h, n_paths) if sim.size == h * n_paths and sim.shape[0] == h else sim.T
    assert sim.shape == (h, n_paths), f"unexpected simulate() shape {sim.shape}"
    return point, np.exp(sim)


def counterfactual(raw_game, cutoff, end, n_paths, seed, simulate=ets_simulate):
    """Counterfactual for `cutoff`..`end`, using ONLY readings before `cutoff`.
    Returns (training series, dates, point forecast, simulated paths)."""
    train = train_series(raw_game, cutoff, TARGET)
    h = (end - cutoff).days + 1
    point, paths = simulate(train.to_numpy(dtype=float), h, n_paths, seed)
    return train, pd.date_range(cutoff, periods=h), point, paths


def window_effect(actual, point, paths, k):
    """Average lift over the first k days: one value per path, plus the point-forecast version, in %."""
    per_path = (actual[:k, None] / paths[:k]).mean(axis=0) - 1
    return 100 * per_path, 100 * (actual[:k] / point[:k]).mean() - 100


def summarise(lifts):
    return {"median_pct": np.median(lifts), "p10_pct": np.percentile(lifts, Q_LO), "p90_pct": np.percentile(lifts, Q_HI)}


def placebo_origins(n_pre, horizon):
    """Days-from-start of each fake cutoff: every STEP days, >= INITIAL days of history, and the whole
    fake post-period (`horizon` days) must end inside the pre-period."""
    return list(range(INITIAL, n_pre - horizon + 1, STEP))


def windows(cutoff):
    """name -> length in days for the three reporting windows (never longer than the post-period)."""
    full = (END - cutoff).days + 1
    return {"first 30 days": min(EARLY, full), "to 30 Jun": min((JUNE_END - cutoff).days + 1, full), "full post-period": full}


# ---------- main ----------
def run(simulate=ets_simulate, out=ROOT / "reports", n_paths=N_PATHS, n_placebo=None, verbose=True):
    daily = read_daily(ROOT / "data/processed/daily_clean.csv")
    raw = read_hourly(ROOT / "steam_sample/steam_hourly.csv")
    raw_by_game = {NAMES.get(a, str(a)): g for a, g in raw.groupby("appid")}
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    start = daily["date"].min()
    n_pre = (INTERVENTION - start).days
    horizon = (END - INTERVENTION).days + 1
    origins = placebo_origins(n_pre, horizon)
    if n_placebo:
        origins = origins[:n_placebo]
    if verbose:
        print(f"intervention {INTERVENTION.date()} | pre-period {n_pre} days | post-period {horizon} days | "
              f"{len(origins)} placebo cutoffs per game (first {(start + pd.Timedelta(days=origins[0])).date()}, "
              f"last {(start + pd.Timedelta(days=origins[-1])).date()})")

    real_rows, placebo_rows, plot_data = [], [], {}
    for gi, (game, g) in enumerate(sorted(daily.groupby("game"))):
        actual_s = g.set_index("date")[TARGET]
        rg = raw_by_game[game]
        # --- real effect at the main date and the two sensitivity dates ---
        for cutoff in [INTERVENTION] + SENSITIVITY:
            seed = SEED + 1000 * gi + (cutoff - INTERVENTION).days
            train, dates, point, paths = counterfactual(rg, cutoff, END, n_paths, seed, simulate)
            actual = actual_s.reindex(dates).to_numpy(dtype=float)
            last30 = float(train.iloc[-30:].mean())
            for name, k in windows(cutoff).items():
                lifts, point_lift = window_effect(actual, point, paths, k)
                cf_mean = np.median(paths[:k].mean(axis=0))
                real_rows.append({"game": game, "intervention": cutoff.date(), "window": name, "days": k,
                                  **summarise(lifts), "point_forecast_pct": point_lift,
                                  "share_paths_above_zero": float((lifts > 0).mean()),
                                  "actual_avg_peak": actual[:k].mean(), "counterfactual_median_avg_peak": cf_mean,
                                  "last_30_pre_days_avg_peak": last30,
                                  "actual_vs_last30_pct": 100 * (actual[:k].mean() / last30 - 1),
                                  "counterfactual_vs_last30_pct": 100 * (cf_mean / last30 - 1)})
            if cutoff == INTERVENTION:
                plot_data[game] = (train, dates, actual, point, paths)
        # --- placebo: same procedure at fake cutoffs inside the pre-period ---
        for oi, o in enumerate(origins):
            cutoff = start + pd.Timedelta(days=o)
            assert cutoff + pd.Timedelta(days=horizon) <= INTERVENTION, "placebo window reaches the real intervention"
            end_p = cutoff + pd.Timedelta(days=horizon - 1)
            train, dates, point, paths = counterfactual(rg, cutoff, end_p, n_paths, SEED + 1000 * gi + 500 + oi, simulate)
            actual = actual_s.reindex(dates).to_numpy(dtype=float)
            for name, k in {"first 30 days": EARLY, "to 30 Jun": (JUNE_END - INTERVENTION).days + 1,
                            "full post-period": horizon}.items():
                lifts, point_lift = window_effect(actual, point, paths, k)
                placebo_rows.append({"game": game, "fake_cutoff": cutoff.date(), "window": name, "days": k,
                                     **summarise(lifts), "point_forecast_pct": point_lift})
        if verbose:
            print("done", game)

    real = pd.DataFrame(real_rows)
    plc = pd.DataFrame(placebo_rows)
    real.round(3).to_csv(out / "causal_lockdown_effect.csv", index=False)
    plc.round(3).to_csv(out / "causal_lockdown_placebo.csv", index=False)

    # ---------- placebo spread vs the real effect (main date) ----------
    srows = []
    main = real[real.intervention == INTERVENTION.date()]
    for (game, w), p in plc.groupby(["game", "window"]):
        r = main[(main.game == game) & (main.window == w)].iloc[0]
        eff = p.median_pct.to_numpy()
        srows.append({"game": game, "window": w, "placebo_cutoffs": len(eff),
                      "placebo_mean_pct": eff.mean(), "placebo_median_pct": np.median(eff),
                      "placebo_p10_pct": np.percentile(eff, Q_LO), "placebo_p90_pct": np.percentile(eff, Q_HI),
                      "placebo_min_pct": eff.min(), "placebo_max_pct": eff.max(),
                      "real_median_pct": r.median_pct, "real_p10_pct": r.p10_pct, "real_p90_pct": r.p90_pct,
                      "real_minus_placebo_median_pct": r.median_pct - np.median(eff),
                      "share_placebo_as_extreme": float((np.abs(eff - np.median(eff)) >= abs(r.median_pct - np.median(eff))).mean()),
                      "real_outside_placebo_p10_p90": bool(r.median_pct < np.percentile(eff, Q_LO) or r.median_pct > np.percentile(eff, Q_HI))})
    summ = pd.DataFrame(srows)
    summ.round(3).to_csv(out / "causal_lockdown_placebo_summary.csv", index=False)

    # ---------- comparison with the SQL insight ----------
    d = daily.pivot(index="date", columns="game", values=TARGET)
    sql = pd.DataFrame({
        "sql_vs_jan_feb_2020_pct": 100 * (d.loc["2020-03-01":"2020-06-30"].mean() / d.loc["2020-01-01":"2020-02-29"].mean() - 1),
        "sql_vs_mar_jun_2019_pct": 100 * (d.loc["2020-03-01":"2020-06-30"].mean() / d.loc["2019-03-01":"2019-06-30"].mean() - 1)})
    sql.index.name = "game"
    j = main[main.window == "to 30 Jun"].set_index("game")[["median_pct", "p10_pct", "p90_pct", "actual_vs_last30_pct", "counterfactual_vs_last30_pct"]]
    j.columns = ["counterfactual_median_pct_11mar_30jun", "p10_pct", "p90_pct", "actual_vs_last30_pre_days_pct", "counterfactual_vs_last30_pre_days_pct"]
    f = main[main.window == "full post-period"].set_index("game")[["median_pct"]].rename(columns={"median_pct": "counterfactual_median_pct_full_post"})
    cmp_ = sql.join(j).join(f).reset_index()
    cmp_.round(2).to_csv(out / "causal_lockdown_vs_sql.csv", index=False)

    if verbose:
        pd.set_option("display.width", 220)
        print("\nREAL EFFECT (median and 10-90% of per-path average lift, %)")
        print(main[["game", "window", "days", "median_pct", "p10_pct", "p90_pct", "point_forecast_pct", "share_paths_above_zero"]].round(1).to_string(index=False))
        print("\nSENSITIVITY TO THE INTERVENTION DATE (median %)")
        print(real.pivot_table(index=["game", "window"], columns="intervention", values="median_pct").round(1).to_string())
        print("\nPLACEBO SPREAD vs REAL (median of path-lifts at each fake cutoff, %)")
        print(summ[["game", "window", "placebo_cutoffs", "placebo_mean_pct", "placebo_p10_pct", "placebo_p90_pct",
                    "real_median_pct", "share_placebo_as_extreme", "real_outside_placebo_p10_p90"]].round(1).to_string(index=False))
        print("\nTHIS METHOD vs THE SQL NUMBERS (%)")
        print(cmp_.round(1).to_string(index=False))

    draw_counterfactual(plot_data, out / "figures/causal_lockdown_counterfactual.png")
    draw_placebo(plc, main, out / "figures/causal_lockdown_placebo.png")
    return real, plc, summ, cmp_


# ---------- figures ----------
def _grid(n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), facecolor=SURFACE)
    return fig, list(axes.ravel())


def draw_counterfactual(plot_data, path, history=90):
    fig, axes = _grid(len(plot_data))
    for ax, game in zip(axes, sorted(plot_data)):
        train, dates, actual, point, paths = plot_data[game]
        style_axes(ax)
        lo, med, hi = (np.percentile(paths, q, axis=1) for q in (Q_LO, 50, Q_HI))
        ax.plot(train.index[-history:], train.iloc[-history:], color=INK2, linewidth=1)
        ax.fill_between(dates, lo, hi, color=BLUE, alpha=0.18, linewidth=0, label="counterfactual 10-90%")
        ax.plot(dates, med, color=BLUE, linewidth=1.4, label="counterfactual median")
        ax.plot(dates, actual, color=ORANGE, linewidth=1.2, label="actual")
        ax.axvline(INTERVENTION, color=MUTED, linestyle="--", linewidth=1)
        ax.set_title(game, fontsize=10, color=INK, loc="left")
        ax.set_ylabel("daily peak players", fontsize=8, color=INK2)
        ax.tick_params(axis="x", labelrotation=30)
    h, l = axes[0].get_legend_handles_labels()
    axes[-1].axis("off")
    axes[-1].legend(h, l, loc="center", frameon=False, fontsize=9)
    fig.suptitle("Actual daily peak vs the no-lockdown counterfactual (ETS fitted on data before 11 Mar 2020 only)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, path)


def draw_placebo(plc, main, path):
    fig, axes = _grid(5)
    w = "full post-period"
    for ax, game in zip(axes, sorted(plc.game.unique())):
        style_axes(ax)
        eff = plc[(plc.game == game) & (plc.window == w)].median_pct.to_numpy()
        ax.hist(eff, bins=min(12, max(4, len(eff) // 2)), color=BLUE, alpha=0.75, edgecolor=SURFACE)
        real = main[(main.game == game) & (main.window == w)].median_pct.iloc[0]
        ax.axvline(real, color=ORANGE, linewidth=2)
        ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
        ax.set_title(game, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("average lift over the full post-period, %", fontsize=8, color=INK2)
        ax.set_yticks([])
    axes[-1].axis("off")
    axes[-1].text(0, 0.6, "blue: effect 'found' at fake dates\n(no lockdown there)\norange: real estimate from 11 Mar\ndashed: zero", fontsize=9, color=INK2,
                  transform=axes[-1].transAxes)
    fig.suptitle("Placebo test: how large are effects at fake cutoffs vs the real one?", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, path)


if __name__ == "__main__":
    run()