"""Prediction ranges for the ensemble forecast, plus forecast-vs-actual and error-by-day charts.

Run from the project root (e.g. /e/chronosgame), AFTER src/backtest_baselines.py:
    pip install matplotlib
    python src/intervals_and_charts.py                # target = peak_players
    python src/intervals_and_charts.py avg_players

How the ranges work (no fancy statistics, easy to explain):
  For each forecast day k and each cutoff, look at how wrong the model was on day k (and the
  3 days either side, so a single freak miss cannot spike the range) at ALL EARLIER cutoffs. Measure the miss as a ratio, actual / forecast.
  The 10th and 90th percentiles of those ratios give an 80% range; the 95th percentile gives an
  upper bound for capacity planning ("peak will stay below this 19 times out of 20").
  Only past misses are used, so nothing peeks at the future. The first MIN_PRIOR cutoffs have
  too little history and get no range.

Outputs: data/processed/backtest_<target>_intervals.csv
         reports/figures/fan_charts.png        forecast, 80% range and 95% upper bound vs actual
         reports/figures/error_by_day.png      how forecast error grows with distance
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "ensemble"
POOL_DAYS = 3                         # pool past misses from this many days either side of day k
MIN_PRIOR = 8                         # cutoffs of history needed before a range is issued
Q_LO, Q_HI, Q_UP = 0.10, 0.90, 0.95   # 80% range and the 95% upper bound
HISTORY_DAYS = 42                     # days of actuals shown before the cutoff in the charts
CALM_DATE = pd.Timestamp("2019-09-15")
LOCKDOWN_DATE = pd.Timestamp("2020-04-15")

# colours (validated light-mode palette; text kept in ink colours, not series colours)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def read_dates(df, col):
    for fmt in ("%Y-%m-%d", "%d-%m-%Y"):   # second one = file re-saved by Excel
        try:
            return pd.to_datetime(df[col], format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(df[col])


def pick_order_stat(sorted_pool, q, upper):
    """Percentile with the usual small-sample correction (conformal prediction): with only n past
    misses, the plain percentile is too optimistic, so step to the ceil((n+1)q)-th smallest value
    for upper bounds, and the floor((n+1)q)-th smallest for lower bounds."""
    n = len(sorted_pool)
    if upper:
        idx = min(n, int(np.ceil((n + 1) * q)))
    else:
        idx = max(1, int(np.floor((n + 1) * q)))
    return float(sorted_pool[idx - 1])


def add_ranges(fc):
    """fc: forecasts for ONE model. Adds lo80, hi80, hi95 using only earlier cutoffs."""
    fc = fc.copy()
    fc["miss"] = np.log(fc.actual / fc.forecast)
    out = []
    for game, g in fc.groupby("game"):
        cutoffs = sorted(g.cutoff.unique())
        for i, c in enumerate(cutoffs):
            if i < MIN_PRIOR:
                continue
            prior = g[(g.cutoff < c) & (g.date < c)]          # actuals already known at cutoff c
            now = g[g.cutoff == c].copy()
            lo, hi, up = [], [], []
            for k in now.day:
                pool = np.sort(prior.loc[(prior.day - k).abs() <= POOL_DAYS, "miss"].to_numpy())
                lo.append(pick_order_stat(pool, Q_LO, upper=False))
                hi.append(pick_order_stat(pool, Q_HI, upper=True))
                up.append(pick_order_stat(pool, Q_UP, upper=True))
            now["lo80"] = now.forecast * np.exp(lo)
            now["hi80"] = now.forecast * np.exp(hi)
            now["hi95"] = now.forecast * np.exp(up)
            out.append(now.drop(columns="miss"))
    return pd.concat(out, ignore_index=True)


def summarise(r):
    r = r.assign(window=np.where(r.day <= 7, "days 1-7", "days 8-14"))
    r["inside80"] = (r.actual >= r.lo80) & (r.actual <= r.hi80)
    r["over95"] = r.actual > r.hi95
    r["headroom"] = r.hi95 / r.forecast - 1
    r["width80"] = r.hi80 / r.lo80 - 1
    t = r.groupby(["game", "window"]).agg(inside80=("inside80", "mean"), over95=("over95", "mean"),
                                          headroom=("headroom", "mean"), width80=("width80", "mean"))
    all_ = r.groupby("window").agg(inside80=("inside80", "mean"), over95=("over95", "mean"),
                                   headroom=("headroom", "mean"), width80=("width80", "mean"))
    return t, all_


# ---------------------------------------------------------------- charts
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def save_fig(fig, path):
    """Save a figure; if Windows refuses (file open in a viewer, synced folder...) save beside it."""
    path = Path(path)
    try:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        return path
    except OSError:
        alt = path.with_name(f"{path.stem}_new{path.suffix}")
        fig.savefig(alt, dpi=150, facecolor=SURFACE)
        print(f"[warn] could not overwrite {path} (is it open in another program?) - saved {alt} instead")
        return alt


def fan_charts(r, daily, target, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    games = sorted(r.game.unique())
    cutoffs = sorted(r.cutoff.unique())
    pick = lambda d: min(cutoffs, key=lambda c: abs(pd.Timestamp(c) - d))
    cols = [("Before the lockdown", pick(CALM_DATE)), ("During the lockdown", pick(LOCKDOWN_DATE))]

    fig, axes = plt.subplots(len(games), 2, figsize=(11, 2.6 * len(games)), facecolor=SURFACE)
    for i, game in enumerate(games):
        series = daily[daily.game == game].set_index("date")[target]
        for j, (title, c) in enumerate(cols):
            ax = axes[i, j]
            style_axes(ax)
            c = pd.Timestamp(c)
            f = r[(r.game == game) & (r.cutoff == c)].sort_values("date")
            hist = series[(series.index >= c - pd.Timedelta(days=HISTORY_DAYS)) & (series.index < c)]
            fut = series[(series.index >= c) & (series.index <= f.date.max())]
            ax.plot(hist.index, hist.values, color=MUTED, linewidth=1.6)
            ax.plot(fut.index, fut.values, color=INK, linewidth=1.8)
            ax.fill_between(f.date, f.lo80, f.hi80, color=BLUE, alpha=0.18, linewidth=0)
            ax.plot(f.date, f.hi95, color=BLUE, linewidth=1.1, linestyle=(0, (4, 3)))
            ax.plot(f.date, f.forecast, color=BLUE, linewidth=2.0)
            ax.axvline(c, color=GRID, linewidth=1.0)
            ax.margins(y=0.08)   # axes follow the data (not forced to zero) so flat games stay readable
            ax.yaxis.set_major_formatter(lambda v, _: f"{v/1000:,.0f}k")
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            when = f"cutoff {c:%d %b %Y}"
            if i == 0:
                ax.set_title(f"{title}  ·  {when}", fontsize=10, color=INK, loc="left", fontweight="bold")
            else:
                ax.set_title(when, fontsize=8, color=INK2, loc="left")
            if j == 0:
                ax.set_ylabel(game, fontsize=10, color=INK, fontweight="bold")
    handles = [Line2D([0], [0], color=MUTED, lw=1.6, label="Earlier actual"),
               Line2D([0], [0], color=INK, lw=1.8, label="Actual (what really happened)"),
               Line2D([0], [0], color=BLUE, lw=2.0, label="Forecast"),
               Patch(facecolor=BLUE, alpha=0.18, label="80% range"),
               Line2D([0], [0], color=BLUE, lw=1.1, linestyle=(0, (4, 3)), label="95% upper bound")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    saved = save_fig(fig, out)
    plt.close(fig)
    return saved


def error_by_day(by_day, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = by_day.groupby(["model", "day"]).scaled_err.mean().unstack(0)
    # (model, colour, width): ensemble is the headline, naive is the grey baseline to beat
    style = {"ensemble": (BLUE, 2.4, "Ensemble"), "ets": (ORANGE, 1.8, "ETS"),
             "sarimax": (AQUA, 1.8, "SARIMAX"), "seasonal_naive": (MUTED, 1.6, "Seasonal naive")}
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=SURFACE)
    style_axes(ax)
    last = {}
    for name, (colour, lw, label) in style.items():
        if name not in m:
            continue
        ax.plot(m.index, m[name], color=colour, linewidth=lw, marker="o", markersize=3.5, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.0)
        last[label] = (m[name].iloc[-1], colour)
    ax.axvline(7, color=GRID, linewidth=1.0)
    ax.text(7.1, ax.get_ylim()[1], "one week", fontsize=8, color=INK2, va="top")
    # direct labels: spread apart, centred on the real line ends, joined to them by a thin
    # leader line in the series colour so a label can never be mistaken for a neighbour
    ends = sorted(last.items(), key=lambda kv: kv[1][0])
    gap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06
    ys = []
    for _, (y, _c) in ends:
        ys.append(y if not ys else max(y, ys[-1] + gap))
    shift = np.mean([y for _, (y, _c) in ends]) - np.mean(ys)
    x_end = m.index.max()
    for (label, (y, colour)), yy in zip(ends, ys):
        ax.annotate(label, xy=(x_end, y), xytext=(x_end + 1.0, yy + shift), fontsize=9, color=INK,
                    va="center", arrowprops=dict(arrowstyle="-", color=colour, lw=1.3, shrinkA=0, shrinkB=3))
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
    ax.set_xlim(0.7, m.index.max() + 4.2)
    ax.set_xticks(range(1, 15))
    ax.set_xlabel("How many days ahead the forecast is", fontsize=9, color=INK2)
    ax.set_ylabel("Average error (MASE scale, lower is better)", fontsize=9, color=INK2)
    ax.set_title("Forecast error by how far ahead (average of the five games)", fontsize=11,
                 color=INK, loc="left", fontweight="bold")
    fig.tight_layout()
    saved = save_fig(fig, out)
    plt.close(fig)
    return saved


def main(target="peak_players"):
    proc = Path("data/processed")
    fc = pd.read_csv(proc / f"backtest_{target}_forecasts.csv")
    for c in ("cutoff", "date"):
        fc[c] = read_dates(fc, c)
    by_day = pd.read_csv(proc / f"backtest_{target}_by_day.csv")
    daily = pd.read_csv(proc / "daily_clean.csv")
    daily["date"] = read_dates(daily, "date")

    if MODEL not in set(fc.model):
        sys.exit(f"No '{MODEL}' forecasts found - run src/backtest_baselines.py with statsmodels installed.")
    r = add_ranges(fc[fc.model == MODEL])
    r.to_csv(proc / f"backtest_{target}_intervals.csv", index=False)

    t, all_ = summarise(r)
    n = r.groupby("game").cutoff.nunique().iloc[0]
    pd.set_option("display.width", 140)
    print(f"Target: {target} | model: {MODEL} | ranges issued for {n} cutoffs per game "
          f"(the first {MIN_PRIOR} have too little history)\n")
    print("How well the ranges hold up (targets: inside the 80% range ~80%, above the 95% bound ~5%):")
    fmt = pd.DataFrame({"inside 80% range": (t.inside80 * 100).round(0),
                        "above 95% bound": (t.over95 * 100).round(1),
                        "headroom needed": (t.headroom * 100).round(0),
                        "80% range width": (t.width80 * 100).round(0)})
    print(fmt.to_string())
    print("\nAll games together:")
    print(pd.DataFrame({"inside 80% range": (all_.inside80 * 100).round(0),
                        "above 95% bound": (all_.over95 * 100).round(1),
                        "headroom needed": (all_.headroom * 100).round(0)}).to_string())
    print("\n(percent; 'headroom needed' = how far the 95% bound sits above the point forecast)")

    figs = Path("reports/figures")
    figs.mkdir(parents=True, exist_ok=True)
    try:
        a = fan_charts(r, daily, target, figs / "fan_charts.png")
        b = error_by_day(by_day, figs / "error_by_day.png")
        print(f"\nwrote {a} and {b}")
    except ImportError:
        print("\n[skip] charts: install matplotlib (pip install matplotlib)")
    print(f"wrote {proc / f'backtest_{target}_intervals.csv'}")


if __name__ == "__main__":
    main(*sys.argv[1:2])