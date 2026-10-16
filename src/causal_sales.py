"""Causal analysis 2: do Steam sales lift the daily peak? (difference-in-differences + permutation test)

Run from the project root:
    python src/causal_sales.py

Question
    The SQL insight (sql/insights/02_sale_effect.sql) compares the first 3 sale days with the 7 days
    before. That comparison mixes in the weekday pattern (a sale that starts on a Friday is compared
    with a Mon-Thu baseline) and any trend or news affecting all games at once. This script removes
    both, and then asks whether what is left is bigger than chance.

Method (all on the log scale, so a difference is a ratio)
  1. Weekday-matched lift for one game on one day:
         log(peak that day) - mean(log(peak on the same weekday in each of the 4 previous weeks))
     The lift of a 3-day window is the mean of the 3 daily lifts (days 1-3 of the sale).
  2. Difference-in-differences: subtract the same window lift for the CONTROL games, i.e. the other
     games that have no discount on any of those 3 days. If no control is available the sale is
     skipped and counted.
  3. Permutation test: draw random start dates on which nobody is on sale and compute the same
     statistic (the game's lift minus its controls' lift). Draw as many dates as the game has usable
     sales, average them, repeat 1,000 times. The observed average is compared with that spread.
     Two-sided p = share of null averages at least as far from the null mean as the observed one
     (with the usual +1 so p is never exactly 0).

  Placebo dates must be: inside the price-file period, at least 7 days away from any sale day of any
  game (so neither the window nor the neighbourhood holds a discount), with all 3 window days
  present in the price file, and not in the March-June 2020 lockdown.

Outputs
    reports/causal_sales_events.csv         one row per sale start (used, or skipped and why)
    reports/causal_sales_summary.csv        per game and pooled: unadjusted, weekday-adjusted, DiD, p-value
    reports/causal_sales_permutation.csv    the 1,000 null averages per game and pooled
    reports/figures/causal_sale_effect.png  observed average vs permutation distribution

What this can and cannot say: it is about the daily PEAK of concurrent players. It says nothing about
revenue, unique buyers or retention, and it is observational (sales are not randomised).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intervals_and_charts import BLUE, GRID, INK, INK2, MUTED, ORANGE, SURFACE, read_dates, save_fig, style_axes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data/processed/daily_clean.csv"
PRICES = ROOT / "steam_sample/steam_prices.csv"
OUT = ROOT / "reports"
FIG = OUT / "figures"

WINDOW = 3                       # first 3 sale days
BASE_WEEKS = 4                   # baseline = same weekday in the previous 4 weeks
BUFFER_DAYS = 7                  # placebo dates stay this far from any sale day
N_PERM = 1000
MIN_POOL = 10                    # a game needs at least this many placebo dates to be tested
SEED = 20200311                  # fixed: every run gives the same numbers
LOCKDOWN = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"))
NAMES = {252490: "Rust", 252950: "Rocket League", 271590: "GTA V", 289070: "Civilization VI", 578080: "PUBG"}


def pct(x):
    """log-ratio -> percent change."""
    return 100 * (np.exp(x) - 1)


def load():
    daily = pd.read_csv(DAILY)
    daily["date"] = read_dates(daily, "date")
    peak = daily.pivot(index="date", columns="game", values="peak_players").sort_index()
    peak = peak.asfreq("D")
    prices = pd.read_csv(PRICES)
    prices["date"] = read_dates(prices, "Date")
    prices["game"] = prices["appid"].map(NAMES)
    disc = prices.pivot(index="date", columns="game", values="Discount").reindex(peak.index)
    return peak, disc


def window_lift(peak):
    """S[date, game] = mean over date..date+2 of [log(peak) - mean log(peak same weekday, previous 4 weeks)].
    Uses only days up to date+2 and the 4 weeks before each day."""
    lp = np.log(peak)
    base = sum(lp.shift(7 * k) for k in range(1, BASE_WEEKS + 1)) / BASE_WEEKS
    day_lift = lp - base
    return day_lift.rolling(WINDOW).mean().shift(-(WINDOW - 1))


def unadjusted(peak, disc):
    """The README / SQL figure: mean peak over first 3 sale days / mean over the 7 days before - 1, in %."""
    first3 = peak.rolling(WINDOW).mean().shift(-(WINDOW - 1))
    prior7 = peak.shift(1).rolling(7).mean()
    return 100 * (first3 / prior7 - 1)


def find_events(disc, last_day):
    """Sale start = discount > 0 today and discount == 0 yesterday (both present in the price file)."""
    rows = []
    for g in disc.columns:
        d = disc[g]
        start = (d > 0) & (d.shift(1) == 0)
        for t in d.index[start]:
            rows.append({"game": g, "start": t, "discount": d.loc[t]})
    ev = pd.DataFrame(rows).sort_values(["game", "start"]).reset_index(drop=True)
    ev["full_window"] = ev["start"] + pd.Timedelta(days=WINDOW - 1) <= last_day
    return ev


def on_sale_in_window(disc, t, game):
    w = disc.loc[t : t + pd.Timedelta(days=WINDOW - 1), game]
    return w.isna().any() or (w > 0).any()    # missing price row = unknown, treated as not a clean control


def did_value(S, disc, t, game):
    """(treated lift, mean control lift, n controls) for one start date."""
    controls = [h for h in S.columns if h != game and not on_sale_in_window(disc, t, h)]
    own = S.loc[t, game]
    if not controls or np.isnan(own):
        return own, np.nan, 0
    ctrl = S.loc[t, controls].mean()
    return own, ctrl, len(controls)


def own_sale_before(disc, game, t):
    """True if the game itself was on sale in the 28 days before t (baseline weeks contaminated)."""
    w = disc.loc[t - pd.Timedelta(days=7 * BASE_WEEKS) : t - pd.Timedelta(days=1), game]
    return bool((w > 0).any())


def placebo_dates(disc):
    """Start dates with no sale anywhere near them, price rows present, outside the lockdown."""
    sale = (disc > 0).any(axis=1)
    known = disc.notna().all(axis=1)
    idx = disc.index
    ok = pd.Series(True, index=idx)
    for off in range(-BUFFER_DAYS, WINDOW + BUFFER_DAYS):     # sale day anywhere in [t-7, t+2+7]
        ok &= ~sale.shift(-off, fill_value=False)
        ok &= known.shift(-off, fill_value=False) if abs(off) <= WINDOW else True
    ok &= ~((idx >= LOCKDOWN[0]) & (idx <= LOCKDOWN[1]))
    # the window itself must be inside the period the price file covers
    covered = disc.notna().all(axis=1)
    first, last = covered.idxmax(), covered[::-1].idxmax()
    ok &= (idx >= first + pd.Timedelta(days=BUFFER_DAYS)) & (idx <= last - pd.Timedelta(days=BUFFER_DAYS + WINDOW))
    return idx[ok]


def main():
    peak, disc = load()
    games = list(peak.columns)
    S = window_lift(peak)
    unadj = unadjusted(peak, disc)
    last_day = peak.index.max()

    # ---------- observed sale events ----------
    ev = find_events(disc, last_day)
    rec = []
    for r in ev.itertuples():
        t, g = r.start, r.game
        row = {"game": g, "start": t.date(), "discount_pct": r.discount, "weekday": t.day_name(), "full_window": bool(r.full_window)}
        if not r.full_window:
            row.update(status="skipped: window runs past the data", used=False)
        else:
            own, ctrl, n_ctrl = did_value(S, disc, t, g)
            row.update(unadjusted_pct=unadj.loc[t, g], weekday_adjusted_pct=pct(own), n_controls=n_ctrl)
            if n_ctrl == 0:
                row.update(status="skipped: no control game off sale", used=False)
            else:
                row.update(did_log=own - ctrl, did_pct=pct(own - ctrl), status="used", used=True)
            row["own_log"] = own
        row["clean_baseline"] = not own_sale_before(disc, g, t)
        in_lock = LOCKDOWN[0] <= t <= LOCKDOWN[1]
        row["in_lockdown"] = in_lock
        rec.append(row)
    events = pd.DataFrame(rec)
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True, parents=True)
    events.to_csv(OUT / "causal_sales_events.csv", index=False)
    n_total, n_used = len(events), int(events.used.sum())
    print(f"sale starts found: {n_total} | used (have controls and a full window): {n_used} | "
          f"skipped: {n_total - n_used}")
    print(events[~events.used].status.value_counts().to_string())

    # ---------- placebo pool ----------
    pool = placebo_dates(disc)
    print(f"placebo start dates available: {len(pool)} "
          f"({pool.min().date()} to {pool.max().date()}), all five games off sale at each")
    # per game: placebo DiD for every pool date (all 4 other games are off sale there by construction)
    pool_did = {}
    pool_own = {}
    for g in games:
        others = [h for h in games if h != g]
        pool_did[g] = (S.loc[pool, g] - S.loc[pool, others].mean(axis=1)).dropna()
        pool_own[g] = S.loc[pool, g].dropna()

    # ---------- permutation test ----------
    rng_master = np.random.SeedSequence(SEED)
    rngs = dict(zip(games + ["Pooled"], [np.random.default_rng(s) for s in rng_master.spawn(len(games) + 1)]))

    def run(used_ev, tag, pdid, pown):
        """Summary rows + null draws for a chosen set of used events and placebo pools.
        A game is only tested if its placebo pool has at least max(MIN_POOL, n_sales) dates."""
        summ, draws = [], {}
        counts = used_ev.groupby("game").size().to_dict()
        counts = {g: n for g, n in counts.items() if len(pdid[g]) >= max(MIN_POOL, n)}
        used_ev = used_ev[used_ev.game.isin(counts)]
        null_cols = {}
        for g in games:
            n = counts.get(g, 0)
            if n == 0:
                continue
            for stat, pool_dict in (("did", pdid), ("nocontrol", pown)):
                vals = pool_dict[g].to_numpy()
                null_cols[(g, stat)] = np.array([rngs[g].choice(vals, n, replace=False).mean() for _ in range(N_PERM)])
        # pooled: draw n_g dates per game, average across all draws
        for stat, pool_dict in (("did", pool_did), ("nocontrol", pool_own)):
            tot = sum(counts.values())
            acc = np.zeros(N_PERM)
            for g in games:
                n = counts.get(g, 0)
                if n:
                    acc += null_cols[(g, stat)] * n          # reuse each game's draws: sum of group sums
            null_cols[("Pooled", stat)] = acc / tot
        for g in games + ["Pooled"]:
            sub = used_ev if g == "Pooled" else used_ev[used_ev.game == g]
            if sub.empty:
                continue
            obs_did = sub.did_log.mean()
            obs_nc = sub.own_log.mean()
            row = {"scope": tag, "game": g, "sales_used": len(sub), "placebo_dates": len(pdid[g]) if g != "Pooled" else np.nan,
                   "unadjusted_avg_pct": sub.unadjusted_pct.mean(),
                   "weekday_adjusted_avg_pct": pct(obs_nc),
                   "did_avg_pct": pct(obs_did)}
            for stat, obs in (("did", obs_did), ("nocontrol", obs_nc)):
                null = null_cols[(g, stat)]
                centre = null.mean()
                p = (1 + np.sum(np.abs(null - centre) >= abs(obs - centre))) / (N_PERM + 1)
                key = "" if stat == "did" else "nocontrol_"
                row[f"{key}null_mean_pct"] = pct(centre)
                row[f"{key}null_p10_pct"] = pct(np.percentile(null, 10))
                row[f"{key}null_p90_pct"] = pct(np.percentile(null, 90))
                row[f"{key}share_null_below_obs"] = float(np.mean(null < obs))
                row[f"{key}p_value"] = p
                if stat == "did":
                    draws[g] = (null, obs)
            summ.append(row)
        return pd.DataFrame(summ), draws, null_cols

    used = events[events.used].copy()
    # clean-baseline version: same-game sale in the 28 baseline days is not allowed, for sales AND placebo dates
    pool_did_c = {g: v[[not own_sale_before(disc, g, t) for t in v.index]] for g, v in pool_did.items()}
    pool_own_c = {g: v[[not own_sale_before(disc, g, t) for t in v.index]] for g, v in pool_own.items()}
    main_summary, draws, null_cols = run(used, "all usable sales", pool_did, pool_own)
    robust_summary, _, _ = run(used[~used.in_lockdown], "excluding Mar-Jun 2020 sales", pool_did, pool_own)
    clean_summary, draws_c, null_c = run(used[used.clean_baseline], "clean baseline (no own sale in prior 28 days)",
                                         pool_did_c, pool_own_c)
    summary = pd.concat([main_summary, robust_summary, clean_summary], ignore_index=True)
    # the README / SQL figure: every sale start with full windows, controls or not
    full = events[events.full_window]
    allu = full.groupby("game").unadjusted_pct.mean().to_dict(); allu["Pooled"] = full.unadjusted_pct.mean()
    summary["unadjusted_all_sales_pct"] = summary.game.map(allu)
    summary["sales_total_found"] = summary.game.map(events.groupby("game").size().to_dict()).fillna(n_total)
    summary.loc[summary.game == "Pooled", "sales_total_found"] = n_total
    summary.round(4).to_csv(OUT / "causal_sales_summary.csv", index=False)
    nulls = pd.DataFrame({f"{g}|{s}": v for (g, s), v in null_cols.items()})
    for (g, s_), v in null_c.items():
        nulls[f"clean|{g}|{s_}"] = v
    nulls.round(6).to_csv(OUT / "causal_sales_permutation.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["game", "sales_used", "unadjusted_avg_pct", "weekday_adjusted_avg_pct", "did_avg_pct",
            "null_mean_pct", "null_p10_pct", "null_p90_pct", "p_value", "nocontrol_p_value"]
    cols.insert(2, "placebo_dates")
    for name, df in (("ALL USABLE SALES", main_summary), ("EXCLUDING MARCH-JUNE 2020 SALES", robust_summary),
                     ("CLEAN BASELINE (no sale of the same game in the 28 baseline days)", clean_summary)):
        print(f"\n{name} (percent; null = random no-sale dates, {N_PERM} draws)")
        print(df[cols].round(3).to_string(index=False))

    # ---------- diagnostics on the placebo pool ----------
    episodes = int((pool.to_series().diff() != pd.Timedelta(days=1)).sum())
    print(f"\nplacebo dates form only {episodes} separate stretches of consecutive days "
          f"(neighbouring dates are not independent draws)")
    for g in games:
        n_dirty = int(sum(own_sale_before(disc, g, pd.Timestamp(t)) for t in used[used.game == g].start))
        print(f"  {g}: {np.mean([own_sale_before(disc, g, t) for t in pool]):.0%} of placebo dates have a sale of the "
              f"same game in the 28 baseline days; {n_dirty} of {int((used.game == g).sum())} real sales do")

    # ---------- figures ----------
    draw_figure(draws, main_summary, "Sale effect on the daily peak (DiD) vs random no-sale dates",
                path=FIG / "causal_sale_effect.png")
    draw_figure(draws_c, clean_summary, "Same, clean baseline only (no sale of the same game in the 28 baseline days)",
                path=FIG / "causal_sale_effect_clean_baseline.png")
    print("\nwrote reports/figures/causal_sale_effect.png, causal_sale_effect_clean_baseline.png, "
          "reports/causal_sales_events.csv, causal_sales_summary.csv, causal_sales_permutation.csv")


def draw_figure(draws, summary, title, path):
    order = [g for g in list(NAMES.values()) + ["Pooled"] if g in draws]
    fig, axes = plt_grid(len(order))
    for ax, g in zip(axes, order):
        null, obs = draws[g]
        style_axes(ax)
        ax.hist(pct(null), bins=40, color=BLUE, alpha=0.75, edgecolor=SURFACE)
        ax.axvline(pct(obs), color=ORANGE, linewidth=2)
        row = summary[summary.game == g].iloc[0]
        ax.set_title(f"{g}  (n={int(row.sales_used)}, p={row.p_value:.3f})", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("average sale effect, %", fontsize=8, color=INK2)
        ax.set_yticks([])
    for ax in axes[len(order):]:
        ax.axis("off")
    fig.suptitle(title + "\norange line = observed average; blue = 1,000 permutation draws from no-sale dates",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, path)


def plt_grid(n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), facecolor=SURFACE)
    return fig, list(axes.ravel())


if __name__ == "__main__":
    main()