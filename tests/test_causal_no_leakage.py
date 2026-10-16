"""Check that the lockdown counterfactual only ever sees data from BEFORE its cutoff.

Run from the project root:
    python tests/test_causal_no_leakage.py
(also works with pytest; the last check needs statsmodels and is skipped without it)

Idea (same as tests/test_no_leakage.py): scramble every raw reading from the cutoff onwards and rebuild
what the counterfactual model is given. If anything it receives changes, the future leaked in.
Checked at the real intervention date, both sensitivity dates and every placebo cutoff, for all 5 games:
  1. the series handed to the model ends the day before the cutoff and does not change when the future is scrambled;
  2. a model that records its input gets identical input and returns identical output either way;
  3. every placebo window (cutoff + post-period length) ends before the real intervention, has >= 365 days
     of history, and the cutoffs are 14 days apart;
  4. a constructed 13-hour gap just before the cutoff: the old "clean everything, then cut" way would leak, ours does not;
  5. (needs statsmodels) the REAL ETS simulation gives identical paths with and without the future.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import causal_lockdown as cl  # noqa: E402
from backtest_baselines import INITIAL, STEP, read_daily  # noqa: E402
from clean_steam import NAMES, clean_game, read_hourly  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def scramble_after(raw_game, cutoff, seed=0):
    r = raw_game.copy()
    late = r["hour"] >= cutoff
    rng = np.random.default_rng(seed)
    for col in ("mean", "max", "min"):
        r.loc[late, col] = r.loc[late, col] * rng.uniform(2, 5, late.sum())
    return r


class Recorder:
    """Stand-in for the model: remembers exactly what it was given, returns something that depends on it."""
    def __init__(self):
        self.seen = []

    def __call__(self, train, h, n_paths, seed):
        self.seen.append(train.copy())
        base = np.resize(np.log(train[-7:]), h)
        rng = np.random.default_rng(seed)
        return np.exp(base), np.exp(base[:, None] + rng.normal(0, 0.1, (h, n_paths)))


def load():
    daily = read_daily(ROOT / "data/processed/daily_clean.csv")
    raw = read_hourly(ROOT / "steam_sample/steam_hourly.csv")
    return daily, {NAMES.get(a, str(a)): g for a, g in raw.groupby("appid")}


def all_cutoffs(daily):
    start = daily["date"].min()
    n_pre = (cl.INTERVENTION - start).days
    horizon = (cl.END - cl.INTERVENTION).days + 1
    origins = cl.placebo_origins(n_pre, horizon)
    return start, n_pre, horizon, origins


def test_inputs_do_not_depend_on_the_future():
    daily, raw_by_game = load()
    start, n_pre, horizon, origins = all_cutoffs(daily)
    cutoffs = [cl.INTERVENTION] + cl.SENSITIVITY + [start + pd.Timedelta(days=o) for o in origins]
    n = 0
    for game, rg in raw_by_game.items():
        for cutoff in cutoffs:
            tampered = scramble_after(rg, cutoff)
            ra, rb = Recorder(), Recorder()
            end = cutoff + pd.Timedelta(days=horizon - 1)
            ta, da, pa, xa = cl.counterfactual(rg, cutoff, end, 20, 1, ra)
            tb, db, pb, xb = cl.counterfactual(tampered, cutoff, end, 20, 1, rb)
            assert ta.index.max() == cutoff - pd.Timedelta(days=1), f"{game} {cutoff.date()}: training series does not stop the day before the cutoff"
            assert np.allclose(ta.to_numpy(), tb.to_numpy(), rtol=0, atol=1e-9), f"LEAK {game} {cutoff.date()}: training series changed"
            assert np.array_equal(ra.seen[0], rb.seen[0]), f"LEAK {game} {cutoff.date()}: model input changed"
            assert np.array_equal(xa, xb) and np.array_equal(pa, pb), f"LEAK {game} {cutoff.date()}: counterfactual changed"
            assert da[0] == cutoff and (da == db).all()
            n += 1
    print(f"OK 1-2: model input and counterfactual unchanged by scrambling the future, {n} game/cutoff pairs")


def test_placebo_windows_stay_inside_the_pre_period():
    daily, _ = load()
    start, n_pre, horizon, origins = all_cutoffs(daily)
    assert len(origins) > 0
    assert origins[0] >= INITIAL, "placebo cutoff with less than 365 days of history"
    assert all(b - a == STEP for a, b in zip(origins, origins[1:])), "placebo cutoffs are not 14 days apart"
    last_day = start + pd.Timedelta(days=origins[-1] + horizon - 1)
    assert last_day < cl.INTERVENTION, "a placebo post-period reaches the real intervention"
    print(f"OK 3: {len(origins)} placebo cutoffs, history >= {origins[0]} days, last window ends {last_day.date()} (< {cl.INTERVENTION.date()})")


def test_constructed_gap_is_not_a_leak():
    """Knock out 13 hours around a day's peak 4 days before the real intervention. Cleaning the whole history
    first (the old way) fills that hole from the following week; ours must not."""
    from backtest_baselines import train_series
    daily, raw_by_game = load()
    rg = raw_by_game["Rust"].copy()
    cutoff = cl.INTERVENTION
    day = cutoff - pd.Timedelta(days=4)
    hrs = rg[(rg["hour"] >= day) & (rg["hour"] < day + pd.Timedelta(days=1))]
    mid = hrs.loc[hrs["max"].idxmax(), "hour"]
    hole = (rg["hour"] >= mid - pd.Timedelta(hours=6)) & (rg["hour"] <= mid + pd.Timedelta(hours=6))
    rg.loc[hole, ["mean", "max", "min"]] = np.nan
    rg.loc[hole, "n_obs"] = 0
    tampered = scramble_after(rg, cutoff)
    ours = np.allclose(train_series(rg, cutoff, cl.TARGET).to_numpy(), train_series(tampered, cutoff, cl.TARGET).to_numpy())

    def old_way(r):
        d = clean_game(r)["peak"].resample("D").max()
        return d[d.index < cutoff].to_numpy()
    old_leaks = not np.allclose(old_way(rg), old_way(tampered))
    assert ours, "LEAK in the constructed case"
    assert old_leaks, "test is not sensitive: it should have caught the old way"
    print("OK 4: constructed 13-hour gap: ours unchanged by the future, the old way is changed (so the test can catch a leak)")


def test_real_model_paths_do_not_depend_on_the_future():
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        print("SKIP 5: statsmodels not installed")
        return
    daily, raw_by_game = load()
    cutoff = cl.INTERVENTION
    end = cl.END
    for game in ("Rust", "PUBG"):
        rg = raw_by_game[game]
        _, _, pa, xa = cl.counterfactual(rg, cutoff, end, 50, 7)
        _, _, pb, xb = cl.counterfactual(scramble_after(rg, cutoff), cutoff, end, 50, 7)
        assert np.allclose(pa, pb) and np.allclose(xa, xb), f"LEAK in real ETS paths for {game}"
    print("OK 5: real ETS counterfactual paths identical with and without the future")


if __name__ == "__main__":
    test_inputs_do_not_depend_on_the_future()
    test_placebo_windows_stay_inside_the_pre_period()
    test_constructed_gap_is_not_a_leak()
    test_real_model_paths_do_not_depend_on_the_future()
    print("All causal no-leakage checks passed.")