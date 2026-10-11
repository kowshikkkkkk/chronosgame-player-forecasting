"""Check that no training series in the backtest can see data from after its cutoff.

Run from the project root:
    python tests/test_no_leakage.py

Idea: take the raw hourly data, scramble every reading from the cutoff onwards, and rebuild the
training series. If the series changes, information from the future leaked in.
  * NEW way (backtest_baselines.train_series): must not change.
  * OLD way (clean the whole history once, then cut at the cutoff): on this dataset it happens not to
    change (no long gap fell in the week before a cutoff), so a second check builds such a gap on
    purpose and confirms the old way leaks while the new way does not.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backtest_baselines import H, INITIAL, STEP, read_daily, train_series  # noqa: E402
from clean_steam import NAMES, clean_game, read_hourly  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TARGET = "peak_players"


def old_way(raw_game, cutoff):
    """Clean everything at once, then take the days before the cutoff (what the first version did)."""
    daily = clean_game(raw_game)["peak"].resample("D").max()
    return daily[daily.index < cutoff]


def scramble_after(raw_game, cutoff, seed=0):
    r = raw_game.copy()
    late = r["hour"] >= cutoff
    rng = np.random.default_rng(seed)
    for col in ("mean", "max", "min"):
        r.loc[late, col] = r.loc[late, col] * rng.uniform(2, 5, late.sum())
    return r


def main():
    daily = read_daily(ROOT / "data/processed/daily_clean.csv")
    raw = read_hourly(ROOT / "steam_sample/steam_hourly.csv")
    by_game = {NAMES.get(a, str(a)): g for a, g in raw.groupby("appid")}
    new_changed = old_changed = n_checked = 0
    for game, g in daily.groupby("game"):
        dates = g["date"].to_numpy()
        for origin in range(INITIAL, len(g) - H + 1, STEP):
            cutoff = pd.Timestamp(dates[origin])
            rg = by_game[game]
            tampered = scramble_after(rg, cutoff)
            n_checked += 1
            new_a, new_b = train_series(rg, cutoff, TARGET), train_series(tampered, cutoff, TARGET)
            if not np.allclose(new_a.to_numpy(), new_b.to_numpy(), rtol=0, atol=1e-9):
                new_changed += 1
            old_a, old_b = old_way(rg, cutoff), old_way(tampered, cutoff)
            if not np.allclose(old_a.to_numpy(), old_b.to_numpy(), rtol=0, atol=1e-9):
                old_changed += 1
    print(f"real data, {n_checked} cutoffs")
    print(f"  NEW way: training series changed after scrambling the future in {new_changed} cutoffs (must be 0)")
    print(f"  OLD way: changed in {old_changed} cutoffs (not a requirement, shown for information)")
    assert new_changed == 0, "LEAK: the training series depends on data after the cutoff"

    # sensitivity check: knock out 13 hours around a day's peak, 4 days before a cutoff
    game = "Rust"
    rg = by_game[game].copy()
    dates = daily[daily.game == game]["date"].to_numpy()
    cutoff = pd.Timestamp(dates[600])
    day = cutoff - pd.Timedelta(days=4)
    hrs = rg[(rg["hour"] >= day) & (rg["hour"] < day + pd.Timedelta(days=1))]
    mid = hrs.loc[hrs["max"].idxmax(), "hour"]
    hole = (rg["hour"] >= mid - pd.Timedelta(hours=6)) & (rg["hour"] <= mid + pd.Timedelta(hours=6))
    rg.loc[hole, ["mean", "max", "min"]] = np.nan
    rg.loc[hole, "n_obs"] = 0
    tampered = scramble_after(rg, cutoff)
    new_ok = np.allclose(train_series(rg, cutoff, TARGET).to_numpy(), train_series(tampered, cutoff, TARGET).to_numpy())
    old_a, old_b = old_way(rg, cutoff), old_way(tampered, cutoff)
    old_leaks = not np.allclose(old_a.to_numpy(), old_b.to_numpy())
    print(f"constructed case (13-hour gap in the week before {cutoff.date()}):")
    print(f"  NEW way unchanged by the future: {new_ok}   OLD way changed by the future: {old_leaks}")
    assert new_ok, "LEAK in the constructed case"
    assert old_leaks, "test is not sensitive: it should have caught the old way"
    print("OK: no leakage in the backtest training data, and the test can catch a leak.")


if __name__ == "__main__":
    main()