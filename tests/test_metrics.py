"""Checks for src/metrics.py: known values, and every edge case from its docstring.

Run from the project root:   python tests/test_metrics.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import metrics as M  # noqa: E402
from backtest_baselines import rmse as bt_rmse, scale_of, smape as bt_smape  # noqa: E402


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except ValueError:
        return True
    return False


def main():
    y = np.array([100.0, 200.0, 300.0, 400.0])
    f = np.array([110.0, 180.0, 330.0, 360.0])          # errors +10 -20 +30 -40
    # known values
    assert close(M.mae(y, f), 25.0)
    assert close(M.rmse(y, f), math.sqrt((100 + 400 + 900 + 1600) / 4))
    assert close(M.wape(y, f), 100 / 1000)
    assert close(M.mape(y, f), np.mean([10 / 100, 20 / 200, 30 / 300, 40 / 400]))
    assert close(M.bias(y, f), -5.0) and close(M.bias_pct(y, f), -20 / 1000)
    # same definitions as the backtest
    assert close(M.smape(y, f), bt_smape(y, f)) and close(M.rmse(y, f), bt_rmse(y, f))
    train = np.arange(1, 31, dtype=float) * 3 + np.tile([0, 5, 2, 8, 1, 9, 4], 5)[:30]
    assert close(M.mase(y, f, train, m=7), M.mae(y, f) / scale_of(train))
    # perfect forecast
    assert M.mae(y, y) == 0 and M.wape(y, y) == 0 and M.smape(y, y) == 0 and M.mape(y, y) == 0
    # zeros in the actuals: nothing is inf or NaN-by-accident
    yz, fz = np.array([0.0, 100.0, 200.0]), np.array([5.0, 90.0, 220.0])
    assert close(M.mape(yz, fz), np.mean([10 / 100, 20 / 200]))                   # skip: zero day left out
    assert close(M.mape(yz, fz, zero_policy="floor", floor=1.0), np.mean([5 / 1, 10 / 100, 20 / 200]))
    assert math.isnan(M.mape(yz, fz, zero_policy="nan"))
    assert math.isfinite(M.wape(yz, fz)) and math.isfinite(M.smape(yz, fz))
    assert M.evaluate(yz, fz)["n_zero_actuals"] == 1
    # everything zero
    z = np.zeros(5)
    assert math.isnan(M.mape(z, z + 1)) and math.isnan(M.wape(z, z + 1)) and math.isnan(M.bias_pct(z, z + 1))
    assert M.wape(z, z) == 0 and M.smape(z, z) == 0                                # 0/0 means a perfect forecast
    # constant or too-short training data: MASE is undefined, not an error or inf
    assert math.isnan(M.mase(y, f, np.ones(30))) and math.isnan(M.mase(y, f, np.arange(5.0)))
    # bad input
    assert raises(M.mae, [], [])
    assert raises(M.mae, [1, 2, 3], [1, 2])
    assert raises(M.mae, [1, np.nan], [1, 2]) and raises(M.mae, [1, 2], [1, np.inf])
    assert close(M.mae([1, np.nan, 3], [2, 2, 5], nan_policy="omit"), 1.5)
    assert raises(M.mae, [np.nan], [np.nan], nan_policy="omit")
    assert raises(M.mape, y, f, zero_policy="bogus") and raises(M.pinball_loss, y, f, 1.5)
    # pandas Series, lists and 2-D arrays are accepted
    assert close(M.mae(pd.Series(y), list(f)), 25.0) and close(M.mae(y.reshape(2, 2), f.reshape(2, 2)), 25.0)
    # ranges and quantiles
    lo, hi = y * 0.9, y * 1.1
    assert close(M.coverage(y, lo, hi), 1.0) and close(M.exceedance(y, y - 1), 1.0)
    assert close(M.coverage(y, lo + 1000, hi + 1000), 0.0)
    assert close(M.pinball_loss(y, f, 0.5), 0.5 * M.mae(y, f))                      # median pinball = half the MAE
    assert M.pinball_loss(y, y + 50, 0.95) < M.pinball_loss(y, y - 50, 0.95)       # a high quantile should sit above
    # grouped table
    df = pd.DataFrame({"game": ["a"] * 4 + ["b"] * 4, "model": "m", "actual": np.r_[y, y], "forecast": np.r_[f, y]})
    t = M.evaluate_frame(df).set_index("game")
    assert close(t.loc["a", "wape"], 0.1) and close(t.loc["b", "wape"], 0.0)
    print("OK: all metric checks passed")


if __name__ == "__main__":
    main()