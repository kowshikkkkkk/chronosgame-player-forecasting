"""Forecast evaluation metrics that survive real data: zeros, NaNs, empty inputs, constant series.

Use as a library:
    from metrics import evaluate, evaluate_frame
    evaluate(actual, forecast, y_train=train_series, m=7)

Or from the command line, on the backtest forecasts file:
    python src/metrics.py                                   # data/processed/backtest_peak_players_forecasts.csv
    python src/metrics.py path/to/forecasts.csv --since 2020-03-01

Conventions (all ratio metrics are fractions: 0.05 means 5%)
  * forecast - actual > 0 means OVER-forecast (bias is positive when the forecast is too high).
  * MAE, RMSE   in the units of the data (players).
  * WAPE        sum|error| / sum|actual|. Total error as a share of total volume. Safe when single days
                are zero or small, which is why it is the usual headline metric in production.
  * MAPE        mean |error| / |actual|. Blows up when an actual is near zero, so zeros are handled by an
                explicit zero_policy (see mape).
  * sMAPE       mean 2|error| / (|actual| + |forecast|), range 0..2. Same definition as backtest_baselines.py.
                When actual and forecast are both 0 the error is defined as 0 (a perfect forecast).
  * MASE        mean |error| / mean |week-over-week change in the training data| (m = 7 for daily data).
                NaN when the training series is too short or perfectly flat.
  * bias, bias_pct   mean signed error, and total signed error as a share of total actual.
  * coverage, exceedance, pinball_loss   for prediction ranges and quantile forecasts (capacity buffers).

Edge-case rules, so nothing divides by zero or fails silently:
  * empty input or different lengths -> ValueError
  * NaN / inf in inputs -> ValueError by default; nan_policy="omit" drops those pairs
  * a metric that is undefined for the data (for example WAPE when all actuals are 0) -> NaN, never inf
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- input handling
def _pair(y_true, y_pred, nan_policy="raise"):
    y = np.asarray(y_true, dtype=float).ravel()
    f = np.asarray(y_pred, dtype=float).ravel()
    if y.size == 0 or f.size == 0:
        raise ValueError("empty input")
    if y.shape != f.shape:
        raise ValueError(f"length mismatch: {y.size} actuals vs {f.size} forecasts")
    ok = np.isfinite(y) & np.isfinite(f)
    if not ok.all():
        if nan_policy == "raise":
            raise ValueError(f"{int((~ok).sum())} NaN/inf value(s); use nan_policy='omit' to drop them")
        if nan_policy != "omit":
            raise ValueError("nan_policy must be 'raise' or 'omit'")
        y, f = y[ok], f[ok]
        if y.size == 0:
            raise ValueError("no valid pairs left after dropping NaN/inf")
    return y, f


# ---------------------------------------------------------------- point metrics
def mae(y_true, y_pred, nan_policy="raise"):
    y, f = _pair(y_true, y_pred, nan_policy)
    return float(np.mean(np.abs(y - f)))


def rmse(y_true, y_pred, nan_policy="raise"):
    y, f = _pair(y_true, y_pred, nan_policy)
    return float(np.sqrt(np.mean((y - f) ** 2)))


def bias(y_true, y_pred, nan_policy="raise"):
    """Mean signed error. Positive = the forecast is too high on average."""
    y, f = _pair(y_true, y_pred, nan_policy)
    return float(np.mean(f - y))


def wape(y_true, y_pred, nan_policy="raise"):
    y, f = _pair(y_true, y_pred, nan_policy)
    total = float(np.sum(np.abs(y)))
    err = float(np.sum(np.abs(y - f)))
    if total == 0:
        return 0.0 if err == 0 else float("nan")
    return err / total


def bias_pct(y_true, y_pred, nan_policy="raise"):
    """Total signed error as a share of total actual (positive = over-forecast)."""
    y, f = _pair(y_true, y_pred, nan_policy)
    total = float(np.sum(np.abs(y)))
    if total == 0:
        return float("nan")
    return float(np.sum(f - y)) / total


def mape(y_true, y_pred, zero_policy="skip", floor=1.0, nan_policy="raise"):
    """Mean absolute percentage error, with an explicit rule for actuals equal to zero.

    zero_policy:
      "skip"  (default) leave zero actuals out of the average; NaN if every actual is zero
      "floor" divide by max(|actual|, floor), so a zero day costs |error| / floor instead of infinity;
              floor is in the data's units (1.0 = one player)
      "nan"   return NaN if any actual is zero
    """
    y, f = _pair(y_true, y_pred, nan_policy)
    err = np.abs(y - f)
    zero = y == 0
    if zero_policy == "skip":
        if zero.all():
            return float("nan")
        return float(np.mean(err[~zero] / np.abs(y[~zero])))
    if zero_policy == "floor":
        if floor <= 0:
            raise ValueError("floor must be positive")
        return float(np.mean(err / np.maximum(np.abs(y), floor)))
    if zero_policy == "nan":
        if zero.any():
            return float("nan")
        return float(np.mean(err / np.abs(y)))
    raise ValueError("zero_policy must be 'skip', 'floor' or 'nan'")


def smape(y_true, y_pred, nan_policy="raise"):
    y, f = _pair(y_true, y_pred, nan_policy)
    denom = np.abs(y) + np.abs(f)
    term = np.zeros_like(denom)
    nz = denom > 0
    term[nz] = 2 * np.abs(y[nz] - f[nz]) / denom[nz]          # 0 / 0 -> 0: both zero is a perfect forecast
    return float(np.mean(term))


def mase(y_true, y_pred, y_train, m=7, nan_policy="raise"):
    y, f = _pair(y_true, y_pred, nan_policy)
    train = np.asarray(y_train, dtype=float).ravel()
    train = train[np.isfinite(train)]
    if train.size <= m:
        return float("nan")
    scale = float(np.mean(np.abs(train[m:] - train[:-m])))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(y - f))) / scale


# ---------------------------------------------------------------- ranges and quantiles
def coverage(y_true, lower, upper, nan_policy="raise"):
    """Share of actuals inside [lower, upper]. An 80% range should give about 0.80."""
    y, lo = _pair(y_true, lower, nan_policy)
    _, hi = _pair(y_true, upper, nan_policy)
    return float(np.mean((y >= lo) & (y <= hi)))


def exceedance(y_true, upper, nan_policy="raise"):
    """Share of actuals ABOVE the upper bound. A 95% bound should give about 0.05."""
    y, hi = _pair(y_true, upper, nan_policy)
    return float(np.mean(y > hi))


def pinball_loss(y_true, y_quantile, q, nan_policy="raise"):
    """Quantile loss for a forecast of the q-th quantile (0 < q < 1). Lower is better."""
    if not 0 < q < 1:
        raise ValueError("q must be between 0 and 1")
    y, p = _pair(y_true, y_quantile, nan_policy)
    d = y - p
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


# ---------------------------------------------------------------- bundles
def evaluate(y_true, y_pred, y_train=None, m=7, zero_policy="skip", floor=1.0, nan_policy="raise"):
    """All point metrics in one dict. MASE is included when y_train is given."""
    y, f = _pair(y_true, y_pred, nan_policy)
    out = {
        "n": int(y.size),
        "n_zero_actuals": int((y == 0).sum()),
        "mae": mae(y, f),
        "rmse": rmse(y, f),
        "wape": wape(y, f),
        "mape": mape(y, f, zero_policy=zero_policy, floor=floor),
        "smape": smape(y, f),
        "bias": bias(y, f),
        "bias_pct": bias_pct(y, f),
    }
    if y_train is not None:
        out["mase"] = mase(y, f, y_train, m=m)
    return out


def evaluate_frame(df, by=("game", "model"), y_col="actual", f_col="forecast", **kwargs):
    """Metrics per group of a long table (for example the backtest forecasts file).

    Errors are pooled over every row in the group, so WAPE here is total error over total actual."""
    rows = []
    for key, g in df.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        res = evaluate(g[y_col], g[f_col], nan_policy="omit", **kwargs)
        rows.append({**dict(zip(by, key)), **res})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- command line
def _show(table):
    t = table.copy()
    for c in ("wape", "mape", "smape", "bias_pct"):
        t[c] = (t[c] * 100).round(1)
    for c in ("mae", "rmse", "bias"):
        t[c] = t[c].round(0)
    cols = [c for c in ("game", "model", "n", "wape", "mape", "smape", "mae", "rmse", "bias", "bias_pct",
                        "n_zero_actuals") if c in t.columns]
    return t[cols].rename(columns={"wape": "wape_%", "mape": "mape_%", "smape": "smape_%", "bias_pct": "bias_%"})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Metrics per game and model from a backtest forecasts file")
    ap.add_argument("path", nargs="?", default="data/processed/backtest_peak_players_forecasts.csv")
    ap.add_argument("--since", help="only forecasts for dates on or after this one, e.g. 2020-03-01")
    args = ap.parse_args(argv)
    if not Path(args.path).exists():
        raise SystemExit(f"{args.path} not found. Run backtest_baselines.py first.")
    df = pd.read_csv(args.path)
    need = {"game", "model", "date", "forecast", "actual"} - set(df.columns)
    if need:
        raise SystemExit(f"missing column(s): {sorted(need)}")
    try:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    except ValueError:
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    if args.since:
        df = df[df["date"] >= pd.Timestamp(args.since)]
    pd.set_option("display.width", 160)
    print(_show(evaluate_frame(df)).to_string(index=False))
    print("\nratios are percent; bias_% > 0 means forecasts are too high; "
          "mape skips days with zero actuals (counted in n_zero_actuals)")


if __name__ == "__main__":
    main()