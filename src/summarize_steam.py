"""Shrink the 5-minute Steam player-count files into one small hourly file + a quality report.

Run from the project root (e.g. /e/chronosgame):
    python src/summarize_steam.py steam_sample

Reads   steam_sample/*PlayerCountHistory*__<appid>.csv   (Time,Playercount, 5-min)
        steam_sample/*PriceHistory*__<appid>.csv         (Date,Initialprice,Finalprice,Discount)
Writes  steam_sample/steam_hourly.csv   appid,hour,mean,max,min,n_obs
        steam_sample/steam_prices.csv   appid,Date,Initialprice,Finalprice,Discount
Prints  a quality report (paste it back to Claude).
"""
import re
import sys
from pathlib import Path

import pandas as pd


def appid(path):
    return re.search(r"__(\d+)\.csv$", path.name).group(1)


def main(folder="steam_sample"):
    folder = Path(folder)
    pc_files = sorted(f for f in folder.glob("*.csv") if "PlayerCountHistory" in f.name)
    pr_files = sorted(f for f in folder.glob("*.csv") if "PriceHistory" in f.name)
    if not pc_files:
        sys.exit(f"No PlayerCountHistory files found in {folder.resolve()}")

    hourly = []
    for f in pc_files:
        a = appid(f)
        d = pd.read_csv(f, parse_dates=["Time"])
        d = d.sort_values("Time")
        dup = int(d.Time.duplicated().sum())
        d = d.drop_duplicates("Time")

        step = d.Time.diff()
        gaps = d.loc[step > pd.Timedelta(minutes=15)].assign(gap=step[step > pd.Timedelta(minutes=15)])
        expected = int((d.Time.max() - d.Time.min()) / pd.Timedelta(minutes=5)) + 1
        zeros = int((d.Playercount == 0).sum())

        print(f"\n--- app {a} ---")
        print(f"rows={len(d):,}  range={d.Time.min()} -> {d.Time.max()}")
        print(f"expected 5-min rows={expected:,}  missing={expected-len(d):,} ({(expected-len(d))/expected:.1%})  duplicates={dup}")
        print(f"playercount min/median/max = {d.Playercount.min():,} / {int(d.Playercount.median()):,} / {d.Playercount.max():,}  zeros={zeros}")
        print(f"gaps >15 min: {len(gaps)}", end="")
        if len(gaps):
            big = gaps.sort_values("gap", ascending=False).head(3)
            print("  largest:", [(str(t), str(g)) for t, g in zip(big.Time, big.gap)])
        else:
            print()

        h = (d.set_index("Time").Playercount.resample("h")
             .agg(["mean", "max", "min", "count"]).rename(columns={"count": "n_obs"}))
        h["mean"] = h["mean"].round(1)
        h.insert(0, "appid", a)
        hourly.append(h.reset_index().rename(columns={"Time": "hour"}))

    out_h = folder / "steam_hourly.csv"
    pd.concat(hourly).to_csv(out_h, index=False)
    print(f"\nwrote {out_h} ({out_h.stat().st_size/1e6:.1f} MB)")

    if pr_files:
        prices = pd.concat([pd.read_csv(f).assign(appid=appid(f)) for f in pr_files])
        prices = prices[["appid", "Date", "Initialprice", "Finalprice", "Discount"]]
        out_p = folder / "steam_prices.csv"
        prices.to_csv(out_p, index=False)
        print(f"wrote {out_p} ({out_p.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:2])