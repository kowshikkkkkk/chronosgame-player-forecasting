"""Run the SQL insight queries in sql/insights/ on the cleaned player data, using DuckDB.

Run from the project root (e.g. /e/chronosgame):
    pip install duckdb
    python src/run_sql.py

Needs data/processed/daily_clean.csv and hourly_clean.csv (from clean_steam.py) and
steam_sample/steam_prices.csv (from summarize_steam.py).

What it does: loads the three CSVs as tables (daily, hourly, prices), runs each .sql file in
sql/insights/ in order, prints every insight_* table and saves it to data/processed/insight_*.csv.
The calendar columns (dow, month, hod) are worked out in pandas, so the SQL stays simple and the
same queries run on any database. dow: 0 = Sunday ... 6 = Saturday.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PRICES = ROOT / "steam_sample" / "steam_prices.csv"
SQL_DIR = ROOT / "sql" / "insights"


def _dates(s):
    """Parse a date column that is either ISO (2019-04-07) or DD-MM-YYYY (re-saved by Excel)."""
    try:
        return pd.to_datetime(s, format="%Y-%m-%d")
    except ValueError:
        return pd.to_datetime(s, format="%d-%m-%Y")


def _flag(s):
    """True/False column read from CSV (also copes with Excel's TRUE/FALSE) -> 1/0."""
    return s.astype(str).str.strip().str.lower().eq("true").astype(int)


def load_tables(proc=PROC, prices=PRICES):
    """The three input tables as DataFrames. Dates are ISO text so SQL can compare them directly."""
    d = pd.read_csv(Path(proc) / "daily_clean.csv")
    d["date"] = _dates(d["date"])
    d["dow"] = (d["date"].dt.dayofweek + 1) % 7          # pandas has Monday = 0; we want Sunday = 0
    d["month"] = d["date"].dt.strftime("%Y-%m")
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")

    h = pd.read_csv(Path(proc) / "hourly_clean.csv")
    ts = pd.to_datetime(h["hour"], format="mixed", dayfirst=False)
    h["hod"] = ts.dt.hour
    h["dow"] = (ts.dt.dayofweek + 1) % 7
    h["imputed"] = _flag(h["imputed"])
    h["dip"] = _flag(h["dip"])
    h["hour"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")

    p = pd.read_csv(prices)
    p.columns = [c.lower() for c in p.columns]
    p["date"] = _dates(p["date"]).dt.strftime("%Y-%m-%d")
    return {"daily": d, "hourly": h, "prices": p}


def statements(sql_text):
    """Split a .sql file into statements, dropping comment-only lines."""
    lines = [ln for ln in sql_text.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def main():
    try:
        import duckdb
    except ImportError:
        sys.exit("DuckDB is not installed. Run:  pip install duckdb")

    tables = load_tables()
    con = duckdb.connect()                                  # in-memory database
    for name, df in tables.items():
        con.register(f"{name}_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_df")
    print("loaded:", ", ".join(f"{n} ({len(df):,} rows)" for n, df in tables.items()))

    pd.set_option("display.width", 160, "display.max_rows", 60)
    for path in sorted(SQL_DIR.glob("*.sql")):
        for stmt in statements(path.read_text()):
            con.execute(stmt)
        made = re.findall(r"CREATE TABLE (insight_\w+)", path.read_text())
        for name in made:
            df = con.execute(f"SELECT * FROM {name}").df()
            df.to_csv(PROC / f"{name}.csv", index=False)
            print(f"\n=== {name}  ({path.name}, {len(df)} rows, saved to data/processed/{name}.csv)")
            # long tables: show a short preview, the CSV has everything
            print(df.to_string(index=False) if len(df) <= 30 else df.head(8).to_string(index=False))
    print("\ndone")


if __name__ == "__main__":
    main()