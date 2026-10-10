"""ChronosGame dashboard: daily peak player forecasts and capacity ranges for five Steam games.

Run from the project root (e.g. /e/chronosgame):
    pip install streamlit
    streamlit run src/dashboard.py

Needs the files in data/processed/ made by clean_steam.py, backtest_baselines.py and
intervals_and_charts.py. All numbers come from a historical replay: each forecast is what the
model would have said on that date, using only data available then.
"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))   # lets us import the sibling modules
import dashboard_data as dd  # noqa: E402

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

st.set_page_config(page_title="ChronosGame: player demand forecasts", layout="wide")


@st.cache_data
def get_data():
    return dd.load(PROC)


try:
    data = get_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

daily, intervals, backtest, by_day = data["daily"], data["intervals"], data["backtest"], data["by_day"]


def style(chart):
    return (chart.configure_view(strokeWidth=0)
            .configure_axis(grid=True, gridColor=GRID, domainColor=GRID, tickColor=GRID,
                            labelColor=INK2, titleColor=INK2)
            .configure_legend(labelColor=INK2, titleColor=INK2, orient="top", title=None))


def fan_chart(hist, f):
    lines = dd.line_frame(hist, f)
    order = ["Earlier actual", "Actual", "Forecast", "95% upper bound"]
    y = alt.Y("value:Q", title="Daily peak players", scale=alt.Scale(zero=False))
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%d %b"))
    band = (alt.Chart(f).mark_area(opacity=0.18, color=BLUE)
            .encode(x=x, y=alt.Y("lo80:Q", title="Daily peak players", scale=alt.Scale(zero=False)),
                    y2=alt.Y2("hi80")))
    line = (alt.Chart(lines).mark_line(strokeWidth=2.2)
            .encode(x=x, y=y,
                    color=alt.Color("series:N", sort=order,
                                    scale=alt.Scale(domain=order, range=[MUTED, INK, BLUE, BLUE])),
                    strokeDash=alt.StrokeDash("series:N", sort=order,
                                              scale=alt.Scale(domain=order,
                                                              range=[[1, 0], [1, 0], [1, 0], [5, 4]])),
                    tooltip=[alt.Tooltip("series:N", title="Line"),
                             alt.Tooltip("date:T", title="Date", format="%a %d %b %Y"),
                             alt.Tooltip("value:Q", title="Players", format=",.0f")]))
    return style(alt.layer(band, line).properties(height=380))


def error_chart(m):
    m = m.rename(columns={"Model": "model", "Days ahead": "days", "Error": "error"})
    order = ["Seasonal naive", "ETS", "SARIMAX", "Ensemble"]
    colours = [MUTED, ORANGE, AQUA, BLUE]
    line = (alt.Chart(m).mark_line(point=True, strokeWidth=2.2)
            .encode(x=alt.X("days:Q", title="Days ahead", axis=alt.Axis(values=list(range(1, 15)), format="d")),
                    y=alt.Y("error:Q", title="Average error (MASE scale, lower is better)",
                            scale=alt.Scale(zero=False)),
                    color=alt.Color("model:N", title=None, sort=order,
                                    scale=alt.Scale(domain=order, range=colours)),
                    tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("days:Q", title="Days ahead"),
                             alt.Tooltip("error:Q", title="Error", format=".2f")]))
    week = alt.Chart(pd.DataFrame({"x": [7]})).mark_rule(color=GRID, strokeWidth=2).encode(x="x:Q")
    return style(alt.layer(week, line).properties(height=360))


def buffer_chart(cap):
    d = cap[cap.Game != "All five games"].rename(
        columns={"Game": "game", "Window": "window", "Buffer above forecast (%)": "buffer"})
    return style(alt.Chart(d).mark_bar()
                 .encode(x=alt.X("game:N", title=None, axis=alt.Axis(labelAngle=0)),
                         xOffset="window:N",
                         y=alt.Y("buffer:Q", title="Buffer above forecast (%)"),
                         color=alt.Color("window:N", title=None,
                                         scale=alt.Scale(domain=["days 1-7", "days 8-14"], range=[BLUE, ORANGE])),
                         tooltip=[alt.Tooltip("game:N", title="Game"), alt.Tooltip("window:N", title="Window"),
                                  alt.Tooltip("buffer:Q", title="Buffer (%)")])
                 .properties(height=340))


def lift_bar(d, value, title):
    """Bar per game for a percent change; blue above zero, orange below."""
    d = d.rename(columns={value: "pct"})
    base = alt.Chart(d).encode(
        x=alt.X("game:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("pct:Q", title=title))
    bars = base.mark_bar().encode(
        color=alt.condition("datum.pct >= 0", alt.value(BLUE), alt.value(ORANGE)),
        tooltip=[alt.Tooltip("game:N", title="Game"), alt.Tooltip("pct:Q", title=title, format="+.1f")])
    labels = base.mark_text(dy=-8, color=INK2).encode(text=alt.Text("pct:Q", format="+.0f"))
    return style(alt.layer(bars, labels).properties(height=300))


def sale_chart(d):
    """One dot per sale (first 3 days vs the 7 days before), with the game's average as a tick."""
    base = alt.Chart(d)
    dots = base.mark_circle(size=70, opacity=0.65, color=BLUE).encode(
        x=alt.X("game:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("lift_pct:Q", title="Change in daily peak (%)"),
        tooltip=[alt.Tooltip("game:N", title="Game"),
                 alt.Tooltip("sale_start:T", title="Sale started", format="%d %b %Y"),
                 alt.Tooltip("discount:Q", title="Discount (%)"),
                 alt.Tooltip("lift_pct:Q", title="Change (%)", format="+.1f")])
    mean = base.mark_tick(color=INK, thickness=3, size=40).encode(
        x="game:N", y=alt.Y("mean(lift_pct):Q", title="Change in daily peak (%)"))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(
        y=alt.Y("y:Q", title="Change in daily peak (%)"))
    return style(alt.layer(zero, dots, mean).properties(height=320))


def trend_chart(r):
    r = r.dropna(subset=["roll7"])
    base = alt.Chart(r).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y")))
    daily_line = base.mark_line(color=MUTED, opacity=0.5, strokeWidth=1).encode(
        y=alt.Y("peak_players:Q", title="Daily peak players", scale=alt.Scale(zero=False)))
    roll = base.mark_line(color=BLUE, strokeWidth=2.4).encode(
        y="roll7:Q",
        tooltip=[alt.Tooltip("date:T", title="Date", format="%a %d %b %Y"),
                 alt.Tooltip("peak_players:Q", title="Daily peak", format=",.0f"),
                 alt.Tooltip("roll7:Q", title="7-day average", format=",.0f")])
    lock = alt.Chart(pd.DataFrame({"d": [dd.LOCKDOWN_START]})).mark_rule(
        color=ORANGE, strokeDash=[5, 4]).encode(x=alt.X("d:T", title=None))
    return style(alt.layer(daily_line, roll, lock).properties(height=320))


def monthly_chart(m):
    return style(alt.Chart(m).mark_bar(color=BLUE).encode(
        x=alt.X("month:T", title=None, timeUnit="yearmonth", axis=alt.Axis(format="%b %y")),
        y=alt.Y("avg_peak:Q", title="Average daily peak"),
        tooltip=[alt.Tooltip("month:T", title="Month", format="%b %Y"),
                 alt.Tooltip("avg_peak:Q", title="Average peak", format=",.0f"),
                 alt.Tooltip("mom_pct:Q", title="vs previous month (%)", format="+.1f")]
    ).properties(height=260))


def dips_heatmap(g):
    return style(alt.Chart(g).mark_rect().encode(
        x=alt.X("hod:O", title="Hour of day"),
        y=alt.Y("day:N", sort=dd.DAY_NAMES, title=None),
        color=alt.Color("dip_hours:Q", title="Dip hours", scale=alt.Scale(range=["#f1f0ec", ORANGE])),
        tooltip=[alt.Tooltip("day:N", title="Day"), alt.Tooltip("hod:O", title="Hour"),
                 alt.Tooltip("dip_hours:Q", title="Dip hours")]
    ).properties(height=240))


# ------------------------------------------------------------------ page
st.title("Player demand forecasts")
st.caption("Daily peak concurrent players for five Steam games, December 2017 to August 2020. "
           "Every forecast is a replay: what the model would have said on that date, using only "
           "data available then.")

tab_view, tab_acc, tab_cap, tab_ins, tab_data = st.tabs(
    ["Forecast viewer", "Accuracy", "Capacity buffers", "Insights", "Data and method"])

with tab_view:
    left, right = st.columns([1, 2])
    game = left.selectbox("Game", dd.game_list(intervals))
    options = dd.cutoffs_for(intervals, game)
    cutoff = right.select_slider("Forecast made on", options=options,
                                 value=dd.nearest(options, "2019-09-15"),
                                 format_func=lambda d: d.strftime("%d %b %Y"))
    hist, f = dd.forecast_window(intervals, daily, game, cutoff)
    k = dd.window_kpis(f)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast peak (next 14 days)", f"{k['forecast_peak']:,.0f}")
    c2.metric("Capacity to hold (95% bound)", f"{k['provision']:,.0f}", f"+{k['buffer']*100:.0f}% buffer",
              delta_color="off")
    c3.metric("Actual peak", f"{k['actual_peak']:,.0f}")
    if k["enough"]:
        c4.metric("Was the capacity enough?", "Yes", f"{k['days_over']} days over the bound",
                  delta_color="off")
    else:
        c4.metric("Was the capacity enough?", "No", f"short by {k['shortfall']*100:.0f}%",
                  delta_color="inverse")

    st.altair_chart(fan_chart(hist, f))
    st.caption("Grey: the 6 weeks before the forecast. Black: what really happened. Blue: forecast, "
               "with the shaded 80% range. Dashed: 95% upper bound, the level to plan capacity "
               "around. Hover for values.")
    with st.expander("Day-by-day numbers"):
        show = f[["date", "forecast", "lo80", "hi80", "hi95", "actual"]].copy()
        show["date"] = show["date"].dt.strftime("%a %d %b")
        show.columns = ["Date", "Forecast", "80% low", "80% high", "95% upper bound", "Actual"]
        st.dataframe(show.round(0), hide_index=True)

with tab_acc:
    period = st.radio("Period", ["All forecasts", "Lockdown period only (from March 2020)"],
                      horizontal=True)
    since = None if period == "All forecasts" else dd.LOCKDOWN_START
    st.subheader("How far off were the forecasts?")
    a, b = st.columns(2)
    a.markdown("**MASE** (error relative to a simple repeat-last-week forecast; lower is better, "
               "comparable across games)")
    a.dataframe(dd.accuracy_table(backtest, "mase", since), hide_index=True)
    b.markdown("**sMAPE** (typical percent error)")
    b.dataframe(dd.accuracy_table(backtest, "smape", since), hide_index=True)
    st.subheader("Error grows with distance")
    st.altair_chart(error_chart(dd.error_by_day(by_day, since)))
    st.caption("Average over the five games. Past about one week every model drifts toward the "
               "simple baseline, so treat forecasts beyond a week as ranges, not single numbers.")

with tab_cap:
    st.subheader("How much spare capacity does a forecast need?")
    cap = dd.capacity_table(intervals)
    st.markdown("The **buffer** is how far the 95% upper bound sits above the point forecast. "
                "Stable games need a small buffer, spiky ones a large one, and every game needs "
                "more in week two. A good set of ranges catches about 80% of outcomes inside the "
                "80% range and goes above the 95% bound on about 5% of days.")
    st.altair_chart(buffer_chart(cap))
    st.dataframe(cap, hide_index=True)

with tab_ins:
    ins = dd.load_insights(PROC)
    if ins is None:
        st.info("The insight tables are not there yet. Run `python src/run_sql.py` from the project "
                "folder (needs `pip install duckdb`), then reload this page.")
    else:
        st.markdown("What happened in the history, answered with SQL (`sql/insights/`). "
                    "These describe the past; the forecasts above look ahead.")
        a, b = st.columns(2)
        a.subheader("Lockdown lift")
        a.altair_chart(lift_bar(ins["lockdown"], "lift_pct", "Change in average daily peak (%)"))
        a.caption("Mar-Jun 2020 against Mar-Jun 2019. Includes each game's own trend: PUBG was "
                  "already shrinking, so its fall is not all lockdown.")
        b.subheader("Weekend lift")
        b.altair_chart(lift_bar(ins["weekend"], "weekend_lift_pct", "Weekend vs weekday peak (%)"))
        b.caption("Average Saturday/Sunday peak against Monday-Friday, whole history.")

        st.subheader("Do sales move the peak?")
        st.altair_chart(sale_chart(ins["sale_effect"]))
        st.caption("Each dot is one sale: average peak over its first 3 days against the 7 days "
                   "before. Black tick: the game's average. Dots scatter widely on both sides of "
                   "zero and there are only 8-15 sales per game, so read this as no reliable effect.")
        with st.expander("Sale summary table"):
            t = ins["sale_summary"].rename(columns={
                "game": "Game", "sales": "Sales", "avg_lift_pct": "Average change (%)",
                "share_positive_pct": "Sales that rose (%)", "worst_pct": "Worst (%)", "best_pct": "Best (%)"})
            st.dataframe(t, hide_index=True)

        st.subheader("Trend and unusual dips by game")
        pick = st.selectbox("Game", sorted(ins["rolling"].game.unique()), key="insight_game")
        st.altair_chart(trend_chart(ins["rolling"][ins["rolling"].game == pick]))
        st.caption("Grey: daily peak. Blue: 7-day average. Dashed orange: start of the March 2020 lockdown.")
        st.altair_chart(monthly_chart(ins["monthly"][ins["monthly"].game == pick]))
        st.caption("Average daily peak per month. The first (Dec 2017) and last (Aug 2020) months are partial.")
        st.altair_chart(dips_heatmap(dd.dips_grid(ins["dips"], pick)))
        st.caption("When hours were flagged as unusually low during cleaning, by weekday and hour "
                   "(dataset's own clock). PUBG's Wednesday early-morning block is weekly maintenance.")

with tab_data:
    st.subheader("Where the data comes from")
    st.markdown(
        "Steam Games Dataset (Wannigamage, Barlow, Lakshika and Kasmarik, Mendeley Data, 2020, "
        "DOI 10.17632/ycy3sy3vj2.1). Players online at the same moment, sampled every 5 minutes. "
        "The licence is not stated on the dataset page, so check it before republishing.")
    st.markdown(
        "- **Metric:** daily peak concurrent players, the number servers have to handle. "
        "It is not daily active users and not unique players.\n"
        "- **Cleaning:** hours with no readings, or with a failed reading stored as 0, were rebuilt: "
        "short gaps (6 hours or fewer) by a straight line, longer ones from the same hour a week "
        "earlier and later. Unusually low hours are flagged but left unchanged (PUBG shows a weekly "
        "maintenance pattern).\n"
        "- **Forecast:** the average of an ETS and a SARIMAX model, 14 days ahead, re-fitted at "
        "43 cutoffs per game.\n"
        "- **Ranges:** built from the model's own past misses, using only misses already known at "
        "the time of each forecast.\n"
        "- **Insights:** plain SQL (window functions) run in DuckDB over the cleaned tables; "
        "see `sql/insights/`.\n"
        "- **Limits:** data ends in August 2020; only five games; the 2020 lockdown breaks every "
        "model, which over-forecast several games afterwards.")
    st.dataframe(dd.data_quality(daily), hide_index=True)