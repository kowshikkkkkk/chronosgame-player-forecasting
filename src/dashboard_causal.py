"""The "Causal analysis" tab of the dashboard.

Reads the tables written by causal_lockdown.py and causal_sales.py (reports/*.csv). Kept in its own
module so dashboard.py only needs two extra lines. All numbers shown come from those files.
"""
from pathlib import Path

import altair as alt
import pandas as pd

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
MAIN_DATE = "2020-03-11"
FILES = ["causal_lockdown_effect.csv", "causal_lockdown_placebo_summary.csv",
         "causal_lockdown_vs_sql.csv", "causal_sales_summary.csv"]


def _style(chart):
    return (chart.configure_view(strokeWidth=0)
            .configure_axis(grid=True, gridColor=GRID, domainColor=GRID, tickColor=GRID,
                            labelColor=INK2, titleColor=INK2)
            .configure_legend(labelColor=INK2, titleColor=INK2, orient="top", title=None))


def lockdown_chart(d):
    """Real estimate (orange, 10-90% line + dot) against the placebo 10-90% band (blue)."""
    base = alt.Chart(d).encode(x=alt.X("game:N", title=None, axis=alt.Axis(labelAngle=0)))
    plc = base.mark_bar(size=34, opacity=0.25, color=BLUE).encode(
        y=alt.Y("placebo_p10_pct:Q", title="Average lift in daily peak (%)"), y2="placebo_p90_pct:Q",
        tooltip=[alt.Tooltip("game:N", title="Game"),
                 alt.Tooltip("placebo_p10_pct:Q", title="Placebo 10th pct (%)", format="+.1f"),
                 alt.Tooltip("placebo_p90_pct:Q", title="Placebo 90th pct (%)", format="+.1f")])
    rng = base.mark_rule(color=ORANGE, strokeWidth=2.5).encode(y="real_p10_pct:Q", y2="real_p90_pct:Q")
    dot = base.mark_point(color=ORANGE, filled=True, size=110).encode(
        y="real_median_pct:Q",
        tooltip=[alt.Tooltip("game:N", title="Game"),
                 alt.Tooltip("real_median_pct:Q", title="Estimated lift, median (%)", format="+.1f"),
                 alt.Tooltip("real_p10_pct:Q", title="10th pct (%)", format="+.1f"),
                 alt.Tooltip("real_p90_pct:Q", title="90th pct (%)", format="+.1f")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    return _style(alt.layer(plc, zero, rng, dot).properties(height=360))


def sales_chart(d):
    long = d.melt(id_vars="game",
                  value_vars=["unadjusted_avg_pct", "weekday_adjusted_avg_pct", "did_avg_pct", "null_mean_pct"],
                  var_name="measure", value_name="pct")
    labels = {"unadjusted_avg_pct": "Unadjusted (README)", "weekday_adjusted_avg_pct": "Weekday-adjusted",
              "did_avg_pct": "With controls (DiD)", "null_mean_pct": "Random no-sale dates"}
    order = list(labels.values())
    long["measure"] = long["measure"].map(labels)
    return _style(alt.Chart(long).mark_bar().encode(
        x=alt.X("game:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset=alt.XOffset("measure:N", sort=order),
        y=alt.Y("pct:Q", title="Average change in daily peak (%)"),
        color=alt.Color("measure:N", sort=order,
                        scale=alt.Scale(domain=order, range=[MUTED, "#9ec5f0", BLUE, ORANGE])),
        tooltip=[alt.Tooltip("game:N", title="Game"), alt.Tooltip("measure:N", title="Measure"),
                 alt.Tooltip("pct:Q", title="Change (%)", format="+.1f")]).properties(height=340))


def render(st, reports_dir):
    reports = Path(reports_dir)
    missing = [f for f in FILES if not (reports / f).exists()]
    if missing:
        st.info("The causal tables are not there yet. Run `python src/causal_lockdown.py` and "
                "`python src/causal_sales.py` from the project folder, then reload this page.")
        return
    effect = pd.read_csv(reports / "causal_lockdown_effect.csv")
    placebo = pd.read_csv(reports / "causal_lockdown_placebo_summary.csv")
    vs_sql = pd.read_csv(reports / "causal_lockdown_vs_sql.csv")
    sales = pd.read_csv(reports / "causal_sales_summary.csv")

    st.markdown("Two estimates, both from observational data, so they are **estimated effects, not proof**. "
                "Both are about daily peak concurrent players only.")

    # ---- lockdown
    st.subheader("1. How much did the March 2020 lockdown change the daily peak?")
    st.markdown("ETS fitted on data before 11 March 2020 only, then 1,000 simulated no-lockdown futures. "
                "**Orange**: estimated average lift (dot = median, line = 10-90% of the simulated paths). "
                "**Blue band**: what the same method 'finds' at 22 fake dates before lockdown, where the true "
                "effect is zero. An estimate is only convincing if it sits clearly outside the blue band.")
    windows = list(placebo["window"].unique())
    window = st.radio("Window", windows, index=0, horizontal=True, key="causal_window")
    d = placebo[placebo["window"] == window]
    st.altair_chart(lockdown_chart(d))
    show = d[["game", "real_median_pct", "real_p10_pct", "real_p90_pct", "placebo_p10_pct",
              "placebo_p90_pct", "real_outside_placebo_p10_p90"]].round(1)
    show.columns = ["Game", "Estimated lift (%)", "10th pct", "90th pct", "Placebo 10th pct",
                    "Placebo 90th pct", "Outside placebo range?"]
    st.dataframe(show, hide_index=True)
    st.caption("The first-30-days window is the cleanest: placebo effects there centre near zero. Over longer "
               "windows the method drifts (GTA V and Rust placebo averages are about +15%), so only an estimate "
               "well outside the band counts. Civilization VI also had a sale starting 12 March 2020.")

    a, b = st.columns(2)
    with a:
        st.markdown("**Sensitivity to the intervention date** (median lift, %)")
        sens = effect[effect["window"] == window].pivot_table(index="game", columns="intervention",
                                                              values="median_pct").round(1)
        st.dataframe(sens)
        st.caption("A 23 March cutoff trains on 12 lockdown days, so it is expected to be lower.")
    with b:
        st.markdown("**Against the SQL insight**")
        cmp_ = vs_sql[["game", "sql_vs_mar_jun_2019_pct", "sql_vs_jan_feb_2020_pct",
                       "counterfactual_median_pct_11mar_30jun"]].round(1)
        cmp_.columns = ["Game", "SQL vs Mar-Jun 2019", "SQL vs Jan-Feb 2020", "Counterfactual (11 Mar-30 Jun)"]
        st.dataframe(cmp_, hide_index=True)
        st.caption("Year-on-year mixes lockdown with each game's own trend; PUBG was already falling.")

    # ---- sales
    st.subheader("2. Do Steam sales lift the daily peak?")
    st.markdown("Lift of the first 3 sale days against the same weekday in the 4 previous weeks, minus the "
                "same measure for games not on sale (difference-in-differences). Compared with random dates "
                "when nobody was on sale (1,000 permutations).")
    scopes = list(sales["scope"].unique())
    scope = st.radio("Sales included", scopes, index=len(scopes) - 1, horizontal=False, key="causal_scope")
    s = sales[sales["scope"] == scope].copy()
    st.altair_chart(sales_chart(s))
    t = s[["game", "sales_used", "unadjusted_avg_pct", "weekday_adjusted_avg_pct", "did_avg_pct",
           "null_mean_pct", "p_value"]].round(3)
    t.columns = ["Game", "Sales", "Unadjusted (%)", "Weekday-adjusted (%)", "With controls (%)",
                 "Random-date average (%)", "p-value"]
    st.dataframe(t, hide_index=True)
    st.caption("The clean-baseline view (no sale of the same game in the 4 baseline weeks, for real and random "
               "dates) is the fairer one. Pooled, a sale lift is not reliably detectable. The permutation pool "
               "has only 55 dates in 3 stretches, so p-values are rough, and the other two views are biased.")