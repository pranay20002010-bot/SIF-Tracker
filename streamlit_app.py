"""
VIKA Wealth — SIF Performance Tracker (single file)

  streamlit run streamlit_app.py              -> the dashboard
  python streamlit_app.py --update            -> append today's AMFI SIF NAVs to data/sif_nav_history.csv
  python streamlit_app.py --backfill          -> parse AMFI NAV-history .txt files in backfill/ into the CSV

The --update mode is what the nightly GitHub Action runs. It needs only pandas.
"""
import glob
import os
import sys
import urllib.request
from datetime import timedelta

import numpy as np
import pandas as pd

HIST_PATH = "data/sif_nav_history.csv"
SIF_NAV_URL = "https://portal.amfiindia.com/spages/SIF_NAVAll.txt"
COLS = ["date", "code", "isin", "brand", "category", "scheme", "plan", "option", "nav"]


# ───────────────────────── AMFI parsing + history (no Streamlit needed) ─────────────────────────
def fetch_latest_text() -> str:
    req = urllib.request.Request(SIF_NAV_URL, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", errors="ignore")


def _norm(s: str) -> str:
    return " ".join(str(s).replace(" ", " ").split())


def parse_amfi_text(text: str) -> pd.DataFrame:
    """Parse AMFI's semicolon format. Header-driven, so it works for both the
    latest-NAV file and NAV-history downloads. Heading lines carry category / AMC."""
    rows, header, category, brand = [], None, "", ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ";" not in line:
            # 'Open Ended Schemes(Equity Oriented ... - Hybrid Long-Short Fund)' or an AMC brand line
            if "(" in line and ")" in line and "Scheme" in line:
                inner = line[line.find("(") + 1: line.rfind(")")]
                category = _norm(inner.split(" - ")[-1])
            else:
                brand = _norm(line)
            continue
        parts = [p.strip() for p in line.split(";")]
        if header is None or parts[0].lower().startswith("scheme code"):
            header = [p.lower() for p in parts]
            continue
        rec = dict(zip(header, parts))

        def pick(*keys):
            for k in header:
                if any(key in k for key in keys):
                    return rec.get(k, "")
            return ""

        rows.append({
            "date": pick("date"),
            "code": pick("scheme code"),
            "isin": pick("isin div payout", "isin growth", "isin"),
            "brand": brand,
            "category": category,
            "scheme": _norm(pick("scheme name")),
            "plan": _norm(pick("plan")),
            "option": _norm(pick("option")),
            "nav": pick("net asset value"),
        })
    df = pd.DataFrame(rows, columns=COLS)
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    return df.dropna(subset=["nav", "date"])


def merge_into_history(new: pd.DataFrame) -> pd.DataFrame:
    os.makedirs(os.path.dirname(HIST_PATH), exist_ok=True)
    if os.path.exists(HIST_PATH):
        old = pd.read_csv(HIST_PATH, parse_dates=["date"], dtype={"code": str})
        # history files lack brand/category: fill them from what we already know
        meta = old.drop_duplicates("code", keep="last").set_index("code")
        for col in ["brand", "category", "plan", "option"]:
            mask = new[col].eq("") | new[col].isna()
            new.loc[mask, col] = new.loc[mask, "code"].map(meta[col]).fillna("")
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df = (df.drop_duplicates(["code", "date"], keep="last")
            .sort_values(["code", "date"]))
    df.to_csv(HIST_PATH, index=False, date_format="%Y-%m-%d")
    return df



if "--update" in sys.argv or "--backfill" in sys.argv:
    if "--backfill" in sys.argv:
        files = glob.glob("backfill/*.txt")
        frames = [parse_amfi_text(open(f, encoding="utf-8", errors="ignore").read()) for f in files]
        new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
        print(f"Backfill: parsed {len(new)} rows from {len(files)} file(s)")
    else:
        new = parse_amfi_text(fetch_latest_text())
        print(f"Fetched {len(new)} SIF NAV rows dated {new['date'].max():%d-%b-%Y}")
    hist = merge_into_history(new)
    print(f"History: {len(hist)} rows, {hist['code'].nunique()} schemes, "
          f"{hist['date'].min():%d-%b-%Y} to {hist['date'].max():%d-%b-%Y}")
    sys.exit(0)


# ───────────────────────── dashboard ─────────────────────────
import plotly.graph_objects as go
import streamlit as st

ISSUE_PRICE = 10.0
BENCHMARKS = {"Nifty 50": "^NSEI", "Nifty 500": "^CRSLDX"}
GOLD, NAVY, GREEN, RED, GREY = "#C9A227", "#1F3A5F", "#3FB950", "#F85149", "#8B949E"
LOOKBACKS = {"1W": 7, "1M": 30, "3M": 91, "6M": 182}
MIN_OBS = 20  # minimum overlapping daily returns before showing risk stats

st.set_page_config(page_title="SIF Tracker · VIKA Wealth", page_icon="📊", layout="wide")
st.markdown(f"""
<style>
  .block-container {{padding-top: 1.2rem;}}
  h1, h2, h3 {{letter-spacing: .3px;}}
  div[data-testid="stMetricValue"] {{color: {GOLD}; font-family: 'IBM Plex Mono', monospace;}}
  .vika-bar {{border-left: 4px solid {GOLD}; padding: .2rem .8rem; color: {GREY}; font-size: .85rem;}}
</style>""", unsafe_allow_html=True)


def plot_layout(fig, title="", h=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)), height=h,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=45, b=10), legend=dict(orientation="h", y=-0.15),
        font=dict(family="IBM Plex Sans, sans-serif", size=12),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)")
    return fig


# ── data ──
@st.cache_data(ttl=3600, show_spinner="Loading SIF NAVs…")
def load_navs():
    try:
        df = pd.read_csv(HIST_PATH, parse_dates=["date"], dtype={"code": str})
        source = "history"
    except FileNotFoundError:
        df = parse_amfi_text(fetch_latest_text())
        source = "live"
    for c in ["brand", "category", "plan", "option", "scheme"]:
        df[c] = df[c].fillna("").astype(str)
    return df, source


@st.cache_data(ttl=6 * 3600, show_spinner="Loading benchmark…")
def load_benchmark(ticker: str, start) -> pd.Series:
    try:
        import yfinance as yf
        s = yf.download(ticker, start=start - timedelta(days=10), progress=False,
                        auto_adjust=True)["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def pick_variant(df: pd.DataFrame, plan: str) -> pd.DataFrame:
    """One NAV series per strategy: chosen plan, Growth option."""
    latest = df.sort_values("date").groupby("code").tail(1).copy()
    pl = latest["plan"].str.lower()
    latest = latest[pl.str.contains(plan.lower()) | pl.eq("")]
    opt = latest["option"].str.lower()
    latest["prio"] = np.where(opt.str.contains("growth"), 0, np.where(opt.eq(""), 1, 2))
    latest = latest[latest["prio"] < 2]
    # where plan is blank, Direct usually carries the higher NAV
    latest = latest.sort_values(["scheme", "prio", "nav"], ascending=[True, True, plan != "Direct"])
    return latest.drop_duplicates("scheme")


# ───────────────────────── metrics ─────────────────────────
def point_return(s: pd.Series, days: int):
    end = s.index[-1]
    past = s[s.index <= end - timedelta(days=days)]
    return s.iloc[-1] / past.iloc[-1] - 1 if len(past) else np.nan


def risk_stats(nav: pd.Series, bench: pd.Series) -> dict:
    r = nav.pct_change().dropna()
    out = dict(vol=np.nan, mdd=np.nan, beta=np.nan, corr=np.nan, up=np.nan, down=np.nan, obs=len(r))
    if len(r) >= MIN_OBS:
        out["vol"] = r.std() * np.sqrt(252)
        out["mdd"] = (nav / nav.cummax() - 1).min()
    if bench.empty:
        return out
    b = bench.pct_change().dropna()
    j = pd.concat([r, b], axis=1, join="inner").dropna()
    j.columns = ["f", "b"]
    if len(j) >= MIN_OBS:
        out["beta"] = j.cov().iloc[0, 1] / j["b"].var()
        out["corr"] = j.corr().iloc[0, 1]
        upd, dnd = j[j.b > 0], j[j.b < 0]
        out["up"] = upd.f.mean() / upd.b.mean() if len(upd) else np.nan
        out["down"] = dnd.f.mean() / dnd.b.mean() if len(dnd) else np.nan
    return out


@st.cache_data(show_spinner=False)
def build_table(df, chosen, bench):
    rows = []
    for _, m in chosen.iterrows():
        s = df[df.code == m.code].set_index("date")["nav"].sort_index()
        row = dict(Strategy=m.scheme, AMC=m.brand, Category=m.category, Code=m.code,
                   NAV=s.iloc[-1], **{"As of": s.index[-1].date()},
                   **{"Since ₹10": s.iloc[-1] / ISSUE_PRICE - 1})
        for k, d in LOOKBACKS.items():
            row[k] = point_return(s, d)
        st_ = risk_stats(s, bench)
        row.update({"Vol": st_["vol"], "Max DD": st_["mdd"], "Beta": st_["beta"],
                    "Up cap": st_["up"], "Down cap": st_["down"], "Days": st_["obs"]})
        rows.append(row)
    return pd.DataFrame(rows)


# ───────────────────────── UI ─────────────────────────
df, source = load_navs()

with st.sidebar:
    st.markdown(f"### <span style='color:{GOLD}'>VIKA</span> · SIF Tracker", unsafe_allow_html=True)
    plan = st.radio("Plan", ["Direct", "Regular"], horizontal=True)
    cats = sorted(c for c in df["category"].unique() if c)
    sel_cats = st.multiselect("Category", cats, default=cats)
    bench_name = st.selectbox("Market benchmark", list(BENCHMARKS))
    st.caption("Beta, capture ratios and drawdown need at least "
               f"{MIN_OBS} trading days of NAV history.")

chosen = pick_variant(df, plan)
if sel_cats:
    chosen = chosen[chosen.category.isin(sel_cats)]
bench = load_benchmark(BENCHMARKS[bench_name], df["date"].min())
table = build_table(df, chosen, bench)

st.title("Specialised Investment Funds — Performance Tracker")
hist_days = df["date"].nunique()
st.markdown(
    f"<div class='vika-bar'>Source: AMFI SIF NAVs · {table['Strategy'].nunique()} strategies · "
    f"history {df['date'].min():%d %b %Y} → {df['date'].max():%d %b %Y} ({hist_days} NAV dates)"
    f"{' · <b>live snapshot only — set up the daily job to build history</b>' if source == 'live' else ''}"
    f"</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Strategies tracked", len(table))
c2.metric("Above ₹10 issue price", f"{(table['Since ₹10'] > 0).sum()} / {len(table)}")
c3.metric("Median since issue", f"{table['Since ₹10'].median():+.1%}" if len(table) else "–")
c4.metric("Median beta", f"{table['Beta'].median():.2f}" if table["Beta"].notna().any() else "–")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏁 Leaderboard", "🔍 Strategy deep-dive", "🎯 Risk map", "🧩 Categories", "🗂 Data"])

pct_cols = ["Since ₹10", *LOOKBACKS, "Vol", "Max DD"]
with tab1:
    sort_by = st.selectbox("Sort by", ["Since ₹10", *LOOKBACKS, "Down cap", "Beta", "Max DD"], index=0)
    view = table.sort_values(sort_by, ascending=sort_by in ["Down cap", "Beta"], na_position="last")
    st.dataframe(
        view.drop(columns=["Code"]).style
            .format({c: "{:+.2%}" for c in pct_cols}, na_rep="–")
            .format({"NAV": "{:.4f}", "Beta": "{:.2f}", "Up cap": "{:.2f}", "Down cap": "{:.2f}"}, na_rep="–")
            .map(lambda v: f"color:{GREEN}" if isinstance(v, float) and v > 0 else
                 (f"color:{RED}" if isinstance(v, float) and v < 0 else ""), subset=["Since ₹10", *LOOKBACKS]),
        width="stretch", hide_index=True, height=560)
    st.caption("Since ₹10 = NAV vs ₹10 issue price (ignores IDCW payouts, so use Growth plans). "
               "Down cap < 1 means the strategy fell less than the market on down days.")

with tab2:
    if table.empty:
        st.info("No strategies match the filters.")
    else:
        pick = st.selectbox("Strategy", table.sort_values("Strategy")["Strategy"])
        m = table[table.Strategy == pick].iloc[0]
        s = df[df.code == m.Code].set_index("date")["nav"].sort_index()
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("NAV", f"{m.NAV:.4f}", f"{m['1M']:+.2%} 1M" if pd.notna(m["1M"]) else None)
        k2.metric("Since ₹10", f"{m['Since ₹10']:+.2%}")
        k3.metric("Beta", "–" if pd.isna(m.Beta) else f"{m.Beta:.2f}")
        k4.metric("Down capture", "–" if pd.isna(m["Down cap"]) else f"{m['Down cap']:.2f}")
        k5.metric("Max drawdown", "–" if pd.isna(m["Max DD"]) else f"{m['Max DD']:.2%}")

        fig = go.Figure()
        fig.add_scatter(x=s.index, y=s / s.iloc[0] * 100, name=pick, line=dict(color=GOLD, width=2.4))
        b = bench[bench.index >= s.index[0]] if not bench.empty else bench
        if not b.empty:
            fig.add_scatter(x=b.index, y=b / b.iloc[0] * 100, name=bench_name,
                            line=dict(color=GREY, width=1.6, dash="dot"))
        st.plotly_chart(plot_layout(fig, "Growth of 100 (rebased to first common date)"),
                        width="stretch")

        col_a, col_b = st.columns(2)
        dd = s / s.cummax() - 1
        fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color=RED, width=1),
                                   name="Strategy"))
        if not b.empty:
            bdd = b / b.cummax() - 1
            fig.add_scatter(x=bdd.index, y=bdd, line=dict(color=GREY, width=1, dash="dot"), name=bench_name)
        fig.update_yaxes(tickformat=".0%")
        col_a.plotly_chart(plot_layout(fig, "Drawdown vs market", 340), width="stretch")

        if bench.empty:
            st.warning(f"{bench_name} data unavailable right now; showing the strategy alone.")
        j = pd.concat([s.pct_change(), bench.pct_change()], axis=1, join="inner").dropna()
        j.columns = ["f", "b"]
        win = st.slider("Rolling window (trading days)", 20, 120, 40, 10) if len(j) > 60 else 20
        if len(j) >= win + 5:
            rb = j.f.rolling(win).cov(j.b) / j.b.rolling(win).var()
            fig = go.Figure(go.Scatter(x=rb.index, y=rb, line=dict(color=GOLD), name="Rolling beta"))
            fig.add_hline(y=1, line=dict(color=GREY, dash="dot"))
            col_b.plotly_chart(plot_layout(fig, f"Rolling {win}-day beta — is it really hedging?", 340),
                               width="stretch")
        else:
            col_b.info("Rolling beta appears once there is enough overlapping history.")

with tab3:
    r = table.dropna(subset=["Beta"])
    if r.empty:
        st.info(f"Risk map needs at least {MIN_OBS} days of NAV history per strategy.")
    else:
        ca, cb = st.columns(2)
        fig = go.Figure(go.Scatter(
            x=r.Beta, y=r["Since ₹10"], mode="markers+text", text=r.AMC, textposition="top center",
            marker=dict(size=13, color=r["Down cap"], colorscale=[[0, GREEN], [0.5, GOLD], [1, RED]],
                        showscale=True, colorbar=dict(title="Down cap")),
            hovertext=r.Strategy, hoverinfo="text+x+y"))
        fig.update_yaxes(tickformat=".0%", title="Return since ₹10")
        fig.update_xaxes(title=f"Beta vs {bench_name}")
        ca.plotly_chart(plot_layout(fig, "Return vs market exposure", 460), width="stretch")

        fig = go.Figure(go.Scatter(
            x=r["Down cap"], y=r["Up cap"], mode="markers+text", text=r.AMC, textposition="top center",
            marker=dict(size=13, color=GOLD), hovertext=r.Strategy, hoverinfo="text+x+y"))
        fig.add_shape(type="line", x0=0, y0=0, x1=1.5, y1=1.5, line=dict(color=GREY, dash="dot"))
        fig.update_xaxes(title="Downside capture (lower is better)")
        fig.update_yaxes(title="Upside capture (higher is better)")
        cb.plotly_chart(plot_layout(fig, "Capture ratios — top-left is the sweet spot", 460),
                        width="stretch")

with tab4:
    if not table.empty:
        g = table.groupby("Category").agg(Strategies=("Strategy", "count"),
                                          Median_since=("Since ₹10", "median"),
                                          Median_1M=("1M", "median"),
                                          Median_beta=("Beta", "median")).reset_index()
        fig = go.Figure(go.Bar(x=g.Median_since, y=g.Category, orientation="h",
                               marker_color=[GREEN if v > 0 else RED for v in g.Median_since],
                               text=[f"{v:+.1%}" for v in g.Median_since], textposition="outside"))
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(plot_layout(fig, "Median return since issue, by category", 380),
                        width="stretch")
        st.dataframe(g.style.format({"Median_since": "{:+.2%}", "Median_1M": "{:+.2%}",
                                     "Median_beta": "{:.2f}"}, na_rep="–"),
                     width="stretch", hide_index=True)

with tab5:
    cov = df.groupby(["brand", "scheme"]).agg(first=("date", "min"), last=("date", "max"),
                                              navs=("nav", "count")).reset_index()
    st.dataframe(cov, width="stretch", hide_index=True)
    st.download_button("Download leaderboard (CSV)", table.to_csv(index=False), "sif_leaderboard.csv")
    st.download_button("Download full NAV history (CSV)", df.to_csv(index=False), "sif_nav_history.csv")

st.caption("For internal research. Past performance is not indicative of future returns. "
           "SIF investments carry higher risk, including potential loss of capital.")
