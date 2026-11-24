"""
VIKA Wealth — SIF Performance Tracker

Data, refreshed at most every 6 hours:
  1. the saved copy  data/sif_nav_history.csv + data/sif_meta.csv  (your own history, never lost)
  2. sif.tigzig.com  full SIF NAV history since launch (free mirror of AMFI)
  3. AMFI            today's SIF NAVs (portal.amfiindia.com/spages/SIF_NAVAll.txt)
New rows are appended and the saved copy is written back: to the `data` branch of your GitHub
repo when [github] secrets are set (Streamlit Cloud), otherwise to the local data/ folder.
"""
import base64
import io
import os
import re
import urllib.request
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ───────────────────────── config ─────────────────────────
NAV_URL = "https://sif.tigzig.com/sif_nav.csv"
META_URL = "https://sif.tigzig.com/sif_meta.csv"
AMFI_LATEST_URL = "https://portal.amfiindia.com/spages/SIF_NAVAll.txt"
ISSUE_PRICE = 10.0
BENCHMARKS = {"Nifty 50": "^NSEI", "Nifty 500": "^CRSLDX"}
LOOKBACKS = {"1M": 30, "3M": 91, "6M": 182, "1Y": 365}
MIN_OBS = 20
GOLD, GREEN, RED, GREY = "#C9A227", "#3FB950", "#F85149", "#8B949E"
SHORT = {  # short labels for the category picker
    "Equity Long-Short Fund": "Equity L/S",
    "Equity Ex-Top 100 Long-Short Fund": "Ex-Top 100 L/S",
    "Sector Rotation Long-Short Fund": "Sector Rotation L/S",
    "Hybrid Long-Short Fund": "Hybrid L/S",
    "Active Asset Allocator Long-Short Fund": "Active Asset Allocator L/S",
}

st.set_page_config(page_title="SIF Tracker · VIKA Wealth", page_icon="📊", layout="wide")
st.markdown(f"""<style>
  .block-container {{padding-top: 1.4rem;}}
  div[data-testid="stMetricValue"] {{color: {GOLD};}}
</style>""", unsafe_allow_html=True)


def plot_layout(fig, title="", h=420):
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=h,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=45 if title else 10, b=10),
                      legend=dict(orientation="h", y=-0.12))
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)")
    return fig


# ───────────────────────── data ─────────────────────────
def fund_name(name: str) -> str:
    """'qsif Equity Long Short Fund - Growth Option - Direct Plan' -> 'qsif Equity Long Short Fund'"""
    n = re.sub(r"(?i)\s*[-–]?\s*\b(direct|regular)\s*plan\b", "", name)
    n = re.sub(r"(?i)\s*[-–]?\s*\b(growth|idcw|dividend)\b(\s*option)?", "", n)
    return re.sub(r"[\s\-–]+$", "", " ".join(n.split()))


def parse_amfi_latest(text: str) -> pd.DataFrame:
    rows, cat, amc = [], "", ""
    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        if ";" not in line:
            if "(" in line and "Scheme" in line:
                cat = line[line.find("(") + 1: line.rfind(")")].split(" - ")[-1].strip()
            else:
                amc = line
            continue
        p = [x.strip() for x in line.split(";")]
        if p[0].lower().startswith("scheme code"):
            continue
        rows.append(dict(code=p[0], date=p[7], nav=p[6], scheme_name=p[3], amc=amc, category=cat,
                         plan="Direct" if "direct" in p[4].lower() else "Regular" if p[4] else "",
                         option="IDCW" if "idcw" in p[5].lower() else "Growth"))
    df = pd.DataFrame(rows)
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    return df.dropna(subset=["nav", "date"])


# ── saved copy: GitHub repo (Streamlit Cloud) or local data/ folder (your Mac) ──
NAV_FILE, META_FILE = "data/sif_nav_history.csv", "data/sif_meta.csv"


def _gh():
    """GitHub settings from .streamlit/secrets.toml / Streamlit Cloud secrets, or None."""
    try:
        g = st.secrets["github"]
        return {"token": g["token"], "repo": g["repo"], "branch": g.get("branch", "data")}
    except Exception:
        return None


def _gh_api(method, url, cfg, **kw):
    import requests
    h = {"Authorization": f"Bearer {cfg['token']}", "Accept": "application/vnd.github+json"}
    h.update(kw.pop("headers", {}))
    return requests.request(method, f"https://api.github.com/repos/{cfg['repo']}{url}",
                            headers=h, timeout=60, **kw)


def _ensure_branch(cfg):
    if _gh_api("GET", f"/git/ref/heads/{cfg['branch']}", cfg).status_code == 200:
        return
    default = _gh_api("GET", "", cfg).json()["default_branch"]
    sha = _gh_api("GET", f"/git/ref/heads/{default}", cfg).json()["object"]["sha"]
    _gh_api("POST", "/git/refs", cfg, json={"ref": f"refs/heads/{cfg['branch']}", "sha": sha})


def read_saved(path):
    cfg = _gh()
    try:
        if cfg:
            r = _gh_api("GET", f"/contents/{path}", cfg, params={"ref": cfg["branch"]},
                        headers={"Accept": "application/vnd.github.raw"})
            return pd.read_csv(io.StringIO(r.text)) if r.status_code == 200 else None
        return pd.read_csv(path) if os.path.exists(path) else None
    except Exception:
        return None


def write_saved(path, df, message):
    """Overwrite the saved CSV. Returns True on success."""
    text = df.to_csv(index=False, date_format="%Y-%m-%d")
    cfg = _gh()
    try:
        if not cfg:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").write(text)
            return True
        _ensure_branch(cfg)
        cur = _gh_api("GET", f"/contents/{path}", cfg, params={"ref": cfg["branch"]})
        body = {"message": message, "branch": cfg["branch"],
                "content": base64.b64encode(text.encode()).decode()}
        if cur.status_code == 200:
            body["sha"] = cur.json()["sha"]
        return _gh_api("PUT", f"/contents/{path}", cfg, json=body).status_code in (200, 201)
    except Exception:
        return False


def _code(x):
    """'SIF-120' / '120' / 120 -> 120"""
    return pd.to_numeric(pd.Series(x).astype(str).str.replace("SIF-", "", regex=False), errors="coerce")


@st.cache_data(ttl=6 * 3600, show_spinner="Updating SIF NAV history…")
def load_data():
    """Saved copy + whatever the live sources add today; saves back only when there is new data.
    Returns (nav, meta, status dict)."""
    status = {"mirror": False, "amfi": False, "saved_rows": 0, "added": 0, "wrote": None}

    saved_nav = read_saved(NAV_FILE)
    saved_meta = read_saved(META_FILE)
    navs = [saved_nav] if saved_nav is not None else []
    metas = [saved_meta] if saved_meta is not None else []
    status["saved_rows"] = 0 if saved_nav is None else len(saved_nav)

    try:  # 1) full history from the mirror
        n = pd.read_csv(NAV_URL, usecols=["scheme_code", "date", "nav"])
        m = pd.read_csv(META_URL)
        navs.append(n.rename(columns={"scheme_code": "code"}))
        metas.append(m.rename(columns={"scheme_code": "code", "category_sub": "category",
                                       "scheme_plan": "plan", "scheme_option": "option"}))
        status["mirror"] = True
    except Exception:
        pass
    try:  # 2) today's NAVs straight from AMFI
        req = urllib.request.Request(AMFI_LATEST_URL, headers={"User-Agent": "Mozilla/5.0"})
        live = parse_amfi_latest(urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore"))
        navs.append(live[["code", "date", "nav"]])
        metas.append(live)
        status["amfi"] = True
    except Exception:
        pass
    if not navs:
        st.error("No saved history yet and both live sources are unreachable. Try again later.")
        st.stop()

    nav = pd.concat(navs, ignore_index=True)
    nav["code"] = _code(nav["code"]).values
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce")
    nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
    nav = (nav.dropna().astype({"code": int})
              .drop_duplicates(["code", "date"], keep="first")   # saved copy wins on overlap
              .sort_values(["code", "date"]).reset_index(drop=True))

    cols = ["code", "scheme_name", "amc", "category", "plan", "option"]
    meta = pd.concat([m.reindex(columns=cols) for m in metas], ignore_index=True)
    meta["code"] = _code(meta["code"]).values
    meta = meta.dropna(subset=["code"]).astype({"code": int})
    for c in cols[1:]:
        meta[c] = meta[c].fillna("").astype(str)
    # prefer the richest row per scheme (one that has category + plan filled)
    meta["filled"] = meta[["category", "plan", "amc"]].ne("").sum(axis=1)
    meta = (meta.sort_values("filled", ascending=False).drop_duplicates("code")
                .drop(columns="filled").sort_values("code").reset_index(drop=True))

    status["added"] = len(nav) - status["saved_rows"]
    if status["added"] > 0:
        stamp = f"{nav['date'].max():%Y-%m-%d}"
        ok = write_saved(NAV_FILE, nav, f"SIF NAVs to {stamp} (+{status['added']} rows)")
        ok = write_saved(META_FILE, meta, f"SIF scheme master {stamp}") and ok
        status["wrote"] = ok
    meta["fund"] = meta["scheme_name"].map(fund_name)
    return nav, meta, status


@st.cache_data(ttl=6 * 3600, show_spinner="Loading benchmark…")
def load_benchmark(ticker: str, start) -> pd.Series:
    try:
        import yfinance as yf
        s = yf.download(ticker, start=start - timedelta(days=10), progress=False, auto_adjust=True)["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def pick_variants(meta: pd.DataFrame, plan: str) -> pd.DataFrame:
    """One row per fund: the chosen plan, Growth option where available."""
    m = meta[meta["plan"].str.lower().eq(plan.lower()) | meta["plan"].eq("")].copy()
    m["prio"] = np.where(m["option"].str.lower().str.contains("growth"), 0, 1)
    m["blank_plan"] = m["plan"].eq("")
    return m.sort_values(["prio", "blank_plan"]).drop_duplicates(["amc", "fund"])


# ───────────────────────── metrics ─────────────────────────
def point_return(s, days):
    past = s[s.index <= s.index[-1] - timedelta(days=days)]
    return s.iloc[-1] / past.iloc[-1] - 1 if len(past) else np.nan


def risk_stats(s, bench):
    r = s.pct_change().dropna()
    out = dict(vol=np.nan, mdd=np.nan, beta=np.nan, up=np.nan, down=np.nan)
    if len(r) >= MIN_OBS:
        out["vol"] = r.std() * np.sqrt(252)
        out["mdd"] = (s / s.cummax() - 1).min()
    if not bench.empty:
        j = pd.concat([r, bench.pct_change()], axis=1, join="inner").dropna()
        j.columns = ["f", "b"]
        if len(j) >= MIN_OBS:
            out["beta"] = j.cov().iloc[0, 1] / j.b.var()
            up, dn = j[j.b > 0], j[j.b < 0]
            out["up"] = up.f.mean() / up.b.mean() if len(up) else np.nan
            out["down"] = dn.f.mean() / dn.b.mean() if len(dn) else np.nan
    return out


@st.cache_data(show_spinner=False)
def build_table(nav, funds, bench):
    rows = []
    for _, m in funds.iterrows():
        s = nav[nav.code == m.code].set_index("date")["nav"]
        if s.empty:
            continue
        rs = risk_stats(s, bench)
        rows.append({"Fund": m.fund, "AMC": m.amc, "Category": m.category, "code": m.code,
                     "Launched": s.index[0].date(), "NAV": s.iloc[-1],
                     "Since launch": s.iloc[-1] / ISSUE_PRICE - 1,
                     **{k: point_return(s, d) for k, d in LOOKBACKS.items()},
                     "Vol": rs["vol"], "Max DD": rs["mdd"], "Beta": rs["beta"],
                     "Up cap": rs["up"], "Down cap": rs["down"]})
    return pd.DataFrame(rows)


# ───────────────────────── UI ─────────────────────────
nav, meta, status = load_data()

with st.sidebar:
    st.markdown(f"### <span style='color:{GOLD}'>VIKA</span> · SIF Tracker", unsafe_allow_html=True)
    plan = st.radio("Plan", ["Direct", "Regular"], horizontal=True)
    bench_name = st.selectbox("Benchmark", list(BENCHMARKS))
    with st.expander("Data status"):
        tick = lambda ok: "✅" if ok else "❌"
        st.markdown(f"{tick(status['mirror'])} History mirror (sif.tigzig.com)  \n"
                    f"{tick(status['amfi'])} AMFI latest NAVs  \n"
                    f"💾 Saved copy: {len(nav):,} rows ({status['saved_rows']:,} before this refresh)  \n"
                    + ("Saved to " + ("GitHub `data` branch" if _gh() else "local data/ folder")
                       + (" ✅" if status["wrote"] else " ❌ (check the token)") if status["wrote"] is not None
                       else "Nothing new to save"))
        if st.button("Refresh now", width="stretch"):
            load_data.clear()
            st.rerun()
    st.caption(f"Risk stats need {MIN_OBS}+ trading days. Source: AMFI SIF NAVs. "
               "Internal research; past performance is not indicative of future returns.")

st.title("SIF Performance Tracker")
st.caption(f"NAVs {nav['date'].min():%d %b %Y} → {nav['date'].max():%d %b %Y} · {plan} plan, Growth")
if not status["mirror"] and not status["amfi"]:
    st.warning("Live sources are down. Showing the saved history; it will catch up on the next refresh.")

# step 1 — category
cats = [c for c in SHORT if c in set(meta["category"])] + \
       sorted(c for c in set(meta["category"]) - set(SHORT) if c)
labels = ["All"] + [SHORT.get(c, c) for c in cats]
choice = st.segmented_control("Category", labels, default="All", label_visibility="collapsed") or "All"
category = None if choice == "All" else cats[labels.index(choice) - 1]

funds = pick_variants(meta, plan)
if category:
    funds = funds[funds["category"] == category]
bench = load_benchmark(BENCHMARKS[bench_name], nav["date"].min())
table = build_table(nav, funds, bench)

if table.empty:
    st.info("No funds in this category yet.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Funds", len(table))
c2.metric("Above ₹10 issue price", f"{(table['Since launch'] > 0).sum()} of {len(table)}")
c3.metric("Median since launch", f"{table['Since launch'].median():+.1%}")
c4.metric("Median beta", f"{table['Beta'].median():.2f}" if table["Beta"].notna().any() else "–")

tab_list, tab_fund = st.tabs(["Funds", "Fund detail"])

with tab_list:
    show = table.sort_values("Since launch", ascending=False).copy()
    pct = ["Since launch", *LOOKBACKS, "Max DD"]
    show[pct] = show[pct] * 100
    cols = ["Fund", "AMC", "Launched", *pct, "Beta", "Down cap"]
    if not category:
        cols.insert(2, "Category")
        show["Category"] = show["Category"].map(lambda c: SHORT.get(c, c))
    st.dataframe(show[cols], hide_index=True, width="stretch", column_config={
        **{c: st.column_config.NumberColumn(format="%+.1f%%") for c in pct},
        "Max DD": st.column_config.NumberColumn(format="%.1f%%"),
        "Beta": st.column_config.NumberColumn(format="%.2f"),
        "Down cap": st.column_config.NumberColumn(format="%.2f",
                    help="Below 1 = fell less than the market on its down days"),
        "Since launch": st.column_config.NumberColumn(format="%+.1f%%", help="NAV vs ₹10 issue price"),
    })

    r = table.dropna(subset=["Beta"])
    if len(r):
        fig = go.Figure(go.Scatter(
            x=r.Beta, y=r["Since launch"], mode="markers+text", text=r.AMC, textposition="top center",
            textfont=dict(color=GREY, size=11), marker=dict(size=12, color=GOLD),
            customdata=r.Fund, hovertemplate="%{customdata}<br>Beta %{x:.2f} · %{y:+.1%}<extra></extra>"))
        fig.update_xaxes(title=f"Beta vs {bench_name} (lower = more hedged)")
        fig.update_yaxes(title="Return since launch", tickformat=".0%")
        st.plotly_chart(plot_layout(fig, "Return vs market exposure", 420), width="stretch")

with tab_fund:
    # step 2 — fund within the chosen category
    opts = table.sort_values(["AMC", "Fund"])
    pick = st.selectbox(f"Fund ({choice})", opts["Fund"] + " · " + opts["AMC"])
    m = opts.iloc[(opts["Fund"] + " · " + opts["AMC"]).tolist().index(pick)]
    s = nav[nav.code == m.code].set_index("date")["nav"]
    f = lambda v, spec: "–" if pd.isna(v) else format(v, spec)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("NAV", f"{m.NAV:.4f}", f"launched {m.Launched:%d %b %Y}", delta_color="off")
    k2.metric("Since launch", f(m["Since launch"], "+.1%"))
    k3.metric("Max drawdown", f(m["Max DD"], ".1%"))
    k4.metric("Beta", f(m.Beta, ".2f"))
    k5.metric("Down capture", f(m["Down cap"], ".2f"))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_scatter(x=s.index, y=s / s.iloc[0] * 100, name=m.Fund, line=dict(color=GOLD, width=2.4), row=1, col=1)
    fig.add_scatter(x=s.index, y=s / s.cummax() - 1, fill="tozeroy", line=dict(color=GOLD, width=1),
                    showlegend=False, row=2, col=1)
    b = bench[bench.index >= s.index[0]] if not bench.empty else bench
    if not b.empty:
        fig.add_scatter(x=b.index, y=b / b.iloc[0] * 100, name=bench_name,
                        line=dict(color=GREY, width=1.5, dash="dot"), row=1, col=1)
        fig.add_scatter(x=b.index, y=b / b.cummax() - 1, line=dict(color=GREY, width=1, dash="dot"),
                        showlegend=False, row=2, col=1)
    fig.update_yaxes(title="Growth of 100", row=1, col=1)
    fig.update_yaxes(title="Drawdown", tickformat=".0%", row=2, col=1)
    st.plotly_chart(plot_layout(fig, "Since launch vs market", 520), width="stretch")

    rets = {k: m[k] for k in LOOKBACKS}
    st.dataframe(pd.DataFrame([{k: (None if pd.isna(v) else v * 100) for k, v in rets.items()}]),
                 hide_index=True, column_config={k: st.column_config.NumberColumn(format="%+.1f%%") for k in rets})

with st.sidebar:
    st.download_button("Download table (CSV)", table.drop(columns=["code"]).to_csv(index=False),
                       "sif_funds.csv", width="stretch")
