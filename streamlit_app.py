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
GOLD, GREEN, RED, GREY = "#C9A227", "#3FB950", "#F85149", "#8B949E"
NAV_URL = "https://sif.tigzig.com/sif_nav.csv"
META_URL = "https://sif.tigzig.com/sif_meta.csv"
AMFI_LATEST_URL = "https://portal.amfiindia.com/spages/SIF_NAVAll.txt"
ISSUE_PRICE = 10.0
BENCHMARKS = {"Nifty 50": "^NSEI", "Nifty 500": "^CRSLDX"}
LOOKBACKS = {"1M": 30, "3M": 91, "6M": 182, "1Y": 365}
MIN_OBS = 20
HEDGE_MIN_OBS = 40  # days needed before a hedge verdict
# SEBI SIF rules per strategy, and the beta band they roughly imply:
#   low  = minimum equity − 25% max unhedged short   high = maximum equity allowed
# Beta also picks up stock selection, so the band is a guide, not a compliance test.
MANDATE = {
    "Equity Long-Short Fund": dict(
        rule="≥80% equity · unhedged shorts ≤25% of NAV", band=(0.55, 1.00), index="Nifty 500"),
    "Equity Ex-Top 100 Long-Short Fund": dict(
        rule="≥65% in stocks outside the top 100 · unhedged shorts ≤25%", band=(0.40, 1.00),
        index="Nifty Midcap 150"),
    "Sector Rotation Long-Short Fund": dict(
        rule="≥80% equity in ≤4 sectors · unhedged shorts ≤25% (sector level)", band=(0.55, 1.20),
        index="Nifty 500"),
    "Hybrid Long-Short Fund": dict(
        rule="≥25% equity and ≥25% debt · unhedged shorts ≤25%", band=(0.00, 0.75), index="Nifty 50"),
    "Active Asset Allocator Long-Short Fund": dict(
        rule="Dynamic across equity, debt, REITs/InvITs, commodity derivatives · unhedged shorts ≤25%",
        band=(0.00, 1.00), index="Nifty 50"),
}
INDEX_TICKERS = {"Nifty 50": "^NSEI", "Nifty 500": "^CRSLDX", "Nifty Midcap 150": "NIFTYMIDCAP150.NS"}
VERDICT_COLOR = {"🟢": GREEN, "🟡": GOLD, "🔴": RED, "⚠️": "#E3842B", "⏳": GREY}
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
    n = re.sub(r"(?i)\s*[-–]?\s*\b(direct|regular)\b(\s*plan)?", "", name)
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
        # the mirror also has its own long 'category' column: keep only what we need, then rename
        m = m[["scheme_code", "scheme_name", "amc", "category_sub", "scheme_plan", "scheme_option"]]
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
    meta = pd.concat([m.loc[:, ~m.columns.duplicated()].reindex(columns=cols) for m in metas],
                     ignore_index=True)
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


# ───────────────────────── hedge check ─────────────────────────
def hedge_stats(s, mkt):
    """Behaviour of a fund's daily returns against its mandate index."""
    out = dict(beta=np.nan, up=np.nan, down=np.nan, win=np.nan, worst_f=np.nan, worst_m=np.nan, obs=0)
    if mkt.empty:
        return out
    j = pd.concat([s.pct_change(), mkt.pct_change()], axis=1, join="inner").dropna()
    j.columns = ["f", "m"]
    out["obs"] = len(j)
    if len(j) < MIN_OBS:
        return out
    up, dn = j[j.m > 0], j[j.m < 0]
    worst = j.nsmallest(min(10, len(dn)), "m")
    out.update(beta=j.cov().iloc[0, 1] / j.m.var(),
               up=up.f.mean() / up.m.mean() if len(up) else np.nan,
               down=dn.f.mean() / dn.m.mean() if len(dn) else np.nan,
               win=(dn.f > dn.m).mean() if len(dn) else np.nan,
               worst_f=worst.f.mean(), worst_m=worst.m.mean())
    return out


def verdict(h, band):
    if h["obs"] < HEDGE_MIN_OBS or pd.isna(h["beta"]):
        return "⏳ Too early to judge"
    if h["beta"] > band[1] + 0.05:
        return "⚠️ More market exposure than the mandate implies"
    if h["beta"] >= 0.85 and h["down"] >= 0.90:
        return "🔴 Long-only in disguise"
    if h["down"] < 0.80 and (h["up"] - h["down"]) >= 0.10:
        return "🟢 Hedging works: falls less, keeps upside"
    if h["down"] < 0.80:
        return "🟡 Lower exposure, no hedging edge"
    return "🟡 Light hedging"


@st.cache_data(show_spinner="Running hedge check…")
def build_hedge(nav, funds, start):
    idx = {name: load_benchmark(t, start) for name, t in INDEX_TICKERS.items()}
    rows = []
    for _, m in funds.iterrows():
        s = nav[nav.code == m.code].set_index("date")["nav"]
        md = MANDATE.get(m.category, dict(rule="–", band=(0.0, 1.0), index="Nifty 500"))
        index_name = md["index"] if not idx[md["index"]].empty else "Nifty 500"
        h = hedge_stats(s, idx[index_name])
        rows.append({"Fund": m.fund, "AMC": m.amc, "Category": m.category, "code": m.code,
                     "Verdict": verdict(h, md["band"]), "Beta": h["beta"],
                     "Mandate band": f"{md['band'][0]:.2f}–{md['band'][1]:.2f}",
                     "Down cap": h["down"], "Up cap": h["up"], "Down-day wins": h["win"],
                     "Worst 10 days: fund": h["worst_f"], "Worst 10 days: index": h["worst_m"],
                     "vs": index_name, "Days": h["obs"], "rule": md["rule"], "band": md["band"]})
    return pd.DataFrame(rows), idx


def render_hedge_check(nav, funds, category, choice, plan):
    hc, idx = build_hedge(nav, funds, nav["date"].min())
    if hc.empty:
        st.info("No funds in this category yet.")
        return
    f = lambda v, spec: "–" if pd.isna(v) else format(v, spec)

    st.subheader("Is it actually hedging?")
    if category:
        md = MANDATE.get(category)
        if md:
            st.markdown(f"**SEBI mandate:** {md['rule']}  \n"
                        f"**Beta this allows (approx.):** {md['band'][0]:.2f} to {md['band'][1]:.2f} "
                        f"vs {md['index']}")
    st.caption("Measured from daily NAVs against the index that fits each strategy. Down capture below 1 "
               "= falls less than the market; a real hedge also keeps up capture above down capture. "
               f"Verdicts need {HEDGE_MIN_OBS}+ trading days.")

    counts = hc["Verdict"].str.split(" ").str[0].value_counts()
    k = st.columns(4)
    k[0].metric("🟢 Hedging works", int(counts.get("🟢", 0)))
    k[1].metric("🟡 Partial", int(counts.get("🟡", 0)))
    k[2].metric("🔴 / ⚠️ Not as promised", int(counts.get("🔴", 0) + counts.get("⚠️", 0)))
    k[3].metric("⏳ Too early", int(counts.get("⏳", 0)))

    order = {"🟢": 0, "🟡": 1, "⚠️": 2, "🔴": 3, "⏳": 4}
    hc = hc.assign(_o=hc["Verdict"].str.split(" ").str[0].map(order)).sort_values(["_o", "Down cap"])
    show = hc.copy()
    for c in ["Down-day wins", "Worst 10 days: fund", "Worst 10 days: index"]:
        show[c] = show[c] * 100
    cols = ["Fund", "AMC", "Verdict", "Beta", "Mandate band", "Down cap", "Up cap", "Down-day wins",
            "Worst 10 days: fund", "Worst 10 days: index", "vs", "Days"]
    if not category:
        cols.insert(2, "Category")
        show["Category"] = show["Category"].map(lambda c: SHORT.get(c, c))
    event = st.dataframe(
        show[cols], hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        key=f"hedge_{choice}_{plan}",
        column_config={
            "Verdict": st.column_config.TextColumn(width="large"),
            "Beta": st.column_config.NumberColumn(format="%.2f", help="Sensitivity to the index ≈ net equity exposure"),
            "Down cap": st.column_config.NumberColumn(format="%.2f", help="Fund move ÷ index move on index down days"),
            "Up cap": st.column_config.NumberColumn(format="%.2f", help="Fund move ÷ index move on index up days"),
            "Down-day wins": st.column_config.NumberColumn(format="%.0f%%", help="Share of index down days the fund fell less"),
            "Worst 10 days: fund": st.column_config.NumberColumn(format="%+.2f%%", help="Average fund move on the index's 10 worst days"),
            "Worst 10 days: index": st.column_config.NumberColumn(format="%+.2f%%"),
            "vs": st.column_config.TextColumn("Index"),
        })
    st.caption("Click a row to see that fund below.")

    r = hc.dropna(subset=["Down cap", "Up cap"])
    if len(r):
        colors = [VERDICT_COLOR.get(v.split(" ")[0], GREY) for v in r["Verdict"]]
        fig = go.Figure(go.Scatter(
            x=r["Down cap"], y=r["Up cap"], mode="markers+text", text=r.AMC, textposition="top center",
            textfont=dict(color=GREY, size=11), marker=dict(size=13, color=colors),
            customdata=np.stack([r.Fund, r.Verdict], axis=1),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>Down %{x:.2f} · Up %{y:.2f}<extra></extra>"))
        hi = max(1.2, float(r[["Down cap", "Up cap"]].max().max()) + 0.1)
        lo = min(0.0, float(r[["Down cap", "Up cap"]].min().min()) - 0.1)
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color=GREY, dash="dot"))
        fig.add_annotation(x=lo + 0.05, y=hi - 0.05, text="Real hedge: falls less, keeps upside",
                           showarrow=False, xanchor="left", font=dict(color=GREEN, size=11))
        fig.add_annotation(x=hi - 0.05, y=lo + 0.05, text="Worst of both", showarrow=False,
                           xanchor="right", font=dict(color=RED, size=11))
        fig.update_xaxes(title="Down capture (lower is better)", range=[lo, hi])
        fig.update_yaxes(title="Up capture (higher is better)", range=[lo, hi])
        st.plotly_chart(plot_layout(fig, "Capture map: above the dotted line = asymmetric (a real hedge)", 460),
                        width="stretch")

    # ── one fund ──
    st.divider()
    st.subheader("Fund hedge detail")
    labels_f = (hc["Fund"] + " · " + hc["AMC"]).tolist()
    key = f"hpick_{choice}_{plan}"
    sel = event.selection.rows if event and event.selection else []
    if sel:
        st.session_state[key] = labels_f[sel[0]]
    pick = st.selectbox("Fund", labels_f, key=key, label_visibility="collapsed")
    m = hc.iloc[labels_f.index(pick)]
    s = nav[nav.code == m.code].set_index("date")["nav"]
    mkt = idx[m["vs"]]
    st.markdown(f"**{m.Verdict}**  \n{SHORT.get(m.Category, m.Category)} · mandate: {m.rule}")

    k = st.columns(4)
    k[0].metric("Beta", f(m.Beta, ".2f"), f"band {m['Mandate band']}", delta_color="off")
    k[1].metric("Down capture", f(m["Down cap"], ".2f"))
    k[2].metric("Up capture", f(m["Up cap"], ".2f"))
    k[3].metric("Down-day wins", f(m["Down-day wins"], ".0%"))

    j = pd.concat([s.pct_change(), mkt.pct_change()], axis=1, join="inner").dropna()
    j.columns = ["f", "m"]
    a, b = st.columns(2)
    win = 40
    if len(j) >= win + 10:
        rb = (j.f.rolling(win).cov(j.m) / j.m.rolling(win).var()).dropna()
        fig = go.Figure()
        fig.add_hrect(y0=m.band[0], y1=m.band[1], fillcolor=GOLD, opacity=0.08, line_width=0,
                      annotation_text="mandate band", annotation_position="top left")
        fig.add_scatter(x=rb.index, y=rb, line=dict(color=GOLD, width=2), name="40-day beta")
        fig.add_hline(y=1, line=dict(color=GREY, dash="dot"))
        a.plotly_chart(plot_layout(fig, f"Rolling 40-day beta vs {m['vs']}", 360), width="stretch")
    else:
        a.info("Rolling beta appears after about 50 trading days.")
    if len(j) >= MIN_OBS:
        w = j.nsmallest(10, "m").sort_index()
        lab = [d.strftime("%d %b %y") for d in w.index]
        fig = go.Figure()
        fig.add_bar(x=lab, y=w.m, name=m["vs"], marker_color=GREY)
        fig.add_bar(x=lab, y=w.f, name="Fund", marker_color=GOLD)
        fig.update_yaxes(tickformat=".1%")
        b.plotly_chart(plot_layout(fig, f"The index's 10 worst days since launch", 360), width="stretch")
    st.caption("NAV-based check: it shows how the fund behaves, not what it holds. Holdings (naked vs hedged "
               "shorts, covered calls, puts) come from the monthly portfolio files.")


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
view = st.segmented_control("View", ["🏁 Leaderboard", "🛡️ Hedge check"], default="🏁 Leaderboard",
                            label_visibility="collapsed") or "🏁 Leaderboard"

funds = pick_variants(meta, plan)
if category:
    funds = funds[funds["category"] == category]
bench = load_benchmark(BENCHMARKS[bench_name], nav["date"].min())
table = build_table(nav, funds, bench)

if table.empty:
    st.info("No funds in this category yet.")
    st.stop()

if view == "🛡️ Hedge check":
    render_hedge_check(nav, funds, category, choice, plan)
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Funds", len(table))
c2.metric("Above ₹10 issue price", f"{(table['Since launch'] > 0).sum()} of {len(table)}")
c3.metric("Median since launch", f"{table['Since launch'].median():+.1%}")
c4.metric("Median beta", f"{table['Beta'].median():.2f}" if table["Beta"].notna().any() else "–")

# ── leaderboard (click a row to open that fund below) ──
st.subheader("Leaderboard")
board = table.sort_values("Since launch", ascending=False).reset_index(drop=True)
show = board.copy()
show.insert(0, "#", range(1, len(show) + 1))
pct = ["Since launch", *LOOKBACKS, "Max DD"]
show[pct] = show[pct] * 100
cols = ["#", "Fund", "AMC", "Launched", *pct, "Beta", "Down cap"]
if not category:
    cols.insert(3, "Category")
    show["Category"] = show["Category"].map(lambda c: SHORT.get(c, c))
event = st.dataframe(
    show[cols], hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
    key=f"board_{choice}_{plan}",
    column_config={
        "#": st.column_config.NumberColumn(width="small"),
        **{c: st.column_config.NumberColumn(format="%+.1f%%") for c in pct},
        "Since launch": st.column_config.NumberColumn(format="%+.1f%%", help="NAV vs ₹10 issue price"),
        "Max DD": st.column_config.NumberColumn(format="%.1f%%"),
        "Beta": st.column_config.NumberColumn(format="%.2f"),
        "Down cap": st.column_config.NumberColumn(
            format="%.2f", help="Below 1 = fell less than the market on its down days"),
    })
st.caption("Click a row to see that fund below. Click a column header to sort.")

r = table.dropna(subset=["Beta"])
if len(r):
    with st.expander("Return vs market exposure"):
        fig = go.Figure(go.Scatter(
            x=r.Beta, y=r["Since launch"], mode="markers+text", text=r.AMC, textposition="top center",
            textfont=dict(color=GREY, size=11), marker=dict(size=12, color=GOLD),
            customdata=r.Fund, hovertemplate="%{customdata}<br>Beta %{x:.2f} · %{y:+.1%}<extra></extra>"))
        fig.update_xaxes(title=f"Beta vs {bench_name} (lower = more hedged)")
        fig.update_yaxes(title="Return since launch", tickformat=".0%")
        st.plotly_chart(plot_layout(fig, "", 400), width="stretch")

# ── fund detail ──
st.divider()
st.subheader("Fund detail")
labels_f = (board["Fund"] + " · " + board["AMC"]).tolist()
pick_key = f"pick_{choice}_{plan}"
rows_sel = event.selection.rows if event and event.selection else []
if rows_sel:
    st.session_state[pick_key] = labels_f[rows_sel[0]]
pick = st.selectbox(f"Fund ({choice})", labels_f, key=pick_key, label_visibility="collapsed")
m = board.iloc[labels_f.index(pick)]
s = nav[nav.code == m.code].set_index("date")["nav"]
f = lambda v, spec: "–" if pd.isna(v) else format(v, spec)

st.caption(f"Launched {m.Launched:%d %b %Y} · rank {labels_f.index(pick) + 1} of {len(board)} "
           f"in {choice if category else 'all SIFs'}")
k = st.columns(5)
k[0].metric("NAV", f"{m.NAV:.4f}")
k[1].metric("Since launch", f(m["Since launch"], "+.1%"))
k[2].metric("Max drawdown", f(m["Max DD"], ".1%"))
k[3].metric("Beta", f(m.Beta, ".2f"))
k[4].metric("Down capture", f(m["Down cap"], ".2f"))

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

p = st.columns(len(LOOKBACKS))
for col, key in zip(p, LOOKBACKS):
    col.metric(f"{key} return", f(m[key], "+.1%"))

with st.sidebar:
    st.download_button("Download table (CSV)", table.drop(columns=["code"]).to_csv(index=False),
                       "sif_funds.csv", width="stretch")
