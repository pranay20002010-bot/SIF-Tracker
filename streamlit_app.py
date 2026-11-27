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
# SEBI category rules (text only). Each fund's own limits come from its ISID (PROMISES below).
MANDATE = {
    "Equity Long-Short Fund": "≥80% equity · unhedged shorts ≤25% of NAV",
    "Equity Ex-Top 100 Long-Short Fund": "≥65% in stocks outside the top 100 · unhedged shorts ≤25%",
    "Sector Rotation Long-Short Fund": "≥80% equity in ≤4 sectors · unhedged shorts ≤25% (sector level)",
    "Hybrid Long-Short Fund": "≥25% equity and ≥25% debt · unhedged shorts ≤25%",
    "Active Asset Allocator Long-Short Fund": "Dynamic across equity, debt, REITs/InvITs, commodity derivatives · unhedged shorts ≤25%",
}
# Yahoo Finance tickers (price indices; TRI not available, so dividends of ~1-1.5% a year are missing)
INDEX_TICKERS = {"Nifty 50": "^NSEI", "Nifty 100": "^CNX100", "Nifty 200": "^CNX200", "Nifty 500": "^CRSLDX",
                 "BSE 500": "BSE-500.BO", "BSE 200": "BSE-200.BO", "Sensex": "^BSESN",
                 "Nifty MidSmallcap 400": "NIFTY_MIDSML_400.NS"}
DEBT_ACCRUAL = 0.07  # stand-in for the debt leg of composite benchmarks (p.a.)


def parse_benchmark(text: str):
    """ISID benchmark text -> (equity index name, equity weight). Debt/commodity legs are modelled as
    a 7% p.a. accrual. Returns the index used for beta/capture and the weight for the composite."""
    t = str(text).lower()
    w = 1.0
    m = re.match(r"\s*(\d+)%", t)
    if m and "+" in t:
        w = int(m.group(1)) / 100
    elif "50:50" in t or "50+50" in t:
        w = 0.5
    elif "85+15" in t:
        w = 0.15
    elif "hybrid" in t:
        w = 0.5
    if "midsmallcap 400" in t or "midsmall" in t:
        idx = "Nifty MidSmallcap 400"
    elif "bse 500" in t:
        idx = "BSE 500"
    elif "bse 200" in t:
        idx = "BSE 200"
    elif "sensex" in t:
        idx = "Sensex"
    elif "500" in t:
        idx = "Nifty 500"
    elif "nifty 200" in t:
        idx = "Nifty 200"
    elif "nifty 100" in t:
        idx = "Nifty 100"
    elif "crisil hybrid" in t or "nifty 50" in t:
        idx = "Nifty 50"
    else:
        idx = "Nifty 500"
    return idx, w


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
        bm, bm_label, _ = own_benchmark(promise_for(m.fund), nav["date"].min(), m.category)
        rows.append({"Fund": m.fund, "AMC": m.amc, "Category": m.category, "code": m.code,
                     "Launched": s.index[0].date(), "NAV": s.iloc[-1],
                     "Since launch": s.iloc[-1] / ISSUE_PRICE - 1,
                     **{k: point_return(s, d) for k, d in LOOKBACKS.items()},
                     "Vol": rs["vol"], "Max DD": rs["mdd"], "Beta": rs["beta"],
                     "Up cap": rs["up"], "Down cap": rs["down"],
                     "vs own benchmark": vs_benchmark(s, bm), "Own benchmark": bm_label})
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


def verdict(h, ceiling):
    """ceiling = most market exposure the fund's ISID allows (max equity, or stated max net)."""
    if h["obs"] < HEDGE_MIN_OBS or pd.isna(h["beta"]):
        return "⏳ Too early to judge"
    if ceiling < 0.99 and h["beta"] > ceiling + 0.10:
        return "⚠️ More market exposure than its ISID allows"
    if h["beta"] >= 0.85 and h["down"] >= 0.90:
        return "🔴 Long-only in disguise"
    if h["down"] < 0.80 and (h["up"] - h["down"]) >= 0.10:
        return "🟢 Hedging works: falls less, keeps upside"
    if h["down"] < 0.80:
        return "🟡 Lower exposure, no hedging edge"
    return "🟡 Light hedging"


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower().replace("long short", "long-short"))


@st.cache_data
def load_promises() -> pd.DataFrame:
    p = pd.read_csv(io.StringIO(ISID_CSV))
    p["key"] = p["fund"].map(_key)
    return p


def promise_for(fund: str):
    p = load_promises()
    hit = p[p["key"] == _key(fund)]
    return None if hit.empty else hit.iloc[0]


def ceiling_of(pr, category):
    """Max net market exposure the ISID allows."""
    if pr is not None:
        m = re.search(r"net\)? equity (\d+)-(\d+)%", str(pr["net_exposure_stated"]), re.I)
        if m:
            return int(m.group(2)) / 100
        if pd.notna(pr["equity_max"]):
            return float(pr["equity_max"]) / 100
    return 0.75 if category == "Hybrid Long-Short Fund" else 1.0


def own_benchmark(pr, start, category=""):
    """Fund's ISID benchmark rebuilt from its equity index + 7% accrual. Returns (series, label, eq index)."""
    text = str(pr["benchmark"]) if pr is not None else ""
    if not text or "not stated" in text.lower():  # fall back to the category's usual benchmark
        text = "NIFTY 50 Hybrid Composite Debt 50:50" if "Hybrid" in category else "Nifty 500"
    idx_name, w = parse_benchmark(text)
    eq = load_benchmark(INDEX_TICKERS[idx_name], start)
    if eq.empty and idx_name != "Nifty 500":
        idx_name = "Nifty 500"
        eq = load_benchmark(INDEX_TICKERS[idx_name], start)
    if eq.empty:
        return eq, "–", idx_name
    r = eq.pct_change().fillna(0) * w + (1 - w) * DEBT_ACCRUAL / 252
    label = idx_name if w == 1 else f"{w:.0%} {idx_name} + {1 - w:.0%} debt"
    return (1 + r).cumprod(), label, idx_name


def vs_benchmark(s, bm):
    b = bm[bm.index >= s.index[0]]
    if len(b) < 2:
        return np.nan
    b = b.reindex(s.index, method="ffill").dropna()
    return (s.iloc[-1] / s.loc[b.index[0]]) - (b.iloc[-1] / b.iloc[0])


@st.cache_data(show_spinner="Running hedge check…")
def build_hedge(nav, funds, start):
    rows, series = [], {}
    for _, m in funds.iterrows():
        s = nav[nav.code == m.code].set_index("date")["nav"]
        pr = promise_for(m.fund)
        bm, bm_label, idx_name = own_benchmark(pr, start, m.category)
        eq = load_benchmark(INDEX_TICKERS[idx_name], start)
        series[idx_name] = eq
        h = hedge_stats(s, eq)
        ceil_ = ceiling_of(pr, m.category)
        eq_rng = (f"{pr['equity_min']:.0f}–{pr['equity_max']:.0f}%" if pr is not None and pd.notna(pr["equity_min"])
                  else (str(pr["net_exposure_stated"])[:40] if pr is not None else "–"))
        rows.append({"Fund": m.fund, "AMC": m.amc, "Category": m.category, "code": m.code,
                     "Verdict": verdict(h, ceil_), "Beta": h["beta"], "Max exposure (ISID)": ceil_,
                     "Equity range (ISID)": eq_rng,
                     "vs own benchmark": vs_benchmark(s, bm), "Own benchmark": bm_label,
                     "Down cap": h["down"], "Up cap": h["up"], "Down-day wins": h["win"],
                     "Worst 10 days: fund": h["worst_f"], "Worst 10 days: index": h["worst_m"],
                     "vs": idx_name, "Days": h["obs"], "has_promise": pr is not None})
    return pd.DataFrame(rows), series


def render_promise(pr, m):
    """'What it promised' card from the ISID."""
    with st.container(border=True):
        if pr is None:
            st.caption("No ISID on file for this fund yet.")
            return
        st.markdown(f"**What it promised** · ISID dated {pr['isid_date']} · [source]({pr['source_url']})")
        a, b, c = st.columns(3)
        rng = lambda lo, hi: "–" if pd.isna(lo) else f"{lo:.0f}–{hi:.0f}%"
        a.markdown(f"**Equity:** {rng(pr['equity_min'], pr['equity_max'])}  \n"
                   f"**Debt:** {rng(pr['debt_min'], pr['debt_max'])}  \n"
                   f"**Unhedged shorts:** up to {pr['unhedged_short_max']:.0f}%")
        b.markdown(f"**Benchmark:** {pr['benchmark']}  \n"
                   f"**Net exposure stated:** {pr['net_exposure_stated']}  \n"
                   f"**Managers:** {pr['fund_managers']}")
        tags = {"covered_calls": "Covered calls", "protective_puts": "Protective puts", "index_hedging": "Index hedging",
                "pair_trades": "Pair trades", "arbitrage": "Arbitrage"}
        named = [v for k, v in tags.items() if str(pr[k]).upper() == "Y"]
        c.markdown("**Named tools:** " + (" · ".join(named) if named else "–") +
                   f"  \n**Option writing:** {pr['options_writing'] if str(pr['options_writing']) not in ('Y', 'N') else ('Yes' if pr['options_writing'] == 'Y' else 'No')}")
        with st.expander("Full list of derivative strategies in the ISID"):
            st.write(pr["derivative_strategies"])
        st.caption(f"Behaviour vs promise: beta {m.Beta:.2f} against an ISID ceiling of "
                   f"{m['Max exposure (ISID)']:.2f}; return {m['vs own benchmark']:+.1%} vs its own benchmark "
                   f"({m['Own benchmark']}). Most ISIDs list strategies as 'illustrative', so whether these tools "
                   "are actually used needs the monthly holdings."
                   if pd.notna(m.Beta) and pd.notna(m["vs own benchmark"]) else
                   "Behaviour vs promise appears once there is enough history.")


def render_hedge_check(nav, funds, category, choice, plan):
    hc, idx = build_hedge(nav, funds, nav["date"].min())
    if hc.empty:
        st.info("No funds in this category yet.")
        return
    f = lambda v, spec: "–" if pd.isna(v) else format(v, spec)

    st.subheader("Is it actually hedging?")
    if category and category in MANDATE:
        st.markdown(f"**SEBI category rule:** {MANDATE[category]}")
    st.caption("Each fund is measured against the equity index in its own ISID benchmark. Hedged positions "
               "count as equity under SIF rules, so a low beta is allowed; the limit that matters is the "
               "ceiling: no more net exposure than the ISID's maximum equity. Down capture below 1 = falls less "
               f"than the market; a real hedge also keeps up capture above it. Verdicts need {HEDGE_MIN_OBS}+ "
               "trading days.")

    counts = hc["Verdict"].str.split(" ").str[0].value_counts()
    k = st.columns(4)
    k[0].metric("🟢 Hedging works", int(counts.get("🟢", 0)))
    k[1].metric("🟡 Partial", int(counts.get("🟡", 0)))
    k[2].metric("🔴 / ⚠️ Not as promised", int(counts.get("🔴", 0) + counts.get("⚠️", 0)))
    k[3].metric("⏳ Too early", int(counts.get("⏳", 0)))

    order = {"🟢": 0, "🟡": 1, "⚠️": 2, "🔴": 3, "⏳": 4}
    hc = hc.assign(_o=hc["Verdict"].str.split(" ").str[0].map(order)).sort_values(["_o", "Down cap"])
    show = hc.copy()
    for c in ["Down-day wins", "Worst 10 days: fund", "Worst 10 days: index", "vs own benchmark"]:
        show[c] = show[c] * 100
    cols = ["Fund", "AMC", "Verdict", "Beta", "Max exposure (ISID)", "Down cap", "Up cap", "Down-day wins",
            "Worst 10 days: fund", "Worst 10 days: index", "vs own benchmark", "vs", "Days"]
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
            "Max exposure (ISID)": st.column_config.NumberColumn(
                format="%.2f", help="Most net market exposure its ISID allows (max equity, or stated max net)"),
            "vs own benchmark": st.column_config.NumberColumn(
                format="%+.1f%%", help="Return since launch minus its ISID benchmark (approximated) over the same period"),
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
    mkt = idx.get(m["vs"], pd.Series(dtype=float, index=pd.DatetimeIndex([])))
    st.markdown(f"**{m.Verdict}**  \n{SHORT.get(m.Category, m.Category)} · SEBI rule: {MANDATE.get(m.Category, '–')}")
    render_promise(promise_for(m.Fund), m)

    k = st.columns(4)
    k[0].metric("Beta", f(m.Beta, ".2f"), f"ISID max {m['Max exposure (ISID)']:.2f}", delta_color="off")
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
        fig.add_hrect(y0=0, y1=m["Max exposure (ISID)"], fillcolor=GOLD, opacity=0.08, line_width=0,
                      annotation_text="allowed by its ISID", annotation_position="top left")
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


# ───────────────────────── ISID promise table ─────────────────────────
# Collected from AMFI-hosted ISIDs (portal.amfiindia.com/spages/S-<n>.pdf), Nov 2026.
# Platinum (S-26*) from Mirae's launch press release because its ISID would not load.
ISID_CSV = """isid,fund,amc_brand,mutual_fund,category,isid_date,benchmark,equity_min,equity_max,debt_min,debt_max,unhedged_short_max,hedged_max,long_deriv_max,reits_invits_max,net_exposure_stated,derivative_strategies,covered_calls,protective_puts,index_hedging,pair_trades,arbitrage,options_writing,fund_managers,source_url
S-1,qsif Equity Long Short Fund,qsif SIF,quant Mutual Fund,Equity Long-Short Fund,2025-08-06,NIFTY 500 TRI,80,100,0,20,25,100,50,20,Not stated,"Short futures; synthetic short; long put; bear put/call spreads; short call; put butterfly/calendar/diagonal; ratio put spread; hedged/unhedged long-only and long-short with up to 25% naked short",N,N,Y,N,N,Y,"Sandeep Tandon; Lokesh Garg; Ankit Pande; Sameer Kate; Sanjeev Sharma",https://portal.amfiindia.com/spages/S-1.pdf
S-2,qsif Hybrid Long-Short Fund,qsif SIF,quant Mutual Fund,Hybrid Long-Short Fund,2025-09-10,NIFTY 50 Hybrid Composite Debt 50:50,25,75,25,75,25,100,50,20,Not stated,"Same tentative short list as qsif Equity L/S; partial/full hedging with index/stock futures or options; long-short with up to 25% naked short",N,N,Y,N,N,Y,"Sandeep Tandon; Lokesh Garg; Ankit Pande; Sameer Kate; Sanjeev Sharma",https://portal.amfiindia.com/spages/S-2.pdf
S-3,Altiva Hybrid Long-Short Fund,Altiva SIF,Edelweiss Mutual Fund,Hybrid Long-Short Fund,2025-09-18,NIFTY 50 Hybrid Composite Debt 50:50 TRI,25,75,25,75,25,100,,20,Not stated (net short monitored),"Cash-futures arbitrage; covered calls; protective puts; index/stock futures-options for hedging and beta adjustment; long/short pair trades; straddles/strangles; directional futures/options; IRS/IRF/FRA",Y,Y,Y,Y,Y,Covered calls only (short straddles conditional),"Bhavesh Jain; Bharat Lahoti; Dhawal Dalal; Pranavi Kulkarni; Amit Vora",https://portal.amfiindia.com/spages/S-3.pdf
S-4,Magnum Hybrid Long Short Fund,Magnum SIF,SBI Mutual Fund,Hybrid Long-Short Fund,2025-09-23,NIFTY 50 Hybrid Composite Debt 50:50 TRI,65,75,25,35,25,75,,10,Not stated,"Covered calls; cash-futures, dividend, merger, buyback and calendar arbitrage; index/stock futures hedging; protective puts; selling calls/call spreads; inverse covered call; short futures; long puts/put spreads",Y,Y,Y,N,Y,Y,Gaurav Mehta,https://portal.amfiindia.com/spages/S-4.pdf
S-5,Diviniti Equity Long Short Fund,Diviniti SIF,ITI Mutual Fund,Equity Long-Short Fund,2025-10-23,Nifty 50 TRI,80,100,0,20,25,100,100,20,Not stated,"Cash-futures arbitrage; stock futures; index/stock futures-options hedging; alpha trades (stock vs sector index futures); covered calls; short index/stock futures; puts and put spreads; short calls; bear call spreads; ratio/calendar/butterfly/condor spreads; OIS/FRA/IRF",Y,Y,Y,Y,Y,Y,Vasav Sahgal; Rajesh Bhatia (overseas),https://portal.amfiindia.com/spages/S-5.pdf
S-6,qsif Equity Ex-Top 100 Long-Short Fund,qsif SIF,quant Mutual Fund,Equity Ex-Top 100 Long-Short Fund,2025-08-14,NIFTY 500 TRI,65,100,0,35,25,100,50,20,Not stated (65-100% outside top 100; 0-35% top 100),"Same tentative short list as qsif Equity L/S; partial/full hedging with index/stock futures or options",N,N,Y,N,N,Y,"Sandeep Tandon; Lokesh Garg; Ankit Pande; Sameer Kate; Sanjeev Sharma",https://portal.amfiindia.com/spages/S-6.pdf
S-7,Titanium Hybrid Long-Short Fund,Titanium SIF,Tata Mutual Fund,Hybrid Long-Short Fund,2026-09-30,CRISIL Hybrid 50+50 Moderate TRI,65,75,25,35,25,75,,10,Not stated (gross and net managed dynamically),"Short futures/options; index-futures hedging; covered calls; protective puts; call and put writing; pair trades; cash-futures arbitrage; collars, spreads, straddles, strangles, butterflies, condors (illustrative)",Y,Y,Y,Y,Y,Y,Suraj Nanda; Akhil Mittal; Hasmukh Vishariya (overseas),https://portal.amfiindia.com/spages/S-7.pdf
S-8,iSIF Equity Ex-Top 100 Long-Short Fund,iSIF SIF,ICICI Prudential Mutual Fund,Equity Ex-Top 100 Long-Short Fund,2026-09-30,Nifty 500 TRI,65,100,0,35,25,100,,20,Not stated (65-100% ex-top-100; 0-35% other equity),"Equity arbitrage; covered calls; index put buying; index futures shorting; protective stock puts/calls; short calls; short puts; long calls/puts/futures; short futures; bull put/bear call spreads; short straddles/strangles",Y,Y,Y,N,Y,Y,Sankaran Naren; Manan Tijoriwala; Divya Jain,https://portal.amfiindia.com/spages/S-8.pdf
S-9,iSIF Hybrid Long-Short Fund,iSIF SIF,ICICI Prudential Mutual Fund,Hybrid Long-Short Fund,2026-09-30,CRISIL Hybrid 50+50 Moderate,65,75,25,35,25,100,,10,Not stated,"Equity arbitrage; covered calls; index futures and index put hedging; protective stock puts/calls; short calls/puts; short straddles/strangles; long calls/puts/futures; short futures; bull put/bear call spreads",Y,Y,Y,N,Y,Y,Rajat Chandak; Ayush Shah; Manish Banthia; Akhil Kakkar,https://portal.amfiindia.com/spages/S-9.pdf
S-10,Arudha Hybrid Long-Short Fund,Arudha SIF,Bandhan Mutual Fund,Hybrid Long-Short Fund,2026-01-02,CRISIL Hybrid 85+15 Conservative TRI,35,65,35,65,25,100,50,20,Not stated,"Covered call; protective put; collar; cash-secured put; long calls/puts; bull/bear spreads; calendar; straddle; strangle; butterfly; iron condor/butterfly; synthetic long/short; pair trades; stock-level hedging; IRS/FRA/IRF/CDS",Y,Y,N,Y,N,Y,Kapil Kankonkar; Brijesh Shah; Debraj Lahiri; Nilesh Saha,https://portal.amfiindia.com/spages/S-10.pdf
S-11,DynaSIF Equity Long-Short Fund,DynaSIF SIF,360 ONE Mutual Fund,Equity Long-Short Fund,2026-01-21,BSE 500 TRI,80,100,0,20,25,,25,20,Not stated (sector long-short net ~20% average),"Long-short equity; sector long-short; pairs trading; merger arbitrage; index rebalancing trades; directional index derivatives; index pair trading; covered calls; long/short straddles-strangles; stock options/futures arbitrage; index-derivative hedging",Y,N,Y,Y,Y,Y,Harsh Aggarwal,https://portal.amfiindia.com/spages/S-11.pdf
S-12,Arudha Equity Long-Short Fund,Arudha SIF,Bandhan Mutual Fund,Equity Long-Short Fund,2026-02-06,NIFTY 500 TRI,80,100,0,20,25,100,50,20,Not stated,"Covered call; protective put; collar; cash-secured put; long calls/puts; bull/bear spreads; calendar; straddle; strangle; butterfly; iron condor/butterfly; synthetic long/short; pair trades; stock-level hedging",Y,Y,N,Y,N,Y,Nilesh Saha; Brijesh Shah,https://portal.amfiindia.com/spages/S-12.pdf
S-13,Apex Hybrid Long-Short Fund,Apex SIF,Aditya Birla Sun Life Mutual Fund,Hybrid Long-Short Fund,2026-02-27,NIFTY 50 Hybrid Composite Debt 50:50 TRI,35,65,35,65,25,,,20,Not stated,"Cash-futures arbitrage; covered calls; protective puts; long/short pair trades; index/stock futures-options for hedging or beta adjustment; long/short straddles-strangles; directional futures/options; put-call parity trades; IRS/IRF/FRA",Y,Y,Y,Y,Y,Y,Lovelish Solanki; Mohit Sharma,https://portal.amfiindia.com/spages/S-13.pdf
S-14,DynaSIF Active Asset Allocator Long-Short Fund,DynaSIF SIF,360 ONE Mutual Fund,Active Asset Allocator Long-Short Fund,2026-03-27,25% BSE Sensex TRI + 60% CRISIL Short Term Bond + 15% iCOMDEX,20,50,20,65,25,100,,20,Not stated (commodity derivatives 0-25%),"Stock and sector long-short; pairs trading; merger arbitrage; index rebalancing; directional index trades; index pair trading; covered calls; straddles/strangles; stock options/futures arbitrage; index-derivative hedging; commodity options",Y,N,Y,Y,Y,Y,Harsh Agarwal; Milan Mody (debt); Rahul Khetawat (commodity),https://portal.amfiindia.com/spages/S-14.pdf
S-15,qsif Active Asset Allocator Long-Short Fund,qsif SIF,quant Mutual Fund,Active Asset Allocator Long-Short Fund,2026-03-27,40% NSE 500 TRI + 30% CRISIL Short Term Bond + 30% iCOMDEX,0,100,0,100,25,100,50,20,Not stated (commodity derivatives 0-30%),"Same tentative short list as qsif Equity L/S; derivatives for hedging and rebalancing",N,N,N,N,N,Y,Sandeep Tandon; Jignesh Shah; Ankit Pande; Sameer Kate; Sanjeev Sharma,https://portal.amfiindia.com/spages/S-15.pdf
S-16,Sapphire Equity Long-Short SIF,Sapphire SIF,Franklin Templeton Mutual Fund,Equity Long-Short Fund,2026-03-26,Nifty 500 TRI,80,100,0,20,25,50,,,"Not stated (approach section: gross long 75-100%, short 0-25%; conflicts with 80% floor)","Unhedged shorts via listed futures/options as stock-level bearish picks; covered calls; imperfect hedging",Y,N,N,N,N,Covered calls only,Arihant Jain,https://portal.amfiindia.com/spages/S-16.pdf
S-17,Titanium Equity Long-Short Fund,Titanium SIF,Tata Mutual Fund,Equity Long-Short Fund,2026-04-09,Nifty 500 TRI,80,100,0,20,25,100,,20,Not stated (gross and net managed dynamically),"Long-short via stock and index futures-options; index futures-options hedging; pair trades; cash-futures arbitrage; sector vs index trades; covered calls; call and put writing; protective puts (illustrative)",Y,Y,Y,Y,Y,Y,Suraj Nanda; Amit Somani; Hasmukh Vishariya (overseas),https://portal.amfiindia.com/spages/S-17.pdf
S-18,WSIF Equity Ex-Top 100 Long-Short Fund,WSIF,The Wealth Company Mutual Fund,Equity Ex-Top 100 Long-Short Fund,2026-03-25,NIFTY 500 TRI,65,100,0,35,25,100,,20,Not stated (65-100% ex-top-100; 0-35% top 100),"Tentative short list (short futures; synthetic short; long puts; put/call spreads; short call; ratio put spread); hedged/unhedged long-only and long-short; covered calls; index/stock futures hedging",Y,Y,Y,N,N,Y,Chinmay Sathe,https://portal.amfiindia.com/spages/S-18.pdf
S-19,WSIF Equity Long-Short Fund,WSIF,The Wealth Company Mutual Fund,Equity Long-Short Fund,2026-03-25,NIFTY 500 TRI,80,100,0,20,25,100,,20,Not stated,"Tentative short list (short futures; synthetic short; long puts; put/call spreads; short call; ratio put spread); covered calls; index/stock futures-options hedging up to 100%",Y,Y,Y,N,N,Y,Chinmay Sathe,https://portal.amfiindia.com/spages/S-19.pdf
S-20,Arthaya Equity Long Short Fund,Arthaya SIF,Union Mutual Fund,Equity Long-Short Fund,2026-04-17,NIFTY 200 TRI,80,100,0,20,25,100,100,20,Not stated,"Hedged/unhedged long-only and long-short with up to 25% unhedged short; short futures; synthetic short; long puts; put/call spreads; short call; ratio put spread; covered calls; index futures hedging; unhedged option writing (risk factors)",Y,Y,Y,N,N,Y,Rajesh Aynor; Hiten Bhadra (arbitrage),https://portal.amfiindia.com/spages/S-20.pdf
S-21,qsif Sector Rotation Long-Short Fund,qsif SIF,quant Mutual Fund,Sector Rotation Long-Short Fund,2026-04-22,NIFTY 500 TRI,80,100,0,20,25,100,50,20,Not stated (80-100% in max 4 sectors),"Same tentative short list as qsif Equity L/S; partial/full hedging with index/stock futures or options",N,N,Y,N,N,Y,Sandeep Tandon; Jignesh Shah; Ankit Pande; Sameer Kate; Sanjeev Sharma,https://portal.amfiindia.com/spages/S-21.pdf
S-22,Altiva Equity Ex-Top 100 Long-Short Fund,Altiva SIF,Edelweiss Mutual Fund,Equity Ex-Top 100 Long-Short Fund,2026-04-20,Nifty 500 TRI,65,100,0,35,25,100,,20,Not stated (65-100% ex-top-100; 0-35% other equity),"Equity arbitrage; covered calls; pair trades; index put buying; index futures shorting; protective stock puts/calls; short calls/puts; short straddles/strangles; long/short futures and options; bear put/call spreads",Y,Y,Y,Y,Y,Y,Nikhil Gada; Trideep Bhattacharya; Amit Vora (overseas),https://portal.amfiindia.com/spages/S-22.pdf
S-23,iSIF Active Asset Allocator Long-Short Fund,iSIF SIF,ICICI Prudential Mutual Fund,Active Asset Allocator Long-Short Fund,2026-11-13,50% Nifty 500 TRI + 40% Nifty Composite Debt + 7% gold + 3% silver,35,80,10,55,25,100,,20,Not stated (commodity derivatives allowed),"Equity arbitrage; covered calls; index put and index futures hedging; protective stock puts/calls; short calls/puts; short straddles/strangles; long/short futures and options; bear put/call spreads; commodity arbitrage",Y,Y,Y,N,Y,Y,Sharmila D'silva; Masoomi Jhurmarvala; Manish Banthia; Akhil Kakkar; Gaurav Chikane (commodity),https://portal.amfiindia.com/spages/S-23.pdf
S-24,iSIF Equity Long-Short Fund,iSIF SIF,ICICI Prudential Mutual Fund,Equity Long-Short Fund,2026-05-07,Nifty 500 TRI,80,100,0,20,25,,,20,Not stated,"Equity arbitrage; covered calls; protective stock puts/calls; index futures and index put hedging; short calls/puts; short straddles/strangles; long/short futures and options; bull put/bear call spreads",Y,Y,Y,N,Y,Y,Mittul Kalawadia; Nitya Mishra; Sri Sharma,https://portal.amfiindia.com/spages/S-24.pdf
S-25,RedHex Hybrid Long-Short Fund,RedHex SIF,HSBC Mutual Fund,Hybrid Long-Short Fund,2026-05-07,NIFTY 50 Hybrid Composite Debt 50:50 TRI,25,75,25,75,25,75,,20,Not stated,"Cash-futures and reverse arbitrage; covered calls; protective puts; index futures hedging; pair trades (same-industry); short straddles/strangles; butterflies; directional futures/options; IRS/IRF/FRA; CDS",Y,Y,Y,Y,Y,Y,Shriram Ramanathan; Praveen Ayathan (arbitrage); Venugopal Manghat; Mayank Chaturvedi (overseas),https://portal.amfiindia.com/spages/S-25.pdf
S-27,Prism Hybrid Long-Short Fund,Prism SIF,JioBlackRock Mutual Fund,Hybrid Long-Short Fund,2026-05-15,NIFTY 50 Hybrid Composite Debt 50:50 TRI,35,75,25,65,25,100,100,20,Not stated,"Merger arbitrage; collars; cash-futures arbitrage; IPO and special-situation trades (open offers, buybacks); indicative list of 29 incl. covered calls, protective/cash-secured/long puts, index put and index futures hedging, pair trades, short calls/puts, short straddles/strangles, market neutral, short futures",Y,Y,Y,Y,Y,Y,Arun Ramachandran; Tanvi Kacheria; Siddharth Deb; Virendra Kumar,https://portal.amfiindia.com/spages/S-27.pdf
S-28,DynaSIF Equity Ex-Top 100 Long-Short Fund,DynaSIF SIF,360 ONE Mutual Fund,Equity Ex-Top 100 Long-Short Fund,2026-05-22,BSE 500 TRI,65,100,0,25,25,100,,20,"Not stated (65-100% ex-top-100; 0-35% top 100; internal: net sector ~20%, index trades ~5%)","Stock and sector long-short; pairs trading; merger arbitrage; index rebalancing; directional index trades; index pair trading; covered calls; straddles/strangles; stock options/futures arbitrage; index-derivative hedging",Y,N,Y,Y,Y,Y,Harsh Agarwal; Mayur Patel; Milan Mody; Pranav Mise,https://portal.amfiindia.com/spages/S-28.pdf
S-29,INFINITY Hybrid Long-Short Fund,Infinity SIF,Kotak Mahindra Mutual Fund,Hybrid Long-Short Fund,2026-05-26,NIFTY 50 Hybrid Composite Debt 50:50,35,75,25,65,25,100,50,20,Not stated,"Covered call; protective put; collar; cash-secured put; long calls/puts; bull/bear spreads; calendar; straddle; strangle; butterfly; iron condor/butterfly; synthetic long/short; pair trades; stock-level hedging; index futures hedging; IRS/FRA/IRF/CDS",Y,Y,Y,Y,N,Y,Hiten Shah; Kalpesh Jain; Abhishek Bisen,https://portal.amfiindia.com/spages/S-29.pdf
S-30,Summit Equity Long-Short Fund,Summit SIF,Invesco Mutual Fund,Equity Long-Short Fund,2026-06-15,BSE 500 TRI,80,100,0,20,25,100,25,20,Not stated (tactical shorts described as 20-25% of assets),"Tentative short list (short futures; synthetic short; long puts; put/call spreads; short call; ratio put spread); index and sector index futures shorts; covered calls (risk factors)",Y,Y,N,N,N,Y,Hiten Jain,https://portal.amfiindia.com/spages/S-30.pdf
S-31,Apex Equity Long-Short Fund,Apex SIF,Aditya Birla Sun Life Mutual Fund,Equity Long-Short Fund,2026-07-22,NIFTY 500 TRI,80,100,0,20,25,,,20,Not stated (actively managed),"Covered calls; protective puts; index/stock futures-options hedging; long/short pair trades; cash-futures and corporate-action arbitrage; short calls; bear call/put spreads; collar; straddle; strangle; synthetic short; iron butterfly; put butterfly/calendar/diagonal; ratio put spread",Y,Y,Y,Y,Y,Y,Manish Gupta; Harshil Suvarnkar,https://portal.amfiindia.com/spages/S-31.pdf
S-32,Apex Equity Ex-Top 100 Long-Short Fund,Apex SIF,Aditya Birla Sun Life Mutual Fund,Equity Ex-Top 100 Long-Short Fund,2026-07-22,NIFTY 500 TRI,65,100,0,35,25,,,20,Not stated (65-100% ex-top-100; 0-35% other equity),"Covered calls (against holdings only); protective puts; cash-futures arbitrage; long/short pair trades; index/stock futures-options for hedging or beta adjustment; long/short straddles-strangles; directional futures/options; synthetic options and put-call parity trades",Y,Y,Y,Y,Y,Y,Manish Gupta; Harshil Suvarnkar,https://portal.amfiindia.com/spages/S-32.pdf
S-33,Magnum Equity Ex-Top 100 Long Short Fund,Magnum SIF,SBI Mutual Fund,Equity Ex-Top 100 Long-Short Fund,2026-07-31,BSE 500 TRI,65,100,0,35,25,100,,20,Not stated (65-100% ex-top-100; 0-35% top 100),"Short futures; long puts; put spreads; selling calls/call spreads; inverse covered call (short puts with short futures); covered calls; index and stock futures-options hedging; cash-futures arbitrage; interest rate futures for imperfect hedging (up to 20%)",Y,Y,Y,N,Y,Y,Gaurav Mehta,https://portal.amfiindia.com/spages/S-33.pdf
S-34,Altiva Equity Long-Short Fund,Altiva SIF,Edelweiss Mutual Fund,Equity Long-Short Fund,2026-08-20,Nifty 100 TRI,80,100,0,20,25,100,,20,Not stated,"Equity arbitrage; covered calls; index put and index futures hedging; protective stock puts/calls; short calls/puts; short straddles/strangles; long/short futures and options; bull/bear spreads; pair trading (risk factors)",Y,Y,Y,Y,Y,Y,Bharat Lahoti; Bhavesh Jain; Amit Vora (overseas),https://portal.amfiindia.com/spages/S-34.pdf
S-35,Titanium Active Asset Allocator Long-Short Fund,Titanium SIF,Tata Mutual Fund,Active Asset Allocator Long-Short Fund,2026-09-08,35% BSE 200 TRI + 50% CRISIL Short Term Bond + 15% iCOMDEX,35,100,0,65,25,100,,20,Net equity may go as low as 0% (commodity derivatives 0-30%),"Cash-futures arbitrage; pair trades; sector vs index trades; hedging long stocks with short futures; covered calls; call and put writing; protective puts, collars, spreads, straddles, condors (illustrative); IRS/FRA/IRF",Y,Y,Y,Y,Y,Y,Sailesh Jain; Akhil Mittal; Tapan Patel (commodity); Hasmukh Vishariya,https://portal.amfiindia.com/spages/S-35.pdf
S-36,MSIF Equity Long-Short Fund,MSIF,Mahindra Manulife Mutual Fund,Equity Long-Short Fund,2026-09-10,Nifty 500 TRI,80,100,0,20,25,100,50,10,Not stated,"Short exposure via stock, index and sector index futures and options; hedging/offsetting positions; long/short calls and puts; covered calls; options writing (risk factors)",Y,Y,Y,N,N,Y,Aalap Shah; Abhishek Jaiswal,https://portal.amfiindia.com/spages/S-36.pdf
S-37,Summit Active Asset Allocator Long-Short Fund,Summit SIF,Invesco Mutual Fund,Active Asset Allocator Long-Short Fund,2026-09-28,40% Nifty 500 TRI + 40% CRISIL Short Term Bond + 20% iCOMDEX,65,90,10,35,25,100,,20,Not stated (commodity derivatives 0-25%),"Short futures; long puts; put spreads; selling calls/call spreads; inverse covered call; covered calls; protective puts; index futures hedging; arbitrage",Y,Y,Y,N,Y,Y,Deepak Gupta,https://portal.amfiindia.com/spages/S-37.pdf
S-38,Eminence Hybrid Long-Short Fund,Eminence SIF,AlphaGrep Mutual Fund,Hybrid Long-Short Fund,2026-09-09,NIFTY 50 Hybrid Composite Debt 50:50 TRI,25,75,25,75,25,100,,20,Not stated,"Cash-futures arbitrage; covered calls; protective puts; collars; index/stock futures-options for hedging or beta adjustment; pair trades; merger/share-swap and dividend arbitrage; straddles/strangles; IRS/IRF/FRA",Y,Y,Y,Y,Y,Y,Dwaipayan Ray; Ravneet Singh; Ashish Jain; Ramesh S,https://portal.amfiindia.com/spages/S-38.pdf
S-40,SELECT Equity Ex-Top 100 Long-Short Fund,Select SIF,Sundaram Mutual Fund,Equity Ex-Top 100 Long-Short Fund,2026-09-28,Nifty MidSmallcap 400 TRI,65,100,0,35,25,100,,10,Not stated (65-100% ex-top-100; 0-35% other equity),"Covered calls; protective stock puts/calls; index put and index futures hedging; pair trades; equity arbitrage; short calls/puts; bear call/put spreads; short straddles/strangles; long/short futures and options",Y,Y,Y,Y,Y,Y,Madanagopal Ramu; Sandeep Agarwal; Shalav Saket (overseas),https://portal.amfiindia.com/spages/S-40.pdf
S-41,H-SIF Hybrid Long-Short Fund,H-SIF,HDFC Mutual Fund,Hybrid Long-Short Fund,2026-10-28,NIFTY 50 Hybrid Composite Debt 50:50,25,75,25,75,25,100,,20,Not stated,"Equity arbitrage; covered calls; protective puts; long puts; index put and index futures hedging; short calls/puts; bull put/bear call spreads; short straddles/strangles; protective calls; premium harvesting",Y,Y,Y,N,Y,Y,Anil Bamboli; Ihab Dalwai; Paul S Parampreet; Gopal Agrawal (overseas),https://portal.amfiindia.com/spages/S-41.pdf
S-26*,Platinum Hybrid Long-Short Fund,Platinum SIF,Mirae Asset Mutual Fund,Hybrid Long-Short Fund,2026-05-13,Not stated in source,,,25,35,25,,,,"Controlled (net) equity 5-70% incl. collars and special situations; fully hedged arbitrage 5-70%; accrual debt 25-35%","Collars; special situations; arbitrage; accrual debt",N,N,N,N,Y,Collars,Gaurik Shah (quoted; not confirmed as manager),https://www.miraeassetmf.co.in/downloads/press-release/platinum-sif-brought-to-you-by-mirae-asset-mutual-fund-launches-platinum-hybrid-long-short-fund
S-42,Altiva Active Asset Allocator Long-Short Fund,Altiva SIF,Edelweiss Mutual Fund,Active Asset Allocator Long-Short Fund,2026-11-05,50% Nifty 50 TRI + 35% Nifty Short Duration Debt + 7.5% gold + 7.5% silver,65,100,0,35,25,,100,20,Not stated (commodity derivatives 0-30%),"Equity arbitrage; covered calls; index put and index futures hedging; protective stock puts/calls; short calls/puts; short straddles/strangles; long/short futures and options; bear put/call spreads; put-call parity trades; pair trades (risk factors)",Y,Y,Y,Y,Y,Y,Bharat Lahoti; Bhavesh Jain; Hetul Raval (debt); Amit Vora (commodity/overseas),https://portal.amfiindia.com/spages/S-42.pdf
S-43,Infinity Equity Long-Short Fund,Infinity SIF,Kotak Mahindra Mutual Fund,Equity Long-Short Fund,2026-10-27,NIFTY 100 TRI,80,100,0,20,25,100,50,20,Not stated,"Covered calls; protective puts; collars; cash-secured puts; pair trades; stock-level hedging; index futures hedging; short straddles/strangles",Y,Y,Y,Y,N,Y,Kalpesh Jain; Rohit Tandon; Abhishek Bisen,https://portal.amfiindia.com/spages/S-43.pdf
S-44,MSIF Hybrid Long-Short Fund,MSIF,Mahindra Manulife Mutual Fund,Hybrid Long-Short Fund,2026-11-09,CRISIL Hybrid 50+50 Moderate,65,75,25,35,25,100,50,10,Not stated,"Covered calls; cash-futures, dividend, merger, buyback and calendar arbitrage; short calls and long puts against longs; short puts and long calls against shorts; index futures-options hedging; interest rate futures",Y,Y,Y,N,Y,Y,Aalap Shah; Abhishek Jaiswal; Amit Garg; Kush Sonigara,https://portal.amfiindia.com/spages/S-44.pdf
"""

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
pct = ["Since launch", "vs own benchmark", *LOOKBACKS, "Max DD"]
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
        "vs own benchmark": st.column_config.NumberColumn(
            "vs ISID benchmark", format="%+.1f%%",
            help="Return since launch minus the fund's own ISID benchmark over the same period (approximated)"),
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
