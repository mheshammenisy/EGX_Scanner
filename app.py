import os
import re
import json
import time
import contextlib
import io

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import matplotlib.pyplot as plt

from egx_watchlist import EGX_SAFE_INTRADAY, display_name
from egx_scan_once import ensure_ohlcv, filter_last_minutes  # must exist in your scanner file
from openai import OpenAI

APP_PASSWORD = os.getenv("APP_PASSWORD") or st.secrets.get("APP_PASSWORD", "")

if APP_PASSWORD:
    if "auth_ok" not in st.session_state:
        st.session_state["auth_ok"] = False

    if not st.session_state["auth_ok"]:
        st.title("🔒 Private App")
        pw = st.text_input("Password", type="password")

        if st.button("Login"):
            if pw == APP_PASSWORD:
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.error("Wrong password.")
        st.stop()
# -----------------------------
# Fixed defaults (no user inputs)
# -----------------------------
INTERVAL = "5m"
PERIOD = "5d"
VOL_LOOKBACK = 20
TOP_N = 10

# Freshness window (for mode detection only).
# With Yahoo delays + market open, 60–90 is safer than 30.
LAST_MINUTES = 90

# Buckets
STRONG_HIT = 3.0      # >= 3
WATCHLIST_LOW = 2.0   # 2 to 3

# -----------------------------
# Simple password gate (Streamlit secrets / env)
# -----------------------------


st.set_page_config(page_title="EGX Intraday Scanner", layout="wide")
st.title("EGX Intraday Scanner (Always Shows Top 10)")

st.caption(
    f"Defaults: interval={INTERVAL}, period={PERIOD}, freshness_window={LAST_MINUTES}m, lookback={VOL_LOOKBACK} bars. "
    f"Buckets: ✅≥{STRONG_HIT}× | ⚠️ {WATCHLIST_LOW}–{STRONG_HIT}× | 👀 <{WATCHLIST_LOW}×. "
    "Nothing runs automatically — click Run Scan."
)


# -----------------------------
# Yahoo fetch (cached + quiet)
# -----------------------------
@st.cache_data(ttl=60)
def fetch_intraday(ticker: str) -> pd.DataFrame:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            df = yf.download(
                tickers=ticker,
                interval=INTERVAL,
                period=PERIOD,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def bucket_label(r: float) -> str:
    if pd.isna(r):
        return "—"
    if r >= STRONG_HIT:
        return "✅ Strong"
    if r >= WATCHLIST_LOW:
        return "⚠️ Watchlist"
    return "👀 Warm (no spike)"


def bucket_rank(r: float) -> int:
    if pd.isna(r):
        return 99
    if r >= STRONG_HIT:
        return 0
    if r >= WATCHLIST_LOW:
        return 1
    return 2


def compute_turnover_egp(df: pd.DataFrame, bars: int = 78) -> float:
    df = ensure_ohlcv(df)
    if df.empty:
        return 0.0
    tail = df.tail(bars)
    return float((tail["Close"] * tail["Volume"]).sum())


def compute_spike_ratio(df: pd.DataFrame) -> float:
    """
    spike_ratio = last completed bar volume / SMA(volume, lookback)
    Use FULL df so SMA has enough bars (VOL_LOOKBACK * 5 minutes).
    """
    df = ensure_ohlcv(df)
    if df.empty or len(df) < VOL_LOOKBACK + 2:
        return float("nan")

    work = df.copy()
    work["vol_sma"] = work["Volume"].rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).mean()

    idx = -2 if len(work) >= 3 else -1  # last completed bar when possible
    v = work["Volume"].iloc[idx]
    sma = work["vol_sma"].iloc[idx]

    if pd.isna(sma) or sma <= 0:
        return float("nan")
    return float(v / sma)


def build_levels_from_last_bar(df: pd.DataFrame) -> dict:
    """
    entry = last completed bar high (small buffer)
    stop  = last completed bar low
    targets = 1R/2R/3R
    """
    df = ensure_ohlcv(df)
    if df.empty:
        return {}

    idx = -2 if len(df) >= 3 else -1
    bar = df.iloc[idx]

    entry = float(bar["High"]) * 1.0005
    stop = float(bar["Low"])
    if stop >= entry:
        return {}

    R = entry - stop
    return {"entry": entry, "stop": stop, "t1": entry + R, "t2": entry + 2 * R, "t3": entry + 3 * R}


def plot_levels_ladder(row: pd.Series):
    entry, stop, t1, t2, t3 = row["entry"], row["stop"], row["t1"], row["t2"], row["t3"]
    last_price = row["last_price"]
    name = row["name"]
    symbol = row["symbol"]
    bucket = row["bucket"]
    ratio = row["spike_ratio"]

    vals = [stop, entry, t1, t2, t3]
    labels = ["STOP", "ENTRY", "T1", "T2", "T3"]

    fig, ax = plt.subplots(figsize=(9, 2.1))
    title_ratio = "n/a" if pd.isna(ratio) else f"{ratio:.2f}×"
    ax.set_title(f"{bucket} | {name} — {symbol} | spike {title_ratio}", fontsize=11)

    ax.scatter(vals, [0] * len(vals), s=70)
    for v, lab in zip(vals, labels):
        ax.annotate(f"{lab}\n{v:.2f}", (v, 0), textcoords="offset points", xytext=(0, 12), ha="center")

    ax.scatter([last_price], [0], s=90, marker="x")
    ax.annotate(f"LAST\n{last_price:.2f}", (last_price, 0), textcoords="offset points", xytext=(0, -26), ha="center")

    ax.set_yticks([])
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.set_xlabel("Price")
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# Run scan
# -----------------------------
run_btn = st.button("Run scan (EGX30 watchlist)")

if "ranked" not in st.session_state:
    st.session_state["ranked"] = pd.DataFrame()
if "mode" not in st.session_state:
    st.session_state["mode"] = "—"

if run_btn:
    t0 = time.time()
    rows = []
    ok_data = 0
    recent_ok = 0
    any_recent = False

    for t in EGX_SAFE_INTRADAY:
        raw = fetch_intraday(t)
        df_full = ensure_ohlcv(raw)
        if df_full.empty:
            continue

        ok_data += 1

        df_recent = filter_last_minutes(df_full, minutes=LAST_MINUTES)
        is_recent = not df_recent.empty
        if is_recent:
            any_recent = True
            recent_ok += 1

        ratio = compute_spike_ratio(df_full)
        turnover = compute_turnover_egp(df_full)
        last_price = float(df_full["Close"].iloc[-1])

        levels = build_levels_from_last_bar(df_full)
        entry = levels.get("entry", float("nan"))
        stop = levels.get("stop", float("nan"))
        t1 = levels.get("t1", float("nan"))
        t2 = levels.get("t2", float("nan"))
        t3 = levels.get("t3", float("nan"))

        rows.append({
            "bucket": bucket_label(ratio),
            "bucket_rank": bucket_rank(ratio),
            "name": display_name(t),
            "symbol": t,
            "spike_ratio": ratio,
            "turnover_egp": turnover,
            "last_price": last_price,
            "entry": entry,
            "stop": stop,
            "t1": t1,
            "t2": t2,
            "t3": t3,
            "live": bool(is_recent),
            "last_bar_time": str(df_full.index[-1]),
        })

    elapsed = time.time() - t0
    ranked = pd.DataFrame(rows)

    if not ranked.empty:
        ranked = ranked.sort_values(
            ["bucket_rank", "spike_ratio", "turnover_egp"],
            ascending=[True, False, False],
        ).reset_index(drop=True)

    st.session_state["ranked"] = ranked
    st.session_state["mode"] = "LIVE" if any_recent else "CLOSED (STALE)"
    st.session_state["meta"] = {
        "elapsed": elapsed,
        "watchlist": len(EGX_SAFE_INTRADAY),
        "data_ok": ok_data,
        "recent_ok": recent_ok,
    }

ranked = st.session_state["ranked"]
meta = st.session_state.get("meta", {"elapsed": None, "watchlist": len(EGX_SAFE_INTRADAY), "data_ok": 0, "recent_ok": 0})
mode = st.session_state.get("mode", "—")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Watchlist", meta["watchlist"])
m2.metric("Tickers with any data", meta["data_ok"])
m3.metric(f"Tickers with bars in last {LAST_MINUTES}m", meta["recent_ok"])
m4.metric("Scan time (sec)", "-" if meta["elapsed"] is None else f'{meta["elapsed"]:.2f}')

st.info(f"Mode: {mode}")

st.divider()

if ranked.empty and meta["data_ok"] == 0:
    st.warning("No data returned from Yahoo for any ticker.")
    st.stop()

if ranked.empty:
    st.warning("No rows built (unexpected).")
    st.stop()


# -----------------------------
# Always show Top N (bucket-ordered)
# -----------------------------
top = ranked.head(TOP_N).copy()

top["stop_%"] = (top["stop"] / top["entry"] - 1.0) * 100.0
top["t1_%"] = (top["t1"] / top["entry"] - 1.0) * 100.0
top["t2_%"] = (top["t2"] / top["entry"] - 1.0) * 100.0
top["t3_%"] = (top["t3"] / top["entry"] - 1.0) * 100.0

st.subheader(f"Top {TOP_N} (bucket order: ≥3 then 2–3 then <2)")

st.dataframe(
    top[
        ["bucket", "live", "last_bar_time", "name", "symbol",
         "spike_ratio", "turnover_egp", "last_price",
         "entry", "stop", "t1", "t2", "t3",
         "stop_%", "t1_%", "t2_%", "t3_%"]
    ],
    use_container_width=True,
    hide_index=True,
)

# Buckets
st.divider()
st.subheader("Buckets (full lists)")

strong = ranked[ranked["bucket_rank"] == 0].copy()
watch = ranked[ranked["bucket_rank"] == 1].copy()
warm = ranked[ranked["bucket_rank"] == 2].copy()

c1, c2, c3 = st.columns(3)
c1.metric("✅ Strong (≥3)", len(strong))
c2.metric("⚠️ Watchlist (2–3)", len(watch))
c3.metric("👀 Warm (<2)", len(warm))

with st.expander("✅ Strong (≥3)"):
    st.dataframe(strong.drop(columns=["bucket_rank"]), use_container_width=True, hide_index=True) if not strong.empty else st.info("No strong spikes right now.")
with st.expander("⚠️ Watchlist (2–3)"):
    st.dataframe(watch.drop(columns=["bucket_rank"]), use_container_width=True, hide_index=True) if not watch.empty else st.info("No medium spikes right now.")
with st.expander("👀 Warm (no spike)"):
    st.dataframe(warm.drop(columns=["bucket_rank"]), use_container_width=True, hide_index=True) if not warm.empty else st.info("No warm names right now (rare).")


# -----------------------------
# AI Recommendation (Top N) + Visuals
# -----------------------------
st.divider()
st.subheader("AI Recommendation (Top 10)")
st.caption("One click only. Output is ACTION + reasons + levels + charts for picks. No JSON shown to you.")

ai_btn = st.button("Ask AI to recommend (Top 10)")

if ai_btn:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY not found. Put it in .streamlit/secrets.toml (not GitHub).")
        st.stop()

    client = OpenAI(api_key=api_key)

    payload = []
    for _, r in top.iterrows():
        payload.append({
            "bucket": r["bucket"],
            "live": bool(r["live"]),
            "last_bar_time": r["last_bar_time"],
            "name": r["name"],
            "symbol": r["symbol"],
            "spike_ratio": None if pd.isna(r["spike_ratio"]) else float(r["spike_ratio"]),
            "turnover_egp": float(r["turnover_egp"]),
            "last_price": float(r["last_price"]),
            "entry": None if pd.isna(r["entry"]) else float(r["entry"]),
            "stop": None if pd.isna(r["stop"]) else float(r["stop"]),
            "t1": None if pd.isna(r["t1"]) else float(r["t1"]),
            "t2": None if pd.isna(r["t2"]) else float(r["t2"]),
            "t3": None if pd.isna(r["t3"]) else float(r["t3"]),
            "stop_pct": None if pd.isna(r["stop_%"]) else float(r["stop_%"]),
            "t1_pct": None if pd.isna(r["t1_%"]) else float(r["t1_%"]),
            "t2_pct": None if pd.isna(r["t2_%"]) else float(r["t2_%"]),
            "t3_pct": None if pd.isna(r["t3_%"]) else float(r["t3_%"]),
        })
    allowed = ", ".join(top["name"].astype(str).str.upper().tolist())

    instruction = f"""
You are an EGX intraday trading assistant.
You are given Top {TOP_N} ranked candidates from EGX30.

Ranking is bucket-based:
1) ✅ Strong (spike_ratio >= {STRONG_HIT})
2) ⚠️ Watchlist ({WATCHLIST_LOW} to {STRONG_HIT})
3) 👀 Warm (below {WATCHLIST_LOW})

Context:
- Data can be delayed. Each row includes 'live' and 'last_bar_time'.
- If live=False, say it's STALE and be conservative.

Scoring guidance:
- Prefer higher spike_ratio AND higher turnover_egp.
- Penalize chasing: if last_price > entry by more than 0.30%, recommend WAIT or SKIP.
- If spike_ratio is missing, explicitly state SPIKE UNKNOWN.

Return ONLY:
PICK: exactly 2 items, ONLY from this allowed list:
{allowed}
Format MUST be exactly:
PICK: CODE1, CODE2

Do NOT use .CA tickers.
Do NOT use company names.
Use ONLY the short codes above.
For each pick:

- ACTION: BUY NOW / WAIT / SKIP
- WHY: 3-5 bullets (must reference spike_ratio or say SPIKE UNKNOWN)
- LEVELS: Entry, Stop, T1, T2, T3 with % to each
- NOTE: one caution line
Then list remaining tickers with one-line reason each.
No JSON, no code.
"""

    resp = client.responses.create(
        model="gpt-5.2",
        input=instruction + "\n\nDATA:\n" + json.dumps(payload, ensure_ascii=False, indent=2),
    )

    st.success(resp.output_text)

    st.markdown("### Visual levels for AI picks")

    m = re.search(r"(?i)PICK:\s*([A-Za-z0-9_,\s\.\-]+)", resp.output_text)
    if not m:
        st.info("Could not parse PICK line from AI output.")
    else:
        picks_raw = m.group(1)
        picks = [p.strip().upper() for p in picks_raw.split(",") if p.strip()]

        shown = 0
        for p in picks:
            p_clean = p.replace(".CA", "").replace("$", "").strip().upper()
            hit = top[top["name"].astype(str).str.upper() == p_clean]

            # fallback: match Yahoo symbol if AI used it
            if hit.empty:
              hit = top[
                  top["symbol"].astype(str).str.upper().str.replace("$", "") == p_clean
             ]

            if hit.empty:
                continue
            plot_levels_ladder(hit.iloc[0])
            shown += 1

        if shown == 0:
            st.info("AI picks did not match the Top 10 names; no charts rendered.")
