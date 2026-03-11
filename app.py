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
from egx_scan_once import ensure_ohlcv, filter_last_minutes
from openai import OpenAI

from zoneinfo import ZoneInfo

CAIRO_TZ = ZoneInfo("Africa/Cairo")
UTC_TZ = ZoneInfo("UTC")


def to_cairo(ts):
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC_TZ)
    return ts.tz_convert(CAIRO_TZ)


def minutes_delay_from_now(ts):
    ts_cairo = to_cairo(ts)
    now_cairo = pd.Timestamp.now(tz=CAIRO_TZ)
    return int((now_cairo - ts_cairo).total_seconds() / 60)


# -----------------------------
# Page config MUST be first Streamlit call
# -----------------------------
st.set_page_config(page_title="EGX Intraday Scanner", layout="wide")


# -----------------------------
# Simple password gate (Secrets first)
# -----------------------------
APP_PASSWORD = st.secrets.get("APP_PASSWORD", os.getenv("APP_PASSWORD", ""))

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

LAST_MINUTES = 90

STRONG_HIT = 2.5
WATCHLIST_LOW = 2.0

RECENT_BARS = 6
IMPULSE_SPIKE_MIN = 1.8

PULLBACK_MIN = 0.20
PULLBACK_MAX = 0.45
RETRACE_REJECT = 0.65

PRICE_WINDOW = 6
MIN_MOVE_PCT = 0.015
VOL_CONFIRM_MULT = 1.2

# Breakout-continuation detector
BREAKOUT_LOOKBACK = 10
BREAKOUT_BUFFER_PCT = 0.001
BREAKOUT_MIN_BARS = 3
BREAKOUT_VOL_MULT = 1.6

MIN_TURNOVER_EGP = 3_000_000
USE_LIQUIDITY_FILTER = False

ENTRY_CHASE_BUFFER_PCT = 0.0025
MIN_REMAINING_UPSIDE_PCT = 0.012

st.title("EGX Intraday Scanner (Always Shows Top 10)")

st.caption(
    f"Defaults: interval={INTERVAL}, period={PERIOD}, freshness_window={LAST_MINUTES}m, "
    f"vol_SMA={VOL_LOOKBACK} bars. "
    f"Buckets: ✅≥{STRONG_HIT}× | ⚠️ {WATCHLIST_LOW}–{STRONG_HIT}× | 👀 <{WATCHLIST_LOW}×. "
    f"Signals: 🚀 spike (≥{IMPULSE_SPIKE_MIN}× in last {RECENT_BARS}) OR "
    f"📈 breakout (≥{MIN_MOVE_PCT*100:.1f}% + vol≥{VOL_CONFIRM_MULT}×) OR "
    f"⚡ momentum breakout (local base + breakout + vol≥{BREAKOUT_VOL_MULT}×). "
    f"Second-leg: pullback 20–45%. "
    f"Late filter: chased if >{ENTRY_CHASE_BUFFER_PCT*100:.2f}% above entry or "
    f"<{MIN_REMAINING_UPSIDE_PCT*100:.1f}% upside left to T2."
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


# -----------------------------
# Labels / ranking helpers
# -----------------------------
def bucket_label(r: float) -> str:
    if pd.isna(r):
        return "—"
    if r >= STRONG_HIT:
        return "✅ Strong"
    if r >= WATCHLIST_LOW:
        return "⚠️ Watchlist"
    return "👀 Warm"


def bucket_rank(r: float) -> int:
    if pd.isna(r):
        return 99
    if r >= STRONG_HIT:
        return 0
    if r >= WATCHLIST_LOW:
        return 1
    return 2


def state_rank(s: str) -> int:
    order = {
        "IN_PULLBACK_ZONE": 0,
        "BREAKOUT_READY": 1,
        "PULLING_BACK": 2,
        "WAIT_BREAKOUT": 3,
        "WAIT_PULLBACK": 4,
        "—": 50,
        "REJECT_DEEP": 80,
        "CHASED": 90,
    }
    return order.get(str(s), 60)


def signal_rank(t: str) -> int:
    if t == "VOLUME_SPIKE":
        return 0
    if t == "MOMENTUM_BREAKOUT":
        return 1
    if t == "PRICE_EXPANSION":
        return 2
    return 9


def signal_label(t: str) -> str:
    if t == "VOLUME_SPIKE":
        return "🚀 Spike"
    if t == "MOMENTUM_BREAKOUT":
        return "⚡ Momentum Breakout"
    if t == "PRICE_EXPANSION":
        return "📈 Breakout"
    return "—"


def liquidity_label(turnover_egp: float) -> str:
    if turnover_egp >= 10_000_000:
        return "✅ Very High"
    if turnover_egp >= 3_000_000:
        return "✅ High"
    if turnover_egp >= 1_000_000:
        return "⚠️ Medium"
    return "❌ Low"


def liquidity_score(turnover_egp: float) -> int:
    if turnover_egp >= 10_000_000:
        return 0
    if turnover_egp >= 3_000_000:
        return 1
    if turnover_egp >= 1_000_000:
        return 2
    return 3


# -----------------------------
# Metrics
# -----------------------------
def compute_turnover_egp(df: pd.DataFrame, bars: int = 78) -> float:
    df = ensure_ohlcv(df)
    if df.empty:
        return 0.0
    tail = df.tail(bars)
    return float((tail["Close"] * tail["Volume"]).sum())


def compute_recent_signal(df: pd.DataFrame) -> dict:
    df = ensure_ohlcv(df)
    if df.empty or len(df) < VOL_LOOKBACK + RECENT_BARS + 2:
        return {
            "signal_found": False,
            "signal_type": "—",
            "signal_time": "",
            "spike_ratio": float("nan"),
            "sig_high": float("nan"),
            "sig_low": float("nan"),
            "move_pct": float("nan"),
        }

    work = df.copy()
    work["vol_sma"] = work["Volume"].rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).mean()
    work["spike_ratio"] = work["Volume"] / work["vol_sma"]

    end = -2 if len(work) >= 3 else -1
    start = end - RECENT_BARS + 1
    window = work.iloc[start:end + 1].dropna(subset=["spike_ratio", "vol_sma"]).copy()

    if window.empty:
        return {
            "signal_found": False,
            "signal_type": "—",
            "signal_time": "",
            "spike_ratio": float("nan"),
            "sig_high": float("nan"),
            "sig_low": float("nan"),
            "move_pct": float("nan"),
        }

    spikes = window[window["spike_ratio"] >= IMPULSE_SPIKE_MIN]
    if not spikes.empty:
        best_idx = spikes["spike_ratio"].idxmax()
        bar = work.loc[best_idx]
        return {
            "signal_found": True,
            "signal_type": "VOLUME_SPIKE",
            "signal_time": str(best_idx),
            "spike_ratio": float(bar["spike_ratio"]),
            "sig_high": float(bar["High"]),
            "sig_low": float(bar["Low"]),
            "move_pct": float("nan"),
        }

    pw = min(PRICE_WINDOW, len(window))
    block = window.tail(pw)
    block_low = float(block["Low"].min())
    block_high = float(block["High"].max())

    move_pct = float("nan")
    if block_low > 0:
        move_pct = (block_high - block_low) / block_low

    if (not np.isnan(move_pct)) and (move_pct >= MIN_MOVE_PCT):
        vol_ok = (block["Volume"] >= VOL_CONFIRM_MULT * block["vol_sma"]).any()
        if vol_ok:
            high_rows = block[block["High"] == block_high]
            best_idx = high_rows.index[0] if not high_rows.empty else block.index[-1]
            bar = work.loc[best_idx]
            return {
                "signal_found": True,
                "signal_type": "PRICE_EXPANSION",
                "signal_time": str(best_idx),
                "spike_ratio": float(bar["spike_ratio"]),
                "sig_high": float(block_high),
                "sig_low": float(block_low),
                "move_pct": float(move_pct),
            }

    return {
        "signal_found": False,
        "signal_type": "—",
        "signal_time": "",
        "spike_ratio": float("nan"),
        "sig_high": float("nan"),
        "sig_low": float("nan"),
        "move_pct": float("nan"),
    }


def compute_breakout_continuation(df: pd.DataFrame) -> dict:
    df = ensure_ohlcv(df)
    if df.empty or len(df) < max(VOL_LOOKBACK + 2, BREAKOUT_LOOKBACK + 3):
        return {
            "signal_found": False,
            "signal_type": "—",
            "signal_time": "",
            "spike_ratio": float("nan"),
            "sig_high": float("nan"),
            "sig_low": float("nan"),
            "move_pct": float("nan"),
        }

    work = df.copy()
    work["vol_sma"] = work["Volume"].rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).mean()

    end = -2 if len(work) >= 3 else -1
    if len(work) < abs(end) + 1:
        return {
            "signal_found": False,
            "signal_type": "—",
            "signal_time": "",
            "spike_ratio": float("nan"),
            "sig_high": float("nan"),
            "sig_low": float("nan"),
            "move_pct": float("nan"),
        }

    base_start = max(0, end - BREAKOUT_LOOKBACK)
    base = work.iloc[base_start:end]

    if len(base) < BREAKOUT_MIN_BARS:
        return {
            "signal_found": False,
            "signal_type": "—",
            "signal_time": "",
            "spike_ratio": float("nan"),
            "sig_high": float("nan"),
            "sig_low": float("nan"),
            "move_pct": float("nan"),
        }

    bar = work.iloc[end]

    if pd.isna(bar["vol_sma"]):
        return {
            "signal_found": False,
            "signal_type": "—",
            "signal_time": "",
            "spike_ratio": float("nan"),
            "sig_high": float("nan"),
            "sig_low": float("nan"),
            "move_pct": float("nan"),
        }

    base_high = float(base["High"].max())
    base_low = float(base["Low"].min())
    bar_high = float(bar["High"])
    bar_close = float(bar["Close"])

    broke_out = bar_high >= base_high * (1.0 + BREAKOUT_BUFFER_PCT)
    vol_ok = float(bar["Volume"]) >= float(bar["vol_sma"]) * BREAKOUT_VOL_MULT
    green_ok = bar_close > float(bar["Open"])

    if not (broke_out and vol_ok and green_ok):
        return {
            "signal_found": False,
            "signal_type": "—",
            "signal_time": "",
            "spike_ratio": float("nan"),
            "sig_high": float("nan"),
            "sig_low": float("nan"),
            "move_pct": float("nan"),
        }

    move_pct = float("nan")
    if base_low > 0:
        move_pct = (bar_high - base_low) / base_low

    return {
        "signal_found": True,
        "signal_type": "MOMENTUM_BREAKOUT",
        "signal_time": str(work.index[end]),
        "spike_ratio": float(bar["Volume"] / bar["vol_sma"]),
        "sig_high": float(bar_high),
        "sig_low": float(base_low),
        "move_pct": float(move_pct),
    }


def compute_pullback_zone(sig_high: float, sig_low: float) -> dict:
    if np.isnan(sig_high) or np.isnan(sig_low):
        return {"pb_low": float("nan"), "pb_high": float("nan"), "sig_range": float("nan")}
    rng = sig_high - sig_low
    if rng <= 0:
        return {"pb_low": float("nan"), "pb_high": float("nan"), "sig_range": float("nan")}

    pb_high = sig_high - PULLBACK_MIN * rng
    pb_low = sig_high - PULLBACK_MAX * rng
    return {"pb_low": float(pb_low), "pb_high": float(pb_high), "sig_range": float(rng)}


def build_levels_from_pullback(*, sig_high: float, sig_low: float, pb_low: float, pb_high: float) -> dict:
    if (
        np.isnan(sig_high)
        or np.isnan(sig_low)
        or np.isnan(pb_low)
        or np.isnan(pb_high)
        or pb_low >= pb_high
        or sig_high <= sig_low
    ):
        return {}

    rng = sig_high - sig_low
    entry = pb_high * 1.0005

    structural_stop = min(pb_low * 0.9985, sig_low * 0.9990)
    stop = structural_stop

    if stop >= entry:
        stop = pb_low * 0.9990

    if stop >= entry:
        return {}

    t1 = max(sig_high, entry * 1.0030)
    t2 = max(sig_high + 0.50 * rng, entry * 1.0100)
    t3 = max(sig_high + 1.00 * rng, entry * 1.0180)

    if not (entry < t1 < t2 < t3):
        return {}

    return {
        "entry": float(entry),
        "stop": float(stop),
        "t1": float(t1),
        "t2": float(t2),
        "t3": float(t3),
    }


def build_levels_from_breakout(*, sig_high: float, sig_low: float) -> dict:
    if np.isnan(sig_high) or np.isnan(sig_low) or sig_high <= sig_low:
        return {}

    rng = sig_high - sig_low
    entry = sig_high * 1.0008

    stop = max(sig_high - 0.45 * rng, sig_low * 1.0000)
    stop = min(stop, entry * 0.9950)

    if stop >= entry:
        stop = sig_high - 0.35 * rng

    if stop >= entry:
        return {}

    t1 = max(sig_high + 0.50 * rng, entry * 1.0040)
    t2 = max(sig_high + 1.00 * rng, entry * 1.0100)
    t3 = max(sig_high + 1.50 * rng, entry * 1.0160)

    if not (entry < t1 < t2 < t3):
        return {}

    return {
        "entry": float(entry),
        "stop": float(stop),
        "t1": float(t1),
        "t2": float(t2),
        "t3": float(t3),
    }


def setup_state(
    df: pd.DataFrame,
    *,
    sig_high: float,
    sig_low: float,
    pb_low: float,
    pb_high: float,
    entry: float,
) -> str:
    df = ensure_ohlcv(df)
    if df.empty or np.isnan(sig_high) or np.isnan(sig_low) or np.isnan(pb_low) or np.isnan(pb_high):
        return "—"

    last = float(df["Close"].iloc[-1])
    rng = sig_high - sig_low
    if rng <= 0:
        return "—"

    if np.isfinite(entry) and last > entry * (1.0 + ENTRY_CHASE_BUFFER_PCT):
        return "CHASED"

    retr = (sig_high - last) / rng
    if retr > RETRACE_REJECT:
        return "REJECT_DEEP"

    if pb_low <= last <= pb_high:
        return "IN_PULLBACK_ZONE"

    if last > pb_high:
        return "WAIT_PULLBACK"

    return "PULLING_BACK"


# -----------------------------
# Visual
# -----------------------------
def plot_levels_ladder(row: pd.Series):
    entry, stop, t1, t2, t3 = row["entry"], row["stop"], row["t1"], row["t2"], row["t3"]
    last_price = row["last_price"]
    name = row["name"]
    symbol = row["symbol"]
    bucket = row["bucket"]
    ratio = row["spike_ratio"]
    state = row.get("setup_state", "—")
    siglbl = row.get("signal_label", "—")
    liq = row.get("liquidity", "—")

    vals = [stop, entry, t1, t2, t3]
    labels = ["STOP", "ENTRY", "T1", "T2", "T3"]

    fig, ax = plt.subplots(figsize=(9, 2.1))
    title_ratio = "n/a" if pd.isna(ratio) else f"{ratio:.2f}×"
    ax.set_title(f"{siglbl} | {bucket} | {state} | {liq} | {name} — {symbol} | vol {title_ratio}", fontsize=11)

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

        sig = compute_recent_signal(df_full)
        if not sig["signal_found"]:
            sig = compute_breakout_continuation(df_full)

        sig_type = sig["signal_type"]
        sig_time = sig["signal_time"]
        ratio = sig["spike_ratio"]
        sig_high = sig["sig_high"]
        sig_low = sig["sig_low"]
        move_pct = sig["move_pct"]

        turnover = compute_turnover_egp(df_full)
        liq = liquidity_label(turnover)
        liq_score = liquidity_score(turnover)

        if USE_LIQUIDITY_FILTER and (turnover < MIN_TURNOVER_EGP):
            continue

        last_price = float(df_full["Close"].iloc[-1])

        if sig_type == "MOMENTUM_BREAKOUT":
            pb_low = float("nan")
            pb_high = float("nan")

            levels = build_levels_from_breakout(sig_high=sig_high, sig_low=sig_low)
            if not levels:
                continue

            entry = levels["entry"]
            stop = levels["stop"]
            t1 = levels["t1"]
            t2 = levels["t2"]
            t3 = levels["t3"]

            if last_price > entry * (1.0 + ENTRY_CHASE_BUFFER_PCT):
                state = "CHASED"
            elif last_price >= sig_high * 0.997:
                state = "BREAKOUT_READY"
            else:
                state = "WAIT_BREAKOUT"

        else:
            pb = compute_pullback_zone(sig_high, sig_low)
            pb_low = pb["pb_low"]
            pb_high = pb["pb_high"]

            levels = build_levels_from_pullback(
                sig_high=sig_high,
                sig_low=sig_low,
                pb_low=pb_low,
                pb_high=pb_high,
            )
            if not levels:
                continue

            entry = levels["entry"]
            stop = levels["stop"]
            t1 = levels["t1"]
            t2 = levels["t2"]
            t3 = levels["t3"]

            state = setup_state(
                df_full,
                sig_high=sig_high,
                sig_low=sig_low,
                pb_low=pb_low,
                pb_high=pb_high,
                entry=entry,
            )

        remaining_upside_pct = float("nan")
        if np.isfinite(t2) and last_price > 0:
            remaining_upside_pct = (t2 - last_price) / last_price

        if np.isfinite(remaining_upside_pct) and remaining_upside_pct < MIN_REMAINING_UPSIDE_PCT:
            state = "CHASED"

        rows.append({
            "signal_type": sig_type,
            "signal_label": signal_label(sig_type),
            "signal_rank": signal_rank(sig_type),
            "signal_time": sig_time,
            "move_pct": move_pct,

            "bucket": bucket_label(ratio),
            "bucket_rank": bucket_rank(ratio),

            "setup_state": state,
            "state_rank": state_rank(state),

            "liquidity": liq,
            "liquidity_score": liq_score,

            "name": display_name(t),
            "symbol": t,

            "spike_ratio": ratio,
            "turnover_egp": turnover,
            "last_price": last_price,
            "remaining_upside_pct": remaining_upside_pct,

            "sig_high": sig_high,
            "sig_low": sig_low,
            "pb_low": pb_low,
            "pb_high": pb_high,

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
            ["signal_rank", "state_rank", "liquidity_score", "remaining_upside_pct", "bucket_rank", "spike_ratio", "turnover_egp"],
            ascending=[True, True, True, False, True, False, False],
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
meta = st.session_state.get(
    "meta",
    {"elapsed": None, "watchlist": len(EGX_SAFE_INTRADAY), "data_ok": 0, "recent_ok": 0},
)
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
# Always show Top N
# -----------------------------
top = ranked.head(TOP_N).copy()

top["stop_%"] = np.where(np.isfinite(top["entry"]), (top["stop"] / top["entry"] - 1.0) * 100.0, np.nan)
top["t1_%"] = np.where(np.isfinite(top["entry"]), (top["t1"] / top["entry"] - 1.0) * 100.0, np.nan)
top["t2_%"] = np.where(np.isfinite(top["entry"]), (top["t2"] / top["entry"] - 1.0) * 100.0, np.nan)
top["t3_%"] = np.where(np.isfinite(top["entry"]), (top["t3"] / top["entry"] - 1.0) * 100.0, np.nan)
top["move_%"] = np.where(np.isfinite(top["move_pct"]), top["move_pct"] * 100.0, np.nan)
top["remaining_upside_%"] = np.where(
    np.isfinite(top["remaining_upside_pct"]),
    top["remaining_upside_pct"] * 100.0,
    np.nan,
)

st.subheader(f"Top {TOP_N} (signal + setup state + liquidity)")

display_cols = [
    "signal_label",
    "setup_state",
    "liquidity",
    "live",
    "last_bar_time",
    "signal_time",
    "name",
    "symbol",
    "spike_ratio",
    "move_%",
    "turnover_egp",
    "last_price",
    "sig_low",
    "sig_high",
    "pb_low",
    "pb_high",
    "entry",
    "stop",
    "t1",
    "t2",
    "t3",
    "stop_%",
    "t1_%",
    "t2_%",
    "t3_%",
    "remaining_upside_%",
]
display_cols = [c for c in display_cols if c in top.columns]

st.dataframe(
    top[display_cols],
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
c1.metric(f"✅ Strong (≥{STRONG_HIT}×)", len(strong))
c2.metric(f"⚠️ Watchlist ({WATCHLIST_LOW}–{STRONG_HIT}×)", len(watch))
c3.metric(f"👀 Warm (<{WATCHLIST_LOW}×)", len(warm))

with st.expander(f"✅ Strong (≥{STRONG_HIT}×)"):
    if not strong.empty:
        st.dataframe(strong.drop(columns=["bucket_rank"]), use_container_width=True, hide_index=True)
    else:
        st.info("No strong names right now.")

with st.expander(f"⚠️ Watchlist ({WATCHLIST_LOW}–{STRONG_HIT}×)"):
    if not watch.empty:
        st.dataframe(watch.drop(columns=["bucket_rank"]), use_container_width=True, hide_index=True)
    else:
        st.info("No medium names right now.")

with st.expander(f"👀 Warm (<{WATCHLIST_LOW}×)"):
    if not warm.empty:
        st.dataframe(warm.drop(columns=["bucket_rank"]), use_container_width=True, hide_index=True)
    else:
        st.info("No warm names right now.")


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
        st.error("OPENAI_API_KEY not found. Put it in Streamlit Secrets.")
        st.stop()

    client = OpenAI(api_key=api_key)

    payload = []
    for _, r in top.iterrows():
        payload.append({
            "signal_type": r.get("signal_type", "—"),
            "signal_time": r.get("signal_time", ""),
            "setup_state": r.get("setup_state", "—"),
            "live": bool(r.get("live", False)),
            "last_bar_time": r.get("last_bar_time", ""),
            "name": r["name"],
            "symbol": r["symbol"],
            "spike_ratio": None if pd.isna(r["spike_ratio"]) else float(r["spike_ratio"]),
            "move_pct": None if pd.isna(r.get("move_pct", np.nan)) else float(r["move_pct"]),
            "turnover_egp": float(r["turnover_egp"]),
            "liquidity": r.get("liquidity", "—"),
            "liquidity_score": int(r.get("liquidity_score", 99)),
            "last_price": float(r["last_price"]),
            "remaining_upside_pct": None if pd.isna(r.get("remaining_upside_pct", np.nan)) else float(r["remaining_upside_pct"]),
            "pb_low": None if pd.isna(r.get("pb_low", np.nan)) else float(r["pb_low"]),
            "pb_high": None if pd.isna(r.get("pb_high", np.nan)) else float(r["pb_high"]),
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

    allowed = ", ".join(top["symbol"].astype(str).str.upper().tolist())

    instruction = f"""
You are an EGX intraday trading assistant.
You are given Top {TOP_N} ranked candidates from EGX30.

Each candidate includes:
- signal_type (VOLUME_SPIKE or PRICE_EXPANSION or MOMENTUM_BREAKOUT)
- spike_ratio (volume spike vs SMA{VOL_LOOKBACK}) and/or move_pct (price expansion / breakout move)
- turnover_egp (money traded today) and liquidity label (Very High/High/Medium/Low)
- setup_state (second-leg readiness or breakout readiness)
- pullback zone pb_low..pb_high and levels (entry/stop/targets)
- live + last_bar_time (data freshness)
- remaining_upside_pct (remaining room to T2)

Scoring guidance:
- Prefer liquidity: ✅ Very High > ✅ High > ⚠️ Medium. Strongly penalize ❌ Low.
- Prefer setup_state in this order: IN_PULLBACK_ZONE > BREAKOUT_READY > PULLING_BACK > WAIT_BREAKOUT > WAIT_PULLBACK. Avoid CHASED/REJECT_DEEP.
- Prefer signal_type = VOLUME_SPIKE, then MOMENTUM_BREAKOUT, then PRICE_EXPANSION when otherwise similar.
- Prefer higher turnover_egp.
- Penalize chasing: if last_price > entry by more than 0.30%, recommend WAIT or SKIP.
- Penalize setups with remaining_upside_pct < {MIN_REMAINING_UPSIDE_PCT:.3f}.
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
- WHY: 3-5 bullets referencing signal_type + (spike_ratio or move_pct) + turnover_egp/liquidity + remaining_upside_pct
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

            hit = top[top["symbol"].astype(str).str.upper().str.replace("$", "", regex=False) == p_clean]

            if hit.empty:
                continue

            plot_levels_ladder(hit.iloc[0])
            shown += 1

        if shown == 0:
            st.info("AI picks did not match the Top 10 symbols; no charts rendered.")