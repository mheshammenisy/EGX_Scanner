# egx_scan_once.py
# Python 3.13+ compatible
# pip install yfinance pandas numpy

from __future__ import annotations

import argparse
import contextlib
import io
from dataclasses import dataclass
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from egx_watchlist import EGX_SAFE_INTRADAY, display_name


# -----------------------------
# Strategy constants
# -----------------------------
PULLBACK_MIN = 0.20
PULLBACK_MAX = 0.45
RETRACE_REJECT = 0.65
ENTRY_CHASE_BUFFER_PCT = 0.0025
MIN_REMAINING_UPSIDE_PCT = 0.012

# Reject weak EGX setups
MIN_T1_PCT = 1.0
MIN_T2_PCT = 2.5
MIN_T3_PCT = 4.0

# Momentum breakout detection
BREAKOUT_LOOKBACK = 10
BREAKOUT_BUFFER_PCT = 0.001
BREAKOUT_MIN_BARS = 3
BREAKOUT_VOL_MULT = 1.6


# -----------------------------
# Helpers
# -----------------------------
def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels == 2 and len(df.columns.levels[1]) == 1:
            df = df.droplevel(1, axis=1)

    if len(df.columns) and isinstance(df.columns[0], tuple):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]

    cols = {str(c).lower(): c for c in df.columns}
    needed = ["open", "high", "low", "close", "volume"]
    if not all(k in cols for k in needed):
        return pd.DataFrame()

    out = df[[cols["open"], cols["high"], cols["low"], cols["close"], cols["volume"]]].copy()
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    return out.dropna().sort_index()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.rolling(n, min_periods=n).mean()


def last_completed_bar_index(df: pd.DataFrame) -> int:
    return -2 if len(df) >= 3 else -1


def compute_turnover_egp(df: pd.DataFrame, bars: int = 78) -> float:
    if df.empty:
        return 0.0
    tail = df.tail(bars)
    return float((tail["Close"] * tail["Volume"]).sum())


def cairo_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ZoneInfo("Africa/Cairo"))


def filter_last_minutes(df: pd.DataFrame, *, minutes: int) -> pd.DataFrame:
    """
    Keep only rows inside the last N minutes (Cairo time).
    Used by the Streamlit app / freshness gate for CLI.
    """
    if df.empty:
        return df

    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize(ZoneInfo("UTC"))

    idx_cairo = idx.tz_convert(ZoneInfo("Africa/Cairo"))
    cutoff = cairo_now() - pd.Timedelta(minutes=minutes)

    out = df.copy()
    out.index = idx_cairo
    return out.loc[out.index >= cutoff]


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
    """
    Structure-based pullback levels:
    - entry: reclaim upper pullback zone
    - stop: under lower pullback zone / signal low buffer
    - T1: retest signal high
    - T2: signal high + 50% of impulse range
    - T3: signal high + 100% of impulse range
    """
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
        "risk_R": float(entry - stop),
        "t1": float(t1),
        "t2": float(t2),
        "t3": float(t3),
    }


def build_levels_from_breakout(*, sig_high: float, sig_low: float) -> dict:
    """
    Structure-based breakout levels:
    - entry: just above breakout high
    - stop: inside/under base, not huge
    - T1/T2/T3: measured move from base range
    """
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

    t1 = max(sig_high + 0.50 * rng, entry * 1.0100)
    t2 = max(sig_high + 1.00 * rng, entry * 1.0250)
    t3 = max(sig_high + 1.50 * rng, entry * 1.0400)

    if not (entry < t1 < t2 < t3):
        return {}

    return {
        "entry": float(entry),
        "stop": float(stop),
        "risk_R": float(entry - stop),
        "t1": float(t1),
        "t2": float(t2),
        "t3": float(t3),
    }


def passes_target_quality(entry: float, t1: float, t2: float, t3: float) -> bool:
    if not np.isfinite(entry) or entry <= 0:
        return False
    if not np.isfinite(t1) or not np.isfinite(t2) or not np.isfinite(t3):
        return False

    t1_pct = ((t1 / entry) - 1.0) * 100.0
    t2_pct = ((t2 / entry) - 1.0) * 100.0
    t3_pct = ((t3 / entry) - 1.0) * 100.0

    return (
        t1_pct >= MIN_T1_PCT
        and t2_pct >= MIN_T2_PCT
        and t3_pct >= MIN_T3_PCT
    )


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
# Detection
# -----------------------------
def find_recent_signal(
    df: pd.DataFrame,
    *,
    vol_lookback: int = 20,
    recent_bars: int = 6,
    spike_min: float = 1.8,
    price_window: int = 6,
    min_move_pct: float = 0.015,
    vol_confirm_mult: float = 1.2,
) -> Optional[dict]:
    """
    Delay-proof detection within last `recent_bars` completed bars.

    Priority:
    1) Volume Spike (>= spike_min * vol_sma)
    2) Price Expansion (>= min_move_pct) + mild volume confirm (>= vol_confirm_mult * vol_sma)
    """
    df = ensure_ohlcv(df)
    if df.empty:
        return None

    need = vol_lookback + recent_bars + 2
    if len(df) < need:
        return None

    work = df.copy()
    work["vol_sma"] = work["Volume"].rolling(vol_lookback, min_periods=vol_lookback).mean()
    work["spike_ratio"] = work["Volume"] / work["vol_sma"]

    end = last_completed_bar_index(work)
    start = end - recent_bars + 1
    window = work.iloc[start:end + 1].copy()
    window = window.dropna(subset=["spike_ratio", "vol_sma"])

    if window.empty:
        return None

    spikes = window[window["spike_ratio"] >= spike_min]
    if not spikes.empty:
        best_time = spikes["spike_ratio"].idxmax()
        bar = work.loc[best_time]
        return {
            "type": "VOLUME_SPIKE",
            "time": best_time,
            "spike_ratio": float(bar["spike_ratio"]),
            "high": float(bar["High"]),
            "low": float(bar["Low"]),
            "move_pct": float("nan"),
        }

    pw = min(price_window, len(window))
    last_block = window.tail(pw)
    if last_block.empty:
        return None

    block_low = float(last_block["Low"].min())
    block_high = float(last_block["High"].max())
    if block_low <= 0:
        return None

    move_pct = (block_high - block_low) / block_low
    if move_pct < min_move_pct:
        return None

    vol_ok = (last_block["Volume"] >= vol_confirm_mult * last_block["vol_sma"]).any()
    if not vol_ok:
        return None

    high_rows = last_block[last_block["High"] == block_high]
    best_time = high_rows.index[0] if not high_rows.empty else last_block.index[-1]
    bar = work.loc[best_time]

    return {
        "type": "PRICE_EXPANSION",
        "time": best_time,
        "spike_ratio": float(bar["spike_ratio"]),
        "high": float(block_high),
        "low": float(block_low),
        "move_pct": float(move_pct),
    }


def find_momentum_breakout(df: pd.DataFrame) -> Optional[dict]:
    """
    Detect breakout continuation without pullback.
    """
    df = ensure_ohlcv(df)

    if len(df) < BREAKOUT_LOOKBACK + 5:
        return None

    work = df.copy()
    work["vol_sma"] = work["Volume"].rolling(20, min_periods=20).mean()

    end = last_completed_bar_index(work)
    if len(work) < abs(end) + 1:
        return None

    bar = work.iloc[end]

    if pd.isna(bar["vol_sma"]):
        return None

    base = work.iloc[end - BREAKOUT_LOOKBACK:end]

    if len(base) < BREAKOUT_MIN_BARS:
        return None

    base_high = float(base["High"].max())
    base_low = float(base["Low"].min())

    bar_high = float(bar["High"])
    bar_close = float(bar["Close"])

    broke_out = bar_high >= base_high * (1 + BREAKOUT_BUFFER_PCT)
    vol_ok = float(bar["Volume"]) >= float(bar["vol_sma"]) * BREAKOUT_VOL_MULT
    green_bar = bar_close > float(bar["Open"])

    if not (broke_out and vol_ok and green_bar):
        return None

    move_pct = float("nan")
    if base_low > 0:
        move_pct = (bar_high - base_low) / base_low

    return {
        "type": "MOMENTUM_BREAKOUT",
        "time": work.index[end],
        "spike_ratio": float(bar["Volume"] / bar["vol_sma"]),
        "high": bar_high,
        "low": base_low,
        "move_pct": float(move_pct),
    }


# -----------------------------
# Trade plan
# -----------------------------
@dataclass
class TradePlan:
    symbol: str
    name: str
    signal_type: str
    spike_time: pd.Timestamp
    spike_mult: float
    last_price: float
    turnover_egp: float

    sig_high: float
    sig_low: float
    pb_low: float
    pb_high: float

    entry: float
    stop: float
    risk_R: float
    t1: float
    t2: float
    t3: float

    setup_state: str
    remaining_upside_pct: float
    triggered: bool


def compute_trade_plan_from_signal(
    symbol: str,
    name: str,
    df: pd.DataFrame,
    *,
    vol_lookback: int = 20,
    recent_bars: int = 6,
    spike_min: float = 1.8,
    price_window: int = 6,
    min_move_pct: float = 0.015,
    vol_confirm_mult: float = 1.2,
    require_green_bar: bool = True,
    atr_n: int = 14,
) -> Optional[TradePlan]:

    df = ensure_ohlcv(df)
    if df.empty:
        return None

    min_len = max(vol_lookback + recent_bars + 3, atr_n + 3)
    if len(df) < min_len:
        return None

    work = df.copy()
    work["atr"] = atr(work, n=atr_n)

    sig = find_recent_signal(
        work,
        vol_lookback=vol_lookback,
        recent_bars=recent_bars,
        spike_min=spike_min,
        price_window=price_window,
        min_move_pct=min_move_pct,
        vol_confirm_mult=vol_confirm_mult,
    )

    if sig is None:
        sig = find_momentum_breakout(work)

    if sig is None:
        return None

    spike_time = sig["time"]
    bar = work.loc[spike_time]

    if pd.isna(bar["atr"]):
        return None

    if require_green_bar and not (float(bar["Close"]) > float(bar["Open"])):
        return None

    sig_high = float(sig["high"])
    sig_low = float(sig["low"])
    last_price = float(work["Close"].iloc[-1])

    if str(sig["type"]) == "MOMENTUM_BREAKOUT":
        pb_low = float("nan")
        pb_high = float("nan")

        levels = build_levels_from_breakout(sig_high=sig_high, sig_low=sig_low)
        if not levels:
            return None

        entry = float(levels["entry"])
        stop = float(levels["stop"])
        risk_R = float(levels["risk_R"])
        t1 = float(levels["t1"])
        t2 = float(levels["t2"])
        t3 = float(levels["t3"])

        if not passes_target_quality(entry, t1, t2, t3):
            return None

        if last_price > entry * (1.0 + ENTRY_CHASE_BUFFER_PCT):
            state = "CHASED"
        elif last_price >= sig_high * 0.997:
            state = "BREAKOUT_READY"
        else:
            state = "WAIT_BREAKOUT"
    else:
        pb = compute_pullback_zone(sig_high, sig_low)
        pb_low = float(pb["pb_low"])
        pb_high = float(pb["pb_high"])

        levels = build_levels_from_pullback(
            sig_high=sig_high,
            sig_low=sig_low,
            pb_low=pb_low,
            pb_high=pb_high,
        )
        if not levels:
            return None

        entry = float(levels["entry"])
        stop = float(levels["stop"])
        risk_R = float(levels["risk_R"])
        t1 = float(levels["t1"])
        t2 = float(levels["t2"])
        t3 = float(levels["t3"])

        if not passes_target_quality(entry, t1, t2, t3):
            return None

        state = setup_state(
            work,
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

    return TradePlan(
        symbol=symbol,
        name=name,
        signal_type=str(sig["type"]),
        spike_time=spike_time,
        spike_mult=float(sig["spike_ratio"]),
        last_price=last_price,
        turnover_egp=compute_turnover_egp(work),
        sig_high=sig_high,
        sig_low=sig_low,
        pb_low=pb_low,
        pb_high=pb_high,
        entry=entry,
        stop=stop,
        risk_R=risk_R,
        t1=t1,
        t2=t2,
        t3=t3,
        setup_state=state,
        remaining_upside_pct=remaining_upside_pct,
        triggered=bool(last_price >= entry),
    )


# -----------------------------
# Yahoo fetch
# -----------------------------
def fetch_intraday(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            df = yf.download(
                tickers=symbol,
                interval=interval,
                period=period,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# -----------------------------
# CLI scan
# -----------------------------
def scan_once(
    tickers: list[str],
    *,
    interval: str = "5m",
    period: str = "5d",
    vol_lookback: int = 20,
    spike_min: float = 1.8,
    recent_bars: int = 6,
    price_window: int = 6,
    min_move_pct: float = 0.015,
    vol_confirm_mult: float = 1.2,
    min_turnover_egp: float = 0.0,
    last_minutes: int = 15,
    require_green_bar: bool = True,
) -> pd.DataFrame:

    plans: list[TradePlan] = []

    for sym in tickers:
        raw = fetch_intraday(sym, interval, period)
        df = ensure_ohlcv(raw)
        if df.empty:
            continue

        df_recent = filter_last_minutes(df, minutes=last_minutes)
        if df_recent.empty:
            continue

        plan = compute_trade_plan_from_signal(
            sym,
            display_name(sym),
            df,
            vol_lookback=vol_lookback,
            recent_bars=recent_bars,
            spike_min=spike_min,
            price_window=price_window,
            min_move_pct=min_move_pct,
            vol_confirm_mult=vol_confirm_mult,
            require_green_bar=require_green_bar,
        )
        if plan is None:
            continue

        if plan.turnover_egp < min_turnover_egp:
            continue

        plans.append(plan)

    if not plans:
        return pd.DataFrame()

    out = pd.DataFrame([p.__dict__ for p in plans])

    state_order = {
        "IN_PULLBACK_ZONE": 0,
        "BREAKOUT_READY": 1,
        "PULLING_BACK": 2,
        "WAIT_BREAKOUT": 3,
        "WAIT_PULLBACK": 4,
        "—": 50,
        "REJECT_DEEP": 80,
        "CHASED": 90,
    }
    signal_order = {
        "VOLUME_SPIKE": 0,
        "MOMENTUM_BREAKOUT": 1,
        "PRICE_EXPANSION": 2,
    }

    out["state_rank"] = out["setup_state"].map(state_order).fillna(60)
    out["signal_rank"] = out["signal_type"].map(signal_order).fillna(9)

    out = out.sort_values(
        ["signal_rank", "state_rank", "remaining_upside_pct", "turnover_egp", "spike_mult"],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)

    return out


# -----------------------------
# CLI entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spike_min", type=float, default=1.8)
    parser.add_argument("--recent_bars", type=int, default=6)
    parser.add_argument("--price_window", type=int, default=6)
    parser.add_argument("--min_move_pct", type=float, default=0.015)
    parser.add_argument("--vol_confirm_mult", type=float, default=1.2)
    parser.add_argument("--last_minutes", type=int, default=15)
    args = parser.parse_args()

    df = scan_once(
        EGX_SAFE_INTRADAY,
        spike_min=args.spike_min,
        recent_bars=args.recent_bars,
        price_window=args.price_window,
        min_move_pct=args.min_move_pct,
        vol_confirm_mult=args.vol_confirm_mult,
        last_minutes=args.last_minutes,
    )

    if df.empty:
        print("No hits.")
    else:
        cols = [
            "name",
            "symbol",
            "signal_type",
            "setup_state",
            "spike_time",
            "spike_mult",
            "last_price",
            "pb_low",
            "pb_high",
            "entry",
            "stop",
            "t1",
            "t2",
            "t3",
            "remaining_upside_pct",
            "turnover_egp",
        ]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()