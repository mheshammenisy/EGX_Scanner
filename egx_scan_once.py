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
PULLBACK_MIN = 0.30          # 30% retrace
PULLBACK_MAX = 0.60          # 60% retrace
RETRACE_REJECT = 0.70        # reject if >70% retrace
ENTRY_CHASE_BUFFER_PCT = 0.003   # 0.30% above planned entry => chased
MIN_REMAINING_UPSIDE_PCT = 0.012 # 1.2% to T2 required


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


def build_levels_from_pullback(*, pb_low: float, pb_high: float) -> dict:
    """
    Second-leg friendly levels:
    - entry: top of pullback zone with tiny buffer
    - stop: bottom of pullback zone
    - targets: 1R / 2R / 3R (capped)
    """
    if np.isnan(pb_low) or np.isnan(pb_high) or pb_low >= pb_high:
        return {}

    entry = pb_high * 1.0001
    stop = pb_low * 0.998
    if stop >= entry:
        return {}

    R = entry - stop
    R_cap = min(R, entry * 0.015)  # cap R to 1.5% of entry to avoid crazy wide levels

    t1 = entry + 1.0 * R_cap
    t2 = entry + 2.0 * R_cap
    t3 = entry + 3.0 * R_cap

    return {
        "entry": float(entry),
        "stop": float(stop),
        "risk_R": float(R),
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
# Detection
# -----------------------------
def find_recent_signal(
    df: pd.DataFrame,
    *,
    vol_lookback: int = 20,
    recent_bars: int = 4,
    spike_min: float = 2.5,
    price_window: int = 4,
    min_move_pct: float = 0.02,
    vol_confirm_mult: float = 1.3,
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
    window = work.iloc[start : end + 1].copy()
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
    recent_bars: int = 4,
    spike_min: float = 2.5,
    price_window: int = 4,
    min_move_pct: float = 0.02,
    vol_confirm_mult: float = 1.3,
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
        return None

    spike_time = sig["time"]
    bar = work.loc[spike_time]

    if pd.isna(bar["atr"]):
        return None

    if require_green_bar and not (float(bar["Close"]) > float(bar["Open"])):
        return None

    sig_high = float(sig["high"])
    sig_low = float(sig["low"])

    pb = compute_pullback_zone(sig_high, sig_low)
    pb_low = float(pb["pb_low"])
    pb_high = float(pb["pb_high"])

    levels = build_levels_from_pullback(pb_low=pb_low, pb_high=pb_high)
    if not levels:
        return None

    entry = float(levels["entry"])
    stop = float(levels["stop"])
    t1 = float(levels["t1"])
    t2 = float(levels["t2"])
    t3 = float(levels["t3"])
    risk_R = float(levels["risk_R"])

    last_price = float(work["Close"].iloc[-1])

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
    spike_min: float = 2.5,
    recent_bars: int = 4,
    price_window: int = 4,
    min_move_pct: float = 0.02,
    vol_confirm_mult: float = 1.3,
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
        "PULLING_BACK": 1,
        "WAIT_PULLBACK": 2,
        "—": 50,
        "REJECT_DEEP": 80,
        "CHASED": 90,
    }
    signal_order = {
        "VOLUME_SPIKE": 0,
        "PRICE_EXPANSION": 1,
    }

    out["state_rank"] = out["setup_state"].map(state_order).fillna(60)
    out["signal_rank"] = out["signal_type"].map(signal_order).fillna(9)

    out = out.sort_values(
        ["signal_rank", "state_rank", "turnover_egp", "spike_mult"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)

    return out


# -----------------------------
# CLI entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spike_min", type=float, default=2.5)
    parser.add_argument("--recent_bars", type=int, default=4)
    parser.add_argument("--price_window", type=int, default=4)
    parser.add_argument("--min_move_pct", type=float, default=0.02)
    parser.add_argument("--vol_confirm_mult", type=float, default=1.3)
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