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
# Helpers
# -----------------------------
def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # Drop ticker level if MultiIndex (Field, Ticker)
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


# -----------------------------
# Detection: 2 types
# -----------------------------
def find_recent_signal(
    df: pd.DataFrame,
    *,
    vol_lookback: int = 20,
    recent_bars: int = 6,
    # Type (1): Volume spike
    spike_min: float = 2.5,
    # Type (2): Range expansion breakout
    price_window: int = 4,          # look at last 4 completed bars inside the recent window
    min_move_pct: float = 0.02,     # >= 2% move
    vol_confirm_mult: float = 1.3,  # volume >= 1.3x SMA confirms price move
) -> Optional[dict]:
    """
    Delay-proof detection within last `recent_bars` completed bars.

    Priority:
    1) Volume Spike (>= spike_min * vol_sma)
    2) Price Expansion (>= min_move_pct) + mild volume confirm (>= vol_confirm_mult * vol_sma)

    Returns dict:
      {
        "type": "VOLUME_SPIKE" | "PRICE_EXPANSION",
        "time": index,
        "spike_ratio": float,
        "high": float,
        "low": float,
      }
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

    end = last_completed_bar_index(work)  # usually -2
    start = end - recent_bars + 1
    window = work.iloc[start : end + 1].copy()
    window = window.dropna(subset=["spike_ratio", "vol_sma"])

    if window.empty:
        return None

    # -----------------------------
    # (1) Volume spike
    # -----------------------------
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

    # -----------------------------
    # (2) Price expansion breakout + mild volume confirmation
    # -----------------------------
    # Take last `price_window` completed bars inside that window
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

    # volume confirmation: at least ONE bar in that block has volume >= vol_confirm_mult * vol_sma
    vol_ok = (last_block["Volume"] >= vol_confirm_mult * last_block["vol_sma"]).any()
    if not vol_ok:
        return None

    # time anchor: use the bar where high == block_high (first occurrence)
    high_rows = last_block[last_block["High"] == block_high]
    best_time = high_rows.index[0] if not high_rows.empty else last_block.index[-1]
    bar = work.loc[best_time]

    return {
        "type": "PRICE_EXPANSION",
        "time": best_time,
        "spike_ratio": float(bar["spike_ratio"]),  # will likely be < spike_min, still useful info
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
    entry: float
    stop: float
    risk_R: float
    t1: float
    t2: float
    t3: float
    turnover_egp: float
    last_price: float
    triggered: bool


def compute_trade_plan_from_signal(
    symbol: str,
    name: str,
    df: pd.DataFrame,
    *,
    vol_lookback: int = 20,
    # detection params
    recent_bars: int = 6,
    spike_min: float = 2.5,
    price_window: int = 4,
    min_move_pct: float = 0.02,
    vol_confirm_mult: float = 1.3,
    # plan params
    require_green_bar: bool = True,
    entry_buffer_pct: float = 0.0005,
    atr_n: int = 14,
    stop_buffer_atr: float = 0.25,
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

    # green bar filter only makes sense for the anchor bar we picked
    if require_green_bar and not (float(bar["Close"]) > float(bar["Open"])):
        return None

    sig_high = float(sig["high"])
    sig_low = float(sig["low"])
    sig_atr = float(bar["atr"])

    entry = sig_high * (1.0 + entry_buffer_pct)
    stop = sig_low - stop_buffer_atr * sig_atr
    if stop >= entry:
        return None

    R = entry - stop

    return TradePlan(
        symbol=symbol,
        name=name,
        signal_type=str(sig["type"]),
        spike_time=spike_time,
        spike_mult=float(sig["spike_ratio"]),
        entry=entry,
        stop=stop,
        risk_R=R,
        t1=entry + 1 * R,
        t2=entry + 2 * R,
        t3=entry + 3 * R,
        turnover_egp=compute_turnover_egp(work),
        last_price=float(work["Close"].iloc[-1]),
        triggered=False,
    )


# -----------------------------
# Yahoo fetch (quiet, safe)
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
# CLI scan (optional)
# -----------------------------
def scan_once(
    tickers: list[str],
    *,
    interval: str = "5m",
    period: str = "5d",
    vol_lookback: int = 20,
    # detection tuning
    spike_min: float = 2.5,
    recent_bars: int = 6,
    price_window: int = 4,
    min_move_pct: float = 0.02,
    vol_confirm_mult: float = 1.3,
    # filters
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

        # freshness gate only (do NOT truncate before detection)
        df_recent = filter_last_minutes(df, minutes=last_minutes)
        if df_recent.empty:
            continue  # stale / market closed / delayed too far

        plan = compute_trade_plan_from_signal(
            sym,
            display_name(sym),
            df,  # IMPORTANT: full df for detection
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

        plan.triggered = plan.last_price >= plan.entry
        plans.append(plan)

    if not plans:
        return pd.DataFrame()

    return pd.DataFrame([p.__dict__ for p in plans])


# -----------------------------
# CLI entrypoint (optional)
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spike_min", type=float, default=2.5)
    parser.add_argument("--recent_bars", type=int, default=6)
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
        # show key columns
        cols = ["name", "symbol", "signal_type", "spike_time", "spike_mult", "entry", "stop", "t1", "t2", "t3", "turnover_egp"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].sort_values(["signal_type", "spike_mult"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
