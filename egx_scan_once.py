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
    Used by the Streamlit app.
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
# Trade plan
# -----------------------------
@dataclass
class TradePlan:
    symbol: str
    name: str
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


def compute_trade_plan_from_spike(
    symbol: str,
    name: str,
    df: pd.DataFrame,
    *,
    vol_lookback: int = 20,
    spike_mult: float = 3.0,
    require_green_bar: bool = True,
    entry_buffer_pct: float = 0.0005,
    atr_n: int = 14,
    stop_buffer_atr: float = 0.25,
) -> Optional[TradePlan]:

    df = ensure_ohlcv(df)
    if df.empty:
        return None

    min_len = max(vol_lookback + 3, atr_n + 3)
    if len(df) < min_len:
        return None

    work = df.copy()
    work["vol_sma"] = work["Volume"].rolling(vol_lookback, min_periods=vol_lookback).mean()
    work["atr"] = atr(work, n=atr_n)

    i = last_completed_bar_index(work)
    bar = work.iloc[i]

    if pd.isna(bar["vol_sma"]) or bar["vol_sma"] <= 0 or pd.isna(bar["atr"]):
        return None

    if bar["Volume"] <= spike_mult * bar["vol_sma"]:
        return None

    if require_green_bar and not (bar["Close"] > bar["Open"]):
        return None

    spike_high = float(bar["High"])
    spike_low = float(bar["Low"])
    spike_atr = float(bar["atr"])

    entry = spike_high * (1.0 + entry_buffer_pct)
    stop = spike_low - stop_buffer_atr * spike_atr
    if stop >= entry:
        return None

    R = entry - stop

    return TradePlan(
        symbol=symbol,
        name=name,
        spike_time=work.index[i],
        spike_mult=float(bar["Volume"] / bar["vol_sma"]),
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
    spike_mult: float = 3.0,
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

        df = filter_last_minutes(df, minutes=last_minutes)
        if df.empty:
            continue

        plan = compute_trade_plan_from_spike(
            sym,
            display_name(sym),
            df,
            vol_lookback=vol_lookback,
            spike_mult=spike_mult,
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
