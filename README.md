
# EGX Intraday Scanner

A focused intraday volume-spike scanner for EGX30 stocks built with Streamlit.

The application scans a predefined watchlist, ranks candidates by relative volume expansion, builds structured trade levels, and generates AI-assisted trade commentary.

# Features

5-minute intraday data

Volume spike detection (vs 20-bar SMA)

Automatic Entry / Stop / T1 / T2 / T3 levels

Top 10 ranked candidates

Turnover-based prioritization

Live vs stale session detection

AI-generated structured trade recommendations

# Clean bucket classification:
✅ Strong (≥ 3×)
⚠️ Watchlist (2–3×)
👀 Warm (< 2×)
Strategy Logic
Spike Ratio
Last Completed Bar Volume / SMA(Volume, 20)


# Risk Model

R = Entry − Stop
T1 = Entry + 1R
T2 = Entry + 2R
T3 = Entry + 3R


# Ranking prioritizes:

Bucket strength
Spike ratio magnitude
Turnover (EGP)

# Dependencies
Streamlit
Pandas
NumPy
yfinance
Matplotlib
OpenAI

# Notes

Market data sourced from Yahoo Finance (may be delayed).
Manual execution — no auto-trading.
Designed for personal intraday analysis.
Not financial advice.
