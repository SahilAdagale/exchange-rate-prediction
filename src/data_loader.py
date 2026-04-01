"""
data_loader.py — Fetches exchange rate data from Yahoo Finance.

This module handles all communication with the yfinance API.
It provides functions to fetch both short-term (live) and
long-term (historical) data for any supported currency pair.

Supported pairs: USD, INR, EUR, GBP (any combination)
Data source: Yahoo Finance via yfinance library
"""

import yfinance as yf
import pandas as pd
import streamlit as st


def build_ticker_symbol(base: str, target: str) -> str:
    """
    Build a Yahoo Finance ticker symbol from base and target currencies.

    Yahoo Finance uses the format 'USDEUR=X' for forex pairs.

    Args:
        base: Base currency code (e.g., 'USD')
        target: Target currency code (e.g., 'INR')

    Returns:
        Ticker symbol string (e.g., 'USDINR=X')
    """
    return f"{base}{target}=X"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_data(ticker: str, period: str = "1mo") -> pd.DataFrame:
    """
    Fetch recent exchange rate data (default: last 1 month).

    Used for displaying the current exchange rate and short-term trends.
    Results are cached for 1 hour (ttl=3600 seconds) to minimize API calls.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g., 'USDINR=X')
        period: Data period — '1mo', '3mo', '6mo', '1y', etc.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index: Date
    """
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        if data.empty:
            return pd.DataFrame()
        # Flatten multi-level columns if present (yfinance sometimes returns multi-index)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        st.error(f"⚠️ Failed to fetch live data: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch long-term historical data for model training (default: 2 years).

    This data is used to train the ML model for exchange rate prediction.
    More data gives the model more patterns to learn from.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g., 'USDINR=X')
        period: Data period — '1y', '2y', '5y', 'max', etc.

    Returns:
        DataFrame with Date as a column (not index), plus OHLCV columns.
    """
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return pd.DataFrame()
        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"⚠️ Failed to fetch historical data: {str(e)}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Quick test — run this file directly to verify data loading works
    ticker = build_ticker_symbol("USD", "INR")
    print(f"Ticker: {ticker}")

    df_live = fetch_live_data.__wrapped__(ticker)  # bypass cache for testing
    print(f"\n--- Live Data (last 5 days) ---")
    print(df_live.tail())

    df_hist = fetch_historical_data.__wrapped__(ticker)
    print(f"\n--- Historical Data Shape: {df_hist.shape} ---")
    print(df_hist.head())
