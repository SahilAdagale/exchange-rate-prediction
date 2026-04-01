"""
preprocess.py — Feature engineering for exchange rate prediction.

This module transforms raw price data into meaningful features
that the ML model can use to learn patterns. Features include:

- Moving Averages (MA_7, MA_30): Smoothed trend indicators
- Daily Return: Day-over-day percentage change
- Volatility (7-day): Rolling standard deviation of returns
- Day of Week: Captures weekly seasonality
- Day Number: Sequential counter for long-term trend detection

Why these features?
- Moving averages help the model understand SHORT and LONG term trends.
- Volatility tells the model how "jumpy" the market has been recently.
- Day of week captures patterns like "markets move differently on Mondays."
- Day number helps model the overall direction (uptrend or downtrend).
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ML-ready features from raw exchange rate data.

    Input DataFrame must have a 'Close' column and a 'Date' column.

    Feature engineering steps:
    1. MA_7:  7-day Simple Moving Average of Close prices
    2. MA_30: 30-day Simple Moving Average of Close prices
    3. Daily_Return: (Today's Close - Yesterday's Close) / Yesterday's Close
    4. Volatility_7: Rolling 7-day standard deviation of Daily_Return
    5. Day_of_Week: Monday=0, Tuesday=1, ..., Friday=4
    6. Day_Num: Sequential day number starting from 0

    Args:
        df: DataFrame with at least 'Close' and 'Date' columns

    Returns:
        DataFrame with all original columns plus engineered features.
        Rows with NaN values (from rolling calculations) are dropped.
    """
    df = df.copy()  # Don't modify the original

    # Ensure Close is a proper numeric Series (flatten if needed)
    if isinstance(df['Close'], pd.DataFrame):
        df['Close'] = df['Close'].iloc[:, 0]
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    # --- Moving Averages ---
    # MA_7: Short-term trend (reacts quickly to price changes)
    df['MA_7'] = df['Close'].rolling(window=7).mean()

    # MA_30: Long-term trend (smooths out daily noise)
    df['MA_30'] = df['Close'].rolling(window=30).mean()

    # --- Returns & Volatility ---
    # Daily percentage change: how much did the rate move today?
    df['Daily_Return'] = df['Close'].pct_change()

    # 7-day rolling standard deviation of returns = recent volatility
    df['Volatility_7'] = df['Daily_Return'].rolling(window=7).std()

    # --- Calendar Features ---
    if 'Date' in df.columns:
        df['Day_of_Week'] = pd.to_datetime(df['Date']).dt.dayofweek
    else:
        df['Day_of_Week'] = 0  # fallback

    # Sequential day counter for trend modeling
    df['Day_Num'] = range(len(df))

    # Drop rows where rolling calculations produced NaN
    df = df.dropna()

    return df


def get_feature_columns() -> list:
    """
    Return the list of feature column names used for model training.

    This is defined in one place so both train.py and predictor.py
    use the exact same features — preventing mismatch bugs.

    Returns:
        List of feature column name strings.
    """
    return ['MA_7', 'MA_30', 'Daily_Return', 'Volatility_7', 'Day_of_Week', 'Day_Num']


if __name__ == "__main__":
    # Quick test with sample data
    import yfinance as yf

    data = yf.download("USDINR=X", period="6mo", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)

    processed = engineer_features(data)
    print(f"Original shape: {data.shape}")
    print(f"Processed shape: {processed.shape}")
    print(f"\nFeature columns: {get_feature_columns()}")
    print(f"\nSample rows:")
    print(processed[['Date', 'Close'] + get_feature_columns()].tail())
