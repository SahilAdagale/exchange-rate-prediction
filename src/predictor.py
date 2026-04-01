"""
predictor.py — Generate future exchange rate predictions.

This module takes a trained model and the most recent data, then
extrapolates features forward to predict exchange rates for the
next N days.

How prediction works:
1. Take the last known data point
2. Estimate what each feature would be in the future:
   - MA_7 / MA_30: Use last known values (trend continuation assumption)
   - Daily_Return: Use average of recent returns
   - Volatility_7: Use last known volatility
   - Day_of_Week: Calculate actual weekdays
   - Day_Num: Increment sequentially
3. Feed estimated features into the model
4. Return predicted rates for each future day
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import get_feature_columns


def predict_future(model, df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """
    Predict exchange rates for the next N days.

    Uses the trained model and extrapolated features from the most
    recent historical data to forecast future rates.

    Args:
        model: Trained sklearn model (e.g., RandomForestRegressor)
        df: Preprocessed DataFrame (with all feature columns)
        days: Number of days to predict into the future

    Returns:
        DataFrame with columns: Date, Predicted_Rate
    """
    feature_cols = get_feature_columns()

    # Get the most recent values to extrapolate from
    last_row = df.iloc[-1]
    last_day_num = last_row['Day_Num']
    last_ma7 = last_row['MA_7']
    last_ma30 = last_row['MA_30']
    avg_return = df['Daily_Return'].tail(30).mean()  # Average recent return
    last_volatility = last_row['Volatility_7']
    last_close = float(last_row['Close'])

    future_dates = []
    predictions = []
    future_features = []

    for i in range(1, days + 1):
        future_date = (datetime.now() + timedelta(days=i)).date()
        future_day_num = last_day_num + i
        future_dow = future_date.weekday()

        # Estimate future feature values
        # MA_7 and MA_30 drift slightly with the predicted trend
        estimated_ma7 = last_ma7 * (1 + avg_return * i * 0.5)
        estimated_ma30 = last_ma30 * (1 + avg_return * i * 0.2)

        features = {
            'MA_7': estimated_ma7,
            'MA_30': estimated_ma30,
            'Daily_Return': avg_return,
            'Volatility_7': last_volatility,
            'Day_of_Week': future_dow,
            'Day_Num': future_day_num,
        }

        future_features.append([features[col] for col in feature_cols])
        future_dates.append(future_date)

    # Make predictions
    predictions = model.predict(np.array(future_features))

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Rate': [round(float(p), 4) for p in predictions]
    })

    return forecast_df


if __name__ == "__main__":
    from data_loader import fetch_historical_data, build_ticker_symbol
    from preprocess import engineer_features
    from train import train_model

    ticker = build_ticker_symbol("USD", "INR")
    raw = fetch_historical_data.__wrapped__(ticker)
    processed = engineer_features(raw)
    model, metrics = train_model(processed)

    forecast = predict_future(model, processed, days=7)
    print("\n📅 7-Day Forecast:")
    print(forecast.to_string(index=False))
