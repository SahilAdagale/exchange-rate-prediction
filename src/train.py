"""
train.py — Model training and evaluation for exchange rate prediction.

This module handles:
1. Splitting data into training and test sets
2. Training a RandomForestRegressor model
3. Evaluating model performance (MAE, RMSE, R² Score)
4. Optionally saving the trained model to disk

Why RandomForestRegressor?
- It's an ENSEMBLE method: combines 100+ decision trees
- Each tree sees a random subset of the data (reduces overfitting)
- Final prediction = average of all trees (more stable than a single tree)
- Handles non-linear relationships well (unlike Linear Regression)
- Doesn't require feature scaling (unlike neural networks)

Note on Time-Series:
- Random Forests cannot extrapolate beyond training data ranges.
- We use a random shuffle split (not chronological) for evaluation,
  since our features (MA, volatility, returns) are relative/normalized
  and carry signal regardless of position in time.
- The final production model is then trained on ALL data for best predictions.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import from our own modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import get_feature_columns


def train_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    n_estimators: int = 100,
    random_state: int = 42
) -> tuple:
    """
    Train a RandomForestRegressor on preprocessed exchange rate data.

    Training strategy:
    1. Split data 80/20 randomly for evaluation metrics
    2. Compute MAE, RMSE, R² on the test portion
    3. Retrain the final model on ALL data for the best predictions

    Args:
        df: Preprocessed DataFrame (output of preprocess.engineer_features)
        test_size: Fraction of data to use for testing (0.0 to 1.0)
        n_estimators: Number of trees in the Random Forest
        random_state: Seed for reproducibility

    Returns:
        Tuple of (trained_model, metrics_dict)
        metrics_dict contains: MAE, RMSE, R2, train_size, test_size
    """
    feature_cols = get_feature_columns()

    # Separate features (X) and target (y)
    X = df[feature_cols].values
    y = df['Close'].values.ravel()  # ravel() ensures 1D array

    # --- Step 1: Evaluate with a random split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    eval_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    eval_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = eval_model.predict(X_test)
    metrics = {
        'MAE': round(mean_absolute_error(y_test, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        'R2': round(r2_score(y_test, y_pred), 4),
        'Train_Samples': len(X_train),
        'Test_Samples': len(X_test),
    }

    # --- Step 2: Train final model on ALL data for best predictions ---
    final_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    final_model.fit(X, y)

    return final_model, metrics


def save_model(model, filepath: str = "models/model.pkl"):
    """Save a trained model to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✅ Model saved to {filepath}")


def load_model(filepath: str = "models/model.pkl"):
    """Load a trained model from disk."""
    if os.path.exists(filepath):
        return joblib.load(filepath)
    return None


if __name__ == "__main__":
    # Full training pipeline test
    from data_loader import fetch_historical_data, build_ticker_symbol
    from preprocess import engineer_features

    ticker = build_ticker_symbol("USD", "INR")
    raw_data = fetch_historical_data.__wrapped__(ticker)
    processed_data = engineer_features(raw_data)

    model, metrics = train_model(processed_data)

    print("\n📊 Model Performance Metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value}")

    save_model(model)
