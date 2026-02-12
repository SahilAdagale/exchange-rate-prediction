# 💱 Smart Currency Converter & AI Predictor

An interactive web application built with Python and Streamlit that provides real-time currency conversion and uses Machine Learning to forecast future exchange rate trends.

## 🚀 Features
- **Real-Time Conversion:** Convert between USD, INR, EUR, and GBP using live market data.
- **Bidirectional Support:** Easily switch between "Base" and "Target" currencies (e.g., INR to USD or USD to INR).
- **AI Forecasting:** Uses a Random Forest Regressor model to predict exchange rates for the next 7 days based on historical trends.
- **Interactive UI:** Clean, modern dashboard with expandable sections to hide complex data until needed.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Data Source:** [yfinance](https://pypi.org/project/yfinance/) (Yahoo Finance API)
- **Machine Learning:** Scikit-Learn (Random Forest Regressor)
- **Data Handling:** Pandas & NumPy
- **Visualization:** Matplotlib

## 📂 Project Structure
```text
ExchangeRatePrediction/
├── app.py              # Main Streamlit application code
├── requirements.txt    # List of required Python libraries
├── README.md           # Project documentation
└── src/                # Source folder for background logic
    ├── data_loader.py  # Script for fetching data
    ├── preprocess.py   # Data cleaning and feature engineering
    └── train.py        # Model training logic