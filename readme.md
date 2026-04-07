# 💱 Smart Currency Converter & AI Predictor

An interactive web application built with **Python** and **Streamlit** that provides real-time currency conversion and uses **Machine Learning** to forecast future exchange rate trends.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)

---

## 🚀 Features

- **Real-Time Conversion** — Convert between USD, INR, EUR, and GBP using live market data from Yahoo Finance.
- **Smart Currency Lock** — Prevents selecting the same currency for both "From" and "To" fields.
- **24-Hour Change Tracker** — Shows how much the exchange rate moved in the last day.
- **Market Overview** — Displays 30-day low, high, average, and volatility statistics.
- **AI-Powered 7-Day Forecast** — Uses a Random Forest Regressor trained on 2 years of historical data to predict the next 7 days.
- **Model Transparency** — Displays model performance metrics (MAE, RMSE, R² Score) so you know how reliable the predictions are.
- **Feature Importance Chart** — Shows which factors the AI considers most important for its predictions.
- **Interactive Charts** — Powered by Plotly for smooth, zoomable, hover-able visualizations.
- **Professional Dark UI** — Sleek gradient-based dark theme with glassmorphism styling.

---

## 🛠️ Tech Stack

| Component         | Technology                                                     |
|-------------------|----------------------------------------------------------------|
| **Language**      | Python 3.10+                                                   |
| **Frontend**      | [Streamlit](https://streamlit.io/)                             |
| **Data Source**   | [yfinance](https://pypi.org/project/yfinance/) (Yahoo Finance)|
| **ML Model**      | Scikit-Learn — Random Forest Regressor                         |
| **Charts**        | Plotly (interactive graphs)                                    |
| **Data Handling** | Pandas & NumPy                                                 |

---

## 📂 Project Structure

```
ExchangeRatePrediction/
├── app.py                  # Main Streamlit dashboard (entry point)
├── requirements.txt        # Python dependencies with version pins
├── README.md               # This file — project documentation
├── models/
│   └── model.pkl           # Saved trained model (auto-generated)
└── src/                    # Source modules (modular architecture)
    ├── __init__.py          # Makes src/ a Python package
    ├── data_loader.py       # Fetches live & historical data from Yahoo Finance
    ├── preprocess.py        # Feature engineering (MA, volatility, returns)
    ├── train.py             # Model training with evaluation metrics
    └── predictor.py         # Generates future exchange rate predictions
```

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ExchangeRatePrediction.git
cd ExchangeRatePrediction
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🧠 How the AI Prediction Works

### Step 1: Data Collection
The app fetches **2 years of daily exchange rate data** from Yahoo Finance using the `yfinance` library.

### Step 2: Feature Engineering
Raw price data is transformed into meaningful features:

| Feature        | What It Measures                          |
|----------------|-------------------------------------------|
| `MA_7`         | 7-day moving average (short-term trend)   |
| `MA_30`        | 30-day moving average (long-term trend)   |
| `Daily_Return` | Day-over-day percentage change            |
| `Volatility_7` | 7-day rolling standard deviation          |
| `Day_of_Week`  | Monday (0) through Friday (4)             |
| `Day_Num`      | Sequential day counter for trend modeling |

### Step 3: Model Training
A **Random Forest Regressor** is trained on 80% of the data and tested on 20%:
- **100 decision trees** work together (ensemble method)
- Each tree sees a different random subset of the data
- The final prediction = average of all tree predictions
- This reduces overfitting and improves accuracy

### Step 4: Evaluation
The model is evaluated using three metrics:
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Square Error): Penalizes larger errors
- **R² Score**: How well the model explains variance (1.0 = perfect)

### Step 5: Forecasting
The model extrapolates features for the **next 7 days** and predicts future exchange rates.

---

## ⚠️ Disclaimer

AI predictions are based on **historical patterns** and should **NOT** be used for actual financial decisions. Exchange rates are influenced by many unpredictable factors including geopolitical events, central bank policies, and market sentiment.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


to run project : python -m streamlit run app.py
