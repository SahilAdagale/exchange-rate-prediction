import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(page_title="Currency AI", layout="centered")

st.title("💰 Smart Currency Converter")
st.markdown("Convert currencies and see the AI trend prediction below.")

# --- 1. Simple Input UI ---
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    base = st.selectbox("From", ["USD", "INR", "EUR", "GBP"])
with col2:
    st.markdown("<h3 style='text-align: center;'>➡️</h3>", unsafe_allow_html=True)
with col3:
    target = st.selectbox("To", ["INR", "USD", "EUR", "GBP"])

amount = st.number_input("Amount to convert:", value=1.0)

# --- 2. Data Fetching ---
symbol = f"{base}{target}=X"

@st.cache_data(ttl=3600)
def get_live_data(ticker):
    data = yf.download(ticker, period="1mo", interval="1d")
    return data

data = get_live_data(symbol)

if not data.empty:
    # Get current rate
    current_rate = float(data['Close'].iloc[-1])
    converted_amount = amount * current_rate
    
    # --- 3. The "Big Result" ---
    st.success(f"### {amount} {base} = {converted_amount:.2f} {target}")
    st.info(f"Current Exchange Rate: 1 {base} = {current_rate:.4f} {target}")

    # --- 4. Prediction "Hidden" in an Expander ---
    with st.expander("Show AI Price Prediction (Next 7 Days)"):
        st.write("Processing historical trends...")
        
        # Prepare ML Data
        data_ml = yf.download(symbol, period="2y")
        data_ml.reset_index(inplace=True)
        data_ml['Day_Num'] = range(len(data_ml))
        
        X = data_ml[['Day_Num']]
        y = data_ml['Close']
        
        model = RandomForestRegressor(n_estimators=50)
        model.fit(X.values, y.values.ravel())
        
        # Predict 7 days
        last_day = data_ml['Day_Num'].iloc[-1]
        future_days = [[last_day + i] for i in range(1, 8)]
        preds = model.predict(future_days)
        
        # Display as a clean table instead of a messy graph
        future_dates = [(datetime.now() + timedelta(days=i)).date() for i in range(1, 8)]
        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Rate": preds})
        st.table(forecast_df)
else:
    st.error("Could not find data for this pair. Try USD to INR.")