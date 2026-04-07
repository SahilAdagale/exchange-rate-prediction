"""
app.py — Main Streamlit Application for Smart Currency Converter & AI Predictor.

This is the entry point for the application. It creates a professional
dashboard with:
  1. Real-time currency conversion
  2. Historical exchange rate chart
  3. AI-powered 7-day price forecast
  4. Model performance metrics

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add src/ to the Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_loader import build_ticker_symbol, fetch_live_data, fetch_historical_data
from preprocess import engineer_features
from train import train_model
from predictor import predict_future


# ──────────────────────────────────────────────
# PAGE CONFIG & CUSTOM STYLING
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Currency AI",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    }

    /* Cards / metric containers */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }

    /* Success/info boxes */
    .stAlert {
        border-radius: 12px;
    }

    /* Headings */
    h1 {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Tables */
    .stTable {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2, #667eea);
        transform: scale(1.02);
    }

    /* Custom metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
        margin-top: 4px;
    }

    /* Divider */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        margin: 24px 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    CURRENCIES = ["USD", "INR", "EUR", "GBP"]

    base = st.selectbox("📤 From Currency", CURRENCIES, index=0)

    # Filter out the base currency from target options to prevent same-pair
    target_options = [c for c in CURRENCIES if c != base]
    target = st.selectbox("📥 To Currency", target_options, index=0)

    amount = st.number_input(
        "💵 Amount",
        min_value=0.01,
        value=1.0,
        step=1.0,
        format="%.2f"
    )

    hist_period = st.selectbox(
        "📅 Historical Data Period",
        ["1y", "2y", "5y"],
        index=1,
        help="More data improves AI predictions but takes longer to load."
    )

    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    This app uses **Machine Learning** (Random Forest) 
    to predict exchange rate trends.
    
    - Data: Yahoo Finance  
    - Model: Random Forest Regressor  
    - Features: Moving Averages, Volatility, Returns  
    """)
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit & Scikit-Learn")


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────

st.markdown("# 💱 Smart Currency Converter & AI Predictor")
st.markdown(
    '<p style="color: rgba(255,255,255,0.6); font-size: 1.1rem;">'
    'Real-time conversion with AI-powered 7-day forecasting'
    '</p>',
    unsafe_allow_html=True
)
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LIVE CONVERSION
# ──────────────────────────────────────────────

ticker = build_ticker_symbol(base, target)

with st.spinner("🔄 Fetching live exchange rates..."):
    live_data = fetch_live_data(ticker)

if live_data.empty:
    st.error(
        f"❌ Could not fetch data for **{base} → {target}**. "
        "Please check your internet connection or try a different currency pair."
    )
    st.stop()

# Extract current rate
current_rate = float(live_data['Close'].iloc[-1])
converted_amount = amount * current_rate

# Calculate 24h change
if len(live_data) >= 2:
    prev_rate = float(live_data['Close'].iloc[-2])
    daily_change = ((current_rate - prev_rate) / prev_rate) * 100
else:
    daily_change = 0.0

# Display conversion result
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{amount:,.2f} {base}</div>
        <div class="metric-label">You Pay</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-color: #667eea;">
        <div class="metric-value">{converted_amount:,.4f} {target}</div>
        <div class="metric-label">You Get</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    change_color = "#4ade80" if daily_change >= 0 else "#f87171"
    change_arrow = "▲" if daily_change >= 0 else "▼"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="background: {change_color}; -webkit-background-clip: text;">
            {change_arrow} {abs(daily_change):.3f}%
        </div>
        <div class="metric-label">24h Change</div>
    </div>
    """, unsafe_allow_html=True)

# Exchange rate info
st.info(f"📊 **Live Rate:** 1 {base} = {current_rate:.4f} {target} &nbsp;&nbsp;|&nbsp;&nbsp; Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HISTORICAL STATS
# ──────────────────────────────────────────────

st.markdown("## 📈 Market Overview")

# Show quick stats from the live data (1 month)
rate_min = float(live_data['Close'].min())
rate_max = float(live_data['Close'].max())
rate_avg = float(live_data['Close'].mean())
rate_std = float(live_data['Close'].std())

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
stat_col1.metric("📉 30-Day Low", f"{rate_min:.4f}")
stat_col2.metric("📈 30-Day High", f"{rate_max:.4f}")
stat_col3.metric("📊 30-Day Average", f"{rate_avg:.4f}")
stat_col4.metric("📐 Volatility (Std)", f"{rate_std:.4f}")


# ──────────────────────────────────────────────
# AI PREDICTION SECTION
# ──────────────────────────────────────────────

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown("## 🤖 AI Price Prediction")

with st.spinner("🧠 Training AI model on historical data..."):
    # Step 1: Fetch historical data
    hist_data = fetch_historical_data(ticker, period=hist_period)

    if hist_data.empty:
        st.error("Could not fetch historical data for model training.")
        st.stop()

    # Step 2: Feature engineering
    processed_data = engineer_features(hist_data)

    if len(processed_data) < 60:
        st.warning("⚠️ Not enough historical data to train a reliable model. Try selecting a longer period.")
        st.stop()

    # Step 3: Train model
    model, metrics = train_model(processed_data)

    # Step 4: Predict future
    forecast_df = predict_future(model, processed_data, days=7)

# ── Model Metrics ──
st.markdown("### 📊 Model Performance")
met_col1, met_col2, met_col3 = st.columns(3)
met_col1.metric("MAE (Mean Absolute Error)", f"{metrics['MAE']:.4f}", help="Average prediction error in rate units. Lower is better.")
met_col2.metric("RMSE (Root Mean Sq Error)", f"{metrics['RMSE']:.4f}", help="Penalizes large errors more. Lower is better.")
met_col3.metric("R² Score", f"{metrics['R2']:.4f}", help="How well the model explains variance. 1.0 is perfect.")

st.caption(f"Trained on {metrics['Train_Samples']} days | Tested on {metrics['Test_Samples']} days")


# ── Interactive Chart ──
st.markdown("### 📉 Historical Trend + 7-Day Forecast")

fig = go.Figure()

# Plot last 60 days of actual data
recent_data = processed_data.tail(60).copy()
recent_data['Date'] = pd.to_datetime(recent_data['Date'])

fig.add_trace(go.Scatter(
    x=recent_data['Date'],
    y=recent_data['Close'],
    mode='lines+markers',
    name='Actual Rate',
    line=dict(color='#667eea', width=2),
    marker=dict(size=4),
    hovertemplate='%{x|%b %d}<br>Rate: %{y:.4f}<extra></extra>'
))

# Plot predictions — convert dates to datetime for consistency
forecast_plot = forecast_df.copy()
forecast_plot['Date'] = pd.to_datetime(forecast_plot['Date'])

fig.add_trace(go.Scatter(
    x=forecast_plot['Date'],
    y=forecast_plot['Predicted_Rate'],
    mode='lines+markers',
    name='AI Prediction',
    line=dict(color='#f59e0b', width=2, dash='dash'),
    marker=dict(size=6, symbol='diamond'),
    hovertemplate='%{x|%b %d}<br>Predicted: %{y:.4f}<extra></extra>'
))

# Add a vertical "Today" marker using shapes (more reliable than add_vline)
last_actual_dt = recent_data['Date'].iloc[-1].to_pydatetime()
fig.add_shape(
    type="line",
    x0=last_actual_dt, x1=last_actual_dt,
    y0=0, y1=1,
    yref="paper",
    line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
)
fig.add_annotation(
    x=last_actual_dt, y=1.05, yref="paper",
    text="Today", showarrow=False,
    font=dict(color="rgba(255,255,255,0.5)", size=11)
)

fig.update_layout(
    template='plotly_dark',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='rgba(255,255,255,0.8)'),
    xaxis_title="Date",
    yaxis_title=f"Exchange Rate ({base}/{target})",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=20, r=20, t=40, b=20),
    height=450,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)


# ── Forecast Table ──
st.markdown("### 📅 7-Day Forecast Table")

# Add day names and trend indicators
forecast_display = forecast_df.copy()
forecast_display['Day'] = pd.to_datetime(forecast_display['Date']).dt.strftime('%A')
forecast_display['Predicted_Rate'] = forecast_display['Predicted_Rate'].apply(lambda x: f"{x:.4f}")

# Add trend vs current rate
forecast_display['vs Current'] = forecast_df['Predicted_Rate'].apply(
    lambda x: f"{'▲' if x > current_rate else '▼'} {abs(x - current_rate):.4f}"
)

forecast_display = forecast_display[['Date', 'Day', 'Predicted_Rate', 'vs Current']]
forecast_display.columns = ['📅 Date', '📆 Day', f'💰 Predicted {target}', f'📊 vs Current Rate']

st.dataframe(
    forecast_display,
    use_container_width=True,
    hide_index=True
)




# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: rgba(255,255,255,0.4); font-size: 0.85rem;">'
    '⚠️ Disclaimer: AI predictions are based on historical patterns and should NOT be used for actual financial decisions. '
    'Exchange rates are influenced by many unpredictable factors.'
    '</p>',
    unsafe_allow_html=True
)