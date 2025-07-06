import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import ta  # For technical indicators

# ===== Load model and scaler =====
model = load_model("model/model.h5")
scaler = joblib.load("model/scaler.save")

# ===== Fetch and process stock data =====
def fetch_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")

    if df.empty:
        return pd.DataFrame()

    # Rename columns to match training
    df.rename(columns={
        "Open": "Open_yfin",
        "High": "High_yfin",
        "Low": "Low_yfin",
        "Close": "Close_yfin",
        "Volume": "Volume_yfin"
    }, inplace=True)

    # Compute technical indicators
    df["SMA_10"] = ta.trend.sma_indicator(df["Close_yfin"], window=10)
    df["EMA_20"] = ta.trend.ema_indicator(df["Close_yfin"], window=20)
    df["RSI_14"] = ta.momentum.rsi(df["Close_yfin"], window=14)
    df["MACD_Line"] = ta.trend.macd(df["Close_yfin"])
    df["SMA_30"] = ta.trend.sma_indicator(df["Close_yfin"], window=30)
    df["EMA_50"] = ta.trend.ema_indicator(df["Close_yfin"], window=50)
    df["Return"] = df["Close_yfin"].pct_change()
    df["Volatility"] = df["Return"].rolling(window=10).std()

    df.dropna(inplace=True)
    return df

# ===== Preprocess latest 60 days =====
def preprocess(df):
    features = ["Close_yfin", "Open_yfin", "High_yfin", "Low_yfin", "Volume_yfin", 
                "SMA_10", "EMA_20", "RSI_14", "MACD_Line", "SMA_30", "EMA_50", "Return", "Volatility"]

    df = df[features].dropna()
    last_60 = df[-60:].values
    X_scaled = scaler.transform(last_60)
    return np.expand_dims(X_scaled, axis=0), df["Close_yfin"].iloc[-1], df.iloc[-1][features].values  # last row for reconstruction

# ===== Streamlit UI =====
st.set_page_config(page_title="Stock LSTM App", layout="centered")
st.title("📈 Real-Time Stock Predictor & Investment Simulator")

ticker = st.text_input("Enter stock symbol (e.g. AAPL):", "AAPL")
investment = st.number_input("💸 Investment Amount ($):", value=1000.0)

# ===== Predict button =====
if st.button("Predict Next Day Price"):
    with st.spinner("Fetching data and predicting..."):
        try:
            df = fetch_data(ticker)
            if df.empty:
                st.error("No data returned for this ticker. Please check the symbol.")
            else:
                X_input, last_price, last_features = preprocess(df)

                predicted_scaled = model.predict(X_input).flatten()[0]

                # Reconstruct a full scaled row and update predicted close
                last_scaled = scaler.transform([last_features])
                last_scaled[0][0] = predicted_scaled  # Replace only Close

                predicted_full = scaler.inverse_transform(last_scaled)
                predicted_price = predicted_full[0][0]

                # ===== Output section =====
                st.success("✅ Prediction Complete")
                st.metric("📉 Last Close Price", f"${last_price:.2f}")
                st.metric("📈 Predicted Next Price", f"${predicted_price:.2f}")

                profit = (predicted_price - last_price) * (investment / last_price)
                st.subheader("💰 Investment Simulation")
                st.write(f"If you invest **${investment:.2f}** now:")
                st.write(f"**Predicted Value Tomorrow:** ${investment + profit:.2f}")
                if profit > 0:
                    st.success(f"📈 Estimated Profit: ${profit:.2f}")
                else:
                    st.error(f"📉 Estimated Loss: ${-profit:.2f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ===== Historical Chart =====
st.subheader(f"📊 Historical Close Price for {ticker}")
chart_data = fetch_data(ticker)
if not chart_data.empty:
    st.line_chart(chart_data["Close_yfin"][-60:])
else:
    st.warning("Could not load chart. Please check the stock symbol.")
