import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import ta

# ===== Load model and scaler =====
model = load_model("model/model.h5")
scaler = joblib.load("model/scaler.save")

# ===== Fetch stock data =====
def fetch_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=False)

    if df.empty:
        raise ValueError("⚠️ Could not fetch data. Please check the stock symbol.")

    if "Close" not in df.columns:
        raise ValueError("⚠️ 'Close' column missing from data.")

    df.rename(columns={
        "Open": "Open_yfin",
        "High": "High_yfin",
        "Low": "Low_yfin",
        "Close": "Close_yfin",
        "Volume": "Volume_yfin"
    }, inplace=True)

    try:
    df["SMA_10"] = ta.trend.sma_indicator(df["Close_yfin"], window=10)
    df["EMA_20"] = ta.trend.ema_indicator(df["Close_yfin"], window=20)
    df["RSI_14"] = ta.momentum.rsi(df["Close_yfin"], window=14)
    df["MACD_Line"] = ta.trend.macd(df["Close_yfin"])
    df["SMA_30"] = ta.trend.sma_indicator(df["Close_yfin"], window=30)
    df["EMA_50"] = ta.trend.ema_indicator(df["Close_yfin"], window=50)
    df["Return"] = df["Close_yfin"].pct_change()
    df["Volatility"] = df["Return"].rolling(window=10).std()
except Exception as e:
    print("Indicator error:", e)
    raise ValueError("⚠️ Indicator computation failed. Please try a different stock.")

    df.dropna(inplace=True)
    return df

# ===== Preprocess for LSTM =====
def preprocess(df):
    features = ["Close_yfin", "Open_yfin", "High_yfin", "Low_yfin", "Volume_yfin",
                "SMA_10", "EMA_20", "RSI_14", "MACD_Line", "SMA_30", "EMA_50", "Return", "Volatility"]

    df = df[features].dropna()
    last_60 = df[-60:].values
    X_scaled = scaler.transform(last_60)
    return np.expand_dims(X_scaled, axis=0), df["Close_yfin"].iloc[-1]

# ===== Streamlit UI =====
st.set_page_config(page_title="Stock LSTM Predictor", layout="centered")
st.title("📈 Real-Time Stock Predictor & Investment Simulator")

ticker = st.text_input("Enter stock symbol (e.g. AAPL, TSLA, INFY.NS):", "AAPL")
investment = st.number_input("💸 Investment Amount ($):", value=1000.0)

# Store fetched dataframe globally
df = None

if st.button("Predict Next Day Price"):
    with st.spinner("Fetching data and predicting..."):
        try:
            df = fetch_data(ticker)
            X_input, last_price = preprocess(df)

            predicted_scaled = model.predict(X_input)[0][0]

            # Build dummy array to inverse transform
            dummy = np.zeros((1, scaler.n_features_in_))
            dummy[0][0] = predicted_scaled
            predicted_price = scaler.inverse_transform(dummy)[0][0]

            # Show results
            st.success("Prediction Complete ✅")
            st.metric("📉 Last Close Price", f"${last_price:.2f}")
            st.metric("📈 Predicted Next Price", f"${predicted_price:.2f}")

            # Investment simulation
            profit = (predicted_price - last_price) * (investment / last_price)
            st.subheader("💰 Investment Simulation")
            st.write(f"If you invest **${investment:.2f}** now:")
            st.write(f"**Predicted Value Tomorrow:** ${investment + profit:.2f}")

            if profit > 0:
                st.success(f"📈 Estimated Profit: ${profit:.2f}")
            else:
                st.error(f"📉 Estimated Loss: ${-profit:.2f}")

        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ===== Historical Chart =====
if df is not None:
    st.subheader(f"📊 Historical Close Price for {ticker}")
    try:
        st.line_chart(df["Close_yfin"][-60:])
    except Exception as e:
        st.warning(f"Could not load chart: {e}")


    try:
        df["SMA_10"] = ta.trend.sma_indicator(df["Close_yfin"], window=10)
        df["EMA_20"] = ta.tren_]()

