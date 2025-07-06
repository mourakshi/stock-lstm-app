import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# ===== Load model and scaler =====
model = load_model("model/model.h5")
scaler = joblib.load("model/scaler.save")

# ===== Fetch stock data =====
def fetch_data(ticker):
    df = yf.download(ticker, period="6mo")
    df["SMA_10"] = df["Close"].rolling(window=10).mean()  # Optional, unused here
    df = df.dropna()
    return df

# ===== Preprocess latest 60 days =====
def preprocess(df):
    features = ["Close", "Open", "High", "Low", "Volume", "SMA_10", "EMA_20",
                "RSI_14", "MACD_Line", "SMA_30", "EMA_50", "Return", "Volatility"]  # match your training features

    df = df[features].dropna()
    last_60 = df[-60:].values
    X_scaled = scaler.transform(last_60)
    return np.expand_dims(X_scaled, axis=0), df["Close"].iloc[-1]


# ===== Streamlit UI =====
st.title("📈 Real-Time Stock Predictor & Investment Simulator")
ticker = st.text_input("Enter stock symbol (e.g. AAPL):", "AAPL")
investment = st.number_input("💸 Investment Amount ($):", value=1000.0)

if st.button("Predict Next Day Price"):
    with st.spinner("Fetching data and predicting..."):
        try:
            df = fetch_data(ticker)
            X_input, last_price = preprocess(df)

            predicted_scaled = model.predict(X_input)[0][0]

            # Inverse transform to get price
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
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ===== Historical Chart =====
st.subheader(f"📊 Historical Close Price for {ticker}")
try:
    st.line_chart(fetch_data(ticker)["Close"][-60:])
except:
    st.warning("Could not load chart. Please check the stock symbol.")
