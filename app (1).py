import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# ===== Load Model and Scaler =====
model = load_model("model/model.h5")
scaler = joblib.load("model/scaler.save")

# ===== Page Config =====
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")
st.title("📈 Real-Time Stock Predictor & Investment Simulator")

# ===== Input =====
ticker = st.text_input("Enter stock symbol (e.g. AAPL):", "AAPL")
investment = st.number_input("💸 Investment Amount ($):", value=1000.0)

# ===== Fetch Data =====
@st.cache_data
def fetch_data(symbol):
    df = yf.download(symbol, period="3mo", interval="1d")
    df = df[["Close"]]
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df = df.dropna()
    return df

# ===== Preprocess Function =====
def preprocess(df):
    # Use only "Close" feature (matches training)
    df = df[["Close"]]
    last_60 = df[-60:].values
    X_scaled = scaler.transform(last_60)
    return np.expand_dims(X_scaled, axis=0), df["Close"].iloc[-1]

# ===== Predict Button =====
if st.button("Predict Next Day Price"):
    with st.spinner("Fetching data and predicting..."):
        df = fetch_data(ticker)
        X_input, last_price = preprocess(df)

        predicted_scaled = model.predict(X_input)[0][0]

        # Recreate array for inverse_transform (must match scaler feature shape)
        dummy_input = np.zeros((1, scaler.n_features_in_))
        dummy_input[0][0] = predicted_scaled  # assuming Close is at index 0

        predicted_price = scaler.inverse_transform(dummy_input)[0][0]

    st.success("✅ Prediction Complete")
    st.metric("📉 Last Close Price", f"${last_price:.2f}")
    st.metric("📈 Predicted Next Price", f"${predicted_price:.2f}")

    # Investment Simulation
    profit = (predicted_price - last_price) * (investment / last_price)
    st.subheader("💰 Investment Simulation")
    st.write(f"If you invest **${investment:.2f}** now:")
    st.write(f"**Predicted Value Tomorrow:** ${investment + profit:.2f}")

    if profit > 0:
        st.success(f"📈 Estimated Profit: ${profit:.2f}")
    else:
        st.error(f"📉 Estimated Loss: ${-profit:.2f}")

# ===== Chart =====
st.subheader(f"📊 Historical Close Price for {ticker}")
st.line_chart(fetch_data(ticker)["Close"][-60:])
