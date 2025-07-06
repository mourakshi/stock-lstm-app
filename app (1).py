import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ================= Load Model and Scaler ====================
model = load_model("model/model.h5")
scaler = joblib.load("model/scaler.save")

# ================= Preprocess Function ====================
def preprocess(df):
    features = ["Close", "SMA_10"]
    df = df[features]

    # Check if we have enough data
    if len(df) < 60:
        raise ValueError("Not enough data to make prediction. Need at least 60 rows.")

    last_60 = df[-60:].values
    X_scaled = scaler.transform(last_60)
    return np.expand_dims(X_scaled, axis=0), df["Close"].iloc[-1]

# ================= Fetch Function ====================
@st.cache_data(show_spinner=False)
def fetch_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df = df.dropna()
    return df

# ================= Streamlit UI ====================
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Real-Time Stock Predictor & Investment Simulator")

ticker = st.text_input("Enter stock symbol (e.g. AAPL):", "AAPL")
investment = st.number_input("💸 Investment Amount ($):", min_value=1.0, value=1000.0)

if st.button("Predict Next Day Price"):
    try:
        df = fetch_data(ticker)
        X_input, last_price = preprocess(df)

        # Predict next day's scaled price
        predicted_scaled = model.predict(X_input)
        predicted_scaled = predicted_scaled.flatten()[0]  # ✅ Safer

        # Create dummy array for inverse transform
        dummy_input = np.zeros((1, scaler.n_features_in_))
        dummy_input[0][0] = predicted_scaled  # assuming Close is the first column

        predicted_price = scaler.inverse_transform(dummy_input)[0][0]

        # ====== UI Output ======
        st.success("Prediction Complete ✅")
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

# ========== Plot Chart ==========
try:
    df = fetch_data(ticker)
    st.subheader(f"📊 Historical Close Price for {ticker.upper()}")
    st.line_chart(df["Close"][-60:])
except:
    st.warning("Could not fetch historical chart.")

