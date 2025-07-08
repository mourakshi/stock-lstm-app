import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import ta

# ========== Load model and scaler ==========
model = load_model("model/model.h5")
scaler = joblib.load("model/scaler.save")

# ========== Define feature columns ==========
FEATURES = [
    'Close_yfin', 'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20',
    'Rolling_STD_10', 'Rolling_Max_10', 'Rolling_Min_10',
    'Momentum_10', 'RSI_14', 'MACD', 'Signal_Line', 'Bollinger_Width'
]

# ========== Fetch stock data ==========
def fetch_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=False)

    if df.empty or "Close" not in df.columns:
        raise ValueError("⚠️ Could not fetch data. Please check the stock symbol.")

    df.rename(columns={
        "Open": "Open_yfin",
        "High": "High_yfin",
        "Low": "Low_yfin",
        "Close": "Close_yfin",
        "Volume": "Volume_yfin"
    }, inplace=True)

    try:
        # Flattened indicators
        df["SMA_10"] = ta.trend.sma_indicator(close=df["Close_yfin"], window=10).astype(float)
        df["SMA_20"] = ta.trend.sma_indicator(close=df["Close_yfin"], window=20).astype(float)
        df["EMA_10"] = ta.trend.ema_indicator(close=df["Close_yfin"], window=10).astype(float)
        df["EMA_20"] = ta.trend.ema_indicator(close=df["Close_yfin"], window=20).astype(float)

        df["Rolling_STD_10"] = df["Close_yfin"].rolling(window=10).std()
        df["Rolling_Max_10"] = df["Close_yfin"].rolling(window=10).max()
        df["Rolling_Min_10"] = df["Close_yfin"].rolling(window=10).min()

        df["Momentum_10"] = df["Close_yfin"] - df["Close_yfin"].shift(10)
        df["RSI_14"] = ta.momentum.rsi(close=df["Close_yfin"], window=14).astype(float)

        # MACD and Signal
        macd_df = ta.trend.macd(close=df["Close_yfin"])
        df["MACD"] = macd_df.squeeze() if isinstance(macd_df, pd.DataFrame) else macd_df
        df["Signal_Line"] = ta.trend.macd_signal(close=df["Close_yfin"]).astype(float)

        # Bollinger Band Width
        bb = ta.volatility.BollingerBands(close=df["Close_yfin"])
        bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        df["Bollinger_Width"] = bb_width.astype(float)

        df.dropna(inplace=True)
        return df

    except Exception as e:
        raise ValueError(f"⚠️ Feature computation failed: {e}")

# ========== Preprocess for LSTM ==========
def preprocess(df):
    df = df[FEATURES].dropna()
    last_60 = df[-60:].values
    X_scaled = scaler.transform(last_60)
    return np.expand_dims(X_scaled, axis=0), df["Close_yfin"].iloc[-1]

# ========== Streamlit UI ==========
st.set_page_config(page_title="Stock LSTM Predictor", layout="centered")
st.title("📈 Real-Time Stock Predictor & Investment Simulator")

ticker = st.text_input("Enter stock symbol (e.g. AAPL, TSLA, INFY.NS):", "AAPL")
investment = st.number_input("💸 Investment Amount ($):", value=1000.0)

df = None

if st.button("Predict Next Day Price"):
    with st.spinner("Fetching data and predicting..."):
        try:
            df = fetch_data(ticker)
            X_input, last_price = preprocess(df)

            predicted_scaled = model.predict(X_input)[0][0]

            dummy = np.zeros((1, scaler.n_features_in_))
            dummy[0][0] = predicted_scaled
            predicted_price = scaler.inverse_transform(dummy)[0][0]

            st.success("✅ Prediction Complete")
            st.metric("📉 Last Close Price", f"${last_price:.2f}")
            st.metric("📈 Predicted Next Price", f"${predicted_price:.2f}")

            # Simulation
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

# ========== Plot ==========
if df is not None:
    st.subheader(f"📊 Historical Close Price for {ticker}")
    try:
        st.line_chart(df["Close_yfin"][-60:])
    except Exception as e:
        st.warning(f"Could not load chart: {e}")

