import streamlit as st
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
import numpy as np

st.title("📈 Stock Price Predictor")

model = load_model("model/model.h5")
scaler = joblib.load("model/scaler.save")

ticker = st.text_input("Enter stock symbol (e.g. AAPL):", "AAPL")

if st.button("Predict"):
    df = yf.download(ticker, period="90d")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df_scaled = scaler.transform(df)
    X_input = df_scaled[-60:]
    X_input = np.expand_dims(X_input, axis=0)
    prediction = model.predict(X_input)[0][0]

    st.subheader("📊 Prediction")
    st.write(f"Predicted Close Price: ${round(prediction, 2)}")