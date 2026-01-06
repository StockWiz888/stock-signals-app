import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Signal App", page_icon="📈", layout="centered")

st.title("📈 Free Stock Buy/Sell Signal App")

ticker = st.text_input("Enter Stock Symbol (e.g. AAPL, MSFT, TSLA)", "AAPL")

if st.button("Generate Signal"):
    try:
        # Download data
        data = yf.download(ticker, start="2020-01-01", progress=False)
        if data.empty:
            st.error("❌ No data found for this ticker. Try another one like AAPL or MSFT.")
        else:
            # --- Technical Indicators ---
            # Handle missing or invalid data
data = data.dropna(subset=["Close"])

if len(data) < 20:
    st.error("⚠️ Not enough data to calculate indicators for this stock.")
else:
    # Safely compute indicators
    try:
        data["RSI"] = RSIIndicator(data["Close"].fillna(method='ffill'), window=14).rsi()
        data["MA50"] = SMAIndicator(data["Close"], window=50).sma_indicator()
        data["MA200"] = SMAIndicator(data["Close"], window=200).sma_indicator()
        macd = MACD(data["Close"])
        data["MACD_line"] = macd.macd()
        data["Signal_line"] = macd.macd_signal()
    except Exception as e:
        st.error(f"Indicator calculation failed: {e}")
        st.stop()
