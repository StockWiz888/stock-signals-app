import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.title("📈 Free Stock Buy/Sell Signal App")

ticker = st.text_input("Enter Stock Symbol", "AAPL")

if st.button("Generate Signal"):
    data = yf.download(ticker, start="2022-01-01")
    
    data["RSI"] = RSIIndicator(data["Close"], window=14).rsi()
    data["MA50"] = SMAIndicator(data["Close"], window=50).sma_indicator()
    data["MA200"] = SMAIndicator(data["Close"], window=200).sma_indicator()
    macd = MACD(data["Close"])
    data["MACD_line"] = macd.macd()
    data["Signal_line"] = macd.macd_signal()
    
    def tech_score(row):
        score = 0
        if row["RSI"] < 30: score += 0.1
        if row["MA50"] > row["MA200"]: score += 0.15
        if row["MACD_line"] > row["Signal_line"]: score += 0.05
        return score

    data["technical_score"] = data.apply(tech_score, axis=1)
    
    data["return"] = data["Close"].pct_change()
    data["target"] = (data["return"].shift(-1) > 0).astype(int)
    features = ["RSI", "MA50", "MA200", "MACD_line", "Signal_line"]
    data = data.dropna()
    model = RandomForestClassifier()
    model.fit(data[features], data["target"])
    data["pred_prob"] = model.predict_proba(data[features])[:, 1]
    data["signal_score"] = 0.5 * data["technical_score"] + 0.5 * data["pred_prob"]

    data["signal"] = np.where(data["signal_score"] > 0.7, "BUY",
                     np.where(data["signal_score"] < 0.4, "SELL", "HOLD"))

    st.metric("Latest Signal", data["signal"].iloc[-1], f"{data['signal_score'].iloc[-1]:.2f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data["Close"], label="Close Price")
    ax.scatter(data[data["signal"] == "BUY"].index, data[data["signal"] == "BUY"]["Close"], color="green", marker="^", label="BUY")
    ax.scatter(data[data["signal"] == "SELL"].index, data[data["signal"] == "SELL"]["Close"], color="red", marker="v", label="SELL")
    ax.legend()
    st.pyplot(fig)
