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

ticker = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, MSFT)", "AAPL")

if st.button("Generate Signal"):
    ticker = ticker.strip().upper()

    try:
        st.write("Fetching data ...")
        stock = yf.Ticker(ticker)
        data = stock.history(period="5y")

        # --- Diagnostics: show what actually came back ---
        st.write("Returned columns:", list(data.columns))
        st.write("Data preview:", data.head())

        if data.empty or "Close" not in data.columns:
            st.error(
                "⚠️ Yahoo Finance returned no usable data. "
                "This often happens on Streamlit Cloud when the free sandbox blocks downloads. "
                "Try again later or run locally."
            )
            st.stop()

        data = data.dropna(subset=["Close"]).copy()
        data["Close"] = data["Close"].astype(float)

        # Indicators
        data["RSI"] = RSIIndicator(data["Close"], window=14).rsi()
        data["MA50"] = SMAIndicator(data["Close"], window=50).sma_indicator()
        data["MA200"] = SMAIndicator(data["Close"], window=200).sma_indicator()
        macd = MACD(data["Close"])
        data["MACD_line"] = macd.macd()
        data["Signal_line"] = macd.macd_signal()
        data = data.dropna()

        # Scores
        def tech_score(row):
            s = 0
            if row["RSI"] < 30: s += 0.1
            if row["MA50"] > row["MA200"]: s += 0.15
            if row["MACD_line"] > row["Signal_line"]: s += 0.05
            return s

        data["technical_score"] = data.apply(tech_score, axis=1)

        # Model
        data["return"] = data["Close"].pct_change()
        data["target"] = (data["return"].shift(-1) > 0).astype(int)
        feats = ["RSI", "MA50", "MA200", "MACD_line", "Signal_line"]
        data = data.dropna()

        if len(data) > 100:
            model = RandomForestClassifier()
            model.fit(data[feats], data["target"])
            data["pred_prob"] = model.predict_proba(data[feats])[:, 1]
        else:
            data["pred_prob"] = 0.5

        data["signal_score"] = 0.5 * data["technical_score"] + 0.5 * data["pred_prob"]
        data["signal"] = np.where(
            data["signal_score"] > 0.7, "BUY",
            np.where(data["signal_score"] < 0.4, "SELL", "HOLD")
        )

        latest_signal = data["signal"].iloc[-1]
        st.metric("Latest Signal", latest_signal)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data["Close"], label="Close")
        ax.scatter(data[data["signal"] == "BUY"].index, data[data["signal"] == "BUY"]["Close"],
                   color="green", marker="^", s=60, label="BUY")
        ax.scatter(data[data["signal"] == "SELL"].index, data[data["signal"] == "SELL"]["Close"],
                   color="red", marker="v", s=60, label="SELL")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ {type(e).__name__}: {e}")
        st.info(
            "If data columns show empty or no 'Close', "
            "run this code locally on your computer using Jupyter Notebook or VS Code. "
            "The Streamlit free cloud sometimes blocks `yfinance`."
        )
