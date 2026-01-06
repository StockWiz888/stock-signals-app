import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# --- Streamlit Settings ---
st.set_page_config(page_title="Stock Signal App", page_icon="📈", layout="centered")
st.title("📈 Free Stock Buy/Sell Signal App")

ticker = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, MSFT)", "AAPL")

if st.button("Generate Signal"):
    try:
        # --- Normalize and Download ---
        ticker = ticker.strip().upper()
        data = yf.download(ticker, start="2020-01-01", progress=False)

        # --- Handle Empty or Multi-Index Data ---
        if data is None or data.empty:
            st.error(f"❌ No data found for '{ticker}'. Try another stock symbol.")
            st.stop()

        # Flatten multi-index columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # --- Verify Close Column ---
        if "Close" not in data.columns:
            st.error("⚠️ Yahoo Finance did not return a 'Close' column. Please try again.")
            st.write("Returned columns:", list(data.columns))
            st.stop()

        # --- Clean Data ---
        data = data.dropna(subset=["Close"]).copy()
        data["Close"] = data["Close"].astype(float)

        # --- Technical Indicators ---
        try:
            data["RSI"] = RSIIndicator(data["Close"], window=14).rsi()
            data["MA50"] = SMAIndicator(data["Close"], window=50).sma_indicator()
            data["MA200"] = SMAIndicator(data["Close"], window=200).sma_indicator()
            macd = MACD(data["Close"])
            data["MACD_line"] = macd.macd()
            data["Signal_line"] = macd.macd_signal()
        except Exception as e:
            st.error(f"⚠️ Indicator calculation error: {e}")
            st.stop()

        data = data.dropna()

        # --- Technical Score ---
        def tech_score(row):
            score = 0
            if row["RSI"] < 30: score += 0.1
            if row["MA50"] > row["MA200"]: score += 0.15
            if row["MACD_line"] > row["Signal_line"]: score += 0.05
            return score

        data["technical_score"] = data.apply(tech_score, axis=1)

        # --- Machine Learning Model ---
        data["return"] = data["Close"].pct_change()
        data["target"] = (data["return"].shift(-1) > 0).astype(int)
        features = ["RSI", "MA50", "MA200", "MACD_line", "Signal_line"]
        data = data.dropna()

        if len(data) > 100:
            model = RandomForestClassifier()
            model.fit(data[features], data["target"])
            data["pred_prob"] = model.predict_proba(data[features])[:, 1]
        else:
            data["pred_prob"] = 0.5

        # --- Combine Scores ---
        data["signal_score"] = 0.5 * data["technical_score"] + 0.5 * data["pred_prob"]
        data["signal"] = np.where(
            data["signal_score"] > 0.7, "BUY",
            np.where(data["signal_score"] < 0.4, "SELL", "HOLD")
        )

        # --- Display Signal ---
        latest_signal = data["signal"].iloc[-1]
        latest_score = data["signal_score"].iloc[-1]
        st.metric("Latest Signal", latest_signal, f"{latest_score:.2f}")

        # --- Plot Chart ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data["Close"], label="Close Price", linewidth=1.2)
        ax.scatter(data[data["signal"] == "BUY"].index,
                   data[data["signal"] == "BUY"]["Close"],
                   color="green", marker="^", label="BUY", s=80)
        ax.scatter(data[data["signal"] == "SELL"].index,
                   data[data["signal"] == "SELL"]["Close"],
                   color="red", marker="v", label="SELL", s=80)
        ax.legend()
        ax.set_title(f"{ticker} Price Chart with Buy/Sell Signals")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error occurred: {e}")
