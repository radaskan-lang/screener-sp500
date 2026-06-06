import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import openpyxl
from io import BytesIO

# ── # ── TICKERS ─────────────────────────────
import pandas as pd

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    tickers = table["Symbol"].tolist()

    # correction pour Yahoo Finance (BRK.B → BRK-B, etc.)
    tickers = [t.replace(".", "-") for t in tickers]

    return tickers

TICKERS = get_sp500_tickers()

# ── RSI ────────────────────────────────
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, 1)
    return 100 - (100 / (1 + rs.iloc[-1]))

# ── FETCH DATA ─────────────────────────
def fetch(ticker):
    t = yf.Ticker(ticker)
    hist = t.history(period="1y")

    if hist.empty or len(hist) < 50:
        return None

    close = hist["Close"]

    price = close.iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]

    return {
        "Ticker": ticker,
        "Prix": price,
        "MA50": ma50,
        "MA200": ma200,
        "RSI": rsi(close)
    }

# ── SCORE ───────────────────────────────
def score(row):
    s = 0
    if row["Prix"] > row["MA50"]: s += 1
    if row["MA50"] > row["MA200"]: s += 1
    if 40 < row["RSI"] < 65: s += 1
    return s

# ── EXCEL EXPORT ────────────────────────
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Screener")
    return output.getvalue()

# ── APP ────────────────────────────────
st.title("📊 Screener S&P 500 (Cloud Version)")

if st.button("🔄 Lancer analyse"):
    data = []

    for t in TICKERS:
        d = fetch(t)
        if d:
            d["Score"] = score(d)
            data.append(d)

    df = pd.DataFrame(data)

    st.dataframe(df.sort_values("Score", ascending=False))

    excel = to_excel(df)

    st.download_button(
        "📥 Télécharger Excel",
        data=excel,
        file_name=f"screener_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    )
