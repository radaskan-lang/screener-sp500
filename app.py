import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from io import BytesIO
import time

# ─────────────────────────────────────
# 📌 S&P 500 LIST AUTO
# ─────────────────────────────────────
@st.cache_data
def get_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    tickers = table["Symbol"].tolist()
    return [t.replace(".", "-") for t in tickers]

# ─────────────────────────────────────
# 📊 INDICATEURS
# ─────────────────────────────────────
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, 1)
    return 100 - (100 / (1 + rs.iloc[-1]))

def fetch(ticker):
    try:
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

    except:
        return None

# ─────────────────────────────────────
# 🧠 IA SCORE
# ─────────────────────────────────────
def ai_score(row):
    score = 0
    reasons = []

    price = float(row["Prix"])
    ma50 = float(row["MA50"])
    ma200 = float(row["MA200"])
    rsi = float(row["RSI"])

    # Trend structure
    if price > ma50 > ma200:
        score += 35
        reasons.append("Tendance haussière forte")
    elif price > ma200:
        score += 20
        reasons.append("Tendance positive")
    else:
        score += 5
        reasons.append("Tendance faible")

    # RSI
    if 40 <= rsi <= 65:
        score += 25
        reasons.append("Momentum sain")
    elif rsi < 30:
        score += 15
        reasons.append("Survente (rebond possible)")
    elif rsi > 70:
        score += 5
        reasons.append("Surachat")

    # Structure MA
    if ma50 > ma200:
        score += 20
        reasons.append("Golden structure MA50 > MA200")

    # Discount vs MA200
    if price > ma200:
        score += 20
    else:
        score += 5

    return score, reasons

def ai_signal(score):
    if score >= 85:
        return "🟢 STRONG BUY"
    elif score >= 70:
        return "🟢 BUY"
    elif score >= 50:
        return "🟡 HOLD"
    else:
        return "🔴 AVOID"

# ─────────────────────────────────────
# 📦 EXCEL EXPORT
# ─────────────────────────────────────
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Screener")
    return output.getvalue()

# ─────────────────────────────────────
# 🚀 APP STREAMLIT
# ─────────────────────────────────────
st.title("📊 S&P 500 IA Screener (Pro Version)")

tickers = get_sp500()

# Mode rapide
if st.checkbox("⚡ Mode rapide (50 actions)"):
    tickers = tickers[:50]

if st.button("🔄 Lancer analyse IA"):

    rows = []

    progress = st.progress(0)

    for i, tkr in enumerate(tickers):

        data = fetch(tkr)

        if data:
            score, reasons = ai_score(data)
            signal = ai_signal(score)

            data["AI Score"] = score
            data["AI Signal"] = signal
            data["AI Reasons"] = " | ".join(reasons)

            rows.append(data)

        progress.progress((i + 1) / len(tickers))
        time.sleep(0.2)

    df = pd.DataFrame(rows)

    df = df.sort_values("AI Score", ascending=False)

    st.subheader("🏆 Top opportunités IA")
    st.dataframe(df)

    st.subheader("📥 Export Excel")
    excel = to_excel(df)

    st.download_button(
        "Télécharger Excel",
        data=excel,
        file_name=f"screener_ia_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    )
