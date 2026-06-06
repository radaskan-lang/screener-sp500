import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from io import BytesIO
import time

# ─────────────────────────────
# 📌 S&P 500 — SOURCE WIKIPEDIA (fiable)
# ─────────────────────────────
@st.cache_data(ttl=86400)
def get_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].tolist()
    # Wikipedia utilise des points, Yahoo Finance utilise des tirets
    return [t.replace(".", "-") for t in tickers]


# ─────────────────────────────
# 📊 RSI
# ─────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    loss_safe = loss.clip(lower=1e-10)
    rs = gain / loss_safe
    return 100 - (100 / (1 + rs.iloc[-1]))


# ─────────────────────────────
# 📈 FETCH SAFE (ANTI CRASH)
# ─────────────────────────────
def fetch(ticker):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1y")

        if hist is None or hist.empty or len(hist) < 50:
            return None

        close = hist["Close"]

        price = float(close.iloc[-1])
        ma50  = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])

        return {
            "Ticker": ticker,
            "Prix":   price,
            "MA50":   ma50,
            "MA200":  ma200,
            "RSI":    calc_rsi(close),
        }

    except Exception:
        return None


# ─────────────────────────────
# 🧠 IA SCORE
# ─────────────────────────────
def ai_score(row):
    score   = 0
    reasons = []

    price   = row["Prix"]
    ma50    = row["MA50"]
    ma200   = row["MA200"]
    rsi_val = row["RSI"]

    # Trend
    if price > ma50 > ma200:
        score += 35
        reasons.append("Trend haussière forte")
    elif price > ma200:
        score += 20
        reasons.append("Trend positive")
    else:
        score += 5
        reasons.append("Trend faible")

    # RSI
    if 40 <= rsi_val <= 65:
        score += 25
        reasons.append("Momentum sain")
    elif rsi_val < 30:
        score += 15
        reasons.append("Survente")
    elif rsi_val > 70:
        score += 5
        reasons.append("Surachat")

    # Structure MA
    if ma50 > ma200:
        score += 20
        reasons.append("Golden structure")

    # Position prix
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


# ─────────────────────────────
# 📦 EXCEL EXPORT
# ─────────────────────────────
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Screener")
    return output.getvalue()


# ─────────────────────────────
# 🚀 APP
# ─────────────────────────────
st.title("📊 S&P 500 IA Screener — VERSION STABLE PRO")

# Chargement des tickers avec message d'erreur clair
try:
    tickers = get_sp500()
    st.success(f"✅ {len(tickers)} actions chargées depuis Wikipedia")
except Exception as e:
    st.error(f"❌ Impossible de charger la liste S&P 500 : {e}")
    st.stop()

if st.checkbox("⚡ Mode rapide (100 actions)"):
    tickers = tickers[:100]

if st.button("🔄 Lancer analyse IA"):

    rows     = []
    progress = st.progress(0)

    for i, tkr in enumerate(tickers):

        data = fetch(tkr)

        if data:
            score, reasons = ai_score(data)
            signal = ai_signal(score)

            data["AI Score"]   = score
            data["AI Signal"]  = signal
            data["AI Reasons"] = " | ".join(reasons)

            rows.append(data)

        progress.progress((i + 1) / len(tickers))
        time.sleep(0.15)

    if len(rows) == 0:
        st.error("Aucune donnée récupérée — problème Yahoo Finance")
    else:
        df = pd.DataFrame(rows)
        df = df.sort_values("AI Score", ascending=False)

        st.subheader("🏆 Top opportunités IA")
        st.dataframe(df)

        st.subheader("📥 Export Excel")
        excel = to_excel(df)

        st.download_button(
            "Télécharger Excel",
            data=excel,
            file_name=f"screener_pro_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
