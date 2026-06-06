import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from io import BytesIO
import time

# ─────────────────────────────
# 📌 S&P 500 — LISTE STATIQUE INTÉGRÉE (aucune dépendance externe)
# ─────────────────────────────
SP500_TICKERS = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE",
    "ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AEP","AXP","AIG","AMT",
    "AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","APO","AAPL","AMAT","APTV","ACGL",
    "ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC",
    "BAX","BDX","BRK-B","BBY","BIO","TECH","BIIB","BLK","BX","BA","BCR","BSX","BMY","AVGO","BR",
    "BRO","BF-B","BLDR","BXP","CHRW","CDNS","CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR",
    "CTLT","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNX","CDAY","CF","CRL","SCHW","CHTR",
    "CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH",
    "CL","CMCSA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY","CTVA","CSGP","COST",
    "CTRA","CRWD","CCI","CSX","CMI","CVS","DHR","DRI","DVA","DAY","DECK","DE","DELL","DAL",
    "DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DHI","DTE","DUK","DD",
    "EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX",
    "EQIX","EQR","ESS","EL","ETSY","EG","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV",
    "FDS","FICO","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA",
    "FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD",
    "GPN","GL","GDDY","GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE","HLT","HOLX",
    "HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW","INCY",
    "IR","PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM","JBAL",
    "JKHY","J","JBL","JNPR","JPM","JNPR","K","KVUE","KDP","KEY","KEYS","KMB","KIM","KMI","KKR",
    "KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LII","LLY","LIN","LYV","LKQ",
    "LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH",
    "MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA",
    "MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX",
    "NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA",
    "NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PANW","PH",
    "PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL",
    "PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF",
    "RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM",
    "SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SNA","SOLV","SO","LUV","SWK","SBUX",
    "STT","STLD","STE","SYK","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL",
    "TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL",
    "TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO","VRSN",
    "VRSK","VZ","VRTX","VTRS","VICI","V","VST","VMC","WRB","GWW","WAB","WBA","WMT","DIS","WBD",
    "WM","WAT","WEC","WFC","WELL","WST","WDC","WY","WMB","WTW","WYNN","XEL","XYL","YUM","ZBRA",
    "ZBH","ZTS"
]

@st.cache_data(ttl=86400)
def get_sp500():
    return SP500_TICKERS


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

tickers = get_sp500()
st.success(f"✅ {len(tickers)} actions chargées")

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
