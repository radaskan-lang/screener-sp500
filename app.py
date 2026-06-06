import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import time
import concurrent.futures
import anthropic
from pattern_detection import detect_all_patterns, pattern_badge
from risk_manager import calc_risk_reward, risk_badge, format_trade_plan

# ─────────────────────────────
# 🎨 PAGE CONFIG
# ─────────────────────────────
st.set_page_config(
    page_title="S&P 500 IA Screener Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────
# 💅 CSS CUSTOM
# ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}
h1, h2, h3 {
    font-family: 'Space Mono', monospace;
    color: #00ff88 !important;
}
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 4px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00ff88;
}
.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}
.stButton > button {
    background: linear-gradient(135deg, #00ff88, #00cc6a) !important;
    color: #0a0e1a !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    letter-spacing: 0.05em;
}
.ai-analysis-box {
    background: linear-gradient(135deg, #0f1f35 0%, #0a1628 100%);
    border: 1px solid #00ff8844;
    border-left: 4px solid #00ff88;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 0.9rem;
    line-height: 1.6;
}
.pattern-box {
    background: linear-gradient(135deg, #0f1f35 0%, #0a1628 100%);
    border: 1px solid #fbbf2444;
    border-left: 4px solid #fbbf24;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.85rem;
    line-height: 1.6;
}
.trade-plan-box {
    background: linear-gradient(135deg, #0f2a1f 0%, #0a1a14 100%);
    border: 1px solid #00ff8833;
    border-left: 4px solid #00ff88;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 0.88rem;
    line-height: 1.8;
}
.rr-excellent { color: #00ff88; font-weight: 700; }
.rr-bon       { color: #4ade80; font-weight: 600; }
.rr-acceptable{ color: #fbbf24; font-weight: 600; }
.rr-faible    { color: #f87171; font-weight: 600; }
.ticker-badge {
    display: inline-block;
    background: #00ff8822;
    border: 1px solid #00ff8866;
    color: #00ff88;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    padding: 2px 10px;
    border-radius: 4px;
    margin-right: 8px;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    overflow: hidden;
}
.stProgress > div > div {
    background: linear-gradient(90deg, #00ff88, #00cc6a) !important;
}
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e3a5f;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# 📌 S&P 500 TICKERS
# ─────────────────────────────
SP500_TICKERS = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE",
    "ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AEP","AXP","AIG","AMT",
    "AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","APO","AAPL","AMAT","APTV","ACGL",
    "ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC",
    "BAX","BDX","BRK-B","BBY","BIO","BIIB","BLK","BX","BA","BSX","BMY","AVGO","BR","BRO","BF-B",
    "BLDR","BXP","CHRW","CDNS","CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR","CAT","CBOE",
    "CBRE","CDW","CE","COR","CNC","CDAY","CF","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI",
    "CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CAG","COP","ED",
    "STZ","CEG","COO","CPRT","GLW","CPAY","CTVA","CSGP","COST","CTRA","CRWD","CCI","CSX","CMI",
    "CVS","DHR","DRI","DVA","DAY","DECK","DE","DELL","DAL","DVN","DXCM","FANG","DLR","DFS","DG",
    "DLTR","D","DPZ","DOV","DOW","DHI","DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA",
    "ELV","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRG",
    "ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT","FDX","FIS","FITB",
    "FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC",
    "GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD","GPN","GL","GDDY","GS","HAL","HIG","HAS",
    "HCA","DOC","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB",
    "HUM","HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG",
    "INTU","ISRG","IVZ","INVH","IQV","IRM","JKHY","J","JBL","JPM","K","KVUE","KDP","KEY","KEYS",
    "KMB","KIM","KMI","KKR","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LII",
    "LLY","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR","MMC",
    "MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU",
    "MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI",
    "NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH",
    "NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR",
    "PKG","PANW","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL",
    "PPG","PPL","PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","PWR","QCOM","DGX","RL",
    "RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI",
    "CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SNA","SOLV","SO","LUV","SWK",
    "SBUX","STT","STLD","STE","SYK","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT",
    "TEL","TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC",
    "TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO",
    "VRSN","VRSK","VZ","VRTX","VTRS","VICI","V","VST","VMC","WRB","GWW","WAB","WBA","WMT","DIS",
    "WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WY","WMB","WTW","WYNN","XEL","XYL","YUM",
    "ZBRA","ZBH","ZTS"
]

# ─────────────────────────────
# 📊 INDICATEURS TECHNIQUES
# ─────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.clip(lower=1e-10)
    return float(100 - (100 / (1 + rs.iloc[-1])))

def calc_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])

def calc_bollinger(series, period=20):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    price = series.iloc[-1]
    bandwidth = float((upper.iloc[-1] - lower.iloc[-1]) / ma.iloc[-1] * 100)
    pct_b = float((price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]))
    return bandwidth, pct_b

def calc_volume_signal(volume, close):
    avg_vol = volume.rolling(20).mean().iloc[-1]
    last_vol = volume.iloc[-1]
    price_up = close.iloc[-1] > close.iloc[-2]
    vol_ratio = float(last_vol / avg_vol) if avg_vol > 0 else 1.0
    return vol_ratio, price_up

# ─────────────────────────────
# 📈 FETCH
# ─────────────────────────────
def fetch(ticker):
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="1y")

        if hist is None or hist.empty or len(hist) < 50:
            return None

        close  = hist["Close"]
        volume = hist["Volume"]
        high   = hist["High"]

        price = float(close.iloc[-1])
        ma50  = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        rsi   = calc_rsi(close)
        macd_line, macd_signal, macd_hist = calc_macd(close)
        bb_bandwidth, bb_pct = calc_bollinger(close)
        vol_ratio, price_up  = calc_volume_signal(volume, close)

        # ── Patterns ──
        patterns_data = detect_all_patterns(hist)

        # ── Risque / Rendement ──
        rr_data = calc_risk_reward(hist)

        # Fondamentaux
        info       = t.info
        pe_ratio   = info.get("trailingPE", None)
        revenue_gr = info.get("revenueGrowth", None)
        net_margin = info.get("profitMargins", None)
        sector     = info.get("sector", "N/A")

        return {
            "Ticker":        ticker,
            "Sector":        sector,
            "Prix":          round(price, 2),
            "MA50":          round(ma50, 2),
            "MA200":         round(ma200, 2),
            "RSI":           round(rsi, 1),
            "MACD":          round(macd_line, 3),
            "MACD_Signal":   round(macd_signal, 3),
            "MACD_Hist":     round(macd_hist, 3),
            "BB_Width":      round(bb_bandwidth, 1),
            "BB_Pct":        round(bb_pct, 2),
            "Vol_Ratio":     round(vol_ratio, 2),
            "Price_Up":      price_up,
            "PE":            round(pe_ratio, 1) if pe_ratio else None,
            "Rev_Growth":    round(revenue_gr * 100, 1) if revenue_gr else None,
            "Net_Margin":    round(net_margin * 100, 1) if net_margin else None,
            # Patterns
            "Top_Pattern":   patterns_data["top_pattern"],
            "Patterns":      patterns_data["summary"],
            "Pattern_Score": patterns_data["bonus_score"],
            "Pattern_Badge": pattern_badge(patterns_data["bonus_score"]),
            # Risque / Rendement
            "Entree":        rr_data["entry"],
            "Stop":          rr_data["stop"],
            "Target":        rr_data["target"],
            "RR_Ratio":      rr_data["rr_ratio"],
            "Risque_Pct":    rr_data["risk_pct"],
            "Gain_Pct":      rr_data["reward_pct"],
            "ATR":           rr_data["atr"],
            "ATR_Pct":       rr_data["atr_pct"],
            "Support":       rr_data["support"],
            "Resistance":    rr_data["resistance"],
            "RR_Quality":    rr_data["quality"],
            "RR_Confidence": rr_data["confidence"],
            "RR_Badge":      risk_badge(rr_data["rr_ratio"], rr_data["risk_pct"]),
            # Stocker rr_data complet pour l'affichage
            "_rr_data":      rr_data,
        }
    except Exception:
        return None

def fetch_parallel(tickers, max_workers=10):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch, t): t for t in tickers}
        progress = st.progress(0)
        status   = st.empty()
        done = 0
        for future in concurrent.futures.as_completed(futures):
            done += 1
            data = future.result()
            if data:
                results.append(data)
            progress.progress(done / len(tickers))
            status.markdown(f"⚡ `{done}/{len(tickers)}` actions analysées...")
        status.empty()
    return results

# ─────────────────────────────
# 🧠 SCORE IA
# ─────────────────────────────
def ai_score(row):
    score   = 0
    reasons = []

    price      = row["Prix"]
    ma50       = row["MA50"]
    ma200      = row["MA200"]
    rsi_val    = row["RSI"]
    macd_hist  = row["MACD_Hist"]
    bb_pct     = row["BB_Pct"]
    vol_ratio  = row["Vol_Ratio"]
    price_up   = row["Price_Up"]
    pe         = row["PE"]
    rev_growth = row["Rev_Growth"]
    net_margin = row["Net_Margin"]

    # Trend (30 pts)
    if price > ma50 > ma200:
        score += 30
        reasons.append("Trend haussière forte")
    elif price > ma200:
        score += 18
        reasons.append("Trend positive")
    else:
        score += 3
        reasons.append("Trend faible")

    # RSI (20 pts)
    if 40 <= rsi_val <= 60:
        score += 20
        reasons.append("RSI idéal")
    elif 60 < rsi_val <= 70:
        score += 12
        reasons.append("RSI légèrement élevé")
    elif rsi_val < 35:
        score += 10
        reasons.append("Survente RSI")
    elif rsi_val > 70:
        score += 3
        reasons.append("Surachat RSI")
    else:
        score += 8

    # MACD (15 pts)
    if macd_hist > 0:
        score += 15
        reasons.append("MACD haussier")
    else:
        score += 3
        reasons.append("MACD baissier")

    # Volume (10 pts)
    if vol_ratio > 1.5 and price_up:
        score += 10
        reasons.append("Volume fort haussier")
    elif vol_ratio > 1.2 and price_up:
        score += 7
        reasons.append("Volume élevé haussier")
    elif vol_ratio < 0.7:
        score += 3
        reasons.append("Volume faible")
    else:
        score += 5

    # Bollinger (10 pts)
    if 0.2 <= bb_pct <= 0.8:
        score += 10
        reasons.append("Position Bollinger saine")
    elif bb_pct < 0.1:
        score += 7
        reasons.append("Proche bande basse Bollinger")
    else:
        score += 3

    # Fondamentaux (15 pts)
    fund_pts = 0
    if pe and 5 < pe < 30:
        fund_pts += 5
        reasons.append(f"PE attractif ({pe}x)")
    if rev_growth and rev_growth > 5:
        fund_pts += 5
        reasons.append(f"Croissance revenus +{rev_growth}%")
    if net_margin and net_margin > 10:
        fund_pts += 5
        reasons.append(f"Marge nette solide ({net_margin}%)")
    score += fund_pts

    # Bonus Patterns
    pattern_bonus = row.get("Pattern_Score", 0)
    if pattern_bonus > 0:
        score += pattern_bonus
        top = row.get("Top_Pattern", "")
        if top and top != "—":
            reasons.append(f"Pattern: {top}")

    # Bonus R/R
    rr = row.get("RR_Ratio", None)
    if rr and rr >= 2.5:
        score += 5
        reasons.append(f"R/R excellent ({rr}:1)")
    elif rr and rr >= 2.0:
        score += 3
        reasons.append(f"R/R bon ({rr}:1)")

    return min(score, 100), reasons

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
# 🤖 ANALYSE CLAUDE
# ─────────────────────────────
def claude_analysis(row, api_key):
    try:
        client = anthropic.Anthropic(api_key=api_key)

        pe_str = f"{row['PE']}x"         if row['PE']         else "N/A"
        rg_str = f"{row['Rev_Growth']}%" if row['Rev_Growth'] else "N/A"
        nm_str = f"{row['Net_Margin']}%" if row['Net_Margin'] else "N/A"
        rr_str = f"{row['RR_Ratio']}:1"  if row['RR_Ratio']  else "N/A"

        prompt = f"""Tu es un trader spécialisé en swing trading (lundi achat → vendredi vente).
Analyse ce titre et donne un avis concis en français (6-8 lignes max).

Ticker: {row['Ticker']} | Secteur: {row['Sector']}
Prix: ${row['Prix']} | Entrée: ${row['Entree']} | Stop: ${row['Stop']} | Target: ${row['Target']}
Ratio R/R: {rr_str} | Risque: {row['Risque_Pct']}% | Gain potentiel: {row['Gain_Pct']}%
RSI: {row['RSI']} | MACD Hist: {row['MACD_Hist']} | ATR: {row['ATR_Pct']}%
Volume ratio: {row['Vol_Ratio']}x | PE: {pe_str}
Score IA: {row['AI Score']}/100 | Signal: {row['AI Signal']}
Pattern: {row['Top_Pattern']} | Qualité setup: {row['RR_Quality']}

Donne:
1) VERDICT (ACHETER / ATTENDRE / ÉVITER)
2) Confirmes-tu l'entrée à ${row['Entree']} et le stop à ${row['Stop']} ?
3) Argument principal pour ce swing cette semaine
4) Risque principal à surveiller
Sois direct, chiffré, sans disclaimer."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Analyse indisponible : {e}"

# ─────────────────────────────
# 📦 EXCEL EXPORT
# ─────────────────────────────
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Screener")
    return output.getvalue()

# ─────────────────────────────
# 🚀 SIDEBAR
# ─────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    mode_rapide = st.checkbox("⚡ Mode rapide (100 actions)", value=True)
    nb_workers  = st.slider("🔀 Threads parallèles", 5, 20, 10)

    st.markdown("---")
    st.markdown("### 🤖 Analyse Claude IA")
    api_key    = st.text_input("Clé API Anthropic", type="password",
                                help="Obtenez votre clé sur console.anthropic.com")
    use_claude = st.checkbox("Activer l'analyse Claude", value=False)

    st.markdown("---")
    st.markdown("### 🔍 Filtres")
    min_score     = st.slider("Score IA minimum", 0, 100, 50)
    min_rr        = st.slider("R/R minimum", 0.0, 3.0, 1.5, step=0.1)
    max_risk      = st.slider("Risque max (%)", 1.0, 10.0, 5.0, step=0.5)
    signal_filter = st.multiselect(
        "Signaux à afficher",
        ["🟢 STRONG BUY", "🟢 BUY", "🟡 HOLD", "🔴 AVOID"],
        default=["🟢 STRONG BUY", "🟢 BUY"]
    )
    only_patterns = st.checkbox("🔍 Avec patterns seulement", value=False)
    only_good_rr  = st.checkbox("📐 R/R ≥ 2.0 seulement", value=False)

    st.markdown("---")
    st.markdown("<div style='color:#64748b;font-size:0.75rem;'>S&P 500 IA Screener Pro<br>Données via Yahoo Finance</div>",
                unsafe_allow_html=True)

# ─────────────────────────────
# 🚀 MAIN
# ─────────────────────────────
st.markdown("# 📊 S&P 500 IA Screener Pro")
st.markdown("<div style='color:#64748b;margin-bottom:2rem;'>Analyse technique · Patterns · Risque/Rendement · Propulsé par Claude IA</div>",
            unsafe_allow_html=True)

tickers = SP500_TICKERS[:100] if mode_rapide else SP500_TICKERS
st.markdown(f"<div style='color:#00ff88;margin-bottom:1rem;'>✅ <b>{len(tickers)}</b> actions prêtes à analyser</div>",
            unsafe_allow_html=True)

if st.button("🔄 Lancer l'analyse complète"):

    with st.spinner("Collecte des données en cours..."):
        rows = fetch_parallel(tickers, max_workers=nb_workers)

    if not rows:
        st.error("❌ Aucune donnée récupérée — vérifiez votre connexion")
        st.stop()

    df = pd.DataFrame(rows)

    # Calcul scores
    scores_data  = df.apply(ai_score, axis=1)
    df["AI Score"]   = scores_data.apply(lambda x: x[0])
    df["AI Signal"]  = df["AI Score"].apply(ai_signal)
    df["AI Reasons"] = scores_data.apply(lambda x: " | ".join(x[1]))
    df = df.sort_values("AI Score", ascending=False).reset_index(drop=True)

    # Filtres
    df_filtered = df[df["AI Score"] >= min_score]
    if signal_filter:
        df_filtered = df_filtered[df_filtered["AI Signal"].isin(signal_filter)]
    if only_patterns:
        df_filtered = df_filtered[df_filtered["Pattern_Score"] > 0]
    if only_good_rr:
        df_filtered = df_filtered[df_filtered["RR_Ratio"] >= 2.0]
    df_filtered = df_filtered[
        (df_filtered["RR_Ratio"].isna()) | (df_filtered["RR_Ratio"] >= min_rr)
    ]
    df_filtered = df_filtered[
        (df_filtered["Risque_Pct"].isna()) | (df_filtered["Risque_Pct"] <= max_risk)
    ]

    # ── MÉTRIQUES ──
    st.markdown("---")
    st.markdown("### 📈 Vue d'ensemble")

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    total         = len(df)
    strong_buy    = len(df[df["AI Signal"] == "🟢 STRONG BUY"])
    buy           = len(df[df["AI Signal"] == "🟢 BUY"])
    hold          = len(df[df["AI Signal"] == "🟡 HOLD"])
    avoid         = len(df[df["AI Signal"] == "🔴 AVOID"])
    avec_pattern  = len(df[df["Pattern_Score"] > 0])
    bon_rr        = len(df[df["RR_Ratio"] >= 2.0]) if "RR_Ratio" in df.columns else 0

    for col, val, label in zip(
        [col1, col2, col3, col4, col5, col6, col7],
        [total, strong_buy, buy, hold, avoid, avec_pattern, bon_rr],
        ["Total", "Strong Buy", "Buy", "Hold", "Avoid", "Patterns", "R/R ≥ 2"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    # ── PLANS DE TRADE TOP 5 ──
    st.markdown("---")
    st.markdown("### 📐 Plans de Trade — Top 5 Opportunités")
    st.markdown("<div style='color:#64748b;font-size:0.85rem;margin-bottom:1rem;'>Entrée · Stop-Loss · Target · R/R — calculés par ATR</div>",
                unsafe_allow_html=True)

    top5_rr = df_filtered[df_filtered["RR_Ratio"].notna()].head(5)
    if not top5_rr.empty:
        for _, row in top5_rr.iterrows():
            rr    = row["RR_Ratio"]
            color = "#00ff88" if rr >= 2.5 else "#4ade80" if rr >= 2.0 else "#fbbf24"
            st.markdown(f"""
            <div class="trade-plan-box">
                <span class="ticker-badge">{row['Ticker']}</span>
                <strong style="color:{color}">{row['RR_Badge']}</strong>
                &nbsp;|&nbsp; Score: <strong>{row['AI Score']}/100</strong>
                &nbsp;|&nbsp; {row['RR_Quality']}
                <br><br>
                🎯 <strong>Entrée :</strong> ${row['Entree']}
                &nbsp;&nbsp;
                🛑 <strong>Stop :</strong> ${row['Stop']} <span style="color:#f87171">(-{row['Risque_Pct']}%)</span>
                &nbsp;&nbsp;
                🏆 <strong>Target :</strong> ${row['Target']} <span style="color:#00ff88">(+{row['Gain_Pct']}%)</span>
                <br>
                📊 <strong>R/R :</strong> {rr}:1
                &nbsp;|&nbsp; ATR : ${row['ATR']} ({row['ATR_Pct']}%)
                &nbsp;|&nbsp; Support : ${row['Support']}
                &nbsp;|&nbsp; Résistance : ${row['Resistance']}
                <br>
                <span style="color:#94a3b8;font-size:0.8rem;">Pattern : {row['Top_Pattern']}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Aucun plan R/R disponible avec les filtres actuels.")

    # ── PATTERNS ──
    st.markdown("---")
    st.markdown("### 🔍 Patterns Swing Détectés")

    top_patterns = df[df["Pattern_Score"] > 0].sort_values("Pattern_Score", ascending=False).head(8)
    if not top_patterns.empty:
        for _, row in top_patterns.iterrows():
            st.markdown(f"""
            <div class="pattern-box">
                <span class="ticker-badge">{row['Ticker']}</span>
                <strong style="color:#fbbf24">{row['Pattern_Badge']}</strong>
                &nbsp;|&nbsp; +{row['Pattern_Score']} pts
                &nbsp;|&nbsp; Score total : <strong>{row['AI Score']}/100</strong>
                <br>
                <span style="color:#94a3b8;font-size:0.8rem;">{row['Patterns']}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Aucun pattern détecté.")

    # ── GRAPHIQUES ──
    st.markdown("---")
    st.markdown("### 📊 Visualisations")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distribution scores", "RSI vs Score", "Top 10", "Patterns", "R/R Ratio"
    ])

    with tab1:
        import plotly.express as px
        fig = px.histogram(df, x="AI Score", nbins=20,
                           color_discrete_sequence=["#00ff88"],
                           title="Distribution des scores IA")
        fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                          font_color="#e2e8f0", title_font_color="#00ff88",
                          xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = px.scatter(df, x="RSI", y="AI Score", color="AI Signal",
                          hover_data=["Ticker", "Sector", "Top_Pattern", "RR_Ratio"],
                          color_discrete_map={
                              "🟢 STRONG BUY": "#00ff88", "🟢 BUY": "#4ade80",
                              "🟡 HOLD": "#fbbf24", "🔴 AVOID": "#f87171"
                          }, title="RSI vs Score IA")
        fig2.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                           font_color="#e2e8f0", title_font_color="#00ff88",
                           xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        top10 = df.head(10)
        fig3 = px.bar(top10, x="Ticker", y="AI Score", color="AI Score",
                      color_continuous_scale=["#1e3a5f", "#00ff88"],
                      title="Top 10 — Score IA")
        fig3.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                           font_color="#e2e8f0", title_font_color="#00ff88",
                           xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        df_pat = df[df["Pattern_Score"] > 0].sort_values("Pattern_Score", ascending=False).head(15)
        if not df_pat.empty:
            fig4 = px.bar(df_pat, x="Ticker", y="Pattern_Score", color="Pattern_Score",
                          color_continuous_scale=["#1e3a5f", "#fbbf24"],
                          hover_data=["Top_Pattern", "AI Score"],
                          title="Top Patterns — Score bonus")
            fig4.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                               font_color="#e2e8f0", title_font_color="#00ff88",
                               xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
            st.plotly_chart(fig4, use_container_width=True)

    with tab5:
        df_rr = df[df["RR_Ratio"].notna()].sort_values("RR_Ratio", ascending=False).head(15)
        if not df_rr.empty:
            fig5 = px.bar(df_rr, x="Ticker", y="RR_Ratio", color="RR_Ratio",
                          color_continuous_scale=["#f87171", "#fbbf24", "#00ff88"],
                          hover_data=["Entree", "Stop", "Target", "Risque_Pct", "Gain_Pct"],
                          title="Top R/R Ratio")
            fig5.add_hline(y=2.0, line_dash="dash", line_color="#fbbf24",
                           annotation_text="R/R min recommandé (2:1)")
            fig5.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                               font_color="#e2e8f0", title_font_color="#00ff88",
                               xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
            st.plotly_chart(fig5, use_container_width=True)

    # ── TABLEAU ──
    st.markdown("---")
    st.markdown(f"### 🏆 Résultats ({len(df_filtered)} actions)")

    cols_display = [
        "Ticker", "Sector", "Prix",
        "Entree", "Stop", "Target", "RR_Ratio", "Risque_Pct", "Gain_Pct",
        "RR_Badge", "RR_Quality",
        "RSI", "MACD_Hist", "Vol_Ratio",
        "Pattern_Badge", "Top_Pattern", "Pattern_Score",
        "AI Score", "AI Signal", "AI Reasons"
    ]
    cols_display = [c for c in cols_display if c in df_filtered.columns]
    st.dataframe(df_filtered[cols_display], use_container_width=True, height=400)

    # ── ANALYSE CLAUDE ──
    if use_claude and api_key:
        st.markdown("---")
        st.markdown("### 🤖 Analyse Claude IA — Top 5")
        st.markdown("<div style='color:#64748b;font-size:0.85rem;margin-bottom:1rem;'>Analyse swing + validation plan de trade par Claude</div>",
                    unsafe_allow_html=True)

        top5 = df_filtered.head(5)
        for _, row in top5.iterrows():
            with st.spinner(f"Claude analyse {row['Ticker']}..."):
                analysis = claude_analysis(row, api_key)
            rr = row.get("RR_Ratio", "N/A")
            st.markdown(f"""
            <div class="ai-analysis-box">
                <span class="ticker-badge">{row['Ticker']}</span>
                <strong style="color:#00ff88">{row['AI Signal']}</strong>
                — Score {row['AI Score']}/100
                &nbsp;|&nbsp; <span style="color:#4ade80">R/R {rr}:1</span>
                &nbsp;|&nbsp; <span style="color:#fbbf24">{row['Pattern_Badge']}</span>
                <br><br>{analysis}
            </div>""", unsafe_allow_html=True)
            time.sleep(0.5)

    elif use_claude and not api_key:
        st.warning("⚠️ Entrez votre clé API Anthropic dans la barre latérale.")

    # ── EXPORT ──
    st.markdown("---")
    st.markdown("### 📥 Export")
    excel_cols = [c for c in cols_display if c in df_filtered.columns]
    excel = to_excel(df_filtered[excel_cols])
    st.download_button(
        "⬇️ Télécharger Excel",
        data=excel,
        file_name=f"screener_pro_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
