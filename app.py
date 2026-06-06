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
from risk_manager import calc_risk_reward, risk_badge
from market_filter import get_market_status, apply_market_filter, market_advice
from pre_filter import run_prefilter, PREFILTER_CONFIG
from advanced_indicators import detect_advanced_signals
from convergence import calc_convergence, build_trade_report, get_day_of_week_advice
from volume_signals import detect_volume_anomaly

# ─────────────────────────────
# 🎨 PAGE CONFIG
# ─────────────────────────────
st.set_page_config(
    page_title="S&P 500 IA Screener Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }
h1, h2, h3 { font-family: 'Space Mono', monospace; color: #00ff88 !important; }
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
    border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 20px; text-align: center; margin: 4px;
}
.metric-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: #00ff88; }
.metric-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }
.stButton > button {
    background: linear-gradient(135deg, #00ff88, #00cc6a) !important;
    color: #0a0e1a !important; font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important; border: none !important;
    border-radius: 8px !important; padding: 12px 24px !important;
}
.market-banner { border-radius: 10px; padding: 16px 20px; margin-bottom: 20px; font-size: 0.9rem; line-height: 1.8; }
.prefilter-banner {
    background: #0d1a2a; border: 1px solid #1e3a5f; border-left: 4px solid #4a90d0;
    border-radius: 8px; padding: 12px 18px; margin: 10px 0;
    font-size: 0.85rem; font-family: 'Space Mono', monospace;
}
.trade-card {
    background: linear-gradient(135deg, #0a1628 0%, #0f2040 100%);
    border: 1px solid #1e4060; border-radius: 14px;
    padding: 20px 24px; margin: 10px 0;
    position: relative; overflow: hidden;
}
.trade-card-gold {
    background: linear-gradient(135deg, #1a1400 0%, #2a2000 100%);
    border: 2px solid #ffd70066; border-radius: 14px;
    padding: 20px 24px; margin: 10px 0;
    box-shadow: 0 0 20px #ffd70022;
}
.trade-card-green {
    background: linear-gradient(135deg, #001a0f 0%, #002a18 100%);
    border: 1px solid #00ff8844; border-radius: 14px;
    padding: 20px 24px; margin: 10px 0;
}
.signal-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.signal-pill {
    padding: 3px 10px; border-radius: 20px; font-size: 11px;
    font-family: 'Space Mono', monospace;
}
.conv-bar { font-family: 'Space Mono', monospace; font-size: 18px; letter-spacing: 2px; }
.ai-analysis-box {
    background: linear-gradient(135deg, #0f1f35 0%, #0a1628 100%);
    border: 1px solid #00ff8844; border-left: 4px solid #00ff88;
    border-radius: 8px; padding: 16px 20px; margin: 8px 0; font-size: 0.9rem; line-height: 1.6;
}
.advice-box { background: #0d1117; border: 1px solid #1e3a5f; border-radius: 8px; padding: 12px 16px; margin: 6px 0; font-size: 0.85rem; }
.ticker-badge {
    display: inline-block; background: #00ff8822; border: 1px solid #00ff8866;
    color: #00ff88; font-family: 'Space Mono', monospace; font-size: 0.9rem;
    padding: 3px 12px; border-radius: 4px; margin-right: 8px; font-weight: 700;
}
.rank-badge {
    display: inline-block; font-family: 'Space Mono', monospace;
    font-size: 1.4rem; font-weight: 700; color: #64748b; margin-right: 8px;
}
div[data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 10px; overflow: hidden; }
.stProgress > div > div { background: linear-gradient(90deg, #00ff88, #00cc6a) !important; }
section[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1e3a5f; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# 📌 S&P 500 COMPLET
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
# 📊 INDICATEURS DE BASE
# ─────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = -delta.where(delta < 0, 0).rolling(period).mean()
    rs    = gain / loss.clip(lower=1e-10)
    return float(100 - (100 / (1 + rs.iloc[-1])))

def calc_macd(series):
    ema12       = series.ewm(span=12, adjust=False).mean()
    ema26       = series.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float((macd_line - signal_line).iloc[-1])

def calc_bollinger(series, period=20):
    ma    = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    price = series.iloc[-1]
    pct_b = float((price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]))
    return pct_b

def calc_volume_signal(volume, close):
    avg_vol   = volume.rolling(20).mean().iloc[-1]
    last_vol  = volume.iloc[-1]
    vol_ratio = float(last_vol / avg_vol) if avg_vol > 0 else 1.0
    return vol_ratio

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

        price     = float(close.iloc[-1])
        ma50      = float(close.rolling(50).mean().iloc[-1])
        ma200     = float(close.rolling(200).mean().iloc[-1])
        rsi       = calc_rsi(close)
        macd_line, macd_signal, macd_hist = calc_macd(close)
        bb_pct    = calc_bollinger(close)
        vol_ratio = calc_volume_signal(volume, close)

        patterns_data = detect_all_patterns(hist)
        rr_data       = calc_risk_reward(hist)
        adv           = detect_advanced_signals(hist)
        vol_anom      = detect_volume_anomaly(hist)

        info       = t.info
        revenue_gr = info.get("revenueGrowth", None)
        sector     = info.get("sector", "N/A")

        return {
            "Ticker":        ticker,
            "Sector":        sector,
            "Prix":          round(price, 2),
            "MA50":          round(ma50, 2),
            "MA200":         round(ma200, 2),
            "RSI":           round(rsi, 1),
            "MACD_Hist":     round(macd_hist, 3),
            "BB_Pct":        round(bb_pct, 2),
            "Vol_Ratio":     round(vol_ratio, 2),
            "Rev_Growth":    round(revenue_gr * 100, 1) if revenue_gr else None,
            # Patterns
            "Top_Pattern":   patterns_data["top_pattern"],
            "Patterns":      patterns_data["summary"],
            "Pattern_Score": patterns_data["bonus_score"],
            "Pattern_Badge": pattern_badge(patterns_data["bonus_score"]),
            # R/R
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
            "RR_Badge":      risk_badge(rr_data["rr_ratio"], rr_data["risk_pct"]),
            # Avancés
            "TTM_Signal":    adv["ttm"]["signal"],
            "TTM_Score":     adv["ttm"]["score"],
            "TTM_Status":    adv["ttm"]["status"],
            "DIV_Signal":    adv["div"]["signal"],
            "DIV_Score":     adv["div"]["score"],
            "DIV_Type":      adv["div"]["type"],
            "EMA_Signal":    adv["ema"]["signal"],
            "EMA_Score":     adv["ema"]["score"],
            "EMA_Level":     adv["ema"]["level"],
            "EMA8_Slope":    adv["ema"].get("ema8_slope", 0),
            "ADV_Score":     adv["total_score"],
            "ADV_Badge":     adv["badge"],
            "ADV_Summary":   adv["summary"],
            "ADV_Active":    adv["n_active"],
            # Volume anormal
            "VOL_Score":     vol_anom["score"],
            "VOL_Badge":     vol_anom["badge"],
            "VOL_Signal":    vol_anom["top_signal"],
            "VOL_Ratio":     vol_anom["vol_ratio"],
            "VOL_52W_Rank":  vol_anom["vol_52w_rank"],
            "VOL_Bullish":   vol_anom["is_bullish"],
            "VOL_Summary":   vol_anom["summary"],
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
            status.markdown(f"🔬 Analyse `{done}/{len(tickers)}`...")
        status.empty()
    return results

# ─────────────────────────────
# 🧠 SCORE IA — NETTOYÉ
# ─────────────────────────────
def ai_score(row):
    score   = 0
    reasons = []

    try:
        price     = float(row.get("Prix", 0) or 0)
        ma50      = float(row.get("MA50", 0) or 0)
        ma200     = float(row.get("MA200", 0) or 0)
        rsi_val   = float(row.get("RSI", 50) or 50)
        macd_hist = float(row.get("MACD_Hist", 0) or 0)
        vol_ratio = float(row.get("Vol_Ratio", 1) or 1)
        rev_growth = row.get("Rev_Growth", None)
    except Exception:
        return 0, ["Erreur calcul"]

    # Trend MA (35 pts)
    if price > ma50 > ma200:
        score += 35; reasons.append("✅ Trend forte (prix>MA50>MA200)")
    elif price > ma50 and price > ma200:
        score += 25; reasons.append("✅ Prix > MA50 & MA200")
    elif price > ma200:
        score += 15; reasons.append("~ Au-dessus MA200")
    else:
        score += 0;  reasons.append("❌ Sous MAs")

    # RSI (25 pts)
    if 45 <= rsi_val <= 65:
        score += 25; reasons.append(f"✅ RSI idéal ({rsi_val})")
    elif 35 <= rsi_val < 45:
        score += 18; reasons.append(f"~ RSI rebond ({rsi_val})")
    elif 65 < rsi_val <= 72:
        score += 15; reasons.append(f"~ RSI momentum ({rsi_val})")
    elif rsi_val < 35:
        score += 10; reasons.append(f"⚠️ RSI survente ({rsi_val})")
    else:
        score += 5;  reasons.append(f"❌ RSI surachat ({rsi_val})")

    # MACD (20 pts)
    if macd_hist > 0.3:
        score += 20; reasons.append(f"✅ MACD fort ({round(macd_hist,3)})")
    elif macd_hist > 0:
        score += 14; reasons.append(f"~ MACD haussier ({round(macd_hist,3)})")
    elif macd_hist > -0.3:
        score += 5;  reasons.append("~ MACD neutre")
    else:
        score += 0;  reasons.append(f"❌ MACD baissier")

    # Volume (20 pts)
    if vol_ratio >= 2.0:
        score += 20; reasons.append(f"✅ Volume très fort ({vol_ratio}x)")
    elif vol_ratio >= 1.5:
        score += 15; reasons.append(f"✅ Volume fort ({vol_ratio}x)")
    elif vol_ratio >= 1.1:
        score += 10; reasons.append(f"~ Volume correct ({vol_ratio}x)")
    elif vol_ratio < 0.7:
        score += 2;  reasons.append(f"❌ Volume faible ({vol_ratio}x)")
    else:
        score += 6

    # Momentum fondamental (5 pts max)
    try:
        if rev_growth and float(rev_growth) > 10:
            score += 5; reasons.append(f"✅ Croissance +{rev_growth}%")
        elif rev_growth and float(rev_growth) > 5:
            score += 2; reasons.append(f"~ Croissance +{rev_growth}%")
    except Exception:
        pass

    # Bonus patterns (max 30 pts)
    try:
        pb = int(row.get("Pattern_Score", 0) or 0)
        if pb > 0:
            score += min(pb, 30)
            top = str(row.get("Top_Pattern", "") or "")
            if top and top != "—":
                reasons.append(f"✅ Pattern: {top}")
    except Exception:
        pass

    # Bonus volume anormal (max 25 pts)
    try:
        vol_score   = int(row.get("VOL_Score", 0) or 0)
        vol_bullish = bool(row.get("VOL_Bullish", True))
        vol_signal  = str(row.get("VOL_Signal") or "")
        if vol_score > 0 and vol_bullish:
            score += min(vol_score, 25)
            if vol_signal and vol_signal not in ["None", "—", ""]:
                reasons.append(f"✅ {vol_signal[:40]}")
    except Exception:
        pass

    # Bonus avancés (max 30 pts)
    try:
        adv = int(row.get("ADV_Score", 0) or 0)
        if adv > 0:
            score += min(adv, 30)
            for sig_key in ["TTM_Signal", "DIV_Signal", "EMA_Signal"]:
                s = str(row.get(sig_key) or "")
                if s and s not in ["None", "—", ""]:
                    reasons.append(f"✅ {s[:40]}")
    except Exception:
        pass

    return min(score, 100), reasons

def ai_signal(score):
    if score >= 85:   return "🟢 STRONG BUY"
    elif score >= 70: return "🟢 BUY"
    elif score >= 50: return "🟡 HOLD"
    else:             return "🔴 AVOID"

# ─────────────────────────────
# 🤖 ANALYSE CLAUDE
# ─────────────────────────────
def claude_analysis(row, api_key, market_status):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        regime = market_status.get("regime", "INCONNU")
        rr_str = f"{row.get('RR_Ratio','N/A')}:1"

        prompt = f"""Tu es un trader spécialisé en swing trading (lundi -> vendredi).
Marché : {regime} | SPY vs MA50: {market_status.get('spy_vs_ma50','N/A')}% | {market_status.get('vix_label','VIX N/A')}

Ticker: {row['Ticker']} ({row['Sector']})
Convergence: {row.get('Conv_N','N/A')}/6 signaux | Score final: {row.get('Score_Final','N/A')}/100
Prix: ${row['Prix']} | Entree: ${row.get('Entree','N/A')} | Stop: ${row.get('Stop','N/A')} | Target: ${row.get('Target','N/A')}
R/R: {rr_str} | Risque: {row.get('Risque_Pct','N/A')}% | Gain: {row.get('Gain_Pct','N/A')}%
RSI: {row['RSI']} | MACD: {row['MACD_Hist']} | Vol: {row['Vol_Ratio']}x
TTM: {row.get('TTM_Signal','—')} | Div RSI: {row.get('DIV_Signal','—')} | EMA: {row.get('EMA_Level','—')}
Pattern: {row.get('Top_Pattern','—')}
Signaux actifs: {row.get('Conv_On','—')}

En 6 lignes max :
1) VERDICT (ACHETER/ATTENDRE/EVITER)
2) Confirmes-tu entree ${row.get('Entree','N/A')} / stop ${row.get('Stop','N/A')} ?
3) Argument principal basé sur la convergence des signaux
4) Risque principal cette semaine
Direct, chiffré, sans disclaimer."""

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

    st.markdown("### ⚡ Mode de scan")
    use_prefilter = st.checkbox("Pré-filtre S&P 500 complet (503)", value=True)
    if use_prefilter:
        st.markdown("<div style='color:#4a90d0;font-size:0.78rem;'>✅ 503 actions · 2 passes automatiques</div>", unsafe_allow_html=True)
        with st.expander("⚙️ Critères pré-filtre"):
            min_price    = st.number_input("Prix min ($)", value=10, min_value=1)
            max_price    = st.number_input("Prix max ($)", value=2000, min_value=50)
            min_volume   = st.number_input("Volume moyen min", value=500000, step=100000)
            min_momentum = st.slider("Momentum 20j min (%)", -10, 5, -2)
            above_ma50   = st.checkbox("Au-dessus MA50", value=True)
            PREFILTER_CONFIG.update({
                "min_price": min_price, "max_price": max_price,
                "min_volume": min_volume, "min_momentum_20d": min_momentum,
                "require_above_ma50": above_ma50,
            })

    st.markdown("---")
    nb_workers = st.slider("🔀 Threads parallèles", 5, 20, 10)

    st.markdown("---")
    st.markdown("### 🎯 Rapport Top Trades")
    top_n       = st.radio("Nombre de trades", [10, 20], index=0, horizontal=True)

    # Mode strict — améliore le win rate
    strict_mode = st.checkbox("🔒 Mode strict (win rate optimisé)", value=True,
                              help="Force min 4/6 signaux + R/R 2.0 + score 70+")

    if strict_mode:
        min_signals  = 4
        min_rr_conv  = 2.0
        min_score    = 70
        st.markdown("""<div style='background:#00ff8812;border:1px solid #00ff8833;
            border-radius:6px;padding:10px;font-size:0.78rem;color:#86efac;margin-top:6px;'>
            🔒 <strong>Mode strict actif</strong><br>
            ✅ Min 4/6 signaux convergents<br>
            ✅ R/R minimum 2.0:1<br>
            ✅ Score ajusté minimum 70<br>
            ✅ Gap ouverture max 1.5%
        </div>""", unsafe_allow_html=True)
    else:
        min_signals  = st.slider("Signaux convergents min", 2, 6, 3,
                                 help="4+ = recommandé pour win rate optimal")
        min_rr_conv  = st.slider("R/R minimum", 1.0, 3.0, 1.5, step=0.1)
        min_score    = st.slider("Score min", 0, 100, 50)

    # Indicateur visuel win rate estimé
    wr_est = {2:"~40%", 3:"~48%", 4:"~58%", 5:"~67%", 6:"~75%"}
    sig_display = min_signals
    st.markdown(
        f"<div style='color:#fbbf24;font-size:0.8rem;margin-top:4px;'>"
        f"📊 Win rate estimé : <strong>{wr_est.get(sig_display,'—')}</strong> "
        f"avec {sig_display}/6 signaux</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### 🤖 Claude IA")
    api_key    = st.text_input("Clé API Anthropic", type="password")
    use_claude = st.checkbox("Activer analyse Claude", value=False)

    st.markdown("---")
    st.markdown("### 🔍 Filtres tableau")
    signal_filter = st.multiselect(
        "Signaux",
        ["🟢 STRONG BUY","🟢 BUY","🟡 HOLD","🔴 AVOID","🟡 HOLD ⚠️"],
        default=["🟢 STRONG BUY","🟢 BUY"]
    )

    st.markdown("---")
    st.markdown("<div style='color:#64748b;font-size:0.75rem;'>S&P 500 IA Screener Pro</div>", unsafe_allow_html=True)

# ─────────────────────────────
# 🚀 MAIN
# ─────────────────────────────
st.markdown("# 📊 S&P 500 IA Screener Pro")
st.markdown("<div style='color:#64748b;margin-bottom:1.5rem;'>Convergence · Patterns · R/R · TTM · Divergence · EMA · Marché · Claude IA</div>", unsafe_allow_html=True)

# BANDEAU MARCHÉ
with st.spinner("Vérification marché global..."):
    market_status = get_market_status()

regime = market_status["regime"]
color  = market_status["color"]
emoji  = market_status["emoji"]

st.markdown(f"""
<div class="market-banner" style="background:{color}11;border:1px solid {color}44;border-left:5px solid {color};">
    <strong style="color:{color};font-size:1.1rem;">{emoji} MARCHÉ {regime}</strong>
    &nbsp;—&nbsp; {market_status['message']}
    <br><span style="color:#94a3b8;font-size:0.82rem;font-family:'Space Mono',monospace;">{market_status['detail']}</span>
</div>
""", unsafe_allow_html=True)

with st.expander("💡 Conseils de trading"):
    advice_list = market_advice(market_status)
    cols = st.columns(2)
    for i, adv in enumerate(advice_list):
        cols[i%2].markdown(f"<div class='advice-box'>{adv}</div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    spy_color = "#00ff88" if market_status.get("spy_vs_ma50",0)>=0 else "#f87171"
    qqq_color = "#00ff88" if market_status.get("qqq_vs_ma50",0)>=0 else "#f87171"
    vix_val   = market_status.get("vix",None)
    vix_color = "#00ff88" if vix_val and vix_val<20 else "#fbbf24" if vix_val and vix_val<30 else "#f87171"
    for col, val, label, clr in [
        (c1, f"{'+' if market_status.get('spy_vs_ma50',0)>=0 else ''}{market_status.get('spy_vs_ma50','—')}%", "SPY vs MA50", spy_color),
        (c2, f"{'+' if market_status.get('qqq_vs_ma50',0)>=0 else ''}{market_status.get('qqq_vs_ma50','—')}%", "QQQ vs MA50", qqq_color),
        (c3, vix_val if vix_val else "—", "VIX", vix_color),
        (c4, market_status.get("spy_rsi","—"), "RSI SPY", "#00ff88"),
    ]:
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{clr}">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 📊 SECTION BACKTEST
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Validation — Backtest du Système")
st.markdown("<div style='color:#64748b;font-size:0.85rem;margin-bottom:1rem;'>Teste les signaux sur 52 semaines de données historiques réelles · Distinct du scan hebdomadaire</div>", unsafe_allow_html=True)

with st.expander("🔬 Lancer le Backtest", expanded=False):
    bt_col1, bt_col2 = st.columns(2)
    with bt_col1:
        bt_tickers_raw = st.text_area(
            "Tickers à backtester (un par ligne)",
            value="AAPL\nMSFT\nNVDA\nGOOGL\nMETA\nAMZN\nTSLA\nAVGO\nJPM\nV",
            height=180
        )
        bt_tickers = [t.strip().upper() for t in bt_tickers_raw.strip().split("\n") if t.strip()]
    with bt_col2:
        bt_workers = st.slider("Threads backtest", 3, 10, 5)
        st.markdown(f"**{len(bt_tickers)} tickers** × ~52 semaines = ~**{len(bt_tickers)*52} trades simulés**")
        st.markdown(f"Durée estimée : ~**{max(1, len(bt_tickers)//5)} minutes**")
        st.markdown("**Logique :** Entrée lundi · Stop ATR×1.5 · Target ATR×3 · Sortie vendredi si aucun niveau atteint")

    if st.button("▶ Lancer le Backtest", key="bt_btn"):
        bt_prog = st.progress(0)
        bt_stat = st.empty()
        def bt_cb(done, total):
            bt_prog.progress(done / total)
            bt_stat.markdown(f"🔬 Backtest `{done}/{total}` tickers...")

        df_bt = run_backtest(bt_tickers, weeks=52, max_workers=bt_workers, progress_callback=bt_cb)
        bt_stat.empty()

        if df_bt.empty:
            st.error("❌ Aucun trade simulé.")
        else:
            stats = backtest_summary(df_bt)
            st.markdown("### 📈 Résultats")

            # Métriques
            m1,m2,m3,m4,m5,m6 = st.columns(6)
            wc  = "#00ff88" if stats["win_rate"]>=55 else "#fbbf24" if stats["win_rate"]>=45 else "#f87171"
            pfc = "#00ff88" if stats["profit_factor"]>=1.5 else "#fbbf24" if stats["profit_factor"]>=1.0 else "#f87171"
            exc = "#00ff88" if stats["expectancy"]>0 else "#f87171"
            for col, val, label, clr in [
                (m1, stats['total'],         "Trades", "#4a90d0"),
                (m2, f"{stats['win_rate']}%","Win Rate", wc),
                (m3, stats['profit_factor'], "Profit Factor", pfc),
                (m4, f"{stats['expectancy']}%","Expectancy", exc),
                (m5, f"+{stats['avg_win']}%","Gain moyen", "#00ff88"),
                (m6, f"{stats['avg_loss']}%","Perte moyenne","#f87171"),
            ]:
                col.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:{clr};font-size:1.3rem;">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            s1,s2,s3,s4 = st.columns(4)
            s1.metric("Meilleur trade",  f"+{stats['best_trade']}%")
            s2.metric("Pire trade",      f"{stats['worst_trade']}%")
            s3.metric("Pertes consécutives max", stats["max_consec_loss"])
            s4.metric("PnL cumulé total", f"{stats['total_pnl']}%")

            # Stats par score et signaux
            st.markdown("---")
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("**📊 Par niveau de score**")
                for label, data in stats["score_stats"].items():
                    wr = data["win_rate"]
                    ic = "🟢" if wr>=55 else "🟡" if wr>=45 else "🔴"
                    st.markdown(f"{ic} **{label}** — Win: `{wr}%` | PnL moy: `{data['avg_pnl']}%` | N=`{data['n']}`")
            with sc2:
                st.markdown("**🎯 Par signaux convergents**")
                for label, data in stats["signal_stats"].items():
                    wr = data["win_rate"]
                    ic = "🟢" if wr>=55 else "🟡" if wr>=45 else "🔴"
                    st.markdown(f"{ic} **{label}** — Win: `{wr}%` | PnL moy: `{data['avg_pnl']}%` | N=`{data['n']}`")

            # Graphiques
            st.markdown("---")
            import plotly.express as px
            import plotly.graph_objects as go

            bt_tab1, bt_tab2, bt_tab3 = st.tabs(["Distribution PnL","Win Rate par score","Courbe de capital"])

            with bt_tab1:
                # Histogramme natif Streamlit
                wins_data  = df_bt[df_bt["result"]=="WIN"]["pnl_pct"]
                loss_data  = df_bt[df_bt["result"]=="LOSS"]["pnl_pct"]
                st.markdown("**Gains (WIN)**")
                st.bar_chart(wins_data.value_counts(bins=15).sort_index())
                st.markdown("**Pertes (LOSS)**")
                st.bar_chart(loss_data.value_counts(bins=15).sort_index())

            with bt_tab2:
                df_bt["score_bucket"] = pd.cut(df_bt["score"],
                    bins=[0,40,50,60,70,80,101],
                    labels=["<40","40-50","50-60","60-70","70-80",">=80"])
                wr_by_score = df_bt.groupby("score_bucket", observed=True).apply(
                    lambda x: round(len(x[x["result"]=="WIN"])/len(x)*100, 1)
                ).reset_index()
                wr_by_score.columns = ["Score","Win Rate %"]
                wr_by_score = wr_by_score.set_index("Score")
                st.markdown("**Win Rate % par niveau de score** *(seuil rentable = 50%)*")
                st.bar_chart(wr_by_score)

            with bt_tab3:
                df_sorted = df_bt.sort_values("week").copy()
                df_sorted["PnL cumulé %"] = df_sorted["pnl_pct"].cumsum()
                df_sorted = df_sorted.reset_index(drop=True)
                st.markdown("**Courbe de capital — PnL cumulé %**")
                st.line_chart(df_sorted["PnL cumulé %"])

            # Export
            bt_excel = BytesIO()
            with pd.ExcelWriter(bt_excel, engine="openpyxl") as writer:
                df_bt.to_excel(writer, index=False, sheet_name="Trades")
            st.download_button("⬇️ Télécharger résultats backtest", data=bt_excel.getvalue(),
                file_name=f"backtest_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
if st.button(f"🔄 Lancer — S&P 500 complet ({len(SP500_TICKERS)} actions)"):

    tickers_to_analyze = SP500_TICKERS

    # PASSE 1 : PRÉ-FILTRE
    if use_prefilter:
        st.markdown("### ⚡ Passe 1 — Pré-filtre rapide")
        pf_prog = st.progress(0)
        pf_stat = st.empty()
        def pf_cb(done, total):
            pf_prog.progress(done/total)
            pf_stat.markdown(f"⚡ Pré-filtre `{done}/{total}`...")
        pf_result = run_prefilter(SP500_TICKERS, max_workers=20, progress_callback=pf_cb)
        pf_stat.empty()
        tickers_to_analyze = pf_result["passed"]
        st.markdown(f"""<div class="prefilter-banner">
            ⚡ PASSE 1 TERMINÉE &nbsp;|&nbsp;
            <span style="color:#00ff88">{pf_result['n_passed']} retenues</span>
            &nbsp;/&nbsp; {pf_result['total']}
            &nbsp;|&nbsp; {pf_result['n_rejected']} éliminées
            &nbsp;|&nbsp; {pf_result['pass_rate']}% passage
        </div>""", unsafe_allow_html=True)
        with st.expander(f"🗑️ {pf_result['n_rejected']} actions éliminées"):
            st.dataframe(pd.DataFrame([
                {"Ticker": t, "Raison": pf_result['details'][t]}
                for t in pf_result['rejected']
            ]), use_container_width=True, height=200)

    if not tickers_to_analyze:
        st.error("❌ Aucune action ne passe le pré-filtre.")
        st.stop()

    # PASSE 2 : ANALYSE COMPLÈTE
    st.markdown(f"### 🔬 Passe 2 — Analyse complète ({len(tickers_to_analyze)} actions)")
    rows = fetch_parallel(tickers_to_analyze, max_workers=nb_workers)

    if not rows:
        st.error("❌ Aucune donnée récupérée.")
        st.stop()

    df = pd.DataFrame(rows)

    scores_data      = df.apply(ai_score, axis=1)
    df["AI Score"]   = scores_data.apply(lambda x: x[0])
    df["AI Signal"]  = df["AI Score"].apply(ai_signal)
    df["AI Reasons"] = scores_data.apply(lambda x: " | ".join(x[1]))

    df = apply_market_filter(df, market_status)
    df.rename(columns={"AI Signal Ajusté":"AI Signal Ajuste","AI Score Ajusté":"AI Score Ajuste"}, errors="ignore", inplace=True)
    if "AI Score Ajuste" not in df.columns:
        df["AI Score Ajuste"] = df["AI Score"]
    if "AI Signal Ajuste" not in df.columns:
        df["AI Signal Ajuste"] = df["AI Signal"]
    df = df.sort_values("AI Score Ajuste", ascending=False).reset_index(drop=True)

    # ════════════════════════════════════════
    # 🎯 RAPPORT DE CONVERGENCE — TOP TRADES
    # ════════════════════════════════════════
    st.markdown("---")

    # Bannière mode strict
    if strict_mode:
        st.markdown("""<div style='background:#00ff8812;border:1px solid #00ff8844;
            border-left:4px solid #00ff88;border-radius:8px;padding:12px 16px;margin-bottom:12px;
            font-size:0.85rem;color:#86efac;'>
            🔒 <strong>MODE STRICT ACTIF</strong> — Min 4/6 signaux · R/R ≥ 2.0 · Score ≥ 70
            — Optimisé pour maximiser le win rate
        </div>""", unsafe_allow_html=True)

    st.markdown(f"## 🎯 Rapport du Dimanche — Top {top_n} Trades Convergents")
    st.markdown(f"<div style='color:#64748b;font-size:0.85rem;margin-bottom:1rem;'>Semaine du {datetime.now().strftime('%d %B %Y')} · Marché {regime} · Min {min_signals}/6 signaux</div>", unsafe_allow_html=True)

    # Conseils d'exécution semaine
    day_advice = get_day_of_week_advice(regime)
    with st.expander("📅 Plan d'exécution de la semaine"):
        cols = st.columns(3)
        days = [("Lundi","🟢"), ("Mercredi","🟡"), ("Vendredi","🔴")]
        for i, (day, em) in enumerate(days):
            cols[i].markdown(f"""<div class="advice-box">
                <strong>{em} {day}</strong><br>{day_advice.get(day,'—')}
            </div>""", unsafe_allow_html=True)

    # Score minimum selon mode
    score_min_rapport = 70 if strict_mode else min_score

    # Construire le rapport de convergence
    report = build_trade_report(
        df,
        top_n=top_n,
        min_signals=min_signals,
        min_rr=min_rr_conv
    )

    # Filtres strict sur le rapport final
    if not report.empty:
        ai_col = "AI Score Ajuste" if "AI Score Ajuste" in report.columns else "AI Score"
        report = report[report[ai_col] >= score_min_rapport]
        # En mode strict : exiger minimum 4/6
        report = report[report["Conv_N"] >= min_signals]

    if report.empty:
        st.warning(f"⚠️ Aucun titre avec {min_signals}+ signaux convergents. Réduire le filtre dans la sidebar.")
    else:
        st.markdown(f"<div style='color:#00ff88;font-size:0.85rem;margin-bottom:16px;'>✅ {len(report)} opportunités identifiées</div>", unsafe_allow_html=True)

        for idx, row in report.iterrows():
            rank      = idx + 1
            n_sig     = int(row.get("Conv_N", 0))
            conv_bar  = str(row.get("Conv_Bar", "░░░░░░"))
            conv_lbl  = str(row.get("Conv_Label", ""))
            conv_clr  = str(row.get("Conv_Color", "#64748b"))
            score_fin = row.get("Score_Final", 0)
            rr        = row.get("RR_Ratio", None)
            rr_str    = str(rr) if rr else "—"
            rr_color  = "#00ff88" if rr and float(rr) >= 2 else "#fbbf24"
            signals_on_list  = str(row.get("Conv_On", "") or "").split(" | ")
            signals_off_list = str(row.get("Conv_Off", "") or "").split(" | ")
            entree    = str(row.get("Entree", "—") or "—")
            stop      = str(row.get("Stop", "—") or "—")
            target    = str(row.get("Target", "—") or "—")
            risque    = str(row.get("Risque_Pct", "—") or "—")
            gain      = str(row.get("Gain_Pct", "—") or "—")
            atr_pct   = str(row.get("ATR_Pct", "—") or "—")
            support   = str(row.get("Support", "—") or "—")
            resist    = str(row.get("Resistance", "—") or "—")

            if rank == 1:
                border_color = "#ffd700"
                bg_color = "#1a130022"
            elif n_sig >= 5:
                border_color = "#00ff88"
                bg_color = "#00ff880a"
            else:
                border_color = "#1e4060"
                bg_color = "#0b142288"

            # ── En-tête de la carte ──
            st.markdown(
                f'<div style="background:{bg_color};border:2px solid {border_color};border-radius:14px;padding:20px 24px;margin:10px 0;">',
                unsafe_allow_html=True
            )

            # Ligne titre
            col_rank, col_ticker, col_conv = st.columns([1, 3, 3])
            col_rank.markdown(f"<div style='font-family:Space Mono,monospace;font-size:2rem;font-weight:700;color:{border_color};'>#{rank}</div>", unsafe_allow_html=True)
            col_ticker.markdown(f"<div style='font-family:Space Mono,monospace;font-size:1.2rem;font-weight:700;color:#00ff88;'>{row['Ticker']}</div><div style='color:#64748b;font-size:0.85rem;'>{row['Sector']}</div>", unsafe_allow_html=True)
            col_conv.markdown(f"<div style='font-family:Space Mono,monospace;font-size:1.1rem;color:{conv_clr};'>{conv_bar} {n_sig}/6</div><div style='color:{conv_clr};font-size:0.8rem;'>{conv_lbl}</div>", unsafe_allow_html=True)

            # Score + Prix
            c1, c2, c3 = st.columns(3)
            c1.metric("Score Final", f"{score_fin}/100")
            c2.metric("Prix actuel", f"${row['Prix']}")
            c3.metric("RSI | Volume", f"{row['RSI']} | {row['Vol_Ratio']}x")

            st.markdown("---")

            # Plan de trade
            col_e, col_s, col_t = st.columns(3)
            col_e.markdown(f"**🎯 ENTRÉE** *(Lundi)*\n\n`${entree}`")
            col_s.markdown(f"**🛑 STOP-LOSS**\n\n`${stop}` **(-{risque}%)**")
            col_t.markdown(f"**🏆 VENTE** *(Jeu/Ven)*\n\n`${target}` **(+{gain}%)**")

            # Règle de gap — avertissement si strict mode
            if strict_mode and entree and entree != "—":
                try:
                    entree_f = float(entree)
                    prix_f   = float(row.get("Prix", entree_f))
                    gap_max  = round(entree_f * 1.015, 2)
                    st.markdown(
                        f"⚠️ **Règle gap :** N'entrer que si ouverture lundi ≤ **${gap_max}** "
                        f"(+1.5% max vs entrée). Au-dessus → passer au trade suivant."
                    )
                except Exception:
                    pass

            # R/R + détails
            st.markdown(
                f"📊 **R/R:** `{rr_str}:1` &nbsp;|&nbsp; "
                f"ATR: `{atr_pct}%` &nbsp;|&nbsp; "
                f"Support: `${support}` &nbsp;|&nbsp; "
                f"Résistance: `${resist}`"
            )

            # Volume anormal
            vol_badge  = str(row.get("VOL_Badge", "") or "")
            vol_signal = str(row.get("VOL_Signal", "") or "")
            vol_ratio_v= row.get("VOL_Ratio", None)
            vol_rank   = row.get("VOL_52W_Rank", None)
            if vol_badge and vol_badge != "—":
                vol_info = f"**Volume :** {vol_badge}"
                if vol_ratio_v:
                    vol_info += f" — `{vol_ratio_v}x` la moyenne"
                if vol_rank:
                    vol_info += f" — Top `{round(100-float(vol_rank),1)}%` annuel"
                st.markdown(vol_info)
                if vol_signal and vol_signal not in ["None","—",""]:
                    st.markdown(
                        f"<span style='color:#00ff88;font-size:0.85rem;'>  {vol_signal}</span>",
                        unsafe_allow_html=True
                    )

            # Signaux actifs
            if signals_on_list and signals_on_list[0]:
                st.markdown("**Signaux actifs :**")
                for s in signals_on_list:
                    if s.strip():
                        st.markdown(f"<span style='color:#86efac;font-size:0.85rem;'>  {s.strip()}</span>", unsafe_allow_html=True)

            # Signaux manquants
            if signals_off_list and signals_off_list[0]:
                st.markdown("**Signaux manquants :**")
                for s in signals_off_list:
                    if s.strip():
                        st.markdown(f"<span style='color:#f87171;font-size:0.85rem;'>  {s.strip()}</span>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("")

    # ── MÉTRIQUES GLOBALES ──
    st.markdown("---")
    st.markdown("### 📈 Vue d'ensemble")
    ai_col  = "AI Score Ajuste"
    sig_col = "AI Signal Ajuste"
    vol_anormal = len(df[df["VOL_Score"] >= 15]) if "VOL_Score" in df.columns else 0
    col1,col2,col3,col4,col5,col6,col7,col8 = st.columns(8)
    for col, val, label in zip(
        [col1,col2,col3,col4,col5,col6,col7,col8],
        [
            len(df),
            len(df[df[sig_col]=="🟢 STRONG BUY"]),
            len(df[df[sig_col]=="🟢 BUY"]),
            len(df[df[sig_col]=="🟡 HOLD"]),
            len(df[df[sig_col]=="🔴 AVOID"]),
            len(report),
            len(df[df.get("Conv_N", pd.Series([0]*len(df))) >= 4]) if "Conv_N" in df.columns else 0,
            vol_anormal,
        ],
        ["Analysées","Strong Buy","Buy","Hold","Avoid",f"Top {top_n}","Conv≥4","Vol Anormal"]
    ):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    # ── GRAPHIQUES ──
    st.markdown("---")
    st.markdown("### 📊 Visualisations")
    import plotly.express as px
    tab1,tab2,tab3,tab4 = st.tabs(["Convergence","Distribution","RSI vs Score","R/R Ratio"])

    with tab1:
        if not report.empty and "Conv_N" in report.columns:
            fig_c = px.bar(report, x="Ticker", y="Conv_N",
                           color="Score_Final",
                           color_continuous_scale=["#1e3a5f","#00ff88"],
                           hover_data=["Score_Final","RR_Ratio","Top_Pattern"],
                           title=f"Top {top_n} — Convergence des signaux")
            fig_c.add_hline(y=6, line_dash="dot", line_color="#ffd700", annotation_text="6/6 parfait")
            fig_c.add_hline(y=4, line_dash="dash", line_color="#00ff88", annotation_text="4/6 minimum recommandé")
            fig_c.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                                font_color="#e2e8f0", title_font_color="#00ff88",
                                xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f",range=[0,7]))
            st.plotly_chart(fig_c, use_container_width=True)

    with tab2:
        fig = px.histogram(df, x=ai_col, nbins=20,
                           color_discrete_sequence=["#00ff88"],
                           title=f"Distribution — {regime} — {len(df)} actions")
        fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                          font_color="#e2e8f0", title_font_color="#00ff88",
                          xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig2 = px.scatter(df, x="RSI", y=ai_col, color=sig_col,
                          hover_data=["Ticker","Top_Pattern","RR_Ratio","ADV_Badge"],
                          color_discrete_map={
                              "🟢 STRONG BUY":"#00ff88","🟢 BUY":"#4ade80",
                              "🟡 HOLD":"#fbbf24","🟡 HOLD ⚠️":"#fb923c","🔴 AVOID":"#f87171"
                          }, title="RSI vs Score")
        fig2.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                           font_color="#e2e8f0", title_font_color="#00ff88",
                           xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        df_rr = df[df["RR_Ratio"].notna()].sort_values("RR_Ratio",ascending=False).head(15)
        if not df_rr.empty:
            fig5 = px.bar(df_rr, x="Ticker", y="RR_Ratio", color="RR_Ratio",
                          color_continuous_scale=["#f87171","#fbbf24","#00ff88"],
                          hover_data=["Entree","Stop","Target","Risque_Pct","Gain_Pct"],
                          title="Top R/R Ratio")
            fig5.add_hline(y=2.0, line_dash="dash", line_color="#fbbf24", annotation_text="R/R min (2:1)")
            fig5.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                               font_color="#e2e8f0", title_font_color="#00ff88",
                               xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
            st.plotly_chart(fig5, use_container_width=True)

    # ── TABLEAU COMPLET ──
    st.markdown("---")
    df_filtered = df[df[ai_col] >= min_score]
    if signal_filter:
        df_filtered = df_filtered[df_filtered[sig_col].isin(signal_filter)]
    st.markdown(f"### 📋 Tableau complet ({len(df_filtered)} actions)")
    cols_display = [
        "Ticker","Sector","Prix","MA50","MA200",
        "RSI","MACD_Hist","Vol_Ratio","Rev_Growth",
        "VOL_Badge","VOL_Signal","VOL_Ratio","VOL_52W_Rank",
        "Entree","Stop","Target","RR_Ratio","Risque_Pct","Gain_Pct","RR_Badge",
        "TTM_Signal","DIV_Signal","EMA_Level","ADV_Score","ADV_Badge",
        "Pattern_Badge","Top_Pattern","Pattern_Score",
        "AI Score","AI Score Ajuste","AI Signal Ajuste","AI Reasons"
    ]
    cols_display = [c for c in cols_display if c in df_filtered.columns]
    st.dataframe(df_filtered[cols_display], use_container_width=True, height=400)

    # ── CLAUDE ──
    if use_claude and api_key:
        st.markdown("---")
        st.markdown("### 🤖 Analyse Claude IA — Top 5 convergents")
        top5 = report.head(5)
        for _, row in top5.iterrows():
            with st.spinner(f"Claude analyse {row['Ticker']}..."):
                analysis = claude_analysis(row, api_key, market_status)
            sig = row.get("AI Signal Ajuste", row.get("AI Signal",""))
            st.markdown(f"""
            <div class="ai-analysis-box">
                <span class="ticker-badge">{row['Ticker']}</span>
                <strong style="color:#00ff88">{sig}</strong>
                — Score {row.get('Score_Final','—')}/100
                &nbsp;|&nbsp; Conv {row.get('Conv_N','—')}/6
                &nbsp;|&nbsp; R/R {row.get('RR_Ratio','—')}:1
                <br><br>{analysis}
            </div>""", unsafe_allow_html=True)
            time.sleep(0.5)
    elif use_claude and not api_key:
        st.warning("⚠️ Entrez votre clé API Anthropic.")

    # ── EXPORT ──
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.markdown("**Export rapport Top trades**")
        if not report.empty:
            excel_report = to_excel(report[cols_display if all(c in report.columns for c in cols_display) else [c for c in cols_display if c in report.columns]])
            st.download_button(
                f"⬇️ Top {top_n} — Rapport convergence",
                data=excel_report,
                file_name=f"top{top_n}_convergence_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    with col_exp2:
        st.markdown("**Export tableau complet**")
        excel_full = to_excel(df_filtered[[c for c in cols_display if c in df_filtered.columns]])
        st.download_button(
            "⬇️ Tableau complet",
            data=excel_full,
            file_name=f"screener_{regime}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
