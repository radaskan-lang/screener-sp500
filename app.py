import streamlit as st

st.set_page_config(
    page_title="S&P 500 IA Screener Pro",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
from earnings_sector import check_earnings, get_sector_strength, sector_bonus_score
from gap_detector import detect_gaps
from relative_strength import calc_relative_strength, get_spy_data
from support_resistance import calc_sr_levels
from fibonacci_bollinger import calc_fibonacci, detect_bollinger_signals
from intraday_signals import calc_vwap_levels, calc_multitf_signals, calc_intraday_momentum
from trading_tools import (
    check_data_quality, save_scan_results, load_scan_results, get_scan_age,
    add_paper_trade, update_paper_results, get_paper_summary,
    add_journal_trade, close_journal_trade, get_journal_summary,
    check_sector_diversity, get_sector_distribution
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

# 
#  S&P 500 COMPLET
# 
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

# 
#  INDICATEURS DE BASE
# 
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

# 
#  FETCH
# 
def detect_volume_anomaly(hist):
    try:
        if hist is None or hist.empty or len(hist) < 20:
            return {"badge":"--","signal":None,"score":0,"vol_ratio":1.0}
        volume = hist["Volume"]; close = hist["Close"]
        avg_vol   = float(volume.rolling(20).mean().iloc[-1])
        last_vol  = float(volume.iloc[-1])
        vol_ratio = round(last_vol / avg_vol, 2) if avg_vol > 0 else 1.0
        price = float(close.iloc[-1]); prev = float(close.iloc[-2])
        score = 0; signal = None
        if vol_ratio >= 3.0 and price > prev:
            score = 20; signal = f"VOLUME CLIMAX ({vol_ratio}x)"
        elif vol_ratio >= 2.0 and price > prev:
            score = 15; signal = f"Volume spike ({vol_ratio}x)"
        elif vol_ratio >= 1.5:
            score = 10; signal = f"Volume fort ({vol_ratio}x)"
            if len(volume) >= 5:
                vols = volume.iloc[-5:].values
                if all(vols[i] <= vols[i+1] for i in range(4)):
                    score = 18; signal = "QUIET BUILDUP"
        elif vol_ratio >= 1.1:
            score = 5
        if score >= 18:   badge = "Volume exceptionnel"
        elif score >= 10: badge = f"Volume fort {vol_ratio}x"
        elif score >= 5:  badge = f"Volume correct ({vol_ratio}x)"
        else:             badge = f"Volume faible ({vol_ratio}x)"
        return {"badge":badge,"signal":signal,"score":score,"vol_ratio":vol_ratio}
    except Exception:
        return {"badge":"--","signal":None,"score":0,"vol_ratio":1.0}


def fetch(ticker):
    try:
        try:
            from curl_cffi import requests as curl_requests
            session = curl_requests.Session(impersonate="chrome")
            t = yf.Ticker(ticker, session=session)
        except Exception:
            t = yf.Ticker(ticker)
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

        # Modules avancs
        earn_data = check_earnings(ticker)
        vol_anom  = detect_volume_anomaly(hist)  # inline below
        gap_data  = detect_gaps(hist)
        rs_data   = calc_relative_strength(hist)
        sr_data   = calc_sr_levels(hist)
        fib_data  = calc_fibonacci(hist)
        bb_data   = detect_bollinger_signals(hist)

        # Intraday (optionnel)
        vwap_data = multitf_data = mom_data = None
        if st.session_state.get("use_intraday", False):
            vwap_data    = calc_vwap_levels(ticker)
            multitf_data = calc_multitf_signals(ticker, hist)
            mom_data     = calc_intraday_momentum(ticker)

        try:
            info = t.info
            revenue_gr = info.get("revenueGrowth", None)
            sector     = info.get("sector", "N/A")
        except Exception:
            revenue_gr = None
            sector     = "N/A"

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
            # Avancs
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
            # Earnings
 "Earnings_Badge": earn_data.get("badge",""),
 "Earnings_Date":  earn_data.get("next_date",""),
 "Earnings_Risk":  earn_data.get("risk","LOW"),
            # Volume anormal
 "VOL_Badge":     vol_anom.get("badge","") if vol_anom else "",
 "VOL_Signal":    vol_anom.get("signal") if vol_anom else None,
 "VOL_Ratio":     vol_anom.get("vol_ratio",1.0) if vol_anom else 1.0,
            # Gap
 "Gap_Badge":     gap_data.get("badge","") if gap_data else "",
 "Gap_Signal":    gap_data.get("signal") if gap_data else None,
 "Gap_Score":     gap_data.get("score",0) if gap_data else 0,
 "Gap_Support":   gap_data.get("gap_support") if gap_data else None,
            # Relative Strength
 "RS_Badge":      rs_data.get("badge","") if rs_data else "",
 "RS_Trend":      rs_data.get("trend","") if rs_data else "",
 "RS_5d":         rs_data.get("rs_5d",0) if rs_data else 0,
 "RS_Perf5d":     rs_data.get("perf_5d",0) if rs_data else 0,
 "SPY_Perf5d":    rs_data.get("spy_perf_5d",0) if rs_data else 0,
            # Support/Rsistance 52w
 "SR_Badge":      sr_data.get("badge","") if sr_data else "",
 "SR_Signal":     sr_data.get("signal") if sr_data else None,
 "SR_High52w":    sr_data.get("high_52w") if sr_data else None,
 "SR_Low52w":     sr_data.get("low_52w") if sr_data else None,
 "SR_Position":   sr_data.get("position_pct") if sr_data else None,
 "SR_DistHigh":   sr_data.get("dist_to_high") if sr_data else None,
 "SR_StopNatural":sr_data.get("stop_natural") if sr_data else None,
 "SR_TargetNatural":sr_data.get("target_natural") if sr_data else None,
            # Fibonacci V2
 "FIB_Badge":      fib_data["badge"],
 "FIB_Signal":     fib_data["signal"],
 "FIB_Score":      fib_data["score"],
 "FIB_Context":    fib_data["price_context"],
 "FIB_EntryValid": fib_data["entry_valid"],
 "FIB_Warning":    fib_data["warning"],
 "FIB_Stop":       fib_data["fib_stop"],
 "FIB_Target":     fib_data["fib_target"],
 "FIB_RR":         fib_data["fib_rr"],
 "FIB_DistResist": fib_data["dist_to_resist"],
 "FIB_Support":    fib_data["nearest_support_fib"][1] if fib_data["nearest_support_fib"] else None,
 "FIB_Resist":     fib_data["nearest_resist_fib"][1] if fib_data["nearest_resist_fib"] else None,
            # Bollinger avanc
 "BB_Badge":      bb_data["badge"],
 "BB_Signal":     bb_data["signal"],
 "BB_Score":      bb_data["score"],
 "BB_Width":      bb_data["bb_width"],
 "BB_Pct":        bb_data["bb_pct"],
 "BB_WidthTrend": bb_data["width_trend"],
            # Intraday
 "ID_VWAP":        vwap_data["vwap"] if vwap_data else None,
 "ID_VWAP_Dist":   vwap_data["price_vs_vwap"] if vwap_data else None,
 "ID_VWAP_Badge":  vwap_data["badge"] if vwap_data else "",
 "ID_VWAP_Score":  vwap_data["score"] if vwap_data else 0,
 "ID_PDH":         vwap_data["pdh"] if vwap_data else None,
 "ID_PDL":         vwap_data["pdl"] if vwap_data else None,
 "ID_ORB_Break":   vwap_data["orb_breakout"] if vwap_data else None,
 "ID_TF_Confirm":  multitf_data["confirmation"] if multitf_data else "",
 "ID_TF_Align":    multitf_data["tf_alignment"] if multitf_data else 0,
 "ID_RSI_1H":      multitf_data["rsi_1h"] if multitf_data else None,
 "ID_RSI_15min":   multitf_data["rsi_15min"] if multitf_data else None,
 "ID_TF_Badge":    multitf_data["badge"] if multitf_data else "",
 "ID_TF_Score":    multitf_data["score"] if multitf_data else 0,
 "ID_Mom_Rel":     mom_data["relative_mom"] if mom_data else None,
 "ID_Mom_Badge":   mom_data["badge"] if mom_data else "",
 "ID_Mom_Score":   mom_data["score"] if mom_data else 0,
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
            status.markdown(f" Analyse `{done}/{len(tickers)}`...")
        status.empty()
    return results

# 
#  SCORE IA  NETTOY
# 
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
        score += 35; reasons.append(" Trend forte (prix>MA50>MA200)")
    elif price > ma50 and price > ma200:
        score += 25; reasons.append(" Prix > MA50 & MA200")
    elif price > ma200:
        score += 15; reasons.append("~ Au-dessus MA200")
    else:
        score += 0;  reasons.append(" Sous MAs")

    # RSI (25 pts)
    if 45 <= rsi_val <= 65:
        score += 25; reasons.append(f" RSI idal ({rsi_val})")
    elif 35 <= rsi_val < 45:
        score += 18; reasons.append(f"~ RSI rebond ({rsi_val})")
    elif 65 < rsi_val <= 72:
        score += 15; reasons.append(f"~ RSI momentum ({rsi_val})")
    elif rsi_val < 35:
        score += 10; reasons.append(f" RSI survente ({rsi_val})")
    else:
        score += 5;  reasons.append(f" RSI surachat ({rsi_val})")

    # MACD (20 pts)
    if macd_hist > 0.3:
        score += 20; reasons.append(f" MACD fort ({round(macd_hist,3)})")
    elif macd_hist > 0:
        score += 14; reasons.append(f"~ MACD haussier ({round(macd_hist,3)})")
    elif macd_hist > -0.3:
        score += 5;  reasons.append("~ MACD neutre")
    else:
        score += 0;  reasons.append(f" MACD baissier")

    # Volume (20 pts)
    if vol_ratio >= 2.0:
        score += 20; reasons.append(f" Volume trs fort ({vol_ratio}x)")
    elif vol_ratio >= 1.5:
        score += 15; reasons.append(f" Volume fort ({vol_ratio}x)")
    elif vol_ratio >= 1.1:
        score += 10; reasons.append(f"~ Volume correct ({vol_ratio}x)")
    elif vol_ratio < 0.7:
        score += 2;  reasons.append(f" Volume faible ({vol_ratio}x)")
    else:
        score += 6

    # Momentum fondamental (5 pts max)
    try:
        if rev_growth and float(rev_growth) > 10:
            score += 5; reasons.append(f" Croissance +{rev_growth}%")
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
            if top and top != "":
                reasons.append(f" Pattern: {top}")
    except Exception:
        pass

    # Bonus avancs (max 30 pts)
    try:
        adv = int(row.get("ADV_Score", 0) or 0)
        if adv > 0:
            score += min(adv, 30)
            for sig_key in ["TTM_Signal", "DIV_Signal", "EMA_Signal"]:
                s = str(row.get(sig_key) or "")
                if s and s not in ["None", "", ""]:
                    reasons.append(f" {s[:40]}")
    except Exception:
        pass

    # Earnings (pnalit)
    try:
        er = str(row.get("Earnings_Risk","LOW") or "LOW")
        if er == "HIGH":
            score -= 25; reasons.append(" Earnings cette semaine  risque lev")
        elif er == "MEDIUM":
            score -= 15; reasons.append(" Earnings proches  prudence")
    except Exception:
        pass

    # Volume anormal (max +25)
    try:
        vs = int(row.get("VOL_Score",0) or 0) if "VOL_Score" in row else 0
        if vs > 0:
            score += min(vs, 25)
            vs_sig = str(row.get("VOL_Signal") or "")
            if vs_sig and vs_sig not in ["None","",""]:
                reasons.append(f" {vs_sig[:40]}")
    except Exception:
        pass

    # Gap (max +20, min -15)
    try:
        gs = int(row.get("Gap_Score",0) or 0)
        if gs != 0:
            score += max(-15, min(gs, 20))
            g_sig = str(row.get("Gap_Signal") or "")
            if g_sig and g_sig not in ["None","",""]:
                reasons.append(f" {g_sig[:40]}")
    except Exception:
        pass

    # Relative Strength vs SPY (max +15, min -8)
    try:
        rs5 = float(row.get("RS_5d",0) or 0)
        rs_trend = str(row.get("RS_Trend","") or "")
        if rs5 >= 2.0:
            score += 15; reasons.append(f" RS Excellente vs SPY (+{rs5}%)")
        elif rs5 >= 1.0:
            score += 10; reasons.append(f" RS Forte vs SPY (+{rs5}%)")
        elif rs5 >= 0.3:
            score += 5
        elif rs5 <= -1.0:
            score -= 8; reasons.append(f" RS Faible vs SPY ({rs5}%)")
    except Exception:
        pass

    # Support/Rsistance 52w
    try:
        sr_sig = str(row.get("SR_Signal") or "")
        sr_pos = float(row.get("SR_Position",50) or 50)
        if sr_pos >= 95:
            score -= 10; reasons.append(" Proche du High 52w  rsistance")
        elif sr_pos >= 80:
            score += 8; reasons.append(f" Position forte 52w ({round(sr_pos)}%)")
        elif sr_pos <= 20:
            score -= 5
    except Exception:
        pass

    # Fibonacci V2  filtre de validation
    try:
        fib_score   = int(row.get("FIB_Score",0) or 0)
        fib_valid   = row.get("FIB_EntryValid", True)
        fib_signal  = str(row.get("FIB_Signal") or "")
        fib_dist_r  = float(row.get("FIB_DistResist",99) or 99)
        if fib_score != 0:
            score += fib_score
            if fib_signal and fib_signal not in ["None","",""]:
                reasons.append(f" {fib_signal[:50]}")
        if fib_valid == False:
            score -= 10
            reasons.append(f" FIB: Rsistance dans {fib_dist_r}%")
    except Exception:
        pass

    # Bollinger avanc (max +15, min -10)
    try:
        bb_score  = int(row.get("BB_Score",0) or 0)
        bb_signal = str(row.get("BB_Signal") or "")
        if bb_score != 0:
            score += bb_score
            if bb_signal and bb_signal not in ["None","",""]:
                reasons.append(f" {bb_signal[:40]}")
    except Exception:
        pass

    # Signaux intraday (max +25, min -20)
    try:
        id_total = (int(row.get("ID_VWAP_Score",0) or 0) +
                    int(row.get("ID_TF_Score",0) or 0) +
                    int(row.get("ID_Mom_Score",0) or 0))
        id_conf  = str(row.get("ID_TF_Confirm","") or "")
        if id_total != 0:
            score += max(-20, min(id_total, 25))
        if id_conf == "CONTRADICTOIRE":
            score -= 10; reasons.append(" TF contradictoires")
        elif id_conf == "FORT" and id_total > 10:
            reasons.append(" Multi-TF aligns + Momentum")
    except Exception:
        pass

    # March BAISSIER  pnalit globale
    try:
        regime = st.session_state.get("regime","HAUSSIER")
        if regime == "BAISSIER":
            score -= 20; reasons.append(" March BAISSIER  prudence")
    except Exception:
        pass

    return min(score, 100), reasons

def ai_signal(score):
    if score >= 85:   return " STRONG BUY"
    elif score >= 70: return " BUY"
    elif score >= 50: return " HOLD"
    else:             return " AVOID"

# 
#  ANALYSE CLAUDE
# 
def claude_analysis(row, api_key, market_status):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        regime = market_status.get("regime", "INCONNU")
        rr_str = f"{row.get('RR_Ratio','N/A')}:1"

        prompt = f"""Tu es un trader spcialis en swing trading (lundi -> vendredi).
March : {regime} | SPY vs MA50: {market_status.get('spy_vs_ma50','N/A')}% | {market_status.get('vix_label','VIX N/A')}

Ticker: {row['Ticker']} ({row['Sector']})
Convergence: {row.get('Conv_N','N/A')}/6 signaux | Score final: {row.get('Score_Final','N/A')}/100
Prix: ${row['Prix']} | Entree: ${row.get('Entree','N/A')} | Stop: ${row.get('Stop','N/A')} | Target: ${row.get('Target','N/A')}
R/R: {rr_str} | Risque: {row.get('Risque_Pct','N/A')}% | Gain: {row.get('Gain_Pct','N/A')}%
RSI: {row['RSI']} | MACD: {row['MACD_Hist']} | Vol: {row['Vol_Ratio']}x
TTM: {row.get('TTM_Signal','')} | Div RSI: {row.get('DIV_Signal','')} | EMA: {row.get('EMA_Level','')}
Pattern: {row.get('Top_Pattern','')}
Signaux actifs: {row.get('Conv_On','')}

En 6 lignes max :
1) VERDICT (ACHETER/ATTENDRE/EVITER)
2) Confirmes-tu entree ${row.get('Entree','N/A')} / stop ${row.get('Stop','N/A')} ?
3) Argument principal bas sur la convergence des signaux
4) Risque principal cette semaine
Direct, chiffr, sans disclaimer."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Analyse indisponible : {e}"

# 
#  EXCEL EXPORT
# 
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Screener")
    return output.getvalue()

# 
#  SIDEBAR
# 
with st.sidebar:
    st.markdown("##  Configuration")
    st.markdown("---")

    st.markdown("###  Mode de scan")
    use_prefilter = st.checkbox("Pr-filtre S&P 500 complet (503)", value=True)
    if use_prefilter:
        st.markdown("<div style='color:#4a90d0;font-size:0.78rem;'> 503 actions  2 passes automatiques</div>", unsafe_allow_html=True)
        with st.expander(" Critres pr-filtre"):
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
    nb_workers = st.slider(" Threads parallles", 5, 20, 10)

    st.markdown("---")
    st.markdown("###  Rapport Top Trades")
    top_n        = st.radio("Nombre de trades", [10, 20], index=0, horizontal=True)
    min_signals  = st.slider("Signaux convergents min", 2, 6, 3)
    min_rr_conv  = st.slider("R/R minimum", 1.0, 3.0, 1.5, step=0.1)

    st.markdown("---")
    st.markdown("###  Claude IA")
    api_key    = st.text_input("Cl API Anthropic", type="password")
    use_claude = st.checkbox("Activer analyse Claude", value=False)

    st.markdown("---")
    st.markdown("###  Signaux Intraday")
    use_intraday = st.checkbox(" Activer signaux intraday", value=False,
        help="VWAP  PDH/PDL  Multi-TF  Momentum vs SPY  dlai 15min")
    if use_intraday:
        st.markdown("<div style='background:#4a90d015;border:1px solid #4a90d033;border-radius:6px;padding:8px;font-size:0.78rem;color:#93c5fd;margin-top:4px;'> VWAP + PDH/PDL + Opening Range<br> Multi-TF RSI (Daily+1H+15min)<br> Momentum intraday vs SPY<br> Dlai 15min  indicatif<br> Ralentit le scan (~2x)</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("###  Filtres tableau")
    min_score     = st.slider("Score min", 0, 100, 50)
    signal_filter = st.multiselect(
 "Signaux",
        [" STRONG BUY"," BUY"," HOLD"," AVOID"," HOLD "],
        default=[" STRONG BUY"," BUY"]
    )

    st.markdown("---")
    st.markdown("<div style='color:#64748b;font-size:0.75rem;'>S&P 500 IA Screener Pro</div>", unsafe_allow_html=True)

# 
#  MAIN
# 
st.markdown("#  S&P 500 IA Screener Pro")
st.markdown("<div style='color:#64748b;margin-bottom:1.5rem;'>Convergence  Patterns  R/R  TTM  Divergence  EMA  March  Claude IA</div>", unsafe_allow_html=True)

# BANDEAU MARCH
with st.spinner("Vrification march global..."):
    market_status = get_market_status()

regime = market_status["regime"]
st.session_state["regime"] = regime
color  = market_status["color"]
emoji  = market_status["emoji"]

st.markdown(f"""
<div class="market-banner" style="background:{color}11;border:1px solid {color}44;border-left:5px solid {color};">
    <strong style="color:{color};font-size:1.1rem;">{emoji} MARCH {regime}</strong>
    &nbsp;&nbsp; {market_status['message']}
    <br><span style="color:#94a3b8;font-size:0.82rem;font-family:'Space Mono',monospace;">{market_status['detail']}</span>
</div>
""", unsafe_allow_html=True)

with st.expander(" Conseils de trading"):
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
        (c1, f"{'+' if market_status.get('spy_vs_ma50',0)>=0 else ''}{market_status.get('spy_vs_ma50','')}%", "SPY vs MA50", spy_color),
        (c2, f"{'+' if market_status.get('qqq_vs_ma50',0)>=0 else ''}{market_status.get('qqq_vs_ma50','')}%", "QQQ vs MA50", qqq_color),
        (c3, vix_val if vix_val else "", "VIX", vix_color),
        (c4, market_status.get("spy_rsi",""), "RSI SPY", "#00ff88"),
    ]:
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{clr}">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

if st.button(f" Lancer  S&P 500 complet ({len(SP500_TICKERS)} actions)"):

    tickers_to_analyze = SP500_TICKERS
    st.session_state["use_intraday"] = use_intraday

    # PASSE 1 : PR-FILTRE
    if use_prefilter:
        st.markdown("###  Passe 1  Pr-filtre rapide")
        pf_prog = st.progress(0)
        pf_stat = st.empty()
        def pf_cb(done, total):
            pf_prog.progress(done/total)
            pf_stat.markdown(f" Pr-filtre `{done}/{total}`...")
        pf_result = run_prefilter(SP500_TICKERS, max_workers=20, progress_callback=pf_cb)
        pf_stat.empty()
        tickers_to_analyze = pf_result["passed"]
        st.markdown(f"""<div class="prefilter-banner">
             PASSE 1 TERMINE &nbsp;|&nbsp;
            <span style="color:#00ff88">{pf_result['n_passed']} retenues</span>
            &nbsp;/&nbsp; {pf_result['total']}
            &nbsp;|&nbsp; {pf_result['n_rejected']} limines
            &nbsp;|&nbsp; {pf_result['pass_rate']}% passage
        </div>""", unsafe_allow_html=True)
        with st.expander(f" {pf_result['n_rejected']} actions limines"):
            st.dataframe(pd.DataFrame([
                {"Ticker": t, "Raison": pf_result['details'][t]}
                for t in pf_result['rejected']
            ]), use_container_width=True, height=200)

    if not tickers_to_analyze:
        st.error(" Aucune action ne passe le pr-filtre.")
        st.stop()

    # PASSE 2 : ANALYSE COMPLTE
    st.markdown(f"###  Passe 2  Analyse complte ({len(tickers_to_analyze)} actions)")
    rows = fetch_parallel(tickers_to_analyze, max_workers=nb_workers)

    if not rows:
        st.error(" Aucune donne rcupre.")
        st.stop()

    df = pd.DataFrame(rows)

    scores_data      = df.apply(ai_score, axis=1)
    df["AI Score"]   = scores_data.apply(lambda x: x[0])
    df["AI Signal"]  = df["AI Score"].apply(ai_signal)
    df["AI Reasons"] = scores_data.apply(lambda x: " | ".join(x[1]))

    df = apply_market_filter(df, market_status)
    df.rename(columns={"AI Signal Ajust":"AI Signal Ajuste","AI Score Ajust":"AI Score Ajuste"}, errors="ignore", inplace=True)
    if "AI Score Ajuste" not in df.columns:
        df["AI Score Ajuste"] = df["AI Score"]
    if "AI Signal Ajuste" not in df.columns:
        df["AI Signal Ajuste"] = df["AI Signal"]
    df = df.sort_values("AI Score Ajuste", ascending=False).reset_index(drop=True)

    # 
    #  RAPPORT DE CONVERGENCE  TOP TRADES
    # 
    st.markdown("---")
    st.markdown(f"##  Rapport du Dimanche  Top {top_n} Trades Convergents")
    st.markdown(f"<div style='color:#64748b;font-size:0.85rem;margin-bottom:1rem;'>Semaine du {datetime.now().strftime('%d %B %Y')}  March {regime}  Min {min_signals}/6 signaux convergents</div>", unsafe_allow_html=True)

    # Conseils d'excution semaine
    day_advice = get_day_of_week_advice(regime)
    with st.expander(" Plan d'excution de la semaine"):
        cols = st.columns(3)
        days = [("Lundi",""), ("Mercredi",""), ("Vendredi","")]
        for i, (day, em) in enumerate(days):
            cols[i].markdown(f"""<div class="advice-box">
                <strong>{em} {day}</strong><br>{day_advice.get(day,'')}
            </div>""", unsafe_allow_html=True)

    # Construire le rapport de convergence
    report = build_trade_report(
        df,
        top_n=top_n,
        min_signals=min_signals,
        min_rr=min_rr_conv
    )

    if report.empty:
        st.warning(f" Aucun titre avec {min_signals}+ signaux convergents. Rduire le filtre dans la sidebar.")
    else:
        st.markdown(f"<div style='color:#00ff88;font-size:0.85rem;margin-bottom:16px;'> {len(report)} opportunits identifies</div>", unsafe_allow_html=True)

        for idx, row in report.iterrows():
            rank      = idx + 1
            n_sig     = int(row.get("Conv_N", 0))
            conv_bar  = str(row.get("Conv_Bar", ""))
            conv_lbl  = str(row.get("Conv_Label", ""))
            conv_clr  = str(row.get("Conv_Color", "#64748b"))
            score_fin = row.get("Score_Final", 0)
            rr        = row.get("RR_Ratio", None)
            signals_on  = str(row.get("Conv_On", ""))
            signals_off = str(row.get("Conv_Off", ""))

            # Choisir le style de carte selon le rang
            if rank == 1:
                card_class = "trade-card-gold"
                rank_color = "#ffd700"
            elif n_sig >= 5:
                card_class = "trade-card-green"
                rank_color = "#00ff88"
            else:
                card_class = "trade-card"
                rank_color = "#4a90d0"

            _html = f"""
            <div class="{card_class}">

                <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap;">
                    <span class="rank-badge" style="color:{rank_color}">#{rank}</span>
                    <span class="ticker-badge">{row['Ticker']}</span>
                    <span style="color:#94a3b8;font-size:0.85rem;">{row['Sector']}</span>
                    <span style="margin-left:auto;font-family:'Space Mono',monospace;font-size:0.8rem;color:{conv_clr};">
                        {conv_lbl}
                    </span>
                </div>

                <div style="display:flex;align-items:center;gap:20px;margin-bottom:14px;flex-wrap:wrap;">
                    <div>
                        <div style="font-size:10px;color:#3a5070;text-transform:uppercase;letter-spacing:0.1em;">Convergence</div>
                        <div class="conv-bar" style="color:{conv_clr};">{conv_bar}</div>
                        <div style="font-size:11px;color:{conv_clr};font-family:'Space Mono',monospace;">{n_sig}/6 signaux</div>
                    </div>
                    <div style="border-left:1px solid #1e3a5f;padding-left:20px;">
                        <div style="font-size:10px;color:#3a5070;text-transform:uppercase;letter-spacing:0.1em;">Score Final</div>
                        <div style="font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;color:{conv_clr};">{score_fin}</div>
                        <div style="font-size:11px;color:#64748b;">Score composite</div>
                    </div>
                    <div style="border-left:1px solid #1e3a5f;padding-left:20px;">
                        <div style="font-size:10px;color:#3a5070;text-transform:uppercase;letter-spacing:0.1em;">Prix actuel</div>
                        <div style="font-family:'Space Mono',monospace;font-size:1.3rem;font-weight:700;color:#e0f0ff;">${row['Prix']}</div>
                        <div style="font-size:11px;color:#64748b;">RSI {row['RSI']} | Vol {row['Vol_Ratio']}x</div>
                    </div>
                </div>

                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px;">
                    <div style="background:#00000033;border-radius:8px;padding:10px;border:1px solid #00ff8833;">
                        <div style="font-size:10px;color:#3a5070;text-transform:uppercase;"> ENTRE (Lundi)</div>
                        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;color:#e0f0ff;">${row.get('Entree','')}</div>
                    </div>
                    <div style="background:#00000033;border-radius:8px;padding:10px;border:1px solid #f8717133;">
                        <div style="font-size:10px;color:#3a5070;text-transform:uppercase;"> STOP-LOSS</div>
                        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;color:#f87171;">${row.get('Stop','')} <span style="font-size:0.75rem;">(-{row.get('Risque_Pct','')}%)</span></div>
                    </div>
                    <div style="background:#00000033;border-radius:8px;padding:10px;border:1px solid #00ff8833;">
                        <div style="font-size:10px;color:#3a5070;text-transform:uppercase;"> VENTE (Jeu/Ven)</div>
                        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;color:#00ff88;">${row.get('Target','')} <span style="font-size:0.75rem;">(+{row.get('Gain_Pct','')}%)</span></div>
                    </div>
                </div>

                <div style="display:flex;gap:16px;margin-bottom:12px;flex-wrap:wrap;">
                    <div style="font-family:'Space Mono',monospace;font-size:0.85rem;">
                         R/R: <strong style="color:{'#00ff88' if rr and rr>=2 else '#fbbf24'}">{rr}:1</strong>
                    </div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.85rem;">
                        ATR: <strong>{row.get('ATR_Pct','')}%</strong>
                    </div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.85rem;">
                        Support: <strong>${row.get('Support','')}</strong>
                    </div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.85rem;">
                        Rsistance: <strong>${row.get('Resistance','')}</strong>
                    </div>
                </div>

                <div style="margin-bottom:8px;">
                    <div style="font-size:10px;color:#3a5070;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Signaux actifs</div>
                    <div style="font-size:0.82rem;color:#86efac;line-height:1.8;">{signals_on.replace(' | ','<br>') if signals_on else ''}</div>
                </div>

                {f'<div style="margin-top:6px;"><div style="font-size:10px;color:#3a5070;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Signaux manquants</div><div style="font-size:0.82rem;color:#f87171;line-height:1.8;">{signals_off.replace(chr(124)+" ","<br>")}</div></div>' if signals_off else ''}

            </div>
            """
            st.html(_html)

            #  FIBONACCI 
            fib_badge   = str(row.get("FIB_Badge","") or "")
            fib_context = str(row.get("FIB_Context","NEUTRE") or "NEUTRE")
            fib_valid   = row.get("FIB_EntryValid", True)
            fib_signal_v= str(row.get("FIB_Signal","") or "")
            fib_warning = str(row.get("FIB_Warning","") or "")
            fib_stop_v  = row.get("FIB_Stop", None)
            fib_target_v= row.get("FIB_Target", None)
            fib_rr_v    = row.get("FIB_RR", None)
            fib_support = row.get("FIB_Support", None)
            fib_resist  = row.get("FIB_Resist", None)
            fib_dist_r  = row.get("FIB_DistResist", None)
            fib_color   = {"REBOND_KEY":"#00ff88","BREAKOUT":"#00ff88","ZONE_SAINE":"#86efac","NEUTRE":"#fbbf24","RESISTANCE_PROCHE":"#f87171"}.get(fib_context,"#fbbf24")
            fib_display = fib_badge if fib_badge and fib_badge != "" else f"Fibonacci {fib_context}"
            st.markdown(f"** Fibonacci :** <span style='color:{fib_color};font-weight:700;'>{fib_display}</span>", unsafe_allow_html=True)
            if not fib_valid and fib_warning:
                st.markdown(f"<div style='background:#f871711a;border-left:3px solid #f87171;border-radius:4px;padding:8px;font-size:0.85rem;color:#f87171;'> {fib_warning}</div>", unsafe_allow_html=True)
            elif fib_signal_v and fib_signal_v not in ["None","",""]:
                st.markdown(f"<span style='color:{fib_color};font-size:0.85rem;'>{fib_signal_v}</span>", unsafe_allow_html=True)
            fib_parts = []
            if fib_support: fib_parts.append(f"Support Fib: `${fib_support}`")
            if fib_resist:  fib_parts.append(f"Rsistance Fib: `${fib_resist}`")
            if fib_dist_r and fib_dist_r < 99: fib_parts.append(f"Dist rsistance: `{fib_dist_r}%`")
            if fib_parts: st.markdown(" &nbsp;|&nbsp; ".join(fib_parts))
            if fib_stop_v and fib_target_v:
                rr_str = f" &nbsp;|&nbsp; R/R Fib: `{fib_rr_v}:1`" if fib_rr_v else ""
                st.markdown(f" Stop Fib: `${fib_stop_v}` &nbsp;|&nbsp;  Target Fib: `${fib_target_v}`{rr_str}")

    #  MTRIQUES GLOBALES 
    st.markdown("---")
    st.markdown("###  Vue d'ensemble")
    ai_col = "AI Score Ajuste"
    sig_col = "AI Signal Ajuste"
    col1,col2,col3,col4,col5,col6,col7 = st.columns(7)
    for col, val, label in zip(
        [col1,col2,col3,col4,col5,col6,col7],
        [len(df),
         len(df[df[sig_col]==" STRONG BUY"]),
         len(df[df[sig_col]==" BUY"]),
         len(df[df[sig_col]==" HOLD"]),
         len(df[df[sig_col]==" AVOID"]),
         len(report),
         len(df[df.get("Conv_N",pd.Series([0]*len(df)))>=4]) if "Conv_N" in df.columns else 0,
        ],
        ["Analyses","Strong Buy","Buy","Hold","Avoid",f"Top {top_n}","Conv 4"]
    ):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    #  GRAPHIQUES 
    st.markdown("---")
    st.markdown("###  Visualisations")
    import plotly.express as px
    tab1,tab2,tab3,tab4 = st.tabs(["Convergence","Distribution","RSI vs Score","R/R Ratio"])

    with tab1:
        if not report.empty and "Conv_N" in report.columns:
            fig_c = px.bar(report, x="Ticker", y="Conv_N",
                           color="Score_Final",
                           color_continuous_scale=["#1e3a5f","#00ff88"],
                           hover_data=["Score_Final","RR_Ratio","Top_Pattern"],
                           title=f"Top {top_n}  Convergence des signaux")
            fig_c.add_hline(y=6, line_dash="dot", line_color="#ffd700", annotation_text="6/6 parfait")
            fig_c.add_hline(y=4, line_dash="dash", line_color="#00ff88", annotation_text="4/6 minimum recommand")
            fig_c.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                                font_color="#e2e8f0", title_font_color="#00ff88",
                                xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f",range=[0,7]))
            st.plotly_chart(fig_c, use_container_width=True)

    with tab2:
        fig = px.histogram(df, x=ai_col, nbins=20,
                           color_discrete_sequence=["#00ff88"],
                           title=f"Distribution  {regime}  {len(df)} actions")
        fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                          font_color="#e2e8f0", title_font_color="#00ff88",
                          xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig2 = px.scatter(df, x="RSI", y=ai_col, color=sig_col,
                          hover_data=["Ticker","Top_Pattern","RR_Ratio","ADV_Badge"],
                          color_discrete_map={
 " STRONG BUY":"#00ff88"," BUY":"#4ade80",
 " HOLD":"#fbbf24"," HOLD ":"#fb923c"," AVOID":"#f87171"
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

    #  TABLEAU COMPLET 
    st.markdown("---")
    df_filtered = df[df[ai_col] >= min_score]
    if signal_filter:
        df_filtered = df_filtered[df_filtered[sig_col].isin(signal_filter)]
    st.markdown(f"###  Tableau complet ({len(df_filtered)} actions)")
    cols_display = [
 "Ticker","Sector","Prix","MA50","MA200",
 "RSI","MACD_Hist","Vol_Ratio","Rev_Growth",
 "Entree","Stop","Target","RR_Ratio","Risque_Pct","Gain_Pct","RR_Badge",
 "TTM_Signal","DIV_Signal","EMA_Level","ADV_Score","ADV_Badge",
 "Pattern_Badge","Top_Pattern","Pattern_Score",
 "AI Score","AI Score Ajuste","AI Signal Ajuste","AI Reasons"
    ]
    cols_display = [c for c in cols_display if c in df_filtered.columns]
    st.dataframe(df_filtered[cols_display], use_container_width=True, height=400)

    #  CLAUDE 
    if use_claude and api_key:
        st.markdown("---")
        st.markdown("###  Analyse Claude IA  Top 5 convergents")
        top5 = report.head(5)
        for _, row in top5.iterrows():
            with st.spinner(f"Claude analyse {row['Ticker']}..."):
                analysis = claude_analysis(row, api_key, market_status)
            sig = row.get("AI Signal Ajuste", row.get("AI Signal",""))
            st.markdown(f"""
            <div class="ai-analysis-box">
                <span class="ticker-badge">{row['Ticker']}</span>
                <strong style="color:#00ff88">{sig}</strong>
                 Score {row.get('Score_Final','')}/100
                &nbsp;|&nbsp; Conv {row.get('Conv_N','')}/6
                &nbsp;|&nbsp; R/R {row.get('RR_Ratio','')}:1
                <br><br>{analysis}
            </div>""", unsafe_allow_html=True)
            time.sleep(0.5)
    elif use_claude and not api_key:
        st.warning(" Entrez votre cl API Anthropic.")

    #  EXPORT 
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.markdown("**Export rapport Top trades**")
        if not report.empty:
            excel_report = to_excel(report[cols_display if all(c in report.columns for c in cols_display) else [c for c in cols_display if c in report.columns]])
            st.download_button(
                f" Top {top_n}  Rapport convergence",
                data=excel_report,
                file_name=f"top{top_n}_convergence_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    with col_exp2:
        st.markdown("**Export tableau complet**")
        excel_full = to_excel(df_filtered[[c for c in cols_display if c in df_filtered.columns]])
        st.download_button(
 " Tableau complet",
            data=excel_full,
            file_name=f"screener_{regime}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )




# ═══════════════════════════════════════════════════════════════
# 🎯 MODULE OPPORTUNITÉS DE CORRECTION — Buy the Dip institutionnel
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 🎯 Opportunités de Correction")

def _corr_compute_indicators(close, high, low, volume):
    """StochRSI + MACD + MFI pour le module correction."""
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = -delta.where(delta < 0, 0).rolling(14).mean()
    rsi_s = 100 - (100 / (1 + gain / loss.clip(lower=1e-10)))
    rsi_min = rsi_s.rolling(14).min()
    rsi_max = rsi_s.rolling(14).max()
    stoch_raw = (rsi_s - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    stoch_k   = stoch_raw.rolling(3).mean()
    stoch_d   = stoch_k.rolling(3).mean()
    ema12     = close.ewm(span=12, adjust=False).mean()
    ema26     = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    sig_line  = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - sig_line
    tp        = (high + low + close) / 3
    raw_mf    = tp * volume
    pos_mf    = pd.Series([raw_mf.iloc[i] if tp.iloc[i] > tp.iloc[i-1] else 0
                            for i in range(len(tp))], index=tp.index)
    neg_mf    = pd.Series([raw_mf.iloc[i] if tp.iloc[i] < tp.iloc[i-1] else 0
                            for i in range(len(tp))], index=tp.index)
    mfi = 100 - (100 / (1 + pos_mf.rolling(14).sum()
                            / neg_mf.rolling(14).sum().clip(lower=1e-10)))
    return {"rsi": rsi_s, "stoch_k": stoch_k, "stoch_d": stoch_d,
            "macd_hist": macd_hist, "mfi": mfi}


def _corr_fib_levels(hi, lo):
    r = hi - lo
    return {"0.236": hi - 0.236*r, "0.382": hi - 0.382*r,
            "0.500": hi - 0.500*r, "0.618": hi - 0.618*r,
            "0.786": hi - 0.786*r}


def _corr_nearest_fib(price, fibs):
    nearest = min(fibs.items(), key=lambda x: abs(x[1] - price))
    return nearest[0], nearest[1], round(abs(price - nearest[1]) / price * 100, 1)


def analyze_correction_opportunity(ticker, hist, min_correction=10.0):
    """
    Détecte un retournement haussier sur un titre en correction.
    Retourne un dict ou None si les critères ne sont pas remplis.
    """
    try:
        if hist is None or len(hist) < 60:
            return None
        close  = hist["Close"]
        high   = hist["High"]
        low    = hist["Low"]
        volume = hist["Volume"]
        price  = float(close.iloc[-1])
        hi52   = float(close.rolling(min(252, len(close))).max().iloc[-1])
        corr_pct = round((price - hi52) / hi52 * 100, 1)

        # ── Critère 1 : correction suffisante ──
        if corr_pct > -min_correction:
            return None

        ind = _corr_compute_indicators(close, high, low, volume)
        sk  = ind["stoch_k"]
        sd  = ind["stoch_d"]
        mh  = ind["macd_hist"]

        # ── Critère 2 : StochRSI K croise D à la hausse depuis zone < 40 ──
        stoch_cross_ok  = False
        stoch_cross_day = None
        lookback = min(8, len(sk) - 2)
        for i in range(lookback, 0, -1):
            idx, idx_p = len(sk)-i, len(sk)-i-1
            if idx < 1 or idx_p < 0: continue
            try:
                k_now  = float(sk.iloc[idx])
                k_prev = float(sk.iloc[idx_p])
                d_now  = float(sd.iloc[idx])
                d_prev = float(sd.iloc[idx_p])
            except Exception:
                continue
            if k_now > d_now and k_prev <= d_prev and d_prev < 40:
                stoch_cross_ok  = True
                stoch_cross_day = i
                break

        if not stoch_cross_ok:
            return None

        # ── Critère 3 : MACD croisement haussier (bonus) ──
        macd_cross_ok  = False
        macd_cross_day = None
        for i in range(lookback, 0, -1):
            idx, idx_p = len(mh)-i, len(mh)-i-1
            if idx < 1 or idx_p < 0: continue
            try:
                h_now  = float(mh.iloc[idx])
                h_prev = float(mh.iloc[idx_p])
            except Exception:
                continue
            if h_now > h_prev and h_prev < 0 and h_now > -0.5:
                macd_cross_ok  = True
                macd_cross_day = i
                break

        # ── Critère 4 : Volume ──
        avg_vol20 = float(volume.rolling(20).mean().iloc[-1])
        avg_vol3  = float(volume.iloc[-3:].mean()) if len(volume) >= 3 else avg_vol20
        vol_ratio = round(avg_vol3 / avg_vol20, 2) if avg_vol20 > 0 else 1.0
        vol_ok    = vol_ratio >= 1.05

        # ── Critère 5 : Fibonacci ──
        fib_lb = min(90, len(close)-1)
        fib_hi = float(high.iloc[-fib_lb:].max())
        fib_lo = float(low.iloc[-fib_lb:].min())
        key_fibs = {k: v for k, v in _corr_fib_levels(fib_hi, fib_lo).items()
                    if k in ("0.382","0.500","0.618","0.786")}
        fib_name, fib_val, fib_dist = _corr_nearest_fib(price, key_fibs)
        fib_ok = fib_dist <= 3.5

        # ── Score signal (0-10) ──
        signal_score = 0
        if stoch_cross_ok: signal_score += 3
        if macd_cross_ok:  signal_score += 2
        if vol_ok:         signal_score += 2
        if fib_ok:
            signal_score += 2
            if fib_name == "0.618": signal_score += 1

        if signal_score < 3:
            return None

        # ── Valeurs display ──
        stoch_k_now = float(sk.iloc[-1]) if not sk.isna().iloc[-1] else 0
        stoch_d_now = float(sd.iloc[-1]) if not sd.isna().iloc[-1] else 0
        macd_h_now  = float(mh.iloc[-1]) if not mh.isna().iloc[-1] else 0
        rsi_now     = float(ind["rsi"].iloc[-1]) if not ind["rsi"].isna().iloc[-1] else 50
        mfi_now     = float(ind["mfi"].iloc[-1]) if not ind["mfi"].isna().iloc[-1] else 50
        ma50_now    = float(close.rolling(min(50, len(close))).mean().iloc[-1])
        ma200_now   = float(close.rolling(min(200, len(close))).mean().iloc[-1])

        # Prix d'entrée suggéré = niveau Fibonacci le plus proche sous le prix
        entry_price = round(fib_val if fib_ok else min(
            (v for v in key_fibs.values() if v <= price * 1.02),
            default=price, key=lambda x: abs(x-price)), 2)

        if signal_score >= 8:     force, force_color = "TRÈS FORT",  "#00ff88"
        elif signal_score >= 6:   force, force_color = "FORT",       "#7DF9FF"
        elif signal_score >= 4:   force, force_color = "MODÉRÉ",     "#fbbf24"
        else:                     force, force_color = "FAIBLE",     "#f97316"

        macd_label = (f"✅ croisé il y a {macd_cross_day}j" if macd_cross_ok
                      else "⚠️ pas encore croisé")

        return {
            "ticker": ticker, "price": round(price, 2), "hi52": round(hi52, 2),
            "corr_pct": corr_pct, "signal_score": signal_score,
            "force": force, "force_color": force_color,
            "stoch_k": round(stoch_k_now, 1), "stoch_d": round(stoch_d_now, 1),
            "stoch_cross_day": stoch_cross_day,
            "macd_ok": macd_cross_ok, "macd_label": macd_label,
            "macd_hist_val": round(macd_h_now, 4),
            "vol_ratio": vol_ratio, "vol_ok": vol_ok,
            "fib_name": fib_name, "fib_val": round(fib_val, 2),
            "fib_dist": fib_dist, "fib_ok": fib_ok,
            "entry_price": entry_price,
            "rsi": round(rsi_now, 1), "mfi": round(mfi_now, 1),
            "ma50": round(ma50_now, 2), "ma200": round(ma200_now, 2),
            "above_ma200": price > ma200_now,
            "stoch_cross_ok": stoch_cross_ok,
        }
    except Exception:
        return None


# ── Paramètres UI ──
corr_c1, corr_c2, corr_c3, corr_c4 = st.columns(4)
with corr_c1:
    corr_min_pct = st.slider("Correction min (%)", 5, 40, 10, 5,
                              key="corr_min_pct",
                              help="Recul minimum depuis le sommet 52 semaines")
with corr_c2:
    corr_min_score = st.slider("Score IA min", 0, 100, 45, 5,
                                key="corr_min_score",
                                help="Score AlphaScreen minimum (0 = ignorer)")
with corr_c3:
    corr_min_signal = st.slider("Force signal min", 1, 10, 3, 1,
                                 key="corr_min_signal",
                                 help="Score retournement minimum (1-10)")
with corr_c4:
    corr_require_fib = st.checkbox("Fibonacci requis", value=False,
                                    key="corr_req_fib",
                                    help="Exiger un niveau Fibonacci ±3.5%")

# ── Lancement du scan ──
if st.button("🎯 Scanner les Corrections", key="btn_corrections"):

    # Récupérer les données déjà analysées si dispo, sinon scanner
    if "df" not in dir() or not isinstance(df if "df" in dir() else None, pd.DataFrame):
        # Pas encore de scan principal lancé — télécharger données directement
        st.info("💡 Lancez d'abord le scan principal pour accélérer cette analyse. Téléchargement en cours...")
        corr_tickers = SP500_TICKERS
        corr_hists   = {}
        corr_prog    = st.progress(0)
        corr_stat    = st.empty()

        def _fetch_hist(t):
            try:
                h = yf.Ticker(t).history(period="1y")
                return t, h if (h is not None and len(h) >= 60) else None
            except Exception:
                return t, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
            futs = {ex.submit(_fetch_hist, t): t for t in corr_tickers}
            done_c = 0
            for f in concurrent.futures.as_completed(futs):
                done_c += 1
                t, h = f.result()
                if h is not None:
                    corr_hists[t] = h
                corr_prog.progress(done_c / len(corr_tickers))
                corr_stat.markdown(f"<span style='color:#64748b;font-size:0.78rem;'>"
                                    f"📥 {t} ({done_c}/{len(corr_tickers)})</span>",
                                    unsafe_allow_html=True)
        corr_prog.empty()
        corr_stat.empty()
        corr_scores = {}  # pas de scores disponibles
    else:
        # Réutiliser les données déjà chargées
        corr_hists  = {}
        corr_scores = {}
        for _, row_s in df.iterrows():
            t = row_s.get("Ticker","")
            if t:
                corr_scores[t] = int(row_s.get("AI Score Ajuste", row_s.get("AI Score", 0)) or 0)
        st.info(f"✅ Réutilisation des {len(df)} titres déjà scannés.")

        # Télécharger l'historique pour les titres qui passent le seuil de score
        eligible = [t for t, s in corr_scores.items() if s >= corr_min_score or corr_min_score == 0]
        corr_prog2 = st.progress(0)
        corr_stat2 = st.empty()

        def _fetch_hist2(t):
            try:
                h = yf.Ticker(t).history(period="1y")
                return t, h if (h is not None and len(h) >= 60) else None
            except Exception:
                return t, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
            futs2 = {ex.submit(_fetch_hist2, t): t for t in eligible}
            done_c2 = 0
            for f in concurrent.futures.as_completed(futs2):
                done_c2 += 1
                t, h = f.result()
                if h is not None:
                    corr_hists[t] = h
                corr_prog2.progress(done_c2 / len(eligible))
                corr_stat2.markdown(f"<span style='color:#64748b;font-size:0.78rem;'>"
                                     f"📥 {t} ({done_c2}/{len(eligible)})</span>",
                                     unsafe_allow_html=True)
        corr_prog2.empty()
        corr_stat2.empty()

    # ── Analyse des corrections ──
    corr_opportunities = []
    for ticker_c, hist_c in corr_hists.items():
        fund_score_c = corr_scores.get(ticker_c, 0)
        if corr_min_score > 0 and fund_score_c > 0 and fund_score_c < corr_min_score:
            continue
        opp = analyze_correction_opportunity(
            ticker_c, hist_c, min_correction=corr_min_pct
        )
        if opp is None:
            continue
        if opp["signal_score"] < corr_min_signal:
            continue
        if corr_require_fib and not opp["fib_ok"]:
            continue
        opp["fund_score"] = fund_score_c
        corr_opportunities.append(opp)

    corr_opportunities.sort(key=lambda x: (x["signal_score"], -x["corr_pct"]), reverse=True)
    st.session_state["corr_opportunities"] = corr_opportunities

# ── Affichage des résultats ──
corr_opps = st.session_state.get("corr_opportunities", [])

if not corr_opps:
    if st.session_state.get("corr_opportunities") is not None:
        st.markdown("""
        <div style='background:#0d1117;border:1px solid #1e3a5f;border-radius:10px;
                    padding:30px;text-align:center;'>
            <div style='font-size:2rem;'>🔍</div>
            <div style='font-family:Space Mono,monospace;font-size:1rem;color:#94a3b8;margin-top:8px;'>
                Aucune opportunité détectée avec ces critères
            </div>
            <div style='font-size:0.82rem;color:#64748b;margin-top:6px;'>
                Essayez de réduire la correction minimum ou le score requis.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#0d1a2a;border:1px solid #1e3a5f;border-left:4px solid #4a90d0;
                    border-radius:8px;padding:16px 20px;font-size:0.88rem;color:#94a3b8;'>
            <strong style='color:#4a90d0;'>🎯 Module prêt</strong> — Cliquez sur
            <strong>"Scanner les Corrections"</strong> pour détecter les titres
            en repli avec retournement de momentum confirmé.
            <br><br>
            <span style='color:#64748b;font-size:0.8rem;'>
            Critères : Correction ≥ min% | StochRSI K > D depuis zone survente |
            MACD croisement haussier | Volume en hausse | Niveau Fibonacci ±3.5%
            </span>
        </div>
        """, unsafe_allow_html=True)
else:
    n_total = len(corr_opps)
    n_fib   = sum(1 for o in corr_opps if o["fib_ok"])
    n_macd  = sum(1 for o in corr_opps if o["macd_ok"])
    avg_corr = round(sum(o["corr_pct"] for o in corr_opps) / n_total, 1)

    st.markdown(f"<div style='color:#00ff88;font-size:0.85rem;margin-bottom:12px;'>"
                f"✅ {n_total} opportunité(s) — Correction moy. {avg_corr}% "
                f"| {n_fib} sur Fibonacci | {n_macd} MACD confirmé</div>",
                unsafe_allow_html=True)

    fib_colors_map = {"0.382":"#7DF9FF","0.500":"#fbbf24","0.618":"#00ff88","0.786":"#f97316"}

    for opp in corr_opps:
        corr_abs  = abs(opp["corr_pct"])
        sig       = opp["signal_score"]
        fc        = opp["force_color"]
        bar_width = min(100, int(corr_abs * 2.5))
        fib_c     = fib_colors_map.get(opp["fib_name"], "#64748b")
        bar_color = "#fbbf24" if corr_abs < 15 else "#f97316" if corr_abs < 25 else "#f87171"
        dots      = "●" * sig + "○" * (10 - sig)

        # Badge émoji titre
        if sig >= 8:   icon = "⭐"
        elif sig >= 6: icon = "🟢"
        elif sig >= 4: icon = "🟡"
        else:          icon = "🟠"

        fund_disp = f"Score IA: {opp['fund_score']}/100" if opp["fund_score"] > 0 else ""
        entry_dist = round((opp["entry_price"] - opp["price"]) / opp["price"] * 100, 1)
        in_zone    = abs(entry_dist) < 1.5

        with st.expander(
            f"{icon} {opp['ticker']}  |  "
            f"Correction: {opp['corr_pct']}%  |  "
            f"Signal: {opp['force']} ({sig}/10)  |  "
            f"Fib {opp['fib_name']}  |  "
            f"Entrée: ${opp['entry_price']}",
            expanded=(sig >= 7)
        ):
            col_left, col_right = st.columns([2, 1])

            with col_left:
                st.markdown(f"""
                <div style='background:#0d1a2a;border:1px solid {fc}33;
                            border-left:4px solid {fc};border-radius:10px;
                            padding:16px 20px;margin-bottom:10px;'>
                    <div style='display:flex;justify-content:space-between;
                                align-items:center;flex-wrap:wrap;gap:8px;'>
                        <div>
                            <span style='font-family:Space Mono,monospace;font-size:1.5rem;
                                         font-weight:700;color:{fc};'>{opp['ticker']}</span>
                            <span style='color:#64748b;font-size:0.8rem;margin-left:10px;'>
                                {fund_disp}</span>
                        </div>
                        <div>
                            <span style='font-family:Space Mono,monospace;font-size:1.3rem;
                                         color:#e2e8f0;font-weight:700;'>${opp['price']}</span>
                            <span style='color:#f87171;font-size:0.85rem;margin-left:8px;'>
                                ▼ {corr_abs}% depuis ${opp['hi52']}</span>
                        </div>
                    </div>

                    <div style='margin-top:10px;'>
                        <div style='display:flex;justify-content:space-between;
                                    font-size:0.72rem;color:#64748b;margin-bottom:3px;'>
                            <span>Correction depuis 52W High</span><span>{corr_abs}%</span>
                        </div>
                        <div style='background:#1e2a3a;border-radius:4px;height:7px;'>
                            <div style='background:linear-gradient(90deg,{bar_color},{bar_color}88);
                                        width:{bar_width}%;height:7px;border-radius:4px;'></div>
                        </div>
                    </div>

                    <div style='margin-top:12px;display:flex;flex-wrap:wrap;gap:6px;'>
                        <span style='background:{fib_c}22;border:1px solid {fib_c}55;border-radius:4px;
                                     padding:2px 10px;font-size:0.72rem;color:{fib_c};
                                     font-family:Space Mono,monospace;'>
                            Fib {opp["fib_name"]} {"✅" if opp["fib_ok"] else "~"} {opp["fib_dist"]}% écart
                        </span>
                        <span style='background:{"#00ff88" if opp["macd_ok"] else "#fbbf24"}22;
                                     border:1px solid {"#00ff88" if opp["macd_ok"] else "#fbbf24"}55;
                                     border-radius:4px;padding:2px 10px;font-size:0.72rem;
                                     color:{"#00ff88" if opp["macd_ok"] else "#fbbf24"};
                                     font-family:Space Mono,monospace;'>
                            MACD {opp["macd_label"]}
                        </span>
                        <span style='background:{"#00ff88" if opp["vol_ok"] else "#94a3b8"}22;
                                     border:1px solid {"#00ff88" if opp["vol_ok"] else "#94a3b8"}55;
                                     border-radius:4px;padding:2px 10px;font-size:0.72rem;
                                     color:{"#00ff88" if opp["vol_ok"] else "#94a3b8"};
                                     font-family:Space Mono,monospace;'>
                            Vol x{opp["vol_ratio"]}
                        </span>
                        <span style='background:{"#00ff88" if opp["above_ma200"] else "#f87171"}22;
                                     border:1px solid {"#00ff88" if opp["above_ma200"] else "#f87171"}55;
                                     border-radius:4px;padding:2px 10px;font-size:0.72rem;
                                     color:{"#00ff88" if opp["above_ma200"] else "#f87171"};
                                     font-family:Space Mono,monospace;'>
                            {"✅ Au-dessus MA200" if opp["above_ma200"] else "❌ Sous MA200"}
                        </span>
                    </div>

                    <div style='margin-top:12px;font-family:Space Mono,monospace;
                                font-size:0.78rem;color:#64748b;'>
                        Force &nbsp;
                        <span style='color:{fc};letter-spacing:1px;'>{dots}</span>
                        &nbsp;{sig}/10 — {opp["force"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Indicateurs mini-tableau
                st.markdown(f"""
                <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;'>
                    <div style='background:#111827;border:1px solid #1e3a5f;
                                border-radius:8px;padding:10px;text-align:center;'>
                        <div style='font-size:0.65rem;color:#64748b;text-transform:uppercase;'>StochRSI K</div>
                        <div style='font-family:Space Mono,monospace;font-size:1.1rem;
                                    color:{"#00ff88" if opp["stoch_k"]<50 else "#fbbf24"};
                                    font-weight:700;'>{opp['stoch_k']}</div>
                        <div style='font-size:0.62rem;color:#64748b;'>croisé il y a {opp['stoch_cross_day']}j</div>
                    </div>
                    <div style='background:#111827;border:1px solid #1e3a5f;
                                border-radius:8px;padding:10px;text-align:center;'>
                        <div style='font-size:0.65rem;color:#64748b;text-transform:uppercase;'>RSI</div>
                        <div style='font-family:Space Mono,monospace;font-size:1.1rem;
                                    color:{"#00ff88" if opp["rsi"]<45 else "#fbbf24"};
                                    font-weight:700;'>{opp['rsi']}</div>
                        <div style='font-size:0.62rem;color:#64748b;'>
                            {"Survente" if opp["rsi"]<30 else "Zone basse" if opp["rsi"]<45 else "Neutre"}</div>
                    </div>
                    <div style='background:#111827;border:1px solid #1e3a5f;
                                border-radius:8px;padding:10px;text-align:center;'>
                        <div style='font-size:0.65rem;color:#64748b;text-transform:uppercase;'>MFI</div>
                        <div style='font-family:Space Mono,monospace;font-size:1.1rem;
                                    color:{"#00ff88" if opp["mfi"]<40 else "#94a3b8"};
                                    font-weight:700;'>{opp['mfi']}</div>
                        <div style='font-size:0.62rem;color:#64748b;'>
                            {"Accumulation" if opp["mfi"]<30 else "Flux -" if opp["mfi"]<45 else "Neutre"}</div>
                    </div>
                    <div style='background:#111827;border:1px solid #1e3a5f;
                                border-radius:8px;padding:10px;text-align:center;'>
                        <div style='font-size:0.65rem;color:#64748b;text-transform:uppercase;'>Volume</div>
                        <div style='font-family:Space Mono,monospace;font-size:1.1rem;
                                    color:{"#00ff88" if opp["vol_ok"] else "#94a3b8"};
                                    font-weight:700;'>x{opp['vol_ratio']}</div>
                        <div style='font-size:0.62rem;color:#64748b;'>
                            {"Confirmé ✅" if opp["vol_ok"] else "Faible"}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_right:
                entry_color = "#00ff88" if in_zone else "#fbbf24"
                st.markdown(f"""
                <div style='background:{"#001a0f" if in_zone else "#0d1a2a"};
                            border:2px solid {entry_color}55;border-radius:10px;
                            padding:16px;text-align:center;'>
                    <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;
                                letter-spacing:0.08em;margin-bottom:6px;'>
                        {"✅ DANS LA ZONE" if in_zone else "💡 PRIX D'ENTRÉE CIBLÉ"}
                    </div>
                    <div style='font-family:Space Mono,monospace;font-size:1.6rem;
                                font-weight:700;color:{entry_color};'>
                        ${opp['entry_price']}
                    </div>
                    <div style='font-size:0.75rem;color:#64748b;margin-top:4px;'>
                        {"Entrer maintenant" if in_zone else f"Recul de {abs(entry_dist):.1f}% requis"}
                    </div>
                    <div style='margin-top:10px;padding-top:10px;border-top:1px solid #1e3a5f;'>
                        <div style='font-size:0.68rem;color:#64748b;'>Niveau Fibonacci</div>
                        <div style='font-family:Space Mono,monospace;font-size:0.88rem;
                                    color:{fib_c};font-weight:700;'>
                            {opp['fib_name']} — ${opp['fib_val']}
                        </div>
                    </div>
                </div>

                <div style='margin-top:8px;background:#0d1117;border:1px solid #1e3a5f;
                            border-radius:8px;padding:10px;font-size:0.72rem;
                            color:#64748b;line-height:1.8;'>
                    <strong style='color:#94a3b8;'>Conditions :</strong><br>
                    {"✅" if opp["stoch_cross_ok"] else "❌"} StochRSI K>D survente<br>
                    {"✅" if opp["macd_ok"] else "⚠️"} MACD croisement<br>
                    {"✅" if opp["vol_ok"] else "⚠️"} Volume en hausse<br>
                    {"✅" if opp["fib_ok"] else "~"} Niveau Fibonacci<br>
                    {"✅" if opp["above_ma200"] else "❌"} Au-dessus MA200<br>
                    MA50: ${opp['ma50']} | MA200: ${opp['ma200']}
                </div>
                """, unsafe_allow_html=True)

    # ── Export CSV ──
    st.markdown("---")
    corr_export = pd.DataFrame([{
        "Ticker": o["ticker"], "Prix $": o["price"],
        "Correction %": o["corr_pct"], "Signal Score": o["signal_score"],
        "Force": o["force"], "StochRSI K": o["stoch_k"], "RSI": o["rsi"],
        "MFI": o["mfi"], "MACD confirmé": "Oui" if o["macd_ok"] else "Non",
        "Volume ratio": o["vol_ratio"], "Fibonacci": o["fib_name"],
        "Dist. Fib %": o["fib_dist"], "Entrée suggérée $": o["entry_price"],
        "Score IA": o["fund_score"], "MA50": o["ma50"], "MA200": o["ma200"],
        "Au-dessus MA200": "Oui" if o["above_ma200"] else "Non",
    } for o in corr_opps])

    col_csv, _ = st.columns([1, 3])
    with col_csv:
        st.download_button(
            label="⬇️ Exporter CSV",
            data=corr_export.to_csv(index=False).encode("utf-8"),
            file_name=f"corrections_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="corr_csv_export"
        )

# PAPER TRADING + JOURNAL
st.markdown("---")
st.markdown("## Paper Trading & Journal")

pt_tab, journal_tab = st.tabs(["Paper Trading", "Journal Reel"])

with pt_tab:
    st.markdown("### Simulation de trades")
    paper_summary = get_paper_summary()
    if paper_summary["n_closed"] > 0:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Trades fermes", paper_summary["n_closed"])
        c2.metric("Win Rate", f"{paper_summary['win_rate']}%")
        c3.metric("P&L moyen", f"{paper_summary['avg_pnl']}%")
        c4.metric("P&L total", f"{paper_summary['total_pnl']}%")
        c5.metric("Ouverts", paper_summary.get("n_open",0))

    with st.expander("Ajouter un trade fictif", expanded=False):
        pa,pb,pc = st.columns(3)
        with pa:
            pt_ticker = st.text_input("Ticker", key="pt_tk").upper()
            pt_entry  = st.number_input("Prix entree", min_value=0.01, value=100.0, key="pt_en")
        with pb:
            pt_stop   = st.number_input("Stop-loss", min_value=0.01, value=97.0, key="pt_st")
            pt_target = st.number_input("Target", min_value=0.01, value=106.0, key="pt_tg")
        with pc:
            pt_strat  = st.selectbox("Strategie", ["A +5%","B +7%","C Vendredi","D Stop 3%","E Stop 5%"], key="pt_sr")
            pt_sector = st.text_input("Secteur", value="N/A", key="pt_sec")
        if st.button("Ajouter trade fictif", key="pt_add"):
            if pt_ticker:
                rr = round((pt_target - pt_entry) / (pt_entry - pt_stop), 2) if pt_entry > pt_stop else 0
                add_paper_trade(ticker=pt_ticker, entry_price=pt_entry, stop_price=pt_stop,
                    target_price=pt_target, conv_n=0, score=0, strategy=pt_strat,
                    sector=pt_sector, week_date=datetime.now().strftime("%Y-W%V"))
                st.success(f"Trade {pt_ticker} ajoute! R/R: {rr}:1")
                st.rerun()

    paper_trades  = update_paper_results()
    open_trades   = [t for t in paper_trades if t.get("status") == "OPEN"]
    closed_trades = [t for t in paper_trades if t.get("status") == "CLOSED"]

    if open_trades:
        st.markdown("**Trades ouverts:**")
        rows_open = []
        for t in open_trades:
            ep = float(t.get("entry_price",0))
            cp = float(t.get("current_price", ep))
            sp = float(t.get("stop_price", ep))
            tp = float(t.get("target_price", ep))
            pnl = round((cp - ep) / ep * 100, 2) if ep > 0 else 0
            rr  = round((tp - ep) / (ep - sp), 2) if ep > sp else 0
            rows_open.append({"Ticker":t.get("ticker"),"Entree":f"${ep}",
 "Stop":f"${sp}","Target":f"${tp}","R/R":f"{rr}:1",
 "Prix act.":f"${cp}","P&L act.":f"{pnl:+.2f}%","Strat":t.get("strategy")})
        st.dataframe(rows_open, use_container_width=True)
        close_options = {f"{t['ticker']} entree ${t['entry_price']}": t["id"] for t in open_trades}
        selected = st.selectbox("Trade a fermer", list(close_options.keys()), key="pt_close_sel")
        exit_price = st.number_input("Prix de sortie", min_value=0.01, value=100.0, key="pt_exit")
        if st.button("Fermer ce trade", key="pt_close_btn"):
            from trading_tools import close_paper_trade
            close_paper_trade(close_options[selected], exit_price)
            st.success("Trade ferme!")
            st.rerun()

    if closed_trades:
        with st.expander(f"Historique ({len(closed_trades)} trades fermes)"):
            rows_cl = []
            for t in closed_trades:
                ep  = float(t.get("entry_price",0))
                xp  = float(t.get("exit_price",0))
                pnl = round((xp - ep) / ep * 100, 2) if ep > 0 else 0
                rows_cl.append({"Ticker":t.get("ticker"),"Entree":f"${ep}",
 "Sortie":f"${xp}","P&L":f"{pnl:+.2f}%",
 "Resultat":"WIN" if pnl > 0.5 else "LOSS" if pnl < -0.5 else "EVEN",
 "Strat":t.get("strategy"),"Semaine":t.get("week")})
            st.dataframe(rows_cl, use_container_width=True)

with journal_tab:
    st.markdown("### Journal de trades reels")
    journal_summary = get_journal_summary()
    if journal_summary["n_closed"] > 0:
        j1,j2,j3,j4 = st.columns(4)
        j1.metric("Trades reels", journal_summary["n_closed"])
        j2.metric("Win Rate reel", f"{journal_summary['win_rate']}%")
        j3.metric("P&L moyen", f"{journal_summary['avg_pnl']}%")
        j4.metric("P&L total", f"{journal_summary['total_pnl']}%")

    with st.expander("Enregistrer un trade reel", expanded=True):
        ja,jb = st.columns(2)
        with ja:
            j_ticker = st.text_input("Ticker", key="j_tk").upper()
            j_entry  = st.number_input("Prix entree", min_value=0.01, value=21.68, key="j_en")
            j_stop   = st.number_input("Stop-loss", min_value=0.01, value=20.95, key="j_st")
        with jb:
            j_target = st.number_input("Target", min_value=0.01, value=23.00, key="j_tg")
            j_strat  = st.selectbox("Strategie", ["A +5%","B +7%","C Vendredi","D Stop 3%","E Stop 5%"], index=2, key="j_sr")
            j_notes  = st.text_input("Notes", key="j_nt")
        if st.button("Enregistrer le trade", key="j_add"):
            if j_ticker:
                rr = round((j_target - j_entry) / (j_entry - j_stop), 2) if j_entry > j_stop else 0
                add_journal_trade(j_ticker, j_entry, j_stop, j_target, j_strat, "N/A", j_notes)
                st.success(f"Trade {j_ticker} enregistre! R/R: {rr}:1")
                st.rerun()

    journal_trades = get_journal_summary().get("trades", [])
    open_j  = [t for t in journal_trades if t.get("status") == "OPEN"]
    closed_j= [t for t in journal_trades if t.get("status") == "CLOSED"]

    if open_j:
        st.markdown("**Trades ouverts:**")
        rows_j = []
        for t in open_j:
            ep = float(t.get("entry_price",0))
            sp = float(t.get("stop_price",ep))
            tp = float(t.get("target_price",ep))
            rr = round((tp - ep) / (ep - sp), 2) if ep > sp else 0
            rows_j.append({"Ticker":t.get("ticker"),"Date":t.get("date_entry"),
 "Entree":f"${ep}","Stop":f"${sp}","Target":f"${tp}",
 "R/R":f"{rr}:1","Strat":t.get("strategy"),"Notes":t.get("notes","")})
        st.dataframe(rows_j, use_container_width=True)
        close_j_opts = {f"{t['ticker']} entree ${t['entry_price']}": t["id"] for t in open_j}
        sel_j  = st.selectbox("Trade a fermer", list(close_j_opts.keys()), key="j_close_sel")
        exit_j = st.number_input("Prix de sortie", min_value=0.01, value=21.76, key="j_exit")
        if st.button("Fermer ce trade reel", key="j_close_btn"):
            close_journal_trade(close_j_opts[sel_j], exit_j)
            st.success("Trade ferme!")
            st.rerun()

    if closed_j:
        with st.expander(f"Historique reel ({len(closed_j)} trades)"):
            rows_jc = []
            for t in closed_j:
                ep  = float(t.get("entry_price",0))
                xp  = float(t.get("exit_price",0))
                pnl = round((xp - ep) / ep * 100, 2) if ep > 0 else 0
                rows_jc.append({"Ticker":t.get("ticker"),"Date":t.get("date_entry"),
 "Entree":f"${ep}","Sortie":f"${xp}","P&L":f"{pnl:+.2f}%",
 "Resultat":"WIN" if pnl > 0.5 else "LOSS" if pnl < -0.5 else "EVEN",
 "Notes":t.get("notes","")})
            st.dataframe(rows_jc, use_container_width=True)
