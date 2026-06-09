import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import concurrent.futures
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
from datetime import datetime
from io import BytesIO

# Module base de données
try:
    from database import (
        test_connection, save_scan, get_latest_scans,
        get_ticker_history, get_top_historical, get_score_evolution,
        watchlist_add, watchlist_remove, watchlist_get, get_global_stats
    )
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

st.set_page_config(page_title="AlphaScreen US", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }
h1, h2, h3 { font-family: 'Space Mono', monospace; color: #00ff88 !important; }
.metric-card { background:linear-gradient(135deg,#111827,#1a2332); border:1px solid #1e3a5f; border-radius:12px; padding:20px; text-align:center; margin:4px; }
.metric-value { font-family:'Space Mono',monospace; font-size:2rem; font-weight:700; color:#00ff88; }
.metric-label { font-size:0.8rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; margin-top:4px; }
.market-banner { border-radius:10px; padding:16px 20px; margin-bottom:20px; }
.prefilter-banner { background:#0d1a2a; border:1px solid #1e3a5f; border-left:4px solid #4a90d0; border-radius:8px; padding:12px 18px; margin:10px 0; font-size:0.85rem; font-family:'Space Mono',monospace; }
.ticker-badge { display:inline-block; background:#00ff8822; border:1px solid #00ff8866; color:#00ff88; font-family:'Space Mono',monospace; font-size:0.9rem; padding:3px 12px; border-radius:4px; margin-right:8px; font-weight:700; }
.ai-analysis-box { background:linear-gradient(135deg,#0f1f35,#0a1628); border:1px solid #00ff8844; border-left:4px solid #00ff88; border-radius:8px; padding:16px 20px; margin:8px 0; font-size:0.9rem; line-height:1.6; }
.conviction-box { background:linear-gradient(135deg,#1a0f35,#0f0a28); border:1px solid #a78bfa44; border-left:4px solid #a78bfa; border-radius:8px; padding:12px 16px; margin:6px 0; font-size:0.85rem; }
.scan-info { background:#0d1a2a; border:1px solid #1e3a5f; border-radius:8px; padding:10px 16px; font-size:0.82rem; color:#64748b; margin-bottom:12px; }
.stButton > button { background:linear-gradient(135deg,#00ff88,#00cc6a) !important; color:#0a0e1a !important; font-family:'Space Mono',monospace !important; font-weight:700 !important; border:none !important; border-radius:8px !important; padding:12px 24px !important; }
div[data-testid="stDataFrame"] { border:1px solid #1e3a5f; border-radius:10px; overflow:hidden; }
.stProgress > div > div { background:linear-gradient(90deg,#00ff88,#00cc6a) !important; }
section[data-testid="stSidebar"] { background:#0d1117 !important; border-right:1px solid #1e3a5f; }
.speed-box { background:#0d1a2a; border:1px solid #00ff8833; border-left:3px solid #00ff88; border-radius:6px; padding:10px 14px; font-size:0.8rem; color:#86efac; margin:6px 0; }
</style>""", unsafe_allow_html=True)

# ─────────────────────────────
# 📌 UNIVERS US
# ─────────────────────────────
SP500 = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE",
    "ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AEP","AXP","AIG","AMT",
    "AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","APO","AAPL","AMAT","APTV","ACGL",
    "ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC",
    "BAX","BDX","BRK-B","BBY","BIO","BIIB","BLK","BX","BA","BSX","BMY","AVGO","BR","BRO",
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
    "NDAQ","NTAP","NFLX","NEM","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE",
    "NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PANW",
    "PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL",
    "PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","PWR","QCOM","DGX","RL","RJF","RTX",
    "O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC",
    "SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SNA","SOLV","SO","LUV","SWK","SBUX","STT",
    "STLD","STE","SYK","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY",
    "TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN",
    "USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO","VRSN",
    "VRSK","VZ","VRTX","VTRS","VICI","V","VST","VMC","WRB","GWW","WAB","WBA","WMT","DIS","WBD",
    "WM","WAT","WEC","WFC","WELL","WST","WDC","WY","WMB","WTW","WYNN","XEL","XYL","YUM","ZBRA",
    "ZBH","ZTS"
]
NASDAQ100_EXTRA = [
    "MELI","ASML","TEAM","TTD","DASH","DDOG","ZS","MDB","OKTA","WDAY",
    "PSTG","CCEP","GFS","SMCI","SIRI","PDD","ILMN","SGEN","MRVL","CRWD"
]

# ─────────────────────────────
# 🌍 MARCHÉ GLOBAL (cache 15 min)
# ─────────────────────────────
@st.cache_data(ttl=900)
def get_market_status():
    try:
        spy=yf.Ticker("SPY").history(period="1y")
        vix=yf.Ticker("^VIX").history(period="5d")
        qqq=yf.Ticker("QQQ").history(period="1y")
        sp=float(spy["Close"].iloc[-1])
        sm50=float(spy["Close"].rolling(50).mean().iloc[-1])
        sm200=float(spy["Close"].rolling(200).mean().iloc[-1])
        sv=round((sp-sm50)/sm50*100,2)
        qp=float(qqq["Close"].iloc[-1])
        qm50=float(qqq["Close"].rolling(50).mean().iloc[-1])
        qv=round((qp-qm50)/qm50*100,2)
        vix_val=float(vix["Close"].iloc[-1]) if not vix.empty else 20.0
        d=spy["Close"].diff()
        g=d.where(d>0,0).rolling(14).mean()
        l=-d.where(d<0,0).rolling(14).mean()
        rsi=float(100-(100/(1+g/l.clip(lower=1e-10))).iloc[-1])
        if sp>sm200 and sp>sm50 and vix_val<20:
            regime,color,emoji,bonus="HAUSSIER","#00ff88","🟢",5
        elif sp>sm200 and vix_val<25:
            regime,color,emoji,bonus="NEUTRE","#fbbf24","🟡",0
        elif vix_val>=30:
            regime,color,emoji,bonus="VOLATIL","#f97316","🟠",-10
        else:
            regime,color,emoji,bonus="BAISSIER","#f87171","🔴",-15
        return dict(regime=regime,color=color,emoji=emoji,bonus=bonus,
                    spy_vs_ma50=sv,qqq_vs_ma50=qv,vix=round(vix_val,1),
                    spy_rsi=round(rsi,1),
                    vix_label="Faible" if vix_val<15 else "Modéré" if vix_val<20 else "Élevé" if vix_val<30 else "Extrême")
    except Exception as e:
        st.warning(f"⚠️ Données marché indisponibles: {e}")
        return dict(regime="INCONNU",color="#64748b",emoji="⚪",bonus=0,
                    spy_vs_ma50=0,qqq_vs_ma50=0,vix=0,spy_rsi=50,vix_label="N/D")

# ─────────────────────────────
# ⚡ BATCH DOWNLOAD PRIX (cache 4h)
# ─────────────────────────────
@st.cache_data(ttl=14400)
def batch_download_prices(tickers_tuple, period="1y"):
    tickers=list(tickers_tuple)
    try:
        raw=yf.download(tickers,period=period,auto_adjust=True,
                        progress=False,group_by="ticker",threads=True)
        prices={}
        if len(tickers)==1:
            t=tickers[0]
            if not raw.empty:
                prices[t]=raw[["Close","Volume","High","Low","Open"]].dropna()
        else:
            for t in tickers:
                try:
                    df_t=raw[t][["Close","Volume","High","Low","Open"]].dropna()
                    if len(df_t)>=50: prices[t]=df_t
                except Exception:
                    continue
        return prices
    except Exception as e:
        st.warning(f"⚠️ Erreur batch download: {e}")
        return {}

# ─────────────────────────────
# 🔬 PRÉ-FILTRE — prix, volume, MA200
# Instantané car tourne sur données locales
# ─────────────────────────────
def prefilter_from_prices(ticker, price_data, min_price, max_price, min_vol):
    try:
        df=price_data.get(ticker)
        if df is None or df.empty or len(df)<50:
            return False,"Données insuffisantes",None
        price=float(df["Close"].iloc[-1])
        avg_vol=float(df["Volume"].rolling(30).mean().iloc[-1])
        ma200=float(df["Close"].rolling(min(200,len(df))).mean().iloc[-1])
        if price<min_price:  return False,f"Prix trop bas ({price:.2f} USD)",None
        if price>max_price:  return False,f"Prix trop eleve ({price:.2f} USD)",None
        if avg_vol<min_vol:  return False,f"Volume faible ({int(avg_vol):,})",None
        if price<ma200:      return False,f"Sous MA200 ({price:.2f} < {ma200:.2f})",None
        return True,"✅ OK",{"df":df,"price":price,"ma200":ma200,"avg_vol":avg_vol}
    except Exception as e:
        return False,f"Erreur: {str(e)[:50]}",None

# ─────────────────────────────
# 📉 TABLEAU DE BORD D'ENTRÉE
# Chandelier + BB + MA + Volume | RSI | MACD | Supports/Résistances
# ─────────────────────────────
# 📐 DÉTECTION DE PATTERNS CHARTISTES
# 8 patterns : 5 haussiers + 3 baissiers
# ─────────────────────────────
def detect_patterns(close, high, low, volume, ma50, ma200):
    """
    Détecte 8 patterns chartistes sur les données de prix.
    Retourne une liste de dicts {name, type, strength, description, icon}
    """
    patterns = []
    n = len(close)
    if n < 50:
        return patterns

    c = close.values
    h = high.values
    l = low.values
    v = volume.values
    m50  = ma50.values
    m200 = ma200.values

    # ── 1. GOLDEN CROSS (haussier) ──
    # MA50 croise au-dessus de MA200 dans les 10 derniers jours
    try:
        for i in range(max(1, n-10), n):
            if (m50[i] > m200[i] and m50[i-1] <= m200[i-1] and
                    not np.isnan(m50[i]) and not np.isnan(m200[i])):
                patterns.append({
                    "name": "Golden Cross",
                    "type": "HAUSSIER",
                    "strength": "FORT",
                    "icon": "⭐",
                    "color": "#00ff88",
                    "desc": "MA50 vient de croiser au-dessus de MA200 — signal haussier long terme très fort"
                })
                break
    except Exception:
        pass

    # ── 2. DEATH CROSS (baissier) ──
    try:
        for i in range(max(1, n-10), n):
            if (m50[i] < m200[i] and m50[i-1] >= m200[i-1] and
                    not np.isnan(m50[i]) and not np.isnan(m200[i])):
                patterns.append({
                    "name": "Death Cross",
                    "type": "BAISSIER",
                    "strength": "FORT",
                    "icon": "💀",
                    "color": "#f87171",
                    "desc": "MA50 vient de croiser sous MA200 — signal baissier long terme — éviter"
                })
                break
    except Exception:
        pass

    # ── 3. BULL FLAG (haussier) ──
    # Fort mouvement haussier suivi d'une consolidation en légère baisse
    try:
        lookback = min(40, n-1)
        # Pole : hausse > 8% sur 5-15 jours
        for pole_len in range(5, 16):
            if n - pole_len - 20 < 0:
                continue
            pole_start = n - pole_len - 20
            pole_end   = n - 20
            if pole_end <= pole_start:
                continue
            pole_gain = (c[pole_end] - c[pole_start]) / c[pole_start] * 100
            if pole_gain > 8:
                # Flag : consolidation légèrement baissière sur 10-20 derniers jours
                flag_c = c[pole_end:]
                if len(flag_c) >= 5:
                    flag_range = (max(flag_c) - min(flag_c)) / max(flag_c) * 100
                    flag_slope = (flag_c[-1] - flag_c[0]) / flag_c[0] * 100
                    if flag_range < pole_gain * 0.5 and -5 < flag_slope < 2:
                        patterns.append({
                            "name": "Bull Flag",
                            "type": "HAUSSIER",
                            "strength": "FORT",
                            "icon": "🚩",
                            "color": "#00ff88",
                            "desc": f"Pole +{pole_gain:.1f}% suivi d'une consolidation — breakout probable vers le haut"
                        })
                        break
    except Exception:
        pass

    # ── 4. BEAR FLAG (baissier) ──
    try:
        for pole_len in range(5, 16):
            if n - pole_len - 20 < 0:
                continue
            pole_start = n - pole_len - 20
            pole_end   = n - 20
            if pole_end <= pole_start:
                continue
            pole_drop = (c[pole_start] - c[pole_end]) / c[pole_start] * 100
            if pole_drop > 8:
                flag_c = c[pole_end:]
                if len(flag_c) >= 5:
                    flag_slope = (flag_c[-1] - flag_c[0]) / flag_c[0] * 100
                    flag_range = (max(flag_c) - min(flag_c)) / max(flag_c) * 100
                    if flag_range < pole_drop * 0.5 and -2 < flag_slope < 5:
                        patterns.append({
                            "name": "Bear Flag",
                            "type": "BAISSIER",
                            "strength": "FORT",
                            "icon": "🏴",
                            "color": "#f87171",
                            "desc": f"Baisse -{pole_drop:.1f}% suivie d'un rebond faible — continuation baissière probable"
                        })
                        break
    except Exception:
        pass

    # ── 5. DOUBLE BOTTOM (haussier) ──
    # Deux creux similaires séparés de 10-40 jours
    try:
        lookback = min(60, n-1)
        lows_idx = []
        for i in range(5, lookback-5):
            idx = n - lookback + i
            if idx <= 0 or idx >= n-1:
                continue
            if l[idx] < l[idx-1] and l[idx] < l[idx+1] and \
               l[idx] < l[idx-2] and l[idx] < l[idx+2]:
                lows_idx.append(idx)

        for i in range(len(lows_idx)-1):
            for j in range(i+1, len(lows_idx)):
                gap = lows_idx[j] - lows_idx[i]
                if 10 <= gap <= 40:
                    diff = abs(l[lows_idx[i]] - l[lows_idx[j]]) / l[lows_idx[i]] * 100
                    if diff < 3.0:
                        # Vérifier que le prix actuel est au-dessus des deux creux
                        neckline = max(h[lows_idx[i]:lows_idx[j]+1])
                        if c[-1] > l[lows_idx[j]] * 1.02:
                            strength = "FORT" if c[-1] > neckline * 0.98 else "EN FORMATION"
                            patterns.append({
                                "name": "Double Bottom",
                                "type": "HAUSSIER",
                                "strength": strength,
                                "icon": "W",
                                "color": "#00ff88",
                                "desc": f"Deux creux similaires ({diff:.1f}% d'écart) — renversement haussier {strength.lower()}"
                            })
                            break
            else:
                continue
            break
    except Exception:
        pass

    # ── 6. DOUBLE TOP (baissier) ──
    try:
        lookback = min(60, n-1)
        highs_idx = []
        for i in range(5, lookback-5):
            idx = n - lookback + i
            if idx <= 0 or idx >= n-1:
                continue
            if h[idx] > h[idx-1] and h[idx] > h[idx+1] and \
               h[idx] > h[idx-2] and h[idx] > h[idx+2]:
                highs_idx.append(idx)

        for i in range(len(highs_idx)-1):
            for j in range(i+1, len(highs_idx)):
                gap = highs_idx[j] - highs_idx[i]
                if 10 <= gap <= 40:
                    diff = abs(h[highs_idx[i]] - h[highs_idx[j]]) / h[highs_idx[i]] * 100
                    if diff < 3.0:
                        neckline = min(l[highs_idx[i]:highs_idx[j]+1])
                        if c[-1] < h[highs_idx[j]] * 0.98:
                            strength = "FORT" if c[-1] < neckline * 1.02 else "EN FORMATION"
                            patterns.append({
                                "name": "Double Top",
                                "type": "BAISSIER",
                                "strength": strength,
                                "icon": "M",
                                "color": "#f87171",
                                "desc": f"Deux sommets similaires ({diff:.1f}% d'écart) — renversement baissier {strength.lower()}"
                            })
                            break
            else:
                continue
            break
    except Exception:
        pass

    # ── 7. CUP AND HANDLE (haussier) ──
    # Formation en U sur 20-60 jours + handle courte
    try:
        for cup_len in range(20, min(61, n-10)):
            cup_start = n - cup_len - 10
            cup_end   = n - 10
            if cup_start < 0:
                continue
            cup_prices = c[cup_start:cup_end+1]
            cup_high_l = c[cup_start]
            cup_high_r = c[cup_end]
            cup_low    = min(cup_prices)

            # Forme en U : les deux bords proches, le bas bien en dessous
            depth = (min(cup_high_l, cup_high_r) - cup_low) / min(cup_high_l, cup_high_r) * 100
            sym   = abs(cup_high_l - cup_high_r) / cup_high_l * 100

            if 8 < depth < 35 and sym < 5:
                # Handle : légère consolidation sur les 10 derniers jours
                handle = c[cup_end:]
                if len(handle) >= 3:
                    handle_drop = (max(handle) - min(handle)) / max(handle) * 100
                    if handle_drop < depth * 0.5:
                        patterns.append({
                            "name": "Cup and Handle",
                            "type": "HAUSSIER",
                            "strength": "TRÈS FORT",
                            "icon": "C",
                            "color": "#7DF9FF",
                            "desc": f"Coupe ({depth:.1f}% de profondeur) + handle — un des patterns les plus fiables (Buffett)"
                        })
                        break
    except Exception:
        pass

    # ── 8. BREAKOUT DE RANGE (haussier ou neutre) ──
    # Prix qui casse au-dessus d'une résistance avec volume élevé
    try:
        lookback = min(40, n-1)
        range_high = max(h[n-lookback:n-3])
        range_low  = min(l[n-lookback:n-3])
        range_pct  = (range_high - range_low) / range_low * 100

        # Vérifier si le prix actuel casse la résistance
        if c[-1] > range_high * 1.005 and range_pct < 15:
            avg_vol = np.mean(v[n-lookback:n-3])
            vol_spike = v[-1] > avg_vol * 1.3 if avg_vol > 0 else False
            strength = "FORT" if vol_spike else "MODÉRÉ"
            patterns.append({
                "name": "Breakout de Range",
                "type": "HAUSSIER",
                "strength": strength,
                "icon": "↗",
                "color": "#a78bfa",
                "desc": f"Cassure au-dessus de la résistance {range_high:.2f} "
                        f"{'avec volume fort' if vol_spike else 'volume modéré'} — "
                        f"continuation probable"
            })
    except Exception:
        pass

    return patterns


# ─────────────────────────────
def render_entry_dashboard(ticker, pre_data):
    """Affiche le tableau de bord complet d'analyse technique pour un titre."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Récupère les données de prix (déjà en cache via pre_data)
        if pre_data and "df" in pre_data:
            df_raw = pre_data["df"].copy()
        else:
            st.info("Données graphiques non disponibles.")
            return

        # Garder les 180 derniers jours pour lisibilité
        df_c = df_raw.tail(180).copy()
        df_c.index = pd.to_datetime(df_c.index)
        dates = df_c.index

        close  = df_c["Close"]
        high   = df_c["High"]
        low    = df_c["Low"]
        open_  = df_c["Open"]
        volume = df_c["Volume"]

        # ── Indicateurs techniques ──
        ma20  = close.rolling(20).mean()
        ma50  = close.rolling(50).mean()
        ma200 = close.rolling(min(200, len(close))).mean()

        # Bandes de Bollinger (20 périodes, ±2σ)
        bb_std  = close.rolling(20).std()
        bb_up   = ma20 + 2 * bb_std
        bb_low  = ma20 - 2 * bb_std

        # RSI
        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = -delta.where(delta < 0, 0).rolling(14).mean()
        rsi   = 100 - (100 / (1 + gain / loss.clip(lower=1e-10)))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist   = macd_line - signal_line

        # Supports & Résistances (pivots sur 52 semaines)
        hi52 = float(df_raw["High"].rolling(min(252, len(df_raw))).max().iloc[-1])
        lo52 = float(df_raw["Low"].rolling(min(252, len(df_raw))).min().iloc[-1])
        pivot = (hi52 + lo52 + float(close.iloc[-1])) / 3
        r1 = 2 * pivot - lo52
        s1 = 2 * pivot - hi52
        r2 = pivot + (hi52 - lo52)
        s2 = pivot - (hi52 - lo52)

        # ── Retracement de Fibonacci ──
        # Détection automatique du dernier swing majeur (90 jours)
        fib_lookback = min(90, len(df_c) - 1)
        fib_high = float(high.iloc[-fib_lookback:].max())
        fib_low  = float(low.iloc[-fib_lookback:].min())
        fib_range = fib_high - fib_low

        # 6 niveaux Fibonacci
        fib_levels = {
            "0.000": fib_low,
            "0.236": fib_high - 0.236 * fib_range,
            "0.382": fib_high - 0.382 * fib_range,
            "0.500": fib_high - 0.500 * fib_range,
            "0.618": fib_high - 0.618 * fib_range,
            "0.786": fib_high - 0.786 * fib_range,
            "1.000": fib_high,
        }
        fib_colors = {
            "0.000": "#64748b",
            "0.236": "#a78bfa",
            "0.382": "#7DF9FF",
            "0.500": "#fbbf24",
            "0.618": "#00ff88",
            "0.786": "#f97316",
            "1.000": "#64748b",
        }

        # Niveau Fibonacci le plus proche du prix actuel
        price_now = float(close.iloc[-1])
        fib_nearest = min(fib_levels.items(),
                          key=lambda x: abs(x[1] - price_now))
        fib_nearest_name  = fib_nearest[0]
        fib_nearest_price = fib_nearest[1]
        fib_dist_pct = abs(price_now - fib_nearest_price) / price_now * 100

        rsi_now   = float(rsi.iloc[-1])
        macd_now  = float(macd_hist.iloc[-1])
        bb_pct    = (price_now - float(bb_low.iloc[-1])) / (float(bb_up.iloc[-1]) - float(bb_low.iloc[-1]) + 1e-10)

        # ── Détection des patterns ──
        patterns = detect_patterns(close, high, low, volume, ma50, ma200)

        # ── Signal d'entrée automatique ──
        entry_signals = []
        entry_score   = 0

        # RSI
        if 40 <= rsi_now <= 60:
            entry_signals.append("✅ RSI en zone neutre-haussière — bon timing")
            entry_score += 2
        elif rsi_now < 40:
            entry_signals.append("✅ RSI bas — zone de rebond potentiel")
            entry_score += 3
        elif rsi_now > 70:
            entry_signals.append("⚠️ RSI suracheté — attendre un repli")
            entry_score -= 1

        # MACD
        if macd_now > 0 and float(macd_hist.iloc[-2]) < 0:
            entry_signals.append("✅ Croisement MACD haussier — signal fort")
            entry_score += 3
        elif macd_now > 0:
            entry_signals.append("✅ MACD positif — momentum haussier")
            entry_score += 1
        else:
            entry_signals.append("❌ MACD négatif — momentum baissier")
            entry_score -= 1

        # Bollinger
        if bb_pct < 0.2:
            entry_signals.append("✅ Prix proche bande basse BB — zone de valeur")
            entry_score += 2
        elif bb_pct > 0.8:
            entry_signals.append("⚠️ Prix proche bande haute BB — risque de rejet")
            entry_score -= 1
        else:
            entry_signals.append("~ Prix dans les bandes Bollinger — zone neutre")

        # Support/Résistance
        dist_s1 = abs(price_now - s1) / price_now * 100
        dist_r1 = abs(price_now - r1) / price_now * 100
        if dist_s1 < 3:
            entry_signals.append(f"✅ Prix proche support S1 ({s1:.2f}) — zone d'achat")
            entry_score += 2
        elif dist_r1 < 3:
            entry_signals.append(f"⚠️ Prix proche resistance R1 ({r1:.2f}) — attendre cassure")
            entry_score -= 1

        # MA50
        if price_now > float(ma50.iloc[-1]):
            entry_signals.append("✅ Prix au-dessus MA50 — tendance à court terme positive")
            entry_score += 1
        else:
            entry_signals.append("❌ Prix sous MA50 — tendance court terme négative")
            entry_score -= 1

        # ── Signal Fibonacci ──
        if fib_dist_pct < 2.0:
            fib_key_levels = {"0.382", "0.500", "0.618"}
            if fib_nearest_name in fib_key_levels:
                if fib_nearest_name == "0.618":
                    entry_signals.append(f"✅ Prix sur Fibonacci 0.618 (Golden Ratio) — zone d'achat optimale")
                    entry_score += 3
                elif fib_nearest_name == "0.500":
                    entry_signals.append(f"✅ Prix sur Fibonacci 0.500 — support psychologique fort")
                    entry_score += 2
                elif fib_nearest_name == "0.382":
                    entry_signals.append(f"✅ Prix sur Fibonacci 0.382 — premier support clé")
                    entry_score += 2
            elif fib_nearest_name == "0.786":
                entry_signals.append(f"⚠️ Prix sur Fibonacci 0.786 — support profond, risque élevé")
                entry_score += 1
            else:
                entry_signals.append(f"~ Prix proche Fibonacci {fib_nearest_name} ({fib_nearest_price:.2f})")
        else:
            entry_signals.append(f"~ Fibonacci: niveau le plus proche = {fib_nearest_name} ({fib_dist_pct:.1f}% d'écart)")

        # Impact patterns sur le score d'entrée
        for p in patterns:
            if p["type"] == "HAUSSIER":
                if p["strength"] == "TRÈS FORT":  entry_score += 3
                elif p["strength"] == "FORT":      entry_score += 2
                else:                              entry_score += 1
            else:  # BAISSIER
                if p["strength"] == "FORT":        entry_score -= 2
                else:                              entry_score -= 1

        # Verdict final
        if entry_score >= 5:
            verdict = "🟢 ENTRER MAINTENANT"
            verdict_color_hex = "#00ff88"
            verdict_bg = "#001a0f"
        elif entry_score >= 2:
            verdict = "🟡 CONDITIONS FAVORABLES"
            verdict_color_hex = "#fbbf24"
            verdict_bg = "#1a1400"
        elif entry_score >= 0:
            verdict = "🟠 ATTENDRE CONFIRMATION"
            verdict_color_hex = "#f97316"
            verdict_bg = "#1a0e00"
        else:
            verdict = "🔴 ÉVITER — REPLI EN COURS"
            verdict_color_hex = "#f87171"
            verdict_bg = "#1a0000"

        # ── Affichage verdict ──
        fib_color = fib_colors.get(fib_nearest_name, "#64748b")
        _macd_sign = "+" if macd_now > 0 else ""
        _signals_joined = " · ".join(entry_signals)
        st.markdown(
            f"<div style='background:{verdict_bg};border:2px solid {verdict_color_hex}55;"
            f"border-radius:10px;padding:14px 18px;margin-bottom:8px;'>"
            f"<span style='font-family:Space Mono,monospace;font-size:1.1rem;font-weight:700;"
            f"color:{verdict_color_hex};'>{verdict}</span>"
            f"&nbsp;&nbsp;<span style='background:{fib_color}22;border:1px solid {fib_color}55;"
            f"border-radius:4px;padding:2px 8px;font-size:0.78rem;color:{fib_color};"
            f"font-family:Space Mono,monospace;'>Fib {fib_nearest_name}</span>"
            f"<br><span style='color:#64748b;font-size:0.78rem;'>"
            f"Score: {entry_score:+d}pts"
            f" | RSI: {rsi_now:.1f}"
            f" | MACD: {_macd_sign}{macd_now:.3f}"
            f" | BB%: {bb_pct*100:.0f}%"
            f" | Fib nearest: {fib_nearest_name} ({fib_dist_pct:.1f}% ecart)"
            f"</span>"
            f"<br><span style='color:#94a3b8;font-size:0.78rem;'>{_signals_joined}</span>"
            f"</div>",
            unsafe_allow_html=True)

        # ── Box Fibonacci ──
        fib_key_display = [
            ("0.236", fib_levels["0.236"], fib_colors["0.236"]),
            ("0.382", fib_levels["0.382"], fib_colors["0.382"]),
            ("0.500", fib_levels["0.500"], fib_colors["0.500"]),
            ("0.618", fib_levels["0.618"], fib_colors["0.618"]),
            ("0.786", fib_levels["0.786"], fib_colors["0.786"]),
        ]
        fib_cols = st.columns(5)
        for idx, (fname, fval, fc) in enumerate(fib_key_display):
            dist = round(abs(price_now - fval) / price_now * 100, 1)
            is_nearest = fname == fib_nearest_name
            border = f"2px solid {fc}" if is_nearest else f"1px solid {fc}44"
            label_extra = " ACTUEL" if is_nearest and fib_dist_pct < 2 else ""
            fib_cols[idx].markdown(
                f"<div style='background:#0d1117;border:{border};"
                f"border-radius:6px;padding:6px;text-align:center;'>"
                f"<div style='font-family:Space Mono,monospace;font-size:0.75rem;"
                f"font-weight:700;color:{fc};'>Fib {fname}{label_extra}</div>"
                f"<div style='font-size:0.85rem;color:#e2e8f0;font-weight:700;'>"
                f"{fval:.2f}</div>"
                f"<div style='font-size:0.68rem;color:#64748b;'>{dist}% ecart</div>"
                f"</div>",
                unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.72rem;color:#64748b;margin:4px 0 8px;'>"
                    f"Swing analysé: {fib_lookback} jours | "
                    f"Haut: {fib_high:.2f} | Bas: {fib_low:.2f}</div>",
                    unsafe_allow_html=True)

        # ── Estimation du prix d'entrée optimal ──
        # Basée sur la confluence : Fibonacci + Support/Résistance + MA50 + Bollinger basse
        ma50_val  = float(ma50.iloc[-1])
        bb_low_val= float(bb_low.iloc[-1])

        # Candidats pour le prix d'entrée (niveaux de confluence)
        candidates = []

        # Fibonacci 0.382 — premier support sérieux
        candidates.append({
            "prix":    round(fib_levels["0.382"], 2),
            "label":   "Fib 0.382",
            "raison":  "Premier support Fibonacci",
            "score_c": 3,
            "color":   "#7DF9FF",
        })
        # Fibonacci 0.500 — support psychologique
        candidates.append({
            "prix":    round(fib_levels["0.500"], 2),
            "label":   "Fib 0.500",
            "raison":  "Support psychologique (50%)",
            "score_c": 4,
            "color":   "#fbbf24",
        })
        # Fibonacci 0.618 — golden ratio
        candidates.append({
            "prix":    round(fib_levels["0.618"], 2),
            "label":   "Fib 0.618 (Golden Ratio)",
            "raison":  "Zone d'achat institutionnelle optimale",
            "score_c": 5,
            "color":   "#00ff88",
        })
        # MA50 actuelle
        candidates.append({
            "prix":    round(ma50_val, 2),
            "label":   "MA50",
            "raison":  "Moyenne mobile 50 jours — support dynamique",
            "score_c": 3,
            "color":   "#fbbf24",
        })
        # Bande basse Bollinger
        candidates.append({
            "prix":    round(bb_low_val, 2),
            "label":   "BB Basse",
            "raison":  "Bande basse Bollinger — zone de valeur",
            "score_c": 3,
            "color":   "#a78bfa",
        })
        # Support S1 pivot
        candidates.append({
            "prix":    round(s1, 2),
            "label":   "Support S1",
            "raison":  "Support pivot annuel",
            "score_c": 2,
            "color":   "#00ff8866",
        })

        # Filtrer : seulement les niveaux EN DESSOUS du prix actuel
        # (sauf si prix déjà sous MA50 ou BB basse)
        below = [c for c in candidates if c["prix"] < price_now * 1.01]
        above = [c for c in candidates if c["prix"] >= price_now * 1.01]

        # Trier par score de confluence décroissant
        below.sort(key=lambda x: x["score_c"], reverse=True)

        # Trouver les confluences — niveaux proches (<3%) qui se renforcent
        confluences = []
        used = set()
        for i, c1 in enumerate(below):
            if i in used:
                continue
            conf_group = [c1]
            for j, c2 in enumerate(below):
                if j != i and j not in used:
                    if abs(c1["prix"] - c2["prix"]) / c1["prix"] * 100 < 3.0:
                        conf_group.append(c2)
                        used.add(j)
            used.add(i)
            if conf_group:
                avg_price = round(sum(c["prix"] for c in conf_group) / len(conf_group), 2)
                total_score = sum(c["score_c"] for c in conf_group)
                labels = " + ".join(c["label"] for c in conf_group)
                raisons = " | ".join(c["raison"] for c in conf_group)
                # Couleur selon score de confluence
                if total_score >= 8:   cc = "#00ff88"; cq = "FORTE"
                elif total_score >= 5: cc = "#fbbf24"; cq = "CORRECTE"
                else:                  cc = "#94a3b8"; cq = "FAIBLE"
                confluences.append({
                    "prix":     avg_price,
                    "labels":   labels,
                    "raisons":  raisons,
                    "score":    total_score,
                    "qualite":  cq,
                    "color":    cc,
                    "dist_pct": round((price_now - avg_price) / price_now * 100, 1),
                })

        confluences.sort(key=lambda x: x["score"], reverse=True)

        # ── Déterminer si le prix est DÉJÀ dans une zone optimale ──
        # Vérifier si le prix est sur ou sous un niveau Fibonacci clé
        in_optimal_zone = False
        optimal_zone_label = ""
        optimal_zone_color = "#00ff88"

        for fname, fval in fib_levels.items():
            dist = abs(price_now - fval) / price_now * 100
            if dist < 2.0 and fname in ("0.382","0.500","0.618"):
                in_optimal_zone = True
                optimal_zone_label = fname
                optimal_zone_color = fib_colors.get(fname, "#00ff88")
                break
        # Aussi vérifier si prix SOUS le 0.618
        already_below_618 = price_now < fib_levels["0.618"] * 1.02

        # ── Affichage ──
        st.markdown("#### Zones d'entrée optimales")

        # ── Message zone d'achat active si prix déjà dans zone optimale ──
        if in_optimal_zone or already_below_618:
            macd_confirm = "positif" if macd_now > 0 else "encore négatif — attendre croisement"
            macd_color   = "#00ff88" if macd_now > 0 else "#fbbf24"
            verdict_achat = "ACHETER MAINTENANT" if macd_now > 0 else "ZONE ACTIVE — CONFIRMER MACD"
            badge_color   = "#00ff88" if macd_now > 0 else "#fbbf24"
            fib_zone_txt  = f"Fib {optimal_zone_label}" if in_optimal_zone else "sous Golden Ratio 0.618"

            st.markdown(
                f"<div style='background:#001a0f;border:3px solid {badge_color};"
                f"border-radius:12px;padding:16px 20px;margin:8px 0;'>"
                f"<div style='font-family:Space Mono,monospace;font-size:1.1rem;"
                f"font-weight:700;color:{badge_color};'>"
                f"✅ ZONE D'ACHAT ACTIVE — {fib_zone_txt}</div>"
                f"<div style='font-size:0.9rem;color:#e2e8f0;margin-top:6px;'>"
                f"Le prix actuel <strong>{price_now:.2f}</strong> est dans la zone optimale. "
                f"Les institutionnels achètent à ce niveau.</div>"
                f"<div style='margin-top:10px;background:{macd_color}22;"
                f"border:1px solid {macd_color}44;border-radius:8px;padding:8px 12px;'>"
                f"<span style='color:{macd_color};font-weight:700;'>MACD: {macd_confirm}</span>"
                f"</div>"
                f"<div style='margin-top:8px;font-size:0.82rem;color:#94a3b8;'>"
                f"{'Entrer maintenant — toutes conditions réunies.' if macd_now > 0 else 'Placer une alerte — entrer dès que le MACD croise au-dessus de zéro.'}"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True)

        if not confluences:
            st.markdown(
                f"<div style='background:#001a0f;border:2px solid #00ff8866;"
                f"border-radius:10px;padding:14px 18px;margin:6px 0;'>"
                f"<span style='color:#00ff88;font-family:Space Mono,monospace;"
                f"font-weight:700;'>PRIX EN ZONE DE VALEUR PROFONDE</span><br>"
                f"<span style='color:#94a3b8;font-size:0.85rem;'>"
                f"Le prix ({price_now:.2f}) est sous les niveaux Fibonacci clés. "
                f"Attendre confirmation MACD avant d'entrer.</span>"
                f"</div>",
                unsafe_allow_html=True)
        else:
            # Afficher les 3 meilleures zones
            for conf in confluences[:3]:
                st.markdown(
                    f"<div style='background:#0d1a2a;border:2px solid {conf['color']}55;"
                    f"border-left:4px solid {conf['color']};border-radius:10px;"
                    f"padding:12px 16px;margin:6px 0;'>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"align-items:center;'>"
                    f"<span style='font-family:Space Mono,monospace;font-size:1.2rem;"
                    f"font-weight:700;color:{conf['color']};'>{conf['prix']}</span>"
                    f"<span style='background:{conf['color']}22;border:1px solid "
                    f"{conf['color']}44;border-radius:6px;padding:2px 10px;"
                    f"font-size:0.78rem;color:{conf['color']};font-weight:700;'>"
                    f"Confluence {conf['qualite']}</span>"
                    f"</div>"
                    f"<div style='color:#94a3b8;font-size:0.82rem;margin-top:4px;'>"
                    f"{conf['labels']}</div>"
                    f"<div style='color:#64748b;font-size:0.75rem;margin-top:2px;'>"
                    f"{conf['raisons']}</div>"
                    f"<div style='color:#64748b;font-size:0.72rem;margin-top:4px;'>"
                    f"Recul necessaire: -{conf['dist_pct']:.1f}% depuis {price_now:.2f}"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True)

            # Résumé — meilleure entrée
            best = confluences[0]
            st.markdown(
                f"<div style='background:#002a1a;border:1px solid #00ff8844;"
                f"border-radius:8px;padding:10px 14px;margin:6px 0;"
                f"font-size:0.82rem;'>"
                f"<strong style='color:#00ff88;'>Meilleure entrée suggérée: "
                f"{best['prix']}</strong>"
                f" | Recul de -{best['dist_pct']:.1f}% | "
                f"Confluence: {best['labels']}"
                f"<br><span style='color:#64748b;font-size:0.72rem;'>"
                f"Attendre que le prix atteigne cette zone ET que le MACD "
                f"redevienne positif avant d'entrer.</span>"
                f"</div>",
                unsafe_allow_html=True)
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.55, 0.15, 0.15, 0.15],
        )

        # 1. Chandelier japonais
        fig.add_trace(go.Candlestick(
            x=dates, open=open_, high=high, low=low, close=close,
            name="Prix",
            increasing_line_color="#00ff88",
            decreasing_line_color="#f87171",
            increasing_fillcolor="#00ff8844",
            decreasing_fillcolor="#f8717144",
        ), row=1, col=1)

        # Bandes de Bollinger
        fig.add_trace(go.Scatter(x=dates, y=bb_up, name="BB+", line=dict(color="#a78bfa44", width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=bb_low, name="BB-", line=dict(color="#a78bfa44", width=1),
                                  fill="tonexty", fillcolor="rgba(167,139,250,0.05)", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=ma20, name="MA20", line=dict(color="#a78bfa", width=1, dash="dot")), row=1, col=1)

        # MA50 et MA200
        fig.add_trace(go.Scatter(x=dates, y=ma50,  name="MA50",  line=dict(color="#fbbf24", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=ma200, name="MA200", line=dict(color="#f87171",  width=1.5)), row=1, col=1)

        # Supports et résistances
        for level, label, color in [
            (s1, f"S1 {s1:.0f}", "#00ff8888"),
            (s2, f"S2 {s2:.0f}", "#00ff8844"),
            (r1, f"R1 {r1:.0f}", "#f8717188"),
            (r2, f"R2 {r2:.0f}", "#f8717144"),
            (pivot, f"Pivot {pivot:.0f}", "#fbbf2466"),
        ]:
            fig.add_shape(type="line", x0=dates[0], x1=dates[-1],
                          y0=level, y1=level,
                          line=dict(color=color, width=1, dash="dash"),
                          row=1, col=1)
            fig.add_annotation(x=dates[-1], y=level, text=label,
                               showarrow=False, xanchor="left",
                               font=dict(color=color, size=9),
                               row=1, col=1)

        # Niveaux Fibonacci
        for fib_name, fib_val in fib_levels.items():
            fc = fib_colors.get(fib_name, "#64748b")
            # Ligne plus épaisse pour les niveaux clés
            fw = 2 if fib_name in ("0.382","0.500","0.618") else 1
            fd = "solid" if fib_name in ("0.500","0.618") else "dot"
            fig.add_shape(type="line", x0=dates[0], x1=dates[-1],
                          y0=fib_val, y1=fib_val,
                          line=dict(color=fc+"99", width=fw, dash=fd),
                          row=1, col=1)
            fig.add_annotation(
                x=dates[0], y=fib_val,
                text=f"Fib {fib_name}",
                showarrow=False, xanchor="right",
                font=dict(color=fc, size=8),
                row=1, col=1)

        # 2. Volume
        vol_colors = ["#00ff8866" if c >= o else "#f8717166"
                      for c, o in zip(close, open_)]
        fig.add_trace(go.Bar(x=dates, y=volume, name="Volume",
                              marker_color=vol_colors, showlegend=False), row=2, col=1)
        vol_ma = volume.rolling(20).mean()
        fig.add_trace(go.Scatter(x=dates, y=vol_ma, name="Vol MA20",
                                  line=dict(color="#fbbf24", width=1), showlegend=False), row=2, col=1)

        # 3. RSI
        fig.add_trace(go.Scatter(x=dates, y=rsi, name="RSI",
                                  line=dict(color="#7DF9FF", width=1.5)), row=3, col=1)
        # Zone idéale RSI
        fig.add_trace(go.Scatter(x=dates, y=[70]*len(dates), name="RSI 70",
                                  line=dict(color="#f8717166", width=0.8, dash="dash"),
                                  showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=dates, y=[30]*len(dates), name="RSI 30",
                                  line=dict(color="#00ff8866", width=0.8, dash="dash"),
                                  showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=dates, y=[50]*len(dates), name="RSI 50",
                                  line=dict(color="#64748b44", width=0.5, dash="dot"),
                                  showlegend=False), row=3, col=1)

        # 4. MACD histogramme
        hist_colors = ["#00ff8888" if v >= 0 else "#f8717188" for v in macd_hist]
        fig.add_trace(go.Bar(x=dates, y=macd_hist, name="MACD Hist",
                              marker_color=hist_colors, showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=dates, y=macd_line,   name="MACD",
                                  line=dict(color="#00ff88", width=1.2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=dates, y=signal_line, name="Signal",
                                  line=dict(color="#f97316", width=1.2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=dates, y=[0]*len(dates), name="MACD 0",
                                  line=dict(color="#64748b44", width=0.5, dash="dash"),
                                  showlegend=False), row=4, col=1)

        fig.update_layout(
            height=700,
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0d1117",
            font=dict(color="#e2e8f0", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=10)),
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=60, t=30, b=0),
            yaxis=dict(title="Prix ($)", gridcolor="#1e2a3a"),
            yaxis2=dict(title="Volume",  gridcolor="#1e2a3a"),
            yaxis3=dict(title="RSI",     gridcolor="#1e2a3a", range=[0,100]),
            yaxis4=dict(title="MACD",    gridcolor="#1e2a3a"),
        )
        for i in range(1, 5):
            fig.update_xaxes(gridcolor="#1e2a3a", row=i, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # ── Affichage des patterns détectés ──
        if patterns:
            st.markdown("#### 📐 Patterns Chartistes Détectés")
            for p in patterns:
                bg  = "#001a0f" if p["type"] == "HAUSSIER" else "#1a0000"
                bdr = p["color"]
                st.markdown(f"""<div style='background:{bg};border:1px solid {bdr}55;
                    border-left:4px solid {bdr};border-radius:8px;
                    padding:10px 16px;margin:6px 0;
                    display:flex;align-items:center;gap:12px;'>
                    <span style='font-size:1.4rem;min-width:30px;text-align:center;'>
                    {p["icon"]}</span>
                    <div>
                        <span style='font-family:Space Mono,monospace;font-weight:700;
                        color:{bdr};font-size:0.9rem;'>{p["name"]}</span>
                        <span style='background:{bdr}22;border:1px solid {bdr}44;
                        border-radius:4px;padding:1px 8px;font-size:0.72rem;
                        color:{bdr};margin-left:8px;'>{p["type"]}</span>
                        <span style='background:#1e2a3a;border-radius:4px;
                        padding:1px 8px;font-size:0.72rem;color:#94a3b8;margin-left:4px;'>
                        {p["strength"]}</span>
                        <div style='color:#94a3b8;font-size:0.8rem;margin-top:3px;'>
                        {p["desc"]}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style='background:#0d1117;border:1px solid #1e3a5f;
                border-radius:8px;padding:10px 16px;margin:6px 0;
                color:#64748b;font-size:0.82rem;'>
                📐 Aucun pattern chartiste clair détecté sur les 60 derniers jours
            </div>""", unsafe_allow_html=True)

    except ImportError:
        st.info("Installez plotly pour les graphiques d'entrée.")
    except Exception as e:
        st.warning(f"Graphique indisponible pour {ticker}: {str(e)[:60]}")


# ─────────────────────────────
# 🏭 ANALYSE SECTORIELLE (cache 1h)
# ─────────────────────────────
SECTOR_ETFS = {
    "Technology":            "XLK",
    "Healthcare":            "XLV",
    "Financial Services":    "XLF",
    "Consumer Cyclical":     "XLY",
    "Consumer Defensive":    "XLP",
    "Industrials":           "XLI",
    "Energy":                "XLE",
    "Basic Materials":       "XLB",
    "Real Estate":           "XLRE",
    "Utilities":             "XLU",
    "Communication Services":"XLC",
}

@st.cache_data(ttl=3600)
def get_sector_strength():
    """Analyse la force relative de chaque secteur via les ETFs SPDR."""
    results = {}
    try:
        etf_tickers = list(SECTOR_ETFS.values())
        raw = yf.download(etf_tickers, period="6mo",
                          auto_adjust=True, progress=False,
                          group_by="ticker", threads=True)
        for sector, etf in SECTOR_ETFS.items():
            try:
                if len(etf_tickers) == 1:
                    closes = raw["Close"]
                else:
                    closes = raw[etf]["Close"]
                closes = closes.dropna()
                if len(closes) < 20:
                    continue
                price    = float(closes.iloc[-1])
                ma20     = float(closes.rolling(20).mean().iloc[-1])
                ma50     = float(closes.rolling(min(50,len(closes))).mean().iloc[-1])
                perf_1m  = round((price - float(closes.iloc[-21])) / float(closes.iloc[-21]) * 100, 2) if len(closes)>=21 else 0
                perf_3m  = round((price - float(closes.iloc[-63])) / float(closes.iloc[-63]) * 100, 2) if len(closes)>=63 else 0
                perf_6m  = round((price - float(closes.iloc[0]))   / float(closes.iloc[0])   * 100, 2)
                above_ma20 = price > ma20
                above_ma50 = price > ma50
                # Score de force sectorielle /10
                score = 0
                if perf_1m > 3:   score += 3
                elif perf_1m > 1: score += 2
                elif perf_1m > 0: score += 1
                if perf_3m > 8:   score += 3
                elif perf_3m > 4: score += 2
                elif perf_3m > 0: score += 1
                if above_ma20: score += 2
                if above_ma50: score += 2
                results[sector] = {
                    "etf": etf, "price": round(price,2),
                    "perf_1m": perf_1m, "perf_3m": perf_3m, "perf_6m": perf_6m,
                    "above_ma20": above_ma20, "above_ma50": above_ma50,
                    "score": score,
                }
            except Exception:
                continue
    except Exception:
        pass
    # Trier par score décroissant
    ranked = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
    top3    = [s for s,_ in ranked[:3]]
    bottom3 = [s for s,_ in ranked[-3:]]
    return {"sectors": results, "ranked": ranked, "top3": top3, "bottom3": bottom3}


def sector_bonus(sector, sector_data):
    """Retourne un bonus/malus selon la force du secteur (+5 à -5 pts)."""
    if not sector_data or "sectors" not in sector_data:
        return 0, "~ Secteur N/D"
    s = sector_data["sectors"].get(sector)
    if not s:
        return 0, "~ Secteur N/D"
    score = s["score"]
    if score >= 8:   return  5, f"Secteur {sector} tres fort (+5pts)"
    elif score >= 6: return  3, f"Secteur {sector} fort (+3pts)"
    elif score >= 4: return  0, f"Secteur {sector} neutre"
    elif score >= 2: return -2, f"Secteur {sector} faible (-2pts)"
    else:            return -5, f"Secteur {sector} tres faible (-5pts)"


# ─────────────────────────────
# 📅 FONDAMENTAUX 5 ANS (via financials annuels)
# ─────────────────────────────
def get_5y_fundamentals(t_obj):
    """
    Calcule les tendances sur 5 ans depuis les états financiers annuels.
    Retourne: rev_cagr, eps_cagr, margin_trend_5y, roe_trend_5y, fcf_trend
    """
    result = {
        "rev_cagr_5y": None, "eps_cagr_5y": None,
        "margin_5y_trend": None, "roe_5y_trend": None,
        "rev_consistency": None, "fcf_5y_positive": None,
        "details": []
    }
    try:
        fin = t_obj.financials          # revenus, bénéfices annuels
        bs  = t_obj.balance_sheet       # capitaux propres
        cf  = t_obj.cashflow            # flux de trésorerie

        # ── Croissance revenus CAGR 5 ans ──
        if fin is not None and not fin.empty:
            rev_row = None
            for name in ["Total Revenue","Revenue","Net Revenue"]:
                if name in fin.index:
                    rev_row = fin.loc[name].dropna().sort_index()
                    break
            if rev_row is not None and len(rev_row) >= 2:
                rev_vals = rev_row.values[::-1]  # du plus ancien au plus récent
                n = min(len(rev_vals)-1, 4)       # max 4 ans (5 points)
                if rev_vals[0] > 0 and rev_vals[-1] > 0 and n > 0:
                    cagr = ((rev_vals[-1]/rev_vals[0])**(1/n) - 1) * 100
                    result["rev_cagr_5y"] = round(cagr, 1)
                # Consistance : combien d'années avec croissance positive
                growths = [(rev_vals[i]-rev_vals[i-1])/abs(rev_vals[i-1])*100
                           for i in range(1,len(rev_vals)) if rev_vals[i-1]!=0]
                pos = sum(1 for g in growths if g > 0)
                result["rev_consistency"] = f"{pos}/{len(growths)} ans en hausse"

            # ── EPS CAGR 5 ans ──
            eps_row = None
            for name in ["Basic EPS","Diluted EPS","EPS"]:
                if name in fin.index:
                    eps_row = fin.loc[name].dropna().sort_index()
                    break
            if eps_row is None:
                # Calculer depuis Net Income / shares
                ni_row = None
                for name in ["Net Income","Net Income Common Stockholders"]:
                    if name in fin.index:
                        ni_row = fin.loc[name].dropna().sort_index()
                        break
                if ni_row is not None and len(ni_row) >= 2:
                    ni_vals = ni_row.values[::-1]
                    n = min(len(ni_vals)-1, 4)
                    if ni_vals[0] != 0 and n > 0 and ni_vals[0] > 0 and ni_vals[-1] > 0:
                        cagr = ((ni_vals[-1]/ni_vals[0])**(1/n) - 1) * 100
                        result["eps_cagr_5y"] = round(min(cagr, 150.0), 1)
            else:
                eps_vals = eps_row.values[::-1]
                n = min(len(eps_vals)-1, 4)
                if len(eps_vals)>=2 and eps_vals[0]>0 and eps_vals[-1]>0 and n>0:
                    cagr = ((eps_vals[-1]/eps_vals[0])**(1/n) - 1) * 100
                    result["eps_cagr_5y"] = round(min(cagr, 150.0), 1)

            # ── Tendance marge nette 5 ans ──
            ni_row2 = None
            for name in ["Net Income","Net Income Common Stockholders"]:
                if name in fin.index:
                    ni_row2 = fin.loc[name].dropna().sort_index()
                    break
            if ni_row2 is not None and rev_row is not None:
                ni_v  = ni_row2.sort_index().values[::-1]
                rev_v = rev_row.sort_index().values[::-1]
                n = min(len(ni_v), len(rev_v))
                if n >= 2:
                    m_recent = ni_v[0]/rev_v[0]*100  if rev_v[0]!=0 else None
                    m_old    = ni_v[-1]/rev_v[-1]*100 if rev_v[-1]!=0 else None
                    if m_recent and m_old:
                        result["margin_5y_trend"] = round(m_recent - m_old, 1)

        # ── FCF positif sur 5 ans ──
        if cf is not None and not cf.empty:
            fcf_row = None
            for name in ["Free Cash Flow","Capital Expenditure"]:
                if name in cf.index:
                    fcf_row = cf.loc[name].dropna()
                    break
            if fcf_row is not None:
                fcf_vals = fcf_row.values
                pos_fcf = sum(1 for v in fcf_vals if v > 0)
                result["fcf_5y_positive"] = f"{pos_fcf}/{len(fcf_vals)} ans FCF positif"

        # ── ROE trend 5 ans via capitaux propres ──
        if bs is not None and not bs.empty and fin is not None and not fin.empty:
            eq_row = None
            for name in ["Stockholders Equity","Total Stockholders Equity","Common Stock Equity"]:
                if name in bs.index:
                    eq_row = bs.loc[name].dropna().sort_index()
                    break
            ni_row3 = None
            for name in ["Net Income","Net Income Common Stockholders"]:
                if name in fin.index:
                    ni_row3 = fin.loc[name].dropna().sort_index()
                    break
            if eq_row is not None and ni_row3 is not None:
                n = min(len(eq_row), len(ni_row3))
                if n >= 2:
                    roe_vals = []
                    for j in range(n):
                        eq  = float(eq_row.iloc[j])
                        ni  = float(ni_row3.iloc[j])
                        if eq > 0:
                            roe_vals.append(ni/eq*100)
                    if len(roe_vals) >= 2:
                        result["roe_5y_trend"] = round(roe_vals[0] - roe_vals[-1], 1)

    except Exception:
        pass
    return result

@st.cache_data(ttl=86400)
def get_ticker_fundamentals(ticker):
    """
    Télécharge TOUS les fondamentaux d'un ticker en un seul bloc.
    Mis en cache 24h — évite les appels répétés à Yahoo Finance.
    """
    import time
    for attempt in range(4):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if not info or len(info) <= 5:
                time.sleep(1.0 * (attempt + 1))
                continue
            # Récupérer tout en une seule session
            try:
                fin  = t.financials
                bs   = t.balance_sheet
                cf   = t.cashflow
                qfin = t.quarterly_financials
            except Exception:
                fin = bs = cf = qfin = None
            return {"info": info, "fin": fin, "bs": bs, "cf": cf, "qfin": qfin}
        except Exception:
            if attempt < 3:
                time.sleep(1.0 * (attempt + 1))
    return None


def get_5y_fundamentals_from_cache(fund_data):
    """Calcule les fondamentaux 5 ans depuis le cache — aucun appel réseau."""
    result = {
        "rev_cagr_5y": None, "eps_cagr_5y": None,
        "margin_5y_trend": None, "roe_5y_trend": None,
        "rev_consistency": None, "fcf_5y_positive": None,
    }
    if not fund_data: return result
    try:
        fin = fund_data.get("fin")
        bs  = fund_data.get("bs")
        cf  = fund_data.get("cf")
        if fin is not None and not fin.empty:
            rev_row = None
            for name in ["Total Revenue","Revenue","Net Revenue"]:
                if name in fin.index:
                    rev_row = fin.loc[name].dropna().sort_index(); break
            if rev_row is not None and len(rev_row) >= 2:
                rv = rev_row.values[::-1]
                n  = min(len(rv)-1, 4)
                if rv[0] > 0 and rv[-1] > 0 and n > 0:
                    result["rev_cagr_5y"] = round(((rv[-1]/rv[0])**(1/n)-1)*100, 1)
                growths = [(rv[i]-rv[i-1])/abs(rv[i-1])*100 for i in range(1,len(rv)) if rv[i-1]!=0]
                result["rev_consistency"] = f"{sum(1 for g in growths if g>0)}/{len(growths)} ans en hausse"
            ni_row = None
            for name in ["Net Income","Net Income Common Stockholders"]:
                if name in fin.index:
                    ni_row = fin.loc[name].dropna().sort_index(); break
            if ni_row is not None and len(ni_row) >= 2:
                nv = ni_row.values[::-1]
                n  = min(len(nv)-1, 4)
                if nv[0] > 0 and nv[-1] > 0 and n > 0:
                    result["eps_cagr_5y"] = round(min(((nv[-1]/nv[0])**(1/n)-1)*100, 150.0), 1)
            if ni_row is not None and rev_row is not None:
                nv2 = ni_row.sort_index().values[::-1]
                rv2 = rev_row.sort_index().values[::-1]
                nm  = min(len(nv2), len(rv2))
                if nm >= 2 and rv2[0] != 0 and rv2[-1] != 0:
                    result["margin_5y_trend"] = round(nv2[0]/rv2[0]*100 - nv2[-1]/rv2[-1]*100, 1)
        if cf is not None and not cf.empty:
            for name in ["Free Cash Flow"]:
                if name in cf.index:
                    fcf_row = cf.loc[name].dropna()
                    result["fcf_5y_positive"] = f"{sum(1 for v in fcf_row.values if v>0)}/{len(fcf_row)} ans FCF positif"
                    break
        if bs is not None and not bs.empty and fin is not None and not fin.empty:
            eq_row = None
            for name in ["Stockholders Equity","Total Stockholders Equity","Common Stock Equity"]:
                if name in bs.index:
                    eq_row = bs.loc[name].dropna().sort_index(); break
            ni_r2 = None
            for name in ["Net Income","Net Income Common Stockholders"]:
                if name in fin.index:
                    ni_r2 = fin.loc[name].dropna().sort_index(); break
            if eq_row is not None and ni_r2 is not None:
                nm2 = min(len(eq_row), len(ni_r2))
                if nm2 >= 2:
                    roe_v = [float(ni_r2.iloc[j])/float(eq_row.iloc[j])*100
                             for j in range(nm2) if float(eq_row.iloc[j]) > 0]
                    if len(roe_v) >= 2:
                        result["roe_5y_trend"] = round(roe_v[0] - roe_v[-1], 1)
    except Exception: pass
    return result


# Qualité 25 | Croissance 20 | Valorisation 15 | Momentum 10 | Conviction 10 | Institutionnel 20
# ─────────────────────────────
def _rsi(s,p=14):
    d=s.diff()
    g=d.where(d>0,0).rolling(p).mean()
    l=-d.where(d<0,0).rolling(p).mean()
    return float(100-(100/(1+g/l.clip(lower=1e-10))).iloc[-1])

def score_ticker(ticker, pre_data, market_bonus=0, spy_hist=None, sector_data=None, fund_data=None):
    try:
        df=pre_data["df"]; price=pre_data["price"]; ma200=pre_data["ma200"]
        close=df["Close"]
        ma50=float(close.rolling(min(50,len(close))).mean().iloc[-1])
        rsi_val=_rsi(close)
        hi52=float(close.rolling(min(252,len(close))).max().iloc[-1])
        pct_hi=round((price-hi52)/hi52*100,1)

        import time
        # Utiliser les fondamentaux pré-chargés si disponibles
        if fund_data is None or not fund_data.get("info"):
            return None  # Pas de données — ignorer ce titre

        info = fund_data["info"]
        if not info or len(info) <= 5: return None

        # ── Données fondamentales ──
        cap_b      =(info.get("marketCap",0) or 0)/1e9
        roe        =info.get("returnOnEquity",None)
        gross_margin=info.get("grossMargins",None)
        net_margin =info.get("profitMargins",None)
        fcf        =info.get("freeCashflow",None)
        debt_eq    =info.get("debtToEquity",None)
        rev_growth =info.get("revenueGrowth",None)
        eps_growth =info.get("earningsGrowth",None)
        eps_qtr    =info.get("earningsQuarterlyGrowth",None)
        pe         =info.get("trailingPE",None)
        peg        =info.get("pegRatio",None)
        ev_ebitda  =info.get("enterpriseToEbitda",None)
        sector     =info.get("sector","N/A")
        name       =info.get("shortName",ticker)
        chg        =info.get("regularMarketChangePercent",0.0) or 0.0
        # Conviction
        insider_pct=info.get("heldPercentInsiders",None)
        target_mean=info.get("targetMeanPrice",None)
        nb_analysts=info.get("numberOfAnalystOpinions",None)
        recommend  =info.get("recommendationMean",None)
        # Signaux institutionnels
        inst_own   =info.get("heldPercentInstitutions",None)
        short_pct  =info.get("shortPercentOfFloat",None)
        beta       =info.get("beta",None)
        ps_ratio   =info.get("priceToSalesTrailing12Months",None)
        current_ratio=info.get("currentRatio",None)
        total_assets =info.get("totalAssets",None)
        total_equity =info.get("totalStockholderEquity",None)
        ebit         =info.get("ebit",None)
        # ROIC = EBIT(1-tax) / (Equity + Debt) — approximation
        tax_rate     =info.get("effectiveTaxRate",None)
        total_debt   =info.get("totalDebt",None)
        # Earnings surprises
        eps_actual   =info.get("earningsPerShare",None)   # dernier trimestre réel
        eps_estimate =info.get("epsCurrentYear",None)     # estimation annuelle

        # Valeurs dérivées de base
        gm=gross_margin*100 if gross_margin else None
        nm=net_margin*100   if net_margin   else None
        rp=roe*100          if roe          else None
        rg=rev_growth*100   if rev_growth   else None
        eg=eps_growth*100   if eps_growth   else None
        eqg=eps_qtr*100     if eps_qtr      else None
        ip=insider_pct*100  if insider_pct  else None
        de=debt_eq/100      if debt_eq is not None else None
        # Valeurs dérivées institutionnelles
        inst_pct=inst_own*100 if inst_own else None
        short_float=short_pct*100 if short_pct else None
        # ROIC calculé : EBIT*(1-tax) / (equity + debt)
        roic=None
        try:
            if ebit and total_equity and total_equity>0:
                tax=tax_rate if tax_rate else 0.21
                invested=total_equity+(total_debt or 0)
                if invested>0: roic=round(ebit*(1-tax)/invested*100,1)
        except Exception: pass
        # Momentum relatif 52W vs SPY — utilise spy_hist pré-chargé (pas de requête réseau)
        rel_momentum=None
        try:
            if spy_hist is not None and not spy_hist.empty and len(df)>=50:
                n_days = min(252, len(df)-1, len(spy_hist)-1)
                price_1y  = float(df["Close"].iloc[-n_days])
                spy_1y    = float(spy_hist["Close"].iloc[-n_days])
                stock_perf = (price - price_1y) / price_1y * 100
                spy_perf   = (float(spy_hist["Close"].iloc[-1]) - spy_1y) / spy_1y * 100
                rel_momentum = round(stock_perf - spy_perf, 1)
        except Exception: pass

        # Tendance marge nette trimestrielle — depuis cache
        margin_trend = None
        try:
            qfin = fund_data.get("qfin")
            if qfin is not None and not qfin.empty:
                if "Net Income" in qfin.index and "Total Revenue" in qfin.index:
                    ni  = qfin.loc["Net Income"].dropna()
                    rev = qfin.loc["Total Revenue"].dropna()
                    if len(ni) >= 2 and len(rev) >= 2:
                        m_recent = float(ni.iloc[0]) / float(rev.iloc[0]) * 100
                        m_older  = float(ni.iloc[1]) / float(rev.iloc[1]) * 100
                        margin_trend = round(m_recent - m_older, 1)
        except Exception: pass

        # Fondamentaux 5 ans — depuis cache (pas de nouveaux appels réseau)
        f5 = get_5y_fundamentals_from_cache(fund_data)
        rev_cagr_5y    = f5["rev_cagr_5y"]
        eps_cagr_5y    = f5["eps_cagr_5y"]
        margin_5y      = f5["margin_5y_trend"]
        roe_5y_trend   = f5["roe_5y_trend"]
        rev_consistency= f5["rev_consistency"]
        fcf_5y         = f5["fcf_5y_positive"]

        # ── Bonus sectoriel ──
        sec_bonus, sec_label = sector_bonus(sector, sector_data)

        # Filtre dur minimal — seulement capitalisation trop faible
        if cap_b > 0 and cap_b < 0.5: return None

        # Plafond EPS growth pour éviter les distorsions (ex: passage de négatif à positif)
        if eg is not None: eg = min(eg, 150.0)
        if eqg is not None: eqg = min(eqg, 200.0)

        # ══ 🏆 QUALITÉ (25 pts) — ROE + Marge brute + Marge nette + FCF + Dette ══
        q=0; qd=[]

        # ROE contextualisé par secteur (6 pts)
        # Seuils différents : banques/assurances ont naturellement un ROE plus élevé
        roe_high = {"Financial Services":18,"Real Estate":10,"Utilities":10,"Energy":12}.get(sector,15)
        roe_mid  = {"Financial Services":12,"Real Estate":7, "Utilities":7, "Energy":8 }.get(sector,10)
        if rp is not None:
            if rp>roe_high+5: q+=6; qd.append(f"✅ ROE exceptionnel ({rp:.1f}%) vs secteur {sector}")
            elif rp>roe_high: q+=5; qd.append(f"✅ ROE solide ({rp:.1f}%) vs secteur {sector}")
            elif rp>roe_mid:  q+=3; qd.append(f"~ ROE correct ({rp:.1f}%) vs secteur {sector}")
            elif rp>0:        q+=1; qd.append(f"~ ROE faible ({rp:.1f}%)")
            else:             q+=0; qd.append(f"❌ ROE négatif ({rp:.1f}%)")
        else: q+=2; qd.append("~ ROE N/D")

        # Marge brute — pricing power (5 pts)
        if gm is not None:
            if gm>50:   q+=5; qd.append(f"✅ Marge brute {gm:.1f}% — pricing power fort")
            elif gm>35: q+=3; qd.append(f"✅ Marge brute {gm:.1f}% — solide")
            elif gm>20: q+=1; qd.append(f"~ Marge brute {gm:.1f}%")
            else:       q+=0; qd.append(f"❌ Marge brute faible ({gm:.1f}%)")
        else: q+=2; qd.append("~ Marge brute N/D")

        # Marge nette (5 pts)
        if nm is not None:
            if nm>20:   q+=5; qd.append(f"✅ Marge nette {nm:.1f}% — excellente")
            elif nm>15: q+=4; qd.append(f"✅ Marge nette {nm:.1f}% — forte")
            elif nm>10: q+=3; qd.append(f"✅ Marge nette {nm:.1f}% — correcte")
            elif nm>5:  q+=1; qd.append(f"~ Marge nette {nm:.1f}% — faible")
            else:       q+=0; qd.append(f"❌ Marge nette {nm:.1f}% — très faible")
        else: q+=2; qd.append("~ Marge nette N/D")

        # FCF (4 pts) — scoré, pas filtré
        if fcf is not None:
            if fcf > 0:
                fb=fcf/1e9
                if fb>5:   q+=4; qd.append(f"✅ FCF {fb:.1f}G$ — très fort")
                elif fb>1: q+=3; qd.append(f"✅ FCF {fb:.2f}G$ — solide")
                else:      q+=2; qd.append(f"~ FCF {fb:.3f}G$ — positif")
            else:
                fb=fcf/1e9
                q+=0; qd.append(f"❌ FCF {fb:.2f}G$ — négatif (investissement massif ou perte)")
        else: q+=1; qd.append("~ FCF N/D")

        # Dette (5 pts)
        if de is not None:
            if de<0.3:   q+=5; qd.append(f"✅ Dette très faible (D/E={de:.2f})")
            elif de<0.75:q+=4; qd.append(f"✅ Dette raisonnable (D/E={de:.2f})")
            elif de<1.5: q+=2; qd.append(f"~ Dette modérée (D/E={de:.2f})")
            else:        q+=0; qd.append(f"❌ Dette élevée (D/E={de:.2f})")
        else: q+=2; qd.append("~ D/E N/D")

        # ══ CROISSANCE (20 pts) — 5Y CAGR prioritaire, TTM en fallback ══
        g_s=0; gd=[]

        # Revenus — CAGR 5 ans (10 pts)
        rev_g = rev_cagr_5y if rev_cagr_5y is not None else rg
        rev_label = "CAGR 5 ans" if rev_cagr_5y is not None else "TTM"
        if rev_g is not None:
            if rev_g>15:   g_s+=10; gd.append(f"✅ Revenus +{rev_g:.1f}%/an ({rev_label}) — excellent")
            elif rev_g>10: g_s+=8;  gd.append(f"✅ Revenus +{rev_g:.1f}%/an ({rev_label}) — fort")
            elif rev_g>8:  g_s+=5;  gd.append(f"✅ Revenus +{rev_g:.1f}%/an ({rev_label}) — correct")
            elif rev_g>5:  g_s+=2;  gd.append(f"~ Revenus +{rev_g:.1f}%/an ({rev_label}) — faible")
            else:          g_s+=0;  gd.append(f"❌ Revenus {rev_g:.1f}%/an ({rev_label}) — stagnation")
        else: g_s+=2; gd.append("~ Croissance revenus N/D")

        # Bonus consistance revenus
        if rev_consistency:
            gd.append(f"~ Consistance: {rev_consistency}")

        # EPS — CAGR 5 ans (8 pts)
        eps_g = eps_cagr_5y if eps_cagr_5y is not None else eg
        eps_label = "CAGR 5 ans" if eps_cagr_5y is not None else "TTM"
        if eps_g is not None:
            if eps_g>15:   g_s+=8; gd.append(f"✅ EPS +{eps_g:.1f}%/an ({eps_label}) — excellent")
            elif eps_g>12: g_s+=6; gd.append(f"✅ EPS +{eps_g:.1f}%/an ({eps_label}) — fort")
            elif eps_g>8:  g_s+=4; gd.append(f"✅ EPS +{eps_g:.1f}%/an ({eps_label}) — correct")
            elif eps_g>0:  g_s+=2; gd.append(f"~ EPS +{eps_g:.1f}%/an ({eps_label}) — lent")
            else:          g_s+=0; gd.append(f"❌ EPS {eps_g:.1f}%/an ({eps_label}) — déclin")
        else: g_s+=2; gd.append("~ EPS N/D")

        # FCF positif sur 5 ans (2 pts)
        if fcf_5y:
            parts = fcf_5y.split("/")
            pos_years = int(parts[0]) if parts[0].isdigit() else 0
            total_years = int(parts[1].split()[0]) if len(parts)>1 else 0
            if total_years>0 and pos_years==total_years:
                g_s+=2; gd.append(f"✅ FCF positif {fcf_5y} — très fiable")
            elif total_years>0 and pos_years>=total_years*0.75:
                g_s+=1; gd.append(f"~ FCF positif {fcf_5y}")
            else:
                g_s+=0; gd.append(f"❌ FCF {fcf_5y} — irrégulier")

        # ══ 💲 VALORISATION (15 pts) — P/E + PEG + EV/EBITDA ══
        v=0; vd=[]

        # P/E (6 pts)
        if pe and pe>0:
            if pe<15:   v+=6; vd.append(f"✅ P/E très bas ({pe:.1f})")
            elif pe<25: v+=4; vd.append(f"✅ P/E raisonnable ({pe:.1f})")
            elif pe<35: v+=2; vd.append(f"~ P/E élevé ({pe:.1f})")
            else:       v+=0; vd.append(f"❌ P/E très élevé ({pe:.1f})")
        else: v+=2; vd.append("~ P/E N/D")

        # PEG (5 pts)
        if peg and peg>0:
            if peg<1.0:  v+=5; vd.append(f"✅ PEG excellent ({peg:.2f})")
            elif peg<1.5:v+=3; vd.append(f"✅ PEG bon ({peg:.2f})")
            elif peg<2.0:v+=1; vd.append(f"~ PEG acceptable ({peg:.2f})")
            else:        v+=0; vd.append(f"❌ PEG élevé ({peg:.2f})")
        else: v+=2; vd.append("~ PEG N/D")

        # EV/EBITDA (4 pts)
        if ev_ebitda and ev_ebitda>0:
            if ev_ebitda<12:  v+=4; vd.append(f"✅ EV/EBITDA très bas ({ev_ebitda:.1f})")
            elif ev_ebitda<18:v+=3; vd.append(f"✅ EV/EBITDA raisonnable ({ev_ebitda:.1f})")
            elif ev_ebitda<25:v+=1; vd.append(f"~ EV/EBITDA élevé ({ev_ebitda:.1f})")
            else:             v+=0; vd.append(f"❌ EV/EBITDA très élevé ({ev_ebitda:.1f})")
        else: v+=1; vd.append("~ EV/EBITDA N/D")

        # ══ 🚀 MOMENTUM (10 pts) — MA + RSI + 52W ══
        m=0; md=[]
        if ma50>ma200: m+=3; md.append("✅ MA50 > MA200 — tendance haussière")
        else:               md.append("❌ MA50 < MA200 — tendance faible")
        if price>ma50: m+=3; md.append(f"✅ Prix > MA50 ({ma50:.2f})")
        else:               md.append(f"❌ Prix < MA50 ({ma50:.2f})")
        if 45<=rsi_val<=70:   m+=3; md.append(f"✅ RSI idéal ({rsi_val:.1f})")
        elif 35<=rsi_val<45:  m+=1; md.append(f"~ RSI bas ({rsi_val:.1f})")
        elif rsi_val>70:      m+=1; md.append(f"⚠️ RSI suracheté ({rsi_val:.1f})")
        else:                      md.append(f"❌ RSI très bas ({rsi_val:.1f})")
        if pct_hi>=-15:  m+=1; md.append(f"✅ À {abs(pct_hi):.1f}% du sommet 52W")
        elif pct_hi>=-25:     md.append(f"~ À {abs(pct_hi):.1f}% du sommet")
        else:                 md.append(f"❌ Loin du sommet ({pct_hi:.1f}%)")

        # ══ 🎯 CONVICTION (10 pts) — Insiders + Upside + Recommandation ══
        c_s=0; cd=[]

        # Insider ownership (3 pts)
        if ip is not None:
            if ip>10:   c_s+=3; cd.append(f"✅ Insiders {ip:.1f}% — fort alignement")
            elif ip>5:  c_s+=2; cd.append(f"✅ Insiders {ip:.1f}% — bon alignement")
            elif ip>2:  c_s+=1; cd.append(f"~ Insiders {ip:.1f}%")
            else:       c_s+=0; cd.append(f"~ Insiders {ip:.1f}% — très faible")
        else: c_s+=1; cd.append("~ Insider ownership N/D")

        # Upside analystes (4 pts)
        upside=None
        if target_mean and price>0:
            upside=round((target_mean-price)/price*100,1)
            if upside>30:   c_s+=4; cd.append(f"✅ Upside +{upside}% — très fort potentiel")
            elif upside>20: c_s+=3; cd.append(f"✅ Upside +{upside}% — fort potentiel")
            elif upside>10: c_s+=2; cd.append(f"✅ Upside +{upside}%")
            elif upside>0:  c_s+=1; cd.append(f"~ Upside +{upside}%")
            else:           c_s+=0; cd.append(f"❌ Upside {upside}% — surcoté")
        else: c_s+=1; cd.append("~ Cible analystes N/D")

        # Recommandation consensus (3 pts)
        if recommend is not None and nb_analysts:
            nb_str=f" ({nb_analysts} analystes)"
            if recommend<=1.5:   c_s+=3; cd.append(f"✅ Strong Buy consensus{nb_str}")
            elif recommend<=2.0: c_s+=2; cd.append(f"✅ Buy consensus{nb_str}")
            elif recommend<=2.5: c_s+=1; cd.append(f"~ Overweight{nb_str}")
            elif recommend<=3.0: c_s+=0; cd.append(f"~ Hold{nb_str}")
            else:                c_s+=0; cd.append(f"❌ Underperform/Sell{nb_str}")
        else: c_s+=1; cd.append("~ Recommandation N/D")

        # ══ 🏦 SIGNAUX INSTITUTIONNELS (20 pts) ══
        # Institutional Ownership | Short Interest | ROIC | Current Ratio
        # P/S Ratio | Beta | Momentum relatif 52W | Tendance marges
        i_s=0; id_=[]

        # Institutional ownership % (3 pts)
        if inst_pct is not None:
            if inst_pct>70:   i_s+=3; id_.append(f"✅ Institutions {inst_pct:.1f}% — très fort intérêt")
            elif inst_pct>50: i_s+=2; id_.append(f"✅ Institutions {inst_pct:.1f}% — intérêt solide")
            elif inst_pct>30: i_s+=1; id_.append(f"~ Institutions {inst_pct:.1f}%")
            else:             i_s+=0; id_.append(f"❌ Institutions {inst_pct:.1f}% — peu d'intérêt")
        else: i_s+=1; id_.append("~ Inst. ownership N/D")

        # Short Interest % du float (3 pts — faible short = bon signe)
        if short_float is not None:
            if short_float<2:    i_s+=3; id_.append(f"✅ Short interest très faible ({short_float:.1f}%) — pas de pression vendeuse")
            elif short_float<5:  i_s+=2; id_.append(f"✅ Short interest faible ({short_float:.1f}%)")
            elif short_float<10: i_s+=1; id_.append(f"~ Short interest modéré ({short_float:.1f}%)")
            elif short_float<15: i_s+=0; id_.append(f"⚠️ Short interest élevé ({short_float:.1f}%) — méfiance des pros")
            else:                i_s+=0; id_.append(f"❌ Short interest très élevé ({short_float:.1f}%) — fort pari baissier")
        else: i_s+=1; id_.append("~ Short interest N/D")

        # ROIC calculé (4 pts)
        if roic is not None:
            if roic>20:   i_s+=4; id_.append(f"✅ ROIC {roic:.1f}% — avantage compétitif exceptionnel")
            elif roic>15: i_s+=3; id_.append(f"✅ ROIC {roic:.1f}% — avantage compétitif solide")
            elif roic>10: i_s+=2; id_.append(f"~ ROIC {roic:.1f}% — correct")
            elif roic>5:  i_s+=1; id_.append(f"~ ROIC {roic:.1f}% — faible")
            else:         i_s+=0; id_.append(f"❌ ROIC {roic:.1f}% — destruction de valeur")
        else: i_s+=1; id_.append("~ ROIC N/D")

        # Current Ratio (2 pts)
        if current_ratio is not None:
            if current_ratio>2.0:   i_s+=2; id_.append(f"✅ Current Ratio {current_ratio:.2f} — liquidité excellente")
            elif current_ratio>1.5: i_s+=2; id_.append(f"✅ Current Ratio {current_ratio:.2f} — liquidité solide")
            elif current_ratio>1.0: i_s+=1; id_.append(f"~ Current Ratio {current_ratio:.2f} — liquidité correcte")
            else:                   i_s+=0; id_.append(f"❌ Current Ratio {current_ratio:.2f} — risque liquidité")
        else: i_s+=1; id_.append("~ Current Ratio N/D")

        # P/S Ratio contextualisé par secteur (2 pts)
        # Seuils différents selon secteur : SaaS/Tech supporte P/S plus élevé
        ps_thresholds = {
            "Technology": (8, 15, 25),           # bas, moyen, élevé
            "Communication Services": (4, 8, 15),
            "Financial Services": (2, 4, 8),
            "Healthcare": (3, 6, 12),
            "Consumer Cyclical": (1, 3, 6),
            "Consumer Defensive": (1, 2, 4),
            "Industrials": (1, 2, 4),
            "Energy": (0.5, 1.5, 3),
            "Basic Materials": (1, 2, 4),
            "Real Estate": (4, 8, 15),
            "Utilities": (1, 2, 4),
        }
        ps_t = ps_thresholds.get(sector, (2, 5, 10))
        if ps_ratio is not None and ps_ratio>0:
            if ps_ratio<ps_t[0]:   i_s+=2; id_.append(f"✅ P/S très bas ({ps_ratio:.1f}) vs secteur {sector}")
            elif ps_ratio<ps_t[1]: i_s+=1; id_.append(f"~ P/S raisonnable ({ps_ratio:.1f}) pour {sector}")
            elif ps_ratio<ps_t[2]: i_s+=0; id_.append(f"~ P/S élevé ({ps_ratio:.1f}) pour {sector}")
            else:                  i_s+=0; id_.append(f"❌ P/S très élevé ({ps_ratio:.1f}) pour {sector}")
        else: i_s+=1; id_.append("~ P/S Ratio N/D")

        # Beta — informatif uniquement, pas pénalisant pour l'investisseur long terme
        # Un beta élevé = volatilité, pas nécessairement mauvais (Amazon, Tesla historiquement)
        if beta is not None:
            if 0.5<=beta<=1.5: i_s+=2; id_.append(f"✅ Beta {beta:.2f} — volatilité acceptable")
            elif beta<0.5:     i_s+=1; id_.append(f"~ Beta {beta:.2f} — très défensif (faible upside)")
            elif beta<=2.0:    i_s+=1; id_.append(f"~ Beta {beta:.2f} — volatile (risque élevé)")
            else:              i_s+=0; id_.append(f"⚠️ Beta {beta:.2f} — très spéculatif")
        else: i_s+=1; id_.append("~ Beta N/D")

        # Momentum relatif 52W vs SPY (2 pts)
        if rel_momentum is not None:
            if rel_momentum>15:   i_s+=2; id_.append(f"✅ Surperformance vs SPY: +{rel_momentum:.1f}% — momentum fort")
            elif rel_momentum>5:  i_s+=1; id_.append(f"✅ Surperformance vs SPY: +{rel_momentum:.1f}%")
            elif rel_momentum>-5: i_s+=1; id_.append(f"~ Performance proche SPY ({rel_momentum:.1f}%)")
            else:                 i_s+=0; id_.append(f"❌ Sous-performance vs SPY: {rel_momentum:.1f}%")
        else: i_s+=1; id_.append("~ Momentum relatif N/D")

        # Tendance des marges (2 pts)
        if margin_trend is not None:
            if margin_trend>2:    i_s+=2; id_.append(f"✅ Marges en hausse (+{margin_trend:.1f}pts) — amélioration opérationnelle")
            elif margin_trend>0:  i_s+=1; id_.append(f"~ Marges légèrement en hausse (+{margin_trend:.1f}pts)")
            elif margin_trend>-2: i_s+=0; id_.append(f"~ Marges stables ({margin_trend:.1f}pts)")
            else:                 i_s+=0; id_.append(f"❌ Marges en baisse ({margin_trend:.1f}pts) — compression opérationnelle")
        else: i_s+=1; id_.append("~ Tendance marges N/D")

        # ── Score final ──
        raw   = q + g_s + v + m + c_s + i_s
        total = min(100, max(0, raw + market_bonus + sec_bonus))

        # Score technique d'entree (Fibonacci + Timing)
        tech_entry = 0
        fib_hi_t = float(close.rolling(min(90,len(close))).max().iloc[-1])
        fib_lo_t = float(close.rolling(min(90,len(close))).min().iloc[-1])
        fib_range_t = fib_hi_t - fib_lo_t
        if fib_range_t > 0:
            fib_618_t = fib_hi_t - 0.618 * fib_range_t
            fib_500_t = fib_hi_t - 0.500 * fib_range_t
            fib_382_t = fib_hi_t - 0.382 * fib_range_t
            if price <= fib_618_t * 1.02:   tech_entry += 5
            elif price <= fib_500_t * 1.02: tech_entry += 3
            elif price <= fib_382_t * 1.02: tech_entry += 2
        if 40 <= rsi_val <= 65: tech_entry += 3
        elif rsi_val < 40:      tech_entry += 4
        elif rsi_val > 75:      tech_entry -= 2
        if ma50 > float(close.rolling(min(200,len(close))).mean().iloc[-1]):
            tech_entry += 2

        # Score Global = fondamental + bonus timing
        timing_bonus = 5 if tech_entry >= 8 else 3 if tech_entry >= 5 else 0 if tech_entry >= 2 else -5
        score_global = min(100, max(0, total + timing_bonus))

        # Verdict timing
        if tech_entry >= 8:   timing_verdict = "ENTRER MAINTENANT"
        elif tech_entry >= 5: timing_verdict = "CONDITIONS FAVORABLES"
        elif tech_entry >= 2: timing_verdict = "ATTENDRE"
        else:                 timing_verdict = "EVITER" 

        # Convergence (6 catégories)
        sigs=[]
        if q>=17:   sigs.append("Qualité")
        if g_s>=14: sigs.append("Croissance")
        if v>=10:   sigs.append("Valorisation")
        if m>=7:    sigs.append("Momentum")
        if c_s>=7:  sigs.append("Conviction")
        if i_s>=14: sigs.append("Institutionnel")
        conv=len(sigs)
        cbar="█"*conv+"░"*(6-conv)

        if total>=83:   sig="🟢 EXCELLENT"
        elif total>=68: sig="🟢 BON"
        elif total>=52: sig="🟡 CORRECT"
        else:           sig="🔴 ÉVITER"

        return {
            "Ticker":ticker,"Nom":name,"Secteur":sector,
            "Cap (G$)":round(cap_b,1),"Prix $":round(price,2),"Var. %":round(chg,2),
            "ROE %":round(rp,1) if rp is not None else "N/D",
            "Marge Brute %":round(gm,1) if gm is not None else "N/D",
            "Marge Nette %":round(nm,1) if nm is not None else "N/D",
            "FCF (G$)":round(fcf/1e9,2) if fcf else "N/D",
            "Rev. Growth % (5Y CAGR)": round(rev_cagr_5y,1) if rev_cagr_5y is not None else ("N/D" if rg is None else f"{round(rg,1)} (TTM)"),
            "EPS Growth % (5Y CAGR)":  round(eps_cagr_5y,1) if eps_cagr_5y is not None else ("N/D" if eg is None else f"{round(eg,1)} (TTM)"),
            "Rev. Consistance":   rev_consistency if rev_consistency else "N/D",
            "Marge 5Y Tendance":  round(margin_5y,1) if margin_5y is not None else "N/D",
            "ROE 5Y Tendance":    round(roe_5y_trend,1) if roe_5y_trend is not None else "N/D",
            "FCF 5Y":             fcf_5y if fcf_5y else "N/D",
            "Secteur Force":      sec_label,
            "Secteur Bonus":      sec_bonus,
            "P/E":round(pe,1) if pe and pe>0 else "N/D",
            "PEG":round(peg,2) if peg and peg>0 else "N/D",
            "EV/EBITDA":round(ev_ebitda,1) if ev_ebitda and ev_ebitda>0 else "N/D",
            "P/S":round(ps_ratio,1) if ps_ratio and ps_ratio>0 else "N/D",
            "RSI":round(rsi_val,1),"MA50":round(ma50,2),"MA200":round(ma200,2),
            "52W High %":pct_hi,
            "Beta":round(beta,2) if beta else "N/D",
            "Momentum Relatif %":rel_momentum if rel_momentum is not None else "N/D",
            "Insider %":round(ip,1) if ip is not None else "N/D",
            "Inst. Ownership %":round(inst_pct,1) if inst_pct is not None else "N/D",
            "Short Interest %":round(short_float,1) if short_float is not None else "N/D",
            "ROIC %":roic if roic is not None else "N/D",
            "Current Ratio":round(current_ratio,2) if current_ratio else "N/D",
            "Tendance Marge":round(margin_trend,1) if margin_trend is not None else "N/D",
            "Upside Analystes %":upside if upside is not None else "N/D",
            "Nb Analystes":nb_analysts if nb_analysts else "N/D",
            "Recommandation":round(recommend,2) if recommend is not None else "N/D",
            "Cible Moy $":round(target_mean,2) if target_mean else "N/D",
            "Score Qualité":q,"Score Croissance":g_s,
            "Score Valorisation":v,"Score Momentum":m,
            "Score Conviction":c_s,"Score Institutionnel":i_s,
            "Score Total":total,"Score Global":score_global,
            "Timing":timing_verdict,"Tech Entry Score":tech_entry,
            "Signal":sig,
            "Convergence":conv,"Conv_Bar":cbar,
            "Signaux":" | ".join(sigs) if sigs else "Aucun",
            "_q":qd,"_g":gd,"_v":vd,"_m":md,"_c":cd,"_i":id_,
        }
    except Exception:
        return None

def run_scoring_parallel(passed, market_bonus=0, max_workers=8, spy_hist=None, sector_data=None, fund_cache=None, cb=None):
    """Score en parallèle — fondamentaux pré-chargés dans fund_cache."""
    results=[]
    if fund_cache is None: fund_cache = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures={ex.submit(score_ticker,i["ticker"],i["data"],market_bonus,spy_hist,sector_data,fund_cache.get(i["ticker"])):i["ticker"]
                 for i in passed}
        done=0
        for f in concurrent.futures.as_completed(futures):
            done+=1
            r=f.result()
            if r: results.append(r)
            if cb: cb(done,len(passed))
    return results

# ─────────────────────────────
# ─────────────────────────────
# 📈 BACKTEST LONG TERME — aligné sur les 6 catégories
# Horizon 6-12 mois | Fondamentaux + Technique
# ─────────────────────────────
@st.cache_data(ttl=86401)  # 24h + 1s pour invalider l'ancien cache
def batch_download_backtest(tickers_tuple):
    tickers = list(tickers_tuple)
    try:
        raw = yf.download(tickers, period="5y", auto_adjust=True,
                          progress=False, group_by="ticker", threads=True)
        prices = {}
        if len(tickers) == 1:
            t = tickers[0]
            if not raw.empty: prices[t] = raw
        else:
            for t in tickers:
                try:
                    df_t = raw[t].dropna()
                    if len(df_t) >= 200: prices[t] = df_t
                except Exception:
                    continue
        return prices
    except Exception as e:
        st.warning(f"Erreur backtest download: {e}")
        return {}


def score_for_backtest(close_series, vol_series, high_series=None, low_series=None,
                       pe=None, roe=None, net_margin=None, rev_growth=None,
                       gross_margin=None, debt_eq=None, roic=None,
                       short_pct=None, inst_own=None, insider_pct=None,
                       upside=None, peg=None, ev_ebitda=None):
    """
    Score aligné sur les 6 catégories du screener + Fibonacci + Timing.
    Utilise tous les paramètres fondamentaux disponibles.
    """
    c = close_series
    v = vol_series
    n = len(c)
    if n < 50: return 0, 0

    score = 0

    # ── Qualité (25 pts) ──
    if roe is not None:
        if roe > 20:   score += 8
        elif roe > 15: score += 6
        elif roe > 10: score += 3
        elif roe > 0:  score += 1
    else: score += 3

    if gross_margin is not None:
        if gross_margin > 50:   score += 5
        elif gross_margin > 35: score += 3
        elif gross_margin > 20: score += 1
    else: score += 2

    if net_margin is not None:
        if net_margin > 20:   score += 5
        elif net_margin > 10: score += 3
        elif net_margin > 5:  score += 1
    else: score += 2

    if debt_eq is not None:
        if debt_eq < 0.3:    score += 4
        elif debt_eq < 0.75: score += 3
        elif debt_eq < 1.5:  score += 1
    else: score += 2

    # FCF proxy — neutre
    score += 3

    # ── Croissance (20 pts) ──
    if rev_growth is not None:
        rg = rev_growth * 100 if rev_growth < 5 else rev_growth
        if rg > 15:   score += 10
        elif rg > 10: score += 8
        elif rg > 8:  score += 5
        elif rg > 5:  score += 3
        elif rg > 0:  score += 1
    else: score += 5
    score += 5  # EPS neutre (pas disponible historiquement)
    score += 2  # Consistance neutre

    # ── Valorisation (15 pts) ──
    if pe is not None and pe > 0:
        if pe < 15:   score += 6
        elif pe < 25: score += 4
        elif pe < 35: score += 2
    else: score += 2

    if peg is not None and peg > 0:
        if peg < 1.0:  score += 5
        elif peg < 1.5:score += 3
        elif peg < 2.0:score += 1
    else: score += 2

    if ev_ebitda is not None and ev_ebitda > 0:
        if ev_ebitda < 12:   score += 4
        elif ev_ebitda < 18: score += 3
        elif ev_ebitda < 25: score += 1
    else: score += 2

    # ── Momentum technique (10 pts) ──
    p    = float(c.iloc[-1])
    m50  = float(c.rolling(50).mean().iloc[-1])
    m200 = float(c.rolling(min(200,n)).mean().iloc[-1])

    if p > m50 > m200: score += 4
    elif p > m200:     score += 2

    d = c.diff()
    g = d.where(d>0,0).rolling(14).mean()
    l = -d.where(d<0,0).rolling(14).mean()
    rsi = float(100-(100/(1+g/l.clip(lower=1e-10))).iloc[-1])
    if 45 <= rsi <= 70:  score += 4
    elif 35 <= rsi < 45: score += 2
    elif rsi > 70:       score += 1

    avg_vol  = float(v.rolling(20).mean().iloc[-1])
    last_vol = float(v.iloc[-1])
    vr = last_vol / avg_vol if avg_vol > 0 else 1.0
    if vr >= 1.5:  score += 2
    elif vr >= 1.1:score += 1

    # ── Conviction (10 pts) ──
    if upside is not None:
        if upside > 30:   score += 4
        elif upside > 20: score += 3
        elif upside > 10: score += 1
    else: score += 2

    if insider_pct is not None:
        if insider_pct > 10:  score += 3
        elif insider_pct > 5: score += 2
        elif insider_pct > 2: score += 1
    else: score += 1

    score += 3  # Recommandation neutre

    # ── Institutionnel (20 pts) ──
    if inst_own is not None:
        if inst_own > 70:   score += 3
        elif inst_own > 50: score += 2
        elif inst_own > 30: score += 1
    else: score += 1

    if short_pct is not None:
        if short_pct < 2:    score += 3
        elif short_pct < 5:  score += 2
        elif short_pct < 10: score += 1
        elif short_pct > 15: score -= 1
    else: score += 1

    if roic is not None:
        if roic > 20:   score += 4
        elif roic > 15: score += 3
        elif roic > 10: score += 2
        elif roic > 5:  score += 1
    else: score += 1

    # Current Ratio, P/S, Beta — neutres (non disponibles historiquement)
    score += 6

    # Momentum relatif — proxy : perf 3 mois vs RSI
    if n >= 63:
        perf_3m = (p - float(c.iloc[-63])) / float(c.iloc[-63]) * 100
        if perf_3m > 15:  score += 2
        elif perf_3m > 5: score += 1
    else: score += 1

    # Tendance marges — neutre
    score += 2

    score = min(score, 100)

    # ── Score technique d'entrée (Fibonacci + Timing) ──
    tech_entry = 0
    if high_series is not None and low_series is not None and len(high_series) >= 90:
        h90 = high_series
        l90 = low_series
        fib_hi = float(h90.rolling(min(90,len(h90))).max().iloc[-1])
        fib_lo = float(l90.rolling(min(90,len(l90))).min().iloc[-1])
        fib_r  = fib_hi - fib_lo
        if fib_r > 0:
            fib618 = fib_hi - 0.618 * fib_r
            fib500 = fib_hi - 0.500 * fib_r
            fib382 = fib_hi - 0.382 * fib_r
            if p <= fib618 * 1.02:   tech_entry += 5
            elif p <= fib500 * 1.02: tech_entry += 3
            elif p <= fib382 * 1.02: tech_entry += 2

    if 40 <= rsi <= 65:  tech_entry += 3
    elif rsi < 40:       tech_entry += 4
    elif rsi > 75:       tech_entry -= 2

    if m50 > m200:       tech_entry += 2

    # Score Global = fondamental + bonus timing
    timing_bonus = 5 if tech_entry >= 8 else 3 if tech_entry >= 5 else 0 if tech_entry >= 2 else -5
    score_global = min(100, max(0, score + timing_bonus))

    return score, score_global


def backtest_long_terme(ticker, hist, horizon_months=6):
    """
    Backtest long terme aligné sur les 6 catégories + Fibonacci + Score Global.
    """
    try:
        if hist is None or hist.empty or len(hist) < 200:
            return []
        hist = hist.copy()
        hist.index = pd.to_datetime(hist.index)

        # Normaliser colonnes si MultiIndex
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        # Vérifier colonnes nécessaires
        needed = ["Close","Volume","High","Low"]
        for col in needed:
            if col not in hist.columns:
                return []

        # Fondamentaux complets via yfinance
        try:
            t_obj      = yf.Ticker(ticker)
            info       = t_obj.info
            pe         = info.get("trailingPE", None)
            peg        = info.get("pegRatio", None)
            ev_ebitda  = info.get("enterpriseToEbitda", None)
            roe        = (info.get("returnOnEquity", None) or 0) * 100
            gross_m    = (info.get("grossMargins", None) or 0) * 100
            net_margin = (info.get("profitMargins", None) or 0) * 100
            rev_growth = info.get("revenueGrowth", None)
            debt_eq    = (info.get("debtToEquity", None) or 0) / 100
            roic       = None
            try:
                ebit   = info.get("ebit", None)
                equity = info.get("totalStockholderEquity", None)
                debt   = info.get("totalDebt", 0) or 0
                tax    = info.get("effectiveTaxRate", 0.21) or 0.21
                if ebit and equity and equity > 0:
                    roic = round(ebit*(1-tax)/(equity+debt)*100, 1)
            except Exception: pass
            short_pct  = (info.get("shortPercentOfFloat", None) or 0) * 100
            inst_own   = (info.get("heldPercentInstitutions", None) or 0) * 100
            insider    = (info.get("heldPercentInsiders", None) or 0) * 100
            target     = info.get("targetMeanPrice", None)
        except Exception:
            pe=peg=ev_ebitda=roe=gross_m=net_margin=rev_growth=None
            debt_eq=roic=short_pct=inst_own=insider=target=None

        # Scan mensuel
        monthly = hist.resample("ME").last()
        n       = len(monthly)
        trades  = []

        for i in range(12, n - horizon_months):
            entry_date    = monthly.index[i]
            exit_date     = monthly.index[i + horizon_months]
            hist_to_entry = hist[hist.index <= entry_date]
            if len(hist_to_entry) < 100: continue

            entry_price_now = float(monthly["Close"].iloc[i])

            # Calcul upside dynamique si target disponible
            upside_pct = round((target - entry_price_now) / entry_price_now * 100, 1) \
                         if target and entry_price_now > 0 else None

            score, score_global = score_for_backtest(
                hist_to_entry["Close"],
                hist_to_entry["Volume"],
                high_series=hist_to_entry["High"],
                low_series=hist_to_entry["Low"],
                pe=pe, roe=roe, net_margin=net_margin,
                rev_growth=rev_growth, gross_margin=gross_m,
                debt_eq=debt_eq, roic=roic,
                short_pct=short_pct, inst_own=inst_own,
                insider_pct=insider, upside=upside_pct,
                peg=peg, ev_ebitda=ev_ebitda,
            )

            # Seuil d'entrée — score >= 60 OU score_global >= 63
            if score < 60 and score_global < 63:
                continue

            entry_price = float(monthly["Close"].iloc[i])
            exit_price  = float(monthly["Close"].iloc[i + horizon_months])
            if entry_price <= 0: continue

            perf = round((exit_price - entry_price) / entry_price * 100, 2)
            if perf > 5:    result = "WIN"
            elif perf < -5: result = "LOSS"
            else:           result = "BREAKEVEN"

            period_data = hist[
                (hist.index >= entry_date) & (hist.index <= exit_date)
            ]["Close"]
            max_dd = round(
                (period_data.min() - entry_price) / entry_price * 100, 2
            ) if len(period_data) > 0 else 0.0

            trades.append({
                "ticker":        ticker,
                "entry_date":    str(entry_date.date()),
                "exit_date":     str(exit_date.date()),
                "horizon":       f"{horizon_months}M",
                "score":         int(score),
                "score_global":  int(score_global),
                "entry_price":   round(entry_price, 2),
                "exit_price":    round(exit_price, 2),
                "perf":          perf,
                "max_drawdown":  max_dd,
                "result":        result,
            })

        return trades
    except Exception:
        return []


def bt_stats(df):
    if df.empty: return {}
    n      = len(df)
    wins   = len(df[df["result"] == "WIN"])
    losses = len(df[df["result"] == "LOSS"])
    wr     = round(wins / n * 100, 1)
    pcol   = "perf" if "perf" in df.columns else "pnl"
    aw     = round(float(df[df["result"]=="WIN"][pcol].mean()),  2) if wins   > 0 else 0.0
    al     = round(float(df[df["result"]=="LOSS"][pcol].mean()), 2) if losses > 0 else 0.0
    gp     = df[df[pcol] > 0][pcol].sum()
    gl     = abs(df[df[pcol] < 0][pcol].sum())
    pf     = round(float(gp/gl), 2) if gl > 0 else 9.9
    exp    = round(wr/100*aw + (1-wr/100)*al, 2)
    avg_p  = round(float(df[pcol].mean()), 2)
    h_months = 6
    if "horizon" in df.columns and len(df) > 0:
        try: h_months = int(df["horizon"].iloc[0].replace("M",""))
        except: pass
    annual  = round(avg_p * (12 / h_months), 1)
    avg_dd  = round(float(df["max_drawdown"].mean()), 2) if "max_drawdown" in df.columns else 0
    sc_stats = {}
    for lo, hi, lbl in [(80,101,">=80"),(65,80,"65-79"),(60,65,"60-64")]:
        sub = df[(df["score"]>=lo) & (df["score"]<hi)]
        if len(sub) > 0:
            sw = len(sub[sub["result"]=="WIN"])
            sc_stats[lbl] = {
                "n": int(len(sub)),
                "win_rate": round(sw/len(sub)*100, 1),
                "avg_perf": round(float(sub[pcol].mean()), 2),
            }

    # Stats par Score Global (si disponible)
    sg_stats = {}
    if "score_global" in df.columns:
        for lo, hi, lbl in [(85,101,"Global>=85"),(70,85,"Global 70-84"),(63,70,"Global 63-69")]:
            sub = df[(df["score_global"]>=lo) & (df["score_global"]<hi)]
            if len(sub) > 0:
                sw = len(sub[sub["result"]=="WIN"])
                sg_stats[lbl] = {
                    "n": int(len(sub)),
                    "win_rate": round(sw/len(sub)*100, 1),
                    "avg_perf": round(float(sub[pcol].mean()), 2),
                }

    return {
        "total": n, "wins": wins, "losses": losses,
        "win_rate": wr, "avg_win": aw, "avg_loss": al,
        "profit_factor": pf, "expectancy": exp,
        "avg_perf": avg_p, "annual_return": annual,
        "avg_drawdown": avg_dd,
        "best":  round(float(df[pcol].max()), 2),
        "worst": round(float(df[pcol].min()), 2),
        "score_stats": sc_stats,
        "score_global_stats": sg_stats,
    }
# ─────────────────────────────
# 🤖 CLAUDE IA
# ─────────────────────────────

# ─────────────────────────────
# 📄 RAPPORT PDF TOP 10
# ─────────────────────────────
def generate_pdf_report(df_top10, ms, scan_date):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, HRFlowable)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=15*mm, leftMargin=15*mm,
                                topMargin=15*mm, bottomMargin=15*mm)

        # Couleurs
        DARK    = colors.HexColor("#0a0e1a")
        GREEN   = colors.HexColor("#00ff88")
        BLUE    = colors.HexColor("#4a90d0")
        YELLOW  = colors.HexColor("#fbbf24")
        RED     = colors.HexColor("#f87171")
        GRAY    = colors.HexColor("#64748b")
        LGRAY   = colors.HexColor("#1e3a5f")
        WHITE   = colors.white
        ORANGE  = colors.HexColor("#f97316")
        PURPLE  = colors.HexColor("#a78bfa")
        CYAN    = colors.HexColor("#38bdf8")

        styles = getSampleStyleSheet()

        def style(name, **kw):
            return ParagraphStyle(name, **kw)

        s_title   = style("T", fontSize=22, textColor=GREEN,   fontName="Helvetica-Bold", spaceAfter=2)
        s_sub     = style("S", fontSize=9,  textColor=GRAY,    fontName="Helvetica",      spaceAfter=8)
        s_h1      = style("H1",fontSize=13, textColor=GREEN,   fontName="Helvetica-Bold", spaceAfter=4, spaceBefore=10)
        s_h2      = style("H2",fontSize=10, textColor=WHITE,   fontName="Helvetica-Bold", spaceAfter=3)
        s_body    = style("B", fontSize=8,  textColor=WHITE,   fontName="Helvetica",      spaceAfter=2)
        s_small   = style("SM",fontSize=7,  textColor=GRAY,    fontName="Helvetica",      spaceAfter=1)
        s_signal_g= style("SG",fontSize=11, textColor=GREEN,   fontName="Helvetica-Bold", spaceAfter=4, alignment=TA_CENTER)
        s_signal_y= style("SY",fontSize=11, textColor=YELLOW,  fontName="Helvetica-Bold", spaceAfter=4, alignment=TA_CENTER)
        s_signal_r= style("SR",fontSize=11, textColor=RED,     fontName="Helvetica-Bold", spaceAfter=4, alignment=TA_CENTER)
        s_center  = style("C", fontSize=8,  textColor=GRAY,    fontName="Helvetica",      alignment=TA_CENTER)

        def signal_style(sig):
            return s_signal_g if "EXCELLENT" in sig or "BON" in sig else \
                   s_signal_y if "CORRECT" in sig else s_signal_r

        story = []

        # ── PAGE DE GARDE ──
        story.append(Spacer(1, 20*mm))
        story.append(Paragraph("📊 AlphaScreen US", s_title))
        story.append(Paragraph(f"Rapport Top 10 — {scan_date}", s_sub))
        story.append(Paragraph(
            f"Marché: <b>{ms['regime']}</b>  |  SPY vs MA50: {ms['spy_vs_ma50']:+.1f}%  "
            f"|  VIX: {ms['vix']} ({ms['vix_label']})  |  Bonus score: {ms['bonus']:+d}pts",
            s_body))
        story.append(Spacer(1, 4*mm))
        story.append(HRFlowable(width="100%", thickness=1, color=GREEN))
        story.append(Spacer(1, 4*mm))

        # Résumé scoring
        story.append(Paragraph("Système de scoring /100 — 6 catégories", s_h2))
        score_table_data = [
            ["Catégorie", "Pts", "Critères"],
            ["🏆 Qualité",      "25", "ROE · Marge brute · Marge nette · FCF · Dette"],
            ["📈 Croissance",   "20", "Revenus · EPS annuel · EPS trimestriel"],
            ["💲 Valorisation", "15", "P/E · PEG · EV/EBITDA"],
            ["🚀 Momentum",     "10", "MA50>MA200 · Prix>MA50 · RSI · 52W High"],
            ["🎯 Conviction",   "10", "Insiders · Upside analystes · Recommandation"],
            ["🏦 Institutionnel","20","ROIC · Short Interest · Inst.Own · Current Ratio · Beta · P/S"],
        ]
        t_score = Table(score_table_data, colWidths=[42*mm, 12*mm, 115*mm])
        t_score.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), LGRAY),
            ("TEXTCOLOR",   (0,0), (-1,0), GREEN),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 7.5),
            ("TEXTCOLOR",   (0,1), (-1,-1), WHITE),
            ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
            ("BACKGROUND",  (0,1), (-1,-1), DARK),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#0d1117"), DARK]),
            ("GRID",        (0,0), (-1,-1), 0.3, LGRAY),
            ("ALIGN",       (1,0), (1,-1), "CENTER"),
            ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        story.append(t_score)
        story.append(Spacer(1, 6*mm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LGRAY))

        # ── FICHE PAR TITRE ──
        for rank, (_, row) in enumerate(df_top10.iterrows(), 1):
            story.append(Spacer(1, 5*mm))

            medals = {1:"🥇", 2:"🥈", 3:"🥉"}
            rank_icon = medals.get(rank, f"#{rank}")
            score = row["Score Total"]
            signal = row["Signal"]

            # En-tête titre
            story.append(Paragraph(
                f"{rank_icon}  {row['Ticker']} — {row['Nom']}",
                s_h1))
            story.append(Paragraph(signal, signal_style(signal)))

            # Scores 6 catégories
            scores_data = [[
                "Score Total", "Qualité\n/25", "Croissance\n/20",
                "Valorisation\n/15", "Momentum\n/10",
                "Conviction\n/10", "Institutionnel\n/20"
            ],[
                f"{score}/100",
                str(row.get("Score Qualité","—")),
                str(row.get("Score Croissance","—")),
                str(row.get("Score Valorisation","—")),
                str(row.get("Score Momentum","—")),
                str(row.get("Score Conviction","—")),
                str(row.get("Score Institutionnel","—")),
            ]]
            t_sc = Table(scores_data, colWidths=[25*mm,24*mm,24*mm,24*mm,21*mm,21*mm,21*mm])
            t_sc.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,0), LGRAY),
                ("TEXTCOLOR",   (0,0), (-1,0), CYAN),
                ("FONTNAME",    (0,0), (-1,-1), "Helvetica-Bold"),
                ("FONTSIZE",    (0,0), (-1,-1), 7.5),
                ("BACKGROUND",  (0,1), (0,1), colors.HexColor("#002a18")),
                ("TEXTCOLOR",   (0,1), (0,1), GREEN),
                ("FONTSIZE",    (0,1), (0,1), 11),
                ("TEXTCOLOR",   (1,1), (-1,1), WHITE),
                ("BACKGROUND",  (1,1), (-1,1), DARK),
                ("ALIGN",       (0,0), (-1,-1), "CENTER"),
                ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
                ("GRID",        (0,0), (-1,-1), 0.3, LGRAY),
                ("TOPPADDING",  (0,0), (-1,-1), 4),
                ("BOTTOMPADDING",(0,0),(-1,-1), 4),
            ]))
            story.append(t_sc)
            story.append(Spacer(1, 3*mm))

            # Données de marché — 2 colonnes
            def v(key): return str(row.get(key, "N/D"))
            left_data = [
                ["💰 Prix",          str(v("Prix $")) + " USD  (" + str(v("Var. %")) + "%)"],
                ["📈 ROE",           f"{v('ROE %')}%"],
                ["📊 Marge brute",   f"{v('Marge Brute %')}%"],
                ["📊 Marge nette",   f"{v('Marge Nette %')}%"],
                ["💵 FCF",           f"{v('FCF (G$)')}G$"],
                ["📈 Rev. Growth",   f"{v('Rev. Growth %')}%"],
                ["📈 EPS Growth",    f"{v('EPS Growth %')}%"],
                ["📈 EPS Trimestriel",f"{v('EPS Qtr %')}%"],
            ]
            right_data = [
                ["💲 P/E",           v("P/E")],
                ["💲 PEG",           v("PEG")],
                ["💲 EV/EBITDA",     v("EV/EBITDA")],
                ["💲 P/S",           v("P/S")],
                ["🏦 ROIC",          f"{v('ROIC %')}%"],
                ["🏦 Short Interest",f"{v('Short Interest %')}%"],
                ["🏦 Inst. Own.",    f"{v('Inst. Ownership %')}%"],
                ["🏦 Current Ratio", v("Current Ratio")],
            ]

            def make_detail_table(data):
                t = Table(data, colWidths=[35*mm, 48*mm])
                t.setStyle(TableStyle([
                    ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
                    ("FONTSIZE",    (0,0), (-1,-1), 7.5),
                    ("TEXTCOLOR",   (0,0), (0,-1), GRAY),
                    ("TEXTCOLOR",   (1,0), (1,-1), WHITE),
                    ("FONTNAME",    (1,0), (1,-1), "Helvetica-Bold"),
                    ("BACKGROUND",  (0,0), (-1,-1), DARK),
                    ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#0d1117"), DARK]),
                    ("GRID",        (0,0), (-1,-1), 0.2, LGRAY),
                    ("TOPPADDING",  (0,0), (-1,-1), 2),
                    ("BOTTOMPADDING",(0,0),(-1,-1), 2),
                ]))
                return t

            from reportlab.platypus import KeepInFrame
            tbl_data = [[make_detail_table(left_data), make_detail_table(right_data)]]
            t_2col = Table(tbl_data, colWidths=[87*mm, 87*mm])
            t_2col.setStyle(TableStyle([
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING", (0,0), (-1,-1), 0),
                ("RIGHTPADDING", (0,0), (-1,-1), 3*mm),
            ]))
            story.append(t_2col)
            story.append(Spacer(1, 3*mm))

            # Conviction & Institutionnel
            conv_data = [
                ["🎯 Insiders", f"{v('Insider %')}%",
                 "🎯 Upside analystes", f"{v('Upside Analystes %')}%",
                 "🎯 Recommandation", f"{v('Recommandation')}/5  ({v('Nb Analystes')} analystes)"],
                ["🏦 Beta", v("Beta"),
                 "🏦 Momentum rel.", f"{v('Momentum Relatif %')}%",
                 "🏦 Tendance marge", f"{v('Tendance Marge')}pts"],
            ]
            t_conv = Table(conv_data, colWidths=[22*mm,28*mm,28*mm,22*mm,30*mm,44*mm])
            t_conv.setStyle(TableStyle([
                ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
                ("FONTSIZE",    (0,0), (-1,-1), 7.5),
                ("TEXTCOLOR",   (0,0), (0,-1), ORANGE),
                ("TEXTCOLOR",   (2,0), (2,-1), ORANGE),
                ("TEXTCOLOR",   (4,0), (4,-1), ORANGE),
                ("TEXTCOLOR",   (1,0), (1,-1), WHITE),
                ("TEXTCOLOR",   (3,0), (3,-1), WHITE),
                ("TEXTCOLOR",   (5,0), (5,-1), WHITE),
                ("FONTNAME",    (1,0), (1,-1), "Helvetica-Bold"),
                ("FONTNAME",    (3,0), (3,-1), "Helvetica-Bold"),
                ("FONTNAME",    (5,0), (5,-1), "Helvetica-Bold"),
                ("BACKGROUND",  (0,0), (-1,-1), DARK),
                ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#0d1117"), DARK]),
                ("GRID",        (0,0), (-1,-1), 0.2, LGRAY),
                ("TOPPADDING",  (0,0), (-1,-1), 2),
                ("BOTTOMPADDING",(0,0),(-1,-1), 2),
            ]))
            story.append(t_conv)
            story.append(Spacer(1, 3*mm))

            # Signaux actifs
            signaux = v("Signaux")
            story.append(Paragraph(
                f"<b>Convergence {row.get('Conv_Bar','')}</b>  {row.get('Convergence',0)}/6  |  "
                f"Signaux actifs: <b>{signaux}</b>  |  "
                f"Secteur: {v('Secteur')}  |  Cap: {v('Cap (G$)')}G$",
                s_small))

            story.append(HRFlowable(width="100%", thickness=0.5, color=LGRAY))

        # Pied de page
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph(
            f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}  |  "
            "AlphaScreen US — Données Yahoo Finance  |  "
            "⚠️ Ne constitue pas un conseil financier",
            s_center))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        return None
    except Exception:
        return None


def calc_position_sizing(row, ms_regime):
    """
    Calcule la recommandation de position basée sur :
    - Score total + convergence
    - Verdict d'entrée technique
    - Régime de marché
    - Upside analystes
    - Short interest (risque)
    Retourne un dict avec taille, stratégie, stop, objectif
    """
    score     = row.get("Score Total", 0)
    conv      = row.get("Convergence", 0)
    price     = row.get("Prix $", 0)
    upside    = row.get("Upside Analystes %", "N/D")
    short     = row.get("Short Interest %", "N/D")
    beta      = row.get("Beta", "N/D")
    signal    = row.get("Signal", "")

    # Convertir les valeurs
    try: upside_f = float(str(upside))
    except: upside_f = 0.0
    try: short_f = float(str(short))
    except: short_f = 5.0
    try: beta_f = float(str(beta))
    except: beta_f = 1.0
    try: price_f = float(str(price))
    except: price_f = 0.0

    # ── Taille de base selon score + convergence ──
    if score >= 85 and conv >= 5:
        base_size = 8.0
        conviction = "TRÈS HAUTE"
    elif score >= 75 and conv >= 4:
        base_size = 6.0
        conviction = "HAUTE"
    elif score >= 65 and conv >= 3:
        base_size = 4.0
        conviction = "CORRECTE"
    elif score >= 55 and conv >= 2:
        base_size = 2.5
        conviction = "MODÉRÉE"
    else:
        base_size = 1.0
        conviction = "FAIBLE"

    # ── Ajustement marché ──
    market_mult = {
        "HAUSSIER": 1.2,
        "NEUTRE":   1.0,
        "VOLATIL":  0.6,
        "BAISSIER": 0.4,
    }.get(ms_regime, 1.0)

    # ── Ajustement risque (short interest + beta) ──
    risk_mult = 1.0
    risk_notes = []
    if short_f > 15:
        risk_mult *= 0.6
        risk_notes.append("Short interest élevé — risque accru")
    elif short_f > 8:
        risk_mult *= 0.8
        risk_notes.append("Short interest modéré")
    if beta_f > 1.8:
        risk_mult *= 0.8
        risk_notes.append("Beta élevé — position réduite")

    # ── Taille finale ──
    final_size = round(base_size * market_mult * risk_mult, 1)
    final_size = max(0.5, min(final_size, 10.0))  # entre 0.5% et 10%

    # ── Stratégie d'entrée ──
    if score >= 80 and conv >= 4:
        strategie = "ENTRÉE IMMÉDIATE"
        strat_detail = "Position complète en 1 fois"
        strat_color = "#00ff88"
    elif score >= 68:
        strategie = "ENTRÉE PROGRESSIVE"
        strat_detail = "50% maintenant, 50% sur confirmation"
        strat_color = "#7DF9FF"
    elif score >= 55:
        strategie = "ATTENDRE SIGNAL"
        strat_detail = "Entrer seulement si RSI < 60 ET MACD haussier"
        strat_color = "#fbbf24"
    else:
        strategie = "SURVEILLER"
        strat_detail = "Watchlist — pas d'achat pour l'instant"
        strat_color = "#f87171"

    # ── Stop loss suggéré (basé sur beta et ATR approximatif) ──
    atr_approx = price_f * beta_f * 0.02  # ATR approximatif = 2% * beta
    stop_pct   = round(max(5.0, min(15.0, beta_f * 7)), 1)
    stop_price = round(price_f * (1 - stop_pct/100), 2) if price_f > 0 else 0

    # ── Objectif de prix ──
    if upside_f > 0 and price_f > 0:
        target_price = round(price_f * (1 + upside_f/100), 2)
        rr_ratio = round(upside_f / stop_pct, 1) if stop_pct > 0 else 0
    else:
        target_price = 0
        rr_ratio = 0

    # ── Horizon recommandé ──
    if score >= 75 and conv >= 4:
        horizon = "12-24 mois (long terme)"
    elif score >= 60:
        horizon = "6-12 mois (moyen terme)"
    else:
        horizon = "À déterminer"

    return {
        "taille":       final_size,
        "conviction":   conviction,
        "strategie":    strategie,
        "strat_detail": strat_detail,
        "strat_color":  strat_color,
        "stop_pct":     stop_pct,
        "stop_price":   stop_price,
        "target_price": target_price,
        "rr_ratio":     rr_ratio,
        "horizon":      horizon,
        "risk_notes":   risk_notes,
        "market_mult":  market_mult,
    }

# ─────────────────────────────
# 📚 HISTORIQUE DES SCORES — stocké dans GitHub
# ─────────────────────────────
import json, os, base64

HISTORY_FILENAME = "alphascreen_history.json"

def github_get_file(token, repo, filename):
    """Lit un fichier depuis GitHub. Retourne (content, sha) ou (None, None)."""
    try:
        import urllib.request
        url = f"https://api.github.com/repos/{repo}/contents/{filename}"
        req = urllib.request.Request(url, headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AlphaScreen"
        })
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        content = json.loads(base64.b64decode(data["content"]).decode("utf-8"))
        return content, data["sha"]
    except Exception:
        return None, None


def github_put_file(token, repo, filename, content, sha=None):
    """Écrit un fichier dans GitHub (create ou update)."""
    try:
        import urllib.request
        url  = f"https://api.github.com/repos/{repo}/contents/{filename}"
        body = {
            "message": f"AlphaScreen history update {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "content": base64.b64encode(
                json.dumps(content, ensure_ascii=False, indent=2).encode("utf-8")
            ).decode("utf-8"),
        }
        if sha:
            body["sha"] = sha
        data = json.dumps(body).encode("utf-8")
        req  = urllib.request.Request(url, data=data, headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
            "User-Agent": "AlphaScreen"
        }, method="PUT")
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status in (200, 201)
    except Exception:
        return False


def save_scan_to_history(df, scan_date, gh_token=None, gh_repo=None):
    """Sauvegarde les scores du scan — GitHub si token dispo, sinon /tmp/."""
    try:
        # Créer l'entrée du scan
        entry = {
            "date":      scan_date,
            "timestamp": datetime.now().isoformat(),
            "scores":    {}
        }
        for _, row in df.iterrows():
            ticker = row["Ticker"]
            entry["scores"][ticker] = {
                "score":       int(row.get("Score Total", 0)),
                "signal":      str(row.get("Signal", "")),
                "convergence": int(row.get("Convergence", 0)),
                "q":  int(row.get("Score Qualité", 0)),
                "g":  int(row.get("Score Croissance", 0)),
                "v":  int(row.get("Score Valorisation", 0)),
                "m":  int(row.get("Score Momentum", 0)),
                "c":  int(row.get("Score Conviction", 0)),
                "i":  int(row.get("Score Institutionnel", 0)),
                "prix":    float(row.get("Prix $", 0)),
                "secteur": str(row.get("Secteur", "")),
            }

        # ── GitHub ──
        if gh_token and gh_repo:
            history, sha = github_get_file(gh_token, gh_repo, HISTORY_FILENAME)
            if history is None:
                history = []
            history.append(entry)
            history = history[-24:]  # garder 6 mois (scan hebdo)
            ok = github_put_file(gh_token, gh_repo, HISTORY_FILENAME, history, sha)
            if ok:
                # Aussi sauvegarder en local comme cache
                with open("/tmp/" + HISTORY_FILENAME, "w") as f:
                    json.dump(history, f)
                return True, "github"

        # ── Fallback /tmp/ ──
        local = "/tmp/" + HISTORY_FILENAME
        history = []
        if os.path.exists(local):
            with open(local, "r") as f:
                history = json.load(f)
        history.append(entry)
        history = history[-12:]
        with open(local, "w") as f:
            json.dump(history, f)
        return True, "local"

    except Exception as e:
        return False, str(e)


def load_history(gh_token=None, gh_repo=None):
    """Charge l'historique — GitHub si token dispo, sinon /tmp/."""
    # Essayer GitHub d'abord
    if gh_token and gh_repo:
        history, _ = github_get_file(gh_token, gh_repo, HISTORY_FILENAME)
        if history:
            # Mettre en cache local
            try:
                with open("/tmp/" + HISTORY_FILENAME, "w") as f:
                    json.dump(history, f)
            except Exception:
                pass
            return history

    # Fallback cache local
    try:
        local = "/tmp/" + HISTORY_FILENAME
        if os.path.exists(local):
            with open(local, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def get_score_trend(ticker, history):
    """Retourne (scores, dates, delta, tendance) pour un ticker."""
    scores = []
    dates  = []
    for entry in history:
        if ticker in entry.get("scores", {}):
            scores.append(entry["scores"][ticker]["score"])
            dates.append(entry["date"])
    if len(scores) < 2:
        return scores, dates, 0, "NOUVEAU"
    delta = scores[-1] - scores[-2]
    if delta >= 5:    tendance = "EN HAUSSE"
    elif delta <= -5: tendance = "EN BAISSE"
    else:             tendance = "STABLE"
    return scores, dates, delta, tendance


def render_history_section(df_show, history):
    """Affiche la section historique complète."""
    if not history:
        st.info("Aucun historique — le premier scan vient d'être enregistré.")
        return

    st.markdown(f"**{len(history)} scans enregistrés** — du {history[0]['date']} au {history[-1]['date']}")

    all_tickers = set()
    for entry in history:
        all_tickers.update(entry.get("scores", {}).keys())

    current_tickers = list(df_show["Ticker"]) if not df_show.empty else []
    relevant = [t for t in current_tickers if t in all_tickers]

    if not relevant:
        st.info("Aucun titre du scan actuel n'a d'historique. Relancez la semaine prochaine.")
        return

    # ── Alertes ──
    st.markdown("#### Mouvements significatifs (variation > 5 pts)")
    alerts = []
    for ticker in relevant:
        _, _, delta, tendance = get_score_trend(ticker, history)
        if abs(delta) >= 5:
            last_score = history[-1]["scores"][ticker]["score"]
            alerts.append((ticker, delta, tendance, last_score))

    if alerts:
        alerts.sort(key=lambda x: abs(x[1]), reverse=True)
        cols = st.columns(min(len(alerts), 4))
        for idx, (ticker, delta, tendance, score) in enumerate(alerts[:4]):
            color = "#00ff88" if delta > 0 else "#f87171"
            arrow = "+" if delta > 0 else ""
            cols[idx].markdown(
                f'<div class="metric-card">'
                f'<div style="font-family:Space Mono,monospace;font-size:1.1rem;'
                f'font-weight:700;color:#e2e8f0;">{ticker}</div>'
                f'<div style="font-family:Space Mono,monospace;font-size:1.8rem;'
                f'font-weight:700;color:{color};">{arrow}{delta} pts</div>'
                f'<div style="font-size:0.75rem;color:{color};">{tendance}</div>'
                f'<div style="font-size:0.72rem;color:#64748b;">Score: {score}/100</div>'
                f'</div>', unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#64748b;font-size:0.85rem;'>Aucune variation > 5 pts depuis le dernier scan.</span>",
                    unsafe_allow_html=True)

    # ── Graphique evolution ──
    st.markdown("#### Evolution des scores dans le temps")
    top_tickers = [t for t in current_tickers[:10] if t in all_tickers]
    if top_tickers:
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            pal = ["#00ff88","#7DF9FF","#fbbf24","#a78bfa","#f97316",
                   "#f87171","#4ade80","#38bdf8","#fb923c","#e879f9"]
            for idx, ticker in enumerate(top_tickers):
                sc, dt, _, _ = get_score_trend(ticker, history)
                if len(sc) >= 2:
                    fig.add_trace(go.Scatter(
                        x=dt, y=sc, name=ticker,
                        mode="lines+markers",
                        line=dict(color=pal[idx % len(pal)], width=2),
                        marker=dict(size=7),
                    ))
            fig.add_hline(y=83, line_dash="dot",  line_color="#00ff8866", annotation_text="Excellent")
            fig.add_hline(y=68, line_dash="dash", line_color="#fbbf2466", annotation_text="Bon")
            fig.update_layout(
                height=380,
                paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                font_color="#e2e8f0",
                xaxis=dict(gridcolor="#1e3a5f", title="Date du scan"),
                yaxis=dict(gridcolor="#1e3a5f", title="Score /100", range=[0, 105]),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

    # ── Tableau comparatif ──
    if len(history) >= 2:
        st.markdown("#### Comparaison — dernier scan vs précédent")
        last = history[-1]["scores"]
        prev = history[-2]["scores"]
        rows = []
        for ticker in relevant[:25]:
            if ticker in last and ticker in prev:
                s_now  = last[ticker]["score"]
                s_prev = prev[ticker]["score"]
                delta  = s_now - s_prev
                rows.append({
                    "Ticker":          ticker,
                    "Score actuel":    s_now,
                    "Score precedent": s_prev,
                    "Delta":           delta,
                    "Tendance":        "EN HAUSSE" if delta>=5 else "EN BAISSE" if delta<=-5 else "STABLE",
                    "Signal":          last[ticker].get("signal", ""),
                })
        if rows:
            df_h = pd.DataFrame(rows).sort_values("Delta", ascending=False)
            st.dataframe(df_h, use_container_width=True, hide_index=True,
                column_config={
                    "Score actuel": st.column_config.ProgressColumn(
                        "Score actuel", min_value=0, max_value=100, format="%d"),
                    "Delta": st.column_config.NumberColumn("Delta", format="%+d"),
                })


# ─────────────────────────────
# 🤖 ANALYSE IA — GEMINI FLASH (gratuit)
# ─────────────────────────────
def gemini_analyse(df, api_key, ms, n_total, n_passed):
    """Analyse IA top 10 via Google Gemini 1.5 Flash — gratuit (1500 req/jour)."""
    top10 = df.head(10)
    lines = "\n".join(
        f"#{i+1} {r['Ticker']} ({r['Nom']}) | Score {r['Score Total']}/100 | {r['Signal']} | "
        f"Conv {r['Conv_Bar']} {r['Convergence']}/6 | "
        f"Q:{r['Score Qualité']}/25 G:{r['Score Croissance']}/20 "
        f"V:{r['Score Valorisation']}/15 M:{r['Score Momentum']}/10 "
        f"C:{r['Score Conviction']}/10 I:{r['Score Institutionnel']}/20 | "
        f"ROE:{r['ROE %']}% MgBrute:{r['Marge Brute %']}% MgNette:{r['Marge Nette %']}% | "
        f"RevG:{r['Rev. Growth %']}% EPS:{r['EPS Growth %']}% | "
        f"P/E:{r['P/E']} PEG:{r['PEG']} | "
        f"ROIC:{r['ROIC %']}% Short:{r['Short Interest %']}% Inst:{r['Inst. Ownership %']}% | "
        f"Insider:{r['Insider %']}% Upside:{r['Upside Analystes %']}%"
        for i, (_, r) in enumerate(top10.iterrows()))

    prompt = f"""Tu es un analyste en investissement long terme (style Buffett+Lynch+institutionnel).
Marché: {ms['regime']} | SPY vs MA50: {ms['spy_vs_ma50']}% | VIX: {ms['vix']} ({ms['vix_label']}) | Bonus: {ms['bonus']:+d}pts

Score /100 — 6 catégories:
• Qualité (25): ROE + Marge brute + Marge nette + FCF + Dette
• Croissance (20): Revenus + EPS annuel + EPS trimestriel
• Valorisation (15): P/E + PEG + EV/EBITDA
• Momentum (10): MA50>MA200 + Prix>MA50 + RSI + 52W
• Conviction (10): Insiders + Upside analystes + Recommandation
• Institutionnel (20): Inst.Ownership + Short Interest + ROIC + Current Ratio + P/S + Beta + Momentum relatif + Tendance marges

Univers: {n_total} titres US → {n_passed} pré-filtre → {len(df)} scorés

TOP 10:
{lines}

Analyse en 8-10 phrases:
1) Qualité générale du top 10 — convergence des 6 catégories
2) Les 3 meilleures opportunités — forces spécifiques et risques long terme
3) Ce que les signaux institutionnels révèlent (ROIC, short interest, ownership)
4) Recommandation concrète pour un investisseur long terme
Français, direct, chiffré, sans disclaimer."""

    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai non installé")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={"max_output_tokens": 1200, "temperature": 0.3}
    )
    response = model.generate_content(prompt)
    return response.text

# ─────────────────────────────
# 🚀 SIDEBAR
# ─────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("### 🌍 Univers US")

    # Mode rapide ou complet
    scan_mode = st.radio("Mode de scan",
                          ["Rapide (60 titres)", "Standard (150 titres)", "Complet (500+ titres)"],
                          index=0,
                          help="Rapide = meilleurs titres S&P500 seulement. Idéal sur mobile.")

    use_nasdaq=st.checkbox(f"Inclure Nasdaq extras ({len(NASDAQ100_EXTRA)} titres)",
                            value=False)

    # Sélection de l'univers selon le mode
    TOP_60 = [
        "AAPL","MSFT","GOOGL","AMZN","NVDA","META","BRK-B","LLY","V","JPM",
        "UNH","XOM","MA","JNJ","PG","HD","MRK","AVGO","COST","ABBV",
        "CVX","KO","PEP","ADBE","TMO","MCD","CSCO","ACN","WMT","BAC",
        "CRM","ABT","ORCL","PFE","NKE","TXN","LIN","DIS","NFLX","AMD",
        "PM","RTX","INTC","QCOM","INTU","NEE","AMGN","HON","UPS","IBM",
        "LOW","GE","SPGI","ISRG","CAT","GS","NOW","AXP","BKNG","T"
    ]
    TOP_150 = TOP_60 + [
        "GILD","MDT","BLK","DE","C","ADP","REGN","VRTX","CI","MMC",
        "ZTS","BSX","SYK","SO","DUK","PLD","CB","ELV","ETN","AON",
        "SHW","MCO","ICE","TJX","APD","ECL","CME","WM","ITW","EMR",
        "NSC","FDX","OKE","FCX","HUM","PNC","USB","MO","GD","TGT",
        "KLAC","LRCX","MCHP","ADI","CDNS","SNPS","FICO","NXPI","TEL","A",
        "ROK","DHR","RMD","EW","HCA","IQV","IDXX","DXCM","PODD","ALGN",
        "MRNA","BIIB","ILMN","VRSN","FTNT","PANW","CRWD","ZS","OKTA","DDOG",
        "WDAY","TEAM","VEEV","PAYC","ANSS","TRMB","EPAM","GDDY","AKAM","CTSH",
        "MTD","WAT","RVTY","LH","DGX","HOLX","BIO","BAX","BDX","COO"
    ]

    combined = set()
    if scan_mode == "Rapide (60 titres)":
        combined.update(TOP_60)
    elif scan_mode == "Standard (150 titres)":
        combined.update(TOP_150)
    else:
        combined.update(SP500)

    if use_nasdaq: combined.update(NASDAQ100_EXTRA)

    mode_color = "#00ff88" if "Rapide" in scan_mode else "#fbbf24" if "Standard" in scan_mode else "#f97316"
    st.markdown(f"""<div class="speed-box">
        <span style='color:{mode_color};font-weight:700;'>{len(combined)} titres</span>
        &nbsp;|&nbsp; Batch download · Cache 4h · Pré-filtre instantané<br>
        <span style='font-size:0.72rem;color:#64748b;'>
        Rapide = top blue chips | Standard = S&P500 partiel | Complet = tout l'indice
        </span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Pré-filtre automatique")
    col_p1,col_p2=st.columns(2)
    with col_p1: min_price=st.number_input("Prix min ($)",value=20.0,min_value=1.0,step=5.0)
    with col_p2: max_price=st.number_input("Prix max ($)",value=500.0,min_value=50.0,step=50.0)
    min_vol=st.number_input("Volume moyen min",value=500000,step=100000,format="%d")
    st.markdown(f"""<div style='background:#0d1a2a;border-left:3px solid #4a90d0;border-radius:6px;
        padding:8px 12px;font-size:0.78rem;color:#86c0e8;margin-top:6px;'>
        ✅ ${min_price:.0f}–${max_price:.0f} · Vol>{int(min_vol/1000)}K · Prix>MA200 · FCF+ · Cap>1G$
    </div>""",unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Pondération")
    st.markdown("""<div style='background:#001a0f;border:1px solid #00ff8833;border-radius:6px;
        padding:10px;font-size:0.82rem;color:#86efac;'>
        🏆 Qualité : <strong>25 pts</strong><br>
        &nbsp;&nbsp;ROE · Marge brute · Marge nette · FCF · Dette<br>
        📈 Croissance : <strong>20 pts</strong><br>
        &nbsp;&nbsp;Revenus · EPS annuel · EPS trimestriel<br>
        💲 Valorisation : <strong>15 pts</strong><br>
        &nbsp;&nbsp;P/E · PEG · EV/EBITDA<br>
        🚀 Momentum : <strong>10 pts</strong><br>
        &nbsp;&nbsp;MA50>MA200 · Prix>MA50 · RSI · 52W<br>
        🎯 Conviction : <strong>10 pts</strong><br>
        &nbsp;&nbsp;Insiders · Upside analystes · Recommandation<br>
        🏦 Institutionnel : <strong>20 pts</strong><br>
        &nbsp;&nbsp;Inst. Ownership · Short Interest · ROIC<br>
        &nbsp;&nbsp;Current Ratio · P/S · Beta · Momentum rel. · Tendance marges<br>
        <span style='color:#64748b;'>+ Bonus marché ±5 à ±15 pts</span>
    </div>""",unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Filtres résultats")
    min_score_f=st.slider("Score minimum",0,100,50,5)
    min_conv_f=st.selectbox("Convergence min",[0,1,2,3,4,5],index=0,
                             format_func=lambda x:f"{x}/5 catégories")
    signal_f=st.multiselect("Signaux",
                             ["🟢 EXCELLENT","🟢 BON","🟡 CORRECT","🔴 ÉVITER"],
                             default=["🟢 EXCELLENT","🟢 BON"])
    sort_f=st.selectbox("Trier par",
                         ["Score Global","Score Total","Convergence","Score Institutionnel",
                          "Score Conviction","ROIC %","Momentum Relatif %",
                          "Upside Analystes %","ROE %","Rev. Growth %"])
    filter_zone = st.checkbox("Seulement titres en zone d'achat", value=False,
                               help="Affiche uniquement les titres avec Timing = ENTRER MAINTENANT ou CONDITIONS FAVORABLES")


    st.markdown("---")
    st.markdown("### 🗄️ Base de données Supabase")
    st.markdown("""<div style='background:#0d1a2a;border:1px solid #4a90d044;border-radius:6px;
        padding:8px 12px;font-size:0.78rem;color:#86c0e8;margin-bottom:8px;'>
        Historique permanent des scores + Watchlist<br>
        <a href='https://supabase.com' target='_blank' style='color:#7DF9FF;'>supabase.com</a>
        → New project → Settings → API
    </div>""", unsafe_allow_html=True)
    sb_url = st.text_input("Supabase URL", placeholder="https://xxx.supabase.co",
                            type="default", key="sb_url")
    sb_key = st.text_input("Supabase anon key", type="password",
                            placeholder="eyJ...", key="sb_key")
    if sb_url and sb_key and DB_AVAILABLE:
        ok, msg = test_connection(sb_url, sb_key)
        if ok:
            st.success("Base de donnees connectee")
        else:
            st.error(f"Erreur: {msg}")
    elif not DB_AVAILABLE:
        st.caption("Ajoutez 'supabase>=2.0.0' dans requirements.txt")
    st.markdown("""<div style='background:#0d1a2a;border:1px solid #4a90d044;border-radius:6px;
        padding:8px 12px;font-size:0.78rem;color:#86c0e8;margin-bottom:8px;'>
        ✅ <strong>100% gratuit</strong> — Google Gemini 1.5 Flash<br>
        📋 <a href='https://aistudio.google.com' target='_blank' style='color:#7DF9FF;'>aistudio.google.com</a>
        → Get API Key · 1 500 req/jour gratuit
    </div>""", unsafe_allow_html=True)
    gemini_key = st.text_input("Clé API Google Gemini", type="password", placeholder="AIza...")
    use_ai = st.checkbox("Analyse IA du top 10 (Gemini)", value=False)

    st.markdown("---")
    nb_sc=st.slider("Threads scoring",3,20,6)
    st.markdown("""<div style='background:#0d1a2a;border:1px solid #fbbf2433;border-radius:6px;
        padding:6px 10px;font-size:0.75rem;color:#fbbf24;margin-top:4px;'>
        Reseau mobile: 3-5 threads | WiFi: 6-10 threads
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    run=st.button("▶ Lancer le scan",use_container_width=True)
    st.markdown("<div style='color:#64748b;font-size:0.75rem;'>AlphaScreen US v4 — 6 catégories · Gemini IA</div>",
                unsafe_allow_html=True)

# ─────────────────────────────
# 🚀 MAIN
# ─────────────────────────────
st.markdown("# 📊 AlphaScreen US — Top 10 Opportunités")
st.markdown("<div style='color:#64748b;margin-bottom:1rem;'>Style Buffett+Lynch · Qualité · Croissance · Valorisation · Momentum · Conviction · Score /100</div>",
            unsafe_allow_html=True)

tickers_list=sorted(list(combined)) if combined else []

# Marché global
with st.spinner("Analyse marché global..."):
    ms=get_market_status()

st.markdown(f"""<div class="market-banner" style="background:{ms['color']}11;border:1px solid {ms['color']}44;border-left:5px solid {ms['color']};">
    <strong style="color:{ms['color']};font-size:1.1rem;">{ms['emoji']} MARCHÉ {ms['regime']}</strong>
    &nbsp;|&nbsp; SPY vs MA50: <strong>{ms['spy_vs_ma50']:+.1f}%</strong>
    &nbsp;|&nbsp; QQQ vs MA50: <strong>{ms['qqq_vs_ma50']:+.1f}%</strong>
    &nbsp;|&nbsp; VIX: <strong>{ms['vix']}</strong> ({ms['vix_label']})
    &nbsp;|&nbsp; RSI SPY: <strong>{ms['spy_rsi']}</strong>
    &nbsp;|&nbsp; Bonus score: <strong style="color:{ms['color']};">{ms['bonus']:+d} pts</strong>
</div>""",unsafe_allow_html=True)

c1,c2,c3,c4,c5=st.columns(5)
for col,val,label,color in[
    (c1,len(tickers_list),"Titres uniques","#e2e8f0"),
    (c2,"6","Catégories score","#00ff88"),
    (c3,"100","Points maximum","#00ff88"),
    (c4,"5 ans","Fondamentaux","#7DF9FF"),
    (c5,ms['regime'],"Régime marché",ms['color']),
]:
    col.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color};">{val}</div><div class="metric-label">{label}</div></div>',
                 unsafe_allow_html=True)

# Analyse sectorielle au démarrage
st.markdown("---")
st.markdown("### 🏭 Force Sectorielle")
with st.spinner("Analyse des 11 secteurs..."):
    sector_preview = get_sector_strength()
if sector_preview and sector_preview["ranked"]:
    sec_cols = st.columns(11)
    for idx,(sec,data) in enumerate(sector_preview["ranked"]):
        color = "#00ff88" if sec in sector_preview["top3"] else \
                "#f87171" if sec in sector_preview["bottom3"] else "#fbbf24"
        icon  = "🔥" if sec in sector_preview["top3"] else \
                "❄️" if sec in sector_preview["bottom3"] else "~"
        short = sec.replace("Consumer ","Conso. ").replace(" Services","").replace("Basic ","")
        sec_cols[idx].markdown(
            f'<div style="background:#0d1117;border:1px solid {color}55;border-radius:8px;'
            f'padding:8px 4px;text-align:center;">'
            f'<div style="font-size:1rem;">{icon}</div>'
            f'<div style="color:#94a3b8;font-size:0.65rem;margin:2px 0;">{short}</div>'
            f'<div style="color:{color};font-family:Space Mono,monospace;font-size:0.85rem;font-weight:700;">'
            f'{data["perf_1m"]:+.1f}%</div>'
            f'<div style="color:#64748b;font-size:0.6rem;">1 mois</div>'
            f'<div style="color:{color};font-size:0.7rem;">{data["perf_3m"]:+.1f}% / 3M</div>'
            f'</div>',
            unsafe_allow_html=True)

st.markdown("---")

# Description système
with st.expander("📖 Philosophie AlphaScreen — 6 catégories Buffett+Lynch+Institutionnel",expanded=False):
    ca,cb_,cc,cd,ce,cf=st.columns(6)
    for col,title,pts,color,items in[
        (ca,"🏆 QUALITÉ","25 pts","#00ff88",
         ["ROE > 15%","Marge brute > 35%","Marge nette > 10%","FCF positif","D/E < 1"]),
        (cb_,"📈 CROISSANCE","20 pts","#7DF9FF",
         ["Revenus +8%/an","EPS annuel +10%","EPS trimestriel consistant"]),
        (cc,"💲 VALORISATION","15 pts","#fbbf24",
         ["P/E < 25","PEG < 1.5","EV/EBITDA < 18"]),
        (cd,"🚀 MOMENTUM","10 pts","#a78bfa",
         ["MA50 > MA200","Prix > MA50","RSI 45-70","< 15% du 52W"]),
        (ce,"🎯 CONVICTION","10 pts","#f97316",
         ["Insiders > 5%","Upside > 20%","Consensus Buy"]),
        (cf,"🏦 INSTITUTIONNEL","20 pts","#38bdf8",
         ["Inst. Ownership > 50%","Short Interest < 5%","ROIC > 15%","Current Ratio > 1.5","Beta équilibré","Momentum rel. vs SPY","Tendance marges","P/S raisonnable"]),
    ]:
        col.markdown(f"""<div style='background:#0d1a2a;border:1px solid {color}33;border-radius:8px;padding:12px;'>
            <strong style='color:{color};'>{title}</strong><br>
            <span style='color:#64748b;font-size:0.78rem;'>{pts}</span><br><br>
            {"<br>".join(f"<span style='font-size:0.78rem;'>• {i}</span>" for i in items)}
        </div>""",unsafe_allow_html=True)

st.markdown("---")

# ── CONSEILLER ──
st.markdown("---")
st.markdown("## 🎯 Conseiller — Analyse & Recommandation")
with st.expander("Obtenir une recommandation personnalisée", expanded=False):

    co1, co2, co3 = st.columns(3)
    with co1:
        cons_ticker  = st.text_input("Ticker", placeholder="Ex: PFE, V, GOOGL",
                                      key="cons_ticker").strip().upper()
    with co2:
        cons_prix    = st.number_input("Prix actuel ($)", min_value=0.01,
                                        value=100.0, step=0.01, key="cons_prix")
    with co3:
        cons_budget  = st.number_input("Budget ($)", min_value=100.0,
                                        value=5000.0, step=100.0, key="cons_budget")

    cons_run = st.button("Analyser et conseiller", key="btn_conseiller",
                          use_container_width=True)

    if cons_run and cons_ticker:
        with st.spinner(f"Analyse complète de {cons_ticker}..."):
            try:
                import time

                # ── Télécharger données ──
                raw_c = yf.download(cons_ticker, period="1y",
                                     auto_adjust=True, progress=False)
                if isinstance(raw_c.columns, pd.MultiIndex):
                    raw_c.columns = raw_c.columns.get_level_values(0)
                raw_c = raw_c[["Close","Volume","High","Low","Open"]].dropna()

                if raw_c.empty:
                    st.error(f"Ticker '{cons_ticker}' introuvable.")
                    st.stop()

                # Fondamentaux
                t_c   = yf.Ticker(cons_ticker)
                info_c = {}
                for attempt in range(3):
                    try:
                        info_c = t_c.info
                        if info_c and len(info_c) > 5: break
                    except Exception:
                        time.sleep(1.0*(attempt+1))

                # Si info_c vide — utiliser des valeurs par défaut
                if not info_c:
                    info_c = {}

                name_c   = info_c.get("shortName", cons_ticker)
                sector_c = info_c.get("sector", "N/A")
                pe_c     = info_c.get("trailingPE", None)
                target_c = info_c.get("targetMeanPrice", None)
                beta_c   = info_c.get("beta", 1.0) or 1.0

                # ── Fibonacci ──
                close_c  = raw_c["Close"]
                high_c   = raw_c["High"]
                low_c    = raw_c["Low"]
                n_look   = min(90, len(raw_c)-1)
                fib_hi_c = float(high_c.iloc[-n_look:].max())
                fib_lo_c = float(low_c.iloc[-n_look:].min())
                fib_rng  = fib_hi_c - fib_lo_c

                fib_c = {
                    "0.236": fib_hi_c - 0.236 * fib_rng,
                    "0.382": fib_hi_c - 0.382 * fib_rng,
                    "0.500": fib_hi_c - 0.500 * fib_rng,
                    "0.618": fib_hi_c - 0.618 * fib_rng,
                    "0.786": fib_hi_c - 0.786 * fib_rng,
                }

                # Niveau Fib actuel
                prix_now = cons_prix
                fib_nearest_c = min(fib_c.items(), key=lambda x: abs(x[1]-prix_now))
                fib_name_c    = fib_nearest_c[0]
                fib_val_c     = fib_nearest_c[1]
                fib_dist_c    = abs(prix_now - fib_val_c) / prix_now * 100

                # ── Indicateurs techniques ──
                ma50_c  = float(close_c.rolling(50).mean().iloc[-1])
                ma200_c = float(close_c.rolling(min(200,len(close_c))).mean().iloc[-1])
                d_c = close_c.diff()
                g_c = d_c.where(d_c>0,0).rolling(14).mean()
                l_c = -d_c.where(d_c<0,0).rolling(14).mean()
                rsi_c = float(100-(100/(1+g_c/l_c.clip(lower=1e-10))).iloc[-1])
                ema12 = close_c.ewm(span=12,adjust=False).mean()
                ema26 = close_c.ewm(span=26,adjust=False).mean()
                macd_c = float((ema12-ema26).iloc[-1])
                macd_sig = float((ema12-ema26).ewm(span=9,adjust=False).mean().iloc[-1])
                macd_pos = macd_c > macd_sig

                # ── Verdict ──
                # Score entrée
                escore = 0
                if prix_now <= fib_c["0.618"] * 1.02:  escore += 5
                elif prix_now <= fib_c["0.500"] * 1.02: escore += 3
                elif prix_now <= fib_c["0.382"] * 1.02: escore += 2
                if 40 <= rsi_c <= 65:  escore += 3
                elif rsi_c < 40:       escore += 4
                elif rsi_c > 75:       escore -= 2
                if macd_pos:           escore += 3
                else:                  escore -= 1
                if prix_now > ma50_c:  escore += 1
                else:                  escore -= 1

                if escore >= 8:
                    verdict_c = "ACHETER MAINTENANT"
                    vc_color  = "#00ff88"
                    vc_bg     = "#001a0f"
                    vc_icon   = "✅"
                elif escore >= 4:
                    verdict_c = "CONDITIONS FAVORABLES — ATTENDRE MACD"
                    vc_color  = "#7DF9FF"
                    vc_bg     = "#001a1f"
                    vc_icon   = "🔵"
                elif escore >= 1:
                    verdict_c = "ATTENDRE MEILLEUR POINT D'ENTREE"
                    vc_color  = "#fbbf24"
                    vc_bg     = "#1a1400"
                    vc_icon   = "⏳"
                else:
                    verdict_c = "EVITER — REPLI EN COURS"
                    vc_color  = "#f87171"
                    vc_bg     = "#1a0000"
                    vc_icon   = "🚫"

                # ── Prix d'entrée optimal ──
                # Trouver la meilleure confluence sous le prix actuel
                entry_optimal = None
                entry_label   = ""
                for fname in ["0.382","0.500","0.618"]:
                    fval = fib_c[fname]
                    if fval < prix_now * 1.01:
                        dist_ma50 = abs(fval - ma50_c) / fval * 100
                        if dist_ma50 < 3.0:
                            entry_optimal = round((fval + ma50_c) / 2, 2)
                            entry_label   = f"Fib {fname} + MA50"
                            break
                        else:
                            entry_optimal = round(fval, 2)
                            entry_label   = f"Fib {fname}"
                            break

                if entry_optimal is None:
                    entry_optimal = round(prix_now, 2)
                    entry_label   = "Prix actuel (deja en zone)"

                # Si déjà dans zone optimale
                if prix_now <= fib_c["0.618"] * 1.02:
                    entry_optimal = round(prix_now, 2)
                    entry_label   = f"Prix actuel — Zone optimale Fib 0.618"

                # ── Stop loss & Objectif ──
                stop_pct    = round(max(5.0, min(15.0, beta_c * 7)), 1)
                stop_price  = round(entry_optimal * (1 - stop_pct/100), 2)
                objectif    = round(target_c, 2) if target_c else round(fib_hi_c, 2)
                upside_pct  = round((objectif - entry_optimal) / entry_optimal * 100, 1)
                rr_ratio    = round(upside_pct / stop_pct, 1) if stop_pct > 0 else 0

                # ── Nombre d'actions ──
                nb_actions  = int(cons_budget / entry_optimal) if entry_optimal > 0 else 0
                cout_total  = round(nb_actions * entry_optimal, 2)
                perte_max   = round(nb_actions * (entry_optimal - stop_price), 2)
                gain_cible  = round(nb_actions * (objectif - entry_optimal), 2)

                # ── Affichage verdict principal ──
                st.markdown(f"""<div style='background:{vc_bg};border:3px solid {vc_color};
                    border-radius:14px;padding:20px 24px;margin:10px 0;'>
                    <div style='font-family:Space Mono,monospace;font-size:1.3rem;
                    font-weight:700;color:{vc_color};margin-bottom:8px;'>
                    {vc_icon} {verdict_c}</div>
                    <div style='color:#e2e8f0;font-size:0.9rem;'>
                    <strong>{cons_ticker}</strong> — {name_c} · {sector_c}
                    &nbsp;|&nbsp; Prix actuel: <strong>{prix_now:.2f}</strong>
                    &nbsp;|&nbsp; Fib: <strong>{fib_name_c}</strong> ({fib_dist_c:.1f}% ecart)
                    &nbsp;|&nbsp; RSI: <strong>{rsi_c:.1f}</strong>
                    &nbsp;|&nbsp; MACD: <strong style='color:{"#00ff88" if macd_pos else "#f87171"}'>
                    {"positif" if macd_pos else "negatif"}</strong>
                    </div>
                </div>""", unsafe_allow_html=True)

                # ── 4 métriques clés ──
                ma1,ma2,ma3,ma4 = st.columns(4)
                for col,val,label,color in [
                    (ma1, f"{entry_optimal}", "Prix d'entree optimal", vc_color),
                    (ma2, f"{stop_price}", "Stop loss", "#f87171"),
                    (ma3, f"{objectif}", "Objectif", "#00ff88"),
                    (ma4, f"{rr_ratio}:1", "Ratio R/R", "#fbbf24"),
                ]:
                    col.markdown(
                        f'<div class="metric-card">'
                        f'<div style="font-family:Space Mono,monospace;font-size:1.3rem;'
                        f'font-weight:700;color:{color};">{val}</div>'
                        f'<div style="font-size:0.72rem;color:#64748b;">{label}</div>'
                        f'</div>', unsafe_allow_html=True)

                # ── Plan avec budget ──
                st.markdown(f"""<div style='background:#0d1a2a;border:1px solid #00ff8844;
                    border-left:4px solid #00ff88;border-radius:10px;
                    padding:16px 20px;margin:10px 0;'>
                    <strong style='color:#00ff88;font-family:Space Mono,monospace;'>
                    PLAN AVEC {cons_budget:.0f}</strong><br><br>
                    <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;'>
                        <div style='background:#0a1117;border-radius:8px;padding:10px;text-align:center;'>
                            <div style='color:#64748b;font-size:0.7rem;'>ACTIONS</div>
                            <div style='color:#e2e8f0;font-size:1.4rem;font-weight:700;
                            font-family:Space Mono,monospace;'>{nb_actions}</div>
                            <div style='color:#64748b;font-size:0.68rem;'>a {entry_optimal}</div>
                        </div>
                        <div style='background:#0a1117;border-radius:8px;padding:10px;text-align:center;'>
                            <div style='color:#64748b;font-size:0.7rem;'>COUT REEL</div>
                            <div style='color:#e2e8f0;font-size:1.4rem;font-weight:700;
                            font-family:Space Mono,monospace;'>{cout_total}</div>
                            <div style='color:#64748b;font-size:0.68rem;'>sur {cons_budget:.0f} budget</div>
                        </div>
                        <div style='background:#0a1117;border-radius:8px;padding:10px;text-align:center;'>
                            <div style='color:#64748b;font-size:0.7rem;'>PERTE MAX</div>
                            <div style='color:#f87171;font-size:1.4rem;font-weight:700;
                            font-family:Space Mono,monospace;'>-{perte_max}</div>
                            <div style='color:#64748b;font-size:0.68rem;'>si stop touche</div>
                        </div>
                        <div style='background:#0a1117;border-radius:8px;padding:10px;text-align:center;'>
                            <div style='color:#64748b;font-size:0.7rem;'>GAIN CIBLE</div>
                            <div style='color:#00ff88;font-size:1.4rem;font-weight:700;
                            font-family:Space Mono,monospace;'>+{gain_cible}</div>
                            <div style='color:#64748b;font-size:0.68rem;'>si objectif atteint</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # ── Explication en langage naturel ──
                # Situation Fibonacci
                if prix_now <= fib_c["0.618"] * 1.02:
                    fib_explain = f"Le prix est sous le Golden Ratio 0.618 ({fib_c['0.618']:.2f}) — zone où les institutionnels achètent historiquement."
                elif prix_now <= fib_c["0.500"] * 1.02:
                    fib_explain = f"Le prix est sur le support psychologique 0.500 ({fib_c['0.500']:.2f}) — mi-chemin du dernier mouvement."
                elif prix_now <= fib_c["0.382"] * 1.02:
                    fib_explain = f"Le prix est sur le premier support Fibonacci 0.382 ({fib_c['0.382']:.2f})."
                elif prix_now <= fib_c["0.236"] * 1.02:
                    fib_explain = f"Le prix est au niveau 0.236 — encore haut, peu de correction. Les meilleures zones sont plus bas: 0.382 ({fib_c['0.382']:.2f}), 0.618 ({fib_c['0.618']:.2f})."
                else:
                    fib_explain = f"Le prix est au-dessus de tous les niveaux Fibonacci — en pleine tendance haussière. Attendre une correction vers 0.382 ({fib_c['0.382']:.2f})."

                macd_explain = "Le MACD est positif — momentum haussier confirmé." if macd_pos else "Le MACD est négatif — attends qu'il redevienne positif avant d'entrer."
                rsi_explain  = f"RSI à {rsi_c:.0f} — {'zone idéale pour acheter.' if 40<=rsi_c<=65 else 'suracheté, risque de repli.' if rsi_c>70 else 'bas, potentiel de rebond.'}"

                if escore >= 8:
                    action_explain = f"Toutes les conditions sont réunies. Achète {nb_actions} actions à {entry_optimal} pour un coût de {cout_total}. Place ton stop à {stop_price} et ton objectif à {objectif}."
                elif escore >= 4:
                    action_explain = f"La zone est bonne mais le MACD doit confirmer. Place une alerte à {entry_optimal} et entre quand le MACD redevient positif."
                elif prix_now > fib_c["0.382"]:
                    action_explain = f"Le titre n'est pas encore en zone d'achat optimale. Attends une correction vers {fib_c['0.382']:.2f} (Fib 0.382) ou mieux {fib_c['0.618']:.2f} (Fib 0.618)."
                else:
                    action_explain = f"Le titre est en repli. Attends la stabilisation et un MACD positif avant d'entrer."

                st.markdown(f"""<div style='background:#0a1628;border:1px solid #4a90d044;
                    border-left:4px solid #4a90d0;border-radius:10px;
                    padding:16px 20px;margin:10px 0;font-size:0.88rem;line-height:1.7;'>
                    <strong style='color:#4a90d0;font-family:Space Mono,monospace;'>
                    ANALYSE EN LANGAGE CLAIR</strong><br><br>
                    <span style='color:#e2e8f0;'>
                    <strong>Fibonacci:</strong> {fib_explain}<br><br>
                    <strong>Momentum:</strong> {macd_explain} {rsi_explain}<br><br>
                    <strong>MA50:</strong> {"Le prix est au-dessus de la MA50 — tendance court terme positive." if prix_now > ma50_c else f"Le prix est sous la MA50 ({ma50_c:.2f}) — tendance court terme négative."}<br><br>
                    <strong style='color:{vc_color};'>Recommandation:</strong>
                    <span style='color:{vc_color};'> {action_explain}</span>
                    </span>
                </div>""", unsafe_allow_html=True)

                # ── Analyse Gemini si disponible ──
                if use_ai and gemini_key:
                    with st.spinner("Gemini analyse..."):
                        try:
                            prompt_cons = f"""Tu es un conseiller en investissement long terme.

Titre: {cons_ticker} ({name_c}) — Secteur: {sector_c}
Prix actuel: {prix_now} | Fibonacci: {fib_name_c} ({fib_dist_c:.1f}% ecart)
RSI: {rsi_c:.1f} | MACD: {"positif" if macd_pos else "negatif"}
MA50: {ma50_c:.2f} | MA200: {ma200_c:.2f}
P/E: {pe_c} | Beta: {beta_c}
Entree optimale: {entry_optimal} | Stop: {stop_price} | Objectif: {objectif}
Ratio R/R: {rr_ratio}:1 | Budget: {cons_budget} | Actions: {nb_actions}
Verdict systeme: {verdict_c}

En 4-5 phrases en francais, donne:
1) Analyse de la situation actuelle
2) Risques specifiques a ce titre
3) Confirmation ou nuance du verdict
Direct, chiffre, sans disclaimer."""

                            genai.configure(api_key=gemini_key)
                            model_g = genai.GenerativeModel(
                                model_name="gemini-2.0-flash",
                                generation_config={"max_output_tokens": 400, "temperature": 0.3}
                            )
                            resp = model_g.generate_content(prompt_cons)
                            st.markdown(
                                f'<div class="ai-analysis-box">'
                                f'<span style="color:#00ff88;font-family:Space Mono,monospace;'
                                f'font-weight:700;">ANALYSE GEMINI</span><br><br>'
                                f'{resp.text}</div>',
                                unsafe_allow_html=True)
                        except Exception as e:
                            st.caption(f"Gemini non disponible: {str(e)[:50]}")

            except Exception as e:
                st.error(f"Erreur: {str(e)[:100]}")

    elif cons_run and not cons_ticker:
        st.warning("Entre un ticker valide.")

# ── ANALYSE INDIVIDUELLE ──
st.markdown("---")
st.markdown("## 🔍 Analyse Individuelle d'un Titre")
with st.expander("Analyser un titre spécifique", expanded=False):
    col_ti1, col_ti2 = st.columns([2,1])
    with col_ti1:
        indiv_ticker = st.text_input(
            "Ticker (ex: V, AAPL, MSFT, INCY)",
            placeholder="Entrez un ticker US...",
            key="indiv_ticker"
        ).strip().upper()
    with col_ti2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_indiv = st.button("Analyser", key="btn_indiv",
                              use_container_width=True)

    if run_indiv and indiv_ticker:
        with st.spinner(f"Analyse de {indiv_ticker} en cours..."):
            try:
                # ── Télécharger les données ──
                indiv_raw = yf.download(indiv_ticker, period="1y",
                                         auto_adjust=True, progress=False)
                if indiv_raw.empty:
                    st.error(f"Ticker '{indiv_ticker}' introuvable.")
                    st.stop()

                # Normaliser les colonnes (single ticker peut avoir MultiIndex)
                if isinstance(indiv_raw.columns, pd.MultiIndex):
                    indiv_raw.columns = indiv_raw.columns.get_level_values(0)
                indiv_raw = indiv_raw[["Close","Volume","High","Low","Open"]].dropna()

                indiv_info = {}
                for attempt in range(3):
                    try:
                        indiv_info = yf.Ticker(indiv_ticker).info
                        if indiv_info and len(indiv_info) > 5:
                            break
                    except Exception:
                        import time; time.sleep(0.5)

                if not indiv_info:
                    st.error("Impossible de récupérer les fondamentaux.")
                    st.stop()

                # ── Calcul du score via score_ticker ──
                # Préparer pre_data
                indiv_df    = indiv_raw.copy()
                indiv_price = float(indiv_df["Close"].iloc[-1])
                indiv_ma200 = float(indiv_df["Close"].rolling(min(200,len(indiv_df))).mean().iloc[-1])
                indiv_pre   = {
                    "df":      indiv_df,
                    "price":   indiv_price,
                    "ma200":   indiv_ma200,
                    "avg_vol": float(indiv_df["Volume"].rolling(30).mean().iloc[-1])
                }

                # Charger SPY pour momentum relatif
                try:
                    spy_indiv = yf.Ticker("SPY").history(period="1y")
                except Exception:
                    spy_indiv = None

                result = score_ticker(indiv_ticker, indiv_pre,
                                      market_bonus=ms.get("bonus",0),
                                      spy_hist=spy_indiv,
                                      sector_data=None)

                if not result:
                    st.warning(f"{indiv_ticker} ne passe pas les filtres (FCF négatif ou cap trop faible).")
                    st.stop()

                st.success(f"Analyse complète de {indiv_ticker} — {result['Nom']}")

                # ── Affichage du score ──
                score_i  = result["Score Total"]
                signal_i = result["Signal"]
                conv_i   = result["Convergence"]

                sc_color = "#00ff88" if "EXCELLENT" in signal_i or "BON" in signal_i \
                           else "#fbbf24" if "CORRECT" in signal_i else "#f87171"

                st.markdown(f"""<div style='background:linear-gradient(135deg,#0d1a2a,#111827);
                    border:2px solid {sc_color}55;border-radius:12px;
                    padding:16px 20px;margin:8px 0;'>
                    <span style='font-family:Space Mono,monospace;font-size:1.4rem;
                    font-weight:700;color:#e2e8f0;'>{indiv_ticker}</span>
                    <span style='color:#64748b;font-size:0.85rem;margin-left:10px;'>
                    {result["Nom"]} · {result["Secteur"]} · {result["Cap (G$)"]}G$</span>
                    <div style='display:flex;align-items:center;gap:16px;margin-top:8px;'>
                        <span style='font-family:Space Mono,monospace;font-size:2.5rem;
                        font-weight:700;color:{sc_color};'>{score_i}</span>
                        <span style='font-size:1rem;color:#64748b;'>/100</span>
                        <span style='color:{sc_color};font-size:1rem;font-weight:700;'>
                        {signal_i}</span>
                        <span style='color:#94a3b8;font-size:0.9rem;'>
                        Conv. {result["Conv_Bar"]} {conv_i}/6</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                # ── Scores 6 catégories ──
                c1i,c2i,c3i,c4i,c5i,c6i,c7i = st.columns(7)
                for col,val,label,maxv,color in [
                    (c1i, score_i,                    "Total",         100, sc_color),
                    (c2i, result["Score Qualité"],     "Qualité",        25, "#00ff88"),
                    (c3i, result["Score Croissance"],  "Croissance",     20, "#7DF9FF"),
                    (c4i, result["Score Valorisation"],"Valorisation",   15, "#fbbf24"),
                    (c5i, result["Score Momentum"],    "Momentum",       10, "#a78bfa"),
                    (c6i, result["Score Conviction"],  "Conviction",     10, "#f97316"),
                    (c7i, result["Score Institutionnel"],"Institutionnel",20,"#38bdf8"),
                ]:
                    col.markdown(
                        f'<div class="metric-card">'
                        f'<div style="font-family:Space Mono,monospace;font-size:1.3rem;'
                        f'font-weight:700;color:{color};">{val}</div>'
                        f'<div style="font-size:0.65rem;color:#64748b;">/{maxv}<br>{label}</div>'
                        f'</div>', unsafe_allow_html=True)

                # ── Métriques clés ──
                st.markdown("---")
                mk1,mk2,mk3,mk4,mk5,mk6,mk7,mk8 = st.columns(8)
                for col,val,label in [
                    (mk1, f"{result['Prix $']}", "Prix $"),
                    (mk2, f"{result['ROE %']}%", "ROE"),
                    (mk3, f"{result['Marge Brute %']}%", "Marge brute"),
                    (mk4, f"{result['P/E']}", "P/E"),
                    (mk5, f"{result['ROIC %']}%", "ROIC"),
                    (mk6, f"{result['Short Interest %']}%", "Short Int."),
                    (mk7, f"{result['Upside Analystes %']}%", "Upside"),
                    (mk8, f"{result['RSI']}", "RSI"),
                ]:
                    col.metric(label, val)

                # ── Fondamentaux 5 ans ──
                st.markdown(f"""<div style='background:#0d1a2a;border:1px solid #7DF9FF33;
                    border-left:3px solid #7DF9FF;border-radius:6px;
                    padding:10px 14px;margin:6px 0;font-size:0.82rem;'>
                    <strong style='color:#7DF9FF;'>Fondamentaux 5 ans</strong>
                    &nbsp;|&nbsp; Rev. CAGR: <strong>{result.get("Rev. Growth % (5Y CAGR)","N/D")}</strong>
                    &nbsp;|&nbsp; EPS CAGR: <strong>{result.get("EPS Growth % (5Y CAGR)","N/D")}</strong>
                    &nbsp;|&nbsp; Consistance: <strong>{result.get("Rev. Consistance","N/D")}</strong>
                    &nbsp;|&nbsp; FCF 5Y: <strong>{result.get("FCF 5Y","N/D")}</strong>
                    &nbsp;|&nbsp; Secteur: <strong style='color:#fbbf24;'>
                    {result.get("Secteur Force","N/D")}</strong>
                </div>""", unsafe_allow_html=True)

                # ── Recommandation de position ──
                ms_current = get_market_status()
                ps_i = calc_position_sizing(result, ms_current.get("regime","NEUTRE"))
                ps_colors_i = {"TRÈS HAUTE":"#00ff88","HAUTE":"#7DF9FF",
                               "CORRECTE":"#fbbf24","MODÉRÉE":"#f97316","FAIBLE":"#f87171"}
                psc = ps_colors_i.get(ps_i["conviction"], "#64748b")
                ps_stop_i   = "soit " + str(ps_i["stop_price"]) if ps_i["stop_price"] > 0 else "N/D"
                ps_target_i = str(ps_i["target_price"]) if ps_i["target_price"] > 0 else "N/D"
                ps_rr_i     = str(ps_i["rr_ratio"]) + ":1" if ps_i["rr_ratio"] > 0 else "N/D"

                st.markdown(f"""<div style='background:#0a1f15;border:2px solid {psc}44;
                    border-radius:10px;padding:14px 18px;margin:8px 0;'>
                    <strong style='color:{psc};font-family:Space Mono,monospace;'>
                    RECOMMANDATION DE POSITION</strong>
                    <span style='background:{psc}22;border:1px solid {psc}44;
                    border-radius:6px;padding:2px 10px;font-size:0.8rem;
                    color:{psc};margin-left:10px;'>Conviction {ps_i["conviction"]}</span>
                    <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;
                    gap:10px;margin-top:10px;'>
                        <div style='background:#0a1117;border-radius:8px;
                        padding:8px;text-align:center;'>
                            <div style='color:#64748b;font-size:0.7rem;'>TAILLE</div>
                            <div style='color:{psc};font-size:1.4rem;font-weight:700;
                            font-family:Space Mono,monospace;'>{ps_i["taille"]}%</div>
                        </div>
                        <div style='background:#0a1117;border-radius:8px;
                        padding:8px;text-align:center;'>
                            <div style='color:#64748b;font-size:0.7rem;'>STOP LOSS</div>
                            <div style='color:#f87171;font-size:1.4rem;font-weight:700;
                            font-family:Space Mono,monospace;'>{ps_i["stop_pct"]}%</div>
                            <div style='color:#f87171;font-size:0.68rem;'>{ps_stop_i}</div>
                        </div>
                        <div style='background:#0a1117;border-radius:8px;
                        padding:8px;text-align:center;'>
                            <div style='color:#64748b;font-size:0.7rem;'>OBJECTIF</div>
                            <div style='color:#00ff88;font-size:1.4rem;font-weight:700;
                            font-family:Space Mono,monospace;'>{ps_target_i}</div>
                        </div>
                        <div style='background:#0a1117;border-radius:8px;
                        padding:8px;text-align:center;'>
                            <div style='color:#64748b;font-size:0.7rem;'>RATIO R/R</div>
                            <div style='color:#fbbf24;font-size:1.4rem;font-weight:700;
                            font-family:Space Mono,monospace;'>{ps_rr_i}</div>
                        </div>
                    </div>
                    <div style='margin-top:8px;background:{ps_i["strat_color"]}11;
                    border:1px solid {ps_i["strat_color"]}33;border-radius:6px;
                    padding:8px 12px;'>
                        <span style='color:{ps_i["strat_color"]};font-weight:700;'>
                        {ps_i["strategie"]}</span>
                        <span style='color:#94a3b8;font-size:0.82rem;'>
                        — {ps_i["strat_detail"]}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                # ── Graphique d'entrée complet avec Fibonacci ──
                st.markdown("---")
                st.markdown(f"#### Analyse graphique — {indiv_ticker}")
                render_entry_dashboard(indiv_ticker, indiv_pre)

            except Exception as e:
                st.error(f"Erreur lors de l'analyse de {indiv_ticker}: {str(e)[:100]}")

    elif run_indiv and not indiv_ticker:
        st.warning("Entre un ticker valide (ex: V, AAPL, MSFT)")

# ── BACKTEST ──
st.markdown("## 📈 Backtest Long Terme — Validation du Système")
with st.expander("🔬 Lancer un backtest long terme",expanded=False):
    bt1,bt2=st.columns(2)
    with bt1:
        bt_raw=st.text_area("Tickers (un par ligne)",
                            value="AAPL\nMSFT\nV\nMA\nJNJ\nKO\nJPM\nHD\nMCD\nNKE\nGOOGL\nNVDA",
                            height=180)
        bt_tickers=[t.strip().upper() for t in bt_raw.strip().split("\n") if t.strip()]
    with bt2:
        horizon_bt = st.selectbox("Horizon de detention", [6, 12], index=0,
                                   format_func=lambda x: f"{x} mois")
        st.markdown(f"""<div class="speed-box">
            📅 5 ans de donnees historiques<br>
            Scan mensuel — score >= 60 pour entrer<br>
            Horizon fixe: {horizon_bt} mois (style Buffett)<br>
            Performance reelle mesuree — pas de stop ATR<br>
            Win = perf > +5% | Loss = perf < -5%<br>
            Win Rate > 60% = systeme viable long terme
        </div>""",unsafe_allow_html=True)

    if st.button("Lancer le Backtest Long Terme",key="bt_run"):
        with st.spinner(f"Batch download {len(bt_tickers)} tickers (5 ans)..."):
            bt_prices=batch_download_backtest(tuple(sorted(bt_tickers)))
        if not bt_prices:
            st.error("Telechargement echoue.")
        else:
            bp=st.progress(0); bs=st.empty(); all_trades=[]
            for i,(t,h) in enumerate(bt_prices.items()):
                bp.progress((i+1)/len(bt_prices))
                bs.markdown(f"Backtest `{t}` ({i+1}/{len(bt_prices)})...")
                trades=backtest_long_terme(t, h, horizon_months=horizon_bt)
                if trades: all_trades.extend(trades)
            bs.empty()
            df_bt=pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
            if df_bt.empty:
                st.warning("Aucune periode trouvee avec score >= 60 ou Score Global >= 63. Essayez AAPL, MSFT, V, MA, GOOGL, NVDA.")
            else:
                s=bt_stats(df_bt)
                st.markdown(f"### Resultats — {s['total']} periodes simulees ({horizon_bt} mois chacune)")

                # Metriques principales
                wrc="#00ff88" if s["win_rate"]>=60 else "#fbbf24" if s["win_rate"]>=50 else "#f87171"
                arc="#00ff88" if s.get("annual_return",0)>=10 else "#fbbf24" if s.get("annual_return",0)>=0 else "#f87171"
                m1,m2,m3,m4,m5,m6,m7=st.columns(7)
                for col,val,label,color in[
                    (m1,f"{s['win_rate']}%","Win Rate",wrc),
                    (m2,f"{s.get('annual_return',0)}%","Rendement annualisé",arc),
                    (m3,f"+{s['avg_win']}%","Perf moy. WIN","#00ff88"),
                    (m4,f"{s['avg_loss']}%","Perf moy. LOSS","#f87171"),
                    (m5,f"{s['expectancy']}%","Expectancy","#e2e8f0"),
                    (m6,f"{s.get('avg_drawdown',0)}%","Drawdown moy.","#fbbf24"),
                    (m7,f"{s['profit_factor']}","Profit Factor","#a78bfa"),
                ]:
                    col.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color};">{val}</div><div class="metric-label">{label}</div></div>',unsafe_allow_html=True)

                # Stats par niveau de score
                if s.get("score_stats") or s.get("score_global_stats"):
                    st.markdown("**Performance par niveau de score :**")
                    all_rows = []
                    for k,v2 in s.get("score_stats",{}).items():
                        all_rows.append({"Niveau":k,"Win Rate":f"{v2['win_rate']}%",
                                        "Perf moy.":f"{v2['avg_perf']}%","N":v2["n"]})
                    for k,v2 in s.get("score_global_stats",{}).items():
                        all_rows.append({"Niveau":k,"Win Rate":f"{v2['win_rate']}%",
                                        "Perf moy.":f"{v2['avg_perf']}%","N":v2["n"]})
                    if all_rows:
                        st.dataframe(pd.DataFrame(all_rows).set_index("Niveau"),
                                     use_container_width=True)

                # Courbe de performance cumulee
                try:
                    import plotly.express as px
                    df_s=df_bt.sort_values("entry_date").copy()
                    df_s["Perf cumulee"]=df_s["perf"].fillna(0).cumsum()
                    fig=px.line(df_s,x="entry_date",y="Perf cumulee",
                                color="ticker",
                                title=f"Performance cumulee — horizon {horizon_bt} mois",
                                color_discrete_sequence=["#00ff88","#7DF9FF","#fbbf24",
                                                         "#a78bfa","#f97316","#f87171"])
                    fig.add_hline(y=0,line_dash="dash",line_color="#f87171")
                    fig.update_layout(paper_bgcolor="#0a0e1a",plot_bgcolor="#111827",
                                      font_color="#e2e8f0",title_font_color="#00ff88",
                                      xaxis=dict(gridcolor="#1e3a5f"),
                                      yaxis=dict(gridcolor="#1e3a5f"))
                    st.plotly_chart(fig,use_container_width=True)
                except ImportError:
                    pass

                # Tableau des trades
                with st.expander("Voir tous les trades"):
                    disp_cols=[c for c in ["ticker","entry_date","exit_date","score",
                                           "entry_price","exit_price","perf","max_drawdown","result"]
                               if c in df_bt.columns]
                    st.dataframe(df_bt[disp_cols].sort_values("entry_date",ascending=False),
                                 use_container_width=True,hide_index=True)

                buf2=BytesIO()
                with pd.ExcelWriter(buf2,engine="openpyxl") as w:
                    df_bt.to_excel(w,index=False,sheet_name="Trades LT")
                st.download_button("Exporter backtest",data=buf2.getvalue(),
                                   file_name=f"backtest_lt_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")

# ── SCAN PRINCIPAL ──
if run:
    if not tickers_list:
        st.error("❌ Sélectionnez au moins un indice dans la sidebar."); st.stop()

    t0=datetime.now()

    # Étape 1 — Batch download
    st.markdown(f"### ⚡ Étape 1 — Batch download ({len(tickers_list)} titres)")
    with st.spinner("📥 Téléchargement batch en cours..."):
        price_data=batch_download_prices(tuple(sorted(tickers_list)),period="1y")
    t1=datetime.now()
    if not price_data:
        st.error("❌ Échec du téléchargement. Réessayez dans quelques instants."); st.stop()
    st.success(f"✅ {len(price_data)} titres téléchargés en {(t1-t0).seconds}s")

    # Étape 2 — Pré-filtre instantané
    st.markdown("### 🔬 Étape 2 — Pré-filtre (instantané)")
    passed=[]; rejected=[]
    pf_bar=st.progress(0)
    for i,ticker in enumerate(tickers_list):
        pf_bar.progress((i+1)/len(tickers_list))
        ok,reason,data=prefilter_from_prices(ticker,price_data,min_price,max_price,min_vol)
        if ok: passed.append({"ticker":ticker,"data":data})
        else:  rejected.append({"ticker":ticker,"reason":reason})

    # Résumé raisons
    rej_df=pd.DataFrame(rejected)
    reasons={}
    if not rej_df.empty:
        for _,r in rej_df.iterrows():
            k=r["reason"].split("(")[0].strip()
            reasons[k]=reasons.get(k,0)+1
    reason_str=" · ".join(f"{k}: {v}" for k,v in
                          sorted(reasons.items(),key=lambda x:-x[1])[:5])

    t2=datetime.now()
    st.markdown(f"""<div class="prefilter-banner">
        🔬 <span style="color:#00ff88;font-weight:700;">{len(passed)} retenus</span>
        &nbsp;/&nbsp; {len(price_data)} téléchargés &nbsp;/&nbsp; {len(tickers_list)} total
        &nbsp;|&nbsp; {len(rejected)} éliminés &nbsp;|&nbsp; ⏱️ {(t2-t0).seconds}s
        <br><span style='color:#64748b;font-size:0.78rem;'>Raisons: {reason_str}</span>
    </div>""",unsafe_allow_html=True)

    with st.expander(f"🗑️ {len(rejected)} titres éliminés — voir détails"):
        if not rej_df.empty:
            st.dataframe(rej_df,use_container_width=True,hide_index=True,height=250)

    if not passed:
        st.error("❌ Aucun titre ne passe le pré-filtre. Assouplissez les seuils."); st.stop()

    # Étape 3 — Scoring fondamentaux
    st.markdown(f"### 📊 Étape 3 — Scoring /100 · 6 catégories ({len(passed)} titres)")

    # Pré-charger SPY une seule fois
    spy_data = None
    try:
        with st.spinner("Chargement SPY..."):
            spy_data = yf.Ticker("SPY").history(period="1y")
    except Exception:
        pass

    # Analyse sectorielle
    sector_info = None
    with st.spinner("Analyse sectorielle..."):
        sector_info = get_sector_strength()
    if sector_info and sector_info["ranked"]:
        top3_sec    = sector_info["top3"]
        bottom3_sec = sector_info["bottom3"]
        sec_cols = st.columns(len(sector_info["ranked"]))
        for idx,(sec,data) in enumerate(sector_info["ranked"]):
            color = "#00ff88" if sec in top3_sec else "#f87171" if sec in bottom3_sec else "#fbbf24"
            icon  = "🔥" if sec in top3_sec else "❄️" if sec in bottom3_sec else "~"
            sec_cols[idx].markdown(
                f'<div style="background:#0d1117;border:1px solid {color}44;border-radius:6px;'
                f'padding:6px;text-align:center;font-size:0.7rem;">'
                f'<span style="color:{color};">{icon}</span><br>'
                f'<span style="color:#e2e8f0;font-size:0.65rem;">{sec.replace(" ",chr(10))}</span><br>'
                f'<span style="color:{color};font-family:Space Mono,monospace;font-size:0.8rem;">'
                f'{data["perf_1m"]:+.1f}%</span></div>',
                unsafe_allow_html=True)

    # ── PRÉ-CHARGEMENT FONDAMENTAUX (thread principal, séquentiel) ──
    # Évite les blocages Yahoo Finance dans les threads parallèles
    fund_cache = {}
    fund_prog = st.progress(0)
    fund_stat = st.empty()
    tickers_to_load = [item["ticker"] for item in passed]
    for idx, ticker in enumerate(tickers_to_load):
        fund_prog.progress((idx+1) / len(tickers_to_load))
        fund_stat.markdown(f"Chargement fondamentaux `{ticker}` ({idx+1}/{len(tickers_to_load)})...")
        try:
            import time
            t_obj = yf.Ticker(ticker)
            info  = {}
            for attempt in range(3):
                try:
                    info = t_obj.info
                    if info and len(info) > 5: break
                except Exception:
                    time.sleep(0.8 * (attempt+1))
            if info and len(info) > 5:
                try:
                    fin  = t_obj.financials
                    bs   = t_obj.balance_sheet
                    cf   = t_obj.cashflow
                    qfin = t_obj.quarterly_financials
                except Exception:
                    fin = bs = cf = qfin = None
                fund_cache[ticker] = {"info": info, "fin": fin, "bs": bs, "cf": cf, "qfin": qfin}
        except Exception:
            pass
        time.sleep(0.15)  # pause légère entre chaque ticker
    fund_stat.empty()
    fund_prog.empty()

    loaded_count = len(fund_cache)
    st.markdown(f"""<div class="prefilter-banner">
        Fondamentaux: <span style="color:#00ff88;font-weight:700;">{loaded_count}/{len(passed)}</span> titres chargés
    </div>""", unsafe_allow_html=True)

    if loaded_count == 0:
        st.error("Impossible de charger les fondamentaux. Yahoo Finance bloqué — attendez 2 minutes et réessayez.")
        st.stop()

    # Scoring parallèle avec fondamentaux déjà en cache local
    sc_prog=st.progress(0); sc_stat=st.empty()
    def sc_cb(d,t_):
        sc_prog.progress(d/t_)
        sc_stat.markdown(f"Scoring `{d}/{t_}`...")
    results=run_scoring_parallel(passed, ms["bonus"], nb_sc,
                                  spy_hist=spy_data,
                                  sector_data=sector_info,
                                  fund_cache=fund_cache,
                                  cb=sc_cb)
    sc_stat.empty()
    t3=datetime.now()

    if not results:
        st.error("Aucun résultat de scoring. Causes possibles: connexion instable ou Yahoo Finance bloqué.")
        st.info("Essayez de: 1) Réduire les threads à 5 dans la sidebar  2) Relancer le scan  3) Attendre 2 minutes")
        st.stop()

    df_all=pd.DataFrame(results).sort_values("Score Total",ascending=False).reset_index(drop=True)
    total_time=(t3-t0).seconds
    scan_date=datetime.now().strftime("%d %B %Y à %H:%M")
    st.success(f"✅ Scan terminé en {total_time}s — {len(df_all)} titres scorés")

    st.session_state.update({
        "df_all":df_all,"n_total":len(tickers_list),
        "n_passed":len(passed),"ms":ms,
        "scan_time":total_time,"scan_date":scan_date,
        "passed_list": passed,
    })

    # Lire les credentials Supabase depuis session_state (définis par la sidebar)
    _sb_url = st.session_state.get("sb_url", "")
    _sb_key = st.session_state.get("sb_key", "")

    # Sauvegarder dans la base de données
    if DB_AVAILABLE and _sb_url and _sb_key:
        with st.spinner("Sauvegarde dans Supabase..."):
            nb, err = save_scan(df_all, scan_date, _sb_url, _sb_key)
        if err:
            st.warning(f"Sauvegarde DB: {err}")
        else:
            st.success(f"{nb} titres sauvegardes dans Supabase")
    else:
        # Fallback JSON local
        save_scan_to_history(df_all, scan_date)

# ── AFFICHAGE RÉSULTATS ──
if "df_all" in st.session_state:
    df_all  =st.session_state["df_all"]
    n_total =st.session_state["n_total"]
    n_passed=st.session_state["n_passed"]
    ms_s    =st.session_state.get("ms",ms)
    scan_t  =st.session_state.get("scan_time","—")
    scan_d  =st.session_state.get("scan_date","—")

    # Filtres
    df_show=df_all[df_all["Score Total"]>=min_score_f].copy()
    df_show["Signal_clean"]=df_show["Signal"].str.extract(r"(EXCELLENT|BON|CORRECT|EVITER|EVITER)")[0].fillna(df_show["Signal"]).str.replace("EVITER","EVITER",regex=False)
    df_show=df_show[df_show["Convergence"]>=min_conv_f]
    if signal_f: df_show=df_show[df_show["Signal"].isin(signal_f)]
    # Filtre zone d'achat
    if filter_zone and "Timing" in df_show.columns:
        df_show=df_show[df_show["Timing"].isin(["ENTRER MAINTENANT","CONDITIONS FAVORABLES"])]
    try:
        sort_col = sort_f if sort_f in df_show.columns else "Score Global"
        df_show=df_show.sort_values(sort_col,ascending=(sort_f=="P/E")).reset_index(drop=True)
    except Exception:
        df_show=df_show.sort_values("Score Global",ascending=False).reset_index(drop=True)

    # Info dernier scan
    st.markdown(f"""<div class="scan-info">
        🕐 Dernier scan: <strong>{scan_d}</strong> &nbsp;|&nbsp;
        ⏱️ {scan_t}s &nbsp;|&nbsp;
        🌍 {n_total} titres → {n_passed} pré-filtre → {len(df_show)} affichés &nbsp;|&nbsp;
        Marché: <strong style="color:{ms_s['color']};">{ms_s['regime']}</strong>
    </div>""",unsafe_allow_html=True)

    # Métriques
    exc=len(df_all[df_all["Signal"].str.contains("EXCELLENT",na=False)])
    bon=len(df_all[df_all["Signal"].str.contains("BON",na=False)])
    cor=len(df_all[df_all["Signal"].str.contains("CORRECT",na=False)])
    c5x=len(df_all[df_all["Convergence"]==5])
    avg=int(df_all["Score Total"].mean()) if not df_all.empty else 0
    upside_num=pd.to_numeric(df_all["Upside Analystes %"],errors="coerce")
    avg_up=upside_num.mean()

    c1,c2,c3,c4,c5,c6,c7=st.columns(7)
    for col,val,label,color in[
        (c1,n_passed,"Pré-filtre OK","#4a90d0"),
        (c2,exc,"Excellent","#00ff88"),
        (c3,bon,"Bon","#4ade80"),
        (c4,cor,"Correct","#fbbf24"),
        (c5,c5x,"Conv. 5/5","#a78bfa"),
        (c6,avg,"Score moyen","#e2e8f0"),
        (c7,f"{avg_up:.0f}%" if not np.isnan(avg_up) else "N/D","Upside moy.","#f97316"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color};">{val}</div><div class="metric-label">{label}</div></div>',
                     unsafe_allow_html=True)

    # ══ TOP 10 ══
    st.markdown("---")
    st.markdown("## 🏆 TOP 10 — Meilleures Opportunités")

    top10=df_show.head(10)

    # Map ticker → pre_data pour les graphiques d'entrée
    pre_data_map = {item["ticker"]: item["data"] for item in st.session_state.get("passed_list",[])}

    if top10.empty:
        st.warning("⚠️ Aucun titre ne correspond aux filtres actifs. Réduisez le score minimum ou ajustez les signaux dans la sidebar.")
    else:
        for rank,(_,row) in enumerate(top10.iterrows(),1):
            score=row["Score Total"]; conv=row["Convergence"]

            if rank==1:
                card_bg="background:linear-gradient(135deg,#1a1400,#2a2000);border:2px solid #ffd700;box-shadow:0 0 40px #ffd70044;"
                bdr="#ffd700"; rank_icon="🥇"
            elif rank==2:
                card_bg="background:linear-gradient(135deg,#141414,#202020);border:2px solid #c0c0c088;"
                bdr="#c0c0c0"; rank_icon="🥈"
            elif rank==3:
                card_bg="background:linear-gradient(135deg,#1a0e00,#2a1800);border:2px solid #cd7f3288;"
                bdr="#cd7f32"; rank_icon="🥉"
            elif score>=68 or conv>=4:
                card_bg="background:linear-gradient(135deg,#001a0f,#002a18);border:1px solid #00ff8844;"
                bdr="#00ff88"; rank_icon=f"#{rank}"
            else:
                card_bg="background:linear-gradient(135deg,#0a1628,#0f2040);border:1px solid #1e4060;"
                bdr="#1e4060"; rank_icon=f"#{rank}"

            _sig_colors={"🟢 EXCELLENT":"#00ff88","🟢 BON":"#4ade80","🟡 CORRECT":"#fbbf24","🔴 ÉVITER":"#f87171"}
            sc=_sig_colors.get(row["Signal"],"#64748b")

            timing_v = row.get("Timing","—")
            timing_colors = {
                "ENTRER MAINTENANT":    "#00ff88",
                "CONDITIONS FAVORABLES":"#7DF9FF",
                "ATTENDRE":             "#fbbf24",
                "EVITER":               "#f87171",
            }
            tc = timing_colors.get(timing_v, "#64748b")
            sg = int(row.get("Score Global", score))

            with st.expander(
                f"{rank_icon}  {row['Ticker']} — {row['Nom']}  |  {row['Signal']}  |  "
                f"Global:{sg}/100  |  {timing_v}  |  Conv: {row['Conv_Bar']} {conv}/6",
                expanded=(rank<=3)
            ):
                st.markdown(f'<div style="{card_bg}border-radius:14px;padding:20px 24px;margin:4px 0;">',
                            unsafe_allow_html=True)

                # En-tête
                r1,r2,r3,r4=st.columns([1,3,2,2])
                r1.markdown(f"<div style='font-family:Space Mono,monospace;font-size:2.2rem;font-weight:700;color:{bdr};'>{rank_icon}</div>",
                            unsafe_allow_html=True)
                r2.markdown(
                    f"<span class='ticker-badge'>{row['Ticker']}</span>"
                    f"<span style='color:#64748b;font-size:0.85rem;'>{row['Secteur']} · {row['Cap (G$)']}G$</span>"
                    f"<div style='color:#94a3b8;font-size:0.8rem;margin-top:2px;'>{row['Nom']}</div>",
                    unsafe_allow_html=True)
                r3.markdown(
                    f"<div style='font-family:Space Mono,monospace;font-size:1.2rem;color:{bdr};'>{row['Conv_Bar']} {conv}/6</div>"
                    f"<div style='color:{sc};font-size:0.9rem;font-weight:700;'>{row['Signal']}</div>"
                    f"<div style='background:{tc}22;border:1px solid {tc}44;border-radius:4px;"
                    f"padding:2px 8px;font-size:0.75rem;color:{tc};font-weight:700;margin-top:3px;'>"
                    f"{timing_v}</div>"
                    f"<div style='color:#64748b;font-size:0.72rem;'>Score Global: {sg}/100</div>",
                    unsafe_allow_html=True)

                # Upside analystes — affiché seulement si valeur numérique
                upside_val=row["Upside Analystes %"]
                try:
                    upside_num=float(str(upside_val))
                    ucolor="#00ff88" if upside_num>20 else "#fbbf24" if upside_num>0 else "#f87171"
                    r4.markdown(
                        f"<div style='text-align:right;'>"
                        f"<div style='font-family:Space Mono,monospace;font-size:1.4rem;font-weight:700;color:{ucolor};'>"
                        f"+{upside_num:.1f}%</div>"
                        f"<div style='color:#64748b;font-size:0.75rem;'>UPSIDE ANALYSTES</div></div>",
                        unsafe_allow_html=True)
                except (ValueError, TypeError):
                    r4.markdown("<div style='color:#64748b;font-size:0.82rem;text-align:right;'>Upside N/D</div>",
                                unsafe_allow_html=True)

                st.markdown("---")

                # Scores 6 catégories
                s1,s2,s3,s4,s5,s6,s7=st.columns(7)
                s1.metric("Score Total",f"{score}/100")
                s2.metric("🏆 Qualité",f"{row['Score Qualité']}/25")
                s3.metric("📈 Croissance",f"{row['Score Croissance']}/20")
                s4.metric("💲 Valorisation",f"{row['Score Valorisation']}/15")
                s5.metric("🚀 Momentum",f"{row['Score Momentum']}/10")
                s6.metric("🎯 Conviction",f"{row['Score Conviction']}/10")
                s7.metric("🏦 Institutionnel",f"{row['Score Institutionnel']}/20")

                # Barres visuelles avec détails
                crit_cfg=[
                    ("_q","🏆 Qualité financière","Score Qualité",25,"#00ff88"),
                    ("_g","📈 Croissance","Score Croissance",20,"#7DF9FF"),
                    ("_v","💲 Valorisation","Score Valorisation",15,"#fbbf24"),
                    ("_m","🚀 Momentum","Score Momentum",10,"#a78bfa"),
                    ("_c","🎯 Conviction","Score Conviction",10,"#f97316"),
                    ("_i","🏦 Signaux Institutionnels","Score Institutionnel",20,"#38bdf8"),
                ]
                for key,title,score_col,max_pts,color in crit_cfg:
                    val2=row[score_col]; pct=int(val2/max_pts*100)
                    st.markdown(
                        f"<div style='margin:4px 0;'>"
                        f"<div style='display:flex;justify-content:space-between;margin-bottom:2px;'>"
                        f"<span style='font-size:0.82rem;color:#94a3b8;'>{title}</span>"
                        f"<span style='font-family:Space Mono,monospace;font-size:0.82rem;color:{color};font-weight:700;'>"
                        f"{int(val2)}/{max_pts}</span></div>"
                        f"<div style='background:#1e2a3a;border-radius:4px;height:8px;'>"
                        f"<div style='width:{pct}%;background:linear-gradient(90deg,{color}88,{color});"
                        f"height:8px;border-radius:4px;'></div></div></div>",
                        unsafe_allow_html=True)
                    for detail in row.get(key,[]):
                        ic="#00ff88" if "✅" in detail else "#fbbf24" if "~" in detail else "#f87171"
                        st.markdown(f"<span style='font-size:0.78rem;color:{ic};margin-left:10px;'>→ {detail}</span>",
                                    unsafe_allow_html=True)

                # Box institutionnel
                st.markdown(f"""<div style='background:linear-gradient(135deg,#0a1628,#0d1f35);
                    border:1px solid #38bdf844;border-left:4px solid #38bdf8;
                    border-radius:8px;padding:12px 16px;margin:6px 0;font-size:0.85rem;'>
                    <strong style='color:#38bdf8;'>🏦 Signaux Institutionnels</strong> &nbsp;|&nbsp;
                    Inst.:{row['Inst. Ownership %']}% &nbsp;|&nbsp;
                    Short:{row['Short Interest %']}% &nbsp;|&nbsp;
                    ROIC:{row['ROIC %']}% &nbsp;|&nbsp;
                    Beta:{row['Beta']} &nbsp;|&nbsp;
                    Mom.Rel:{row['Momentum Relatif %']}% &nbsp;|&nbsp;
                    Marge↗:{row['Tendance Marge']}pts &nbsp;|&nbsp;
                    P/S:{row['P/S']} &nbsp;|&nbsp;
                    Curr.Ratio:{row['Current Ratio']}
                </div>""",unsafe_allow_html=True)

                # Box conviction
                st.markdown(f"""<div class="conviction-box">
                    <strong style='color:#f97316;'>🎯 Conviction</strong> &nbsp;|&nbsp;
                    Insiders: <strong>{row['Insider %']}%</strong> &nbsp;|&nbsp;
                    Cible: <strong>${row['Cible Moy $']}</strong> &nbsp;|&nbsp;
                    Upside: <strong style='color:#f97316;'>{row['Upside Analystes %']}%</strong> &nbsp;|&nbsp;
                    {row['Nb Analystes']} analystes &nbsp;|&nbsp;
                    Score: <strong>{row['Recommandation']}/5</strong>
                </div>""",unsafe_allow_html=True)

                st.markdown("---")
                m1,m2,m3,m4,m5,m6,m7,m8=st.columns(8)
                m1.metric("Prix","$" + str(row["Prix $"]), f"{row['Var. %']:+.1f}%")
                m2.metric("ROE",f"{row['ROE %']}%")
                m3.metric("Marge brute",f"{row['Marge Brute %']}%")
                m4.metric("Marge nette",f"{row['Marge Nette %']}%")
                m5.metric("P/E",row["P/E"])
                m6.metric("PEG",row["PEG"])
                m7.metric("ROIC",f"{row['ROIC %']}%")
                m8.metric("Beta",row["Beta"])

                # Fondamentaux 5 ans
                st.markdown(f"""<div style='background:#0d1a2a;border:1px solid #7DF9FF33;
                    border-left:3px solid #7DF9FF;border-radius:6px;
                    padding:10px 14px;margin:6px 0;font-size:0.82rem;'>
                    <strong style='color:#7DF9FF;'>📅 Fondamentaux 5 ans</strong><br>
                    <span style='color:#94a3b8;'>Rev. CAGR:</span>
                    <strong style='color:#e2e8f0;'>{row.get("Rev. Growth % (5Y CAGR)","N/D")}</strong>
                    &nbsp;|&nbsp;
                    <span style='color:#94a3b8;'>EPS CAGR:</span>
                    <strong style='color:#e2e8f0;'>{row.get("EPS Growth % (5Y CAGR)","N/D")}</strong>
                    &nbsp;|&nbsp;
                    <span style='color:#94a3b8;'>Consistance:</span>
                    <strong style='color:#e2e8f0;'>{row.get("Rev. Consistance","N/D")}</strong>
                    &nbsp;|&nbsp;
                    <span style='color:#94a3b8;'>FCF:</span>
                    <strong style='color:#e2e8f0;'>{row.get("FCF 5Y","N/D")}</strong>
                    &nbsp;|&nbsp;
                    <span style='color:#94a3b8;'>Marge 5Y:</span>
                    <strong style='color:#e2e8f0;'>{row.get("Marge 5Y Tendance","N/D")}pts</strong>
                    &nbsp;|&nbsp;
                    <span style='color:#94a3b8;'>Secteur:</span>
                    <strong style='color:#fbbf24;'>{row.get("Secteur Force","N/D")}</strong>
                </div>""", unsafe_allow_html=True)

                st.markdown(
                    f"**Short:** {row['Short Interest %']}% &nbsp;·&nbsp; "
                    f"**Inst.Own:** {row['Inst. Ownership %']}% &nbsp;·&nbsp; "
                    f"**EV/EBITDA:** {row['EV/EBITDA']} &nbsp;·&nbsp; "
                    f"**Signaux:** <span style='color:#00ff88;'>{row['Signaux']}</span>",
                    unsafe_allow_html=True)

                # ── RECOMMANDATION DE POSITION ──
                st.markdown("---")
                ps = calc_position_sizing(row, ms_s.get("regime","NEUTRE"))
                ps_colors = {
                    "TRÈS HAUTE": "#00ff88",
                    "HAUTE":      "#7DF9FF",
                    "CORRECTE":   "#fbbf24",
                    "MODÉRÉE":    "#f97316",
                    "FAIBLE":     "#f87171",
                }
                ps_color      = ps_colors.get(ps["conviction"], "#64748b")
                ps_horizon    = ps["horizon"]
                ps_regime     = ms_s.get("regime", "?")
                ps_mult       = ps["market_mult"]
                ps_stop_txt   = "soit " + str(ps["stop_price"]) if ps["stop_price"] > 0 else "N/D"
                ps_target_txt = str(ps["target_price"]) if ps["target_price"] > 0 else "N/D"
                ps_rr_txt     = str(ps["rr_ratio"]) + ":1" if ps["rr_ratio"] > 0 else "N/D"

                # Box principale — métriques
                st.markdown(f"""<div style='background:linear-gradient(135deg,#0a1f15,#0d2a1e);
                    border:2px solid {ps_color}44;border-radius:12px;
                    padding:16px 20px;margin:8px 0;'>
                    <div style='display:flex;justify-content:space-between;
                    align-items:center;margin-bottom:12px;'>
                        <span style='font-family:Space Mono,monospace;font-size:1rem;
                        font-weight:700;color:{ps_color};'>RECOMMANDATION DE POSITION</span>
                        <span style='background:{ps_color}22;border:1px solid {ps_color}55;
                        border-radius:6px;padding:3px 10px;font-size:0.8rem;
                        color:{ps_color};font-weight:700;'>Conviction {ps["conviction"]}</span>
                    </div>
                    <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;
                    gap:12px;margin-bottom:12px;'>
                        <div style='background:#0a1117;border-radius:8px;
                        padding:10px;text-align:center;'>
                            <div style='color:#64748b;font-size:0.72rem;
                            margin-bottom:4px;'>TAILLE SUGGEREE</div>
                            <div style='font-family:Space Mono,monospace;font-size:1.6rem;
                            font-weight:700;color:{ps_color};'>{ps["taille"]}%</div>
                            <div style='color:#64748b;font-size:0.68rem;'>du portefeuille</div>
                        </div>
                        <div style='background:#0a1117;border-radius:8px;
                        padding:10px;text-align:center;'>
                            <div style='color:#64748b;font-size:0.72rem;
                            margin-bottom:4px;'>STOP LOSS</div>
                            <div style='font-family:Space Mono,monospace;font-size:1.6rem;
                            font-weight:700;color:#f87171;'>{ps["stop_pct"]}%</div>
                            <div style='color:#f87171;font-size:0.68rem;'>{ps_stop_txt}</div>
                        </div>
                        <div style='background:#0a1117;border-radius:8px;
                        padding:10px;text-align:center;'>
                            <div style='color:#64748b;font-size:0.72rem;
                            margin-bottom:4px;'>OBJECTIF</div>
                            <div style='font-family:Space Mono,monospace;font-size:1.6rem;
                            font-weight:700;color:#00ff88;'>{ps_target_txt}</div>
                            <div style='color:#64748b;font-size:0.68rem;'>cible analystes</div>
                        </div>
                        <div style='background:#0a1117;border-radius:8px;
                        padding:10px;text-align:center;'>
                            <div style='color:#64748b;font-size:0.72rem;
                            margin-bottom:4px;'>RATIO R/R</div>
                            <div style='font-family:Space Mono,monospace;font-size:1.6rem;
                            font-weight:700;color:#fbbf24;'>{ps_rr_txt}</div>
                            <div style='color:#64748b;font-size:0.68rem;'>risque/rendement</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Stratégie d'entrée
                st.markdown(f"""<div style='background:{ps["strat_color"]}11;
                    border:1px solid {ps["strat_color"]}44;border-left:4px solid {ps["strat_color"]};
                    border-radius:8px;padding:10px 14px;margin:4px 0;'>
                    <span style='color:{ps["strat_color"]};font-weight:700;font-size:0.9rem;'>
                    {ps["strategie"]}</span>
                    <span style='color:#94a3b8;font-size:0.82rem;'> — {ps["strat_detail"]}</span>
                </div>""", unsafe_allow_html=True)

                # Infos horizon + risques
                risk_line = " | " + " · ".join(ps["risk_notes"]) if ps["risk_notes"] else ""
                st.markdown(
                    f"<div style='font-size:0.78rem;color:#64748b;padding:4px 0;'>"
                    f"Horizon: <span style='color:#94a3b8;'>{ps_horizon}</span>"
                    f" | Marche: <span style='color:#94a3b8;'>{ps_regime} (x{ps_mult})</span>"
                    f"{risk_line}"
                    f"<br><span style='font-size:0.7rem;color:#3a5a7a;'>"
                    f"Indicatif uniquement — adaptez selon votre profil de risque</span></div>",
                    unsafe_allow_html=True)

                # ── TABLEAU DE BORD D'ENTRÉE ──
                st.markdown("---")
                st.markdown("#### 📉 Analyse graphique — Point d'entrée")
                render_entry_dashboard(row["Ticker"], pre_data_map.get(row["Ticker"]))

                st.markdown("</div>",unsafe_allow_html=True)

    # Tableau complet
    st.markdown("---")
    st.markdown(f"### 📋 Tableau complet — {len(df_show)} titres")
    tcols=["Ticker","Nom","Secteur","Secteur Force","Cap (G$)","Prix $","Var. %",
           "ROE %","Marge Brute %","Marge Nette %",
           "Rev. Growth % (5Y CAGR)","EPS Growth % (5Y CAGR)",
           "Rev. Consistance","FCF 5Y","Marge 5Y Tendance","ROE 5Y Tendance",
           "P/E","PEG","EV/EBITDA","P/S","RSI","Beta",
           "Inst. Ownership %","Short Interest %","ROIC %","Current Ratio",
           "Insider %","Upside Analystes %","Nb Analystes",
           "Score Qualité","Score Croissance","Score Valorisation",
           "Score Momentum","Score Conviction","Score Institutionnel",
           "Score Total","Conv_Bar","Signal"]
    tcols=[c for c in tcols if c in df_show.columns]
    st.dataframe(df_show[tcols].rename(columns={"Conv_Bar":"Conv."}),
        use_container_width=True,hide_index=True,
        column_config={
            "Score Total":       st.column_config.ProgressColumn("Score /100",min_value=0,max_value=100,format="%d"),
            "Var. %":            st.column_config.NumberColumn("Var. %",format="%.1f%%"),
            "Prix $":            st.column_config.NumberColumn("Prix $",format="$%.2f"),
            "Score Qualité":     st.column_config.NumberColumn("Qualité(25)",format="%d"),
            "Score Croissance":  st.column_config.NumberColumn("Croissance(20)",format="%d"),
            "Score Valorisation":st.column_config.NumberColumn("Valeur(15)",format="%d"),
            "Score Momentum":    st.column_config.NumberColumn("Momentum(10)",format="%d"),
            "Score Conviction":  st.column_config.NumberColumn("Conviction(10)",format="%d"),
            "Score Institutionnel":st.column_config.NumberColumn("Institutionnel(20)",format="%d"),
        })

    # Graphiques
    st.markdown("---")
    st.markdown("### 📊 Visualisations")
    try:
        import plotly.express as px
        t1,t2,t3,t4,t5,t6=st.tabs([
            "Score Total","Décomposition 6 catégories","ROE vs Croissance",
            "🏦 Signaux Institutionnels","Conviction","P/E vs PEG"
        ])
        with t1:
            fig=px.bar(df_show.head(25),x="Ticker",y="Score Total",color="Score Total",
                       color_continuous_scale=["#f87171","#fbbf24","#00ff88"],
                       hover_data=["Nom","Secteur","Signal","ROE %","Marge Brute %",
                                   "Insider %","Upside Analystes %"],
                       title="Top 25 — Score Long Terme /100")
            fig.add_hline(y=82,line_dash="dot",line_color="#00ff88",annotation_text="Excellent(82)")
            fig.add_hline(y=68,line_dash="dash",line_color="#fbbf24",annotation_text="Bon(68)")
            fig.update_layout(paper_bgcolor="#0a0e1a",plot_bgcolor="#111827",
                              font_color="#e2e8f0",title_font_color="#00ff88",
                              xaxis=dict(gridcolor="#1e3a5f"),
                              yaxis=dict(gridcolor="#1e3a5f",range=[0,105]))
            st.plotly_chart(fig,use_container_width=True)
        with t2:
            fig2=px.bar(df_show.head(20),x="Ticker",
                        y=["Score Qualité","Score Croissance","Score Valorisation",
                           "Score Momentum","Score Conviction","Score Institutionnel"],
                        color_discrete_map={
                            "Score Qualité":"#00ff88","Score Croissance":"#7DF9FF",
                            "Score Valorisation":"#fbbf24","Score Momentum":"#a78bfa",
                            "Score Conviction":"#f97316","Score Institutionnel":"#38bdf8"},
                        barmode="stack",title="Décomposition — 6 catégories")
            fig2.update_layout(paper_bgcolor="#0a0e1a",plot_bgcolor="#111827",
                               font_color="#e2e8f0",title_font_color="#00ff88",
                               xaxis=dict(gridcolor="#1e3a5f"),yaxis=dict(gridcolor="#1e3a5f"))
            st.plotly_chart(fig2,use_container_width=True)
        with t3:
            dfs3=df_show.copy()
            dfs3["ROE_n"]=pd.to_numeric(dfs3["ROE %"],errors="coerce")
            dfs3["RG_n"]=pd.to_numeric(dfs3["Rev. Growth %"],errors="coerce")
            dfs3b=dfs3.dropna(subset=["ROE_n","RG_n"])
            if not dfs3b.empty:
                fig3=px.scatter(dfs3b,x="ROE_n",y="RG_n",color="Signal_clean",size="Score Total",
                                hover_data=["Ticker","Nom","Marge Brute %","Insider %"],
                                color_discrete_map={"EXCELLENT":"#00ff88","BON":"#4ade80","CORRECT":"#fbbf24","EVITER":"#f87171"},
                                labels={"ROE_n":"ROE (%)","RG_n":"Croissance revenus (%)","Signal_clean":"Signal"},
                                title="Qualité (ROE) vs Croissance revenus")
                fig3.add_vline(x=15,line_dash="dash",line_color="#00ff88",annotation_text="ROE 15%")
                fig3.add_hline(y=8,line_dash="dash",line_color="#7DF9FF",annotation_text="Rev. +8%")
                fig3.update_layout(paper_bgcolor="#0a0e1a",plot_bgcolor="#111827",
                                   font_color="#e2e8f0",title_font_color="#00ff88",
                                   xaxis=dict(gridcolor="#1e3a5f"),yaxis=dict(gridcolor="#1e3a5f"))
                st.plotly_chart(fig3,use_container_width=True)
            else:
                st.info("Pas assez de données ROE/Croissance pour ce graphique.")
        with t4:
            # 🏦 Signaux institutionnels — Short Interest vs ROIC
            dfs4=df_show.copy()
            dfs4["SI_n"]=pd.to_numeric(dfs4["Short Interest %"],errors="coerce")
            dfs4["ROIC_n"]=pd.to_numeric(dfs4["ROIC %"],errors="coerce")
            dfs4b=dfs4.dropna(subset=["SI_n","ROIC_n"])
            if not dfs4b.empty:
                fig4=px.scatter(dfs4b,x="SI_n",y="ROIC_n",color="Signal_clean",size="Score Total",
                                hover_data=["Ticker","Nom","Inst. Ownership %","Beta","Current Ratio"],
                                color_discrete_map={"EXCELLENT":"#00ff88","BON":"#4ade80","CORRECT":"#fbbf24","EVITER":"#f87171"},
                                labels={"SI_n":"Short Interest (%)","ROIC_n":"ROIC (%)","Signal_clean":"Signal"},
                                title="🏦 Signaux Institutionnels — Short Interest vs ROIC")
                fig4.add_vline(x=5,line_dash="dash",line_color="#f87171",annotation_text="Short 5%")
                fig4.add_hline(y=15,line_dash="dash",line_color="#38bdf8",annotation_text="ROIC 15%")
                fig4.update_layout(paper_bgcolor="#0a0e1a",plot_bgcolor="#111827",
                                   font_color="#e2e8f0",title_font_color="#00ff88",
                                   xaxis=dict(gridcolor="#1e3a5f"),yaxis=dict(gridcolor="#1e3a5f"))
                st.plotly_chart(fig4,use_container_width=True)
            else:
                st.info("Pas assez de données Short Interest / ROIC.")
        with t5:
            dfs5=df_show.copy()
            dfs5["IP_n"]=pd.to_numeric(dfs5["Insider %"],errors="coerce")
            dfs5["UP_n"]=pd.to_numeric(dfs5["Upside Analystes %"],errors="coerce")
            dfs5b=dfs5.dropna(subset=["IP_n","UP_n"])
            if not dfs5b.empty:
                fig5=px.scatter(dfs5b,x="IP_n",y="UP_n",color="Signal_clean",size="Score Total",
                                hover_data=["Ticker","Nom","Score Conviction","Nb Analystes"],
                                color_discrete_map={"EXCELLENT":"#00ff88","BON":"#4ade80","CORRECT":"#fbbf24","EVITER":"#f87171"},
                                labels={"IP_n":"Insider ownership (%)","UP_n":"Upside analystes (%)","Signal_clean":"Signal"},
                                title="🎯 Conviction — Insider % vs Upside analystes %")
                fig5.add_vline(x=5,line_dash="dash",line_color="#f97316",annotation_text="Insiders 5%")
                fig5.add_hline(y=20,line_dash="dash",line_color="#f97316",annotation_text="Upside 20%")
                fig5.update_layout(paper_bgcolor="#0a0e1a",plot_bgcolor="#111827",
                                   font_color="#e2e8f0",title_font_color="#00ff88",
                                   xaxis=dict(gridcolor="#1e3a5f"),yaxis=dict(gridcolor="#1e3a5f"))
                st.plotly_chart(fig5,use_container_width=True)
            else:
                st.info("Pas assez de données Insider/Upside.")
        with t6:
            dfs6=df_show.copy()
            dfs6["PE_n"]=pd.to_numeric(dfs6["P/E"],errors="coerce")
            dfs6["PEG_n"]=pd.to_numeric(dfs6["PEG"],errors="coerce")
            dfs6b=dfs6.dropna(subset=["PE_n","PEG_n"])
            if not dfs6b.empty:
                signal_colors = {"EXCELLENT":"#00ff88","BON":"#4ade80","CORRECT":"#fbbf24","EVITER":"#f87171"}
                dfs6b = dfs6b.copy()
                dfs6b["Signal_clean"] = dfs6b["Signal"].str.extract(r"(EXCELLENT|BON|CORRECT|EVITER|ÉVITER)")[0].fillna(dfs6b["Signal"]).str.replace("ÉVITER","EVITER")
                fig6=px.scatter(dfs6b,x="PE_n",y="PEG_n",
                                color="Signal_clean",
                                size="Score Total",
                                hover_data=["Ticker","Nom","EV/EBITDA","P/S","Score Valorisation"],
                                color_discrete_map=signal_colors,
                                labels={"PE_n":"P/E","PEG_n":"PEG Ratio","Signal_clean":"Signal"},
                                title="Valorisation — P/E vs PEG")
                fig6.add_vline(x=25,line_dash="dash",line_color="#fbbf24",annotation_text="P/E 25")
                fig6.add_hline(y=1.5,line_dash="dash",line_color="#fbbf24",annotation_text="PEG 1.5")
                fig6.update_layout(paper_bgcolor="#0a0e1a",plot_bgcolor="#111827",
                                   font_color="#e2e8f0",title_font_color="#00ff88",
                                   xaxis=dict(gridcolor="#1e3a5f"),yaxis=dict(gridcolor="#1e3a5f"))
                st.plotly_chart(fig6,use_container_width=True)
            else:
                st.info("Pas assez de données P/E & PEG.")
    except ImportError:
        st.info("Installez plotly pour les graphiques : pip install plotly")

    # ── HISTORIQUE DES SCORES ──
    st.markdown("---")
    sb_url_s = st.session_state.get("sb_url")
    sb_key_s = st.session_state.get("sb_key")

    if DB_AVAILABLE and sb_url_s and sb_key_s:
        # Charger depuis Supabase
        history = get_latest_scans(sb_url_s, sb_key_s, n=12)

        # Stats globales
        stats = get_global_stats(sb_url_s, sb_key_s)
        if stats:
            st.markdown(f"""<div style='background:#0d1a2a;border:1px solid #00ff8833;
                border-radius:8px;padding:10px 16px;font-size:0.82rem;color:#64748b;'>
                Base de donnees: <strong style='color:#00ff88;'>{stats.get("nb_scans",0)} scans</strong>
                &nbsp;|&nbsp; {stats.get("total_rows",0)} titres enregistres
                &nbsp;|&nbsp; Du {stats.get("first_scan","—")} au {stats.get("last_scan","—")}
            </div>""", unsafe_allow_html=True)

        # Titres recurrents excellent
        top_rec = get_top_historical(sb_url_s, sb_key_s, min_score=80, min_appearances=2)
        if top_rec:
            st.markdown("#### Titres recurrents score >= 80")
            rec_cols = st.columns(min(len(top_rec[:5]), 5))
            for idx, r in enumerate(top_rec[:5]):
                rec_cols[idx].markdown(
                    f'<div class="metric-card">'
                    f'<div style="font-family:Space Mono,monospace;font-weight:700;'
                    f'color:#00ff88;font-size:1rem;">{r["ticker"]}</div>'
                    f'<div style="font-size:1.4rem;font-weight:700;color:#fbbf24;">'
                    f'{r["score_moyen"]}</div>'
                    f'<div style="font-size:0.72rem;color:#64748b;">'
                    f'{r["apparitions"]} scans</div>'
                    f'</div>', unsafe_allow_html=True)

        if history:
            with st.expander(f"📚 Historique — {len(history)} scans", expanded=False):
                render_history_section(df_show, history)

        # Watchlist
        st.markdown("---")
        st.markdown("#### Watchlist")
        wl = watchlist_get(sb_url_s, sb_key_s)
        wl_tickers = [w["ticker"] for w in wl]

        wl_col1, wl_col2 = st.columns([2,1])
        with wl_col1:
            wl_add_ticker = st.selectbox(
                "Ajouter a la watchlist",
                options=[""] + list(df_show["Ticker"]) if not df_show.empty else [""],
                key="wl_add")
            wl_note = st.text_input("Note (optionnel)", placeholder="Ex: Attendre RSI < 55")
        with wl_col2:
            if wl_add_ticker and st.button("Ajouter"):
                watchlist_add(wl_add_ticker, wl_note, sb_url_s, sb_key_s)
                st.rerun()
            if wl_tickers:
                wl_rm = st.selectbox("Retirer", options=[""] + wl_tickers)
                if wl_rm and st.button("Retirer"):
                    watchlist_remove(wl_rm, sb_url_s, sb_key_s)
                    st.rerun()

        if wl:
            # Enrichir la watchlist avec le score actuel
            wl_rows = []
            for w in wl:
                t = w["ticker"]
                score_now = "N/A"
                signal_now = "—"
                if not df_show.empty and t in df_show["Ticker"].values:
                    row_wl = df_show[df_show["Ticker"]==t].iloc[0]
                    score_now  = int(row_wl["Score Total"])
                    signal_now = str(row_wl["Signal"])
                wl_rows.append({
                    "Ticker":       t,
                    "Score actuel": score_now,
                    "Signal":       signal_now,
                    "Note":         w.get("note",""),
                    "Ajoute le":    w.get("added_date","")[:10],
                })
            st.dataframe(pd.DataFrame(wl_rows),
                         use_container_width=True, hide_index=True)
        else:
            st.caption("Watchlist vide — ajoute des titres pour les suivre.")

    else:
        # Fallback local
        history = load_history()
        if history:
            with st.expander(f"📚 Historique local — {len(history)} scans", expanded=False):
                render_history_section(df_show, history)
        else:
            st.markdown("""<div style='background:#0d1a2a;border:1px solid #1e3a5f;
                border-left:3px solid #4a90d0;border-radius:8px;padding:12px 16px;
                font-size:0.82rem;color:#64748b;'>
                📚 <strong style='color:#4a90d0;'>Historique</strong> —
                Configure Supabase dans la sidebar pour un historique permanent.
                Sans DB: historique en memoire locale (perdu au reboot).
            </div>""", unsafe_allow_html=True)

    # Analyse IA Gemini
    if use_ai and gemini_key and not df_show.empty:
        st.markdown("---")
        st.markdown("### 🤖 Analyse Gemini Flash — Top 10")
        with st.spinner("Gemini analyse le top 10..."):
            try:
                analysis = gemini_analyse(df_show, gemini_key, ms_s, n_total, n_passed)
                st.markdown(
                    f'<div class="ai-analysis-box">'
                    f'<span style="color:#00ff88;font-family:Space Mono,monospace;font-weight:700;">'
                    f'🤖 ANALYSE IA — Google Gemini Flash · Buffett+Lynch</span><br><br>{analysis}</div>',
                    unsafe_allow_html=True)
            except ImportError:
                st.error("❌ Installez google-generativeai : ajoutez-le à requirements.txt")
            except Exception as e:
                st.error(f"Erreur API Gemini: {e}")
    elif use_ai and not gemini_key:
        st.warning("⚠️ Entrez votre clé API Google Gemini dans la sidebar (aistudio.google.com).")

    # Export Excel
    st.markdown("---")
    ecols=[c for c in [
        "Ticker","Nom","Secteur","Cap (G$)","Prix $","Var. %",
        "ROE %","Marge Brute %","Marge Nette %","FCF (G$)",
        "Rev. Growth %","EPS Growth %","EPS Qtr %",
        "P/E","PEG","EV/EBITDA","P/S","RSI","MA50","MA200","52W High %",
        "Beta","Momentum Relatif %",
        "Inst. Ownership %","Short Interest %","ROIC %","Current Ratio","Tendance Marge",
        "Insider %","Upside Analystes %","Nb Analystes","Recommandation","Cible Moy $",
        "Score Qualité","Score Croissance","Score Valorisation",
        "Score Momentum","Score Conviction","Score Institutionnel",
        "Score Total","Convergence","Signal","Signaux"
    ] if c in df_show.columns]
    buf=BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        df_show[ecols].to_excel(w,index=False,sheet_name="Top Opportunités")
        df_all[ecols].to_excel(w,index=False,sheet_name="Tous les titres scorés")

    # ── Boutons export côte à côte ──
    col_xl, col_pdf = st.columns(2)
    with col_xl:
        st.download_button(
            "⬇️ Exporter Excel",
            data=buf.getvalue(),
            file_name=f"alphascreen_top10_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col_pdf:
        if st.button("📄 Générer rapport PDF Top 10"):
            with st.spinner("Génération du PDF..."):
                pdf_bytes = generate_pdf_report(df_show.head(10), ms_s, scan_d)
                if pdf_bytes:
                    st.download_button(
                        "⬇️ Télécharger le PDF",
                        data=pdf_bytes,
                        file_name=f"alphascreen_rapport_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error("❌ Erreur génération PDF. Vérifiez que reportlab est installé.")

st.markdown("---")
st.caption("⚠️ Données Yahoo Finance à titre informatif uniquement. AlphaScreen US ne constitue pas un conseil financier. Investir comporte des risques de perte en capital.")
