import yfinance as yf
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 📊 INTRADAY SIGNALS
# VWAP · PDH/PDL · Multi-TF · Momentum
# ─────────────────────────────────────────────

def calc_vwap_levels(ticker):
    try:
        t = yf.Ticker(ticker)
        intraday = t.history(period="2d", interval="5m")
        if intraday is None or intraday.empty or len(intraday) < 10:
            return _empty_vwap()
        intraday.index = pd.to_datetime(intraday.index)
        today = intraday.index[-1].date()
        td = intraday[intraday.index.date == today]
        pd_data = intraday[intraday.index.date < today]
        if td.empty: return _empty_vwap()
        tp = (td["High"] + td["Low"] + td["Close"]) / 3
        cv = td["Volume"].cumsum()
        ctp = (tp * td["Volume"]).cumsum()
        vwap = round(float(ctp.iloc[-1] / cv.iloc[-1]), 2) if cv.iloc[-1] > 0 else None
        price = float(td["Close"].iloc[-1])
        pvwap = round((price - vwap) / vwap * 100, 2) if vwap else 0
        pdh = pdl = None
        if not pd_data.empty:
            prev = pd_data[pd_data.index.date == pd_data.index[-1].date()]
            if not prev.empty:
                pdh = round(float(prev["High"].max()), 2)
                pdl = round(float(prev["Low"].min()), 2)
        orb = td.iloc[:6]
        orb_h = round(float(orb["High"].max()), 2) if not orb.empty else None
        orb_l = round(float(orb["Low"].min()), 2) if not orb.empty else None
        score = 0; signal = None
        if vwap and pvwap > 0.5: score += 8; signal = f"Au-dessus VWAP ${vwap} (+{pvwap}%)"
        elif vwap and pvwap < -1.0: score -= 8; signal = f"Sous VWAP ${vwap} ({pvwap}%)"
        if pdh and price > pdh: score += 12; signal = f"Breakout PDH ${pdh}"
        elif pdh and (pdh - price)/price*100 <= 1.5: score -= 5
        if orb_h and price > orb_h: score += 10; signal = signal or f"ORB Breakout ${orb_h}"
        elif orb_l and price < orb_l: score -= 10
        score = max(-15, min(score, 22))
        badge = "Setup intraday excellent" if score>=18 else "Favorable" if score>=10 else "Neutre" if score>=0 else "Defavorable"
        return {
            "vwap": vwap, "price_vs_vwap": pvwap, "above_vwap": price > vwap if vwap else True,
            "pdh": pdh, "pdl": pdl, "orb_high": orb_h, "orb_low": orb_l,
            "orb_breakout": price > orb_h if orb_h else False,
            "signal": signal, "score": score, "badge": badge,
        }
    except Exception:
        return _empty_vwap()


def calc_multitf_signals(ticker, hist_daily=None):
    try:
        t = yf.Ticker(ticker)
        results = {}
        for interval, label in [("1h","1H"), ("15m","15min")]:
            try:
                h = t.history(period="5d", interval=interval)
                if h is None or h.empty or len(h) < 20: results[label] = None; continue
                c = h["Close"]
                d = c.diff(); g = d.where(d>0,0).rolling(14).mean(); l = -d.where(d<0,0).rolling(14).mean()
                rsi = round(float((100-(100/(1+g/l.clip(lower=1e-10)))).iloc[-1]), 1)
                ma = c.rolling(20).mean(); std = c.rolling(20).std()
                up = ma+2*std; lo = ma-2*std
                bp = round((float(c.iloc[-1])-float(lo.iloc[-1]))/(float(up.iloc[-1])-float(lo.iloc[-1])),3) if float(up.iloc[-1])-float(lo.iloc[-1])>0 else 0.5
                e12 = c.ewm(span=12,adjust=False).mean(); e26 = c.ewm(span=26,adjust=False).mean()
                mh = float((e12-e26-(e12-e26).ewm(span=9,adjust=False).mean()).iloc[-1])
                results[label] = {"rsi":rsi,"bb_pct":bp,"macd_h":round(mh,4),
                                  "bullish":45<=rsi<=75 and mh>0 and bp>0.3,"overbought":rsi>75}
            except Exception:
                results[label] = None

        rsi_d = None
        if hist_daily is not None and not hist_daily.empty:
            try:
                c = hist_daily["Close"]; d = c.diff()
                g = d.where(d>0,0).rolling(14).mean(); l = -d.where(d<0,0).rolling(14).mean()
                rsi_d = round(float((100-(100/(1+g/l.clip(lower=1e-10)))).iloc[-1]),1)
            except Exception: pass

        tf_b = 0; tf_t = 0
        if rsi_d and 45<=rsi_d<=75: tf_b += 1
        if rsi_d: tf_t += 1
        for lb in ["1H","15min"]:
            if results.get(lb): tf_t += 1; tf_b += 1 if results[lb]["bullish"] else 0

        contra = False
        if results.get("1H") and results.get("15min"):
            if abs(results["1H"]["rsi"]-results["15min"]["rsi"]) > 20: contra = True
            if results["1H"]["bullish"] and results["15min"]["overbought"]: contra = True

        if tf_t == 0: conf = "INDISPONIBLE"; score = 0
        elif contra:  conf = "CONTRADICTOIRE"; score = -5
        elif tf_b == tf_t: conf = "FORT"; score = 15
        elif tf_b >= tf_t-1: conf = "MODERE"; score = 8
        elif tf_b >= 1: conf = "FAIBLE"; score = 2
        else: conf = "BAISSIER"; score = -10

        r1 = f"1H RSI:{results['1H']['rsi']}" if results.get("1H") else "1H N/A"
        r2 = f"15m RSI:{results['15min']['rsi']}" if results.get("15min") else "15m N/A"
        rd = f"D RSI:{rsi_d}" if rsi_d else "D N/A"

        if conf == "FORT": signal = f"Confirmation {tf_b}/{tf_t} TF — {rd} | {r1} | {r2}"
        elif conf == "CONTRADICTOIRE": signal = f"Signal contradictoire entre TF — attendre"
        elif conf == "BAISSIER": signal = f"Tous les TF baissiers"
        else: signal = f"Confirmation partielle {tf_b}/{tf_t} TF — {rd} | {r1} | {r2}"

        badge = f"Multi-TF {tf_b}/{tf_t}" if conf=="FORT" else "TF contradictoires" if conf=="CONTRADICTOIRE" else "TF baissier" if conf=="BAISSIER" else f"Multi-TF partiel {tf_b}/{tf_t}"

        return {
            "rsi_1h": results.get("1H",{}).get("rsi") if results.get("1H") else None,
            "rsi_15min": results.get("15min",{}).get("rsi") if results.get("15min") else None,
            "rsi_daily": rsi_d, "tf_alignment": tf_b, "tf_total": tf_t,
            "confirmation": conf, "contradiction": contra,
            "signal": signal, "score": score, "badge": badge,
        }
    except Exception:
        return _empty_multitf()


def calc_intraday_momentum(ticker):
    try:
        t = yf.Ticker(ticker)
        intra = t.history(period="1d", interval="5m")
        if intra is None or intra.empty or len(intra) < 3: return _empty_momentum()
        op = float(intra["Open"].iloc[0]); lp = float(intra["Close"].iloc[-1])
        perf = round((lp-op)/op*100, 2) if op > 0 else 0
        spy_perf = 0
        try:
            sh = yf.Ticker("SPY").history(period="1d", interval="5m")
            if sh is not None and not sh.empty:
                so = float(sh["Open"].iloc[0]); sl = float(sh["Close"].iloc[-1])
                spy_perf = round((sl-so)/so*100, 2) if so > 0 else 0
        except Exception: pass
        rm = round(perf - spy_perf, 2)
        acc = False
        if len(intra) >= 6:
            mid = len(intra)//2; h1 = float(intra["Close"].iloc[mid-1])
            h2 = float(intra["Close"].iloc[-1])
            p1 = (h1-op)/op*100; p2 = (h2-h1)/h1*100
            acc = p2 > p1*0.5 and p2 > 0
        if rm >= 2.0:    score=15; trend="FORT_HAUSSIER"; signal=f"Momentum intraday +{rm}% vs SPY"
        elif rm >= 1.0:  score=10; trend="HAUSSIER";      signal=f"Momentum fort +{rm}% vs SPY"
        elif rm >= 0.3:  score=5;  trend="HAUSSIER";      signal=f"Momentum positif +{rm}% vs SPY"
        elif rm >= -0.5: score=0;  trend="NEUTRE";        signal=f"Momentum neutre ({rm}% vs SPY)"
        elif rm >= -1.5: score=-5; trend="BAISSIER";      signal=f"Sous-performe SPY {abs(rm)}%"
        else:            score=-12;trend="BAISSIER";      signal=f"Faiblesse intraday {rm}% vs SPY"
        if acc and score > 0: score += 3; signal += " acceleration"
        score = max(-15, min(score, 18))
        badge = f"Momentum fort ({perf:+.1f}% | SPY {spy_perf:+.1f}%)" if score>=12 else f"Momentum positif ({perf:+.1f}%)" if score>=5 else f"Neutre ({perf:+.1f}%)" if score>=0 else f"Faible ({perf:+.1f}%)"
        return {
            "perf_open": perf, "spy_perf_open": spy_perf, "relative_mom": rm,
            "trend_intraday": trend, "acceleration": acc,
            "signal": signal, "score": score, "badge": badge,
        }
    except Exception:
        return _empty_momentum()


def _empty_vwap():
    return {"vwap":None,"price_vs_vwap":0,"above_vwap":True,"pdh":None,"pdl":None,
            "orb_high":None,"orb_low":None,"orb_breakout":False,"signal":None,"score":0,"badge":"—"}

def _empty_multitf():
    return {"rsi_1h":None,"rsi_15min":None,"rsi_daily":None,"tf_alignment":0,"tf_total":0,
            "confirmation":"INDISPONIBLE","contradiction":False,"signal":None,"score":0,"badge":"—"}

def _empty_momentum():
    return {"perf_open":0,"spy_perf_open":0,"relative_mom":0,"trend_intraday":"NEUTRE",
            "acceleration":False,"signal":None,"score":0,"badge":"—"}
