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
from claude_scorer import claude_score_batch, verdict_color, conviction_badge
from earnings_sector import check_earnings, get_sector_strength, sector_bonus_score
from gap_detector import detect_gaps
from relative_strength import calc_relative_strength, get_spy_data
from support_resistance import calc_sr_levels
from cheat_sheet import _generate_cheat_sheet

# ─────────────────────────────────────────────
# 📐 FIBONACCI INLINE — FILTRE DE VALIDATION
# ─────────────────────────────────────────────

def calc_fibonacci(hist, swing_lookback=60):
    import numpy as np
    if hist is None or hist.empty or len(hist) < 20:
        return {"swing_low":None,"swing_high":None,"swing_range":0,"retracements":{},"extensions":{},
                "price_context":"NEUTRE","entry_valid":True,"entry_reason":"—","fib_stop":None,
                "fib_target":None,"fib_rr":None,"nearest_support_fib":None,"nearest_resist_fib":None,
                "dist_to_resist":99.0,"dist_to_support":99.0,"signal":None,"score":0,"badge":"—",
                "warning":None,"all_levels":[],"current_level":"—","at_key_level":False}
    try:
        close = hist["Close"]; high = hist["High"]; low = hist["Low"]
        price = float(close.iloc[-1])
        recent = hist.iloc[-swing_lookback:] if len(hist) >= swing_lookback else hist
        swing_low = float(recent["Low"].min())
        swing_high = float(recent["High"].max())
        swing_range = swing_high - swing_low
        if swing_range <= 0 or swing_low <= 0:
            return {"swing_low":None,"swing_high":None,"swing_range":0,"retracements":{},"extensions":{},
                    "price_context":"NEUTRE","entry_valid":True,"entry_reason":"—","fib_stop":None,
                    "fib_target":None,"fib_rr":None,"nearest_support_fib":None,"nearest_resist_fib":None,
                    "dist_to_resist":99.0,"dist_to_support":99.0,"signal":None,"score":0,"badge":"—",
                    "warning":None,"all_levels":[],"current_level":"—","at_key_level":False}

        retrace = {
            "23.6%": round(swing_high - 0.236 * swing_range, 2),
            "38.2%": round(swing_high - 0.382 * swing_range, 2),
            "50.0%": round(swing_high - 0.500 * swing_range, 2),
            "61.8%": round(swing_high - 0.618 * swing_range, 2),
            "78.6%": round(swing_high - 0.786 * swing_range, 2),
        }
        extend = {
            "100%":   round(swing_low + 1.000 * swing_range, 2),
            "127.2%": round(swing_low + 1.272 * swing_range, 2),
            "161.8%": round(swing_low + 1.618 * swing_range, 2),
        }
        all_lvl = {**retrace, **extend}
        below = {k:v for k,v in all_lvl.items() if v < price * 0.999}
        above = {k:v for k,v in all_lvl.items() if v > price * 1.001}
        ns = max(below.items(), key=lambda x: x[1]) if below else None
        nr = min(above.items(), key=lambda x: x[1]) if above else None
        dist_r = round((nr[1] - price) / price * 100, 2) if nr else 99.0
        dist_s = round((price - ns[1]) / price * 100, 2) if ns else 99.0

        rebond = None
        for lvl in ["38.2%","50.0%","61.8%"]:
            lp = retrace[lvl]
            if abs(price - lp) / price * 100 <= 2.0 and price >= lp:
                rebond = (lvl, lp); break

        resist_danger = nr is not None and dist_r <= 2.0
        breakout = price >= swing_high * 0.998
        zone_saine = retrace["50.0%"] <= price <= retrace["23.6%"]

        score = 0; signal = None; warning = None; entry_valid = True

        if breakout:
            price_context = "BREAKOUT"; score = 20; entry_valid = True
            signal = f"Breakout Fib — au-dessus du swing high ${round(swing_high,2)}"
        elif rebond:
            ln, lp = rebond; price_context = "REBOND_KEY"; entry_valid = True
            if ln == "61.8%":   score = 18; signal = f"Rebond Golden Ratio 61.8% à ${lp} — entrée optimale"
            elif ln == "50.0%": score = 14; signal = f"Rebond niveau 50% à ${lp} — setup solide"
            elif ln == "38.2%": score = 10; signal = f"Rebond niveau 38.2% à ${lp}"
        elif resist_danger:
            price_context = "RESISTANCE_PROCHE"; score = -15; entry_valid = False
            signal = f"Résistance Fib {nr[0]} dans {dist_r}% — ENTRÉE RISQUÉE"
            warning = f"Prix à {dist_r}% d'une résistance Fib majeure ({nr[0]} à ${nr[1]}). Risque de rejet."
        elif zone_saine:
            price_context = "ZONE_SAINE"; score = 8; entry_valid = True
            signal = f"Zone Fib saine (entre 23.6% et 50%)"
        else:
            price_context = "NEUTRE"; score = 2; entry_valid = True
            signal = f"Position Fib neutre"

        fib_stop = round(ns[1] * 0.995, 2) if ns else None
        fib_target = None
        for ext_label in ["127.2%","161.8%"]:
            if ext_label in extend and extend[ext_label] > price * 1.01:
                fib_target = extend[ext_label]; break
        fib_rr = None
        if fib_stop and fib_target and fib_stop < price:
            risk = price - fib_stop; reward = fib_target - price
            if risk > 0: fib_rr = round(reward / risk, 2)

        if score >= 18:   badge = "Fib Optimal (rebond cle)"
        elif score >= 12: badge = "Fib Favorable"
        elif score >= 5:  badge = "Fib Zone correcte"
        elif score > 0:   badge = "Fib Neutre"
        elif score <= -10:badge = "RESISTANCE Fib — Eviter"
        else:             badge = "Fib Defavorable"

        return {"swing_low":round(swing_low,2),"swing_high":round(swing_high,2),
                "swing_range":round(swing_range,2),"retracements":retrace,"extensions":extend,
                "price_context":price_context,"entry_valid":entry_valid,"entry_reason":signal or "—",
                "fib_stop":fib_stop,"fib_target":fib_target,"fib_rr":fib_rr,
                "nearest_support_fib":ns,"nearest_resist_fib":nr,
                "dist_to_resist":dist_r,"dist_to_support":dist_s,
                "signal":signal,"score":score,"badge":badge,"warning":warning,
                "all_levels":[],"current_level":ns[0] if ns else "—","at_key_level":rebond is not None or breakout}
    except Exception:
        return {"swing_low":None,"swing_high":None,"swing_range":0,"retracements":{},"extensions":{},
                "price_context":"NEUTRE","entry_valid":True,"entry_reason":"—","fib_stop":None,
                "fib_target":None,"fib_rr":None,"nearest_support_fib":None,"nearest_resist_fib":None,
                "dist_to_resist":99.0,"dist_to_support":99.0,"signal":None,"score":0,"badge":"—",
                "warning":None,"all_levels":[],"current_level":"—","at_key_level":False}


def detect_bollinger_signals(hist, period=20, std_mult=2.0):
    if hist is None or hist.empty or len(hist) < period + 5:
        return {"signal":None,"all_signals":[],"score":0,"badge":"—","bb_upper":None,
                "bb_lower":None,"bb_mid":None,"bb_width":0,"bb_pct":0.5,"width_trend":"NEUTRAL","vol_ratio":1.0}
    try:
        close = hist["Close"]; volume = hist["Volume"]
        price = float(close.iloc[-1])
        ma = close.rolling(period).mean(); std = close.rolling(period).std()
        upper = ma + std_mult * std; lower = ma - std_mult * std
        bb_upper = float(upper.iloc[-1]); bb_lower = float(lower.iloc[-1]); bb_mid = float(ma.iloc[-1])
        bb_width = round((bb_upper-bb_lower)/bb_mid*100,2) if bb_mid>0 else 0
        bb_pct = round((price-bb_lower)/(bb_upper-bb_lower),3) if (bb_upper-bb_lower)>0 else 0.5
        if len(std)>=10:
            wn=float(std.iloc[-1]); wp=float(std.iloc[-6])
            width_trend = "EXPANDING" if wn>wp*1.1 else "CONTRACTING" if wn<wp*0.9 else "NEUTRAL"
        else: width_trend = "NEUTRAL"
        avg_vol=float(volume.rolling(20).mean().iloc[-1]); last_vol=float(volume.iloc[-1])
        vol_ratio=last_vol/avg_vol if avg_vol>0 else 1.0
        signals=[]; score=0
        prev_close=float(close.iloc[-2]); prev_upper=float(upper.iloc[-2]); prev_lower=float(lower.iloc[-2])
        if price>bb_upper and prev_close<=prev_upper:
            if vol_ratio>=1.5: signals.append("BB Breakout haussier avec volume fort"); score+=15
            else: signals.append("BB Breakout haussier"); score+=8
        elif price>bb_upper:
            wd=0; rc=close.iloc[-10:].values; ru=upper.iloc[-10:].values
            for i in range(len(rc)-1,-1,-1):
                if rc[i]>ru[i]: wd+=1
                else: break
            if wd>=3: signals.append(f"BB Walk haussier ({wd} jours)"); score+=12
            elif wd>=2: signals.append(f"BB Walk ({wd} jours)"); score+=6
        if width_trend=="EXPANDING" and bb_width<5.0:
            signals.append("BB Squeeze libere"); score+=10
        elif bb_pct<=0.1 and price>prev_close:
            signals.append(f"Rebond bande BB inferieure (${round(bb_lower,2)})"); score+=8
        elif 0.1<bb_pct<0.45 and price>prev_close:
            signals.append(f"Retour vers BB moyenne"); score+=4
        if price<bb_lower and prev_close>=prev_lower:
            signals.append("BB Breakout baissier"); score-=10
        if score>=15: badge="BB Breakout fort"
        elif score>=10: badge="BB Signal haussier"
        elif score>=5: badge=f"BB Position {round(bb_pct*100,0)}%"
        elif score<0: badge="BB Signal baissier"
        else: badge=f"BB Neutre ({width_trend})"
        return {"signal":signals[0] if signals else None,"all_signals":signals,"score":score,"badge":badge,
                "bb_upper":round(bb_upper,2),"bb_lower":round(bb_lower,2),"bb_mid":round(bb_mid,2),
                "bb_width":bb_width,"bb_pct":bb_pct,"width_trend":width_trend,"vol_ratio":round(vol_ratio,2)}
    except Exception:
        return {"signal":None,"all_signals":[],"score":0,"badge":"—","bb_upper":None,
                "bb_lower":None,"bb_mid":None,"bb_width":0,"bb_pct":0.5,"width_trend":"NEUTRAL","vol_ratio":1.0}

from trading_tools import (
    check_data_quality, save_scan_results, load_scan_results, get_scan_age,
    add_paper_trade, update_paper_results, get_paper_summary,
    add_journal_trade, close_journal_trade, get_journal_summary,
    check_sector_diversity, get_sector_distribution
)

# ─────────────────────────────────────────────
# 📊 INTRADAY SIGNALS INLINE
# VWAP · PDH/PDL · Multi-TF · Momentum
# ─────────────────────────────────────────────

def _calc_vwap(ticker):
    try:
        t = yf.Ticker(ticker)
        intraday = t.history(period="2d", interval="5m")
        if intraday is None or intraday.empty or len(intraday) < 10:
            return None
        intraday.index = pd.to_datetime(intraday.index)
        today = intraday.index[-1].date()
        td = intraday[intraday.index.date == today]
        pd_data = intraday[intraday.index.date < today]
        if td.empty: return None
        tp  = (td["High"] + td["Low"] + td["Close"]) / 3
        cv  = td["Volume"].cumsum()
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
        if vwap and pvwap > 0.5: score += 8; signal = f"✅ Au-dessus VWAP ${vwap} (+{pvwap}%)"
        elif vwap and pvwap < -1.0: score -= 8; signal = f"🔴 Sous VWAP ${vwap} ({pvwap}%)"
        if pdh and price > pdh: score += 12; signal = f"🚀 Breakout PDH ${pdh}"
        elif pdh and pdh and (pdh - price)/price*100 <= 1.5: score -= 5
        if orb_h and price > orb_h: score += 10; signal = signal or f"🚀 ORB Breakout ${orb_h}"
        elif orb_l and price < orb_l: score -= 10
        score = max(-15, min(score, 22))
        badge = "🚀 Setup intraday excellent" if score>=18 else "✅ Favorable" if score>=10 else "~ Neutre" if score>=0 else "🔴 Défavorable"
        return {"vwap":vwap,"price_vs_vwap":pvwap,"above_vwap":price>vwap if vwap else True,
                "pdh":pdh,"pdl":pdl,"orb_high":orb_h,"orb_low":orb_l,
                "orb_breakout":price>orb_h if orb_h else False,
                "signal":signal,"score":score,"badge":badge}
    except Exception:
        return None


def _calc_multitf(ticker, hist_daily=None):
    try:
        t = yf.Ticker(ticker)
        results = {}
        for interval, label in [("1h","1H"),("15m","15min")]:
            try:
                h = t.history(period="5d", interval=interval)
                if h is None or h.empty or len(h)<20: results[label]=None; continue
                c = h["Close"]
                d = c.diff(); g = d.where(d>0,0).rolling(14).mean(); l = -d.where(d<0,0).rolling(14).mean()
                rsi = round(float((100-(100/(1+g/l.clip(lower=1e-10)))).iloc[-1]),1)
                ma = c.rolling(20).mean(); std = c.rolling(20).std()
                up = ma+2*std; lo = ma-2*std
                bp = round((float(c.iloc[-1])-float(lo.iloc[-1]))/(float(up.iloc[-1])-float(lo.iloc[-1])),3) if float(up.iloc[-1])-float(lo.iloc[-1])>0 else 0.5
                e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
                mh=float((e12-e26-(e12-e26).ewm(span=9,adjust=False).mean()).iloc[-1])
                results[label]={"rsi":rsi,"bb_pct":bp,"macd_h":round(mh,4),"bullish":45<=rsi<=75 and mh>0 and bp>0.3,"overbought":rsi>75}
            except Exception:
                results[label]=None
        rsi_d = None
        if hist_daily is not None and not hist_daily.empty:
            try:
                c=hist_daily["Close"]; d=c.diff(); g=d.where(d>0,0).rolling(14).mean(); l=-d.where(d<0,0).rolling(14).mean()
                rsi_d=round(float((100-(100/(1+g/l.clip(lower=1e-10)))).iloc[-1]),1)
            except Exception: pass
        tf_b=0; tf_t=0
        if rsi_d and 45<=rsi_d<=75: tf_b+=1
        if rsi_d: tf_t+=1
        for lb in ["1H","15min"]:
            if results.get(lb): tf_t+=1; tf_b += 1 if results[lb]["bullish"] else 0
        contra = False
        if results.get("1H") and results.get("15min"):
            if abs(results["1H"]["rsi"]-results["15min"]["rsi"])>20: contra=True
            if results["1H"]["bullish"] and results["15min"]["overbought"]: contra=True
        if tf_t==0: conf="INDISPONIBLE"; score=0
        elif contra: conf="CONTRADICTOIRE"; score=-5
        elif tf_b==tf_t: conf="FORT"; score=15
        elif tf_b>=tf_t-1: conf="MODÉRÉ"; score=8
        elif tf_b>=1: conf="FAIBLE"; score=2
        else: conf="BAISSIER"; score=-10
        r1=f"1H RSI:{results['1H']['rsi']}" if results.get("1H") else "1H N/A"
        r2=f"15m RSI:{results['15min']['rsi']}" if results.get("15min") else "15m N/A"
        rd=f"D RSI:{rsi_d}" if rsi_d else "D N/A"
        if conf=="FORT": signal=f"✅ Confirmation {tf_b}/{tf_t} TF — {rd} | {r1} | {r2}"
        elif conf=="CONTRADICTOIRE": signal=f"⚠️ Signal contradictoire entre TF"
        elif conf=="BAISSIER": signal=f"🔴 Tous les TF baissiers"
        else: signal=f"~ Confirmation {tf_b}/{tf_t} TF — {rd} | {r1} | {r2}"
        badge=f"✅ Multi-TF {tf_b}/{tf_t}" if conf=="FORT" else f"⚠️ TF contradictoires" if conf=="CONTRADICTOIRE" else f"🔴 TF baissier" if conf=="BAISSIER" else f"~ Multi-TF {tf_b}/{tf_t}"
        return {"rsi_1h":results.get("1H",{}).get("rsi") if results.get("1H") else None,
                "rsi_15min":results.get("15min",{}).get("rsi") if results.get("15min") else None,
                "rsi_daily":rsi_d,"tf_alignment":tf_b,"tf_total":tf_t,
                "confirmation":conf,"contradiction":contra,"signal":signal,"score":score,"badge":badge}
    except Exception:
        return None


def _calc_intraday_momentum(ticker):
    try:
        t=yf.Ticker(ticker); intra=t.history(period="1d",interval="5m")
        if intra is None or intra.empty or len(intra)<3: return None
        op=float(intra["Open"].iloc[0]); lp=float(intra["Close"].iloc[-1])
        perf=round((lp-op)/op*100,2) if op>0 else 0
        spy_perf=0
        try:
            sh=yf.Ticker("SPY").history(period="1d",interval="5m")
            if sh is not None and not sh.empty:
                so=float(sh["Open"].iloc[0]); sl=float(sh["Close"].iloc[-1])
                spy_perf=round((sl-so)/so*100,2) if so>0 else 0
        except Exception: pass
        rm=round(perf-spy_perf,2)
        acc=False
        if len(intra)>=6:
            mid=len(intra)//2; h1=float(intra["Close"].iloc[mid-1])
            h2=float(intra["Close"].iloc[-1]); p1=(h1-op)/op*100; p2=(h2-h1)/h1*100
            acc=p2>p1*0.5 and p2>0
        if rm>=2.0: score=15; trend="FORT_HAUSSIER"; signal=f"🚀 Momentum intraday +{rm}% vs SPY"
        elif rm>=1.0: score=10; trend="HAUSSIER"; signal=f"✅ Momentum fort +{rm}% vs SPY"
        elif rm>=0.3: score=5; trend="HAUSSIER"; signal=f"~ Momentum positif +{rm}% vs SPY"
        elif rm>=-0.5: score=0; trend="NEUTRE"; signal=f"~ Momentum neutre ({rm}% vs SPY)"
        elif rm>=-1.5: score=-5; trend="BAISSIER"; signal=f"⚠️ Sous-performe SPY {abs(rm)}%"
        else: score=-12; trend="BAISSIER"; signal=f"🔴 Faiblesse intraday {rm}% vs SPY"
        if acc and score>0: score+=3; signal+=" ↗"
        score=max(-15,min(score,18))
        badge=f"🚀 Momentum fort ({perf:+.1f}% | SPY {spy_perf:+.1f}%)" if score>=12 else f"✅ Momentum positif ({perf:+.1f}%)" if score>=5 else f"~ Neutre ({perf:+.1f}%)" if score>=0 else f"🔴 Faible ({perf:+.1f}%)"
        return {"perf_open":perf,"spy_perf_open":spy_perf,"relative_mom":rm,"trend_intraday":trend,"acceleration":acc,"signal":signal,"score":score,"badge":badge}
    except Exception:
        return None

# ─────────────────────────────────────────────
# 📊 BACKTEST INLINE — 6 STRATÉGIES DE SORTIE
# A: +5% fixe | B: +7% fixe | C: Vendredi
# D: Stop suiveur 3% | E: Stop suiveur 5%
# F: 50% à +5% + stop suiveur 3%
# ─────────────────────────────────────────────

STRATEGIES = {
    "A": "A — Vente fixe +5%",
    "B": "B — Vente fixe +7%",
    "C": "C — Vente vendredi",
    "D": "D — Stop suiveur 3%",
    "E": "E — Stop suiveur 5%",
    "F": "F — 50% à +5% + stop 3%",
}

def _simulate_strategy(strategy, opens, highs, lows, closes, entry, stop):
    n = len(closes)
    actual_entry = float(opens[0])
    ratio        = actual_entry / entry if entry and entry > 0 else 1.0
    actual_stop  = stop * ratio if stop else actual_entry * 0.97

    def _pnl(exit_p): return round((exit_p - actual_entry) / actual_entry * 100, 2)
    def _res(p): return "WIN" if p > 0.5 else "LOSS" if p < -0.5 else "BREAKEVEN"

    if strategy == "A":
        tgt = actual_entry * 1.05
        for d in range(n):
            if lows[d] <= actual_stop:  return _pnl(actual_stop), d, "LOSS"
            if highs[d] >= tgt:         return _pnl(tgt), d, "WIN"
        p = _pnl(closes[-1]); return p, n-1, _res(p)

    elif strategy == "B":
        tgt = actual_entry * 1.07
        for d in range(n):
            if lows[d] <= actual_stop:  return _pnl(actual_stop), d, "LOSS"
            if highs[d] >= tgt:         return _pnl(tgt), d, "WIN"
        p = _pnl(closes[-1]); return p, n-1, _res(p)

    elif strategy == "C":
        for d in range(n):
            if lows[d] <= actual_stop:  return _pnl(actual_stop), d, "LOSS"
        p = _pnl(closes[-1]); return p, n-1, _res(p)

    elif strategy == "D":
        trail = actual_stop; hi = actual_entry
        for d in range(n):
            if highs[d] > hi:
                hi = highs[d]; trail = max(trail, hi * 0.97)
            if lows[d] <= trail:
                p = _pnl(trail); return p, d, _res(p)
        p = _pnl(closes[-1]); return p, n-1, _res(p)

    elif strategy == "E":
        trail = actual_stop; hi = actual_entry
        for d in range(n):
            if highs[d] > hi:
                hi = highs[d]; trail = max(trail, hi * 0.95)
            if lows[d] <= trail:
                p = _pnl(trail); return p, d, _res(p)
        p = _pnl(closes[-1]); return p, n-1, _res(p)

    elif strategy == "F":
        tgt_half = actual_entry * 1.05
        half_sold = False; trail = actual_stop; hi = actual_entry; total = 0.0
        for d in range(n):
            if half_sold and highs[d] > hi:
                hi = highs[d]; trail = max(trail, hi * 0.97)
            if not half_sold and highs[d] >= tgt_half:
                total += _pnl(tgt_half) * 0.5
                half_sold = True; hi = tgt_half; trail = max(trail, tgt_half * 0.97)
            if lows[d] <= trail:
                w = 0.5 if half_sold else 1.0
                total += _pnl(trail) * w
                return round(total, 2), d, _res(total)
        w = 0.5 if half_sold else 1.0
        total += _pnl(closes[-1]) * w
        return round(total, 2), n-1, _res(total)

    return 0.0, 0, "BREAKEVEN"


def _backtest_ticker(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="2y")
        if hist is None or hist.empty or len(hist) < 100:
            return []
        trades = []
        hist.index   = pd.to_datetime(hist.index)
        hist["week"] = hist.index.to_period("W")
        groups       = list(hist.groupby("week"))
        for i in range(12, len(groups) - 1):
            hw    = groups[i][1]
            hdata = hist[hist.index <= hw.index[-1]]
            nw    = groups[i+1][1]
            if len(hdata) < 50 or nw.empty: continue
            try:
                close  = hdata["Close"]
                volume = hdata["Volume"]
                price  = float(close.iloc[-1])
                ma50   = float(close.rolling(50).mean().iloc[-1])
                ma200  = float(close.rolling(min(200,len(close))).mean().iloc[-1])
                delta  = close.diff()
                gain   = delta.where(delta>0,0).rolling(14).mean()
                loss   = -delta.where(delta<0,0).rolling(14).mean()
                rsi    = float(100-(100/(1+gain/loss.clip(lower=1e-10))).iloc[-1])
                e12    = close.ewm(span=12,adjust=False).mean()
                e26    = close.ewm(span=26,adjust=False).mean()
                mh     = float((e12-e26-(e12-e26).ewm(span=9,adjust=False).mean()).iloc[-1])
                vr     = float(volume.iloc[-1]/volume.rolling(20).mean().iloc[-1])
                hi_s   = hdata["High"]; lo_s = hdata["Low"]
                tr     = pd.concat([(hi_s-lo_s),(hi_s-close.shift(1)).abs(),(lo_s-close.shift(1)).abs()],axis=1).max(axis=1)
                atr    = float(tr.rolling(14).mean().iloc[-1])

                score = 0
                if price>ma50>ma200: score+=35
                elif price>ma200:    score+=15
                if 45<=rsi<=65:      score+=25
                elif 35<=rsi<45:     score+=18
                elif 65<rsi<=72:     score+=15
                else:                score+=5
                if mh>0.3:           score+=20
                elif mh>0:           score+=14
                if vr>=2:            score+=20
                elif vr>=1.5:        score+=15
                elif vr>=1.1:        score+=10

                n_sig = sum([price>ma50>ma200, 45<=rsi<=65 or 35<=rsi<45, mh>0, vr>=1.5])
                entry = round(price*1.003, 2)
                stop  = round(entry - atr*1.5, 2)

                opens  = nw["Open"].values
                highs  = nw["High"].values
                lows   = nw["Low"].values
                closes = nw["Close"].values

                if len(opens) < 2: continue

                row = {"ticker":ticker,"week":str(groups[i][0]),"score":int(score),"n_signals":int(n_sig),
                       "rsi":round(rsi,1),"macd_hist":round(mh,3),"vol_ratio":round(vr,2),"entry":entry,"stop":stop}
                for s in STRATEGIES:
                    p, d, r = _simulate_strategy(s, opens, highs, lows, closes, entry, stop)
                    row[f"pnl_{s}"] = float(p)
                    row[f"result_{s}"] = str(r)
                trades.append(row)
            except Exception:
                continue
        return trades
    except Exception:
        return []


def run_backtest(tickers, weeks=52, max_workers=8, progress_callback=None):
    all_trades = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_backtest_ticker, t): t for t in tickers}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            done += 1
            trades = future.result()
            if trades: all_trades.extend(trades)
            if progress_callback: progress_callback(done, len(tickers))
    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


def backtest_summary(df):
    if df.empty: return {}
    summary = {}
    for s, label in STRATEGIES.items():
        pc = f"pnl_{s}"; rc = f"result_{s}"
        if pc not in df.columns: continue
        d    = df[[pc,rc,"score"]].dropna()
        n    = len(d)
        if n == 0: continue
        wins = len(d[d[rc]=="WIN"]); losses = len(d[d[rc]=="LOSS"])
        wr   = round(wins/n*100, 1)
        aw   = round(float(d[d[rc]=="WIN"][pc].mean()), 2) if wins>0 else 0.0
        al   = round(float(d[d[rc]=="LOSS"][pc].mean()), 2) if losses>0 else 0.0
        tp   = round(float(d[pc].sum()), 1)
        gp   = d[d[pc]>0][pc].sum(); gl = abs(d[d[pc]<0][pc].sum())
        pf   = round(float(gp/gl), 2) if gl>0 else 9.9
        exp  = round(float(wr/100*aw + (1-wr/100)*al), 2)
        best = round(float(d[pc].max()), 2); worst = round(float(d[pc].min()), 2)
        mc   = 0; cur = 0
        for r in df.sort_values("week")[rc]:
            cur = cur+1 if r=="LOSS" else 0
            mc  = max(mc, cur)
        sc_stats = {}
        for sm,sx,sl in [(80,101,">=80"),(60,80,"60-79"),(0,60,"<60")]:
            sub = d[(d["score"]>=sm)&(d["score"]<sx)]
            if len(sub)>0:
                sw = len(sub[sub[rc]=="WIN"])
                sc_stats[sl] = {"n":int(len(sub)),"win_rate":round(sw/len(sub)*100,1),"avg_pnl":round(float(sub[pc].mean()),2)}
        summary[s] = {"label":str(label),"total":int(n),"wins":int(wins),"losses":int(losses),
                      "win_rate":float(wr),"avg_win":float(aw),"avg_loss":float(al),
                      "total_pnl":float(tp),"best":float(best),"worst":float(worst),
                      "profit_factor":float(pf),"expectancy":float(exp),
                      "max_consec_loss":int(mc),"score_stats":sc_stats}
    return summary

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

        # Vérification qualité des données
        dq_valid, dq_issues, dq_quality = check_data_quality(hist, ticker)
        if not dq_valid:
            return None  # Données corrompues — ignorer ce ticker

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

        # Earnings
        earn     = check_earnings(ticker)
        # Gaps
        gap_data = detect_gaps(hist)
        # Relative Strength vs SPY
        rs_data  = calc_relative_strength(hist)
        # Support & Résistance 52 semaines
        sr_data  = calc_sr_levels(hist)
        # Fibonacci
        fib_data = calc_fibonacci(hist)
        # Bollinger avancé
        bb_data  = detect_bollinger_signals(hist)

        # Signaux intraday (optionnel — activé dans sidebar)
        vwap_data = None; multitf_data = None; mom_data = None
        if st.session_state.get("use_intraday", False):
            vwap_data   = _calc_vwap(ticker)
            multitf_data= _calc_multitf(ticker, hist)
            mom_data    = _calc_intraday_momentum(ticker)

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
            # Earnings
            "Earnings_Badge":  earn["badge"],
            "Earnings_Date":   earn["earnings_date"],
            "Earnings_Days":   earn["days_until"],
            "Earnings_Risk":   earn["risk_level"],
            "Earnings_Avoid":  earn["should_avoid"],
            # Gaps
            "Gap_Badge":       gap_data["badge"],
            "Gap_Signal":      gap_data["top_signal"],
            "Gap_Score":       gap_data["score"],
            "Gap_Pct":         gap_data["recent_gap_pct"],
            "Gap_Direction":   gap_data["gap_direction"],
            "Gap_Support":     gap_data["nearest_support"],
            "Gap_Summary":     gap_data["summary"],
            # Relative Strength vs SPY
            "RS_Score":        rs_data["rs_score"],
            "RS_Badge":        rs_data["badge"],
            "RS_Signal":       rs_data["signal"],
            "RS_Trend":        rs_data["rs_trend"],
            "RS_5d":           rs_data["rs_5d"],
            "RS_10d":          rs_data["rs_10d"],
            "RS_20d":          rs_data["rs_20d"],
            "RS_Bonus":        rs_data["bonus_pts"],
            "RS_Perf5d":       rs_data["perf_5d"],
            "SPY_Perf5d":      rs_data["spy_perf_5d"],
            "RS_Outperform":   rs_data["outperform"],
            # Support & Résistance 52 semaines
            "SR_High52w":      sr_data["high_52w"],
            "SR_Low52w":       sr_data["low_52w"],
            "SR_Position":     sr_data["position_pct"],
            "SR_DistHigh":     sr_data["dist_to_high"],
            "SR_DistLow":      sr_data["dist_to_low"],
            "SR_Signal":       sr_data["signal"],
            "SR_Badge":        sr_data["badge"],
            "SR_Score":        sr_data["score"],
            "SR_Quality":      sr_data["setup_quality"],
            "SR_StopNatural":  sr_data["stop_natural"],
            "SR_TargetNatural":sr_data["target_natural"],
            "SR_Round":        sr_data["nearest_round"],
            # Fibonacci V2 — filtre de validation
            "FIB_Signal":      fib_data["signal"],
            "FIB_Score":       fib_data["score"],
            "FIB_Badge":       fib_data["badge"],
            "FIB_Context":     fib_data["price_context"],
            "FIB_EntryValid":  fib_data["entry_valid"],
            "FIB_EntryReason": fib_data["entry_reason"],
            "FIB_Stop":        fib_data["fib_stop"],
            "FIB_Target":      fib_data["fib_target"],
            "FIB_RR":          fib_data["fib_rr"],
            "FIB_DistResist":  fib_data["dist_to_resist"],
            "FIB_Warning":     fib_data["warning"],
            "FIB_Support":     fib_data["nearest_support_fib"][1] if fib_data["nearest_support_fib"] else None,
            "FIB_Resist":      fib_data["nearest_resist_fib"][1] if fib_data["nearest_resist_fib"] else None,
            "FIB_AtKey":       fib_data["at_key_level"],
            # Bollinger avancé
            "BB_Signal":       bb_data["signal"],
            "BB_Score":        bb_data["score"],
            "BB_Badge":        bb_data["badge"],
            "BB_Width":        bb_data["bb_width"],
            "BB_Pct":          bb_data["bb_pct"],
            "BB_WidthTrend":   bb_data["width_trend"],
            "BB_Upper":        bb_data["bb_upper"],
            "BB_Lower":        bb_data["bb_lower"],
            # Intraday (optionnel)
            "ID_VWAP":         vwap_data["vwap"] if vwap_data else None,
            "ID_VWAP_Dist":    vwap_data["price_vs_vwap"] if vwap_data else None,
            "ID_AboveVWAP":    vwap_data["above_vwap"] if vwap_data else None,
            "ID_PDH":          vwap_data["pdh"] if vwap_data else None,
            "ID_PDL":          vwap_data["pdl"] if vwap_data else None,
            "ID_ORB_Break":    vwap_data["orb_breakout"] if vwap_data else None,
            "ID_VWAP_Badge":   vwap_data["badge"] if vwap_data else "—",
            "ID_VWAP_Signal":  vwap_data["signal"] if vwap_data else None,
            "ID_VWAP_Score":   vwap_data["score"] if vwap_data else 0,
            "ID_TF_Confirm":   multitf_data["confirmation"] if multitf_data else "—",
            "ID_TF_Align":     multitf_data["tf_alignment"] if multitf_data else 0,
            "ID_TF_Total":     multitf_data["tf_total"] if multitf_data else 0,
            "ID_RSI_1H":       multitf_data["rsi_1h"] if multitf_data else None,
            "ID_RSI_15min":    multitf_data["rsi_15min"] if multitf_data else None,
            "ID_TF_Badge":     multitf_data["badge"] if multitf_data else "—",
            "ID_TF_Signal":    multitf_data["signal"] if multitf_data else None,
            "ID_TF_Score":     multitf_data["score"] if multitf_data else 0,
            "ID_Mom_Rel":      mom_data["relative_mom"] if mom_data else None,
            "ID_Mom_Trend":    mom_data["trend_intraday"] if mom_data else "—",
            "ID_Mom_Badge":    mom_data["badge"] if mom_data else "—",
            "ID_Mom_Signal":   mom_data["signal"] if mom_data else None,
            "ID_Mom_Score":    mom_data["score"] if mom_data else 0,
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

    # Bonus signaux intraday (max +25, min -20) — si activé
    try:
        id_vwap_score = int(row.get("ID_VWAP_Score", 0) or 0)
        id_tf_score   = int(row.get("ID_TF_Score", 0) or 0)
        id_mom_score  = int(row.get("ID_Mom_Score", 0) or 0)
        id_total      = id_vwap_score + id_tf_score + id_mom_score
        id_tf_conf    = str(row.get("ID_TF_Confirm", "") or "")
        if id_total != 0:
            score += max(-20, min(id_total, 25))
        if id_tf_conf == "CONTRADICTOIRE":
            score -= 10
            reasons.append("⚠️ TF contradictoires — prudence")
        elif id_tf_conf == "FORT" and id_total > 10:
            reasons.append("✅ Multi-TF alignés + Momentum fort")
        elif id_tf_conf == "BAISSIER":
            reasons.append("🔴 Multi-TF baissier")
    except Exception:
        pass

    # Fibonacci V2 — filtre de validation (max +20, min -20)
    try:
        fib_score    = int(row.get("FIB_Score", 0) or 0)
        fib_signal   = str(row.get("FIB_Signal") or "")
        fib_valid    = row.get("FIB_EntryValid", True)
        fib_warning  = str(row.get("FIB_Warning") or "")
        fib_context  = str(row.get("FIB_Context", "") or "")
        fib_dist_r   = float(row.get("FIB_DistResist", 99) or 99)

        if fib_score != 0:
            score += fib_score
            if fib_signal and fib_signal not in ["None","—",""]:
                reasons.append(f"📐 {fib_signal[:50]}")

        # Pénalité forte si entrée invalide (résistance Fib proche)
        if fib_valid == False:
            score -= 10  # Pénalité additionnelle
            reasons.append(f"🔴 FIB: Entrée invalide — résistance dans {fib_dist_r}%")

    except Exception:
        pass

    # Bonus Bollinger avancé (max +15, min -10)
    try:
        bb_score  = int(row.get("BB_Score", 0) or 0)
        bb_signal = str(row.get("BB_Signal") or "")
        if bb_score != 0:
            score += bb_score
            if bb_signal and bb_signal not in ["None","—",""]:
                reasons.append(f"📊 {bb_signal[:45]}")
    except Exception:
        pass

    # Bonus Support/Résistance 52 semaines (max +22, min -15)
    try:
        sr_score  = int(row.get("SR_Score", 0) or 0)
        sr_signal = str(row.get("SR_Signal") or "")
        sr_qual   = str(row.get("SR_Quality", "") or "")
        if sr_score != 0:
            score += sr_score
            if sr_signal and sr_signal not in ["None","—",""]:
                reasons.append(f"📐 {sr_signal[:45]}")
    except Exception:
        pass

    # Bonus Relative Strength vs SPY (max +15, min -8)
    try:
        rs_bonus  = int(row.get("RS_Bonus", 0) or 0)
        rs_signal = str(row.get("RS_Signal") or "")
        rs_trend  = str(row.get("RS_Trend", "") or "")
        if rs_bonus != 0:
            score += rs_bonus
            if rs_signal and rs_signal not in ["None","—","~","~ RS neutre vs SPY",""]:
                reasons.append(rs_signal[:50])
    except Exception:
        pass

    # Bonus / Pénalité Gap (max +20, min -15)
    try:
        gap_score = int(row.get("Gap_Score", 0) or 0)
        gap_sig   = str(row.get("Gap_Signal") or "")
        gap_dir   = str(row.get("Gap_Direction", "NONE") or "NONE")
        if gap_score > 0:
            score += min(gap_score, 20)
            if gap_sig and gap_sig not in ["None", "—", ""]:
                reasons.append(f"✅ {gap_sig[:45]}")
        elif gap_score < 0:
            score += max(gap_score, -15)
            reasons.append(f"🔴 Gap baissier récent")
    except Exception:
        pass

    # Pénalité Earnings — risque élevé cette semaine
    try:
        earn_avoid = bool(row.get("Earnings_Avoid", False))
        earn_risk  = str(row.get("Earnings_Risk", "") or "")
        earn_days  = row.get("Earnings_Days", None)
        if earn_avoid and earn_risk == "ÉLEVÉ":
            score -= 25
            reasons.append(f"🔴 Earnings dans {earn_days}j — risque élevé")
        elif earn_avoid and earn_risk == "MODÉRÉ":
            score -= 15
            reasons.append(f"⚠️ Earnings dans {earn_days}j — prudence")
    except Exception:
        pass

    # Bonus secteur fort
    try:
        sec_bonus, sec_label = sector_bonus_score(
            str(row.get("Sector", "") or ""),
            st.session_state.get("sector_data", {})
        )
        if sec_bonus != 0:
            score += sec_bonus
            if sec_label and sec_label != "—":
                reasons.append(sec_label)
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
    api_key = st.text_input("Clé API Anthropic", type="password")

    use_claude_scorer = st.checkbox("🎯 Score dynamique Claude (Top 20)", value=False,
                                     help="Claude re-score les 20 meilleures actions avec raisonnement IA")
    if use_claude_scorer:
        claude_top_n = st.slider("Actions à scorer par Claude", 5, 30, 15)
        st.markdown("""<div style='background:#a78bfa15;border:1px solid #a78bfa33;
            border-radius:6px;padding:8px;font-size:0.78rem;color:#c4b5fd;margin-top:4px;'>
            🤖 Claude analyse chaque setup en profondeur<br>
            et remplace le score algorithmique<br>
            ⏱️ ~{} secondes pour {} actions
        </div>""".format(claude_top_n * 2, claude_top_n), unsafe_allow_html=True)
    else:
        claude_top_n = 15

    use_claude = st.checkbox("💬 Analyse Claude Top 5", value=False,
                              help="Analyse textuelle détaillée pour les 5 meilleurs trades")
    st.markdown("---")
    st.markdown("### 📊 Signaux Intraday")
    use_intraday = st.checkbox("⚡ Activer signaux intraday", value=False,
                                help="VWAP · PDH/PDL · Multi-TF · Momentum vs SPY — données 15min délai")
    if use_intraday:
        st.markdown("""<div style='background:#4a90d015;border:1px solid #4a90d033;
            border-radius:6px;padding:8px;font-size:0.78rem;color:#93c5fd;margin-top:4px;'>
            ⚡ VWAP + PDH/PDL + Opening Range<br>
            ⚡ Multi-timeframe RSI (Daily+1H+15min)<br>
            ⚡ Momentum intraday vs SPY<br>
            ⚠️ Délai 15min — indicatif seulement<br>
            ⏱️ Ralentit le scan (~2x plus long)
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 Filtres tableau")
    signal_filter = st.multiselect(
        "Signaux",
        ["🟢 STRONG BUY","🟢 BUY","🟡 HOLD","🔴 AVOID","🟡 HOLD ⚠️"],
        default=["🟢 STRONG BUY","🟢 BUY"]
    )
    filter_earnings = st.checkbox(
        "📅 Exclure earnings cette semaine", value=True,
        help="Exclut les actions avec earnings dans les 5 prochains jours"
    )
    filter_top_sectors = st.checkbox(
        "💪 Top 5 secteurs seulement", value=False,
        help="Garde seulement les actions dans les 5 secteurs les plus forts"
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

# ── FORCE SECTORIELLE ──
with st.spinner("Analyse sectorielle..."):
    sector_data = get_sector_strength()

# Stocker en session state pour ai_score
st.session_state["sector_data"]  = sector_data
st.session_state["use_intraday"] = use_intraday

if sector_data["rankings"]:
    with st.expander("💪 Force Sectorielle cette semaine", expanded=False):
        st.markdown(f"**🔥 Secteur dominant : {sector_data['top_sector']}** &nbsp;|&nbsp; 🔴 Secteur faible : {sector_data['worst_sector']}")
        rows_sec = []
        for sector, data in sector_data["rankings"]:
            badge = sector_data["sector_badges"].get(sector, "")
            rows_sec.append({
                "Secteur":    sector,
                "Badge":      badge,
                "Mom 5j":     f"{'+' if data['mom_5d']>=0 else ''}{data['mom_5d']}%",
                "Mom 20j":    f"{'+' if data['mom_20d']>=0 else ''}{data['mom_20d']}%",
                "vs MA20":    "✅" if data['above_ma20'] else "❌",
                "Force":      str(data['strength']),
            })
        st.dataframe(pd.DataFrame(rows_sec).set_index("Secteur"), use_container_width=True)

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
            pnl_cols = [c for c in df_bt.columns if c.startswith("pnl_")]
            stats    = backtest_summary(df_bt)
            total_trades = len(df_bt)

            # Détecter quelle version du backtest est utilisée
            valid_keys = [k for k in stats.keys() if k in ["A","B","C","D","E","F"]]
            is_new_backtest = len(valid_keys) > 0

            if is_new_backtest:
                # ── NOUVEAU BACKTEST : 6 stratégies ──
                stats = {k: stats[k] for k in valid_keys}
                st.markdown(f"### 📈 Comparatif 6 Stratégies — {total_trades} trades simulés")

                strat_labels = {
                    "A": "A — +5% fixe",
                    "B": "B — +7% fixe",
                    "C": "C — Vendredi",
                    "D": "D — Stop 3%",
                    "E": "E — Stop 5%",
                    "F": "F — 50%+Stop 3%",
                }

                try:
                    best_strat = max(
                        stats.keys(),
                        key=lambda s: float(stats[s].get("expectancy", -99) or -99)
                    )
                except Exception:
                    best_strat = list(stats.keys())[0]

                rows_display = []
                for strat, data in stats.items():
                    try:
                        crown = " 👑" if strat == best_strat else ""
                        rows_display.append({
                            "Strategie":     strat_labels.get(strat, strat) + crown,
                            "Trades":        str(data.get("total", 0)),
                            "Win Rate":      str(data.get("win_rate", 0)) + "%",
                            "Expectancy":    str(data.get("expectancy", 0)) + "%",
                            "Profit Factor": str(data.get("profit_factor", 0)),
                            "Gain moyen":    "+" + str(data.get("avg_win", 0)) + "%",
                            "Perte moyenne": str(data.get("avg_loss", 0)) + "%",
                            "PnL Total":     str(data.get("total_pnl", 0)) + "%",
                            "Max Pertes":    str(data.get("max_consec_loss", 0)),
                        })
                    except Exception:
                        pass

                if rows_display:
                    st.dataframe(
                        pd.DataFrame(rows_display).set_index("Strategie"),
                        use_container_width=True
                    )

                st.markdown("---")
                selected_strat = st.selectbox(
                    "📊 Voir le détail d'une stratégie",
                    options=list(stats.keys()),
                    format_func=lambda s: strat_labels.get(s, s),
                    index=0
                )

                if selected_strat not in stats:
                    selected_strat = list(stats.keys())[0]

                sd = stats[selected_strat]
                st.markdown(f"#### Détail — {strat_labels.get(selected_strat, selected_strat)}")

                try:
                    d1,d2,d3,d4 = st.columns(4)
                    d1.metric("Meilleur trade",    f"+{sd.get('best', 0)}%")
                    d2.metric("Pire trade",         f"{sd.get('worst', 0)}%")
                    d3.metric("Max pertes consec.", str(sd.get("max_consec_loss", 0)))
                    d4.metric("PnL cumulé",         f"{sd.get('total_pnl', 0)}%")
                except Exception:
                    pass

                try:
                    score_stats = sd.get("score_stats") or {}
                    if score_stats:
                        st.markdown("**Performance par niveau de score IA :**")
                        rows_ss = []
                        for slabel, sdata in score_stats.items():
                            rows_ss.append({
                                "Score":    str(slabel),
                                "Win Rate": str(sdata.get("win_rate", 0)) + "%",
                                "PnL moy":  str(sdata.get("avg_pnl", 0)) + "%",
                                "N trades": str(sdata.get("n", 0)),
                            })
                        st.dataframe(pd.DataFrame(rows_ss).set_index("Score"), use_container_width=True)
                except Exception:
                    pass

                pnl_col = f"pnl_{selected_strat}"
                res_col = f"result_{selected_strat}"

                if pnl_col in df_bt.columns:
                    bt_tab1, bt_tab2 = st.tabs(["Courbe capital", "Win Rate/Score"])
                    with bt_tab1:
                        try:
                            df_s2 = df_bt.sort_values("week").copy()
                            df_s2["PnL cumule"] = df_s2[pnl_col].fillna(0).cumsum()
                            st.line_chart(df_s2["PnL cumule"])
                        except Exception as e:
                            st.error(f"Graphique indisponible: {e}")
                    with bt_tab2:
                        try:
                            df_bt2 = df_bt.copy()
                            df_bt2["score_bucket"] = pd.cut(df_bt2["score"],
                                bins=[0,40,50,60,70,80,101],
                                labels=["<40","40-50","50-60","60-70","70-80",">=80"])
                            wr_s = df_bt2.groupby("score_bucket", observed=True).apply(
                                lambda x: round(len(x[x[res_col]=="WIN"])/max(len(x),1)*100, 1)
                            ).reset_index()
                            wr_s.columns = ["Score","Win Rate %"]
                            st.bar_chart(wr_s.set_index("Score"))
                        except Exception as e:
                            st.error(f"Graphique indisponible: {e}")

            else:
                # ── ANCIEN BACKTEST : version simple ──
                st.info("ℹ️ Ancienne version du backtest détectée — mets à jour backtest.py pour les 6 stratégies.")
                st.markdown(f"### 📈 Résultats — {total_trades} trades simulés")

                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Win Rate",      f"{stats.get('win_rate', 0)}%")
                m2.metric("Profit Factor", f"{stats.get('profit_factor', 0)}")
                m3.metric("Gain moyen",    f"+{stats.get('avg_win', 0)}%")
                m4.metric("PnL Total",     f"{stats.get('total_pnl', 0)}%")

                if "pnl_pct" in df_bt.columns:
                    df_sorted = df_bt.sort_values("week").copy()
                    df_sorted["PnL cumule"] = df_sorted["pnl_pct"].fillna(0).cumsum()
                    st.line_chart(df_sorted["PnL cumule"])

            # ── EXPORT (commun aux deux versions) ──
            bt_excel = BytesIO()
            with pd.ExcelWriter(bt_excel, engine="openpyxl") as writer:
                df_bt.to_excel(writer, index=False, sheet_name="Trades")
                try:
                    strat_labels_export = {
                        "A": "A — +5% fixe", "B": "B — +7% fixe",
                        "C": "C — Vendredi", "D": "D — Stop 3%",
                        "E": "E — Stop 5%",  "F": "F — 50%+Stop 3%",
                    }
                    rows = []
                    for s, d in stats.items():
                        if s in ["A","B","C","D","E","F"]:
                            rows.append({
                                "Strategie":    strat_labels_export.get(s, s),
                                "Win Rate %":   d.get("win_rate", 0),
                                "Expectancy %": d.get("expectancy", 0),
                                "Profit Factor":d.get("profit_factor", 0),
                                "Gain Moy %":   d.get("avg_win", 0),
                                "Perte Moy %":  d.get("avg_loss", 0),
                                "PnL Total %":  d.get("total_pnl", 0),
                                "Max Pertes":   d.get("max_consec_loss", 0),
                            })
                    if rows:
                        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="Comparatif")
                except Exception:
                    pass
            st.download_button("⬇️ Télécharger résultats backtest", data=bt_excel.getvalue(),
                file_name=f"backtest_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
# ── Bouton charger dernier scan ──
saved_scan = load_scan_results()
if saved_scan:
    age = get_scan_age(saved_scan.get("timestamp", ""))
    st.markdown(
        f"<div class='prefilter-banner'>💾 Dernier scan sauvegardé : "
        f"<strong>{age}</strong> · {saved_scan.get('n_actions','?')} actions · "
        f"Marché {saved_scan.get('regime','?')}</div>",
        unsafe_allow_html=True
    )

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

    # Charger SPY une seule fois pour la Relative Strength
    with st.spinner("Chargement données SPY pour Relative Strength..."):
        spy_close = get_spy_data()

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

    # Sauvegarder les résultats du scan
    save_scan_results(df, market_status, regime)
    st.success(f"✅ Résultats sauvegardés — {len(df)} actions analysées")

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
        report = report[report["Conv_N"] >= min_signals]
        # Limite sectorielle — max 2 titres par secteur
        report = check_sector_diversity(report, top_n=top_n, max_per_sector=2)
        # Filtre Fibonacci — en mode strict, exclure les entrées invalides
        if strict_mode and "FIB_EntryValid" in report.columns:
            n_before = len(report)
            report_fib = report[report["FIB_EntryValid"] != False]
            n_removed  = n_before - len(report_fib)
            if n_removed > 0:
                st.info(f"📐 {n_removed} titre(s) exclu(s) — résistance Fibonacci trop proche")

    # ── SCORE CLAUDE DYNAMIQUE ──
    if not report.empty and use_claude_scorer and api_key:
        st.markdown("---")
        st.markdown("### 🤖 Score Claude Dynamique en cours...")
        cs_prog = st.progress(0)
        cs_stat = st.empty()

        def cs_cb(done, total):
            cs_prog.progress(done / total)
            cs_stat.markdown(f"🤖 Claude analyse `{done}/{total}` actions...")

        report = claude_score_batch(
            report,
            market_status=market_status,
            api_key=api_key,
            top_n=min(claude_top_n, len(report)),
            delay=0.3,
            progress_callback=cs_cb,
        )
        cs_stat.empty()
        cs_prog.empty()
        st.markdown(
            "<div style='color:#a78bfa;font-size:0.85rem;margin-bottom:8px;'>"
            f"✅ Score Claude calculé pour {min(claude_top_n, len(report))} actions"
            " — rapport trié par score Claude</div>",
            unsafe_allow_html=True
        )
    elif use_claude_scorer and not api_key:
        st.warning("⚠️ Entrez votre clé API Anthropic dans la sidebar pour activer le score Claude.")

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

            # Score + Prix + Claude
            claude_ok      = bool(row.get("claude_ok", False))
            claude_score_v = row.get("claude_score", None)
            claude_verdict_v = str(row.get("claude_verdict", "") or "")
            claude_conviction_v = str(row.get("claude_conviction", "") or "")
            claude_raison_v = str(row.get("claude_raison", "") or "")
            claude_risque_v = str(row.get("claude_risque", "") or "")
            claude_stop_v  = row.get("claude_stop_adj", None)

            if claude_ok and claude_score_v is not None:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Score Algo", f"{score_fin}/100")
                vc = verdict_color(claude_verdict_v)
                c2.markdown(
                    f"**Score Claude**\n\n"
                    f"<span style='font-size:1.5rem;font-weight:700;color:{vc};'>"
                    f"{claude_score_v}/100</span>",
                    unsafe_allow_html=True
                )
                c3.markdown(
                    f"**Verdict Claude**\n\n"
                    f"<span style='color:{vc};font-weight:700;'>{claude_verdict_v}</span><br>"
                    f"<span style='color:#64748b;font-size:0.8rem;'>{conviction_badge(claude_conviction_v)}</span>",
                    unsafe_allow_html=True
                )
                c4.metric("Prix actuel", f"${row['Prix']}")
            else:
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

            # ── SIGNAUX INTRADAY (si activés) ──
            if st.session_state.get("use_intraday", False):
                id_vwap_b   = str(row.get("ID_VWAP_Badge", "") or "")
                id_vwap_s   = str(row.get("ID_VWAP_Signal", "") or "")
                id_vwap_v   = row.get("ID_VWAP", None)
                id_vwap_d   = row.get("ID_VWAP_Dist", None)
                id_pdh      = row.get("ID_PDH", None)
                id_pdl      = row.get("ID_PDL", None)
                id_orb      = row.get("ID_ORB_Break", False)
                id_tf_b     = str(row.get("ID_TF_Badge", "") or "")
                id_tf_s     = str(row.get("ID_TF_Signal", "") or "")
                id_tf_conf  = str(row.get("ID_TF_Confirm", "") or "")
                id_rsi_1h   = row.get("ID_RSI_1H", None)
                id_rsi_15   = row.get("ID_RSI_15min", None)
                id_mom_b    = str(row.get("ID_Mom_Badge", "") or "")
                id_mom_rel  = row.get("ID_Mom_Rel", None)

                st.markdown("**⚡ Signaux Intraday (délai 15min) :**")

                # VWAP
                if id_vwap_b and id_vwap_b != "—":
                    vwap_c = "#00ff88" if (row.get("ID_VWAP_Score",0) or 0)>=0 else "#f87171"
                    st.markdown(f"<span style='color:{vwap_c};'>📊 VWAP: {id_vwap_b}</span>", unsafe_allow_html=True)
                    parts = []
                    if id_vwap_v:  parts.append(f"VWAP: `${id_vwap_v}`")
                    if id_pdh:     parts.append(f"PDH: `${id_pdh}`")
                    if id_pdl:     parts.append(f"PDL: `${id_pdl}`")
                    if id_orb:     parts.append(f"✅ ORB Breakout")
                    if parts: st.markdown(" | ".join(parts))

                # Multi-TF
                if id_tf_b and id_tf_b != "—":
                    tf_c = "#00ff88" if id_tf_conf=="FORT" else "#f87171" if id_tf_conf in ["CONTRADICTOIRE","BAISSIER"] else "#fbbf24"
                    st.markdown(f"<span style='color:{tf_c};'>🔄 Multi-TF: {id_tf_b}</span>", unsafe_allow_html=True)
                    rsi_parts = []
                    if id_rsi_1h:  rsi_parts.append(f"RSI 1H: `{id_rsi_1h}`")
                    if id_rsi_15:  rsi_parts.append(f"RSI 15m: `{id_rsi_15}`")
                    if rsi_parts: st.markdown(" | ".join(rsi_parts))

                # Momentum
                if id_mom_b and id_mom_b != "—":
                    mom_c = "#00ff88" if (row.get("ID_Mom_Score",0) or 0)>0 else "#f87171"
                    st.markdown(f"<span style='color:{mom_c};'>🚀 Momentum: {id_mom_b}</span>", unsafe_allow_html=True)

            # ── FIBONACCI V2 ──
            fib_badge    = str(row.get("FIB_Badge", "") or "")
            fib_context  = str(row.get("FIB_Context", "NEUTRE") or "NEUTRE")
            fib_valid    = row.get("FIB_EntryValid", True)
            fib_signal_v = str(row.get("FIB_Signal", "") or "")
            fib_warning  = str(row.get("FIB_Warning", "") or "")
            fib_stop_v   = row.get("FIB_Stop", None)
            fib_target_v = row.get("FIB_Target", None)
            fib_rr_v     = row.get("FIB_RR", None)
            fib_support  = row.get("FIB_Support", None)
            fib_resist   = row.get("FIB_Resist", None)
            fib_dist_r   = row.get("FIB_DistResist", None)
            fib_score_v  = int(row.get("FIB_Score", 0) or 0)

            # Couleur selon contexte
            fib_color = {
                "REBOND_KEY":        "#00ff88",
                "BREAKOUT":          "#00ff88",
                "ZONE_SAINE":        "#86efac",
                "NEUTRE":            "#fbbf24",
                "RESISTANCE_PROCHE": "#f87171",
            }.get(fib_context, "#fbbf24")

            fib_display = fib_badge if fib_badge and fib_badge != "—" else f"~ Fibonacci {fib_context}"
            st.markdown(
                f"**📐 Fibonacci :** <span style='color:{fib_color};font-weight:700;'>{fib_display}</span>",
                unsafe_allow_html=True
            )

            # Avertissement rouge si entrée invalide
            if not fib_valid and fib_warning:
                st.markdown(
                    f"<div style='background:#f871711a;border-left:3px solid #f87171;"
                    f"border-radius:4px;padding:8px;margin:4px 0;font-size:0.85rem;color:#f87171;'>"
                    f"⚠️ {fib_warning}</div>",
                    unsafe_allow_html=True
                )
            elif fib_signal_v and fib_signal_v not in ["None","—",""]:
                st.markdown(
                    f"<span style='color:{fib_color};font-size:0.85rem;'>  {fib_signal_v}</span>",
                    unsafe_allow_html=True
                )

            # Niveaux Fib clés
            fib_parts = []
            if fib_support: fib_parts.append(f"Support: `${fib_support}`")
            if fib_resist:  fib_parts.append(f"Résistance: `${fib_resist}`")
            if fib_dist_r and fib_dist_r < 99: fib_parts.append(f"Dist. résist: `{fib_dist_r}%`")
            if fib_parts: st.markdown(" &nbsp;|&nbsp; ".join(fib_parts))

            # Stop + Target + R/R Fibonacci
            if fib_stop_v and fib_target_v:
                fib_rr_str = f"R/R Fib: `{fib_rr_v}:1`" if fib_rr_v else ""
                st.markdown(
                    f"🛡️ Stop Fib: `${fib_stop_v}` &nbsp;|&nbsp; "
                    f"🎯 Target Fib: `${fib_target_v}`"
                    + (f" &nbsp;|&nbsp; {fib_rr_str}" if fib_rr_str else "")
                )

            # Bollinger avancé — toujours afficher
            bb_badge  = str(row.get("BB_Badge", "") or "")
            bb_trend  = str(row.get("BB_WidthTrend", "") or "")
            bb_signal_v = str(row.get("BB_Signal", "") or "")
            bb_score_v  = int(row.get("BB_Score", 0) or 0)
            bb_color   = "#00ff88" if bb_score_v > 0 else "#fbbf24" if bb_score_v == 0 else "#f87171"
            trend_icon = "📈" if bb_trend=="EXPANDING" else "📉" if bb_trend=="CONTRACTING" else "~"
            bb_display = bb_badge if bb_badge and bb_badge != "—" else f"~ BB {bb_trend}"
            st.markdown(
                f"**📊 Bollinger :** <span style='color:{bb_color};'>{bb_display}</span> {trend_icon}",
                unsafe_allow_html=True
            )
            if bb_signal_v and bb_signal_v not in ["None","—",""]:
                st.markdown(f"<span style='color:{bb_color};font-size:0.85rem;'>  {bb_signal_v}</span>", unsafe_allow_html=True)

            # Support & Résistance 52 semaines
            sr_badge    = str(row.get("SR_Badge", "") or "")
            sr_high     = row.get("SR_High52w", None)
            sr_low      = row.get("SR_Low52w", None)
            sr_pos      = row.get("SR_Position", None)
            sr_stop_nat = row.get("SR_StopNatural", None)
            sr_tgt_nat  = row.get("SR_TargetNatural", None)

            if sr_badge and sr_badge not in ["—", ""]:
                sr_color = "#00ff88" if (row.get("SR_Score", 0) or 0) >= 10 else \
                           "#fbbf24" if (row.get("SR_Score", 0) or 0) >= 0 else "#f87171"
                st.markdown(
                    f"**📐 S/R 52 sem :** <span style='color:{sr_color};'>{sr_badge}</span>",
                    unsafe_allow_html=True
                )
                info_parts = []
                if sr_high: info_parts.append(f"High 52w: `${sr_high}`")
                if sr_low:  info_parts.append(f"Low 52w: `${sr_low}`")
                if sr_pos is not None: info_parts.append(f"Position: `{sr_pos}%` du range")
                if info_parts:
                    st.markdown(" &nbsp;|&nbsp; ".join(info_parts))
                if sr_stop_nat and sr_tgt_nat:
                    st.markdown(
                        f"🛡️ Stop S/R: `${sr_stop_nat}` &nbsp;|&nbsp; "
                        f"🎯 Target S/R: `${sr_tgt_nat}`"
                    )

            # Relative Strength
            rs_badge    = str(row.get("RS_Badge", "") or "")
            rs_trend_v  = str(row.get("RS_Trend", "") or "")
            rs_perf     = row.get("RS_Perf5d", None)
            spy_perf    = row.get("SPY_Perf5d", None)
            rs_5d_v     = row.get("RS_5d", None)
            if rs_badge and rs_badge not in ["—", ""]:
                rs_color = "#00ff88" if row.get("RS_Outperform") else "#f87171"
                rs_text  = f"**💪 RS :** {rs_badge}"
                if rs_perf is not None and spy_perf is not None:
                    diff = round(float(rs_perf) - float(spy_perf), 2)
                    sign = "+" if diff >= 0 else ""
                    rs_text += f" &nbsp;|&nbsp; Action: `{'+' if float(rs_perf)>=0 else ''}{rs_perf}%` vs SPY: `{'+' if float(spy_perf)>=0 else ''}{spy_perf}%` → <span style='color:{rs_color};'>{sign}{diff}% vs marché</span>"
                st.markdown(rs_text, unsafe_allow_html=True)

            # Gap signal
            gap_badge   = str(row.get("Gap_Badge", "") or "")
            gap_signal  = str(row.get("Gap_Signal", "") or "")
            gap_score_v = row.get("Gap_Score", 0)
            gap_support = row.get("Gap_Support", None)
            if gap_badge and gap_badge not in ["—", ""]:
                gap_color = "#00ff88" if (gap_score_v or 0) > 0 else "#f87171"
                st.markdown(
                    f"<span style='color:{gap_color};font-weight:700;font-size:0.85rem;'>{gap_badge}</span>",
                    unsafe_allow_html=True
                )
                if gap_support:
                    st.markdown(f"🛡️ Support gap : `${gap_support}` — niveau de stop solide")

            # Earnings warning
            earn_badge = str(row.get("Earnings_Badge", "") or "")
            earn_risk  = str(row.get("Earnings_Risk", "") or "")
            if earn_badge and earn_badge not in ["✅ Pas d'earnings","—",""]:
                earn_color = "#f87171" if earn_risk == "ÉLEVÉ" else "#fbbf24" if earn_risk == "MODÉRÉ" else "#86efac"
                st.markdown(
                    f"<span style='color:{earn_color};font-weight:700;font-size:0.9rem;'>{earn_badge}</span>",
                    unsafe_allow_html=True
                )

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

            # Raisonnement Claude
            if claude_ok and claude_raison_v and claude_raison_v != "—":
                vc = verdict_color(claude_verdict_v)
                st.markdown("**🤖 Analyse Claude :**")
                st.markdown(
                    f"<div style='background:#a78bfa0d;border-left:3px solid #a78bfa;"
                    f"border-radius:4px;padding:10px 14px;margin:6px 0;font-size:0.85rem;'>"
                    f"<span style='color:{vc};font-weight:700;'>{claude_verdict_v}</span>"
                    f" — {claude_raison_v}<br>"
                    f"<span style='color:#f87171;font-size:0.82rem;'>⚠️ {claude_risque_v}</span>"
                    + (f"<br><span style='color:#fbbf24;font-size:0.82rem;'>📐 Stop ajusté: ${claude_stop_v}</span>" if claude_stop_v else "")
                    + "</div>",
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("")

    # ── MÉTRIQUES GLOBALES ──
    st.markdown("---")
    st.markdown("### 📈 Vue d'ensemble")
    ai_col  = "AI Score Ajuste"
    
