import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ─────────────────────────────────────────────
# 📊 SIGNAUX INTRADAY — ANTI-FAUX SIGNAUX
# VWAP · PDH/PDL · ORB · Multi-TF · Momentum
# from intraday_signals import (
#     calc_vwap_levels, calc_multitf_signals, calc_intraday_momentum
# )
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 1. VWAP + PDH/PDL + OPENING RANGE
# ─────────────────────────────────────────────

def calc_vwap_levels(ticker):
    """
    Calcule VWAP, PDH/PDL et Opening Range pour aujourd'hui.

    VWAP  = Prix moyen pondéré par le volume depuis l'ouverture
    PDH   = Plus haut de la journée précédente (résistance clé)
    PDL   = Plus bas de la journée précédente (support clé)
    ORB   = Opening Range = high/low des 30 premières minutes

    Retourne :
    - vwap          : prix VWAP actuel
    - price_vs_vwap : position vs VWAP en %
    - above_vwap    : bool
    - pdh / pdl     : high/low de hier
    - dist_to_pdh   : distance au PDH en %
    - dist_to_pdl   : distance au PDL en %
    - orb_high / orb_low : opening range high/low
    - orb_breakout  : bool — prix au-dessus de l'ORB
    - signal        : signal principal
    - score         : bonus pts
    - badge         : label lisible
    """
    try:
        t = yf.Ticker(ticker)

        # Données 5min pour aujourd'hui (délai 15min yfinance)
        intraday = t.history(period="2d", interval="5m")

        if intraday is None or intraday.empty or len(intraday) < 10:
            return _empty_vwap()

        # Séparer aujourd'hui et hier
        intraday.index = pd.to_datetime(intraday.index)
        today = intraday.index[-1].date()

        today_data = intraday[intraday.index.date == today]
        prev_data  = intraday[intraday.index.date < today]

        if today_data.empty:
            return _empty_vwap()

        # ── VWAP ──
        typical_price = (today_data["High"] + today_data["Low"] + today_data["Close"]) / 3
        cumvol = today_data["Volume"].cumsum()
        cumtp  = (typical_price * today_data["Volume"]).cumsum()
        vwap   = round(float(cumtp.iloc[-1] / cumvol.iloc[-1]), 2) if cumvol.iloc[-1] > 0 else None

        price = float(today_data["Close"].iloc[-1])
        price_vs_vwap = round((price - vwap) / vwap * 100, 2) if vwap else 0
        above_vwap    = price > vwap if vwap else True

        # ── PDH / PDL ──
        pdh = pdl = None
        if not prev_data.empty:
            prev_day = prev_data[prev_data.index.date == prev_data.index[-1].date()]
            if not prev_day.empty:
                pdh = round(float(prev_day["High"].max()), 2)
                pdl = round(float(prev_day["Low"].min()), 2)

        dist_to_pdh = round((pdh - price) / price * 100, 2) if pdh else None
        dist_to_pdl = round((price - pdl) / price * 100, 2) if pdl else None

        # ── OPENING RANGE (30 premières minutes) ──
        orb_data  = today_data.iloc[:6]  # 6 bougies de 5min = 30 min
        orb_high  = round(float(orb_data["High"].max()), 2) if not orb_data.empty else None
        orb_low   = round(float(orb_data["Low"].min()), 2) if not orb_data.empty else None
        orb_breakout = price > orb_high if orb_high else False
        orb_breakdown= price < orb_low  if orb_low  else False

        # ── Scoring ──
        score  = 0
        signal = None

        # Prix au-dessus du VWAP
        if above_vwap and price_vs_vwap > 0.5:
            score += 8
            signal = f"✅ Au-dessus du VWAP (${vwap}) +{price_vs_vwap}%"
        elif above_vwap:
            score += 3
        elif price_vs_vwap < -1.0:
            score -= 8
            signal = f"🔴 Sous le VWAP (${vwap}) {price_vs_vwap}%"

        # Breakout PDH
        if pdh and price > pdh:
            score += 12
            signal = f"🚀 Breakout PDH ${pdh} — force haussière forte"

        # Proche du PDH (résistance)
        elif pdh and dist_to_pdh and dist_to_pdh <= 1.5:
            score -= 5
            if not signal:
                signal = f"⚠️ PDH résistance proche (${pdh}, -{dist_to_pdh}%)"

        # Support PDL tenu
        if pdl and dist_to_pdl and dist_to_pdl >= 0 and dist_to_pdl <= 2.0:
            score += 5
            if not signal:
                signal = f"✅ Support PDL tenu (${pdl})"

        # ORB Breakout
        if orb_breakout and orb_high:
            score += 10
            signal = f"🚀 ORB Breakout (${orb_high}) — direction confirmée"
        elif orb_breakdown and orb_low:
            score -= 10
            signal = f"🔴 ORB Breakdown (${orb_low})"

        score = max(-15, min(score, 22))

        # Badge
        if score >= 18:
            badge = f"🚀 Setup intraday excellent"
        elif score >= 10:
            badge = f"✅ Setup intraday favorable"
        elif score >= 0:
            badge = f"~ Intraday neutre"
        else:
            badge = f"🔴 Setup intraday défavorable"

        return {
            "vwap":         vwap,
            "price_vs_vwap":price_vs_vwap,
            "above_vwap":   above_vwap,
            "pdh":          pdh,
            "pdl":          pdl,
            "dist_to_pdh":  dist_to_pdh,
            "dist_to_pdl":  dist_to_pdl,
            "orb_high":     orb_high,
            "orb_low":      orb_low,
            "orb_breakout": orb_breakout,
            "orb_breakdown":orb_breakdown,
            "signal":       signal,
            "score":        score,
            "badge":        badge,
            "price":        price,
        }

    except Exception:
        return _empty_vwap()


def _empty_vwap():
    return {
        "vwap": None, "price_vs_vwap": 0, "above_vwap": True,
        "pdh": None, "pdl": None, "dist_to_pdh": None, "dist_to_pdl": None,
        "orb_high": None, "orb_low": None, "orb_breakout": False, "orb_breakdown": False,
        "signal": None, "score": 0, "badge": "—", "price": None,
    }


# ─────────────────────────────────────────────
# 2. MULTI-TIMEFRAME RSI + BOLLINGER
# Élimine les faux signaux par confirmation
# ─────────────────────────────────────────────

def calc_multitf_signals(ticker, hist_daily=None):
    """
    Calcule RSI et Bollinger sur 3 timeframes :
    Daily (existant) + 1H + 15min

    Retourne :
    - rsi_1h / rsi_15min
    - bb_pct_1h / bb_pct_15min
    - tf_alignment   : nombre de TF alignés haussiers (0-3)
    - confirmation   : "FORT" / "MODÉRÉ" / "FAIBLE" / "CONTRADICTOIRE"
    - signal         : description
    - score          : bonus/pénalité
    - badge          : label lisible
    - details        : dict des valeurs par TF
    """
    try:
        t = yf.Ticker(ticker)

        results = {}

        for interval, label in [("1h", "1H"), ("15m", "15min")]:
            try:
                hist = t.history(period="5d", interval=interval)
                if hist is None or hist.empty or len(hist) < 20:
                    results[label] = None
                    continue

                close = hist["Close"]

                # RSI 14
                delta = close.diff()
                gain  = delta.where(delta > 0, 0).rolling(14).mean()
                loss  = -delta.where(delta < 0, 0).rolling(14).mean()
                rs    = gain / loss.clip(lower=1e-10)
                rsi   = round(float((100 - (100 / (1 + rs))).iloc[-1]), 1)

                # Bollinger
                ma    = close.rolling(20).mean()
                std   = close.rolling(20).std()
                upper = ma + 2 * std
                lower = ma - 2 * std
                price = float(close.iloc[-1])
                bb_pct = round((price - float(lower.iloc[-1])) /
                               (float(upper.iloc[-1]) - float(lower.iloc[-1])), 3) \
                         if (float(upper.iloc[-1]) - float(lower.iloc[-1])) > 0 else 0.5

                # MACD
                ema12  = close.ewm(span=12, adjust=False).mean()
                ema26  = close.ewm(span=26, adjust=False).mean()
                macd_h = float((ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()).iloc[-1])

                results[label] = {
                    "rsi":    rsi,
                    "bb_pct": bb_pct,
                    "macd_h": round(macd_h, 4),
                    "bullish": 45 <= rsi <= 75 and macd_h > 0 and bb_pct > 0.3,
                    "overbought": rsi > 75,
                    "oversold":   rsi < 30,
                }
            except Exception:
                results[label] = None

        # RSI daily depuis l'historique existant (déjà calculé)
        rsi_daily = None
        if hist_daily is not None and not hist_daily.empty:
            try:
                close_d = hist_daily["Close"]
                delta_d = close_d.diff()
                gain_d  = delta_d.where(delta_d > 0, 0).rolling(14).mean()
                loss_d  = -delta_d.where(delta_d < 0, 0).rolling(14).mean()
                rs_d    = gain_d / loss_d.clip(lower=1e-10)
                rsi_daily = round(float((100 - (100 / (1 + rs_d))).iloc[-1]), 1)
            except Exception:
                pass

        # ── Alignement des timeframes ──
        tf_bullish = 0
        tf_total   = 0

        if rsi_daily and 45 <= rsi_daily <= 75:
            tf_bullish += 1
        if rsi_daily:
            tf_total += 1

        for label in ["1H", "15min"]:
            if results.get(label):
                tf_total += 1
                if results[label]["bullish"]:
                    tf_bullish += 1

        # ── Contradiction détectée ──
        contradiction = False
        if results.get("1H") and results.get("15min"):
            r1h    = results["1H"]
            r15    = results["15min"]
            # RSI divergent entre 1H et 15min
            if abs(r1h["rsi"] - r15["rsi"]) > 20:
                contradiction = True
            # 1H haussier mais 15min surachat extrême
            if r1h["bullish"] and r15["overbought"]:
                contradiction = True

        # Confirmation globale
        if tf_total == 0:
            confirmation = "INDISPONIBLE"
            score = 0
        elif contradiction:
            confirmation = "CONTRADICTOIRE"
            score = -5
        elif tf_bullish == tf_total:
            confirmation = "FORT"
            score = 15
        elif tf_bullish >= tf_total - 1:
            confirmation = "MODÉRÉ"
            score = 8
        elif tf_bullish >= 1:
            confirmation = "FAIBLE"
            score = 2
        else:
            confirmation = "BAISSIER"
            score = -10

        # Signal descriptif
        r1h_str   = f"1H RSI:{results['1H']['rsi']}" if results.get("1H") else "1H N/A"
        r15_str   = f"15m RSI:{results['15min']['rsi']}" if results.get("15min") else "15m N/A"
        d_str     = f"D RSI:{rsi_daily}" if rsi_daily else "D N/A"

        if confirmation == "FORT":
            signal = f"✅ Confirmation {tf_bullish}/{tf_total} TF — {d_str} | {r1h_str} | {r15_str}"
        elif confirmation == "CONTRADICTOIRE":
            signal = f"⚠️ Signal contradictoire entre TF — attendre alignement"
        elif confirmation == "MODÉRÉ":
            signal = f"~ Confirmation partielle {tf_bullish}/{tf_total} TF"
        elif confirmation == "BAISSIER":
            signal = f"🔴 Tous les TF baissiers — éviter"
        else:
            signal = f"~ Confirmation {tf_bullish}/{tf_total} TF"

        if confirmation == "FORT":
            badge = f"✅ Multi-TF alignés ({tf_bullish}/{tf_total})"
        elif confirmation == "CONTRADICTOIRE":
            badge = f"⚠️ TF contradictoires — prudence"
        elif confirmation == "MODÉRÉ":
            badge = f"~ Multi-TF partiel ({tf_bullish}/{tf_total})"
        elif confirmation == "BAISSIER":
            badge = f"🔴 Multi-TF baissier"
        else:
            badge = f"~ Multi-TF {tf_bullish}/{tf_total}"

        return {
            "rsi_1h":      results.get("1H", {}).get("rsi") if results.get("1H") else None,
            "rsi_15min":   results.get("15min", {}).get("rsi") if results.get("15min") else None,
            "rsi_daily":   rsi_daily,
            "bb_pct_1h":   results.get("1H", {}).get("bb_pct") if results.get("1H") else None,
            "bb_pct_15min":results.get("15min", {}).get("bb_pct") if results.get("15min") else None,
            "tf_alignment": tf_bullish,
            "tf_total":     tf_total,
            "confirmation": confirmation,
            "contradiction":contradiction,
            "signal":       signal,
            "score":        score,
            "badge":        badge,
            "details":      results,
        }

    except Exception:
        return _empty_multitf()


def _empty_multitf():
    return {
        "rsi_1h": None, "rsi_15min": None, "rsi_daily": None,
        "bb_pct_1h": None, "bb_pct_15min": None,
        "tf_alignment": 0, "tf_total": 0,
        "confirmation": "INDISPONIBLE", "contradiction": False,
        "signal": None, "score": 0, "badge": "—", "details": {},
    }


# ─────────────────────────────────────────────
# 3. MOMENTUM INTRADAY VS SPY
# Force relative depuis l'ouverture
# ─────────────────────────────────────────────

def calc_intraday_momentum(ticker, spy_intraday=None):
    """
    Mesure la force relative de l'action vs SPY depuis l'ouverture.

    Retourne :
    - perf_open      : performance depuis l'ouverture en %
    - spy_perf_open  : performance SPY depuis l'ouverture
    - relative_mom   : surperformance vs SPY en %
    - momentum_score : force du momentum (0-100)
    - trend_intraday : "FORT_HAUSSIER" / "HAUSSIER" / "NEUTRE" / "BAISSIER"
    - signal         : description
    - score          : bonus pts
    - badge          : label lisible
    - acceleration   : bool — momentum qui s'accélère
    """
    try:
        t        = yf.Ticker(ticker)
        intraday = t.history(period="1d", interval="5m")

        if intraday is None or intraday.empty or len(intraday) < 3:
            return _empty_momentum()

        open_price  = float(intraday["Open"].iloc[0])
        last_price  = float(intraday["Close"].iloc[-1])
        perf_open   = round((last_price - open_price) / open_price * 100, 2) if open_price > 0 else 0

        # SPY depuis l'ouverture
        spy_perf_open = 0
        if spy_intraday is not None and not spy_intraday.empty:
            try:
                spy_open  = float(spy_intraday["Open"].iloc[0])
                spy_last  = float(spy_intraday["Close"].iloc[-1])
                spy_perf_open = round((spy_last - spy_open) / spy_open * 100, 2) if spy_open > 0 else 0
            except Exception:
                pass
        else:
            try:
                spy_hist = yf.Ticker("SPY").history(period="1d", interval="5m")
                if spy_hist is not None and not spy_hist.empty:
                    spy_open      = float(spy_hist["Open"].iloc[0])
                    spy_last      = float(spy_hist["Close"].iloc[-1])
                    spy_perf_open = round((spy_last - spy_open) / spy_open * 100, 2) if spy_open > 0 else 0
            except Exception:
                pass

        relative_mom = round(perf_open - spy_perf_open, 2)

        # Accélération du momentum
        # Comparer la performance de la première moitié vs deuxième moitié
        acceleration = False
        if len(intraday) >= 6:
            mid     = len(intraday) // 2
            half1_p = float(intraday["Close"].iloc[mid-1])
            half2_p = float(intraday["Close"].iloc[-1])
            perf_h1 = (half1_p - open_price) / open_price * 100
            perf_h2 = (half2_p - half1_p) / half1_p * 100
            acceleration = perf_h2 > perf_h1 * 0.5 and perf_h2 > 0

        # Score
        score = 0
        if relative_mom >= 2.0:
            score = 15
            trend = "FORT_HAUSSIER"
            signal = f"🚀 Momentum intraday exceptionnel +{relative_mom}% vs SPY"
        elif relative_mom >= 1.0:
            score = 10
            trend = "HAUSSIER"
            signal = f"✅ Momentum intraday fort +{relative_mom}% vs SPY"
        elif relative_mom >= 0.3:
            score = 5
            trend = "HAUSSIER"
            signal = f"~ Momentum positif +{relative_mom}% vs SPY"
        elif relative_mom >= -0.5:
            score = 0
            trend = "NEUTRE"
            signal = f"~ Momentum neutre ({relative_mom}% vs SPY)"
        elif relative_mom >= -1.5:
            score = -5
            trend = "BAISSIER"
            signal = f"⚠️ Sous-performe SPY de {abs(relative_mom)}% aujourd'hui"
        else:
            score = -12
            trend = "BAISSIER"
            signal = f"🔴 Faiblesse intraday forte {relative_mom}% vs SPY"

        if acceleration and score > 0:
            score += 3
            signal += " ↗ accélération"

        score = max(-15, min(score, 18))

        if score >= 12:
            badge = f"🚀 Momentum fort ({perf_open:+.1f}% | SPY {spy_perf_open:+.1f}%)"
        elif score >= 5:
            badge = f"✅ Momentum positif ({perf_open:+.1f}% | SPY {spy_perf_open:+.1f}%)"
        elif score >= 0:
            badge = f"~ Momentum neutre ({perf_open:+.1f}%)"
        else:
            badge = f"🔴 Momentum faible ({perf_open:+.1f}% | SPY {spy_perf_open:+.1f}%)"

        return {
            "perf_open":      perf_open,
            "spy_perf_open":  spy_perf_open,
            "relative_mom":   relative_mom,
            "trend_intraday": trend,
            "acceleration":   acceleration,
            "signal":         signal,
            "score":          score,
            "badge":          badge,
        }

    except Exception:
        return _empty_momentum()


def _empty_momentum():
    return {
        "perf_open": 0, "spy_perf_open": 0, "relative_mom": 0,
        "trend_intraday": "NEUTRE", "acceleration": False,
        "signal": None, "score": 0, "badge": "—",
    }


# ─────────────────────────────────────────────
# SCORE GLOBAL INTRADAY
# Combine les 3 modules
# ─────────────────────────────────────────────

def calc_intraday_score(ticker, hist_daily=None):
    """
    Score intraday global combinant VWAP, Multi-TF et Momentum.
    Appel unique pour les 3 modules.
    """
    vwap_data    = calc_vwap_levels(ticker)
    multitf_data = calc_multitf_signals(ticker, hist_daily)
    momentum_data= calc_intraday_momentum(ticker)

    total_score = (
        vwap_data["score"] +
        multitf_data["score"] +
        momentum_data["score"]
    )
    total_score = max(-20, min(total_score, 30))

    # Pénalité contradictions
    if multitf_data["contradiction"]:
        total_score -= 8

    # Badge global
    if total_score >= 25:
        global_badge = "🚀 Setup intraday exceptionnel"
    elif total_score >= 15:
        global_badge = "✅ Setup intraday solide"
    elif total_score >= 5:
        global_badge = "~ Setup intraday correct"
    elif total_score >= -5:
        global_badge = "~ Intraday neutre"
    else:
        global_badge = "🔴 Setup intraday défavorable"

    return {
        "vwap":     vwap_data,
        "multitf":  multitf_data,
        "momentum": momentum_data,
        "total_score": total_score,
        "global_badge": global_badge,
    }
