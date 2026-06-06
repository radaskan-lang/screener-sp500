import yfinance as yf
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 💪 RELATIVE STRENGTH VS SPY
# Mesure si l'action surperforme le marché
# RS = (Action / SPY) ratio normalisé
# from relative_strength import calc_relative_strength, get_spy_data
# ─────────────────────────────────────────────

# Cache SPY pour éviter de le télécharger à chaque ticker
_SPY_CACHE = {"data": None, "loaded": False}


def get_spy_data():
    """
    Charge les données SPY une seule fois et les met en cache.
    Retourne l'historique SPY (Close prices).
    """
    if _SPY_CACHE["loaded"] and _SPY_CACHE["data"] is not None:
        return _SPY_CACHE["data"]
    try:
        spy_hist = yf.Ticker("SPY").history(period="1y")
        if spy_hist is not None and not spy_hist.empty:
            _SPY_CACHE["data"]   = spy_hist["Close"]
            _SPY_CACHE["loaded"] = True
            return _SPY_CACHE["data"]
    except Exception:
        pass
    return None


def calc_relative_strength(hist, spy_close=None):
    """
    Calcule la Relative Strength d'une action vs SPY.

    Méthode :
    RS = performance action / performance SPY sur la même période.
    RS > 1 = surperforme le marché (bullish)
    RS < 1 = sous-performe le marché (bearish)

    Périodes analysées : 5j, 10j, 20j, 60j

    Retourne un dict :
    - rs_5d        : RS sur 5 jours
    - rs_10d       : RS sur 10 jours
    - rs_20d       : RS sur 20 jours
    - rs_60d       : RS sur 60 jours
    - rs_score     : score composite 0-100
    - rs_trend     : "FORTE" / "POSITIVE" / "NEUTRE" / "FAIBLE"
    - badge        : label lisible
    - bonus_pts    : bonus à ajouter au score IA
    - perf_5d      : performance action sur 5j en %
    - spy_perf_5d  : performance SPY sur 5j en %
    - outperform   : bool — surperforme SPY sur 5j
    - signal       : texte du signal principal
    - ranking_pct  : percentile estimé vs le marché (0-100)
    """
    if hist is None or hist.empty or len(hist) < 25:
        return _empty_rs()

    try:
        close = hist["Close"]

        # Charger SPY si pas fourni
        if spy_close is None:
            spy_close = get_spy_data()

        if spy_close is None or len(spy_close) < 25:
            return _empty_rs()

        # Aligner les dates
        common_dates = close.index.intersection(spy_close.index)
        if len(common_dates) < 20:
            return _empty_rs()

        stock_aligned = close.reindex(common_dates)
        spy_aligned   = spy_close.reindex(common_dates)

        def rs_period(n):
            """Calcule RS sur n jours."""
            if len(stock_aligned) < n + 1:
                return 1.0
            stock_perf = float(stock_aligned.iloc[-1] / stock_aligned.iloc[-n]) - 1
            spy_perf   = float(spy_aligned.iloc[-1] / spy_aligned.iloc[-n]) - 1
            if spy_perf == 0:
                return 1.0 + stock_perf
            return round((1 + stock_perf) / (1 + spy_perf), 3)

        rs_5d  = rs_period(5)
        rs_10d = rs_period(10)
        rs_20d = rs_period(20)
        rs_60d = rs_period(min(60, len(stock_aligned)-1))

        # Performance brute sur 5j
        perf_5d     = round((float(stock_aligned.iloc[-1]) - float(stock_aligned.iloc[-6])) / float(stock_aligned.iloc[-6]) * 100, 2) if len(stock_aligned) > 5 else 0
        spy_perf_5d = round((float(spy_aligned.iloc[-1]) - float(spy_aligned.iloc[-6])) / float(spy_aligned.iloc[-6]) * 100, 2) if len(spy_aligned) > 5 else 0
        outperform  = perf_5d > spy_perf_5d

        # ── Score composite RS ──
        rs_score = 50  # neutre = 50

        # RS 5j (poids fort pour swing)
        if rs_5d >= 1.03:    rs_score += 20
        elif rs_5d >= 1.01:  rs_score += 12
        elif rs_5d >= 1.00:  rs_score += 5
        elif rs_5d >= 0.98:  rs_score -= 5
        else:                rs_score -= 15

        # RS 10j
        if rs_10d >= 1.05:   rs_score += 15
        elif rs_10d >= 1.02: rs_score += 8
        elif rs_10d >= 1.00: rs_score += 3
        elif rs_10d >= 0.97: rs_score -= 5
        else:                rs_score -= 10

        # RS 20j
        if rs_20d >= 1.08:   rs_score += 15
        elif rs_20d >= 1.03: rs_score += 8
        elif rs_20d >= 1.00: rs_score += 3
        elif rs_20d >= 0.95: rs_score -= 3
        else:                rs_score -= 8

        rs_score = max(0, min(100, rs_score))

        # ── Percentile estimé ──
        # Basé sur le score composite
        ranking_pct = rs_score

        # ── Trend RS ──
        # RS qui s'améliore dans le temps = momentum de surperformance
        if rs_5d > rs_10d > rs_20d:
            rs_trend = "ACCÉLÉRATION"
        elif rs_5d >= 1.0 and rs_10d >= 1.0 and rs_20d >= 1.0:
            rs_trend = "FORTE"
        elif rs_5d >= 1.0 and rs_10d >= 1.0:
            rs_trend = "POSITIVE"
        elif rs_5d >= 1.0:
            rs_trend = "COURT TERME"
        elif rs_5d < 0.97 and rs_10d < 0.97:
            rs_trend = "FAIBLE"
        else:
            rs_trend = "NEUTRE"

        # ── Bonus points pour ai_score ──
        if rs_trend == "ACCÉLÉRATION":
            bonus_pts = 15
            signal = f"🚀 RS en accélération — surperforme SPY sur 5j/10j/20j"
        elif rs_trend == "FORTE":
            bonus_pts = 12
            signal = f"✅ RS forte — surperforme SPY ({round((rs_5d-1)*100,1)}% mieux sur 5j)"
        elif rs_trend == "POSITIVE":
            bonus_pts = 8
            signal = f"✅ RS positive — surperforme SPY sur 5j et 10j"
        elif rs_trend == "COURT TERME":
            bonus_pts = 4
            signal = f"~ RS court terme — surperforme SPY sur 5j seulement"
        elif rs_trend == "FAIBLE":
            bonus_pts = -8
            signal = f"🔴 RS faible — sous-performe SPY ({round((rs_5d-1)*100,1)}% sur 5j)"
        else:
            bonus_pts = 0
            signal = f"~ RS neutre vs SPY"

        # ── Badge ──
        if rs_score >= 80:
            badge = f"🚀 RS Excellente (top {100-rs_score}%)"
        elif rs_score >= 65:
            badge = f"✅ RS Forte"
        elif rs_score >= 50:
            badge = f"~ RS Neutre"
        elif rs_score >= 35:
            badge = f"⚠️ RS Faible"
        else:
            badge = f"🔴 RS Très faible"

        return {
            "rs_5d":       rs_5d,
            "rs_10d":      rs_10d,
            "rs_20d":      rs_20d,
            "rs_60d":      rs_60d,
            "rs_score":    rs_score,
            "rs_trend":    rs_trend,
            "badge":       badge,
            "bonus_pts":   bonus_pts,
            "perf_5d":     perf_5d,
            "spy_perf_5d": spy_perf_5d,
            "outperform":  outperform,
            "signal":      signal,
            "ranking_pct": ranking_pct,
        }

    except Exception:
        return _empty_rs()


def _empty_rs():
    return {
        "rs_5d":       1.0,
        "rs_10d":      1.0,
        "rs_20d":      1.0,
        "rs_60d":      1.0,
        "rs_score":    50,
        "rs_trend":    "NEUTRE",
        "badge":       "—",
        "bonus_pts":   0,
        "perf_5d":     0.0,
        "spy_perf_5d": 0.0,
        "outperform":  False,
        "signal":      "—",
        "ranking_pct": 50,
    }
