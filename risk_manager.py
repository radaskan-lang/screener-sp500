import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 📐 RISK MANAGER — SWING TRADING
# Calcul ATR, entrée, stop-loss, target, R/R
# Intégrer dans app.py via :
# from risk_manager import calc_risk_reward, risk_badge
# ─────────────────────────────────────────────

def calc_atr(hist, period=14):
    """
    Average True Range sur 14 jours.
    Mesure la volatilité réelle (gaps inclus).
    """
    high  = hist["High"]
    low   = hist["Low"]
    close = hist["Close"]

    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr)


def find_support_resistance(hist, lookback=20):
    """
    Trouve le support (plus bas récent) et la résistance (plus haut récent)
    sur les N derniers jours — utile pour valider stop et target.
    """
    recent = hist.iloc[-lookback:]
    support    = float(recent["Low"].min())
    resistance = float(recent["High"].max())
    return support, resistance


def calc_risk_reward(hist, atr_multiplier_stop=1.5, atr_multiplier_target=3.0):
    """
    Calcule le plan de trade complet pour le swing.

    Paramètres :
    - atr_multiplier_stop   : distance stop en multiple d'ATR (défaut 1.5)
    - atr_multiplier_target : distance target en multiple d'ATR (défaut 3.0)

    Retourne un dict avec :
    - entry       : prix d'entrée suggéré
    - stop        : stop-loss
    - target      : objectif de prix
    - rr_ratio    : ratio risque/rendement
    - risk_pct    : risque en % du capital sur ce trade
    - reward_pct  : gain potentiel en %
    - atr         : valeur ATR brute
    - atr_pct     : ATR en % du prix (volatilité relative)
    - support     : support technique récent
    - resistance  : résistance technique récente
    - quality     : qualité du setup (Excellent / Bon / Acceptable / Faible)
    - confidence  : nombre de confirmations du setup
    """
    try:
        close      = hist["Close"]
        price      = float(close.iloc[-1])
        atr        = calc_atr(hist)
        support, resistance = find_support_resistance(hist)

        # ── Entrée ──
        # Légère confirmation au-dessus du prix actuel (+0.3%)
        entry = round(price * 1.003, 2)

        # ── Stop-loss ──
        # Basé sur ATR × multiplicateur
        stop_atr  = round(entry - (atr * atr_multiplier_stop), 2)
        # Ne pas descendre sous le support récent
        stop_final = round(max(stop_atr, support * 0.995), 2)

        # ── Target ──
        # Basé sur ATR × multiplicateur
        target_atr   = round(entry + (atr * atr_multiplier_target), 2)
        # Ne pas dépasser la résistance si elle est plus proche
        if resistance > entry * 1.01:
            target_final = round(min(target_atr, resistance * 0.99), 2)
        else:
            target_final = target_atr

        # ── Calculs R/R ──
        risk_pts   = entry - stop_final
        reward_pts = target_final - entry

        if risk_pts <= 0:
            return _empty_rr()

        rr_ratio   = round(reward_pts / risk_pts, 2)
        risk_pct   = round((risk_pts / entry) * 100, 2)
        reward_pct = round((reward_pts / entry) * 100, 2)
        atr_pct    = round((atr / price) * 100, 2)

        # ── Qualité du setup ──
        confidence = 0
        if rr_ratio >= 2.5:   confidence += 1
        if risk_pct <= 3.0:   confidence += 1
        if atr_pct >= 1.5:    confidence += 1  # assez de volatilité pour bouger
        if atr_pct <= 5.0:    confidence += 1  # pas trop volatile
        if reward_pct >= 5.0: confidence += 1

        if confidence >= 5:   quality = "🔥 Excellent"
        elif confidence >= 4: quality = "✅ Bon"
        elif confidence >= 3: quality = "⚠️ Acceptable"
        else:                 quality = "❌ Faible"

        return {
            "entry":      entry,
            "stop":       stop_final,
            "target":     target_final,
            "rr_ratio":   rr_ratio,
            "risk_pct":   risk_pct,
            "reward_pct": reward_pct,
            "atr":        round(atr, 2),
            "atr_pct":    atr_pct,
            "support":    round(support, 2),
            "resistance": round(resistance, 2),
            "quality":    quality,
            "confidence": confidence,
        }

    except Exception:
        return _empty_rr()


def _empty_rr():
    """Retourne un dict vide en cas d'erreur."""
    return {
        "entry": None, "stop": None, "target": None,
        "rr_ratio": None, "risk_pct": None, "reward_pct": None,
        "atr": None, "atr_pct": None,
        "support": None, "resistance": None,
        "quality": "—", "confidence": 0,
    }


def risk_badge(rr_ratio, risk_pct):
    """
    Badge lisible pour le tableau principal.
    """
    if rr_ratio is None or risk_pct is None:
        return "—"
    if rr_ratio >= 2.5 and risk_pct <= 3:
        return "🔥 R/R Excellent"
    elif rr_ratio >= 2.0 and risk_pct <= 4:
        return "✅ R/R Bon"
    elif rr_ratio >= 1.5:
        return "⚠️ R/R Acceptable"
    else:
        return "❌ R/R Faible"


def format_trade_plan(rr_data, ticker):
    """
    Formate un plan de trade lisible pour l'affichage Streamlit.
    """
    if rr_data["entry"] is None:
        return f"Plan indisponible pour {ticker}"

    return f"""
📌 **{ticker} — Plan de Trade Swing**

| | Prix | % |
|---|---|---|
| 🎯 Entrée | ${rr_data['entry']} | — |
| 🛑 Stop-Loss | ${rr_data['stop']} | -{rr_data['risk_pct']}% |
| 🏆 Target | ${rr_data['target']} | +{rr_data['reward_pct']}% |

**Ratio R/R : {rr_data['rr_ratio']}:1** &nbsp;|&nbsp; ATR : ${rr_data['atr']} ({rr_data['atr_pct']}%)

Support : ${rr_data['support']} &nbsp;|&nbsp; Résistance : ${rr_data['resistance']}

Qualité setup : {rr_data['quality']} ({rr_data['confidence']}/5 confirmations)
"""
