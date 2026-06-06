import yfinance as yf
import pandas as pd
from datetime import datetime


# ─────────────────────────────────────────────
# 🌍 MARKET FILTER — FILTRE MARCHÉ GLOBAL
# Vérifie l'état de SPY et QQQ avant de scanner
# Intégrer dans app.py via :
# from market_filter import get_market_status, apply_market_filter
# ─────────────────────────────────────────────

def get_market_status():
    """
    Analyse SPY (S&P 500) et QQQ (Nasdaq) pour déterminer
    l'état global du marché.

    Retourne un dict avec :
    - regime        : "HAUSSIER" / "MIXTE" / "BAISSIER"
    - color         : couleur associée (#00ff88 / #fbbf24 / #f87171)
    - emoji         : emoji du régime
    - message       : message principal
    - detail        : détails SPY + QQQ
    - spy_price     : prix actuel SPY
    - spy_ma50      : MA50 SPY
    - spy_ma200     : MA200 SPY
    - spy_vs_ma50   : % distance SPY vs MA50
    - spy_vs_ma200  : % distance SPY vs MA200
    - spy_rsi       : RSI SPY
    - spy_trend     : tendance SPY
    - qqq_price     : prix actuel QQQ
    - qqq_ma50      : MA50 QQQ
    - qqq_vs_ma50   : % distance QQQ vs MA50
    - qqq_trend     : tendance QQQ
    - vix           : niveau de peur du marché (si disponible)
    - allow_long    : True si le marché autorise les positions longues
    - caution       : True si prudence recommandée
    - score_modifier: multiplicateur à appliquer aux scores (-20 / 0 / +5)
    """
    try:
        # ── SPY ──
        spy_hist  = yf.Ticker("SPY").history(period="1y")
        spy_close = spy_hist["Close"]
        spy_price = float(spy_close.iloc[-1])
        spy_ma50  = float(spy_close.rolling(50).mean().iloc[-1])
        spy_ma200 = float(spy_close.rolling(200).mean().iloc[-1])
        spy_rsi   = _calc_rsi(spy_close)
        spy_vs_ma50  = round((spy_price - spy_ma50)  / spy_ma50  * 100, 2)
        spy_vs_ma200 = round((spy_price - spy_ma200) / spy_ma200 * 100, 2)
        spy_above_ma50  = spy_price > spy_ma50
        spy_above_ma200 = spy_price > spy_ma200

        # Momentum SPY sur 5 jours
        spy_momentum = round((spy_price - float(spy_close.iloc[-6])) / float(spy_close.iloc[-6]) * 100, 2)

        if spy_above_ma50 and spy_above_ma200:
            spy_trend = "Haussière forte"
        elif spy_above_ma200:
            spy_trend = "Haussière modérée"
        elif spy_above_ma50:
            spy_trend = "Mixte"
        else:
            spy_trend = "Baissière"

        # ── QQQ ──
        qqq_hist  = yf.Ticker("QQQ").history(period="200d")
        qqq_close = qqq_hist["Close"]
        qqq_price = float(qqq_close.iloc[-1])
        qqq_ma50  = float(qqq_close.rolling(50).mean().iloc[-1])
        qqq_vs_ma50 = round((qqq_price - qqq_ma50) / qqq_ma50 * 100, 2)
        qqq_above_ma50 = qqq_price > qqq_ma50
        qqq_momentum   = round((qqq_price - float(qqq_close.iloc[-6])) / float(qqq_close.iloc[-6]) * 100, 2)

        if qqq_above_ma50:
            qqq_trend = "Haussière"
        else:
            qqq_trend = "Baissière"

        # ── VIX (peur du marché) ──
        try:
            vix_hist  = yf.Ticker("^VIX").history(period="5d")
            vix_level = round(float(vix_hist["Close"].iloc[-1]), 1)
            if vix_level < 15:
                vix_label = f"VIX {vix_level} — Complacence 😌"
            elif vix_level < 20:
                vix_label = f"VIX {vix_level} — Normal ✅"
            elif vix_level < 30:
                vix_label = f"VIX {vix_level} — Nervosité ⚠️"
            else:
                vix_label = f"VIX {vix_level} — Panique 🔴"
        except Exception:
            vix_level = None
            vix_label = "VIX — indisponible"

        # ── Détermination du régime ──
        if spy_above_ma50 and spy_above_ma200 and qqq_above_ma50:
            regime         = "HAUSSIER"
            color          = "#00ff88"
            emoji          = "✅"
            message        = "Marché haussier — Conditions favorables au swing"
            allow_long     = True
            caution        = False
            score_modifier = 0

        elif spy_above_ma50 and not qqq_above_ma50:
            regime         = "MIXTE"
            color          = "#fbbf24"
            emoji          = "⚠️"
            message        = "Marché mixte — Prudence, sélectivité recommandée"
            allow_long     = True
            caution        = True
            score_modifier = -10

        elif not spy_above_ma50 and qqq_above_ma50:
            regime         = "MIXTE"
            color          = "#fbbf24"
            emoji          = "⚠️"
            message        = "Marché mixte — SPY faible, tech résistant"
            allow_long     = True
            caution        = True
            score_modifier = -10

        else:
            regime         = "BAISSIER"
            color          = "#f87171"
            emoji          = "🔴"
            message        = "Marché baissier — Mode défensif activé"
            allow_long     = False
            caution        = True
            score_modifier = -20

        # Surcharge VIX si panique
        if vix_level and vix_level > 30 and regime != "BAISSIER":
            regime         = "MIXTE"
            color          = "#fbbf24"
            emoji          = "⚠️"
            message        = f"VIX élevé ({vix_level}) — Volatilité extrême, prudence"
            caution        = True
            score_modifier = max(score_modifier, -15)

        detail = (
            f"SPY ${round(spy_price,2)} | MA50 ${round(spy_ma50,2)} "
            f"({'+' if spy_vs_ma50 >= 0 else ''}{spy_vs_ma50}%) | "
            f"MA200 ({'+' if spy_vs_ma200 >= 0 else ''}{spy_vs_ma200}%) | "
            f"Momentum 5j: {'+' if spy_momentum >= 0 else ''}{spy_momentum}%"
            f"\nQQQ ${round(qqq_price,2)} | MA50 ${round(qqq_ma50,2)} "
            f"({'+' if qqq_vs_ma50 >= 0 else ''}{qqq_vs_ma50}%) | "
            f"Momentum 5j: {'+' if qqq_momentum >= 0 else ''}{qqq_momentum}%"
            f"\n{vix_label}"
        )

        return {
            "regime":         regime,
            "color":          color,
            "emoji":          emoji,
            "message":        message,
            "detail":         detail,
            "spy_price":      round(spy_price, 2),
            "spy_ma50":       round(spy_ma50, 2),
            "spy_ma200":      round(spy_ma200, 2),
            "spy_vs_ma50":    spy_vs_ma50,
            "spy_vs_ma200":   spy_vs_ma200,
            "spy_rsi":        round(spy_rsi, 1),
            "spy_trend":      spy_trend,
            "spy_momentum":   spy_momentum,
            "qqq_price":      round(qqq_price, 2),
            "qqq_ma50":       round(qqq_ma50, 2),
            "qqq_vs_ma50":    qqq_vs_ma50,
            "qqq_trend":      qqq_trend,
            "qqq_momentum":   qqq_momentum,
            "vix":            vix_level,
            "vix_label":      vix_label,
            "allow_long":     allow_long,
            "caution":        caution,
            "score_modifier": score_modifier,
        }

    except Exception as e:
        return {
            "regime": "INCONNU", "color": "#64748b", "emoji": "❓",
            "message": f"Données marché indisponibles : {e}",
            "detail": "", "allow_long": True, "caution": False,
            "score_modifier": 0, "spy_price": None, "qqq_price": None,
            "vix": None, "vix_label": "—",
        }


def apply_market_filter(df, market_status):
    """
    Applique le filtre marché sur le DataFrame de résultats.

    - Marché BAISSIER  : convertit STRONG BUY et BUY en HOLD
    - Marché MIXTE     : réduit le score de 10 pts
    - Marché HAUSSIER  : aucune modification

    Ajoute une colonne 'AI Signal Ajusté' et 'AI Score Ajusté'.
    """
    df = df.copy()

    modifier = market_status.get("score_modifier", 0)
    regime   = market_status.get("regime", "HAUSSIER")

    if modifier != 0:
        df["AI Score Ajusté"] = (df["AI Score"] + modifier).clip(0, 100)
    else:
        df["AI Score Ajusté"] = df["AI Score"]

    # Recalculer le signal ajusté
    def adjusted_signal(row):
        score  = row["AI Score Ajusté"]
        signal = row["AI Signal"]

        if regime == "BAISSIER":
            if signal in ["🟢 STRONG BUY", "🟢 BUY"]:
                return "🟡 HOLD ⚠️"
        if score >= 85: return "🟢 STRONG BUY"
        if score >= 70: return "🟢 BUY"
        if score >= 50: return "🟡 HOLD"
        return "🔴 AVOID"

    df["AI Signal Ajusté"] = df.apply(adjusted_signal, axis=1)

    return df


def _calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = -delta.where(delta < 0, 0).rolling(period).mean()
    rs    = gain / loss.clip(lower=1e-10)
    return float(100 - (100 / (1 + rs.iloc[-1])))


def market_advice(market_status):
    """
    Retourne des conseils concrets selon le régime de marché.
    """
    regime = market_status.get("regime", "HAUSSIER")
    vix    = market_status.get("vix", None)

    if regime == "HAUSSIER":
        advice = [
            "✅ Conditions optimales pour le swing trading",
            "✅ Viser des positions sur 3-5 jours",
            "✅ R/R minimum 2:1 recommandé",
        ]
        if vix and vix > 20:
            advice.append("⚠️ VIX modéré — Réduire légèrement la taille des positions")

    elif regime == "MIXTE":
        advice = [
            "⚠️ Réduire la taille des positions de 30-50%",
            "⚠️ Privilégier les setups avec R/R ≥ 2.5:1",
            "⚠️ Éviter les secteurs les plus faibles",
            "⚠️ Stop-loss plus serrés que d'habitude",
        ]
    else:  # BAISSIER
        advice = [
            "🔴 Éviter les nouvelles positions longues",
            "🔴 Protéger les positions existantes",
            "🔴 Cash is king — Attendre la reprise",
            "🔴 Surveiller le retour de SPY au-dessus de la MA50",
        ]

    return advice
