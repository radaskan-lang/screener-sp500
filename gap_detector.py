import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 📊 GAP DETECTOR — SWING TRADING
# Détecte les gaps haussiers significatifs
# Gap rupture · Gap continuation · Gap comblement
# from gap_detector import detect_gaps
# ─────────────────────────────────────────────

def detect_gaps(hist, min_gap_pct=0.5, volume_multiplier=1.5, lookback=60):
    """
    Détecte et classifie les gaps sur l'historique OHLCV.

    Types de gaps :
    - BREAKOUT_GAP  : gap au-dessus d'une résistance avec volume fort → +20 pts
    - CONTINUATION  : gap dans une tendance établie → +15 pts
    - RECENT_GAP    : gap haussier récent (< 5 jours) → +12 pts
    - GAP_SUPPORT   : gap non comblé = niveau de support solide → +8 pts
    - BEARISH_GAP   : gap baissier récent → pénalité -10 pts

    Retourne un dict :
    - signals       : liste des signaux détectés
    - top_signal    : signal principal
    - score         : bonus pts (0-20, négatif si bearish)
    - badge         : label lisible
    - recent_gap_pct: taille du dernier gap en %
    - gap_direction : "UP" / "DOWN" / "NONE"
    - unfilled_gaps : liste des gaps non comblés (niveaux de support)
    - nearest_support: niveau de support gap le plus proche
    - summary       : résumé textuel
    """
    if hist is None or hist.empty or len(hist) < 10:
        return _empty_gap()

    try:
        opens  = hist["Open"]
        closes = hist["Close"]
        highs  = hist["High"]
        lows   = hist["Low"]
        volume = hist["Volume"]

        avg_vol = float(volume.rolling(20).mean().iloc[-1])
        price   = float(closes.iloc[-1])

        # ── Détecter tous les gaps sur la période ──
        gaps = []
        for i in range(1, len(hist)):
            prev_close = float(closes.iloc[i-1])
            curr_open  = float(opens.iloc[i])
            curr_vol   = float(volume.iloc[i])

            if prev_close == 0:
                continue

            gap_pct = (curr_open - prev_close) / prev_close * 100

            # Seulement les gaps significatifs
            if abs(gap_pct) < min_gap_pct:
                continue

            vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
            days_ago  = len(hist) - 1 - i

            # Vérifier si le gap a été comblé
            gap_filled = False
            if gap_pct > 0:  # Gap haussier — comblé si prix descend sous prev_close
                future_lows = lows.iloc[i+1:].values if i+1 < len(hist) else []
                gap_filled  = any(l <= prev_close for l in future_lows)
            else:  # Gap baissier — comblé si prix remonte au-dessus prev_close
                future_highs = highs.iloc[i+1:].values if i+1 < len(hist) else []
                gap_filled   = any(h >= prev_close for h in future_highs)

            gaps.append({
                "idx":        i,
                "days_ago":   days_ago,
                "gap_pct":    round(gap_pct, 2),
                "prev_close": round(prev_close, 2),
                "open":       round(curr_open, 2),
                "vol_ratio":  round(vol_ratio, 2),
                "filled":     gap_filled,
                "direction":  "UP" if gap_pct > 0 else "DOWN",
            })

        if not gaps:
            return _empty_gap()

        # ── Analyser les gaps récents et non comblés ──
        recent_gaps   = [g for g in gaps if g["days_ago"] <= 5]
        unfilled_gaps = [g for g in gaps if not g["filled"] and g["direction"] == "UP"]
        bullish_gaps  = [g for g in gaps if g["direction"] == "UP"]
        bearish_gaps  = [g for g in gaps if g["direction"] == "DOWN" and g["days_ago"] <= 5]

        signals   = []
        score     = 0
        direction = "NONE"

        # ── 1. Gap haussier très récent (1-3 jours) avec volume ──
        very_recent = [g for g in recent_gaps if g["direction"]=="UP" and g["days_ago"] <= 3]
        if very_recent:
            g = very_recent[-1]  # Le plus récent
            if g["vol_ratio"] >= 2.0:
                signals.append({
                    "name":   f"🚀 GAP RUPTURE HAUSSIER +{g['gap_pct']}% (vol {g['vol_ratio']}x)",
                    "score":  20,
                    "detail": f"Gap il y a {g['days_ago']}j — ${g['prev_close']} → ${g['open']} avec volume exceptionnel",
                })
                score     += 20
                direction  = "UP"
            elif g["vol_ratio"] >= volume_multiplier:
                signals.append({
                    "name":   f"⚡ Gap haussier récent +{g['gap_pct']}% (vol {g['vol_ratio']}x)",
                    "score":  15,
                    "detail": f"Gap il y a {g['days_ago']}j — volume fort confirmé",
                })
                score     += 15
                direction  = "UP"
            else:
                signals.append({
                    "name":   f"📈 Gap haussier +{g['gap_pct']}%",
                    "score":  8,
                    "detail": f"Gap il y a {g['days_ago']}j — volume normal",
                })
                score     += 8
                direction  = "UP"

        # ── 2. Gap de continuation dans une tendance ──
        elif bullish_gaps:
            # Vérifier si on est dans une tendance haussière
            ma20_now  = float(closes.rolling(20).mean().iloc[-1])
            ma20_prev = float(closes.rolling(20).mean().iloc[-10]) if len(closes) > 10 else ma20_now
            in_uptrend = price > ma20_now > ma20_prev

            recent_bull = [g for g in bullish_gaps if g["days_ago"] <= 10]
            if recent_bull and in_uptrend:
                g = recent_bull[-1]
                signals.append({
                    "name":   f"✅ Gap continuation +{g['gap_pct']}% (tendance haussière)",
                    "score":  12,
                    "detail": f"Gap il y a {g['days_ago']}j dans une tendance établie",
                })
                score     += 12
                direction  = "UP"

        # ── 3. Gaps non comblés = niveaux de support ──
        if unfilled_gaps:
            # Trouver le support le plus proche en dessous du prix actuel
            supports = [(g["prev_close"], g["gap_pct"]) for g in unfilled_gaps
                        if g["prev_close"] < price]
            if supports:
                nearest_support = max(supports, key=lambda x: x[0])
                dist_pct = round((price - nearest_support[0]) / price * 100, 1)
                signals.append({
                    "name":   f"🛡️ Gap support non comblé à ${round(nearest_support[0],2)} ({dist_pct}% sous le prix)",
                    "score":  6,
                    "detail": f"Niveau de support solide — bon endroit pour stop-loss",
                })
                score += 6

        # ── 4. Gap baissier récent — pénalité ──
        if bearish_gaps:
            g = bearish_gaps[-1]
            if g["vol_ratio"] >= 1.5:
                signals.append({
                    "name":   f"🔴 Gap baissier récent {g['gap_pct']}% (vol {g['vol_ratio']}x)",
                    "score":  -10,
                    "detail": f"Gap de faiblesse il y a {g['days_ago']}j — signal négatif",
                })
                score    -= 10
                direction = "DOWN"
            else:
                signals.append({
                    "name":   f"⚠️ Gap baissier {g['gap_pct']}%",
                    "score":  -5,
                    "detail": f"Gap de faiblesse il y a {g['days_ago']}j",
                })
                score    -= 5

        # ── Score final plafonné ──
        score = max(-15, min(score, 20))

        # ── Badge ──
        if score >= 18:
            badge = "🚀 Gap institutionnel fort"
        elif score >= 12:
            badge = "⚡ Gap haussier actif"
        elif score >= 6:
            badge = "📈 Gap support détecté"
        elif score > 0:
            badge = "~ Gap faible"
        elif score < 0:
            badge = "🔴 Gap baissier récent"
        else:
            badge = "—"

        # ── Support le plus proche ──
        nearest_support = None
        if unfilled_gaps:
            supports = [g["prev_close"] for g in unfilled_gaps if g["prev_close"] < price]
            if supports:
                nearest_support = round(max(supports), 2)

        # ── Gap récent principal ──
        recent_gap_pct = 0.0
        if recent_gaps:
            latest = sorted(recent_gaps, key=lambda x: x["days_ago"])[0]
            recent_gap_pct = latest["gap_pct"]

        top_signal = signals[0]["name"] if signals else None
        summary    = " | ".join(s["name"] for s in signals) if signals else "Aucun gap significatif"

        return {
            "signals":        signals,
            "top_signal":     top_signal,
            "score":          score,
            "badge":          badge,
            "recent_gap_pct": recent_gap_pct,
            "gap_direction":  direction,
            "unfilled_gaps":  unfilled_gaps,
            "nearest_support":nearest_support,
            "n_gaps":         len(gaps),
            "n_bullish":      len(bullish_gaps),
            "summary":        summary,
        }

    except Exception:
        return _empty_gap()


def _empty_gap():
    return {
        "signals":         [],
        "top_signal":      None,
        "score":           0,
        "badge":           "—",
        "recent_gap_pct":  0.0,
        "gap_direction":   "NONE",
        "unfilled_gaps":   [],
        "nearest_support": None,
        "n_gaps":          0,
        "n_bullish":       0,
        "summary":         "—",
    }
