import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 📐 SUPPORT & RÉSISTANCE 52 SEMAINES
# Contexte de prix annuel pour valider l'entrée
# from support_resistance import calc_sr_levels
# ─────────────────────────────────────────────

def calc_sr_levels(hist):
    """
    Calcule les niveaux de support et résistance clés sur 52 semaines.

    Niveaux détectés :
    - Plus haut 52 semaines (résistance majeure)
    - Plus bas 52 semaines (support majeur)
    - Niveaux ronds psychologiques proches
    - Position dans le range annuel (0-100%)
    - Zones de consolidation (clustering de prix)

    Retourne un dict :
    - high_52w       : plus haut 52 semaines
    - low_52w        : plus bas 52 semaines
    - range_52w      : range total en $
    - range_pct      : range total en %
    - position_pct   : position dans le range (0%=bas, 100%=haut)
    - dist_to_high   : distance au plus haut en %
    - dist_to_low    : distance au plus bas en %
    - nearest_round  : niveau rond psychologique le plus proche
    - dist_to_round  : distance au niveau rond en %
    - signal         : signal principal
    - score          : bonus/pénalité pts
    - badge          : label lisible
    - setup_quality  : "EXCELLENT" / "BON" / "ACCEPTABLE" / "RISQUÉ"
    - stop_natural   : niveau de stop naturel suggéré
    - target_natural : niveau de target naturel suggéré
    - key_levels     : liste des niveaux clés
    """
    if hist is None or hist.empty or len(hist) < 50:
        return _empty_sr()

    try:
        close  = hist["Close"]
        high   = hist["High"]
        low    = hist["Low"]
        volume = hist["Volume"]

        # Données 52 semaines
        hist_252 = hist.iloc[-252:] if len(hist) >= 252 else hist
        high_52w = float(hist_252["High"].max())
        low_52w  = float(hist_252["Low"].min())
        price    = float(close.iloc[-1])

        range_52w  = round(high_52w - low_52w, 2)
        range_pct  = round((high_52w - low_52w) / low_52w * 100, 1) if low_52w > 0 else 0

        # Position dans le range annuel (0% = plus bas, 100% = plus haut)
        position_pct = round((price - low_52w) / (high_52w - low_52w) * 100, 1) if range_52w > 0 else 50

        # Distance aux extrêmes
        dist_to_high = round((high_52w - price) / price * 100, 2)
        dist_to_low  = round((price - low_52w) / price * 100, 2)

        # ── Niveaux ronds psychologiques ──
        round_levels = _find_round_levels(price)
        nearest_round = None
        dist_to_round = 999
        for lvl in round_levels:
            d = abs(price - lvl) / price * 100
            if d < dist_to_round:
                dist_to_round = round(d, 2)
                nearest_round = lvl

        # ── Zones de consolidation (support/résistance dynamiques) ──
        key_levels = _find_key_levels(hist_252)

        # ── Stop et target naturels basés sur S/R ──
        # Stop : premier support significatif sous le prix
        support_levels = [lvl for lvl in key_levels if lvl < price * 0.999]
        stop_natural   = round(max(support_levels), 2) if support_levels else round(price * 0.96, 2)

        # Target : première résistance significative au-dessus
        resist_levels  = [lvl for lvl in key_levels if lvl > price * 1.001]
        target_natural = round(min(resist_levels), 2) if resist_levels else round(price * 1.06, 2)

        # ── Signaux et scoring ──
        signals = []
        score   = 0

        # Breakout 52 semaines
        if price >= high_52w * 0.998:
            # Prix AU niveau du plus haut → breakout en cours
            signals.append("🚀 BREAKOUT 52 SEMAINES en cours")
            score += 20
            setup_quality = "EXCELLENT"

        elif dist_to_high <= 2.0:
            # Très proche du plus haut → pre-breakout
            signals.append(f"⚡ Pré-breakout 52w — résistance dans {dist_to_high}%")
            score += 12
            setup_quality = "BON"

        elif dist_to_high <= 5.0:
            # Proche du plus haut → attention résistance
            signals.append(f"⚠️ Résistance 52w proche ({dist_to_high}% au-dessus)")
            score -= 5
            setup_quality = "ACCEPTABLE"

        elif 20 <= position_pct <= 80:
            # Zone saine du range
            signals.append(f"✅ Zone saine du range annuel ({position_pct}%)")
            score += 5
            setup_quality = "BON"

        elif position_pct > 80:
            # Haut du range mais pas encore au sommet
            signals.append(f"~ Haut du range annuel ({position_pct}%)")
            score += 2
            setup_quality = "ACCEPTABLE"

        elif position_pct < 15:
            # Très bas du range → rebond possible mais risqué
            signals.append(f"⚠️ Bas du range annuel ({position_pct}%) — support ${round(low_52w,2)}")
            score -= 5
            setup_quality = "RISQUÉ"

        else:
            setup_quality = "ACCEPTABLE"

        # Niveau rond proche
        if dist_to_round <= 1.0 and nearest_round and nearest_round > price:
            signals.append(f"⚠️ Niveau rond ${nearest_round} dans {dist_to_round}% (résistance psychologique)")
            score -= 3
        elif dist_to_round <= 1.0 and nearest_round and nearest_round < price:
            signals.append(f"✅ Niveau rond ${nearest_round} comme support ({dist_to_round}% sous le prix)")
            score += 3

        # Rebond depuis support annuel
        if dist_to_low <= 5.0 and dist_to_high >= 10.0:
            signals.append(f"✅ Rebond depuis support annuel (${round(low_52w,2)} — +{dist_to_low}%)")
            score += 8

        score = max(-15, min(score, 22))

        # ── Badge ──
        if score >= 18:
            badge = f"🚀 Breakout 52w"
        elif score >= 10:
            badge = f"✅ Zone favorable (pos. {position_pct}%)"
        elif score >= 0:
            badge = f"~ Position neutre ({position_pct}% du range)"
        elif score >= -5:
            badge = f"⚠️ Résistance proche ({dist_to_high}%)"
        else:
            badge = f"🔴 Zone risquée"

        # Signal principal
        main_signal = signals[0] if signals else f"Position {position_pct}% du range annuel"

        return {
            "high_52w":       round(high_52w, 2),
            "low_52w":        round(low_52w, 2),
            "range_52w":      range_52w,
            "range_pct":      range_pct,
            "position_pct":   position_pct,
            "dist_to_high":   dist_to_high,
            "dist_to_low":    dist_to_low,
            "nearest_round":  nearest_round,
            "dist_to_round":  round(dist_to_round, 2),
            "signal":         main_signal,
            "all_signals":    signals,
            "score":          score,
            "badge":          badge,
            "setup_quality":  setup_quality,
            "stop_natural":   stop_natural,
            "target_natural": target_natural,
            "key_levels":     key_levels,
        }

    except Exception:
        return _empty_sr()


def _find_round_levels(price):
    """Trouve les niveaux ronds psychologiques autour du prix."""
    levels = []

    if price < 20:
        step = 1
    elif price < 50:
        step = 5
    elif price < 100:
        step = 5
    elif price < 500:
        step = 25
    elif price < 1000:
        step = 50
    else:
        step = 100

    base = round(price / step) * step
    for mult in [-3, -2, -1, 0, 1, 2, 3]:
        lvl = base + mult * step
        if lvl > 0:
            levels.append(round(lvl, 2))

    return sorted(set(levels))


def _find_key_levels(hist, n_levels=5):
    """
    Trouve les niveaux de prix clés par clustering de hauts/bas.
    Retourne les n_levels niveaux les plus significatifs.
    """
    try:
        closes = hist["Close"].values
        highs  = hist["High"].values
        lows   = hist["Low"].values

        # Hauts et bas locaux
        pivots = []
        for i in range(2, len(closes) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1] and \
               highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                pivots.append(float(highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i+1] and \
               lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                pivots.append(float(lows[i]))

        if not pivots:
            return []

        # Clustering simple — regrouper les niveaux proches
        pivots_sorted = sorted(pivots)
        clusters      = []
        current       = [pivots_sorted[0]]

        for p in pivots_sorted[1:]:
            if p <= current[-1] * 1.02:  # Dans 2% du niveau précédent
                current.append(p)
            else:
                clusters.append(np.mean(current))
                current = [p]
        clusters.append(np.mean(current))

        return [round(c, 2) for c in sorted(clusters)]

    except Exception:
        return []


def _empty_sr():
    return {
        "high_52w":       None,
        "low_52w":        None,
        "range_52w":      0,
        "range_pct":      0,
        "position_pct":   50,
        "dist_to_high":   0,
        "dist_to_low":    0,
        "nearest_round":  None,
        "dist_to_round":  0,
        "signal":         "—",
        "all_signals":    [],
        "score":          0,
        "badge":          "—",
        "setup_quality":  "ACCEPTABLE",
        "stop_natural":   None,
        "target_natural": None,
        "key_levels":     [],
    }
