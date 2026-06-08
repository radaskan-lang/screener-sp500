import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 📐 FIBONACCI — FILTRE DE VALIDATION
# Version 2 — Fibonacci comme filtre clé
# Valide l'entrée · Affine stop · Définit target
# Bloque les setups dangereux
# ─────────────────────────────────────────────

FIB_RETRACE = {
    "23.6%": 0.236,
    "38.2%": 0.382,
    "50.0%": 0.500,
    "61.8%": 0.618,
    "78.6%": 0.786,
}

FIB_EXTEND = {
    "100%":   1.000,
    "127.2%": 1.272,
    "161.8%": 1.618,
    "200%":   2.000,
    "261.8%": 2.618,
}

# Niveaux clés — ceux qui comptent vraiment
KEY_RETRACES  = ["38.2%", "50.0%", "61.8%"]
KEY_EXTENDS   = ["127.2%", "161.8%"]
TOLERANCE_PCT = 2.0  # ±2% pour considérer "sur le niveau"


def calc_fibonacci(hist, swing_lookback=60):
    """
    Fibonacci comme filtre de validation complet.

    Logique :
    1. Calcule le swing récent (haut/bas sur 60 jours)
    2. Calcule tous les niveaux retracements + extensions
    3. Évalue le contexte du prix actuel
    4. Valide ou invalide l'entrée
    5. Suggère stop et target Fibonacci optimaux

    Retourne :
    - swing_low / swing_high / swing_range
    - retracements / extensions : dict {label: prix}
    - price_context  : "REBOND_KEY" / "ZONE_SAINE" / "RESISTANCE_PROCHE" / "BREAKOUT" / "NEUTRE"
    - entry_valid    : bool — Fibonacci valide-t-il l'entrée ?
    - entry_reason   : explication
    - fib_stop       : stop-loss Fibonacci optimal
    - fib_target     : target Fibonacci optimal
    - fib_rr         : R/R basé sur niveaux Fib
    - nearest_support_fib : support Fib le plus proche sous le prix
    - nearest_resist_fib  : résistance Fib la plus proche au-dessus
    - dist_to_resist : distance à la résistance en %
    - dist_to_support: distance au support en %
    - signal         : signal principal
    - score          : pts (positif ou négatif fort)
    - badge          : label lisible
    - warning        : avertissement si setup dangereux
    - all_levels     : tous les niveaux pour affichage
    """
    if hist is None or hist.empty or len(hist) < 20:
        return _empty_fib()

    try:
        close = hist["Close"]
        high  = hist["High"]
        low   = hist["Low"]
        price = float(close.iloc[-1])

        recent      = hist.iloc[-swing_lookback:] if len(hist) >= swing_lookback else hist
        swing_low   = float(recent["Low"].min())
        swing_high  = float(recent["High"].max())
        swing_range = swing_high - swing_low

        if swing_range <= 0 or swing_low <= 0:
            return _empty_fib()

        # ── Calcul de tous les niveaux ──
        retracements = {}
        for label, ratio in FIB_RETRACE.items():
            retracements[label] = round(swing_high - ratio * swing_range, 2)

        extensions = {}
        for label, ratio in FIB_EXTEND.items():
            extensions[label] = round(swing_low + ratio * swing_range, 2)

        all_levels = {**retracements, **extensions}

        # ── Niveaux les plus proches ──
        levels_below = {k: v for k, v in all_levels.items() if v < price * 0.999}
        levels_above = {k: v for k, v in all_levels.items() if v > price * 1.001}

        nearest_support_fib = max(levels_below.items(), key=lambda x: x[1]) if levels_below else None
        nearest_resist_fib  = min(levels_above.items(), key=lambda x: x[1]) if levels_above else None

        dist_to_resist  = round((nearest_resist_fib[1] - price) / price * 100, 2) if nearest_resist_fib else 99.0
        dist_to_support = round((price - nearest_support_fib[1]) / price * 100, 2) if nearest_support_fib else 99.0

        # ── Contexte du prix ──
        # 1. REBOND sur niveau clé de retracement
        rebond_level = None
        for lvl in KEY_RETRACES:
            lvl_price = retracements[lvl]
            dist      = abs(price - lvl_price) / price * 100
            if dist <= TOLERANCE_PCT and price >= lvl_price:
                rebond_level = (lvl, lvl_price)
                break

        # 2. RÉSISTANCE proche (extension ou retracement au-dessus)
        resist_danger = nearest_resist_fib and dist_to_resist <= TOLERANCE_PCT

        # 3. BREAKOUT au-dessus du swing high
        breakout = price >= swing_high * 0.998

        # 4. Zone saine (entre 23.6% et 50% du retracement)
        r236 = retracements["23.6%"]
        r500 = retracements["50.0%"]
        zone_saine = r500 <= price <= r236

        # ── Évaluation ──
        score       = 0
        signal      = None
        warning     = None
        entry_valid = True

        if breakout:
            price_context = "BREAKOUT"
            score         = 20
            entry_valid   = True
            signal        = f"🚀 Breakout Fibonacci — au-dessus du swing high ${round(swing_high,2)}"

        elif rebond_level:
            lvl_name, lvl_price = rebond_level
            price_context = "REBOND_KEY"
            if lvl_name == "61.8%":
                score  = 18
                signal = f"🚀 Rebond Golden Ratio 61.8% à ${lvl_price} — entrée optimale"
            elif lvl_name == "50.0%":
                score  = 14
                signal = f"✅ Rebond niveau 50% à ${lvl_price} — setup solide"
            elif lvl_name == "38.2%":
                score  = 10
                signal = f"✅ Rebond niveau 38.2% à ${lvl_price}"
            entry_valid = True

        elif resist_danger:
            price_context = "RESISTANCE_PROCHE"
            resist_name   = nearest_resist_fib[0]
            resist_price  = nearest_resist_fib[1]
            score         = -15
            entry_valid   = False
            signal        = f"🔴 Résistance Fib {resist_name} dans {dist_to_resist}% — ENTRÉE RISQUÉE"
            warning       = f"⚠️ Prix à {dist_to_resist}% d'une résistance Fibonacci majeure ({resist_name} à ${resist_price}). Le titre va probablement se heurter à ce niveau."

        elif zone_saine:
            price_context = "ZONE_SAINE"
            score         = 8
            entry_valid   = True
            signal        = f"✅ Zone Fib saine (entre 23.6% et 50%) — bon contexte"

        else:
            price_context = "NEUTRE"
            score         = 2
            entry_valid   = True
            signal        = f"~ Position Fib neutre"

        # ── Stop Fibonacci optimal ──
        # Le stop se place juste sous le support Fib le plus proche
        fib_stop = None
        if nearest_support_fib:
            fib_stop = round(nearest_support_fib[1] * 0.995, 2)  # 0.5% sous le support

        # ── Target Fibonacci optimal ──
        # Utiliser l'extension la plus proche au-dessus
        fib_target = None
        for ext_label in KEY_EXTENDS:
            ext_price = extensions[ext_label]
            if ext_price > price * 1.01:
                fib_target = ext_price
                break

        # ── R/R Fibonacci ──
        fib_rr = None
        if fib_stop and fib_target and fib_stop < price:
            risk   = price - fib_stop
            reward = fib_target - price
            if risk > 0:
                fib_rr = round(reward / risk, 2)

        # ── Badge ──
        if score >= 18:
            badge = f"🚀 Fib Optimal (rebond clé)"
        elif score >= 12:
            badge = f"✅ Fib Favorable"
        elif score >= 5:
            badge = f"~ Fib Zone correcte"
        elif score > 0:
            badge = f"~ Fib Neutre"
        elif score <= -10:
            badge = f"🔴 RÉSISTANCE Fib — Éviter"
        else:
            badge = f"⚠️ Fib Défavorable"

        # Résumé de tous les niveaux pour affichage
        all_levels_display = []
        for label, lvl_price in sorted(retracements.items(), key=lambda x: x[1], reverse=True):
            marker = "▶" if abs(price - lvl_price) / price * 100 <= TOLERANCE_PCT else " "
            all_levels_display.append(f"{marker} {label}: ${lvl_price}")
        all_levels_display.append(f"--- Prix actuel: ${round(price,2)} ---")
        for label, lvl_price in sorted(extensions.items(), key=lambda x: x[1]):
            if lvl_price > price * 0.9:
                marker = "▶" if abs(price - lvl_price) / price * 100 <= TOLERANCE_PCT else " "
                all_levels_display.append(f"{marker} Ext {label}: ${lvl_price}")

        return {
            "swing_low":           round(swing_low, 2),
            "swing_high":          round(swing_high, 2),
            "swing_range":         round(swing_range, 2),
            "retracements":        retracements,
            "extensions":          extensions,
            "price_context":       price_context,
            "entry_valid":         entry_valid,
            "entry_reason":        signal or "—",
            "fib_stop":            fib_stop,
            "fib_target":          fib_target,
            "fib_rr":              fib_rr,
            "nearest_support_fib": nearest_support_fib,
            "nearest_resist_fib":  nearest_resist_fib,
            "dist_to_resist":      dist_to_resist,
            "dist_to_support":     dist_to_support,
            "signal":              signal,
            "score":               score,
            "badge":               badge,
            "warning":             warning,
            "all_levels":          all_levels_display,
            "current_level":       nearest_support_fib[0] if nearest_support_fib else "—",
            "at_key_level":        rebond_level is not None or breakout,
        }

    except Exception:
        return _empty_fib()


# ─────────────────────────────────────────────
# 📊 BOLLINGER SIGNALS AVANCÉS
# ─────────────────────────────────────────────

def detect_bollinger_signals(hist, period=20, std_mult=2.0):
    """
    Signaux avancés Bollinger Bands.
    BB Breakout · BB Walk · BB Squeeze libéré · Rebond bande inf
    """
    if hist is None or hist.empty or len(hist) < period + 5:
        return _empty_bb()

    try:
        close  = hist["Close"]
        volume = hist["Volume"]
        price  = float(close.iloc[-1])

        ma    = close.rolling(period).mean()
        std   = close.rolling(period).std()
        upper = ma + std_mult * std
        lower = ma - std_mult * std

        bb_upper = float(upper.iloc[-1])
        bb_lower = float(lower.iloc[-1])
        bb_mid   = float(ma.iloc[-1])
        bb_width = round((bb_upper - bb_lower) / bb_mid * 100, 2) if bb_mid > 0 else 0
        bb_pct   = round((price - bb_lower) / (bb_upper - bb_lower), 3) if (bb_upper - bb_lower) > 0 else 0.5

        if len(std) >= 10:
            width_now  = float(std.iloc[-1])
            width_prev = float(std.iloc[-6])
            if width_now > width_prev * 1.1:
                width_trend = "EXPANDING"
            elif width_now < width_prev * 0.9:
                width_trend = "CONTRACTING"
            else:
                width_trend = "NEUTRAL"
        else:
            width_trend = "NEUTRAL"

        avg_vol   = float(volume.rolling(20).mean().iloc[-1])
        last_vol  = float(volume.iloc[-1])
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

        signals = []
        score   = 0

        prev_close = float(close.iloc[-2])
        prev_upper = float(upper.iloc[-2])
        prev_lower = float(lower.iloc[-2])

        # 1. BB Breakout haussier
        if price > bb_upper and prev_close <= prev_upper:
            if vol_ratio >= 1.5:
                signals.append("🚀 BB Breakout haussier avec volume fort")
                score += 15
            else:
                signals.append("⚡ BB Breakout haussier")
                score += 8

        # 2. BB Walk (marche sur bande sup)
        elif price > bb_upper:
            walk_days = 0
            rc = close.iloc[-10:].values
            ru = upper.iloc[-10:].values
            for i in range(len(rc)-1, -1, -1):
                if rc[i] > ru[i]: walk_days += 1
                else: break
            if walk_days >= 3:
                signals.append(f"✅ BB Walk haussier ({walk_days} jours)")
                score += 12
            elif walk_days >= 2:
                signals.append(f"~ BB Walk ({walk_days} jours)")
                score += 6

        # 3. BB Squeeze libéré
        if width_trend == "EXPANDING" and bb_width < 5.0:
            signals.append("⚡ BB Squeeze libéré")
            score += 10

        # 4. Rebond bande inférieure
        elif bb_pct <= 0.1 and price > prev_close:
            signals.append(f"✅ Rebond bande BB inférieure (${round(bb_lower,2)})")
            score += 8

        # 5. Retour vers moyenne
        elif 0.1 < bb_pct < 0.45 and price > prev_close:
            signals.append(f"~ Retour vers BB moyenne (${round(bb_mid,2)})")
            score += 4

        # 6. Breakout baissier
        if price < bb_lower and prev_close >= prev_lower:
            signals.append(f"🔴 BB Breakout baissier")
            score -= 10

        if score >= 15:
            badge = "🚀 BB Breakout fort"
        elif score >= 10:
            badge = "✅ BB Signal haussier"
        elif score >= 5:
            badge = f"~ BB Position {round(bb_pct*100,0)}%"
        elif score < 0:
            badge = "🔴 BB Signal baissier"
        else:
            badge = f"~ BB Neutre ({width_trend})"

        return {
            "signal":      signals[0] if signals else None,
            "all_signals": signals,
            "score":       score,
            "badge":       badge,
            "bb_upper":    round(bb_upper, 2),
            "bb_lower":    round(bb_lower, 2),
            "bb_mid":      round(bb_mid, 2),
            "bb_width":    bb_width,
            "bb_pct":      bb_pct,
            "width_trend": width_trend,
            "vol_ratio":   round(vol_ratio, 2),
        }

    except Exception:
        return _empty_bb()


def _empty_fib():
    return {
        "swing_low": None, "swing_high": None, "swing_range": 0,
        "retracements": {}, "extensions": {}, "price_context": "NEUTRE",
        "entry_valid": True, "entry_reason": "—",
        "fib_stop": None, "fib_target": None, "fib_rr": None,
        "nearest_support_fib": None, "nearest_resist_fib": None,
        "dist_to_resist": 99.0, "dist_to_support": 99.0,
        "signal": None, "score": 0, "badge": "—",
        "warning": None, "all_levels": [], "current_level": "—",
        "at_key_level": False,
    }


def _empty_bb():
    return {
        "signal": None, "all_signals": [], "score": 0, "badge": "—",
        "bb_upper": None, "bb_lower": None, "bb_mid": None,
        "bb_width": 0, "bb_pct": 0.5, "width_trend": "NEUTRAL",
        "vol_ratio": 1.0,
    }
