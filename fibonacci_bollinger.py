import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 📐 FIBONACCI LEVELS + BOLLINGER SIGNALS
# Retracements · Extensions · BB Breakout · BB Walk
# from fibonacci_bollinger import calc_fibonacci, detect_bollinger_signals
# ─────────────────────────────────────────────

def calc_fibonacci(hist, swing_lookback=60):
    """
    Calcule les niveaux de Fibonacci basés sur le dernier swing haussier.

    Retracements : 23.6%, 38.2%, 50%, 61.8%, 78.6%
    Extensions   : 127.2%, 161.8%, 200%, 261.8%

    Retourne :
    - swing_low      : bas du swing
    - swing_high     : haut du swing
    - retracements   : dict {niveau: prix}
    - extensions     : dict {niveau: prix}
    - current_level  : niveau Fibonacci le plus proche du prix actuel
    - nearest_support: support Fib le plus proche sous le prix
    - nearest_resist : résistance Fib la plus proche au-dessus
    - signal         : signal principal
    - score          : bonus pts
    - badge          : label lisible
    - at_key_level   : bool — prix sur un niveau clé
    """
    if hist is None or hist.empty or len(hist) < 20:
        return _empty_fib()

    try:
        close  = hist["Close"]
        high   = hist["High"]
        low    = hist["Low"]
        price  = float(close.iloc[-1])

        # Données récentes pour le swing
        recent = hist.iloc[-swing_lookback:] if len(hist) >= swing_lookback else hist

        # Trouver le swing low et swing high récents
        swing_low  = float(recent["Low"].min())
        swing_high = float(recent["High"].max())
        swing_range = swing_high - swing_low

        if swing_range <= 0:
            return _empty_fib()

        # Niveaux de retracement (depuis le haut vers le bas)
        retrace_levels = {
            "23.6%": round(swing_high - 0.236 * swing_range, 2),
            "38.2%": round(swing_high - 0.382 * swing_range, 2),
            "50.0%": round(swing_high - 0.500 * swing_range, 2),
            "61.8%": round(swing_high - 0.618 * swing_range, 2),
            "78.6%": round(swing_high - 0.786 * swing_range, 2),
        }

        # Extensions (au-dessus du swing high)
        extend_levels = {
            "127.2%": round(swing_high + 0.272 * swing_range, 2),
            "161.8%": round(swing_high + 0.618 * swing_range, 2),
            "200.0%": round(swing_high + 1.000 * swing_range, 2),
            "261.8%": round(swing_high + 1.618 * swing_range, 2),
        }

        # Tous les niveaux combinés
        all_levels = {**retrace_levels, **extend_levels}

        # Niveau le plus proche du prix actuel
        nearest = min(all_levels.items(), key=lambda x: abs(x[1] - price))
        dist_nearest = round(abs(nearest[1] - price) / price * 100, 2)
        at_key_level = dist_nearest <= 1.5

        # Support le plus proche sous le prix
        supports = {k: v for k, v in all_levels.items() if v < price * 0.999}
        nearest_support = max(supports.items(), key=lambda x: x[1]) if supports else None

        # Résistance la plus proche au-dessus
        resists = {k: v for k, v in all_levels.items() if v > price * 1.001}
        nearest_resist = min(resists.items(), key=lambda x: x[1]) if resists else None

        # Position dans le range (0% = bas, 100% = haut)
        position_pct = round((price - swing_low) / swing_range * 100, 1)

        # ── Scoring ──
        score  = 0
        signal = None

        # Prix sur un niveau clé de retracement
        key_retracements = ["38.2%", "50.0%", "61.8%"]
        for lvl_name in key_retracements:
            lvl_price = retrace_levels[lvl_name]
            dist = abs(price - lvl_price) / price * 100
            if dist <= 1.5:
                if price >= lvl_price:  # Prix au-dessus = rebond confirmé
                    if lvl_name == "61.8%":
                        score  = 18
                        signal = f"🚀 Rebond niveau Fib 61.8% (Golden Ratio) à ${lvl_price}"
                    elif lvl_name == "50.0%":
                        score  = 14
                        signal = f"✅ Rebond niveau Fib 50% à ${lvl_price}"
                    elif lvl_name == "38.2%":
                        score  = 10
                        signal = f"✅ Rebond niveau Fib 38.2% à ${lvl_price}"
                else:  # Prix sous le niveau = résistance
                    score  = -5
                    signal = f"⚠️ Résistance Fib {lvl_name} à ${lvl_price} (+{round(dist,1)}%)"
                break

        # Prix proche d'une extension (target naturel)
        for lvl_name, lvl_price in extend_levels.items():
            dist = abs(price - lvl_price) / price * 100
            if dist <= 2.0 and price < lvl_price:
                if nearest_resist and nearest_resist[0] == lvl_name:
                    score += 5
                    if not signal:
                        signal = f"📈 Extension Fib {lvl_name} comme target: ${lvl_price}"
                break

        # Zone saine (entre 38.2% et 61.8% du retracement)
        r382 = retrace_levels["38.2%"]
        r618 = retrace_levels["61.8%"]
        if r618 <= price <= r382 and score == 0:
            score  = 6
            signal = f"~ Zone Fib saine (entre 38.2% et 61.8%)"

        # Badge
        if score >= 15:
            badge = f"🚀 Niveau Fib clé (Golden Ratio)"
        elif score >= 10:
            badge = f"✅ Niveau Fib fort"
        elif score >= 5:
            badge = f"📈 Zone Fib correcte"
        elif score > 0:
            badge = f"~ Zone Fib neutre"
        elif score < 0:
            badge = f"⚠️ Résistance Fib proche"
        else:
            badge = "—"

        return {
            "swing_low":       round(swing_low, 2),
            "swing_high":      round(swing_high, 2),
            "swing_range":     round(swing_range, 2),
            "retracements":    retrace_levels,
            "extensions":      extend_levels,
            "current_level":   nearest[0],
            "dist_to_nearest": dist_nearest,
            "nearest_support": nearest_support,
            "nearest_resist":  nearest_resist,
            "position_pct":    position_pct,
            "at_key_level":    at_key_level,
            "signal":          signal,
            "score":           score,
            "badge":           badge,
        }

    except Exception:
        return _empty_fib()


def detect_bollinger_signals(hist, period=20, std_mult=2.0):
    """
    Détecte les signaux avancés des Bandes de Bollinger.

    Signaux :
    - BB_BREAKOUT_UP   : prix casse au-dessus de la bande sup avec volume → +15 pts
    - BB_WALK_UP       : prix marche le long de la bande sup 3+ jours → +12 pts
    - BB_SQUEEZE_FIRE  : squeeze qui se libère → +18 pts (complément TTM)
    - BB_REVERSAL      : prix revient vers la moyenne depuis la bande inf → +8 pts
    - BB_BREAKOUT_DOWN : cassure baissière → -10 pts

    Retourne :
    - signal     : signal principal
    - score      : bonus pts
    - badge      : label lisible
    - bb_upper   : bande supérieure actuelle
    - bb_lower   : bande inférieure actuelle
    - bb_mid     : bande moyenne (MA20)
    - bb_width   : largeur des bandes en %
    - bb_pct     : position dans les bandes (0=bas, 1=haut)
    - width_trend: "EXPANDING" / "CONTRACTING" / "NEUTRAL"
    """
    if hist is None or hist.empty or len(hist) < period + 5:
        return _empty_bb()

    try:
        close  = hist["Close"]
        volume = hist["Volume"]
        price  = float(close.iloc[-1])

        # Calcul Bollinger
        ma     = close.rolling(period).mean()
        std    = close.rolling(period).std()
        upper  = ma + std_mult * std
        lower  = ma - std_mult * std

        bb_upper = float(upper.iloc[-1])
        bb_lower = float(lower.iloc[-1])
        bb_mid   = float(ma.iloc[-1])
        bb_width = round((bb_upper - bb_lower) / bb_mid * 100, 2) if bb_mid > 0 else 0
        bb_pct   = round((price - bb_lower) / (bb_upper - bb_lower), 3) if (bb_upper - bb_lower) > 0 else 0.5

        # Tendance de la largeur (expansion vs contraction)
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

        # Volume moyen
        avg_vol   = float(volume.rolling(20).mean().iloc[-1])
        last_vol  = float(volume.iloc[-1])
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

        signals = []
        score   = 0

        # ── 1. BB BREAKOUT HAUSSIER ──
        # Prix qui casse au-dessus de la bande supérieure avec volume
        prev_close = float(close.iloc[-2])
        prev_upper = float(upper.iloc[-2])

        if price > bb_upper and prev_close <= prev_upper:
            if vol_ratio >= 1.5:
                signals.append("🚀 BB Breakout haussier avec volume fort")
                score += 15
            else:
                signals.append("⚡ BB Breakout haussier")
                score += 8

        # ── 2. BB WALK (marche le long de la bande) ──
        # Prix reste au-dessus de la bande supérieure 3+ jours consécutifs
        elif price > bb_upper:
            walk_days = 0
            recent_prices = close.iloc[-10:].values
            recent_uppers = upper.iloc[-10:].values
            for i in range(len(recent_prices) - 1, -1, -1):
                if recent_prices[i] > recent_uppers[i]:
                    walk_days += 1
                else:
                    break
            if walk_days >= 3:
                signals.append(f"✅ BB Walk haussier ({walk_days} jours) — tendance très forte")
                score += 12
            elif walk_days >= 2:
                signals.append(f"~ BB Walk haussier ({walk_days} jours)")
                score += 6

        # ── 3. BB SQUEEZE LIBÉRÉ ──
        # Bandes qui se contractent puis s'expandent — énergie libérée
        if width_trend == "EXPANDING" and bb_width < 5.0:
            # Squeeze qui se libère
            signals.append("⚡ BB Squeeze libéré — expansion de volatilité")
            score += 10

        # ── 4. REBOND DEPUIS BANDE INFÉRIEURE ──
        # Prix rebondit depuis la bande basse vers la moyenne
        elif bb_pct <= 0.1 and price > prev_close:
            signals.append(f"✅ Rebond bande BB inférieure (${round(bb_lower,2)})")
            score += 8

        # ── 5. RETOUR VERS LA MOYENNE ──
        # Prix entre bande inf et moyenne, remontant vers MA20
        elif 0.1 < bb_pct < 0.5 and price > prev_close:
            signals.append(f"~ Retour vers BB moyenne (${round(bb_mid,2)})")
            score += 4

        # ── 6. BREAKOUT BAISSIER ──
        if price < bb_lower and prev_close >= float(lower.iloc[-2]):
            signals.append(f"🔴 BB Breakout baissier — faiblesse")
            score -= 10

        # ── Badge ──
        if score >= 15:
            badge = "🚀 BB Breakout fort"
        elif score >= 10:
            badge = "✅ BB Signal haussier"
        elif score >= 5:
            badge = f"~ BB Position correcte ({round(bb_pct*100,0)}%)"
        elif score < 0:
            badge = "🔴 BB Signal baissier"
        else:
            badge = "—"

        main_signal = signals[0] if signals else None

        return {
            "signal":      main_signal,
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
        "retracements": {}, "extensions": {}, "current_level": "—",
        "dist_to_nearest": 0, "nearest_support": None, "nearest_resist": None,
        "position_pct": 50, "at_key_level": False,
        "signal": None, "score": 0, "badge": "—",
    }


def _empty_bb():
    return {
        "signal": None, "all_signals": [], "score": 0, "badge": "—",
        "bb_upper": None, "bb_lower": None, "bb_mid": None,
        "bb_width": 0, "bb_pct": 0.5, "width_trend": "NEUTRAL", "vol_ratio": 1.0,
    }
