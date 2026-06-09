import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 📐 FIBONACCI V2 — FILTRE DE VALIDATION
# + BOLLINGER SIGNALS AVANCÉS
# ─────────────────────────────────────────────

def calc_fibonacci(hist, swing_lookback=60):
    if hist is None or hist.empty or len(hist) < 20:
        return _empty_fib()
    try:
        close = hist["Close"]
        price = float(close.iloc[-1])
        recent = hist.iloc[-swing_lookback:] if len(hist) >= swing_lookback else hist
        swing_low  = float(recent["Low"].min())
        swing_high = float(recent["High"].max())
        swing_range = swing_high - swing_low
        if swing_range <= 0 or swing_low <= 0:
            return _empty_fib()

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
        below = {k: v for k, v in all_lvl.items() if v < price * 0.999}
        above = {k: v for k, v in all_lvl.items() if v > price * 1.001}
        ns = max(below.items(), key=lambda x: x[1]) if below else None
        nr = min(above.items(), key=lambda x: x[1]) if above else None
        dist_r = round((nr[1] - price) / price * 100, 2) if nr else 99.0
        dist_s = round((price - ns[1]) / price * 100, 2) if ns else 99.0

        rebond = None
        for lvl in ["38.2%", "50.0%", "61.8%"]:
            lp = retrace[lvl]
            if abs(price - lp) / price * 100 <= 2.0 and price >= lp:
                rebond = (lvl, lp)
                break

        resist_danger = nr is not None and dist_r <= 2.0
        breakout      = price >= swing_high * 0.998
        zone_saine    = retrace["50.0%"] <= price <= retrace["23.6%"]

        score = 0; signal = None; warning = None; entry_valid = True

        if breakout:
            price_context = "BREAKOUT"; score = 20; entry_valid = True
            signal = f"Breakout Fib — au-dessus du swing high ${round(swing_high,2)}"
        elif rebond:
            ln, lp = rebond; price_context = "REBOND_KEY"; entry_valid = True
            if ln == "61.8%":   score = 18; signal = f"Rebond Golden Ratio 61.8% a ${lp} — entree optimale"
            elif ln == "50.0%": score = 14; signal = f"Rebond niveau 50% a ${lp} — setup solide"
            elif ln == "38.2%": score = 10; signal = f"Rebond niveau 38.2% a ${lp}"
        elif resist_danger:
            price_context = "RESISTANCE_PROCHE"; score = -15; entry_valid = False
            signal  = f"Resistance Fib {nr[0]} dans {dist_r}% — ENTREE RISQUEE"
            warning = f"Prix a {dist_r}% d'une resistance Fib majeure ({nr[0]} a ${nr[1]}). Risque de rejet."
        elif zone_saine:
            price_context = "ZONE_SAINE"; score = 8; entry_valid = True
            signal = f"Zone Fib saine (entre 23.6% et 50%)"
        else:
            price_context = "NEUTRE"; score = 2; entry_valid = True
            signal = f"Position Fib neutre"

        fib_stop = round(ns[1] * 0.995, 2) if ns else None
        fib_target = None
        for ext_label in ["127.2%", "161.8%"]:
            if ext_label in extend and extend[ext_label] > price * 1.01:
                fib_target = extend[ext_label]
                break
        fib_rr = None
        if fib_stop and fib_target and fib_stop < price:
            risk = price - fib_stop; reward = fib_target - price
            if risk > 0: fib_rr = round(reward / risk, 2)

        if score >= 18:    badge = "Fib Optimal (rebond cle)"
        elif score >= 12:  badge = "Fib Favorable"
        elif score >= 5:   badge = "Fib Zone correcte"
        elif score > 0:    badge = "Fib Neutre"
        elif score <= -10: badge = "RESISTANCE Fib — Eviter"
        else:              badge = "Fib Defavorable"

        return {
            "swing_low": round(swing_low,2), "swing_high": round(swing_high,2),
            "swing_range": round(swing_range,2), "retracements": retrace, "extensions": extend,
            "price_context": price_context, "entry_valid": entry_valid, "entry_reason": signal or "—",
            "fib_stop": fib_stop, "fib_target": fib_target, "fib_rr": fib_rr,
            "nearest_support_fib": ns, "nearest_resist_fib": nr,
            "dist_to_resist": dist_r, "dist_to_support": dist_s,
            "signal": signal, "score": score, "badge": badge, "warning": warning,
            "all_levels": [], "current_level": ns[0] if ns else "—",
            "at_key_level": rebond is not None or breakout,
        }
    except Exception:
        return _empty_fib()


def detect_bollinger_signals(hist, period=20, std_mult=2.0):
    if hist is None or hist.empty or len(hist) < period + 5:
        return _empty_bb()
    try:
        close = hist["Close"]; volume = hist["Volume"]
        price = float(close.iloc[-1])
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = ma + std_mult * std
        lower = ma - std_mult * std
        bb_upper = float(upper.iloc[-1]); bb_lower = float(lower.iloc[-1]); bb_mid = float(ma.iloc[-1])
        bb_width = round((bb_upper - bb_lower) / bb_mid * 100, 2) if bb_mid > 0 else 0
        bb_pct = round((price - bb_lower) / (bb_upper - bb_lower), 3) if (bb_upper - bb_lower) > 0 else 0.5
        if len(std) >= 10:
            wn = float(std.iloc[-1]); wp = float(std.iloc[-6])
            width_trend = "EXPANDING" if wn > wp*1.1 else "CONTRACTING" if wn < wp*0.9 else "NEUTRAL"
        else:
            width_trend = "NEUTRAL"
        avg_vol = float(volume.rolling(20).mean().iloc[-1])
        last_vol = float(volume.iloc[-1])
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
        signals = []; score = 0
        prev_close = float(close.iloc[-2])
        prev_upper = float(upper.iloc[-2])
        prev_lower = float(lower.iloc[-2])

        if price > bb_upper and prev_close <= prev_upper:
            if vol_ratio >= 1.5: signals.append("BB Breakout haussier avec volume fort"); score += 15
            else: signals.append("BB Breakout haussier"); score += 8
        elif price > bb_upper:
            wd = 0; rc = close.iloc[-10:].values; ru = upper.iloc[-10:].values
            for i in range(len(rc)-1, -1, -1):
                if rc[i] > ru[i]: wd += 1
                else: break
            if wd >= 3: signals.append(f"BB Walk haussier ({wd} jours)"); score += 12
            elif wd >= 2: signals.append(f"BB Walk ({wd} jours)"); score += 6

        if width_trend == "EXPANDING" and bb_width < 5.0:
            signals.append("BB Squeeze libere"); score += 10
        elif bb_pct <= 0.1 and price > prev_close:
            signals.append(f"Rebond bande BB inferieure"); score += 8
        elif 0.1 < bb_pct < 0.45 and price > prev_close:
            signals.append("Retour vers BB moyenne"); score += 4

        if price < bb_lower and prev_close >= prev_lower:
            signals.append("BB Breakout baissier"); score -= 10

        if score >= 15:   badge = "BB Breakout fort"
        elif score >= 10: badge = "BB Signal haussier"
        elif score >= 5:  badge = f"BB Position {round(bb_pct*100,0)}%"
        elif score < 0:   badge = "BB Signal baissier"
        else:             badge = f"BB Neutre ({width_trend})"

        return {
            "signal": signals[0] if signals else None, "all_signals": signals,
            "score": score, "badge": badge,
            "bb_upper": round(bb_upper,2), "bb_lower": round(bb_lower,2), "bb_mid": round(bb_mid,2),
            "bb_width": bb_width, "bb_pct": bb_pct, "width_trend": width_trend, "vol_ratio": round(vol_ratio,2),
        }
    except Exception:
        return _empty_bb()


def _empty_fib():
    return {
        "swing_low": None, "swing_high": None, "swing_range": 0,
        "retracements": {}, "extensions": {}, "price_context": "NEUTRE",
        "entry_valid": True, "entry_reason": "—", "fib_stop": None,
        "fib_target": None, "fib_rr": None, "nearest_support_fib": None,
        "nearest_resist_fib": None, "dist_to_resist": 99.0, "dist_to_support": 99.0,
        "signal": None, "score": 0, "badge": "—", "warning": None,
        "all_levels": [], "current_level": "—", "at_key_level": False,
    }


def _empty_bb():
    return {
        "signal": None, "all_signals": [], "score": 0, "badge": "—",
        "bb_upper": None, "bb_lower": None, "bb_mid": None,
        "bb_width": 0, "bb_pct": 0.5, "width_trend": "NEUTRAL", "vol_ratio": 1.0,
    }
