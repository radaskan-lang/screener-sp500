import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 🔍 PATTERN DETECTION — SWING TRADING
# Intégrer dans app.py via : from pattern_detection import detect_all_patterns
# ─────────────────────────────────────────────

def detect_golden_death_cross(close, ma50, ma200):
    """
    Golden Cross : MA50 croise MA200 à la hausse (très récent = +25 pts)
    Death Cross  : MA50 croise MA200 à la baisse (-15 pts)
    """
    if len(close) < 5:
        return None, 0

    # Comparer les 5 derniers jours pour détecter un croisement récent
    ma50_series  = close.rolling(50).mean()
    ma200_series = close.rolling(200).mean()

    recent_50  = ma50_series.iloc[-5:]
    recent_200 = ma200_series.iloc[-5:]

    # Golden Cross : MA50 était sous MA200 puis passe au-dessus
    was_below = (recent_50.iloc[0] < recent_200.iloc[0])
    is_above  = (recent_50.iloc[-1] > recent_200.iloc[-1])

    if was_below and is_above:
        return "🟢 Golden Cross (croisement récent)", 25
    elif ma50 > ma200:
        return "🟡 Golden Cross (en place)", 15
    elif not was_below and not is_above:
        return "🔴 Death Cross", -15
    return None, 0


def detect_bull_flag(close, volume, lookback=20):
    """
    Bull Flag :
    - Fort mouvement haussier (mât) sur 5-10 jours
    - Consolidation légèrement baissière sur 5-10 jours (drapeau)
    - Volume décroissant pendant la consolidation
    """
    if len(close) < lookback:
        return None, 0

    recent = close.iloc[-lookback:]
    vol_recent = volume.iloc[-lookback:]

    # Mât : hausse forte sur première moitié
    pole = recent.iloc[:10]
    flag = recent.iloc[10:]

    pole_return = (pole.iloc[-1] - pole.iloc[0]) / pole.iloc[0] * 100
    flag_return  = (flag.iloc[-1] - flag.iloc[0]) / flag.iloc[0] * 100

    # Volume décroissant pendant le drapeau
    vol_flag = vol_recent.iloc[10:]
    vol_declining = vol_flag.iloc[-1] < vol_flag.mean()

    if pole_return > 8 and -5 < flag_return < 1 and vol_declining:
        return "🟢 Bull Flag (setup complet)", 22
    elif pole_return > 5 and flag_return < 2:
        return "🟡 Bull Flag (partiel)", 12
    return None, 0


def detect_double_bottom(close, lookback=60):
    """
    Double Bottom (W) :
    - Deux creux similaires séparés par un sommet intermédiaire
    - Deuxième creux légèrement plus haut que le premier
    - Prix remonte au-dessus du sommet intermédiaire = confirmation
    """
    if len(close) < lookback:
        return None, 0

    prices = close.iloc[-lookback:].values
    n = len(prices)

    # Trouver les creux locaux
    bottoms = []
    for i in range(2, n - 2):
        if prices[i] < prices[i-1] and prices[i] < prices[i+1] \
           and prices[i] < prices[i-2] and prices[i] < prices[i+2]:
            bottoms.append((i, prices[i]))

    if len(bottoms) < 2:
        return None, 0

    # Prendre les deux derniers creux
    b1_idx, b1_price = bottoms[-2]
    b2_idx, b2_price = bottoms[-1]

    # Vérifier écart temporel suffisant (au moins 10 jours)
    if b2_idx - b1_idx < 10:
        return None, 0

    # Les deux creux doivent être proches en prix (±5%)
    diff_pct = abs(b2_price - b1_price) / b1_price * 100
    if diff_pct > 5:
        return None, 0

    # Prix actuel au-dessus du deuxième creux (sortie du W)
    current = prices[-1]
    if current > b2_price * 1.03:
        return "🟢 Double Bottom confirmé (W)", 20
    elif diff_pct < 3:
        return "🟡 Double Bottom en formation", 10
    return None, 0


def detect_cup_and_handle(close, lookback=90):
    """
    Cup & Handle :
    - Forme en U sur 30-90 jours (coupe)
    - Légère consolidation après le sommet droit (anse)
    - Volume en baisse pendant l'anse
    """
    if len(close) < lookback:
        return None, 0

    prices = close.iloc[-lookback:].values
    n = len(prices)

    left_high  = max(prices[:15])
    cup_bottom = min(prices[15:n-15])
    right_high = max(prices[n-15:])

    # Profondeur de la coupe (15-35%)
    depth = (left_high - cup_bottom) / left_high * 100

    # Les deux côtés de la coupe doivent être similaires
    symmetry = abs(left_high - right_high) / left_high * 100

    # Prix actuel dans la zone de l'anse (légère retraite après sommet droit)
    current = prices[-1]
    handle_retrace = (right_high - current) / right_high * 100

    if 15 <= depth <= 35 and symmetry < 8 and 2 <= handle_retrace <= 12:
        return "🟢 Cup & Handle (anse en cours)", 22
    elif 10 <= depth <= 40 and symmetry < 15:
        return "🟡 Cup & Handle (formation)", 11
    return None, 0


def detect_breakout_52w(close, high):
    """
    Breakout 52 semaines :
    - Prix casse au-dessus du plus haut des 52 dernières semaines
    - Signal très fort en swing trading
    """
    if len(close) < 252:
        return None, 0

    high_52w = high.iloc[-252:].max()
    current  = close.iloc[-1]
    prev     = close.iloc[-2]

    # Breakout récent (dans les 5 derniers jours)
    recent_high = high.iloc[-252:-5].max()

    if current > high_52w * 0.995 and prev <= recent_high:
        return "🟢 Breakout 52 semaines (rupture)", 25
    elif current >= high_52w * 0.97:
        return "🟡 Proche du sommet 52 semaines", 12
    return None, 0


def detect_ascending_triangle(close, high, lookback=40):
    """
    Ascending Triangle :
    - Résistance horizontale (hauts similaires)
    - Série de creux ascendants (lows montants)
    - Compression vers le point de rupture
    """
    if len(close) < lookback:
        return None, 0

    highs  = high.iloc[-lookback:].values
    lows   = close.iloc[-lookback:].values
    n      = len(highs)

    # Résistance : les hauts sont stables (±2%)
    recent_highs = highs[n//2:]
    high_std = np.std(recent_highs) / np.mean(recent_highs) * 100

    # Support ascendant : régression linéaire des bas
    x = np.arange(n)
    low_slope, _ = np.polyfit(x, lows, 1)

    # Prix proche de la résistance = compression
    resistance = np.mean(recent_highs)
    current = close.iloc[-1]
    proximity = (current / resistance) * 100

    if high_std < 2 and low_slope > 0 and proximity > 96:
        return "🟢 Ascending Triangle (rupture imminente)", 20
    elif high_std < 3 and low_slope > 0:
        return "🟡 Ascending Triangle (en formation)", 10
    return None, 0


# ─────────────────────────────────────────────
# 🧠 FONCTION PRINCIPALE — APPELER DANS APP.PY
# ─────────────────────────────────────────────

def detect_all_patterns(hist):
    """
    Entrée  : hist = yf.Ticker(ticker).history(period="1y")
    Sortie  : dict avec patterns détectés + score bonus + résumé
    
    Usage dans app.py :
        hist = yf.Ticker(ticker).history(period="1y")
        patterns = detect_all_patterns(hist)
        
        # Ajouter au DataFrame :
        row["Patterns"]      = patterns["summary"]
        row["Pattern_Score"] = patterns["bonus_score"]
        row["Top_Pattern"]   = patterns["top_pattern"]
    """
    if hist is None or hist.empty or len(hist) < 60:
        return {"patterns": [], "bonus_score": 0, "summary": "—", "top_pattern": "—"}

    close  = hist["Close"]
    high   = hist["High"]
    volume = hist["Volume"]

    ma50  = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])

    detected = []

    # Lancer toutes les détections
    checkers = [
        detect_golden_death_cross(close, ma50, ma200),
        detect_bull_flag(close, volume),
        detect_double_bottom(close),
        detect_cup_and_handle(close),
        detect_breakout_52w(close, high),
        detect_ascending_triangle(close, high),
    ]

    for name, score in checkers:
        if name is not None and score > 0:
            detected.append({"pattern": name, "score": score})

    if not detected:
        return {"patterns": [], "bonus_score": 0, "summary": "Aucun pattern", "top_pattern": "—"}

    # Trier par score décroissant
    detected = sorted(detected, key=lambda x: x["score"], reverse=True)

    # Score bonus total (plafonné à 40 pour ne pas écraser le score principal)
    bonus = min(sum(d["score"] for d in detected), 40)

    # Résumé textuel
    summary = " | ".join(d["pattern"] for d in detected)
    top     = detected[0]["pattern"]

    return {
        "patterns":     detected,
        "bonus_score":  bonus,
        "summary":      summary,
        "top_pattern":  top,
    }


# ─────────────────────────────────────────────
# 📋 INTERPRÉTATION DU SCORE PATTERN
# ─────────────────────────────────────────────

def pattern_badge(bonus_score):
    """Retourne un label lisible selon le score pattern total."""
    if bonus_score >= 30:
        return "🔥 Patterns forts"
    elif bonus_score >= 20:
        return "⚡ Patterns actifs"
    elif bonus_score >= 10:
        return "👀 Pattern en formation"
    elif bonus_score > 0:
        return "📊 Signal faible"
    return "—"
