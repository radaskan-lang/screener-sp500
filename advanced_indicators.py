import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 🔬 INDICATEURS TECHNIQUES AVANCÉS
# TTM Squeeze · Divergence RSI · EMA Alignment
# from advanced_indicators import detect_advanced_signals
# ─────────────────────────────────────────────

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def detect_ttm_squeeze(hist, lookback=20):
    """
    TTM Squeeze (Lazybear) :
    Compare Bollinger Bands vs Keltner Channels.

    - SQUEEZE ON  (🔴) : BB à l'intérieur des KC → énergie accumulée
    - SQUEEZE OFF (🟢) : BB sort des KC → explosion imminente ou en cours
    - Momentum histogram : direction probable du mouvement

    Retourne :
    - status     : "FIRE" / "ARMED" / "NEUTRAL"
    - squeeze_on : bool — compression active
    - momentum   : float — direction et force
    - bars_armed : int — nombre de barres en compression
    - signal     : texte lisible
    - score      : bonus pts
    """
    try:
        close  = hist["Close"]
        high   = hist["High"]
        low    = hist["Low"]
        n      = len(close)

        if n < lookback + 10:
            return _empty_ttm()

        # ── Bollinger Bands ──
        bb_mid = close.rolling(lookback).mean()
        bb_std = close.rolling(lookback).std()
        bb_up  = bb_mid + 1.5 * bb_std
        bb_low = bb_mid - 1.5 * bb_std

        # ── Keltner Channels ──
        kc_mid = calc_ema(close, lookback)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr     = tr.rolling(lookback).mean()
        kc_mult = 1.5
        kc_up   = kc_mid + kc_mult * atr
        kc_low  = kc_mid - kc_mult * atr

        # ── Détection squeeze ──
        squeeze = (bb_up < kc_up) & (bb_low > kc_low)

        # ── Momentum (linreg delta) ──
        mid_val  = (high.rolling(lookback).max() + low.rolling(lookback).min()) / 2
        mid_val2 = (mid_val + bb_mid) / 2
        delta    = close - mid_val2
        x        = np.arange(lookback)

        momentum_vals = []
        for i in range(lookback, n):
            y = delta.iloc[i - lookback:i].values
            if len(y) == lookback:
                m, _ = np.polyfit(x, y, 1)
                momentum_vals.append(m)
            else:
                momentum_vals.append(0)

        momentum = float(momentum_vals[-1]) if momentum_vals else 0

        # Compter les barres consécutives en squeeze
        bars_armed = 0
        for i in range(len(squeeze) - 1, -1, -1):
            if squeeze.iloc[i]:
                bars_armed += 1
            else:
                break

        # Squeeze vient de se libérer (2 dernières barres)
        just_fired = (not squeeze.iloc[-1]) and squeeze.iloc[-2] if len(squeeze) >= 2 else False

        # ── Scoring ──
        if just_fired and momentum > 0:
            status = "FIRE"
            signal = "🔥 TTM Squeeze déclenché (haussier)"
            score  = 20
        elif just_fired and momentum < 0:
            status = "FIRE_BEAR"
            signal = "🔥 TTM Squeeze déclenché (baissier)"
            score  = 0
        elif bars_armed >= 3 and momentum > 0:
            status = "ARMED"
            signal = f"⚡ TTM Squeeze armé ({bars_armed} barres) — haussier"
            score  = 14
        elif bars_armed >= 3:
            status = "ARMED_NEUTRAL"
            signal = f"⚡ TTM Squeeze armé ({bars_armed} barres)"
            score  = 8
        elif squeeze.iloc[-1]:
            status = "SQUEEZE"
            signal = "🔴 TTM Squeeze actif"
            score  = 5
        else:
            status = "NEUTRAL"
            signal = None
            score  = 0

        return {
            "status":     status,
            "squeeze_on": bool(squeeze.iloc[-1]),
            "momentum":   round(momentum, 4),
            "bars_armed": bars_armed,
            "just_fired": just_fired,
            "signal":     signal,
            "score":      score,
        }

    except Exception:
        return _empty_ttm()


def detect_rsi_divergence(hist, rsi_period=14, lookback=30):
    """
    Divergence RSI Haussière :
    Prix fait un nouveau bas MAIS RSI fait un bas plus haut
    → Signal de retournement à la hausse

    Divergence RSI Baissière :
    Prix fait un nouveau haut MAIS RSI fait un haut plus bas
    → Signal d'épuisement

    Retourne :
    - type   : "BULLISH" / "BEARISH" / None
    - signal : texte lisible
    - score  : bonus pts
    - strength : "forte" / "modérée"
    """
    try:
        close = hist["Close"]
        if len(close) < lookback + rsi_period + 5:
            return _empty_div()

        # Calcul RSI
        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss  = -delta.where(delta < 0, 0).rolling(rsi_period).mean()
        rs    = gain / loss.clip(lower=1e-10)
        rsi   = 100 - (100 / (1 + rs))

        recent_close = close.iloc[-lookback:]
        recent_rsi   = rsi.iloc[-lookback:]

        # Trouver les creux locaux sur le prix
        price_lows = []
        for i in range(2, len(recent_close) - 2):
            if (recent_close.iloc[i] < recent_close.iloc[i-1] and
                recent_close.iloc[i] < recent_close.iloc[i+1] and
                recent_close.iloc[i] < recent_close.iloc[i-2] and
                recent_close.iloc[i] < recent_close.iloc[i+2]):
                price_lows.append((i, float(recent_close.iloc[i]), float(recent_rsi.iloc[i])))

        # Trouver les sommets locaux sur le prix
        price_highs = []
        for i in range(2, len(recent_close) - 2):
            if (recent_close.iloc[i] > recent_close.iloc[i-1] and
                recent_close.iloc[i] > recent_close.iloc[i+1] and
                recent_close.iloc[i] > recent_close.iloc[i-2] and
                recent_close.iloc[i] > recent_close.iloc[i+2]):
                price_highs.append((i, float(recent_close.iloc[i]), float(recent_rsi.iloc[i])))

        # ── Divergence haussière ──
        if len(price_lows) >= 2:
            idx1, p1, r1 = price_lows[-2]
            idx2, p2, r2 = price_lows[-1]

            price_lower = p2 < p1           # Prix fait un plus bas
            rsi_higher  = r2 > r1 + 2      # RSI fait un plus haut (marge 2pts)
            rsi_oversold = r2 < 50          # RSI encore en zone basse

            if price_lower and rsi_higher and rsi_oversold:
                diff = round(r2 - r1, 1)
                strength = "forte" if diff > 8 else "modérée"
                return {
                    "type":     "BULLISH",
                    "signal":   f"🟢 Divergence RSI haussière ({strength}) +{diff}pts",
                    "score":    18 if strength == "forte" else 12,
                    "strength": strength,
                    "rsi_low1": round(r1, 1),
                    "rsi_low2": round(r2, 1),
                }

        # ── Divergence baissière ──
        if len(price_highs) >= 2:
            idx1, p1, r1 = price_highs[-2]
            idx2, p2, r2 = price_highs[-1]

            price_higher = p2 > p1
            rsi_lower    = r2 < r1 - 2
            rsi_overbought = r2 > 55

            if price_higher and rsi_lower and rsi_overbought:
                diff = round(r1 - r2, 1)
                return {
                    "type":     "BEARISH",
                    "signal":   f"🔴 Divergence RSI baissière -{diff}pts",
                    "score":    0,
                    "strength": "modérée",
                    "rsi_low1": round(r1, 1),
                    "rsi_low2": round(r2, 1),
                }

        return _empty_div()

    except Exception:
        return _empty_div()


def detect_ema_alignment(hist):
    """
    Alignement EMA 8 / 21 / MA50 / MA200

    Niveaux :
    - PARFAIT   : EMA8 > EMA21 > MA50 > MA200 + prix au-dessus de tout → +15 pts
    - FORT      : EMA8 > EMA21 > MA50 → +10 pts
    - MODERE    : EMA8 > EMA21 → +6 pts
    - NEUTRE    : alignement partiel
    - BAISSIER  : ordre inverse

    Retourne également :
    - ema8_slope : pente EMA8 (momentum court terme)
    - fan_opening : expansion du fan (les EMAs s'écartent = accélération)
    """
    try:
        close = hist["Close"]
        if len(close) < 200:
            return _empty_ema()

        price  = float(close.iloc[-1])
        ema8   = float(calc_ema(close, 8).iloc[-1])
        ema21  = float(calc_ema(close, 21).iloc[-1])
        ma50   = float(close.rolling(50).mean().iloc[-1])
        ma200  = float(close.rolling(200).mean().iloc[-1])

        # Pente EMA8 (momentum 5 jours)
        ema8_series = calc_ema(close, 8)
        ema8_slope  = float((ema8_series.iloc[-1] - ema8_series.iloc[-6]) / ema8_series.iloc[-6] * 100)

        # Fan opening (EMA8 s'écarte de EMA21)
        ema21_series  = calc_ema(close, 21)
        fan_now       = (ema8_series.iloc[-1] - ema21_series.iloc[-1]) / ema21_series.iloc[-1] * 100
        fan_5d        = (ema8_series.iloc[-6] - ema21_series.iloc[-6]) / ema21_series.iloc[-6] * 100
        fan_opening   = float(fan_now - fan_5d)

        # Distances en %
        dist_ema8_21  = round((ema8 - ema21) / ema21 * 100, 2)
        dist_ema21_50 = round((ema21 - ma50) / ma50 * 100, 2)
        dist_ma50_200 = round((ma50 - ma200) / ma200 * 100, 2)

        # Conditions
        p_above_all   = price > ema8 > ema21 > ma50 > ma200
        ema8_above_21 = ema8 > ema21
        ema21_above_50= ema21 > ma50
        ma50_above_200= ma50 > ma200
        slope_positive= ema8_slope > 0

        # Compter les conditions alignées
        conditions = [ema8_above_21, ema21_above_50, ma50_above_200, slope_positive]
        n_aligned  = sum(conditions)

        if p_above_all and slope_positive and fan_opening > 0:
            level  = "PARFAIT"
            signal = f"⭐ EMA Alignement parfait (8>21>50>200) pente +{round(ema8_slope,2)}%"
            score  = 15
        elif ema8_above_21 and ema21_above_50 and ma50_above_200:
            level  = "FORT"
            signal = f"✅ EMA Alignement fort (8>21>50>200)"
            score  = 10
        elif ema8_above_21 and ema21_above_50:
            level  = "MODERE"
            signal = f"🟡 EMA Alignement modéré (8>21>50)"
            score  = 6
        elif ema8_above_21:
            level  = "PARTIEL"
            signal = f"~ EMA Partiel (8>21)"
            score  = 3
        elif not ema8_above_21 and not ema21_above_50:
            level  = "BAISSIER"
            signal = None
            score  = 0
        else:
            level  = "NEUTRE"
            signal = None
            score  = 0

        return {
            "level":        level,
            "signal":       signal,
            "score":        score,
            "ema8":         round(ema8, 2),
            "ema21":        round(ema21, 2),
            "ma50":         round(ma50, 2),
            "ma200":        round(ma200, 2),
            "ema8_slope":   round(ema8_slope, 2),
            "fan_opening":  round(fan_opening, 3),
            "dist_8_21":    dist_ema8_21,
            "dist_21_50":   dist_ema21_50,
            "dist_50_200":  dist_ma50_200,
            "n_aligned":    n_aligned,
        }

    except Exception:
        return _empty_ema()


# ─────────────────────────────────────────────
# 🧠 FONCTION PRINCIPALE
# ─────────────────────────────────────────────

def detect_advanced_signals(hist):
    """
    Lance les 3 détections avancées sur un historique yfinance.

    Usage dans fetch() :
        adv = detect_advanced_signals(hist)
        row["TTM_Signal"]   = adv["ttm"]["signal"]
        row["TTM_Score"]    = adv["ttm"]["score"]
        row["DIV_Signal"]   = adv["div"]["signal"]
        row["DIV_Score"]    = adv["div"]["score"]
        row["EMA_Signal"]   = adv["ema"]["signal"]
        row["EMA_Score"]    = adv["ema"]["score"]
        row["ADV_Score"]    = adv["total_score"]
        row["ADV_Badge"]    = adv["badge"]
        row["ADV_Summary"]  = adv["summary"]
    """
    if hist is None or hist.empty or len(hist) < 60:
        return _empty_advanced()

    ttm = detect_ttm_squeeze(hist)
    div = detect_rsi_divergence(hist)
    ema = detect_ema_alignment(hist)

    # Score total avancé
    total = ttm["score"] + div["score"] + ema["score"]

    # Bonus combinaison (2+ signaux forts)
    n_active = sum([
        ttm["score"] >= 10,
        div["score"] >= 10,
        ema["score"] >= 10,
    ])
    if n_active >= 2:
        total += 5

    total = min(total, 43)  # Plafonné à 43 pts bonus max

    # Badge global
    if total >= 30:
        badge = "🚀 Signaux avancés excellents"
    elif total >= 20:
        badge = "⚡ Signaux avancés forts"
    elif total >= 10:
        badge = "🟡 Signaux avancés modérés"
    elif total > 0:
        badge = "📊 Signaux avancés faibles"
    else:
        badge = "—"

    # Résumé
    signals = [s for s in [ttm["signal"], div["signal"], ema["signal"]] if s]
    summary = " | ".join(signals) if signals else "Aucun signal avancé"

    return {
        "ttm":         ttm,
        "div":         div,
        "ema":         ema,
        "total_score": total,
        "badge":       badge,
        "summary":     summary,
        "n_active":    n_active,
    }


# ─────────────────────────────────────────────
# 🔧 HELPERS
# ─────────────────────────────────────────────

def _empty_ttm():
    return {"status":"NEUTRAL","squeeze_on":False,"momentum":0,
            "bars_armed":0,"just_fired":False,"signal":None,"score":0}

def _empty_div():
    return {"type":None,"signal":None,"score":0,"strength":None,
            "rsi_low1":None,"rsi_low2":None}

def _empty_ema():
    return {"level":"NEUTRE","signal":None,"score":0,"ema8":None,
            "ema21":None,"ma50":None,"ma200":None,"ema8_slope":0,
            "fan_opening":0,"dist_8_21":0,"dist_21_50":0,"dist_50_200":0,"n_aligned":0}

def _empty_advanced():
    return {
        "ttm": _empty_ttm(), "div": _empty_div(), "ema": _empty_ema(),
        "total_score": 0, "badge": "—", "summary": "—", "n_active": 0,
    }
