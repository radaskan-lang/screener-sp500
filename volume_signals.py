import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 📊 VOLUME ANORMAL — DÉTECTION AVANCÉE
# Spike · Climax · Accumulation · Distribution · Quiet Buildup
# from volume_signals import detect_volume_anomaly
# ─────────────────────────────────────────────

def detect_volume_anomaly(hist, lookback_avg=20):
    """
    Détecte les anomalies de volume sur un historique yfinance.

    Signaux détectés :
    - SPIKE        : volume jour > 3x moyenne 20j (événement rare)
    - CLIMAX       : plus gros volume des 52 dernières semaines
    - ACCUMULATION : 3+ jours consécutifs haussiers avec volume fort
    - DISTRIBUTION : 3+ jours consécutifs baissiers avec volume fort
    - QUIET_BUILDUP: volume faible qui s'accélère progressivement
    - DRYING_UP    : volume qui se comprime avant un mouvement

    Retourne un dict avec :
    - signals      : liste des signaux détectés
    - top_signal   : signal le plus fort
    - score        : bonus pts (0-25)
    - badge        : label lisible
    - vol_ratio    : ratio volume actuel / moyenne 20j
    - vol_52w_rank : rang du volume actuel sur 52 semaines (0-100%)
    - is_bullish   : True si volume haussier dominant
    - summary      : résumé textuel
    """
    if hist is None or hist.empty or len(hist) < 30:
        return _empty_vol()

    try:
        close  = hist["Close"]
        volume = hist["Volume"]
        high   = hist["High"]
        low    = hist["Low"]

        # ── Métriques de base ──
        avg_vol_20  = float(volume.rolling(lookback_avg).mean().iloc[-1])
        last_vol    = float(volume.iloc[-1])
        vol_ratio   = round(last_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 1.0

        # Rang percentile sur 52 semaines
        vol_252 = volume.iloc[-252:] if len(volume) >= 252 else volume
        vol_52w_rank = round(
            (vol_252 < last_vol).sum() / len(vol_252) * 100, 1
        )

        # Prix up/down aujourd'hui
        price_up = float(close.iloc[-1]) > float(close.iloc[-2])

        signals  = []
        bullish_pts  = 0
        bearish_pts  = 0

        # ── 1. VOLUME SPIKE ──
        # Volume > 3x la moyenne → événement inhabituel
        if vol_ratio >= 4.0 and price_up:
            signals.append({
                "name":    "🚀 VOLUME SPIKE HAUSSIER",
                "detail":  f"Volume {vol_ratio}x la moyenne — achat institutionnel probable",
                "score":   25,
                "bullish": True,
            })
            bullish_pts += 25
        elif vol_ratio >= 3.0 and price_up:
            signals.append({
                "name":    "⚡ Volume spike haussier",
                "detail":  f"Volume {vol_ratio}x la moyenne",
                "score":   18,
                "bullish": True,
            })
            bullish_pts += 18
        elif vol_ratio >= 4.0 and not price_up:
            signals.append({
                "name":    "🔴 VOLUME SPIKE BAISSIER",
                "detail":  f"Volume {vol_ratio}x — pression vendeuse forte",
                "score":   0,
                "bullish": False,
            })
            bearish_pts += 20

        # ── 2. VOLUME CLIMAX ──
        # Plus gros volume des 52 semaines
        if vol_52w_rank >= 98 and price_up:
            signals.append({
                "name":    "🏆 CLIMAX HAUSSIER (top 2% volume annuel)",
                "detail":  f"Volume dans le top 2% annuel — momentum exceptionnel",
                "score":   22,
                "bullish": True,
            })
            bullish_pts += 22
        elif vol_52w_rank >= 95 and price_up:
            signals.append({
                "name":    "📈 Volume exceptionnel (top 5% annuel)",
                "detail":  f"Rang {vol_52w_rank}% sur 52 semaines",
                "score":   15,
                "bullish": True,
            })
            bullish_pts += 15
        elif vol_52w_rank >= 98 and not price_up:
            signals.append({
                "name":    "🔴 CLIMAX BAISSIER (top 2% volume annuel)",
                "detail":  f"Volume exceptionnel sur baisse — capitulation possible",
                "score":   5,  # peut être rebond
                "bullish": False,
            })

        # ── 3. ACCUMULATION ──
        # 3+ jours consécutifs : hausse prix + volume > moyenne
        recent_5 = hist.iloc[-6:-1]  # 5 derniers jours (hors aujourd'hui)
        if len(recent_5) >= 3:
            consec_acc = 0
            for i in range(len(recent_5) - 1, -1, -1):
                day_up  = float(recent_5["Close"].iloc[i]) > float(recent_5["Close"].iloc[i-1]) if i > 0 else True
                day_vol = float(recent_5["Volume"].iloc[i])
                if day_up and day_vol > avg_vol_20 * 1.1:
                    consec_acc += 1
                else:
                    break

            if consec_acc >= 4:
                signals.append({
                    "name":    f"✅ ACCUMULATION FORTE ({consec_acc} jours)",
                    "detail":  f"{consec_acc} jours consécutifs haussiers avec volume élevé",
                    "score":   20,
                    "bullish": True,
                })
                bullish_pts += 20
            elif consec_acc >= 3:
                signals.append({
                    "name":    f"✅ Accumulation ({consec_acc} jours)",
                    "detail":  f"{consec_acc} jours de hausse avec volume",
                    "score":   13,
                    "bullish": True,
                })
                bullish_pts += 13

        # ── 4. DISTRIBUTION ──
        # 3+ jours consécutifs : baisse prix + volume > moyenne
        if len(recent_5) >= 3:
            consec_dist = 0
            for i in range(len(recent_5) - 1, -1, -1):
                day_down = float(recent_5["Close"].iloc[i]) < float(recent_5["Close"].iloc[i-1]) if i > 0 else True
                day_vol  = float(recent_5["Volume"].iloc[i])
                if day_down and day_vol > avg_vol_20 * 1.1:
                    consec_dist += 1
                else:
                    break

            if consec_dist >= 3:
                signals.append({
                    "name":    f"🔴 Distribution ({consec_dist} jours)",
                    "detail":  f"{consec_dist} jours de baisse avec volume fort — pression vendeuse",
                    "score":   0,
                    "bullish": False,
                })
                bearish_pts += 15

        # ── 5. QUIET BUILDUP ──
        # Volume faible qui s'accélère sur 10 jours = compression avant explosion
        if len(volume) >= 15:
            vol_10d_first = float(volume.iloc[-15:-10].mean())
            vol_10d_last  = float(volume.iloc[-5:].mean())
            buildup_ratio = vol_10d_last / vol_10d_first if vol_10d_first > 0 else 1.0

            if 0.5 < vol_10d_first / avg_vol_20 < 0.85 and buildup_ratio > 1.4 and price_up:
                signals.append({
                    "name":    "⚡ QUIET BUILDUP (volume qui s'accélère)",
                    "detail":  f"Volume calme puis accélération ×{round(buildup_ratio,1)} — setup de breakout",
                    "score":   17,
                    "bullish": True,
                })
                bullish_pts += 17

        # ── 6. DRYING UP ──
        # Volume qui se comprime = énergie accumulée (souvent avant TTM Squeeze)
        if len(volume) >= 10:
            vol_recent_5  = float(volume.iloc[-5:].mean())
            vol_prev_10   = float(volume.iloc[-15:-5].mean())
            drying_ratio  = vol_recent_5 / vol_prev_10 if vol_prev_10 > 0 else 1.0

            if drying_ratio < 0.6 and vol_recent_5 < avg_vol_20 * 0.7:
                signals.append({
                    "name":    "🔄 Volume Drying Up (compression)",
                    "detail":  f"Volume réduit à {round(drying_ratio*100)}% — énergie qui s'accumule",
                    "score":   10,
                    "bullish": True,  # neutre mais positif pour le swing
                })
                bullish_pts += 10

        # ── Score final ──
        # Favoriser les signaux haussiers, pénaliser les baissiers
        if bearish_pts > bullish_pts:
            final_score = 0
            is_bullish  = False
        else:
            final_score = min(bullish_pts, 25)
            is_bullish  = True

        # Top signal
        bullish_signals = [s for s in signals if s["bullish"]]
        top_signal = bullish_signals[0]["name"] if bullish_signals else (
            signals[0]["name"] if signals else None
        )

        # Badge
        if final_score >= 20:
            badge = "🚀 Volume exceptionnel"
        elif final_score >= 15:
            badge = "⚡ Volume fort"
        elif final_score >= 10:
            badge = "📊 Volume notable"
        elif final_score > 0:
            badge = "~ Volume légèrement élevé"
        elif bearish_pts > 0:
            badge = "🔴 Volume baissier"
        else:
            badge = "—"

        # Résumé
        names   = [s["name"] for s in signals]
        summary = " | ".join(names) if names else "Volume normal"

        return {
            "signals":     signals,
            "top_signal":  top_signal,
            "score":       final_score,
            "badge":       badge,
            "vol_ratio":   vol_ratio,
            "vol_52w_rank":vol_52w_rank,
            "is_bullish":  is_bullish,
            "bullish_pts": bullish_pts,
            "bearish_pts": bearish_pts,
            "summary":     summary,
        }

    except Exception:
        return _empty_vol()


def _empty_vol():
    return {
        "signals":      [],
        "top_signal":   None,
        "score":        0,
        "badge":        "—",
        "vol_ratio":    1.0,
        "vol_52w_rank": 50.0,
        "is_bullish":   True,
        "bullish_pts":  0,
        "bearish_pts":  0,
        "summary":      "—",
    }
