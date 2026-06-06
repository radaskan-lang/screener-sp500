import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 🎯 CONVERGENCE — RAPPORT DU DIMANCHE SOIR
# Combine tous les signaux pour Top 10/20
# from convergence import calc_convergence, build_trade_report
# ─────────────────────────────────────────────

# Les 6 signaux vérifiés pour la convergence
SIGNAL_CHECKS = [
    "trend",    # MA50 / MA200 alignées
    "rsi",      # RSI zone idéale swing
    "macd",     # MACD histogramme haussier
    "volume",   # Volume ratio fort
    "pattern",  # Pattern technique détecté
    "advanced", # TTM / Divergence / EMA
]


def calc_convergence(row):
    """
    Calcule le score de convergence d'une action.
    Vérifie chaque signal indépendamment et retourne :
    - n_signals    : nombre de signaux actifs (0-6)
    - signals_on   : liste des signaux actifs
    - signals_off  : liste des signaux inactifs
    - conv_score   : score de convergence pondéré (0-100)
    - conv_label   : label lisible
    - conv_color   : couleur associée
    - bar          : barre de progression visuelle
    """
    signals_on  = []
    signals_off = []
    details     = {}

    try:
        price     = float(row.get("Prix", 0) or 0)
        ma50      = float(row.get("MA50", 0) or 0)
        ma200     = float(row.get("MA200", 0) or 0)
        rsi_val   = float(row.get("RSI", 50) or 50)
        macd_hist = float(row.get("MACD_Hist", 0) or 0)
        vol_ratio = float(row.get("Vol_Ratio", 1) or 1)
        pattern_score = int(row.get("Pattern_Score", 0) or 0)
        adv_score     = int(row.get("ADV_Score", 0) or 0)

        # ── 1. Trend ──
        if price > ma50 > ma200:
            signals_on.append("✅ Trend haussière forte (prix>MA50>MA200)")
            details["trend"] = {"ok": True, "weight": 2, "detail": f"${round(price,2)} > MA50 ${round(ma50,2)} > MA200 ${round(ma200,2)}"}
        elif price > ma50 and price > ma200:
            signals_on.append("✅ Trend haussière (prix>MA50 & MA200)")
            details["trend"] = {"ok": True, "weight": 1, "detail": f"Prix au-dessus des deux MAs"}
        else:
            signals_off.append("❌ Trend — Sous MA50 ou MA200")
            details["trend"] = {"ok": False, "weight": 0, "detail": f"Prix ${round(price,2)} | MA50 ${round(ma50,2)} | MA200 ${round(ma200,2)}"}

        # ── 2. RSI ──
        if 45 <= rsi_val <= 65:
            signals_on.append(f"✅ RSI idéal swing ({rsi_val})")
            details["rsi"] = {"ok": True, "weight": 2, "detail": f"RSI {rsi_val} — zone parfaite 45-65"}
        elif 35 <= rsi_val < 45:
            signals_on.append(f"✅ RSI zone rebond ({rsi_val})")
            details["rsi"] = {"ok": True, "weight": 1, "detail": f"RSI {rsi_val} — potentiel rebond"}
        elif 65 < rsi_val <= 72:
            signals_on.append(f"~ RSI momentum ({rsi_val})")
            details["rsi"] = {"ok": True, "weight": 1, "detail": f"RSI {rsi_val} — momentum fort mais surveiller"}
        else:
            signals_off.append(f"❌ RSI hors zone ({rsi_val})")
            details["rsi"] = {"ok": False, "weight": 0, "detail": f"RSI {rsi_val} — {'surachat >72' if rsi_val > 72 else 'survente <35'}"}

        # ── 3. MACD ──
        if macd_hist > 0.3:
            signals_on.append(f"✅ MACD fort haussier ({round(macd_hist,3)})")
            details["macd"] = {"ok": True, "weight": 2, "detail": f"Histogramme {round(macd_hist,3)} — momentum accéléré"}
        elif macd_hist > 0:
            signals_on.append(f"✅ MACD haussier ({round(macd_hist,3)})")
            details["macd"] = {"ok": True, "weight": 1, "detail": f"Histogramme positif {round(macd_hist,3)}"}
        else:
            signals_off.append(f"❌ MACD baissier ({round(macd_hist,3)})")
            details["macd"] = {"ok": False, "weight": 0, "detail": f"Histogramme négatif {round(macd_hist,3)}"}

        # ── 4. Volume ──
        if vol_ratio >= 1.5:
            signals_on.append(f"✅ Volume fort ({vol_ratio}x moyenne)")
            details["volume"] = {"ok": True, "weight": 2, "detail": f"Volume {vol_ratio}x la moyenne 20j"}
        elif vol_ratio >= 1.1:
            signals_on.append(f"~ Volume correct ({vol_ratio}x)")
            details["volume"] = {"ok": True, "weight": 1, "detail": f"Volume légèrement au-dessus {vol_ratio}x"}
        else:
            signals_off.append(f"❌ Volume faible ({vol_ratio}x)")
            details["volume"] = {"ok": False, "weight": 0, "detail": f"Volume insuffisant {vol_ratio}x"}

        # ── 5. Pattern ──
        top_pattern = str(row.get("Top_Pattern", "") or "")
        if pattern_score >= 15:
            signals_on.append(f"✅ Pattern fort: {top_pattern}")
            details["pattern"] = {"ok": True, "weight": 2, "detail": top_pattern}
        elif pattern_score > 0:
            signals_on.append(f"~ Pattern: {top_pattern}")
            details["pattern"] = {"ok": True, "weight": 1, "detail": top_pattern}
        else:
            signals_off.append("❌ Aucun pattern détecté")
            details["pattern"] = {"ok": False, "weight": 0, "detail": "—"}

        # ── 6. Indicateurs Avancés ──
        adv_badge = str(row.get("ADV_Badge", "") or "")
        if adv_score >= 20:
            signals_on.append(f"✅ Avancés forts: {adv_badge}")
            details["advanced"] = {"ok": True, "weight": 2, "detail": str(row.get("ADV_Summary", "") or "")}
        elif adv_score >= 8:
            signals_on.append(f"~ Avancés: {adv_badge}")
            details["advanced"] = {"ok": True, "weight": 1, "detail": str(row.get("ADV_Summary", "") or "")}
        else:
            signals_off.append("❌ Signaux avancés absents")
            details["advanced"] = {"ok": False, "weight": 0, "detail": "—"}

    except Exception as e:
        return _empty_conv()

    # ── Score de convergence pondéré ──
    n_signals  = len(signals_on)
    total_weight = sum(d["weight"] for d in details.values())
    max_weight   = 12  # max 2 pts × 6 signaux
    conv_score   = round((total_weight / max_weight) * 100)

    # ── Barre visuelle ──
    filled = "█" * n_signals
    empty  = "░" * (6 - n_signals)
    bar    = filled + empty

    # ── Label et couleur ──
    if n_signals == 6:
        conv_label = "🚀 CONVERGENCE PARFAITE"
        conv_color = "#00ff88"
    elif n_signals == 5:
        conv_label = "⭐ TRÈS FORTE"
        conv_color = "#4ade80"
    elif n_signals == 4:
        conv_label = "✅ FORTE"
        conv_color = "#86efac"
    elif n_signals == 3:
        conv_label = "🟡 MODÉRÉE"
        conv_color = "#fbbf24"
    elif n_signals == 2:
        conv_label = "⚠️ FAIBLE"
        conv_color = "#fb923c"
    else:
        conv_label = "❌ INSUFFISANTE"
        conv_color = "#f87171"

    return {
        "n_signals":   n_signals,
        "signals_on":  signals_on,
        "signals_off": signals_off,
        "details":     details,
        "conv_score":  conv_score,
        "conv_label":  conv_label,
        "conv_color":  conv_color,
        "bar":         bar,
    }


def build_trade_report(df, top_n=10, min_signals=3, min_rr=1.5):
    """
    Construit le rapport final du dimanche soir.

    Paramètres :
    - df         : DataFrame avec toutes les colonnes calculées
    - top_n      : nombre de titres à retenir (10 ou 20)
    - min_signals: minimum de signaux convergents requis
    - min_rr     : R/R minimum requis

    Retourne un DataFrame enrichi trié par convergence + score IA.
    """
    df = df.copy()

    # Calcul convergence sur chaque ligne
    conv_data = df.apply(calc_convergence, axis=1)

    df["Conv_N"]      = conv_data.apply(lambda x: x["n_signals"])
    df["Conv_Score"]  = conv_data.apply(lambda x: x["conv_score"])
    df["Conv_Label"]  = conv_data.apply(lambda x: x["conv_label"])
    df["Conv_Color"]  = conv_data.apply(lambda x: x["conv_color"])
    df["Conv_Bar"]    = conv_data.apply(lambda x: x["bar"])
    df["Conv_On"]     = conv_data.apply(lambda x: " | ".join(x["signals_on"]))
    df["Conv_Off"]    = conv_data.apply(lambda x: " | ".join(x["signals_off"]))

    # Score composite : convergence (60%) + score IA (40%)
    ai_col = "AI Score Ajusté" if "AI Score Ajusté" in df.columns else "AI Score"
    df["Score_Final"] = (
        df["Conv_Score"] * 0.60 +
        df[ai_col] * 0.40
    ).round(1)

    # Filtres
    report = df[df["Conv_N"] >= min_signals].copy()

    if "RR_Ratio" in report.columns:
        report = report[
            (report["RR_Ratio"].isna()) |
            (report["RR_Ratio"] >= min_rr)
        ]

    # Tri : convergence d'abord, score IA ensuite
    report = report.sort_values(
        ["Conv_N", "Score_Final"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return report.head(top_n)


def _empty_conv():
    return {
        "n_signals": 0, "signals_on": [], "signals_off": [],
        "details": {}, "conv_score": 0,
        "conv_label": "❌ INSUFFISANTE", "conv_color": "#f87171",
        "bar": "░░░░░░",
    }


def get_day_of_week_advice(market_regime):
    """
    Conseils d'exécution selon le jour et le régime de marché.
    """
    return {
        "Lundi":    "🟢 Entrée à l'ouverture si gap < 1% vs prix d'entrée",
        "Mardi":    "🟢 Confirmation si pas entré lundi — momentum toujours actif ?",
        "Mercredi": "🟡 Mi-semaine — surveiller le stop, ajuster si +3% de gain",
        "Jeudi":    "🟡 Commencer à sécuriser les gains — déplacer stop au breakeven",
        "Vendredi": "🔴 Vente à la clôture — ne pas tenir le weekend",
        "regime":   f"Marché {market_regime} — adapter la taille des positions en conséquence",
    }
