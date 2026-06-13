# ─────────────────────────────────────────────
# SIMULATIONS — A/B Testing des parametres
# 4 simulations paralleles avec parametres differents
# ─────────────────────────────────────────────

SIMULATION_CONFIGS = {
    "Sim1": {
        "name":        "Sim1 - Baseline",
        "description": "Parametres standards du systeme",
        "score_min":   70,
        "conv_min":    4,
        "rr_min":      2.0,
        "adx_min":     20,
        "fibonacci":   True,
        "strict":      False,
        "color":       "#4a90d0",
    },
    "Sim2": {
        "name":        "Sim2 - Momentum Fort",
        "description": "ADX eleve, forte tendance confirmee",
        "score_min":   80,
        "conv_min":    5,
        "rr_min":      2.0,
        "adx_min":     25,
        "fibonacci":   True,
        "strict":      False,
        "color":       "#00ff88",
    },
    "Sim3": {
        "name":        "Sim3 - Filtres Larges",
        "description": "Filet large, sans Fibonacci, ADX flexible",
        "score_min":   65,
        "conv_min":    3,
        "rr_min":      1.5,
        "adx_min":     15,
        "fibonacci":   False,
        "strict":      False,
        "color":       "#fbbf24",
    },
    "Sim4": {
        "name":        "Sim4 - Ultra Strict",
        "description": "Seulement le meilleur du meilleur",
        "score_min":   85,
        "conv_min":    6,
        "rr_min":      2.5,
        "adx_min":     30,
        "fibonacci":   True,
        "strict":      True,
        "color":       "#f87171",
    },
}


def filter_for_simulation(df, sim_key):
    """Filtre le DataFrame selon les parametres de la simulation."""
    if df is None or df.empty:
        return df

    config = SIMULATION_CONFIGS.get(sim_key, SIMULATION_CONFIGS["Sim1"])
    result = df.copy()

    ai_col = "AI Score Ajuste" if "AI Score Ajuste" in result.columns else "AI Score"

    # Score minimum
    result = result[result[ai_col] >= config["score_min"]]

    # Convergence minimum
    if "Conv_N" in result.columns:
        result = result[result["Conv_N"] >= config["conv_min"]]

    # R/R minimum
    if "RR_Ratio" in result.columns:
        result = result[result["RR_Ratio"] >= config["rr_min"]]

    # ADX minimum
    if "ADX" in result.columns and config["adx_min"] > 0:
        result = result[result["ADX"] >= config["adx_min"]]

    # Fibonacci validation
    if not config["fibonacci"] and "FIB_EntryValid" in result.columns:
        pass  # Ignore Fibonacci filter
    elif config["fibonacci"] and "FIB_EntryValid" in result.columns:
        result = result[result["FIB_EntryValid"] != False]

    # Top 10
    result = result.head(10).reset_index(drop=True)

    return result


def get_simulation_summary(trades, sim_key):
    """Calcule les stats pour une simulation donnee."""
    sim_trades = [t for t in trades if t.get("simulation") == sim_key]
    closed     = [t for t in sim_trades if t.get("status") == "CLOSED"]
    open_t     = [t for t in sim_trades if t.get("status") == "OPEN"]

    if not closed:
        return {
            "sim_key":   sim_key,
            "name":      SIMULATION_CONFIGS.get(sim_key, {}).get("name", sim_key),
            "n_closed":  0,
            "n_open":    len(open_t),
            "wins":      0,
            "losses":    0,
            "win_rate":  0,
            "avg_pnl":   0,
            "total_pnl": 0,
            "best":      0,
            "worst":     0,
        }

    wins   = [t for t in closed if t.get("result") == "WIN"]
    pnls   = [float(t.get("pnl_pct", 0) or 0) for t in closed]

    return {
        "sim_key":   sim_key,
        "name":      SIMULATION_CONFIGS.get(sim_key, {}).get("name", sim_key),
        "color":     SIMULATION_CONFIGS.get(sim_key, {}).get("color", "#4a90d0"),
        "n_closed":  len(closed),
        "n_open":    len(open_t),
        "wins":      len(wins),
        "losses":    len(closed) - len(wins),
        "win_rate":  round(len(wins) / len(closed) * 100, 1),
        "avg_pnl":   round(sum(pnls) / len(pnls), 2),
        "total_pnl": round(sum(pnls), 1),
        "best":      round(max(pnls), 2),
        "worst":     round(min(pnls), 2),
    }
