import anthropic
import time


# ─────────────────────────────────────────────
# 🤖 CLAUDE SCORER DYNAMIQUE
# Re-score le Top 30 avec raisonnement IA
# from claude_scorer import claude_score_batch
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un analyste quantitatif senior spécialisé en swing trading sur actions américaines.
Horizon : lundi achat → vendredi vente (5 jours).

Ton rôle : évaluer la qualité d'un setup de swing trading en analysant la CONVERGENCE et la COHÉRENCE des signaux — pas juste leur nombre.

Règles d'évaluation :
- Un RSI élevé (65-72) est ACCEPTABLE si le MACD accélère ET le volume confirme (momentum setup)
- Un volume spike seul sans trend est SUSPECT
- Une distribution volume ANNULE un bon score technique
- Le contexte marché (SPY/QQQ) doit être compatible avec le trade
- Un R/R < 1.5 est éliminatoire même avec de bons signaux
- Les patterns techniques forts (Bull Flag, Cup & Handle confirmé) méritent une prime

Tu réponds UNIQUEMENT en JSON valide, sans markdown, sans texte avant ou après.
Format exact :
{
  "score": 85,
  "verdict": "ACHETER",
  "conviction": "FORTE",
  "raison_principale": "...",
  "risque_principal": "...",
  "ajustement_stop": null
}

Champs :
- score : entier 0-100
- verdict : "ACHETER" | "ATTENDRE" | "EVITER"
- conviction : "FORTE" | "MODEREE" | "FAIBLE"
- raison_principale : 1 phrase max, chiffrée
- risque_principal : 1 phrase max, chiffrée
- ajustement_stop : nouveau stop suggéré en $ si tu veux l'ajuster, sinon null"""


def build_prompt(row, market_status):
    """Construit le prompt pour scorer un titre."""
    regime     = market_status.get("regime", "INCONNU")
    spy_vs_50  = market_status.get("spy_vs_ma50", "N/A")
    qqq_vs_50  = market_status.get("qqq_vs_ma50", "N/A")
    vix        = market_status.get("vix_label", "VIX N/A")

    def safe(key, default="N/A"):
        v = row.get(key)
        return str(round(v, 2)) if isinstance(v, float) else str(v) if v else default

    return f"""CONTEXTE MARCHÉ : {regime} | SPY vs MA50: {spy_vs_50}% | QQQ vs MA50: {qqq_vs_50}% | {vix}

TICKER : {safe('Ticker')} | SECTEUR : {safe('Sector')}

TECHNIQUE :
- Prix: ${safe('Prix')} | MA50: ${safe('MA50')} | MA200: ${safe('MA200')}
- RSI: {safe('RSI')} | MACD Hist: {safe('MACD_Hist')} | Vol Ratio: {safe('Vol_Ratio')}x

SIGNAUX AVANCÉS :
- TTM Squeeze: {safe('TTM_Signal')}
- Divergence RSI: {safe('DIV_Signal')}
- EMA Alignement: {safe('EMA_Signal')}
- Pattern: {safe('Top_Pattern')}

VOLUME ANORMAL :
- Badge: {safe('VOL_Badge')}
- Signal: {safe('VOL_Signal')}
- Ratio: {safe('VOL_Ratio')}x | Rang annuel: {safe('VOL_52W_Rank')}%
- Haussier: {safe('VOL_Bullish')}

PLAN DE TRADE :
- Entrée: ${safe('Entree')} | Stop: ${safe('Stop')} (-{safe('Risque_Pct')}%)
- Target: ${safe('Target')} (+{safe('Gain_Pct')}%) | R/R: {safe('RR_Ratio')}:1
- ATR: {safe('ATR_Pct')}% | Support: ${safe('Support')} | Résistance: ${safe('Resistance')}

CONVERGENCE : {safe('Conv_N')}/6 signaux | Score algo: {safe('AI Score Ajuste')}/100
Signaux actifs: {safe('Conv_On')}
Signaux manquants: {safe('Conv_Off')}

Évalue ce setup de swing trading et retourne le JSON."""


def claude_score_single(row, market_status, client, model="claude-sonnet-4-20250514"):
    """
    Score un seul ticker avec Claude.
    Retourne un dict avec score, verdict, conviction, etc.
    """
    try:
        prompt = build_prompt(row, market_status)

        message = client.messages.create(
            model=model,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text.strip()

        # Nettoyer au cas où Claude ajoute du markdown
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        import json
        data = json.loads(raw)

        return {
            "claude_score":    int(data.get("score", 50)),
            "claude_verdict":  str(data.get("verdict", "ATTENDRE")),
            "claude_conviction": str(data.get("conviction", "FAIBLE")),
            "claude_raison":   str(data.get("raison_principale", "")),
            "claude_risque":   str(data.get("risque_principal", "")),
            "claude_stop_adj": data.get("ajustement_stop", None),
            "claude_ok":       True,
        }

    except Exception as e:
        return {
            "claude_score":    50,
            "claude_verdict":  "ATTENDRE",
            "claude_conviction": "FAIBLE",
            "claude_raison":   f"Analyse indisponible: {str(e)[:50]}",
            "claude_risque":   "—",
            "claude_stop_adj": None,
            "claude_ok":       False,
        }


def claude_score_batch(df, market_status, api_key, top_n=20, delay=0.3, progress_callback=None):
    """
    Score les top_n premières lignes du DataFrame avec Claude.

    Paramètres :
    - df              : DataFrame trié par score décroissant
    - market_status   : dict du marché global
    - api_key         : clé API Anthropic
    - top_n           : nombre de tickers à scorer (max 30 recommandé)
    - delay           : délai entre chaque appel API (éviter rate limit)
    - progress_callback : function(done, total)

    Retourne le DataFrame avec colonnes Claude ajoutées,
    trié par claude_score décroissant.
    """
    if not api_key:
        return df

    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception:
        return df

    df = df.copy()
    subset = df.head(top_n).copy()

    claude_cols = {
        "claude_score":     50,
        "claude_verdict":   "—",
        "claude_conviction":"—",
        "claude_raison":    "—",
        "claude_risque":    "—",
        "claude_stop_adj":  None,
        "claude_ok":        False,
    }

    # Initialiser les colonnes Claude sur tout le DataFrame
    for col, default in claude_cols.items():
        df[col] = default

    done = 0
    for idx, row in subset.iterrows():
        result = claude_score_single(row, market_status, client)

        for col, val in result.items():
            df.at[idx, col] = val

        done += 1
        if progress_callback:
            progress_callback(done, top_n)

        if delay > 0:
            time.sleep(delay)

    # Trier par score Claude pour les lignes scorées, algo pour le reste
    scored   = df[df["claude_ok"] == True].sort_values("claude_score", ascending=False)
    unscored = df[df["claude_ok"] == False]

    return pd.concat([scored, unscored]).reset_index(drop=True)


def verdict_color(verdict):
    """Couleur selon le verdict Claude."""
    return {
        "ACHETER": "#00ff88",
        "ATTENDRE": "#fbbf24",
        "EVITER": "#f87171",
    }.get(verdict, "#64748b")


def conviction_badge(conviction):
    """Badge selon la conviction."""
    return {
        "FORTE":   "🔥 Conviction forte",
        "MODEREE": "⚡ Conviction modérée",
        "FAIBLE":  "~ Conviction faible",
    }.get(conviction, "—")


# Import pandas ici pour éviter les imports circulaires
import pandas as pd
