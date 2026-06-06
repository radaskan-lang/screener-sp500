import yfinance as yf
import pandas as pd
import concurrent.futures


# ─────────────────────────────────────────────
# ⚡ PRÉ-FILTRE RAPIDE — S&P 500 COMPLET
# Passe 1 : élimine les actions sans intérêt en 2-3s chacune
# Passe 2 : analyse complète seulement sur les survivantes
# ─────────────────────────────────────────────

# Critères du pré-filtre (modifiables)
PREFILTER_CONFIG = {
    "min_price":        10,      # Prix minimum ($)
    "max_price":        2000,    # Prix maximum ($)
    "min_volume":       500_000, # Volume moyen 20j minimum
    "min_momentum_20d": -2,      # Momentum 20 jours minimum (%)
    "require_above_ma50": True,  # Prix doit être au-dessus de MA50
}


def prefilter_ticker(ticker):
    """
    Analyse rapide d'un ticker — seulement prix, volume, MA50, momentum.
    Retourne (ticker, True/False, raison_exclusion).
    """
    try:
        hist = yf.Ticker(ticker).history(period="3mo")

        if hist is None or hist.empty or len(hist) < 25:
            return ticker, False, "Données insuffisantes"

        close  = hist["Close"]
        volume = hist["Volume"]

        price      = float(close.iloc[-1])
        ma50_short = float(close.rolling(min(50, len(close))).mean().iloc[-1])
        avg_vol    = float(volume.rolling(20).mean().iloc[-1])
        momentum   = float((price - close.iloc[-21]) / close.iloc[-21] * 100) if len(close) > 21 else 0

        cfg = PREFILTER_CONFIG

        # Critère 1 : prix dans la fourchette
        if not (cfg["min_price"] <= price <= cfg["max_price"]):
            return ticker, False, f"Prix hors fourchette (${round(price,2)})"

        # Critère 2 : volume suffisant
        if avg_vol < cfg["min_volume"]:
            return ticker, False, f"Volume faible ({int(avg_vol):,})"

        # Critère 3 : momentum positif
        if momentum < cfg["min_momentum_20d"]:
            return ticker, False, f"Momentum négatif ({round(momentum,1)}%)"

        # Critère 4 : au-dessus de la MA50
        if cfg["require_above_ma50"] and price < ma50_short:
            return ticker, False, f"Sous MA50 (${round(ma50_short,2)})"

        return ticker, True, f"OK — ${round(price,2)} | Vol {int(avg_vol):,} | Mom {round(momentum,1)}%"

    except Exception as e:
        return ticker, False, f"Erreur: {str(e)[:40]}"


def run_prefilter(tickers, max_workers=20, progress_callback=None):
    """
    Lance le pré-filtre en parallèle sur tous les tickers.

    Paramètres :
    - tickers          : liste complète des tickers à tester
    - max_workers      : threads parallèles (20 recommandé — requêtes légères)
    - progress_callback: fonction(done, total) appelée à chaque ticker traité

    Retourne un dict :
    - passed    : liste des tickers qui passent le filtre
    - rejected  : liste des tickers rejetés
    - details   : dict {ticker: raison} pour tous les tickers
    - pass_rate : % de tickers retenus
    """
    passed   = []
    rejected = []
    details  = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(prefilter_ticker, t): t for t in tickers}
        done = 0

        for future in concurrent.futures.as_completed(futures):
            done += 1
            ticker, ok, reason = future.result()
            details[ticker] = reason

            if ok:
                passed.append(ticker)
            else:
                rejected.append(ticker)

            if progress_callback:
                progress_callback(done, len(tickers))

    pass_rate = round(len(passed) / len(tickers) * 100, 1) if tickers else 0

    return {
        "passed":    passed,
        "rejected":  rejected,
        "details":   details,
        "pass_rate": pass_rate,
        "total":     len(tickers),
        "n_passed":  len(passed),
        "n_rejected":len(rejected),
    }


def prefilter_summary(result):
    """
    Retourne un résumé lisible du pré-filtre.
    """
    return (
        f"⚡ Pré-filtre terminé : "
        f"**{result['n_passed']}** actions retenues sur **{result['total']}** "
        f"({result['pass_rate']}%) — "
        f"{result['n_rejected']} éliminées"
    )
