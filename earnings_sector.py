import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures


# ─────────────────────────────────────────────
# 📅 EARNINGS FILTER + 💪 SECTOR STRENGTH
# Évite les semaines avec earnings
# Détecte le secteur le plus fort
# from earnings_sector import check_earnings, get_sector_strength
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 📅 EARNINGS FILTER
# ─────────────────────────────────────────────

def check_earnings(ticker):
    """
    Vérifie si un ticker a des earnings prévus dans les 7 prochains jours.

    Retourne un dict :
    - has_earnings   : bool
    - earnings_date  : date ou None
    - days_until     : int ou None
    - risk_level     : "ÉLEVÉ" / "MODÉRÉ" / "FAIBLE" / "AUCUN"
    - badge          : label lisible
    - should_avoid   : bool — True si on doit éviter ce trade
    """
    try:
        t    = yf.Ticker(ticker)
        cal  = t.calendar

        if cal is None or cal.empty:
            return _empty_earnings()

        # Chercher la date d'earnings
        earnings_date = None

        # Format DataFrame (nouvelle API yfinance)
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                raw = cal.loc["Earnings Date"]
                if hasattr(raw, "__iter__"):
                    for v in raw:
                        try:
                            earnings_date = pd.to_datetime(v).date()
                            break
                        except Exception:
                            continue
                else:
                    try:
                        earnings_date = pd.to_datetime(raw).date()
                    except Exception:
                        pass

        # Format dict (ancienne API)
        elif isinstance(cal, dict):
            for key in ["Earnings Date", "earningsDate", "earnings_date"]:
                if key in cal:
                    try:
                        val = cal[key]
                        if isinstance(val, (list, tuple)):
                            val = val[0]
                        earnings_date = pd.to_datetime(val).date()
                        break
                    except Exception:
                        continue

        if earnings_date is None:
            return _empty_earnings()

        today      = datetime.now().date()
        days_until = (earnings_date - today).days

        # Évaluation du risque
        if days_until < 0:
            # Earnings passés récemment — pas de risque immédiat
            if days_until >= -3:
                risk_level   = "MODÉRÉ"
                badge        = f"📊 Earnings il y a {abs(days_until)}j"
                should_avoid = False
            else:
                return _empty_earnings()

        elif days_until <= 2:
            risk_level   = "ÉLEVÉ"
            badge        = f"🔴 Earnings dans {days_until}j — ÉVITER"
            should_avoid = True

        elif days_until <= 5:
            risk_level   = "MODÉRÉ"
            badge        = f"⚠️ Earnings dans {days_until}j"
            should_avoid = True  # Dans la fenêtre swing lundi-vendredi

        elif days_until <= 10:
            risk_level   = "FAIBLE"
            badge        = f"🟡 Earnings dans {days_until}j"
            should_avoid = False

        else:
            risk_level   = "AUCUN"
            badge        = f"✅ Pas d'earnings proches ({days_until}j)"
            should_avoid = False

        return {
            "has_earnings":  True,
            "earnings_date": str(earnings_date),
            "days_until":    days_until,
            "risk_level":    risk_level,
            "badge":         badge,
            "should_avoid":  should_avoid,
        }

    except Exception:
        return _empty_earnings()


def _empty_earnings():
    return {
        "has_earnings":  False,
        "earnings_date": None,
        "days_until":    None,
        "risk_level":    "AUCUN",
        "badge":         "✅ Pas d'earnings",
        "should_avoid":  False,
    }


def check_earnings_batch(tickers, max_workers=15):
    """
    Vérifie les earnings pour une liste de tickers en parallèle.
    Retourne un dict {ticker: earnings_data}.
    """
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_earnings, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception:
                results[ticker] = _empty_earnings()
    return results


# ─────────────────────────────────────────────
# 💪 SECTOR STRENGTH
# ─────────────────────────────────────────────

# ETFs sectoriels S&P 500
SECTOR_ETFS = {
    "Technology":             "XLK",
    "Health Care":            "XLV",
    "Financials":             "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials":            "XLI",
    "Consumer Staples":       "XLP",
    "Energy":                 "XLE",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
    "Materials":              "XLB",
}


def get_sector_strength():
    """
    Calcule la force relative de chaque secteur sur 5 jours et 20 jours.

    Retourne un dict :
    - rankings      : liste triée par force 5j décroissante
    - top_sector    : secteur le plus fort cette semaine
    - worst_sector  : secteur le plus faible
    - sector_scores : dict {secteur: score_force}
    - momentum_5d   : dict {secteur: % sur 5 jours}
    - momentum_20d  : dict {secteur: % sur 20 jours}
    """
    try:
        sector_data = {}

        for sector, etf in SECTOR_ETFS.items():
            try:
                hist = yf.Ticker(etf).history(period="2mo")
                if hist is None or hist.empty or len(hist) < 25:
                    continue

                close      = hist["Close"]
                price      = float(close.iloc[-1])
                ma20       = float(close.rolling(20).mean().iloc[-1])
                mom_5d     = round((price - float(close.iloc[-6])) / float(close.iloc[-6]) * 100, 2) if len(close) > 5 else 0
                mom_20d    = round((price - float(close.iloc[-21])) / float(close.iloc[-21]) * 100, 2) if len(close) > 20 else 0
                above_ma20 = price > ma20

                # Score de force sectorielle
                strength = 0
                if mom_5d > 2:       strength += 30
                elif mom_5d > 0:     strength += 20
                elif mom_5d > -1:    strength += 10
                else:                strength += 0

                if mom_20d > 5:      strength += 30
                elif mom_20d > 2:    strength += 20
                elif mom_20d > 0:    strength += 10
                else:                strength += 0

                if above_ma20:       strength += 20
                if mom_5d > mom_20d: strength += 20  # accélération court terme

                sector_data[sector] = {
                    "etf":        etf,
                    "price":      round(price, 2),
                    "ma20":       round(ma20, 2),
                    "mom_5d":     mom_5d,
                    "mom_20d":    mom_20d,
                    "above_ma20": above_ma20,
                    "strength":   strength,
                }

            except Exception:
                continue

        if not sector_data:
            return _empty_sector()

        # Trier par force 5j
        rankings = sorted(
            sector_data.items(),
            key=lambda x: x[1]["strength"],
            reverse=True
        )

        top_sector   = rankings[0][0] if rankings else "N/A"
        worst_sector = rankings[-1][0] if rankings else "N/A"

        sector_scores  = {s: d["strength"] for s, d in sector_data.items()}
        momentum_5d    = {s: d["mom_5d"]   for s, d in sector_data.items()}
        momentum_20d   = {s: d["mom_20d"]  for s, d in sector_data.items()}

        # Badges pour les top secteurs
        sector_badges = {}
        for i, (sector, data) in enumerate(rankings):
            if i == 0:
                sector_badges[sector] = f"🔥 #{i+1} Secteur le plus fort"
            elif i == 1:
                sector_badges[sector] = f"⚡ #{i+1} Secteur fort"
            elif i <= 3:
                sector_badges[sector] = f"✅ #{i+1} Secteur correct"
            elif i >= len(rankings) - 2:
                sector_badges[sector] = f"🔴 #{i+1} Secteur faible"
            else:
                sector_badges[sector] = f"~ #{i+1} Secteur neutre"

        return {
            "rankings":      rankings,
            "top_sector":    top_sector,
            "worst_sector":  worst_sector,
            "sector_scores": sector_scores,
            "momentum_5d":   momentum_5d,
            "momentum_20d":  momentum_20d,
            "sector_badges": sector_badges,
            "sector_data":   sector_data,
        }

    except Exception:
        return _empty_sector()


def _empty_sector():
    return {
        "rankings":      [],
        "top_sector":    "N/A",
        "worst_sector":  "N/A",
        "sector_scores": {},
        "momentum_5d":   {},
        "momentum_20d":  {},
        "sector_badges": {},
        "sector_data":   {},
    }


def sector_bonus_score(sector, sector_strength_data):
    """
    Retourne un bonus de score basé sur la force du secteur.
    Top 3 secteurs : +10 pts
    Secteurs moyens : 0 pts
    Bottom 2 secteurs : -5 pts
    """
    if not sector or not sector_strength_data:
        return 0, "—"

    rankings = sector_strength_data.get("rankings", [])
    if not rankings:
        return 0, "—"

    sector_names = [s for s, _ in rankings]
    n = len(sector_names)

    try:
        rank = sector_names.index(sector)
    except ValueError:
        return 0, "—"

    if rank == 0:
        return 10, f"🔥 Secteur #1 ({sector})"
    elif rank == 1:
        return 7, f"⚡ Secteur #2 ({sector})"
    elif rank == 2:
        return 4, f"✅ Secteur #3 ({sector})"
    elif rank >= n - 2:
        return -5, f"🔴 Secteur faible #{rank+1} ({sector})"
    else:
        return 0, f"~ Secteur #{rank+1} ({sector})"
