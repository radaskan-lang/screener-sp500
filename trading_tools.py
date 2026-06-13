import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import yfinance as yf


# ─────────────────────────────────────────────
# PAPER TRADING + JOURNAL + DATA QUALITY
# Stockage GitHub persistant
# ─────────────────────────────────────────────

PAPER_FILE  = "/tmp/screener_paper_trades.json"
JOURNAL_FILE= "/tmp/screener_journal.json"
MAX_PER_SECTOR = 2


def _use_github():
    """Verifie si GitHub storage est disponible."""
    try:
        from github_storage import github_storage_available
        return github_storage_available()
    except Exception:
        return False


def load_paper_trades():
    """Charge les paper trades — GitHub en priorite, /tmp en fallback."""
    if _use_github():
        try:
            from github_storage import load_paper_trades_github
            data = load_paper_trades_github()
            if data is not None:
                return data
        except Exception:
            pass
    try:
        if os.path.exists(PAPER_FILE):
            with open(PAPER_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_paper_trades(trades):
    """Sauvegarde les paper trades — GitHub + /tmp."""
    # Toujours sauvegarder en local pour la session
    try:
        with open(PAPER_FILE, "w") as f:
            json.dump(trades, f, indent=2)
    except Exception:
        pass
    # Sauvegarder sur GitHub si disponible
    if _use_github():
        try:
            from github_storage import save_paper_trades_github
            return save_paper_trades_github(trades)
        except Exception:
            pass
    return True


def load_journal():
    """Charge le journal — GitHub en priorite, /tmp en fallback."""
    if _use_github():
        try:
            from github_storage import load_journal_github
            data = load_journal_github()
            if data is not None:
                return data
        except Exception:
            pass
    try:
        if os.path.exists(JOURNAL_FILE):
            with open(JOURNAL_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_journal(trades):
    """Sauvegarde le journal — GitHub + /tmp."""
    try:
        with open(JOURNAL_FILE, "w") as f:
            json.dump(trades, f, indent=2)
    except Exception:
        pass
    if _use_github():
        try:
            from github_storage import save_journal_github
            return save_journal_github(trades)
        except Exception:
            pass
    return True


# ─────────────────────────────────────────────
# 🔍 FILTRE QUALITÉ DES DONNÉES
# ─────────────────────────────────────────────

def check_data_quality(hist, ticker=""):
    """
    Vérifie que les données yfinance sont propres et exploitables.
    Détecte les spikes aberrants, données manquantes, incohérences OHLCV.

    Retourne :
    - is_valid   : bool
    - issues     : liste des problèmes détectés
    - quality    : "BONNE" / "ACCEPTABLE" / "MAUVAISE"
    """
    issues = []

    if hist is None or hist.empty:
        return False, ["Données vides"], "MAUVAISE"

    if len(hist) < 30:
        return False, ["Historique trop court"], "MAUVAISE"

    try:
        close  = hist["Close"]
        high   = hist["High"]
        low    = hist["Low"]
        volume = hist["Volume"]

        # ── 1. Détection de spikes aberrants ──
        # Variation journalière > 20% = suspect (sauf split)
        daily_returns = close.pct_change().abs()
        spikes = daily_returns[daily_returns > 0.20]
        if len(spikes) > 0:
            max_spike = round(float(spikes.max()) * 100, 1)
            issues.append(f"Spike aberrant détecté: {max_spike}% en 1 jour")

        # Variation sur 5 derniers jours > 25% = données corrompues
        recent_returns = close.iloc[-5:].pct_change().abs()
        recent_spikes  = recent_returns[recent_returns > 0.20]
        if len(recent_spikes) > 0:
            issues.append(f"Spike récent suspect (données possiblement corrompues)")

        # ── 2. Cohérence OHLCV ──
        # High doit être >= Close et Open
        incoherent_high = ((high < close) | (high < hist["Open"])).sum()
        if incoherent_high > 5:
            issues.append(f"Données High incohérentes ({incoherent_high} jours)")

        # Low doit être <= Close et Open
        incoherent_low = ((low > close) | (low > hist["Open"])).sum()
        if incoherent_low > 5:
            issues.append(f"Données Low incohérentes ({incoherent_low} jours)")

        # ── 3. Valeurs nulles ou négatives ──
        null_close = close.isna().sum()
        if null_close > 3:
            issues.append(f"{null_close} valeurs manquantes dans Close")

        zero_price = (close <= 0).sum()
        if zero_price > 0:
            issues.append(f"Prix zéro ou négatif détecté")

        # ── 4. Volume anormalement nul ──
        zero_vol = (volume == 0).sum()
        if zero_vol > 10:
            issues.append(f"Volume nul sur {zero_vol} jours")

        # ── 5. Prix constant (données gelées) ──
        price_std = float(close.rolling(10).std().iloc[-1])
        if price_std < 0.01:
            issues.append("Prix constant — données gelées")

        # ── Verdict ──
        if len(issues) == 0:
            return True, [], "BONNE"
        elif len(issues) <= 1 and "Spike aberrant" not in str(issues):
            return True, issues, "ACCEPTABLE"
        else:
            return False, issues, "MAUVAISE"

    except Exception as e:
        return False, [f"Erreur vérification: {str(e)[:50]}"], "MAUVAISE"


# ─────────────────────────────────────────────
# 💾 SAUVEGARDE / CHARGEMENT RÉSULTATS
# ─────────────────────────────────────────────

def save_scan_results(df, market_status, regime):
    """Sauvegarde les résultats du dernier scan."""
    try:
        data = {
            "timestamp":     datetime.now().isoformat(),
            "regime":        regime,
            "n_actions":     len(df),
            "spy_vs_ma50":   market_status.get("spy_vs_ma50", 0),
            "vix":           market_status.get("vix", None),
            "top20_tickers": df.head(20)["Ticker"].tolist() if "Ticker" in df.columns else [],
            "df_json":       df.head(30).to_json(orient="records", default_handler=str),
        }
        with open(SAVE_FILE, "w") as f:
            json.dump(data, f)
        return True
    except Exception:
        return False


def load_scan_results():
    """Charge les résultats du dernier scan sauvegardé."""
    try:
        if not os.path.exists(SAVE_FILE):
            return None
        with open(SAVE_FILE, "r") as f:
            data = json.load(f)
        df = pd.read_json(data["df_json"], orient="records")
        return {
            "df":          df,
            "timestamp":   data.get("timestamp"),
            "regime":      data.get("regime"),
            "n_actions":   data.get("n_actions"),
            "top20":       data.get("top20_tickers", []),
        }
    except Exception:
        return None


def get_scan_age(timestamp_str):
    """Retourne l'âge du dernier scan en format lisible."""
    try:
        ts   = datetime.fromisoformat(timestamp_str)
        diff = datetime.now() - ts
        h    = int(diff.total_seconds() // 3600)
        m    = int((diff.total_seconds() % 3600) // 60)
        if h >= 24:
            return f"il y a {h//24}j {h%24}h"
        elif h > 0:
            return f"il y a {h}h {m}min"
        else:
            return f"il y a {m} minutes"
    except Exception:
        return "date inconnue"


# ─────────────────────────────────────────────
# 📊 PAPER TRADING
# ─────────────────────────────────────────────

def load_paper_trades():
    try:
        if not os.path.exists(PAPER_FILE):
            return []
        with open(PAPER_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_paper_trades(trades):
    try:
        with open(PAPER_FILE, "w") as f:
            json.dump(trades, f, indent=2)
        return True
    except Exception:
        return False


def close_paper_trade(trade_id, exit_price):
    """Ferme un trade paper avec son prix de sortie."""
    trades = load_paper_trades()
    for t in trades:
        if t.get("id") == trade_id:
            ep  = float(t["entry_price"])
            xp  = float(exit_price)
            pnl = round((xp - ep) / ep * 100, 2)
            t["exit_price"] = round(xp, 2)
            t["exit_date"]  = datetime.now().strftime("%Y-%m-%d")
            t["pnl_pct"]    = pnl
            t["result"]     = "WIN" if pnl >= 0 else "LOSS"
            t["status"]     = "CLOSED"
            break
    save_paper_trades(trades)


def add_paper_trade(ticker, entry_price, stop_price, target_price,
                    conv_n, score, strategy, sector, week_date, simulation="Sim1"):
    """Ajoute un trade fictif."""
    trades = load_paper_trades()
    trade = {
        "id":           f"{ticker}_{week_date}_{simulation}",
        "ticker":       ticker,
        "week":         week_date,
        "simulation":   simulation,
        "entry_price":  round(float(entry_price), 2),
        "stop_price":   round(float(stop_price), 2),
        "target_price": round(float(target_price), 2),
        "conv_n":       int(conv_n),
        "score":        float(score),
        "strategy":     strategy,
        "sector":       sector,
        "status":       "OPEN",
        "exit_price":   None,
        "exit_date":    None,
        "pnl_pct":      None,
        "result":       None,
        "added_at":     datetime.now().isoformat(),
    }
    # Éviter les doublons
    trades = [t for t in trades if t.get("id") != trade["id"]]
    trades.append(trade)
    save_paper_trades(trades)
    return trade


def update_paper_results():
    """
    Met a jour les trades paper ouverts.
    Gere les sorties selon la strategie choisie.
    Stop-loss declenche SEULEMENT pendant les heures de marche (9h30-16h00 EST).
    """
    from datetime import datetime
    import pytz

    trades  = load_paper_trades()
    updated = False

    # Verifier si c'est vendredi apres 15h30 EST
    try:
        est     = pytz.timezone("US/Eastern")
        now_est = datetime.now(est)
        is_friday_close = (now_est.weekday() == 4 and now_est.hour >= 15 and now_est.minute >= 30)
    except Exception:
        # Si pytz pas disponible, verifier heure UTC (15h30 EST = 20h30 UTC)
        now_utc = datetime.utcnow()
        is_friday_close = (now_utc.weekday() == 4 and now_utc.hour >= 20 and now_utc.minute >= 30)

    for trade in trades:
        if trade.get("status") != "OPEN":
            continue
        try:
            ticker   = trade["ticker"]
            entry    = float(trade["entry_price"])
            stop     = float(trade["stop_price"])
            target   = float(trade["target_price"])
            strategy = str(trade.get("strategy", "C")).upper()

            # Calculer le stop et target selon la strategie
            is_trailing = "T" in strategy.upper()
            if "A" in strategy or "+5" in strategy:
                strat_target = round(entry * 1.05, 2)
                strat_stop   = stop
            elif "B" in strategy or "+7" in strategy:
                strat_target = round(entry * 1.07, 2)
                strat_stop   = stop
            elif "D" in strategy or "STOP 3" in strategy or "3%" in strategy:
                strat_target = target
                strat_stop   = round(entry * 0.97, 2)
            elif "E" in strategy or "STOP 5" in strategy or "5%" in strategy:
                strat_target = target
                strat_stop   = round(entry * 0.95, 2)
            elif "F" in strategy:
                strat_target = round(entry * 1.05, 2)
                strat_stop   = round(entry * 0.97, 2)
            elif is_trailing:
                # Trailing stop - calcule dynamiquement bougie par bougie
                strat_target = target
                strat_stop   = stop  # stop initial
            else:
                # Strategie C (vendredi) - stop et target ATR
                strat_target = target
                strat_stop   = stop

            # Donnees intraday
            try:
                from curl_cffi import requests as cr
                session  = cr.Session(impersonate="chrome")
                hist_intra = yf.Ticker(ticker, session=session).history(period="5d", interval="30m")
            except Exception:
                hist_intra = yf.Ticker(ticker).history(period="5d", interval="30m")

            if hist_intra is None or hist_intra.empty:
                hist_day = yf.Ticker(ticker).history(period="2d")
                current_price = float(hist_day["Close"].iloc[-1]) if hist_day is not None and not hist_day.empty else entry
                trade["current_price"] = round(current_price, 2)
                trade["current_pnl"]   = round((current_price - entry) / entry * 100, 2)

                # Vendredi close pour strategie C
                if "C" in strategy and is_friday_close:
                    pnl = round((current_price - entry) / entry * 100, 2)
                    trade["exit_price"] = round(current_price, 2)
                    trade["exit_date"]  = datetime.now().strftime("%Y-%m-%d")
                    trade["pnl_pct"]    = pnl
                    trade["result"]     = "WIN" if pnl >= 0 else "LOSS"
                    trade["status"]     = "CLOSED"
                    trade["exit_note"]  = "Vente vendredi (strategie C)"
                    updated = True
                continue

            # Filtrer heures de marche + apres date d'entree
            hist_intra.index = pd.to_datetime(hist_intra.index)
            entry_date = trade.get("added_at", "")[:10]
            try:
                if entry_date:
                    hist_intra = hist_intra[hist_intra.index.date >= pd.to_datetime(entry_date).date()]
            except Exception:
                pass

            try:
                market_hours = hist_intra.between_time("09:30", "16:00")
            except Exception:
                market_hours = hist_intra

            current_price = float(market_hours["Close"].iloc[-1]) if not market_hours.empty else entry
            trade["current_price"] = round(current_price, 2)
            trade["current_pnl"]   = round((current_price - entry) / entry * 100, 2)

            # Strategie C — vente vendredi
            if "C" in strategy and is_friday_close:
                pnl = round((current_price - entry) / entry * 100, 2)
                trade["exit_price"] = round(current_price, 2)
                trade["exit_date"]  = datetime.now().strftime("%Y-%m-%d")
                trade["pnl_pct"]    = pnl
                trade["result"]     = "WIN" if pnl >= 0 else "LOSS"
                trade["status"]     = "CLOSED"
                trade["exit_note"]  = "Vente vendredi (strategie C)"
                updated = True
                continue

            # Verifier stop et target pendant heures de marche
            hit_stop   = False
            hit_target = False
            exit_price_final = None

            trailing_high = entry  # Prix max atteint depuis l'entree
            for _, candle in market_hours.iterrows():
                open_p = float(candle["Open"])
                low_p  = float(candle["Low"])
                high_p = float(candle["High"])

                # Mettre a jour le trailing stop
                if is_trailing:
                    if high_p > trailing_high:
                        trailing_high = high_p
                    trailing_pct = 0.03  # 3% par defaut
                    if "T5" in strategy.upper(): trailing_pct = 0.05
                    elif "T3" in strategy.upper(): trailing_pct = 0.03
                    strat_stop = round(trailing_high * (1 - trailing_pct), 2)
                    # Ne jamais descendre sous le stop initial
                    strat_stop = max(strat_stop, stop)

                # Gap baissier sous le stop a l'ouverture
                if open_p <= strat_stop and not hit_stop:
                    hit_stop = True
                    exit_price_final = round(open_p, 2)
                    break
                # Stop touche intraday
                if low_p <= strat_stop and not hit_stop:
                    hit_stop = True
                    exit_price_final = round(strat_stop, 2)
                    break
                # Target atteint (sauf trailing qui n'a pas de target fixe)
                if not is_trailing and high_p >= strat_target:
                    hit_target = True
                    exit_price_final = round(strat_target, 2)
                    break

            if hit_stop and not hit_target:
                pnl = round((exit_price_final - entry) / entry * 100, 2)
                trade["exit_price"] = exit_price_final
                trade["exit_date"]  = datetime.now().strftime("%Y-%m-%d")
                trade["pnl_pct"]    = pnl
                trade["result"]     = "LOSS"
                trade["status"]     = "CLOSED"
                trade["exit_note"]  = f"Stop-loss ({strategy})"
                updated = True
            elif hit_target:
                pnl = round((exit_price_final - entry) / entry * 100, 2)
                trade["exit_price"] = exit_price_final
                trade["exit_date"]  = datetime.now().strftime("%Y-%m-%d")
                trade["pnl_pct"]    = pnl
                trade["result"]     = "WIN"
                trade["status"]     = "CLOSED"
                trade["exit_note"]  = f"Target atteint ({strategy})"
                updated = True

        except Exception:
            continue

    if updated:
        save_paper_trades(trades)

    return trades
    """
    Met a jour les trades paper ouverts.
    Stop-loss declenche SEULEMENT pendant les heures de marche (9h30-16h00 EST).
    Si gap baissier a l'ouverture sous le stop → execution au prix d'ouverture.
    """
    trades = load_paper_trades()
    updated = False

    for trade in trades:
        if trade.get("status") != "OPEN":
            continue
        try:
            ticker = trade["ticker"]
            # Donnees intraday 30min pour respecter les heures de marche
            hist_intra = yf.Ticker(ticker).history(period="5d", interval="30m")

            if hist_intra is None or hist_intra.empty:
                # Fallback donnees journalieres
                hist_day = yf.Ticker(ticker).history(period="5d")
                if hist_day is None or hist_day.empty:
                    continue
                current_price = float(hist_day["Close"].iloc[-1])
                trade["current_price"] = round(current_price, 2)
                trade["current_pnl"]   = round((current_price - float(trade["entry_price"])) / float(trade["entry_price"]) * 100, 2)
                continue

            entry  = float(trade["entry_price"])
            stop   = float(trade["stop_price"])
            target = float(trade["target_price"])
            # Date d'entree du trade
            entry_date = trade.get("added_at", trade.get("week", ""))[:10]

            # Filtrer uniquement les bougies APRES l'entree
            hist_intra.index = pd.to_datetime(hist_intra.index)
            try:
                if entry_date:
                    hist_intra = hist_intra[hist_intra.index.date >= pd.to_datetime(entry_date).date()]
            except Exception:
                pass

            # Filtrer heures de marche 9h30-16h00 EST
            try:
                market_hours = hist_intra.between_time("09:30", "16:00")
            except Exception:
                market_hours = hist_intra

            if market_hours.empty:
                # Pas encore de donnees de marche depuis l'entree - trade toujours ouvert
                try:
                    current_price = float(hist_intra["Close"].iloc[-1])
                    trade["current_price"] = round(current_price, 2)
                    trade["current_pnl"]   = round((current_price - entry) / entry * 100, 2)
                except Exception:
                    pass
                continue

            current_price = float(market_hours["Close"].iloc[-1])
            trade["current_price"] = round(current_price, 2)
            trade["current_pnl"]   = round((current_price - entry) / entry * 100, 2)

            # Verifier stop et target pendant heures de marche seulement
            hit_stop   = False
            hit_target = False
            exit_price_final = None

            for _, candle in market_hours.iterrows():
                # Gap baissier a l'ouverture sous le stop
                if float(candle["Open"]) <= stop and not hit_stop:
                    hit_stop = True
                    exit_price_final = round(float(candle["Open"]), 2)
                    break
                # Stop touche intraday
                if float(candle["Low"]) <= stop and not hit_stop:
                    hit_stop = True
                    exit_price_final = round(stop, 2)
                    break
                # Target atteint
                if float(candle["High"]) >= target:
                    hit_target = True
                    exit_price_final = round(target, 2)
                    break

            if hit_stop and not hit_target:
                pnl = round((exit_price_final - entry) / entry * 100, 2)
                trade["exit_price"] = exit_price_final
                trade["exit_date"]  = datetime.now().strftime("%Y-%m-%d")
                trade["pnl_pct"]    = pnl
                trade["result"]     = "LOSS"
                trade["status"]     = "CLOSED"
                trade["exit_note"]  = "Stop-loss marche"
                updated = True

            elif hit_target:
                pnl = round((exit_price_final - entry) / entry * 100, 2)
                trade["exit_price"] = exit_price_final
                trade["exit_date"]  = datetime.now().strftime("%Y-%m-%d")
                trade["pnl_pct"]    = pnl
                trade["result"]     = "WIN"
                trade["status"]     = "CLOSED"
                trade["exit_note"]  = "Target atteint"
                updated = True

        except Exception:
            continue

    if updated:
        save_paper_trades(trades)

    return trades


def get_paper_summary():
    """Calcule les statistiques du paper trading."""
    trades = load_paper_trades()
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("pnl_pct") is not None]

    if not closed:
        return {"n_closed": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0, "wins": 0, "losses": 0}

    wins   = [t for t in closed if t.get("result") == "WIN"]
    losses = [t for t in closed if t.get("result") == "LOSS"]
    pnls   = [t["pnl_pct"] for t in closed]

    return {
        "n_closed":  len(closed),
        "n_open":    len([t for t in trades if t.get("status") == "OPEN"]),
        "wins":      len(wins),
        "losses":    len(losses),
        "win_rate":  round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "avg_pnl":   round(sum(pnls) / len(pnls), 2),
        "total_pnl": round(sum(pnls), 1),
        "best":      round(max(pnls), 2),
        "worst":     round(min(pnls), 2),
        "trades":    trades,
    }


# ─────────────────────────────────────────────
# 📓 JOURNAL DE TRADES RÉELS
# ─────────────────────────────────────────────

def load_journal():
    try:
        if not os.path.exists(JOURNAL_FILE):
            return []
        with open(JOURNAL_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_journal(trades):
    try:
        with open(JOURNAL_FILE, "w") as f:
            json.dump(trades, f, indent=2)
        return True
    except Exception:
        return False


def add_journal_trade(ticker, entry_price, stop_price, target_price,
                      strategy, sector, notes=""):
    """Ajoute un trade réel au journal."""
    trades = load_journal()
    trade = {
        "id":           f"real_{ticker}_{datetime.now().strftime('%Y%m%d')}",
        "ticker":       ticker,
        "date_entry":   datetime.now().strftime("%Y-%m-%d"),
        "week":         datetime.now().strftime("%Y-W%V"),
        "entry_price":  round(float(entry_price), 2),
        "stop_price":   round(float(stop_price), 2),
        "target_price": round(float(target_price), 2),
        "strategy":     strategy,
        "sector":       sector,
        "notes":        notes,
        "status":       "OPEN",
        "exit_price":   None,
        "exit_date":    None,
        "pnl_pct":      None,
        "result":       None,
    }
    trades.append(trade)
    save_journal(trades)
    return trade


def close_journal_trade(trade_id, exit_price, notes=""):
    """Ferme un trade réel avec son prix de sortie."""
    trades = load_journal()
    for t in trades:
        if t.get("id") == trade_id:
            ep  = float(exit_price)
            en  = float(t["entry_price"])
            pnl = round((ep - en) / en * 100, 2)
            t["exit_price"] = round(ep, 2)
            t["exit_date"]  = datetime.now().strftime("%Y-%m-%d")
            t["pnl_pct"]    = pnl
            t["result"]     = "WIN" if pnl >= 0 else "LOSS"
            t["status"]     = "CLOSED"
            if notes:
                t["notes"] += f" | Sortie: {notes}"
            break
    save_journal(trades)


def get_journal_summary():
    """Statistiques du journal de trades réels."""
    trades = load_journal()
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("pnl_pct") is not None]

    if not closed:
        return {"n_closed": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}

    wins  = [t for t in closed if t.get("result") == "WIN"]
    pnls  = [t["pnl_pct"] for t in closed]

    return {
        "n_closed":  len(closed),
        "n_open":    len([t for t in trades if t.get("status") == "OPEN"]),
        "wins":      len(wins),
        "losses":    len(closed) - len(wins),
        "win_rate":  round(len(wins) / len(closed) * 100, 1),
        "avg_pnl":   round(sum(pnls) / len(pnls), 2),
        "total_pnl": round(sum(pnls), 1),
        "trades":    trades,
    }


# ─────────────────────────────────────────────
# 🏢 DIVERSIFICATION SECTORIELLE
# ─────────────────────────────────────────────

def check_sector_diversity(df, top_n=10, max_per_sector=MAX_PER_SECTOR):
    """
    Filtre le Top N pour respecter la limite par secteur.
    Garde les max_per_sector meilleures actions par secteur.

    Retourne le DataFrame filtré avec une colonne 'Sector_Rank'.
    """
    if df.empty or "Sector" not in df.columns:
        return df

    result   = []
    sector_counts = {}

    for _, row in df.iterrows():
        sector = str(row.get("Sector", "Unknown") or "Unknown")
        count  = sector_counts.get(sector, 0)

        if count < max_per_sector:
            row_dict = row.to_dict()
            row_dict["Sector_Rank"] = count + 1
            result.append(row_dict)
            sector_counts[sector] = count + 1

        if len(result) >= top_n:
            break

    if not result:
        return df

    return pd.DataFrame(result)


def get_sector_distribution(df):
    """Retourne la distribution sectorielle du Top N."""
    if "Sector" not in df.columns:
        return {}
    return df["Sector"].value_counts().to_dict()
