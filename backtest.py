import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures


# ─────────────────────────────────────────────
# 📊 BACKTEST ENGINE — SWING TRADING
# Simule lundi entrée → vendredi sortie
# sur 52 semaines de données historiques
# from backtest import run_backtest, backtest_summary
# ─────────────────────────────────────────────

def simulate_week_trade(hist_week, entry_price, stop_price, target_price):
    """
    Simule un trade sur une semaine donnée.
    
    Logique :
    - Lundi  : entrée au prix d'ouverture (ou entry_price si gap)
    - Mardi-Jeudi : vérification stop / target intraday
    - Vendredi : sortie à la clôture si ni stop ni target atteint
    
    Retourne :
    - result   : 'WIN' / 'LOSS' / 'BREAKEVEN'
    - exit_price : prix de sortie réel
    - exit_day   : jour de sortie (0=Lundi ... 4=Vendredi)
    - pnl_pct    : gain/perte en %
    - hit_target : bool
    - hit_stop   : bool
    """
    if hist_week is None or hist_week.empty or len(hist_week) < 3:
        return None

    opens  = hist_week["Open"].values
    highs  = hist_week["High"].values
    lows   = hist_week["Low"].values
    closes = hist_week["Close"].values

    # Entrée lundi — prix réel d'ouverture
    actual_entry = float(opens[0])

    # Ajuster stop et target proportionnellement si gap
    if entry_price and entry_price > 0:
        ratio = actual_entry / entry_price
        actual_stop   = stop_price * ratio if stop_price else actual_entry * 0.97
        actual_target = target_price * ratio if target_price else actual_entry * 1.06
    else:
        actual_stop   = stop_price or actual_entry * 0.97
        actual_target = target_price or actual_entry * 1.06

    exit_price = float(closes[-1])  # défaut : clôture vendredi
    exit_day   = len(hist_week) - 1
    hit_target = False
    hit_stop   = False

    # Simulation jour par jour
    for day_idx in range(len(hist_week)):
        day_high  = float(highs[day_idx])
        day_low   = float(lows[day_idx])
        day_close = float(closes[day_idx])

        # Target atteint
        if day_high >= actual_target:
            exit_price = actual_target
            exit_day   = day_idx
            hit_target = True
            break

        # Stop touché
        if day_low <= actual_stop:
            exit_price = actual_stop
            exit_day   = day_idx
            hit_stop   = True
            break

    pnl_pct = round((exit_price - actual_entry) / actual_entry * 100, 2)

    if hit_target:
        result = "WIN"
    elif hit_stop:
        result = "LOSS"
    elif pnl_pct > 0.5:
        result = "WIN"
    elif pnl_pct < -0.5:
        result = "LOSS"
    else:
        result = "BREAKEVEN"

    return {
        "result":      result,
        "entry_actual":round(actual_entry, 2),
        "exit_price":  round(exit_price, 2),
        "exit_day":    exit_day,
        "pnl_pct":     pnl_pct,
        "hit_target":  hit_target,
        "hit_stop":    hit_stop,
    }


def backtest_ticker(ticker, score_fn, weeks=52):
    """
    Backtest complet sur un ticker sur N semaines.
    
    Pour chaque semaine :
    1. Calcule les signaux sur les données précédentes (look-ahead free)
    2. Simule le trade sur la semaine suivante
    3. Enregistre le résultat
    
    Retourne une liste de trades avec leurs résultats.
    """
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="2y")  # 2 ans pour avoir 1 an de backtest

        if hist is None or hist.empty or len(hist) < 100:
            return []

        trades = []
        
        # Grouper par semaine
        hist.index = pd.to_datetime(hist.index)
        hist["week"] = hist.index.to_period("W")
        weekly_groups = list(hist.groupby("week"))

        # On a besoin d'au moins 60 jours de données avant pour calculer les indicateurs
        min_lookback = 60

        for i in range(min_lookback // 5, len(weekly_groups) - 1):
            # Données jusqu'à la fin de la semaine i (pour calculer signaux)
            week_end_idx = weekly_groups[i][1].index[-1]
            hist_so_far  = hist[hist.index <= week_end_idx]

            if len(hist_so_far) < min_lookback:
                continue

            # Semaine suivante pour simuler le trade
            next_week_data = weekly_groups[i + 1][1]
            if next_week_data.empty:
                continue

            # Calcul simplifié du score (sans appel API pour la vitesse)
            try:
                close     = hist_so_far["Close"]
                volume    = hist_so_far["Volume"]
                price     = float(close.iloc[-1])
                ma50      = float(close.rolling(50).mean().iloc[-1])
                ma200     = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else ma50
                
                # RSI
                delta = close.diff()
                gain  = delta.where(delta > 0, 0).rolling(14).mean()
                loss  = -delta.where(delta < 0, 0).rolling(14).mean()
                rs    = gain / loss.clip(lower=1e-10)
                rsi   = float(100 - (100 / (1 + rs.iloc[-1])))

                # MACD
                ema12     = close.ewm(span=12, adjust=False).mean()
                ema26     = close.ewm(span=26, adjust=False).mean()
                macd_hist = float((ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()).iloc[-1])

                # Volume
                avg_vol   = float(volume.rolling(20).mean().iloc[-1])
                last_vol  = float(volume.iloc[-1])
                vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

                # ATR pour stop/target
                high_s = hist_so_far["High"]
                low_s  = hist_so_far["Low"]
                tr = pd.concat([
                    high_s - low_s,
                    (high_s - close.shift(1)).abs(),
                    (low_s  - close.shift(1)).abs()
                ], axis=1).max(axis=1)
                atr = float(tr.rolling(14).mean().iloc[-1])

                # Score simplifié (mêmes critères que ai_score)
                score = 0
                if price > ma50 > ma200:   score += 35
                elif price > ma200:        score += 15
                if 45 <= rsi <= 65:        score += 25
                elif 35 <= rsi < 45:       score += 18
                elif 65 < rsi <= 72:       score += 15
                elif rsi > 72:             score += 5
                if macd_hist > 0.3:        score += 20
                elif macd_hist > 0:        score += 14
                if vol_ratio >= 2.0:       score += 20
                elif vol_ratio >= 1.5:     score += 15
                elif vol_ratio >= 1.1:     score += 10

                # Nombre de signaux convergents
                n_signals = sum([
                    price > ma50 > ma200,
                    45 <= rsi <= 65 or (35 <= rsi < 45),
                    macd_hist > 0,
                    vol_ratio >= 1.5,
                ])

                # Prix d'entrée / stop / target
                entry  = round(price * 1.003, 2)
                stop   = round(entry - atr * 1.5, 2)
                target = round(entry + atr * 3.0, 2)

                # Simuler le trade sur la semaine suivante
                trade_result = simulate_week_trade(next_week_data, entry, stop, target)

                if trade_result:
                    week_label = str(weekly_groups[i][0])
                    trades.append({
                        "ticker":     ticker,
                        "week":       week_label,
                        "score":      score,
                        "n_signals":  n_signals,
                        "rsi":        round(rsi, 1),
                        "macd_hist":  round(macd_hist, 3),
                        "vol_ratio":  round(vol_ratio, 2),
                        "entry":      entry,
                        "stop":       stop,
                        "target":     target,
                        "rr_ratio":   round((target - entry) / (entry - stop), 2) if entry > stop else 0,
                        **trade_result,
                    })

            except Exception:
                continue

        return trades

    except Exception:
        return []


def run_backtest(tickers, weeks=52, max_workers=8, progress_callback=None):
    """
    Lance le backtest sur une liste de tickers.
    
    Retourne un DataFrame avec tous les trades simulés.
    """
    all_trades = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(backtest_ticker, t, None, weeks): t for t in tickers}
        done = 0

        for future in concurrent.futures.as_completed(futures):
            done += 1
            trades = future.result()
            if trades:
                all_trades.extend(trades)
            if progress_callback:
                progress_callback(done, len(tickers))

    if not all_trades:
        return pd.DataFrame()

    return pd.DataFrame(all_trades)


def backtest_summary(df_trades):
    """
    Calcule les statistiques complètes du backtest.
    
    Retourne un dict avec toutes les métriques clés.
    """
    if df_trades.empty:
        return {}

    total  = len(df_trades)
    wins   = len(df_trades[df_trades["result"] == "WIN"])
    losses = len(df_trades[df_trades["result"] == "LOSS"])
    be     = len(df_trades[df_trades["result"] == "BREAKEVEN"])

    win_rate    = round(wins / total * 100, 1) if total > 0 else 0
    avg_win     = round(df_trades[df_trades["result"]=="WIN"]["pnl_pct"].mean(), 2) if wins > 0 else 0
    avg_loss    = round(df_trades[df_trades["result"]=="LOSS"]["pnl_pct"].mean(), 2) if losses > 0 else 0
    avg_pnl     = round(df_trades["pnl_pct"].mean(), 2)
    total_pnl   = round(df_trades["pnl_pct"].sum(), 1)
    best_trade  = round(df_trades["pnl_pct"].max(), 2)
    worst_trade = round(df_trades["pnl_pct"].min(), 2)

    # Profit factor
    gross_profit = df_trades[df_trades["pnl_pct"] > 0]["pnl_pct"].sum()
    gross_loss   = abs(df_trades[df_trades["pnl_pct"] < 0]["pnl_pct"].sum())
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.0

    # Expectancy
    expectancy = round((win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss), 2)

    # Max drawdown consécutif
    results_list = df_trades.sort_values("week")["result"].tolist()
    max_consec_loss = 0
    curr_loss = 0
    for r in results_list:
        if r == "LOSS":
            curr_loss += 1
            max_consec_loss = max(max_consec_loss, curr_loss)
        else:
            curr_loss = 0

    # Stats par niveau de score
    score_stats = {}
    for score_min, score_max, label in [
        (80, 101, "Score ≥ 80"),
        (60, 80,  "Score 60-79"),
        (0,  60,  "Score < 60"),
    ]:
        subset = df_trades[(df_trades["score"] >= score_min) & (df_trades["score"] < score_max)]
        if len(subset) > 0:
            s_wins = len(subset[subset["result"]=="WIN"])
            score_stats[label] = {
                "n":        len(subset),
                "win_rate": round(s_wins / len(subset) * 100, 1),
                "avg_pnl":  round(subset["pnl_pct"].mean(), 2),
            }

    # Stats par nombre de signaux convergents
    signal_stats = {}
    for n in range(2, 5):
        subset = df_trades[df_trades["n_signals"] == n]
        if len(subset) > 0:
            s_wins = len(subset[subset["result"]=="WIN"])
            signal_stats[f"{n}/4 signaux"] = {
                "n":        len(subset),
                "win_rate": round(s_wins / len(subset) * 100, 1),
                "avg_pnl":  round(subset["pnl_pct"].mean(), 2),
            }
    subset_4 = df_trades[df_trades["n_signals"] >= 4]
    if len(subset_4) > 0:
        s_wins = len(subset_4[subset_4["result"]=="WIN"])
        signal_stats["4/4 signaux"] = {
            "n":        len(subset_4),
            "win_rate": round(s_wins / len(subset_4) * 100, 1),
            "avg_pnl":  round(subset_4["pnl_pct"].mean(), 2),
        }

    return {
        "total":           total,
        "wins":            wins,
        "losses":          losses,
        "breakeven":       be,
        "win_rate":        win_rate,
        "avg_win":         avg_win,
        "avg_loss":        avg_loss,
        "avg_pnl":         avg_pnl,
        "total_pnl":       total_pnl,
        "best_trade":      best_trade,
        "worst_trade":     worst_trade,
        "profit_factor":   profit_factor,
        "expectancy":      expectancy,
        "max_consec_loss": max_consec_loss,
        "score_stats":     score_stats,
        "signal_stats":    signal_stats,
    }
