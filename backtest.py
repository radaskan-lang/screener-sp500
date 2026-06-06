import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures


# ─────────────────────────────────────────────
# 📊 BACKTEST ENGINE — 6 STRATÉGIES DE SORTIE
#
# A : Vente fixe à +5%
# B : Vente fixe à +7%
# C : Vente le vendredi (clôture)
# D : Stop suiveur 3%
# E : Stop suiveur 5%
# F : 50% à +5%, reste stop suiveur 3%
#
# from backtest import run_backtest, backtest_summary
# ─────────────────────────────────────────────

STRATEGIES = {
    "A": "Vente fixe +5%",
    "B": "Vente fixe +7%",
    "C": "Vente vendredi",
    "D": "Stop suiveur 3%",
    "E": "Stop suiveur 5%",
    "F": "50% à +5% + stop suiveur 3%",
}


def simulate_strategy(strategy, opens, highs, lows, closes, actual_entry, actual_stop):
    """
    Simule UNE stratégie de sortie sur une semaine.

    Retourne : (pnl_pct, exit_day, result, detail)
    """
    n = len(closes)

    # ── Stratégie A : Vente fixe à +5% ──
    if strategy == "A":
        target = actual_entry * 1.05
        for day in range(n):
            if lows[day] <= actual_stop:
                pnl = (actual_stop - actual_entry) / actual_entry * 100
                return round(pnl, 2), day, "LOSS", "Stop touché"
            if highs[day] >= target:
                pnl = (target - actual_entry) / actual_entry * 100
                return round(pnl, 2), day, "WIN", "Target +5% atteint"
        # Sortie vendredi
        pnl = (closes[-1] - actual_entry) / actual_entry * 100
        result = "WIN" if pnl > 0.5 else "LOSS" if pnl < -0.5 else "BREAKEVEN"
        return round(pnl, 2), n-1, result, "Clôture vendredi"

    # ── Stratégie B : Vente fixe à +7% ──
    elif strategy == "B":
        target = actual_entry * 1.07
        for day in range(n):
            if lows[day] <= actual_stop:
                pnl = (actual_stop - actual_entry) / actual_entry * 100
                return round(pnl, 2), day, "LOSS", "Stop touché"
            if highs[day] >= target:
                pnl = (target - actual_entry) / actual_entry * 100
                return round(pnl, 2), day, "WIN", "Target +7% atteint"
        pnl = (closes[-1] - actual_entry) / actual_entry * 100
        result = "WIN" if pnl > 0.5 else "LOSS" if pnl < -0.5 else "BREAKEVEN"
        return round(pnl, 2), n-1, result, "Clôture vendredi"

    # ── Stratégie C : Vente vendredi clôture ──
    elif strategy == "C":
        for day in range(n):
            if lows[day] <= actual_stop:
                pnl = (actual_stop - actual_entry) / actual_entry * 100
                return round(pnl, 2), day, "LOSS", "Stop touché"
        pnl = (closes[-1] - actual_entry) / actual_entry * 100
        result = "WIN" if pnl > 0.5 else "LOSS" if pnl < -0.5 else "BREAKEVEN"
        return round(pnl, 2), n-1, result, "Clôture vendredi"

    # ── Stratégie D : Stop suiveur 3% ──
    elif strategy == "D":
        trailing_stop = actual_stop
        highest       = actual_entry
        for day in range(n):
            # Mettre à jour le plus haut et le stop suiveur
            if highs[day] > highest:
                highest       = highs[day]
                new_stop      = highest * (1 - 0.03)
                trailing_stop = max(trailing_stop, new_stop)
            # Vérifier stop
            if lows[day] <= trailing_stop:
                pnl = (trailing_stop - actual_entry) / actual_entry * 100
                result = "WIN" if pnl > 0.5 else "LOSS" if pnl < -0.5 else "BREAKEVEN"
                return round(pnl, 2), day, result, f"Stop suiveur 3% ({round(trailing_stop,2)})"
        pnl = (closes[-1] - actual_entry) / actual_entry * 100
        result = "WIN" if pnl > 0.5 else "LOSS" if pnl < -0.5 else "BREAKEVEN"
        return round(pnl, 2), n-1, result, "Clôture vendredi"

    # ── Stratégie E : Stop suiveur 5% ──
    elif strategy == "E":
        trailing_stop = actual_stop
        highest       = actual_entry
        for day in range(n):
            if highs[day] > highest:
                highest       = highs[day]
                new_stop      = highest * (1 - 0.05)
                trailing_stop = max(trailing_stop, new_stop)
            if lows[day] <= trailing_stop:
                pnl = (trailing_stop - actual_entry) / actual_entry * 100
                result = "WIN" if pnl > 0.5 else "LOSS" if pnl < -0.5 else "BREAKEVEN"
                return round(pnl, 2), day, result, f"Stop suiveur 5% ({round(trailing_stop,2)})"
        pnl = (closes[-1] - actual_entry) / actual_entry * 100
        result = "WIN" if pnl > 0.5 else "LOSS" if pnl < -0.5 else "BREAKEVEN"
        return round(pnl, 2), n-1, result, "Clôture vendredi"

    # ── Stratégie F : 50% à +5%, reste stop suiveur 3% ──
    elif strategy == "F":
        target_half   = actual_entry * 1.05
        half_sold     = False
        trailing_stop = actual_stop
        highest       = actual_entry
        total_pnl     = 0.0

        for day in range(n):
            # Mise à jour stop suiveur (seulement après vente partielle)
            if half_sold and highs[day] > highest:
                highest       = highs[day]
                new_stop      = highest * (1 - 0.03)
                trailing_stop = max(trailing_stop, new_stop)

            # Vente partielle à +5%
            if not half_sold and highs[day] >= target_half:
                half_pnl  = (target_half - actual_entry) / actual_entry * 100
                total_pnl += half_pnl * 0.5
                half_sold  = True
                highest    = target_half
                trailing_stop = max(trailing_stop, target_half * 0.97)

            # Stop touché
            if lows[day] <= trailing_stop:
                remaining_pnl = (trailing_stop - actual_entry) / actual_entry * 100
                weight = 0.5 if half_sold else 1.0
                total_pnl += remaining_pnl * weight
                result = "WIN" if total_pnl > 0.5 else "LOSS" if total_pnl < -0.5 else "BREAKEVEN"
                detail = f"50% vendu +5% + stop suiveur 3% ({round(trailing_stop,2)})"
                return round(total_pnl, 2), day, result, detail

        # Clôture vendredi
        close_pnl  = (closes[-1] - actual_entry) / actual_entry * 100
        weight     = 0.5 if half_sold else 1.0
        total_pnl += close_pnl * weight
        result = "WIN" if total_pnl > 0.5 else "LOSS" if total_pnl < -0.5 else "BREAKEVEN"
        return round(total_pnl, 2), n-1, result, "Clôture vendredi (reste)"

    return 0.0, 0, "BREAKEVEN", "—"


def simulate_all_strategies(hist_week, entry_price, stop_price):
    """
    Simule les 6 stratégies sur la même semaine.
    Retourne un dict {stratégie: résultat}.
    """
    if hist_week is None or hist_week.empty or len(hist_week) < 3:
        return {}

    opens  = hist_week["Open"].values
    highs  = hist_week["High"].values
    lows   = hist_week["Low"].values
    closes = hist_week["Close"].values

    actual_entry = float(opens[0])

    if entry_price and entry_price > 0:
        ratio       = actual_entry / entry_price
        actual_stop = stop_price * ratio if stop_price else actual_entry * 0.97
    else:
        actual_stop = stop_price or actual_entry * 0.97

    results = {}
    for strat in STRATEGIES:
        pnl, exit_day, result, detail = simulate_strategy(
            strat, opens, highs, lows, closes, actual_entry, actual_stop
        )
        results[strat] = {
            "pnl_pct":  pnl,
            "exit_day": exit_day,
            "result":   result,
            "detail":   detail,
        }
    return results


def backtest_ticker(ticker, weeks=52):
    """Backtest complet sur un ticker sur N semaines."""
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="2y")

        if hist is None or hist.empty or len(hist) < 100:
            return []

        trades = []
        hist.index    = pd.to_datetime(hist.index)
        hist["week"]  = hist.index.to_period("W")
        weekly_groups = list(hist.groupby("week"))
        min_lookback  = 60

        for i in range(min_lookback // 5, len(weekly_groups) - 1):
            week_end_idx = weekly_groups[i][1].index[-1]
            hist_so_far  = hist[hist.index <= week_end_idx]

            if len(hist_so_far) < min_lookback:
                continue

            next_week_data = weekly_groups[i + 1][1]
            if next_week_data.empty:
                continue

            try:
                close  = hist_so_far["Close"]
                volume = hist_so_far["Volume"]
                price  = float(close.iloc[-1])
                ma50   = float(close.rolling(50).mean().iloc[-1])
                ma200  = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else ma50

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

                # ATR
                high_s = hist_so_far["High"]
                low_s  = hist_so_far["Low"]
                tr = pd.concat([
                    high_s - low_s,
                    (high_s - close.shift(1)).abs(),
                    (low_s  - close.shift(1)).abs()
                ], axis=1).max(axis=1)
                atr = float(tr.rolling(14).mean().iloc[-1])

                # Score
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

                n_signals = sum([
                    price > ma50 > ma200,
                    45 <= rsi <= 65 or (35 <= rsi < 45),
                    macd_hist > 0,
                    vol_ratio >= 1.5,
                ])

                entry = round(price * 1.003, 2)
                stop  = round(entry - atr * 1.5, 2)

                # Simuler les 6 stratégies
                strat_results = simulate_all_strategies(next_week_data, entry, stop)

                if strat_results:
                    week_label = str(weekly_groups[i][0])
                    base = {
                        "ticker":    ticker,
                        "week":      week_label,
                        "score":     score,
                        "n_signals": n_signals,
                        "rsi":       round(rsi, 1),
                        "macd_hist": round(macd_hist, 3),
                        "vol_ratio": round(vol_ratio, 2),
                        "entry":     entry,
                        "stop":      stop,
                        "atr":       round(atr, 2),
                    }
                    # Ajouter résultats par stratégie
                    for strat, res in strat_results.items():
                        base[f"pnl_{strat}"]    = res["pnl_pct"]
                        base[f"result_{strat}"]  = res["result"]
                        base[f"exit_day_{strat}"]= res["exit_day"]

                    trades.append(base)

            except Exception:
                continue

        return trades

    except Exception:
        return []


def run_backtest(tickers, weeks=52, max_workers=8, progress_callback=None):
    """Lance le backtest sur une liste de tickers."""
    all_trades = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(backtest_ticker, t, weeks): t for t in tickers}
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
    Calcule les statistiques pour chaque stratégie + comparatif.
    """
    if df_trades.empty:
        return {}

    summary = {}

    for strat, label in STRATEGIES.items():
        pnl_col    = f"pnl_{strat}"
        result_col = f"result_{strat}"

        if pnl_col not in df_trades.columns:
            continue

        df_s   = df_trades[[pnl_col, result_col, "score", "n_signals"]].dropna()
        total  = len(df_s)
        if total == 0:
            continue

        wins   = len(df_s[df_s[result_col] == "WIN"])
        losses = len(df_s[df_s[result_col] == "LOSS"])

        win_rate  = round(wins / total * 100, 1)
        avg_win   = round(df_s[df_s[result_col]=="WIN"][pnl_col].mean(), 2) if wins > 0 else 0
        avg_loss  = round(df_s[df_s[result_col]=="LOSS"][pnl_col].mean(), 2) if losses > 0 else 0
        avg_pnl   = round(df_s[pnl_col].mean(), 2)
        total_pnl = round(df_s[pnl_col].sum(), 1)
        best      = round(df_s[pnl_col].max(), 2)
        worst     = round(df_s[pnl_col].min(), 2)

        gross_profit = df_s[df_s[pnl_col] > 0][pnl_col].sum()
        gross_loss   = abs(df_s[df_s[pnl_col] < 0][pnl_col].sum())
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.0
        expectancy   = round((win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss), 2)

        # Max pertes consécutives
        results_list = df_trades.sort_values("week")[result_col].tolist()
        max_consec = 0
        curr = 0
        for r in results_list:
            if r == "LOSS":
                curr += 1
                max_consec = max(max_consec, curr)
            else:
                curr = 0

        # Stats par niveau de score
        score_stats = {}
        for smin, smax, slabel in [(80,101,">=80"),(60,80,"60-79"),(0,60,"<60")]:
            sub = df_s[(df_s["score"]>=smin)&(df_s["score"]<smax)]
            if len(sub) > 0:
                sw = len(sub[sub[result_col]=="WIN"])
                wr_val = round(sw/len(sub)*100, 1)
                ap_val = round(sub[pnl_col].mean(), 2)
                score_stats[slabel] = {
                    "n":        int(len(sub)),
                    "win_rate": float(wr_val),
                    "avg_pnl":  float(ap_val) if ap_val == ap_val else 0.0,
                }

        summary[strat] = {
            "label":          str(label),
            "total":          int(total),
            "wins":           int(wins),
            "losses":         int(losses),
            "win_rate":       float(win_rate),
            "avg_win":        float(avg_win) if avg_win == avg_win else 0.0,
            "avg_loss":       float(avg_loss) if avg_loss == avg_loss else 0.0,
            "avg_pnl":        float(avg_pnl) if avg_pnl == avg_pnl else 0.0,
            "total_pnl":      float(total_pnl),
            "best":           float(best) if best == best else 0.0,
            "worst":          float(worst) if worst == worst else 0.0,
            "profit_factor":  float(profit_factor),
            "expectancy":     float(expectancy),
            "max_consec_loss":int(max_consec),
            "score_stats":    score_stats,
            "pnl_series":     [float(x) for x in df_trades.sort_values("week")[pnl_col].fillna(0).tolist()],
        }

    return summary
