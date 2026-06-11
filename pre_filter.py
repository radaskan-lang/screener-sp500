import yfinance as yf
import pandas as pd
import concurrent.futures


# ─────────────────────────────────────────────
# PRE-FILTRE RAPIDE — S&P 500 COMPLET
# Utilise yf.download() en batch pour eviter le rate limiting
# ─────────────────────────────────────────────

PREFILTER_CONFIG = {
    "min_price":          10,
    "max_price":          2000,
    "min_volume":         500_000,
    "min_momentum_20d":   -5,
    "require_above_ma50": False,
}


def prefilter_ticker(ticker):
    """Analyse rapide individuelle avec curl_cffi."""
    try:
        try:
            from curl_cffi import requests as cr
            session = cr.Session(impersonate="chrome")
            hist = yf.Ticker(ticker, session=session).history(period="3mo")
        except Exception:
            hist = yf.Ticker(ticker).history(period="3mo")

        if hist is None or hist.empty or len(hist) < 10:
            return ticker, False, "Donnees insuffisantes"

        close  = hist["Close"].dropna()
        volume = hist["Volume"].dropna()

        if len(close) < 5:
            return ticker, False, "Donnees insuffisantes"

        price   = float(close.iloc[-1])
        avg_vol = float(volume.rolling(min(20, len(volume))).mean().iloc[-1])

        import math
        if math.isnan(price) or price <= 0:
            return ticker, False, "Prix invalide"
        if math.isnan(avg_vol):
            avg_vol = 0

        cfg = PREFILTER_CONFIG

        if not (cfg["min_price"] <= price <= cfg["max_price"]):
            return ticker, False, f"Prix hors fourchette (${round(price,2)})"

        if avg_vol < cfg["min_volume"]:
            return ticker, False, f"Volume faible ({int(avg_vol):,})"

        if len(close) > 21:
            momentum = float((price - float(close.iloc[-21])) / float(close.iloc[-21]) * 100)
            if momentum < cfg["min_momentum_20d"]:
                return ticker, False, f"Momentum negatif ({round(momentum,1)}%)"
        
        if cfg["require_above_ma50"] and len(close) >= 50:
            ma50 = float(close.rolling(50).mean().iloc[-1])
            if not math.isnan(ma50) and price < ma50:
                return ticker, False, f"Sous MA50 (${round(ma50,2)})"

        return ticker, True, f"OK ${round(price,2)}"

    except Exception as e:
        return ticker, False, f"Erreur: {str(e)[:40]}"


def run_prefilter(tickers, max_workers=20, progress_callback=None):
    """
    Lance le pre-filtre.
    Essaie d'abord un batch yf.download() pour les prix.
    Fallback sur appels individuels si le batch echoue.
    """
    passed  = []
    failed  = []
    reasons = {}

    # Tentative batch pour les prix (beaucoup plus rapide)
    batch_prices = {}
    try:
        import math
        # Telecharger les prix en batch par groupes de 100
        batch_size = 100
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            try:
                from curl_cffi import requests as cr
                session = cr.Session(impersonate="chrome")
                data = yf.download(
                    batch,
                    period="3mo",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    session=session,
                )
            except Exception:
                data = yf.download(
                    batch,
                    period="3mo",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )

            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    close_data = data["Close"] if "Close" in data else None
                    vol_data   = data["Volume"] if "Volume" in data else None
                else:
                    close_data = data[["Close"]] if "Close" in data.columns else None
                    vol_data   = data[["Volume"]] if "Volume" in data.columns else None

                if close_data is not None:
                    for t in batch:
                        try:
                            if t in close_data.columns:
                                prices = close_data[t].dropna()
                                vols   = vol_data[t].dropna() if vol_data is not None and t in vol_data.columns else pd.Series()
                                if len(prices) >= 5:
                                    price   = float(prices.iloc[-1])
                                    avg_vol = float(vols.rolling(min(20,len(vols))).mean().iloc[-1]) if len(vols) > 0 else 0
                                    if not math.isnan(price) and price > 0:
                                        batch_prices[t] = {"price": price, "avg_vol": avg_vol, "close": prices}
                        except Exception:
                            pass

        if progress_callback:
            progress_callback(0.3, f"Batch: {len(batch_prices)}/{len(tickers)} prix recuperes")

    except Exception:
        pass

    # Appliquer les filtres sur les donnees batch
    cfg = PREFILTER_CONFIG
    import math

    batch_checked = set()
    for ticker in tickers:
        if ticker not in batch_prices:
            continue
        batch_checked.add(ticker)
        d = batch_prices[ticker]
        price   = d["price"]
        avg_vol = d["avg_vol"]
        close   = d["close"]

        if not (cfg["min_price"] <= price <= cfg["max_price"]):
            failed.append(ticker)
            reasons[ticker] = f"Prix hors fourchette (${round(price,2)})"
            continue

        if avg_vol < cfg["min_volume"]:
            failed.append(ticker)
            reasons[ticker] = f"Volume faible ({int(avg_vol):,})"
            continue

        if len(close) > 21:
            try:
                momentum = float((price - float(close.iloc[-21])) / float(close.iloc[-21]) * 100)
                if momentum < cfg["min_momentum_20d"]:
                    failed.append(ticker)
                    reasons[ticker] = f"Momentum negatif ({round(momentum,1)}%)"
                    continue
            except Exception:
                pass

        if cfg["require_above_ma50"] and len(close) >= 50:
            try:
                ma50 = float(close.rolling(50).mean().iloc[-1])
                if not math.isnan(ma50) and price < ma50:
                    failed.append(ticker)
                    reasons[ticker] = f"Sous MA50 (${round(ma50,2)})"
                    continue
            except Exception:
                pass

        passed.append(ticker)
        reasons[ticker] = f"OK ${round(price,2)}"

    # Fallback individuel pour les tickers non traites par le batch
    remaining = [t for t in tickers if t not in batch_checked]
    if remaining:
        if progress_callback:
            progress_callback(0.4, f"Fallback individuel: {len(remaining)} tickers...")

        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, 10)) as executor:
            futures = {executor.submit(prefilter_ticker, t): t for t in remaining}
            for future in concurrent.futures.as_completed(futures):
                done += 1
                try:
                    ticker, ok, reason = future.result(timeout=15)
                    if ok:
                        passed.append(ticker)
                    else:
                        failed.append(ticker)
                    reasons[ticker] = reason
                except Exception as e:
                    t = futures[future]
                    failed.append(t)
                    reasons[t] = "Timeout"

                if progress_callback and done % 20 == 0:
                    pct = 0.4 + (done / len(remaining)) * 0.5
                    progress_callback(pct, f"Passe 1: {len(passed)} retenus / {done+len(batch_checked)} analyses")

    if progress_callback:
        progress_callback(1.0, f"Passe 1 terminee: {len(passed)} retenus")

    return {
        "passed":  passed,
        "failed":  failed,
        "reasons": reasons,
        "n_total": len(tickers),
        "n_passed":len(passed),
        "n_failed":len(failed),
        "pass_rate": round(len(passed)/len(tickers)*100, 1) if tickers else 0,
    }
