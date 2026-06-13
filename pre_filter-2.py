import yfinance as yf
import pandas as pd
import concurrent.futures
import math


PREFILTER_CONFIG = {
    "min_price":          10,
    "max_price":          2000,
    "min_volume":         500_000,
    "min_momentum_20d":   -5,
    "require_above_ma50": False,
}


def prefilter_ticker(ticker):
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
    passed  = []
    failed  = []
    reasons = {}
    done    = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(prefilter_ticker, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            done += 1
            try:
                ticker, ok, reason = future.result(timeout=20)
                if ok:
                    passed.append(ticker)
                else:
                    failed.append(ticker)
                reasons[ticker] = reason
            except Exception:
                t = futures[future]
                failed.append(t)
                reasons[t] = "Timeout"

            if progress_callback and done % 25 == 0:
                pct = done / len(tickers)
                progress_callback(pct, f"Passe 1: {len(passed)} retenus / {done} analyses")

    if progress_callback:
        progress_callback(1.0, f"Passe 1 terminee: {len(passed)} retenus")

    return {
        "passed":   passed,
        "failed":   failed,
        "reasons":  reasons,
        "n_total":  len(tickers),
        "n_passed": len(passed),
        "n_failed": len(failed),
        "pass_rate": round(len(passed) / len(tickers) * 100, 1) if tickers else 0,
    }
