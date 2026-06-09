import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import time
import concurrent.futures
import anthropic
from pattern_detection import detect_all_patterns, pattern_badge
from risk_manager import calc_risk_reward, risk_badge
from market_filter import get_market_status, apply_market_filter, market_advice
from pre_filter import run_prefilter, PREFILTER_CONFIG
from advanced_indicators import detect_advanced_signals
from convergence import calc_convergence, build_trade_report, get_day_of_week_advice
from volume_signals import detect_volume_anomaly
from claude_scorer import claude_score_batch, verdict_color, conviction_badge
from earnings_sector import check_earnings, get_sector_strength, sector_bonus_score
from gap_detector import detect_gaps
from relative_strength import calc_relative_strength, get_spy_data
from support_resistance import calc_sr_levels
from cheat_sheet import _generate_cheat_sheet
from fibonacci_bollinger import calc_fibonacci, detect_bollinger_signals
from intraday_signals import calc_vwap_levels, calc_multitf_signals, calc_intraday_momentum
from trading_tools import (
    check_data_quality, save_scan_results, load_scan_results, get_scan_age,
    add_paper_trade, update_paper_results, get_paper_summary,
    add_journal_trade, close_journal_trade, get_journal_summary,
    check_sector_diversity, get_sector_distribution
)



# ─────────────────────────────
# 📦 EXCEL EXPORT
# ─────────────────────────────
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Screener")
    return output.getvalue()
# ─────────────────────────────
# 🚀 SIDEBAR
# ─────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("### ⚡ Mode de scan")
    use_prefilter = st.checkbox("Pré-filtre S&P 500 complet (503)", value=True)
    if use_prefilter:
        st.markdown("<div style='color:#4a90d0;font-size:0.78rem;'>✅ 503 actions · 2 passes automatiques</div>", unsafe_allow_html=True)
        with st.expander("⚙️ Critères pré-filtre"):
            min_price    = st.number_input("Prix min ($)", value=10, min_value=1)
            max_price    = st.number_input("Prix max ($)", value=2000, min_value=50)
            min_volume   = st.number_input("Volume moyen min", value=500000, step=100000)
            min_momentum = st.slider("Momentum 20j min (%)", -10, 5, -2)
            above_ma50   = st.checkbox("Au-dessus MA50", value=True)
            PREFILTER_CONFIG.update({
                "min_price": min_price, "max_price": max_price,
                "min_volume": min_volume, "min_momentum_20d": min_momentum,
                "require_above_ma50": above_ma50,
            })

    st.markdown("---")
    nb_workers = st.slider("🔀 Threads parallèles", 5, 20, 10)

    st.markdown("---")
    st.markdown("### 🎯 Rapport Top Trades")
    top_n       = st.radio("Nombre de trades", [10, 20], index=0, horizontal=True)

    # Mode strict — améliore le win rate
    strict_mode = st.checkbox("🔒 Mode strict (win rate optimisé)", value=True,
                              help="Force min 4/6 signaux + R/R 2.0 + score 70+")

    if strict_mode:
        min_signals  = 4
        min_rr_conv  = 2.0
        min_score    = 70
        st.markdown("""<div style='background:#00ff8812;border:1px solid #00ff8833;
            border-radius:6px;padding:10px;font-size:0.78rem;color:#86efac;margin-top:6px;'>
            🔒 <strong>Mode strict actif</strong><br>
            ✅ Min 4/6 signaux convergents<br>
            ✅ R/R minimum 2.0:1<br>
            ✅ Score ajusté minimum 70<br>
            ✅ Gap ouverture max 1.5%
        </div>""", unsafe_allow_html=True)
    else:
        min_signals  = st.slider("Signaux convergents min", 2, 6, 3,
                                 help="4+ = recommandé pour win rate optimal")
        min_rr_conv  = st.slider("R/R minimum", 1.0, 3.0, 1.5, step=0.1)
        min_score    = st.slider("Score min", 0, 100, 50)

    # Indicateur visuel win rate estimé
    wr_est = {2:"~40%", 3:"~48%", 4:"~58%", 5:"~67%", 6:"~75%"}
    sig_display = min_signals
    st.markdown(
        f"<div style='color:#fbbf24;font-size:0.8rem;margin-top:4px;'>"
        f"📊 Win rate estimé : <strong>{wr_est.get(sig_display,'—')}</strong> "
        f"avec {sig_display}/6 signaux</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### 🤖 Claude IA")
    api_key = st.text_input("Clé API Anthropic", type="password")

    use_claude_scorer = st.checkbox("🎯 Score dynamique Claude (Top 20)", value=False,
                                     help="Claude re-score les 20 meilleures actions avec raisonnement IA")
    if use_claude_scorer:
        claude_top_n = st.slider("Actions à scorer par Claude", 5, 30, 15)
        st.markdown("""<div style='background:#a78bfa15;border:1px solid #a78bfa33;
            border-radius:6px;padding:8px;font-size:0.78rem;color:#c4b5fd;margin-top:4px;'>
            🤖 Claude analyse chaque setup en profondeur<br>
            et remplace le score algorithmique<br>
            ⏱️ ~{} secondes pour {} actions
        </div>""".format(claude_top_n * 2, claude_top_n), unsafe_allow_html=True)
    else:
        claude_top_n = 15

    use_claude = st.checkbox("💬 Analyse Claude Top 5", value=False,
                              help="Analyse textuelle détaillée pour les 5 meilleurs trades")
    st.markdown("---")
    st.markdown("### 📊 Signaux Intraday")
    use_intraday = st.checkbox("⚡ Activer signaux intraday", value=False,
                                help="VWAP · PDH/PDL · Multi-TF · Momentum vs SPY — données 15min délai")
    if use_intraday:
        st.markdown("""<div style='background:#4a90d015;border:1px solid #4a90d033;
            border-radius:6px;padding:8px;font-size:0.78rem;color:#93c5fd;margin-top:4px;'>
            ⚡ VWAP + PDH/PDL + Opening Range<br>
            ⚡ Multi-timeframe RSI (Daily+1H+15min)<br>
            ⚡ Momentum intraday vs SPY<br>
            ⚠️ Délai 15min — indicatif seulement<br>
            ⏱️ Ralentit le scan (~2x plus long)
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 Filtres tableau")
    signal_filter = st.multiselect(
        "Signaux",
        ["🟢 STRONG BUY","🟢 BUY","🟡 HOLD","🔴 AVOID","🟡 HOLD ⚠️"],
        default=["🟢 STRONG BUY","🟢 BUY"]
    )
    filter_earnings = st.checkbox(
        "📅 Exclure earnings cette semaine", value=True,
        help="Exclut les actions avec earnings dans les 5 prochains jours"
    )
    filter_top_sectors = st.checkbox(
        "💪 Top 5 secteurs seulement", value=False,
        help="Garde seulement les actions dans les 5 secteurs les plus forts"
    )

    st.markdown("---")
    st.markdown("<div style='color:#64748b;font-size:0.75rem;'>S&P 500 IA Screener Pro</div>", unsafe_allow_html=True)

# ─────────────────────────────
# 🚀 MAIN
# ─────────────────────────────
st.markdown("# 📊 S&P 500 IA Screener Pro")
st.markdown("<div style='color:#64748b;margin-bottom:1.5rem;'>Convergence · Patterns · R/R · TTM · Divergence · EMA · Marché · Claude IA</div>", unsafe_allow_html=True)

# BANDEAU MARCHÉ
with st.spinner("Vérification marché global..."):
    market_status = get_market_status()

regime = market_status["regime"]
color  = market_status["color"]
emoji  = market_status["emoji"]

st.markdown(f"""
<div class="market-banner" style="background:{color}11;border:1px solid {color}44;border-left:5px solid {color};">
    <strong style="color:{color};font-size:1.1rem;">{emoji} MARCHÉ {regime}</strong>
    &nbsp;—&nbsp; {market_status['message']}
    <br><span style="color:#94a3b8;font-size:0.82rem;font-family:'Space Mono',monospace;">{market_status['detail']}</span>
</div>
""", unsafe_allow_html=True)

with st.expander("💡 Conseils de trading"):
    advice_list = market_advice(market_status)
    cols = st.columns(2)
    for i, adv in enumerate(advice_list):
        cols[i%2].markdown(f"<div class='advice-box'>{adv}</div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    spy_color = "#00ff88" if market_status.get("spy_vs_ma50",0)>=0 else "#f87171"
    qqq_color = "#00ff88" if market_status.get("qqq_vs_ma50",0)>=0 else "#f87171"
    vix_val   = market_status.get("vix",None)
    vix_color = "#00ff88" if vix_val and vix_val<20 else "#fbbf24" if vix_val and vix_val<30 else "#f87171"
    for col, val, label, clr in [
        (c1, f"{'+' if market_status.get('spy_vs_ma50',0)>=0 else ''}{market_status.get('spy_vs_ma50','—')}%", "SPY vs MA50", spy_color),
        (c2, f"{'+' if market_status.get('qqq_vs_ma50',0)>=0 else ''}{market_status.get('qqq_vs_ma50','—')}%", "QQQ vs MA50", qqq_color),
        (c3, vix_val if vix_val else "—", "VIX", vix_color),
        (c4, market_status.get("spy_rsi","—"), "RSI SPY", "#00ff88"),
    ]:
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{clr}">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

# ── FORCE SECTORIELLE ──
with st.spinner("Analyse sectorielle..."):
    sector_data = get_sector_strength()

# Stocker en session state pour ai_score
st.session_state["sector_data"]  = sector_data
st.session_state["use_intraday"] = use_intraday

if sector_data["rankings"]:
    with st.expander("💪 Force Sectorielle cette semaine", expanded=False):
        st.markdown(f"**🔥 Secteur dominant : {sector_data['top_sector']}** &nbsp;|&nbsp; 🔴 Secteur faible : {sector_data['worst_sector']}")
        rows_sec = []
        for sector, data in sector_data["rankings"]:
            badge = sector_data["sector_badges"].get(sector, "")
            rows_sec.append({
                "Secteur":    sector,
                "Badge":      badge,
                "Mom 5j":     f"{'+' if data['mom_5d']>=0 else ''}{data['mom_5d']}%",
                "Mom 20j":    f"{'+' if data['mom_20d']>=0 else ''}{data['mom_20d']}%",
                "vs MA20":    "✅" if data['above_ma20'] else "❌",
                "Force":      str(data['strength']),
            })
        st.dataframe(pd.DataFrame(rows_sec).set_index("Secteur"), use_container_width=True)

# ─────────────────────────────────────────
# 📊 SECTION BACKTEST
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Validation — Backtest du Système")
st.markdown("<div style='color:#64748b;font-size:0.85rem;margin-bottom:1rem;'>Teste les signaux sur 52 semaines de données historiques réelles · Distinct du scan hebdomadaire</div>", unsafe_allow_html=True)

with st.expander("🔬 Lancer le Backtest", expanded=False):
    bt_col1, bt_col2 = st.columns(2)
    with bt_col1:
        bt_tickers_raw = st.text_area(
            "Tickers à backtester (un par ligne)",
            value="AAPL\nMSFT\nNVDA\nGOOGL\nMETA\nAMZN\nTSLA\nAVGO\nJPM\nV",
            height=180
        )
        bt_tickers = [t.strip().upper() for t in bt_tickers_raw.strip().split("\n") if t.strip()]
    with bt_col2:
        bt_workers = st.slider("Threads backtest", 3, 10, 5)
        st.markdown(f"**{len(bt_tickers)} tickers** × ~52 semaines = ~**{len(bt_tickers)*52} trades simulés**")
        st.markdown(f"Durée estimée : ~**{max(1, len(bt_tickers)//5)} minutes**")
        st.markdown("**Logique :** Entrée lundi · Stop ATR×1.5 · Target ATR×3 · Sortie vendredi si aucun niveau atteint")

    if st.button("▶ Lancer le Backtest", key="bt_btn"):
        bt_prog = st.progress(0)
        bt_stat = st.empty()
        def bt_cb(done, total):
            bt_prog.progress(done / total)
            bt_stat.markdown(f"🔬 Backtest `{done}/{total}` tickers...")

        df_bt = run_backtest(bt_tickers, weeks=52, max_workers=bt_workers, progress_callback=bt_cb)
        bt_stat.empty()

        if df_bt.empty:
            st.error("❌ Aucun trade simulé.")
        else:
            pnl_cols = [c for c in df_bt.columns if c.startswith("pnl_")]
            stats    = backtest_summary(df_bt)
            total_trades = len(df_bt)

            # Détecter quelle version du backtest est utilisée
            valid_keys = [k for k in stats.keys() if k in ["A","B","C","D","E","F"]]
            is_new_backtest = len(valid_keys) > 0

            if is_new_backtest:
                # ── NOUVEAU BACKTEST : 6 stratégies ──
                stats = {k: stats[k] for k in valid_keys}
                st.markdown(f"### 📈 Comparatif 6 Stratégies — {total_trades} trades simulés")

                strat_labels = {
                    "A": "A — +5% fixe",
                    "B": "B — +7% fixe",
                    "C": "C — Vendredi",
                    "D": "D — Stop 3%",
                    "E": "E — Stop 5%",
                    "F": "F — 50%+Stop 3%",
                }

                try:
                    best_strat = max(
                        stats.keys(),
                        key=lambda s: float(stats[s].get("expectancy", -99) or -99)
                    )
                except Exception:
                    best_strat = list(stats.keys())[0]

                rows_display = []
                for strat, data in stats.items():
                    try:
                        crown = " 👑" if strat == best_strat else ""
                        rows_display.append({
                            "Strategie":     strat_labels.get(strat, strat) + crown,
                            "Trades":        str(data.get("total", 0)),
                            "Win Rate":      str(data.get("win_rate", 0)) + "%",
                            "Expectancy":    str(data.get("expectancy", 0)) + "%",
                            "Profit Factor": str(data.get("profit_factor", 0)),
                            "Gain moyen":    "+" + str(data.get("avg_win", 0)) + "%",
                            "Perte moyenne": str(data.get("avg_loss", 0)) + "%",
                            "PnL Total":     str(data.get("total_pnl", 0)) + "%",
                            "Max Pertes":    str(data.get("max_consec_loss", 0)),
                        })
                    except Exception:
                        pass

                if rows_display:
                    st.dataframe(
                        pd.DataFrame(rows_display).set_index("Strategie"),
                        use_container_width=True
                    )

                st.markdown("---")
                selected_strat = st.selectbox(
                    "📊 Voir le détail d'une stratégie",
                    options=list(stats.keys()),
                    format_func=lambda s: strat_labels.get(s, s),
                    index=0
                )

                if selected_strat not in stats:
                    selected_strat = list(stats.keys())[0]

                sd = stats[selected_strat]
                st.markdown(f"#### Détail — {strat_labels.get(selected_strat, selected_strat)}")

                try:
                    d1,d2,d3,d4 = st.columns(4)
                    d1.metric("Meilleur trade",    f"+{sd.get('best', 0)}%")
                    d2.metric("Pire trade",         f"{sd.get('worst', 0)}%")
                    d3.metric("Max pertes consec.", str(sd.get("max_consec_loss", 0)))
                    d4.metric("PnL cumulé",         f"{sd.get('total_pnl', 0)}%")
                except Exception:
                    pass

                try:
                    score_stats = sd.get("score_stats") or {}
                    if score_stats:
                        st.markdown("**Performance par niveau de score IA :**")
                        rows_ss = []
                        for slabel, sdata in score_stats.items():
                            rows_ss.append({
                                "Score":    str(slabel),
                                "Win Rate": str(sdata.get("win_rate", 0)) + "%",
                                "PnL moy":  str(sdata.get("avg_pnl", 0)) + "%",
                                "N trades": str(sdata.get("n", 0)),
                            })
                        st.dataframe(pd.DataFrame(rows_ss).set_index("Score"), use_container_width=True)
                except Exception:
                    pass

                pnl_col = f"pnl_{selected_strat}"
                res_col = f"result_{selected_strat}"

                if pnl_col in df_bt.columns:
                    bt_tab1, bt_tab2 = st.tabs(["Courbe capital", "Win Rate/Score"])
                    with bt_tab1:
                        try:
                            df_s2 = df_bt.sort_values("week").copy()
                            df_s2["PnL cumule"] = df_s2[pnl_col].fillna(0).cumsum()
                            st.line_chart(df_s2["PnL cumule"])
                        except Exception as e:
                            st.error(f"Graphique indisponible: {e}")
                    with bt_tab2:
                        try:
                            df_bt2 = df_bt.copy()
                            df_bt2["score_bucket"] = pd.cut(df_bt2["score"],
                                bins=[0,40,50,60,70,80,101],
                                labels=["<40","40-50","50-60","60-70","70-80",">=80"])
                            wr_s = df_bt2.groupby("score_bucket", observed=True).apply(
                                lambda x: round(len(x[x[res_col]=="WIN"])/max(len(x),1)*100, 1)
                            ).reset_index()
                            wr_s.columns = ["Score","Win Rate %"]
                            st.bar_chart(wr_s.set_index("Score"))
                        except Exception as e:
                            st.error(f"Graphique indisponible: {e}")

            else:
                # ── ANCIEN BACKTEST : version simple ──
                st.info("ℹ️ Ancienne version du backtest détectée — mets à jour backtest.py pour les 6 stratégies.")
                st.markdown(f"### 📈 Résultats — {total_trades} trades simulés")

                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Win Rate",      f"{stats.get('win_rate', 0)}%")
                m2.metric("Profit Factor", f"{stats.get('profit_factor', 0)}")
                m3.metric("Gain moyen",    f"+{stats.get('avg_win', 0)}%")
                m4.metric("PnL Total",     f"{stats.get('total_pnl', 0)}%")

                if "pnl_pct" in df_bt.columns:
                    df_sorted = df_bt.sort_values("week").copy()
                    df_sorted["PnL cumule"] = df_sorted["pnl_pct"].fillna(0).cumsum()
                    st.line_chart(df_sorted["PnL cumule"])

            # ── EXPORT (commun aux deux versions) ──
            bt_excel = BytesIO()
            with pd.ExcelWriter(bt_excel, engine="openpyxl") as writer:
                df_bt.to_excel(writer, index=False, sheet_name="Trades")
                try:
                    strat_labels_export = {
                        "A": "A — +5% fixe", "B": "B — +7% fixe",
                        "C": "C — Vendredi", "D": "D — Stop 3%",
                        "E": "E — Stop 5%",  "F": "F — 50%+Stop 3%",
                    }
                    rows = []
                    for s, d in stats.items():
                        if s in ["A","B","C","D","E","F"]:
                            rows.append({
                                "Strategie":    strat_labels_export.get(s, s),
                                "Win Rate %":   d.get("win_rate", 0),
                                "Expectancy %": d.get("expectancy", 0),
                                "Profit Factor":d.get("profit_factor", 0),
                                "Gain Moy %":   d.get("avg_win", 0),
                                "Perte Moy %":  d.get("avg_loss", 0),
                                "PnL Total %":  d.get("total_pnl", 0),
                                "Max Pertes":   d.get("max_consec_loss", 0),
                            })
                    if rows:
                        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="Comparatif")
                except Exception:
                    pass
            st.download_button("⬇️ Télécharger résultats backtest", data=bt_excel.getvalue(),
                file_name=f"backtest_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
# ── Bouton charger dernier scan ──
saved_scan = load_scan_results()
if saved_scan:
    age = get_scan_age(saved_scan.get("timestamp", ""))
    st.markdown(
        f"<div class='prefilter-banner'>💾 Dernier scan sauvegardé : "
        f"<strong>{age}</strong> · {saved_scan.get('n_actions','?')} actions · "
        f"Marché {saved_scan.get('regime','?')}</div>",
        unsafe_allow_html=True
    )

st.markdown("---")
if st.button(f"🔄 Lancer — S&P 500 complet ({len(SP500_TICKERS)} actions)"):

    tickers_to_analyze = SP500_TICKERS

    # PASSE 1 : PRÉ-FILTRE
    if use_prefilter:
        st.markdown("### ⚡ Passe 1 — Pré-filtre rapide")
        pf_prog = st.progress(0)
        pf_stat = st.empty()
        def pf_cb(done, total):
            pf_prog.progress(done/total)
            pf_stat.markdown(f"⚡ Pré-filtre `{done}/{total}`...")
        pf_result = run_prefilter(SP500_TICKERS, max_workers=20, progress_callback=pf_cb)
        pf_stat.empty()
        tickers_to_analyze = pf_result["passed"]
        st.markdown(f"""<div class="prefilter-banner">
            ⚡ PASSE 1 TERMINÉE &nbsp;|&nbsp;
            <span style="color:#00ff88">{pf_result['n_passed']} retenues</span>
            &nbsp;/&nbsp; {pf_result['total']}
            &nbsp;|&nbsp; {pf_result['n_rejected']} éliminées
            &nbsp;|&nbsp; {pf_result['pass_rate']}% passage
        </div>""", unsafe_allow_html=True)
        with st.expander(f"🗑️ {pf_result['n_rejected']} actions éliminées"):
            st.dataframe(pd.DataFrame([
                {"Ticker": t, "Raison": pf_result['details'][t]}
                for t in pf_result['rejected']
            ]), use_container_width=True, height=200)

    if not tickers_to_analyze:
        st.error("❌ Aucune action ne passe le pré-filtre.")
        st.stop()

    # PASSE 2 : ANALYSE COMPLÈTE
    st.markdown(f"### 🔬 Passe 2 — Analyse complète ({len(tickers_to_analyze)} actions)")

    # Charger SPY une seule fois pour la Relative Strength
    with st.spinner("Chargement données SPY pour Relative Strength..."):
        spy_close = get_spy_data()

    rows = fetch_parallel(tickers_to_analyze, max_workers=nb_workers)

    if not rows:
        st.error("❌ Aucune donnée récupérée.")
        st.stop()

    df = pd.DataFrame(rows)

    scores_data      = df.apply(ai_score, axis=1)
    df["AI Score"]   = scores_data.apply(lambda x: x[0])
    df["AI Signal"]  = df["AI Score"].apply(ai_signal)
    df["AI Reasons"] = scores_data.apply(lambda x: " | ".join(x[1]))

    df = apply_market_filter(df, market_status)
    df.rename(columns={"AI Signal Ajusté":"AI Signal Ajuste","AI Score Ajusté":"AI Score Ajuste"}, errors="ignore", inplace=True)
    if "AI Score Ajuste" not in df.columns:
        df["AI Score Ajuste"] = df["AI Score"]
    if "AI Signal Ajuste" not in df.columns:
        df["AI Signal Ajuste"] = df["AI Signal"]
    df = df.sort_values("AI Score Ajuste", ascending=False).reset_index(drop=True)

    # Sauvegarder les résultats du scan
    save_scan_results(df, market_status, regime)
    st.success(f"✅ Résultats sauvegardés — {len(df)} actions analysées")

    # ════════════════════════════════════════
    # 🎯 RAPPORT DE CONVERGENCE — TOP TRADES
    # ════════════════════════════════════════
    st.markdown("---")

    # Bannière mode strict
    if strict_mode:
        st.markdown("""<div style='background:#00ff8812;border:1px solid #00ff8844;
            border-left:4px solid #00ff88;border-radius:8px;padding:12px 16px;margin-bottom:12px;
            font-size:0.85rem;color:#86efac;'>
            🔒 <strong>MODE STRICT ACTIF</strong> — Min 4/6 signaux · R/R ≥ 2.0 · Score ≥ 70
            — Optimisé pour maximiser le win rate
        </div>""", unsafe_allow_html=True)

    st.markdown(f"## 🎯 Rapport du Dimanche — Top {top_n} Trades Convergents")
    st.markdown(f"<div style='color:#64748b;font-size:0.85rem;margin-bottom:1rem;'>Semaine du {datetime.now().strftime('%d %B %Y')} · Marché {regime} · Min {min_signals}/6 signaux</div>", unsafe_allow_html=True)

    # Conseils d'exécution semaine
    day_advice = get_day_of_week_advice(regime)
    with st.expander("📅 Plan d'exécution de la semaine"):
        cols = st.columns(3)
        days = [("Lundi","🟢"), ("Mercredi","🟡"), ("Vendredi","🔴")]
        for i, (day, em) in enumerate(days):
            cols[i].markdown(f"""<div class="advice-box">
                <strong>{em} {day}</strong><br>{day_advice.get(day,'—')}
            </div>""", unsafe_allow_html=True)

    # Score minimum selon mode
    score_min_rapport = 70 if strict_mode else min_score

    # Construire le rapport de convergence
    report = build_trade_report(
        df,
        top_n=top_n,
        min_signals=min_signals,
        min_rr=min_rr_conv
    )

    # Filtres strict sur le rapport final
    if not report.empty:
        ai_col = "AI Score Ajuste" if "AI Score Ajuste" in report.columns else "AI Score"
        report = report[report[ai_col] >= score_min_rapport]
        report = report[report["Conv_N"] >= min_signals]
        # Limite sectorielle — max 2 titres par secteur
        report = check_sector_diversity(report, top_n=top_n, max_per_sector=2)
        # Filtre Fibonacci — en mode strict, exclure les entrées invalides
        if strict_mode and "FIB_EntryValid" in report.columns:
            n_before = len(report)
            report_fib = report[report["FIB_EntryValid"] != False]
            n_removed  = n_before - len(report_fib)
            if n_removed > 0:
                st.info(f"📐 {n_removed} titre(s) exclu(s) — résistance Fibonacci trop proche")

    # ── SCORE CLAUDE DYNAMIQUE ──
    if not report.empty and use_claude_scorer and api_key:
        st.markdown("---")
        st.markdown("### 🤖 Score Claude Dynamique en cours...")
        cs_prog = st.progress(0)
        cs_stat = st.empty()

        def cs_cb(done, total):
            cs_prog.progress(done / total)
            cs_stat.markdown(f"🤖 Claude analyse `{done}/{total}` actions...")

        report = claude_score_batch(
            report,
            market_status=market_status,
            api_key=api_key,
            top_n=min(claude_top_n, len(report)),
            delay=0.3,
            progress_callback=cs_cb,
        )
        cs_stat.empty()
        cs_prog.empty()
        st.markdown(
            "<div style='color:#a78bfa;font-size:0.85rem;margin-bottom:8px;'>"
            f"✅ Score Claude calculé pour {min(claude_top_n, len(report))} actions"
            " — rapport trié par score Claude</div>",
            unsafe_allow_html=True
        )
    elif use_claude_scorer and not api_key:
        st.warning("⚠️ Entrez votre clé API Anthropic dans la sidebar pour activer le score Claude.")

    if report.empty:
        st.warning(f"⚠️ Aucun titre avec {min_signals}+ signaux convergents. Réduire le filtre dans la sidebar.")
    else:
        st.markdown(f"<div style='color:#00ff88;font-size:0.85rem;margin-bottom:16px;'>✅ {len(report)} opportunités identifiées</div>", unsafe_allow_html=True)

        for idx, row in report.iterrows():
            rank      = idx + 1
            n_sig     = int(row.get("Conv_N", 0))
            conv_bar  = str(row.get("Conv_Bar", "░░░░░░"))
            conv_lbl  = str(row.get("Conv_Label", ""))
            conv_clr  = str(row.get("Conv_Color", "#64748b"))
            score_fin = row.get("Score_Final", 0)
            rr        = row.get("RR_Ratio", None)
            rr_str    = str(rr) if rr else "—"
            rr_color  = "#00ff88" if rr and float(rr) >= 2 else "#fbbf24"
            signals_on_list  = str(row.get("Conv_On", "") or "").split(" | ")
            signals_off_list = str(row.get("Conv_Off", "") or "").split(" | ")
            entree    = str(row.get("Entree", "—") or "—")
            stop      = str(row.get("Stop", "—") or "—")
            target    = str(row.get("Target", "—") or "—")
            risque    = str(row.get("Risque_Pct", "—") or "—")
            gain      = str(row.get("Gain_Pct", "—") or "—")
            atr_pct   = str(row.get("ATR_Pct", "—") or "—")
            support   = str(row.get("Support", "—") or "—")
            resist    = str(row.get("Resistance", "—") or "—")

            if rank == 1:
                border_color = "#ffd700"
                bg_color = "#1a130022"
            elif n_sig >= 5:
                border_color = "#00ff88"
                bg_color = "#00ff880a"
            else:
                border_color = "#1e4060"
                bg_color = "#0b142288"

            # ── En-tête de la carte ──
            st.markdown(
                f'<div style="background:{bg_color};border:2px solid {border_color};border-radius:14px;padding:20px 24px;margin:10px 0;">',
                unsafe_allow_html=True
            )

            # Ligne titre
            col_rank, col_ticker, col_conv = st.columns([1, 3, 3])
            col_rank.markdown(f"<div style='font-family:Space Mono,monospace;font-size:2rem;font-weight:700;color:{border_color};'>#{rank}</div>", unsafe_allow_html=True)
            col_ticker.markdown(f"<div style='font-family:Space Mono,monospace;font-size:1.2rem;font-weight:700;color:#00ff88;'>{row['Ticker']}</div><div style='color:#64748b;font-size:0.85rem;'>{row['Sector']}</div>", unsafe_allow_html=True)
            col_conv.markdown(f"<div style='font-family:Space Mono,monospace;font-size:1.1rem;color:{conv_clr};'>{conv_bar} {n_sig}/6</div><div style='color:{conv_clr};font-size:0.8rem;'>{conv_lbl}</div>", unsafe_allow_html=True)

            # Score + Prix + Claude
            claude_ok      = bool(row.get("claude_ok", False))
            claude_score_v = row.get("claude_score", None)
            claude_verdict_v = str(row.get("claude_verdict", "") or "")
            claude_conviction_v = str(row.get("claude_conviction", "") or "")
            claude_raison_v = str(row.get("claude_raison", "") or "")
            claude_risque_v = str(row.get("claude_risque", "") or "")
            claude_stop_v  = row.get("claude_stop_adj", None)

            if claude_ok and claude_score_v is not None:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Score Algo", f"{score_fin}/100")
                vc = verdict_color(claude_verdict_v)
                c2.markdown(
                    f"**Score Claude**\n\n"
                    f"<span style='font-size:1.5rem;font-weight:700;color:{vc};'>"
                    f"{claude_score_v}/100</span>",
                    unsafe_allow_html=True
                )
                c3.markdown(
                    f"**Verdict Claude**\n\n"
                    f"<span style='color:{vc};font-weight:700;'>{claude_verdict_v}</span><br>"
                    f"<span style='color:#64748b;font-size:0.8rem;'>{conviction_badge(claude_conviction_v)}</span>",
                    unsafe_allow_html=True
                )
                c4.metric("Prix actuel", f"${row['Prix']}")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Score Final", f"{score_fin}/100")
                c2.metric("Prix actuel", f"${row['Prix']}")
                c3.metric("RSI | Volume", f"{row['RSI']} | {row['Vol_Ratio']}x")

            st.markdown("---")

            # Plan de trade
            col_e, col_s, col_t = st.columns(3)
            col_e.markdown(f"**🎯 ENTRÉE** *(Lundi)*\n\n`${entree}`")
            col_s.markdown(f"**🛑 STOP-LOSS**\n\n`${stop}` **(-{risque}%)**")
            col_t.markdown(f"**🏆 VENTE** *(Jeu/Ven)*\n\n`${target}` **(+{gain}%)**")

            # Règle de gap — avertissement si strict mode
            if strict_mode and entree and entree != "—":
                try:
                    entree_f = float(entree)
                    prix_f   = float(row.get("Prix", entree_f))
                    gap_max  = round(entree_f * 1.015, 2)
                    st.markdown(
                        f"⚠️ **Règle gap :** N'entrer que si ouverture lundi ≤ **${gap_max}** "
                        f"(+1.5% max vs entrée). Au-dessus → passer au trade suivant."
                    )
                except Exception:
                    pass

            # ── SIGNAUX INTRADAY (si activés) ──
            if st.session_state.get("use_intraday", False):
                id_vwap_b   = str(row.get("ID_VWAP_Badge", "") or "")
                id_vwap_s   = str(row.get("ID_VWAP_Signal", "") or "")
                id_vwap_v   = row.get("ID_VWAP", None)
                id_vwap_d   = row.get("ID_VWAP_Dist", None)
                id_pdh      = row.get("ID_PDH", None)
                id_pdl      = row.get("ID_PDL", None)
                id_orb      = row.get("ID_ORB_Break", False)
                id_tf_b     = str(row.get("ID_TF_Badge", "") or "")
                id_tf_s     = str(row.get("ID_TF_Signal", "") or "")
                id_tf_conf  = str(row.get("ID_TF_Confirm", "") or "")
                id_rsi_1h   = row.get("ID_RSI_1H", None)
                id_rsi_15   = row.get("ID_RSI_15min", None)
                id_mom_b    = str(row.get("ID_Mom_Badge", "") or "")
                id_mom_rel  = row.get("ID_Mom_Rel", None)

                st.markdown("**⚡ Signaux Intraday (délai 15min) :**")

                # VWAP
                if id_vwap_b and id_vwap_b != "—":
                    vwap_c = "#00ff88" if (row.get("ID_VWAP_Score",0) or 0)>=0 else "#f87171"
                    st.markdown(f"<span style='color:{vwap_c};'>📊 VWAP: {id_vwap_b}</span>", unsafe_allow_html=True)
                    parts = []
                    if id_vwap_v:  parts.append(f"VWAP: `${id_vwap_v}`")
                    if id_pdh:     parts.append(f"PDH: `${id_pdh}`")
                    if id_pdl:     parts.append(f"PDL: `${id_pdl}`")
                    if id_orb:     parts.append(f"✅ ORB Breakout")
                    if parts: st.markdown(" | ".join(parts))

                # Multi-TF
                if id_tf_b and id_tf_b != "—":
                    tf_c = "#00ff88" if id_tf_conf=="FORT" else "#f87171" if id_tf_conf in ["CONTRADICTOIRE","BAISSIER"] else "#fbbf24"
                    st.markdown(f"<span style='color:{tf_c};'>🔄 Multi-TF: {id_tf_b}</span>", unsafe_allow_html=True)
                    rsi_parts = []
                    if id_rsi_1h:  rsi_parts.append(f"RSI 1H: `{id_rsi_1h}`")
                    if id_rsi_15:  rsi_parts.append(f"RSI 15m: `{id_rsi_15}`")
                    if rsi_parts: st.markdown(" | ".join(rsi_parts))

                # Momentum
                if id_mom_b and id_mom_b != "—":
                    mom_c = "#00ff88" if (row.get("ID_Mom_Score",0) or 0)>0 else "#f87171"
                    st.markdown(f"<span style='color:{mom_c};'>🚀 Momentum: {id_mom_b}</span>", unsafe_allow_html=True)

            # ── FIBONACCI V2 ──
            fib_badge    = str(row.get("FIB_Badge", "") or "")
            fib_context  = str(row.get("FIB_Context", "NEUTRE") or "NEUTRE")
            fib_valid    = row.get("FIB_EntryValid", True)
            fib_signal_v = str(row.get("FIB_Signal", "") or "")
            fib_warning  = str(row.get("FIB_Warning", "") or "")
            fib_stop_v   = row.get("FIB_Stop", None)
            fib_target_v = row.get("FIB_Target", None)
            fib_rr_v     = row.get("FIB_RR", None)
            fib_support  = row.get("FIB_Support", None)
            fib_resist   = row.get("FIB_Resist", None)
            fib_dist_r   = row.get("FIB_DistResist", None)
            fib_score_v  = int(row.get("FIB_Score", 0) or 0)

            # Couleur selon contexte
            fib_color = {
                "REBOND_KEY":        "#00ff88",
                "BREAKOUT":          "#00ff88",
                "ZONE_SAINE":        "#86efac",
                "NEUTRE":            "#fbbf24",
                "RESISTANCE_PROCHE": "#f87171",
            }.get(fib_context, "#fbbf24")

            fib_display = fib_badge if fib_badge and fib_badge != "—" else f"~ Fibonacci {fib_context}"
            st.markdown(
                f"**📐 Fibonacci :** <span style='color:{fib_color};font-weight:700;'>{fib_display}</span>",
                unsafe_allow_html=True
            )

            # Avertissement rouge si entrée invalide
            if not fib_valid and fib_warning:
                st.markdown(
                    f"<div style='background:#f871711a;border-left:3px solid #f87171;"
                    f"border-radius:4px;padding:8px;margin:4px 0;font-size:0.85rem;color:#f87171;'>"
                    f"⚠️ {fib_warning}</div>",
                    unsafe_allow_html=True
                )
            elif fib_signal_v and fib_signal_v not in ["None","—",""]:
                st.markdown(
                    f"<span style='color:{fib_color};font-size:0.85rem;'>  {fib_signal_v}</span>",
                    unsafe_allow_html=True
                )

            # Niveaux Fib clés
            fib_parts = []
            if fib_support: fib_parts.append(f"Support: `${fib_support}`")
            if fib_resist:  fib_parts.append(f"Résistance: `${fib_resist}`")
            if fib_dist_r and fib_dist_r < 99: fib_parts.append(f"Dist. résist: `{fib_dist_r}%`")
            if fib_parts: st.markdown(" &nbsp;|&nbsp; ".join(fib_parts))

            # Stop + Target + R/R Fibonacci
            if fib_stop_v and fib_target_v:
                fib_rr_str = f"R/R Fib: `{fib_rr_v}:1`" if fib_rr_v else ""
                st.markdown(
                    f"🛡️ Stop Fib: `${fib_stop_v}` &nbsp;|&nbsp; "
                    f"🎯 Target Fib: `${fib_target_v}`"
                    + (f" &nbsp;|&nbsp; {fib_rr_str}" if fib_rr_str else "")
                )

            # Bollinger avancé — toujours afficher
            bb_badge  = str(row.get("BB_Badge", "") or "")
            bb_trend  = str(row.get("BB_WidthTrend", "") or "")
            bb_signal_v = str(row.get("BB_Signal", "") or "")
            bb_score_v  = int(row.get("BB_Score", 0) or 0)
            bb_color   = "#00ff88" if bb_score_v > 0 else "#fbbf24" if bb_score_v == 0 else "#f87171"
            trend_icon = "📈" if bb_trend=="EXPANDING" else "📉" if bb_trend=="CONTRACTING" else "~"
            bb_display = bb_badge if bb_badge and bb_badge != "—" else f"~ BB {bb_trend}"
            st.markdown(
                f"**📊 Bollinger :** <span style='color:{bb_color};'>{bb_display}</span> {trend_icon}",
                unsafe_allow_html=True
            )
            if bb_signal_v and bb_signal_v not in ["None","—",""]:
                st.markdown(f"<span style='color:{bb_color};font-size:0.85rem;'>  {bb_signal_v}</span>", unsafe_allow_html=True)

            # Support & Résistance 52 semaines
            sr_badge    = str(row.get("SR_Badge", "") or "")
            sr_high     = row.get("SR_High52w", None)
            sr_low      = row.get("SR_Low52w", None)
            sr_pos      = row.get("SR_Position", None)
            sr_stop_nat = row.get("SR_StopNatural", None)
            sr_tgt_nat  = row.get("SR_TargetNatural", None)

            if sr_badge and sr_badge not in ["—", ""]:
                sr_color = "#00ff88" if (row.get("SR_Score", 0) or 0) >= 10 else \
                           "#fbbf24" if (row.get("SR_Score", 0) or 0) >= 0 else "#f87171"
                st.markdown(
                    f"**📐 S/R 52 sem :** <span style='color:{sr_color};'>{sr_badge}</span>",
                    unsafe_allow_html=True
                )
                info_parts = []
                if sr_high: info_parts.append(f"High 52w: `${sr_high}`")
                if sr_low:  info_parts.append(f"Low 52w: `${sr_low}`")
                if sr_pos is not None: info_parts.append(f"Position: `{sr_pos}%` du range")
                if info_parts:
                    st.markdown(" &nbsp;|&nbsp; ".join(info_parts))
                if sr_stop_nat and sr_tgt_nat:
                    st.markdown(
                        f"🛡️ Stop S/R: `${sr_stop_nat}` &nbsp;|&nbsp; "
                        f"🎯 Target S/R: `${sr_tgt_nat}`"
                    )

            # Relative Strength
            rs_badge    = str(row.get("RS_Badge", "") or "")
            rs_trend_v  = str(row.get("RS_Trend", "") or "")
            rs_perf     = row.get("RS_Perf5d", None)
            spy_perf    = row.get("SPY_Perf5d", None)
            rs_5d_v     = row.get("RS_5d", None)
            if rs_badge and rs_badge not in ["—", ""]:
                rs_color = "#00ff88" if row.get("RS_Outperform") else "#f87171"
                rs_text  = f"**💪 RS :** {rs_badge}"
                if rs_perf is not None and spy_perf is not None:
                    diff = round(float(rs_perf) - float(spy_perf), 2)
                    sign = "+" if diff >= 0 else ""
                    rs_text += f" &nbsp;|&nbsp; Action: `{'+' if float(rs_perf)>=0 else ''}{rs_perf}%` vs SPY: `{'+' if float(spy_perf)>=0 else ''}{spy_perf}%` → <span style='color:{rs_color};'>{sign}{diff}% vs marché</span>"
                st.markdown(rs_text, unsafe_allow_html=True)

            # Gap signal
            gap_badge   = str(row.get("Gap_Badge", "") or "")
            gap_signal  = str(row.get("Gap_Signal", "") or "")
            gap_score_v = row.get("Gap_Score", 0)
            gap_support = row.get("Gap_Support", None)
            if gap_badge and gap_badge not in ["—", ""]:
                gap_color = "#00ff88" if (gap_score_v or 0) > 0 else "#f87171"
                st.markdown(
                    f"<span style='color:{gap_color};font-weight:700;font-size:0.85rem;'>{gap_badge}</span>",
                    unsafe_allow_html=True
                )
                if gap_support:
                    st.markdown(f"🛡️ Support gap : `${gap_support}` — niveau de stop solide")

            # Earnings warning
            earn_badge = str(row.get("Earnings_Badge", "") or "")
            earn_risk  = str(row.get("Earnings_Risk", "") or "")
            if earn_badge and earn_badge not in ["✅ Pas d'earnings","—",""]:
                earn_color = "#f87171" if earn_risk == "ÉLEVÉ" else "#fbbf24" if earn_risk == "MODÉRÉ" else "#86efac"
                st.markdown(
                    f"<span style='color:{earn_color};font-weight:700;font-size:0.9rem;'>{earn_badge}</span>",
                    unsafe_allow_html=True
                )

            # R/R + détails
            st.markdown(
                f"📊 **R/R:** `{rr_str}:1` &nbsp;|&nbsp; "
                f"ATR: `{atr_pct}%` &nbsp;|&nbsp; "
                f"Support: `${support}` &nbsp;|&nbsp; "
                f"Résistance: `${resist}`"
            )

            # Volume anormal
            vol_badge  = str(row.get("VOL_Badge", "") or "")
            vol_signal = str(row.get("VOL_Signal", "") or "")
            vol_ratio_v= row.get("VOL_Ratio", None)
            vol_rank   = row.get("VOL_52W_Rank", None)
            if vol_badge and vol_badge != "—":
                vol_info = f"**Volume :** {vol_badge}"
                if vol_ratio_v:
                    vol_info += f" — `{vol_ratio_v}x` la moyenne"
                if vol_rank:
                    vol_info += f" — Top `{round(100-float(vol_rank),1)}%` annuel"
                st.markdown(vol_info)
                if vol_signal and vol_signal not in ["None","—",""]:
                    st.markdown(
                        f"<span style='color:#00ff88;font-size:0.85rem;'>  {vol_signal}</span>",
                        unsafe_allow_html=True
                    )

            # Signaux actifs
            if signals_on_list and signals_on_list[0]:
                st.markdown("**Signaux actifs :**")
                for s in signals_on_list:
                    if s.strip():
                        st.markdown(f"<span style='color:#86efac;font-size:0.85rem;'>  {s.strip()}</span>", unsafe_allow_html=True)

            # Signaux manquants
            if signals_off_list and signals_off_list[0]:
                st.markdown("**Signaux manquants :**")
                for s in signals_off_list:
                    if s.strip():
                        st.markdown(f"<span style='color:#f87171;font-size:0.85rem;'>  {s.strip()}</span>", unsafe_allow_html=True)

            # Raisonnement Claude
            if claude_ok and claude_raison_v and claude_raison_v != "—":
                vc = verdict_color(claude_verdict_v)
                st.markdown("**🤖 Analyse Claude :**")
                st.markdown(
                    f"<div style='background:#a78bfa0d;border-left:3px solid #a78bfa;"
                    f"border-radius:4px;padding:10px 14px;margin:6px 0;font-size:0.85rem;'>"
                    f"<span style='color:{vc};font-weight:700;'>{claude_verdict_v}</span>"
                    f" — {claude_raison_v}<br>"
                    f"<span style='color:#f87171;font-size:0.82rem;'>⚠️ {claude_risque_v}</span>"
                    + (f"<br><span style='color:#fbbf24;font-size:0.82rem;'>📐 Stop ajusté: ${claude_stop_v}</span>" if claude_stop_v else "")
                    + "</div>",
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("")

    # ── MÉTRIQUES GLOBALES ──
    st.markdown("---")
    st.markdown("### 📈 Vue d'ensemble")
    ai_col  = "AI Score Ajuste"
    sig_col = "AI Signal Ajuste"
    vol_anormal  = len(df[df["VOL_Score"] >= 15]) if "VOL_Score" in df.columns else 0
    gaps_forts   = len(df[df["Gap_Score"] >= 12]) if "Gap_Score" in df.columns else 0
    rs_forts     = len(df[df["RS_Score"] >= 65]) if "RS_Score" in df.columns else 0
    col1,col2,col3,col4,col5,col6,col7,col8,col9,col10 = st.columns(10)
    for col, val, label in zip(
        [col1,col2,col3,col4,col5,col6,col7,col8,col9,col10],
        [
            len(df),
            len(df[df[sig_col]=="🟢 STRONG BUY"]),
            len(df[df[sig_col]=="🟢 BUY"]),
            len(df[df[sig_col]=="🟡 HOLD"]),
            len(df[df[sig_col]=="🔴 AVOID"]),
            len(report),
            len(df[df.get("Conv_N", pd.Series([0]*len(df))) >= 4]) if "Conv_N" in df.columns else 0,
            vol_anormal,
            gaps_forts,
            rs_forts,
        ],
        ["Analysées","Strong Buy","Buy","Hold","Avoid",f"Top {top_n}","Conv≥4","Vol","Gaps","RS Fort"]
    ):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    # ── GRAPHIQUES ──
    st.markdown("---")
    st.markdown("### 📊 Visualisations")
    import plotly.express as px
    tab1,tab2,tab3,tab4 = st.tabs(["Convergence","Distribution","RSI vs Score","R/R Ratio"])

    with tab1:
        if not report.empty and "Conv_N" in report.columns:
            fig_c = px.bar(report, x="Ticker", y="Conv_N",
                           color="Score_Final",
                           color_continuous_scale=["#1e3a5f","#00ff88"],
                           hover_data=["Score_Final","RR_Ratio","Top_Pattern"],
                           title=f"Top {top_n} — Convergence des signaux")
            fig_c.add_hline(y=6, line_dash="dot", line_color="#ffd700", annotation_text="6/6 parfait")
            fig_c.add_hline(y=4, line_dash="dash", line_color="#00ff88", annotation_text="4/6 minimum recommandé")
            fig_c.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                                font_color="#e2e8f0", title_font_color="#00ff88",
                                xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f",range=[0,7]))
            st.plotly_chart(fig_c, use_container_width=True)

    with tab2:
        fig = px.histogram(df, x=ai_col, nbins=20,
                           color_discrete_sequence=["#00ff88"],
                           title=f"Distribution — {regime} — {len(df)} actions")
        fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                          font_color="#e2e8f0", title_font_color="#00ff88",
                          xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig2 = px.scatter(df, x="RSI", y=ai_col, color=sig_col,
                          hover_data=["Ticker","Top_Pattern","RR_Ratio","ADV_Badge"],
                          color_discrete_map={
                              "🟢 STRONG BUY":"#00ff88","🟢 BUY":"#4ade80",
                              "🟡 HOLD":"#fbbf24","🟡 HOLD ⚠️":"#fb923c","🔴 AVOID":"#f87171"
                          }, title="RSI vs Score")
        fig2.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                           font_color="#e2e8f0", title_font_color="#00ff88",
                           xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        df_rr = df[df["RR_Ratio"].notna()].sort_values("RR_Ratio",ascending=False).head(15)
        if not df_rr.empty:
            fig5 = px.bar(df_rr, x="Ticker", y="RR_Ratio", color="RR_Ratio",
                          color_continuous_scale=["#f87171","#fbbf24","#00ff88"],
                          hover_data=["Entree","Stop","Target","Risque_Pct","Gain_Pct"],
                          title="Top R/R Ratio")
            fig5.add_hline(y=2.0, line_dash="dash", line_color="#fbbf24", annotation_text="R/R min (2:1)")
            fig5.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                               font_color="#e2e8f0", title_font_color="#00ff88",
                               xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
            st.plotly_chart(fig5, use_container_width=True)

    # ── PAPER TRADING + JOURNAL ──
    st.markdown("---")

    pt_tab, journal_tab = st.tabs(["📊 Paper Trading", "📓 Journal Réel"])

    with pt_tab:
        st.markdown("### 📊 Paper Trading — Simulation semaine")
        st.markdown("<div style='color:#64748b;font-size:0.85rem;margin-bottom:12px;'>Simule le Top 10 sans risquer d'argent réel — résultats mis à jour automatiquement</div>", unsafe_allow_html=True)

        # Mettre à jour les trades ouverts
        paper_trades = update_paper_results()
        paper_summary = get_paper_summary()

        # Stats paper trading
        if paper_summary["n_closed"] > 0:
            pc1,pc2,pc3,pc4 = st.columns(4)
            wc = "#00ff88" if paper_summary["win_rate"]>=55 else "#fbbf24" if paper_summary["win_rate"]>=45 else "#f87171"
            pc1.metric("Trades fermés", paper_summary["n_closed"])
            pc2.metric("Win Rate", f"{paper_summary['win_rate']}%")
            pc3.metric("PnL moyen", f"{paper_summary['avg_pnl']}%")
            pc4.metric("PnL total", f"{paper_summary['total_pnl']}%")

        # Ajouter un trade paper
        st.markdown("**➕ Ajouter un trade fictif du Top 10 :**")
        if not report.empty:
            tickers_list = report["Ticker"].tolist() if "Ticker" in report.columns else []
            if tickers_list:
                col_pt1, col_pt2, col_pt3 = st.columns(3)
                with col_pt1:
                    pt_ticker = st.selectbox("Ticker", tickers_list, key="pt_ticker")
                with col_pt2:
                    pt_strategy = st.selectbox("Stratégie", ["A","B","C","D","E","F"], key="pt_strat")
                with col_pt3:
                    if st.button("➕ Ajouter trade fictif", key="pt_add"):
                        pt_row = report[report["Ticker"]==pt_ticker].iloc[0]
                        add_paper_trade(
                            ticker=pt_ticker,
                            entry_price=pt_row.get("Entree", pt_row.get("Prix", 0)),
                            stop_price=pt_row.get("Stop", 0),
                            target_price=pt_row.get("Target", 0),
                            conv_n=pt_row.get("Conv_N", 0),
                            score=pt_row.get("Score_Final", pt_row.get("AI Score Ajuste", 0)),
                            strategy=pt_strategy,
                            sector=pt_row.get("Sector", ""),
                            week_date=datetime.now().strftime("%Y-W%V"),
                        )
                        st.success(f"✅ Trade fictif {pt_ticker} ajouté !")

        # Afficher les trades paper ouverts
        open_trades = [t for t in paper_trades if t.get("status") == "OPEN"]
        if open_trades:
            st.markdown("**Trades fictifs ouverts :**")
            rows_pt = []
            for t in open_trades:
                rows_pt.append({
                    "Ticker":    t.get("ticker"),
                    "Entrée":    f"${t.get('entry_price')}",
                    "Stop":      f"${t.get('stop_price')}",
                    "Target":    f"${t.get('target_price')}",
                    "Stratégie": t.get("strategy"),
                    "P&L actuel":f"{t.get('current_pnl', '—')}%",
                    "Conv":      f"{t.get('conv_n')}/6",
                    "Semaine":   t.get("week"),
                })
            st.dataframe(pd.DataFrame(rows_pt), use_container_width=True)

        # Trades fermés
        closed_trades = [t for t in paper_trades if t.get("status") == "CLOSED"]
        if closed_trades:
            with st.expander(f"📋 Historique ({len(closed_trades)} trades fermés)"):
                rows_cl = []
                for t in closed_trades:
                    rows_cl.append({
                        "Ticker":  t.get("ticker"),
                        "Entrée":  f"${t.get('entry_price')}",
                        "Sortie":  f"${t.get('exit_price')}",
                        "PnL":     f"{t.get('pnl_pct')}%",
                        "Résultat":t.get("result"),
                        "Strat":   t.get("strategy"),
                        "Semaine": t.get("week"),
                    })
                st.dataframe(pd.DataFrame(rows_cl), use_container_width=True)

    with journal_tab:
        st.markdown("### 📓 Journal de Trades Réels")
        st.markdown("<div style='color:#64748b;font-size:0.85rem;margin-bottom:12px;'>Enregistre tes vrais trades pour mesurer tes résultats réels</div>", unsafe_allow_html=True)

        journal_summary = get_journal_summary()

        if journal_summary["n_closed"] > 0:
            jc1,jc2,jc3,jc4 = st.columns(4)
            jc1.metric("Trades réels", journal_summary["n_closed"])
            jc2.metric("Win Rate réel", f"{journal_summary['win_rate']}%")
            jc3.metric("PnL moyen réel", f"{journal_summary['avg_pnl']}%")
            jc4.metric("PnL total réel", f"{journal_summary['total_pnl']}%")

        # Ajouter trade réel
        st.markdown("**➕ Enregistrer un trade réel :**")
        jcol1, jcol2 = st.columns(2)
        with jcol1:
            j_ticker  = st.text_input("Ticker (ex: KEY)", key="j_ticker").upper()
            j_entry   = st.number_input("Prix d'entrée", min_value=0.01, value=21.68, key="j_entry")
            j_stop    = st.number_input("Stop-loss", min_value=0.01, value=20.95, key="j_stop")
        with jcol2:
            j_target  = st.number_input("Target", min_value=0.01, value=23.00, key="j_target")
            j_strat   = st.selectbox("Stratégie", ["A","B","C","D","E","F"], index=2, key="j_strat")
            j_sector  = st.text_input("Secteur", value="Financial Services", key="j_sector")
            j_notes   = st.text_input("Notes (optionnel)", key="j_notes")

        if st.button("➕ Enregistrer le trade", key="j_add"):
            if j_ticker:
                add_journal_trade(j_ticker, j_entry, j_stop, j_target, j_strat, j_sector, j_notes)
                st.success(f"✅ Trade {j_ticker} enregistré dans le journal !")

        # Fermer un trade ouvert
        journal_trades = get_journal_summary().get("trades", [])
        open_j = [t for t in journal_trades if t.get("status") == "OPEN"]
        if open_j:
            st.markdown("**Fermer un trade :**")
            close_options = {f"{t['ticker']} — entré ${t['entry_price']}": t["id"] for t in open_j}
            selected_close = st.selectbox("Trade à fermer", list(close_options.keys()), key="j_close_sel")
            exit_price_val = st.number_input("Prix de sortie", min_value=0.01, value=21.76, key="j_exit")
            if st.button("✅ Fermer ce trade", key="j_close_btn"):
                close_journal_trade(close_options[selected_close], exit_price_val)
                st.success(f"✅ Trade fermé !")

        # Afficher le journal
        all_journal = get_journal_summary().get("trades", [])
        if all_journal:
            with st.expander(f"📋 Journal complet ({len(all_journal)} trades)"):
                rows_j = []
                for t in all_journal:
                    rows_j.append({
                        "Ticker":   t.get("ticker"),
                        "Date":     t.get("date_entry"),
                        "Entrée":   f"${t.get('entry_price')}",
                        "Stop":     f"${t.get('stop_price')}",
                        "Target":   f"${t.get('target_price')}",
                        "Sortie":   f"${t.get('exit_price','—')}",
                        "PnL":      f"{t.get('pnl_pct','—')}%",
                        "Résultat": t.get("result","OPEN"),
                        "Strat":    t.get("strategy"),
                        "Notes":    t.get("notes",""),
                    })
                st.dataframe(pd.DataFrame(rows_j), use_container_width=True)

    # ── TABLEAU COMPLET ──
    st.markdown("---")
    df_filtered = df[df[ai_col] >= min_score]
    if signal_filter:
        df_filtered = df_filtered[df_filtered[sig_col].isin(signal_filter)]

    # Filtre earnings
    if filter_earnings and "Earnings_Avoid" in df_filtered.columns:
        df_before = len(df_filtered)
        df_filtered = df_filtered[df_filtered["Earnings_Avoid"] != True]
        n_removed = df_before - len(df_filtered)
        if n_removed > 0:
            st.info(f"📅 {n_removed} action(s) exclues pour cause d'earnings cette semaine.")
    st.markdown(f"### 📋 Tableau complet ({len(df_filtered)} actions)")
    cols_display = [
        "Ticker","Sector","Prix","MA50","MA200",
        "RSI","MACD_Hist","Vol_Ratio","Rev_Growth",
        "FIB_Badge","FIB_Context","FIB_EntryValid","FIB_EntryReason",
        "FIB_Stop","FIB_Target","FIB_RR","FIB_DistResist","FIB_Warning",
        "BB_Badge","BB_Signal","BB_Width","BB_WidthTrend",
        "SR_Badge","SR_Signal","SR_Position","SR_DistHigh",
        "SR_High52w","SR_Low52w","SR_StopNatural","SR_TargetNatural",
        "RS_Badge","RS_Trend","RS_5d","RS_10d","RS_20d","RS_Perf5d","SPY_Perf5d",
        "Earnings_Badge","Earnings_Date","Earnings_Risk",
        "Gap_Badge","Gap_Signal","Gap_Score","Gap_Support",
        "VOL_Badge","VOL_Signal","VOL_Ratio",
        "Entree","Stop","Target","RR_Ratio","Risque_Pct","Gain_Pct","RR_Badge",
        "TTM_Signal","DIV_Signal","EMA_Level","ADV_Score","ADV_Badge",
        "Pattern_Badge","Top_Pattern","Pattern_Score",
        "AI Score","AI Score Ajuste","AI Signal Ajuste","AI Reasons"
    ]
    cols_display = [c for c in cols_display if c in df_filtered.columns]
    st.dataframe(df_filtered[cols_display], use_container_width=True, height=400)

    # ── CLAUDE ──
    if use_claude and api_key:
        st.markdown("---")
        st.markdown("### 🤖 Analyse Claude IA — Top 5 convergents")
        top5 = report.head(5)
        for _, row in top5.iterrows():
            with st.spinner(f"Claude analyse {row['Ticker']}..."):
                analysis = claude_analysis(row, api_key, market_status)
            sig = row.get("AI Signal Ajuste", row.get("AI Signal",""))
            st.markdown(f"""
            <div class="ai-analysis-box">
                <span class="ticker-badge">{row['Ticker']}</span>
                <strong style="color:#00ff88">{sig}</strong>
                — Score {row.get('Score_Final','—')}/100
                &nbsp;|&nbsp; Conv {row.get('Conv_N','—')}/6
                &nbsp;|&nbsp; R/R {row.get('RR_Ratio','—')}:1
                <br><br>{analysis}
            </div>""", unsafe_allow_html=True)
            time.sleep(0.5)
    elif use_claude and not api_key:
        st.warning("⚠️ Entrez votre clé API Anthropic.")

    # ── EXPORT ──
    st.markdown("---")
    st.markdown("### 📥 Export")

    col_exp1, col_exp2, col_exp3 = st.columns(3)

    # ── Pense-bête trades ──
    with col_exp1:
        st.markdown("**📋 Pense-bête Lundi matin**")
        if not report.empty:
            try:
                pense_bete = _generate_cheat_sheet(report, market_status, regime)
                st.download_button(
                    "📋 Télécharger pense-bête",
                    data=pense_bete,
                    file_name=f"pense_bete_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.error(f"Erreur pense-bête: {e}")

    with col_exp2:
        st.markdown("**Export rapport Top trades**")
        if not report.empty:
            excel_report = to_excel(report[[c for c in cols_display if c in report.columns]])
            st.download_button(
                f"⬇️ Top {top_n} — Rapport convergence",
                data=excel_report,
                file_name=f"top{top_n}_convergence_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with col_exp3:
        st.markdown("**Export tableau complet**")
        excel_full = to_excel(df_filtered[[c for c in cols_display if c in df_filtered.columns]])
        st.download_button(
            "⬇️ Tableau complet",
            data=excel_full,
            file_name=f"screener_{regime}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

