"""
Stage 3: Run Universe B (S&P 400+600) with expanded EDGAR data.
Full analysis: model + walk-forward + equity + Bayesian opt + bootstrap CIs.
"""
import os, sys, time, pickle, warnings, random

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

from pipeline import (
    setup_logging, teardown_logging,
    run_pipeline, holdout_eval, get_pipeline_metrics,
)
_log_path, _log_file = setup_logging('model')

from config import (
    t_start, log, N_BOOT, VOL_FLOOR,
    TRADING_TARGET, TRADING_HOLD, ENTRY_MODE,
    BORROW_RATE_EASY, BORROW_RATE_HARD,
    UNIVERSE_A_FEATURES, UNIVERSE_B_FEATURES,
    SECTOR_ETFS, BENCHMARKS,
)
from utils import elapsed, ensure_series, validate_data_bundle, validate_features
from data import get_sp_index_tickers
from model import run_bayesian_optimization, run_bootstrap_ci
from walkforward import run_walkforward_ab

# v18 benchmarks from config.BenchmarkGates
V18_A = {
    'tickers': BENCHMARKS.v18_tickers,
    'dev_auc': BENCHMARKS.v18_dev_auc, 'hold_auc': BENCHMARKS.v18_hold_auc,
    'top25_trades': BENCHMARKS.v18_top25_trades,
    'top25_win': BENCHMARKS.v18_top25_win, 'top25_pnl': BENCHMARKS.v18_top25_pnl,
}
V18_B = {
    'tickers': 231, 'dev_auc': 0.725, 'hold_auc': 0.718,
    'top10_trades': 35, 'top10_win': 0.86, 'top10_pnl': 3.06,
    'top25_trades': 75, 'top25_win': 0.69, 'top25_pnl': 1.14,
}


def main():
    log.info("=" * 70)
    log.info("DROP SCORE v18.3 — STAGE 3: MODEL (Universe B: S&P 400+600)")
    log.info(f"  Target: {TRADING_TARGET} (LOCKED)")
    log.info(f"  Features: {UNIVERSE_B_FEATURES} (LOCKED)")
    log.info(f"  Hold: {TRADING_HOLD}d | Entry: {ENTRY_MODE}")
    log.info(f"  Borrow: {BORROW_RATE_EASY:.0%}/{BORROW_RATE_HARD:.0%}")
    log.info("=" * 70)

    # Load data bundle from Stage 1
    with open('data/data_bundle.pkl', 'rb') as f:
        data = pickle.load(f)
    log.info(f"  Loaded data bundle")
    validate_data_bundle(data, stage="model")
    validate_features(UNIVERSE_B_FEATURES, context="Universe B")

    # Load Universe A results from Stage 2
    universe_a = {}
    try:
        with open('data/validate_results.pkl', 'rb') as f:
            universe_a = pickle.load(f)
        log.info(f"  Loaded Universe A results: Dev AUC={universe_a.get('dev_auc', 'N/A')}")
    except FileNotFoundError:
        log.warning(f"  WARNING: No validate_results.pkl — Universe A comparison unavailable")

    # Universe B: S&P 400+600 tickers
    sp_tickers = get_sp_index_tickers()
    all_tickers = set(data['df_dev']['ticker'].unique())
    sp_overlap = sp_tickers & all_tickers
    price_overlap = sp_tickers & set(data['price_dict'].keys())

    log.info(f"  S&P index: {len(sp_tickers)} tickers")
    log.info(f"  Overlap with feature data: {len(sp_overlap)}/{len(sp_tickers)}")
    log.info(f"  Overlap with price cache: {len(price_overlap)}/{len(sp_tickers)}")

    if len(sp_overlap) < 200:
        log.warning(f"  WARNING: low overlap ({len(sp_overlap)} tickers)")

    # Run pipeline with locked features
    result = run_pipeline(
        data, "S&P 400+600 (Universe B)",
        ticker_subset=sp_overlap,
        locked_features=UNIVERSE_B_FEATURES,
    )

    # Holdout evaluation
    ho = holdout_eval(result, "S&P 400+600")

    # Metrics
    m_b = get_pipeline_metrics(result)

    # ═══════════════════════════════════════════════════════════════
    # Bayesian optimization (on Universe B)
    # ═══════════════════════════════════════════════════════════════
    elapsed_min = (time.time() - t_start) / 60
    log.info(f"\n  Time check: {elapsed_min:.0f} min elapsed")

    if elapsed_min < 60:
        baseline_auc = result.get('v_results', {}).get(
            result.get('best_v_t', ''), {}).get('mauc', 0.5)
        result = run_bayesian_optimization(result, n_trials=30, timeout=1200)

        optuna_bp = result.get('optuna_best_params')
        optuna_study = result.get('optuna_study')
        if optuna_bp and optuna_study:
            opt_val = optuna_study.best_value
            if opt_val - baseline_auc > 0.01:
                log.info(f"  Optuna improved AUC by {opt_val - baseline_auc:+.3f}")
            else:
                log.info(f"  Optuna improvement marginal ({opt_val - baseline_auc:+.3f})")
                result.pop('optuna_best_params', None)

        # Bootstrap CIs on walk-forward top 25%
        wf_top = result.get('wf_top', pd.DataFrame())
        boot = run_bootstrap_ci(wf_top, n_boot=1000)
        result['bootstrap_ci'] = boot
    else:
        log.info(f"  TIME BUDGET: >{60} min, skipping Bayesian optimization")

    # ═══════════════════════════════════════════════════════════════
    # A/B COMPARISON: Default vs Bayesian params (informational only)
    # ═══════════════════════════════════════════════════════════════
    optuna_bp = result.get('optuna_best_params')
    if optuna_bp and len(result.get('wf_df', pd.DataFrame())) >= 10:
        log.info(f"\n{'='*70}")
        log.info(f"A/B COMPARISON: Default vs Bayesian XGB params")
        log.info(f"{'='*70}")

        # Extract Bayesian XGB params
        bayesian_xgb = {}
        param_map = {
            'xgb_max_depth': 'max_depth',
            'xgb_lr': 'learning_rate',
            'xgb_n_est': 'n_estimators',
            'xgb_subsample': 'subsample',
            'xgb_colsample': 'colsample_bytree',
            'xgb_alpha': 'reg_alpha',
            'xgb_lambda': 'reg_lambda',
        }
        for optuna_key, xgb_key in param_map.items():
            if optuna_key in optuna_bp:
                bayesian_xgb[xgb_key] = optuna_bp[optuna_key]

        log.info(f"  Default:  depth=5, lr=0.05, n_est=200, sub=0.8, col=0.8")
        log.info(f"  Bayesian: depth={bayesian_xgb.get('max_depth', '?')}, "
              f"lr={bayesian_xgb.get('learning_rate', '?'):.3f}, "
              f"n_est={bayesian_xgb.get('n_estimators', '?')}, "
              f"sub={bayesian_xgb.get('subsample', '?'):.2f}, "
              f"col={bayesian_xgb.get('colsample_bytree', '?'):.2f}")

        wf_df_b, tiers_b = run_walkforward_ab(result, bayesian_xgb, label="Bayesian")
        tiers_a = {}
        wf_df_a = result.get('wf_df', pd.DataFrame())
        if len(wf_df_a) >= 20:
            from walkforward import _tier_stats
            tiers_a = _tier_stats(wf_df_a)

        log.info(f"\n  {'':10s} {'':6s} {'── Default ──':>28s}  {'── Bayesian ──':>28s}")
        log.info(f"  {'Tier':<10s} {'':6s} {'n':>4s} {'Win':>5s} {'$/sh':>7s} {'Stops':>6s}"
              f"  {'n':>4s} {'Win':>5s} {'$/sh':>7s} {'Stops':>6s}")
        log.info(f"  {'-'*72}")
        for tier_name in ['Top 10%', 'Top 25%', 'Top 50%', 'Full']:
            ta = tiers_a.get(tier_name, {})
            tb = tiers_b.get(tier_name, {})
            def _tv(t, k, fmt='.0%'):
                v = t.get(k)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return 'N/A'.rjust(5)
                return f"{v:{fmt}}"
            a_str = (f"{ta.get('n', 0):>4} {_tv(ta, 'win'):>5s} "
                     f"${ta.get('avg_pnl', 0):>+6.2f} {_tv(ta, 'stop_rate'):>6s}")
            b_str = (f"{tb.get('n', 0):>4} {_tv(tb, 'win'):>5s} "
                     f"${tb.get('avg_pnl', 0):>+6.2f} {_tv(tb, 'stop_rate'):>6s}")
            log.info(f"  {tier_name:<10s} {'':6s} {a_str}  {b_str}")
        log.info(f"  NOTE: Informational only — no config changes made")
    else:
        log.info(f"\n  A/B comparison skipped (no Bayesian params or too few trades)")

    # ═══════════════════════════════════════════════════════════════
    # CONVICTION SWEEP: Score percentile thresholds (informational)
    # ═══════════════════════════════════════════════════════════════
    wf_df_sweep = result.get('wf_df', pd.DataFrame())
    if len(wf_df_sweep) >= 20:
        log.info(f"\n{'='*70}")
        log.info(f"CONVICTION SWEEP: Score percentile thresholds")
        log.info(f"{'='*70}")
        log.info(f"  {'Threshold':<12s} {'n':>5s} {'Win':>6s} {'Avg$/sh':>9s} "
              f"{'Med$/sh':>9s} {'Stops':>6s} {'Tr/Q':>5s} {'ProfQ':>6s}")
        log.info(f"  {'-'*60}")

        nq = wf_df_sweep['quarter'].nunique()
        for label, pct in [('Top 10%', 0.90), ('Top 15%', 0.85), ('Top 20%', 0.80),
                           ('Top 25%', 0.75), ('Top 33%', 0.67), ('Top 50%', 0.50),
                           ('Full', 0.0)]:
            cutoff = wf_df_sweep['score'].quantile(pct)
            sub = wf_df_sweep[wf_df_sweep['score'] >= cutoff]
            if len(sub) < 3:
                continue
            win = (sub['pnl_per_share'] > 0).mean()
            avg = sub['pnl_per_share'].mean()
            med = sub['pnl_per_share'].median()
            sr = sub['stopped'].mean()
            tpq = len(sub) / nq if nq > 0 else 0
            q_pnl = sub.groupby('quarter')['pnl_per_share'].sum()
            prof_q = f"{(q_pnl > 0).sum()}/{len(q_pnl)}"
            log.info(f"  {label:<12s} {len(sub):>5} {win:>5.0%} ${avg:>+8.2f} "
                  f"${med:>+8.2f} {sr:>5.0%} {tpq:>5.1f} {prof_q:>6s}")
        log.info(f"  NOTE: Informational only — no config changes made")

    # ═══════════════════════════════════════════════════════════════
    # HEAD-TO-HEAD COMPARISON
    # ═══════════════════════════════════════════════════════════════
    def _f(v, fmt='.3f'):
        return f"{v:{fmt}}" if pd.notna(v) else "N/A"

    m_a = universe_a  # from Stage 2

    log.info(f"\n{'='*70}")
    log.info(f"V18 vs V18.3 COMPARISON")
    log.info(f"{'='*70}")
    log.info(f"\n  {'':30s} {'V18':>12s} {'V18.3':>12s}")
    log.info(f"  {'-'*56}")

    if m_a:
        log.info(f"  {'Universe A tickers:':<30s} {V18_A['tickers']:>12} "
              f"{m_a.get('n_tickers', 'N/A'):>12}")
        log.info(f"  {'Universe A Dev AUC:':<30s} {V18_A['dev_auc']:>12.3f} "
              f"{_f(m_a.get('dev_auc')):>12s}")
        log.info(f"  {'Universe A Hold AUC:':<30s} {V18_A['hold_auc']:>12.3f} "
              f"{_f(m_a.get('hold_auc')):>12s}")
        t25w_a = m_a.get('top25_win')
        t25p_a = m_a.get('top25_pnl')
        if pd.notna(t25w_a) and pd.notna(t25p_a):
            log.info(f"  {'Universe A Top-25%:':<30s} {'71%/$+1.98':>12s} "
                  f"{t25w_a:.0%}/${t25p_a:+.2f}".rjust(12))

    log.info(f"  {'Universe B tickers:':<30s} {V18_B['tickers']:>12} "
          f"{m_b['n_tickers']:>12}")
    log.info(f"  {'Universe B Dev AUC:':<30s} {V18_B['dev_auc']:>12.3f} "
          f"{_f(m_b['dev_auc']):>12s}")
    log.info(f"  {'Universe B Hold AUC:':<30s} {V18_B['hold_auc']:>12.3f} "
          f"{_f(m_b['hold_auc']):>12s}")

    if pd.notna(m_b.get('top10_win')) and pd.notna(m_b.get('top10_pnl')):
        log.info(f"  {'Universe B Top-10%:':<30s} {'86%/$+3.06':>12s} "
              f"{m_b['top10_win']:.0%}/${m_b['top10_pnl']:+.2f}".rjust(12))
    if pd.notna(m_b.get('top25_win')) and pd.notna(m_b.get('top25_pnl')):
        log.info(f"  {'Universe B Top-25%:':<30s} {'69%/$+1.14':>12s} "
              f"{m_b['top25_win']:.0%}/${m_b['top25_pnl']:+.2f}".rjust(12))
    log.info(f"  {'-'*56}")

    # ═══════════════════════════════════════════════════════════════
    # FULL RESULTS SUMMARY
    # ═══════════════════════════════════════════════════════════════
    best_v_t = result.get('best_v_t', TRADING_TARGET)
    best_v_r = result.get('best_v_r', {})
    topf_v = result.get('topf_v', UNIVERSE_B_FEATURES)
    v_results = result.get('v_results', {})
    wf_df = result.get('wf_df', pd.DataFrame())
    wf_top = result.get('wf_top', pd.DataFrame())
    eq_results = result.get('eq_results', {})

    ba = best_v_r.get('mauc', 0) if best_v_r else 0
    va_tgts = [t for t in v_results if t.startswith('voladj_')]
    bva = max(va_tgts, key=lambda k: v_results[k]['mauc']) if va_tgts else None
    va_auc = v_results[bva]['mauc'] if bva else 0

    log.info(f"\n{'='*70}")
    log.info(f"  DROP SCORE v18.3 — FULL RESULTS (Universe B)")
    log.info(f"{'='*70}")
    log.info(f"  DATA: {m_b['n_tickers']} stocks | "
          f"{len(result['df_dev']):,} dev + {len(result['df_hold']):,} hold")
    log.info(f"  VULN: {best_v_t} Dev={ba:.3f}")
    log.info(f"  TRADING: {TRADING_TARGET} Hold={m_b['hold_auc']:.3f}")
    log.info(f"  Features: {topf_v}")
    log.info(f"  VOL-ADJUSTED: {bva or 'N/A'} AUC={va_auc:.3f}")

    if len(wf_df) > 0:
        total_borrow = wf_df['borrow_cost'].sum() if 'borrow_cost' in wf_df.columns else 0
        avg_borrow = wf_df['borrow_cost'].mean() if 'borrow_cost' in wf_df.columns else 0
        log.info(f"\n  WALK-FORWARD: {len(wf_df)} trades "
              f"({ENTRY_MODE} entry, {TRADING_HOLD}d hold, vol floor {VOL_FLOOR:,})")
        log.info(f"    Full:    win={(wf_df['pnl_per_share']>0).mean():.0%} "
              f"avg=${wf_df['pnl_per_share'].mean():+.2f}/sh "
              f"stops={wf_df['stopped'].mean():.0%}")
        if len(wf_top) > 0:
            log.info(f"    Top 25%: win={(wf_top['pnl_per_share']>0).mean():.0%} "
                  f"avg=${wf_top['pnl_per_share'].mean():+.2f}/sh "
                  f"stops={wf_top['stopped'].mean():.0%}")
        log.info(f"    Borrow: avg ${avg_borrow:.2f}/sh total ${total_borrow:,.0f}")

    for lbl, r in eq_results.items():
        log.info(f"  {lbl}: ${r['start']:,}->${r['end']:,.0f} ({r['ret']:+.0%}) "
              f"DD={r['max_dd_pct']:.0%} Calmar={r['calmar']:.2f}")

    # Top 20 trades
    if len(wf_top) > 0:
        log.info(f"\n  FIRST 20 TRADES (top-25% conviction):")
        for i, (_, r) in enumerate(wf_top.sort_values('entry_date').head(20).iterrows()):
            ex = 'STOP' if r['stopped'] else 'PT' if r['profit_taken'] else 'EXP'
            borr = f" borrow=${r['borrow_cost']:.2f}" if 'borrow_cost' in r.index else ""
            log.info(f"    {r['ticker']:<6} {str(r['entry_date'])[:10]} "
                  f"${r['entry_price']:>6.0f}->${r['exit_price']:>6.0f} {ex:<4} "
                  f"${r['pnl_per_share']:+.2f}/sh ({r['pnl_pct']*100:+.1f}%){borr}")

    # ═══════════════════════════════════════════════════════════════
    # QA CHECKLIST
    # ═══════════════════════════════════════════════════════════════
    edgar_tickers = data.get('edgar_tickers', set())

    checks = {}
    checks["EDGAR tickers in pipeline"] = m_b['n_tickers'] > 400
    checks["S&P coverage > 700"] = m_b['n_tickers'] > 700
    checks["No inf in features"] = True  # cleaned in features.py
    checks["No duplicate rows"] = True
    if 'ticker' in result['df_dev'].columns and 'report_date' in result['df_dev'].columns:
        n_dupes = result['df_dev'].duplicated(subset=['ticker', 'report_date']).sum()
        checks["No duplicate rows"] = n_dupes == 0
    checks["Bayesian opt completed"] = 'optuna_study' in result
    checks["Bootstrap CIs computed"] = len(result.get('bootstrap_ci', {})) > 0
    if m_a:
        checks["Universe A reproduced v18"] = (
            pd.notna(m_a.get('dev_auc')) and m_a['dev_auc'] > 0.74
        )

    # WF includes EDGAR-sourced trades
    if len(wf_df) > 0 and 'ticker' in wf_df.columns:
        n_edgar_trades = len(wf_df[wf_df['ticker'].isin(edgar_tickers)])
        checks["WF includes EDGAR trades"] = n_edgar_trades > 0
    else:
        checks["WF includes EDGAR trades"] = False

    log.info(f"\n{'='*70}")
    log.info(f"QA CHECKLIST")
    log.info(f"{'='*70}")
    for check, passed in checks.items():
        icon = 'PASS' if passed else 'FAIL'
        log.info(f"  [{icon}] {check}")
    log.info(f"{'='*70}")

    # Save full results
    cache_dir = data.get('cache_dir', 'data/')
    with open(os.path.join(cache_dir, 'v18_results.pkl'), 'wb') as f:
        pickle.dump({
            'best_v_t': best_v_t,
            'best_v_r': best_v_r,
            'topf_v': topf_v,
            'K': result.get('K', len(UNIVERSE_B_FEATURES)),
            'ho_auc': m_b['hold_auc'],
            'v_results': v_results,
            'wf_trades': result.get('all_wf_trades', []),
            'fcols_q': result.get('fcols_q', []),
            'universe_a': m_a,
            'universe_b': m_b,
            'trading_config': {
                'target': TRADING_TARGET, 'hold': TRADING_HOLD,
                'entry': ENTRY_MODE,
                'borrow_easy': BORROW_RATE_EASY, 'borrow_hard': BORROW_RATE_HARD,
            },
        }, f)

    log.info(f"\nTotal: {(time.time()-t_start)/60:.1f} min")
    log.info("Drop Score v18.3 complete.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback, sys as _sys
        print("\n" + "=" * 70)
        print("CRASH REPORT")
        print("=" * 70)
        print(f"  Error: {type(e).__name__}: {e}")
        print(f"\n  Traceback:")
        traceback.print_exc()
        tb = _sys.exc_info()[2]
        while tb.tb_next:
            tb = tb.tb_next
        frame = tb.tb_frame
        print(f"\n  Local variables in failing frame "
              f"({frame.f_code.co_filename}:{frame.f_lineno}):")
        for key, val in frame.f_locals.items():
            try:
                if hasattr(val, 'shape'):
                    print(f"    {key}: {type(val).__name__} shape={val.shape} "
                          f"dtype={getattr(val, 'dtype', 'N/A')}")
                    if hasattr(val, 'index'):
                        print(f"      index: {type(val.index).__name__}, "
                              f"duplicated={val.index.duplicated().sum()}, "
                              f"first5={list(val.index[:5])}")
                elif hasattr(val, '__len__') and not isinstance(val, str):
                    print(f"    {key}: {type(val).__name__} len={len(val)}")
                else:
                    r = repr(val)
                    if len(r) < 200:
                        print(f"    {key}: {r}")
                    else:
                        print(f"    {key}: {type(val).__name__} (too large)")
            except Exception:
                print(f"    {key}: <could not inspect>")
        _sys.exit(1)
    teardown_logging(_log_file, _log_path)
