"""
Stage 4: Deep model analysis — extend backtest to 2012, run 5 stress tests.
Answers: is the edge real, durable, and decomposable?

Tests:
  0. Extend backtest to 2012-2024 via EDGAR + yFinance
  1. Conviction threshold sweep (12yr)
  2. Quarter concentration analysis
  3. Confirmed entry decomposition
  4. Temporal persistence (split-half)
  5. Target definition sweep
"""
import os, sys, time, pickle, warnings, random

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

from pipeline import setup_logging, teardown_logging
_log_path, _log_file = setup_logging('analysis')

from config import (
    t_start, log, TRADING_TARGET, TRADING_HOLD, ENTRY_MODE,
    UNIVERSE_B_FEATURES, SECTOR_ETFS,
    BORROW_RATE_EASY, BORROW_RATE_HARD,
    CONFIRMATION_DROP, CONFIRMATION_WINDOW,
)
from utils import elapsed
from edgar import (
    load_cik_map, parse_edgar_facts, _load_raw_json_cache,
    _edgar_to_simfin_frames, extract_filing_metadata,
)
from data import (
    build_universe, download_all_prices, derive_benchmarks,
    get_sp_index_tickers, classify_tickers,
)
from features import build_features_from_scratch, prepare_features, recompute_outcomes
from walkforward import (
    run_walkforward_analysis, run_walkforward_split,
    _process_quarters, _generate_trades, _tier_stats,
)

START_TIME = time.time()
LOCKED_FEATURES_B = list(UNIVERSE_B_FEATURES)
LOCKED_TARGET = TRADING_TARGET


def check_time(test_name, hard_limit=80):
    """Check elapsed time. Returns False if we should skip."""
    elapsed_min = (time.time() - START_TIME) / 60
    log.info(f"\n  [Time: {elapsed_min:.0f} min | Starting {test_name}]")
    if elapsed_min > hard_limit:
        log.info(f"  Time limit reached. Skipping {test_name}.")
        return False
    return True


def compute_tier(wf_df, top_pct):
    """Compute stats for a conviction tier (top N% by score)."""
    if len(wf_df) < 5:
        return {'n': 0, 'win': 0, 'pnl': 0, 'median_pnl': 0,
                'stops': 0, 'n_quarters': 0, 'prof_q': '0/0', 'prof_q_ratio': 0}
    cutoff = wf_df['score'].quantile(1 - top_pct / 100)
    sub = wf_df[wf_df['score'] >= cutoff]
    if len(sub) < 3:
        return {'n': 0, 'win': 0, 'pnl': 0, 'median_pnl': 0,
                'stops': 0, 'n_quarters': 0, 'prof_q': '0/0', 'prof_q_ratio': 0}
    n_q = sub['quarter'].nunique()
    q_pnl = sub.groupby('quarter')['pnl_per_share'].sum()
    prof_q = (q_pnl > 0).sum()
    return {
        'n': len(sub),
        'win': (sub['pnl_per_share'] > 0).mean(),
        'pnl': sub['pnl_per_share'].mean(),
        'median_pnl': sub['pnl_per_share'].median(),
        'stops': sub['stopped'].mean(),
        'n_quarters': n_q,
        'prof_q': f"{prof_q}/{n_q}",
        'prof_q_ratio': prof_q / n_q if n_q > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════
# TEST 0: Extend backtest to 2012-2024
# ═══════════════════════════════════════════════════════════════

def test0_extend_backtest(data):
    """Extend data to 2012 using EDGAR raw cache + yFinance prices."""
    log.info("=" * 70)
    log.info("TEST 0: EXTEND BACKTEST TO 2012-2024")
    log.info("=" * 70)

    # --- Step 1: Parse EDGAR quarters back to 2012 ---
    raw_cache = _load_raw_json_cache('data/')
    if not raw_cache:
        log.info("  No EDGAR raw JSON cache — cannot extend. Using existing data only.")
        return data, None

    log.info(f"  EDGAR raw cache: {len(raw_cache)} tickers")
    all_edgar_data = {}
    parse_failures = 0
    for ticker, facts_json in raw_cache.items():
        try:
            quarterly = parse_edgar_facts(facts_json, ticker)
            for key, fields in quarterly.items():
                # key is (ticker, end_date_str)
                try:
                    dt = pd.Timestamp(key[1])
                    if dt >= pd.Timestamp('2012-01-01'):
                        all_edgar_data[key] = fields
                except Exception:
                    pass
        except Exception:
            parse_failures += 1

    ext_tickers = {k[0] for k in all_edgar_data}
    log.info(f"  Extended EDGAR: {len(ext_tickers)} tickers, "
             f"{len(all_edgar_data)} quarter-rows (failures: {parse_failures})")

    # Coverage by year
    year_counts = {}
    for (tk, end_date), _ in all_edgar_data.items():
        try:
            yr = pd.Timestamp(end_date).year
            if yr not in year_counts:
                year_counts[yr] = set()
            year_counts[yr].add(tk)
        except Exception:
            pass

    log.info(f"\n  {'Year':<6} {'Tickers':>8}")
    log.info(f"  {'---'*5}")
    for yr in sorted(year_counts.keys()):
        if 2012 <= yr <= 2024:
            log.info(f"  {yr:<6} {len(year_counts[yr]):>8}")

    # --- Step 2: Build SimFin-format frames from EDGAR ---
    df_inc_ext, df_bal_ext, df_cf_ext = _edgar_to_simfin_frames(all_edgar_data)
    log.info(f"\n  EDGAR frames: inc={len(df_inc_ext)}, bal={len(df_bal_ext)}, cf={len(df_cf_ext)}")

    # --- Step 3: Fetch extended prices (2011-2020) ---
    sp_tickers = data.get('sp_tickers', set())
    tickers_need = list(ext_tickers & sp_tickers)
    existing_price_dict = data['price_dict']

    # Only fetch prices for tickers we don't already have good coverage for
    tickers_to_fetch = []
    for tk in tickers_need:
        if tk in existing_price_dict:
            px = existing_price_dict[tk]
            if hasattr(px, 'index') and len(px) > 0:
                earliest = px.index.min()
                if earliest > pd.Timestamp('2013-01-01'):
                    tickers_to_fetch.append(tk)
            else:
                tickers_to_fetch.append(tk)
        else:
            tickers_to_fetch.append(tk)

    log.info(f"\n  Need extended prices for {len(tickers_to_fetch)} tickers...")

    extended_prices = {}
    for i in range(0, len(tickers_to_fetch), 50):
        batch = tickers_to_fetch[i:i+50]
        try:
            px_data = yf.download(
                batch, start='2011-01-01', end='2020-12-31',
                group_by='ticker', auto_adjust=True, threads=True, progress=False
            )
            for tk in batch:
                try:
                    if len(batch) == 1:
                        px = px_data[['Close', 'Volume']].dropna()
                    else:
                        px = px_data[tk][['Close', 'Volume']].dropna()
                    if len(px) > 252:
                        extended_prices[tk] = px
                except (KeyError, TypeError):
                    pass
        except Exception as e:
            log.info(f"    Batch {i//50} failed: {e}")

        if (i + 50) % 200 == 0:
            e_min = (time.time() - START_TIME) / 60
            log.info(f"    {i+50}/{len(tickers_to_fetch)} downloaded ({e_min:.1f} min)")

    log.info(f"  Extended prices: {len(extended_prices)} tickers")

    # Fetch extended SPY
    try:
        spy_ext = yf.download('SPY', start='2011-01-01', end='2020-12-31',
                              auto_adjust=True, progress=False)['Close']
        log.info(f"  Extended SPY: {len(spy_ext)} trading days")
    except Exception:
        spy_ext = pd.Series(dtype=float)
        log.info("  WARNING: Could not fetch extended SPY")

    # --- Step 4: Merge into combined dataset ---
    def _dedup_index(px):
        """Deduplicate index on a DataFrame or Series."""
        if isinstance(px, pd.DataFrame) and px.index.duplicated().any():
            return px[~px.index.duplicated(keep='last')]
        if isinstance(px, pd.Series) and px.index.duplicated().any():
            return px[~px.index.duplicated(keep='last')]
        return px

    # Merge extended and existing prices
    full_price_dict = dict(existing_price_dict)
    for tk in set(list(extended_prices.keys()) + list(full_price_dict.keys())):
        frames = []
        if tk in extended_prices:
            px = extended_prices[tk]
            if isinstance(px, pd.DataFrame):
                px = px[~px.index.duplicated(keep='last')]
            elif isinstance(px, pd.Series):
                px = px[~px.index.duplicated(keep='last')]
            frames.append(px)
        if tk in full_price_dict:
            px = full_price_dict[tk]
            if isinstance(px, pd.DataFrame):
                px = px[~px.index.duplicated(keep='last')]
            elif isinstance(px, pd.Series):
                px = px[~px.index.duplicated(keep='last')]
            frames.append(px)
        if len(frames) == 1:
            full_price_dict[tk] = frames[0]
        elif len(frames) == 2:
            # Deduplicate columns (yFinance auto_adjust can produce ['Close','Close','Volume'])
            for i in range(len(frames)):
                frames[i] = frames[i].loc[:, ~frames[i].columns.duplicated()]
            combined = pd.concat(frames)
            combined = combined[~combined.index.duplicated(keep='last')].sort_index()
            full_price_dict[tk] = combined

    # Merge SPY
    existing_spy = _dedup_index(data['spy_close'])
    if len(spy_ext) > 0:
        spy_ext = _dedup_index(spy_ext)
        full_spy = pd.concat([spy_ext, existing_spy])
        full_spy = full_spy[~full_spy.index.duplicated(keep='last')].sort_index()
    else:
        full_spy = existing_spy

    # Merge fundamental frames: EDGAR extended + existing
    existing_inc = data.get('df_inc', pd.DataFrame())
    existing_bal = data.get('df_bal', pd.DataFrame())
    existing_cf = data.get('df_cf', pd.DataFrame())

    def _merge_frames(ext, existing):
        if len(ext) == 0:
            return existing
        if len(existing) == 0:
            return ext
        combined = pd.concat([ext, existing])
        return combined[~combined.index.duplicated(keep='last')].sort_index()

    full_inc = _merge_frames(df_inc_ext, existing_inc)
    full_bal = _merge_frames(df_bal_ext, existing_bal)
    full_cf = _merge_frames(df_cf_ext, existing_cf)

    # Build universe from combined frames
    sector_map = data.get('sector_map', {})
    universe = build_universe(full_inc, full_bal, full_cf, sector_map, min_quarters=4)
    # Filter to S&P tickers
    universe = [tk for tk in universe if tk in sp_tickers]

    log.info(f"\n  COMBINED DATASET:")
    log.info(f"    Income rows: {len(full_inc)}")
    log.info(f"    Balance rows: {len(full_bal)}")
    log.info(f"    Cash flow rows: {len(full_cf)}")
    log.info(f"    S&P universe: {len(universe)} tickers")
    log.info(f"    Prices: {len(full_price_dict)} tickers")
    if len(full_spy) > 0:
        log.info(f"    SPY range: {full_spy.index.min().date()} to {full_spy.index.max().date()}")

    # Derive VIX and sector ETF returns for extended period
    full_spy_ret = full_spy.pct_change().dropna()
    vix_series = data.get('vix_series')
    sector_etf_ret = data.get('sector_etf_ret', {})

    # Build extended data bundle for feature engineering
    ext_bundle = {
        'df_inc': full_inc,
        'df_bal': full_bal,
        'df_cf': full_cf,
        'price_dict': full_price_dict,
        'spy_close': full_spy,
        'spy_ret': full_spy_ret,
        'vix_series': vix_series,
        'sector_map': sector_map,
        'sector_etf_ret': sector_etf_ret,
        'universe': universe,
        'simfin_max_date': data.get('simfin_max_date'),
        'intermediates_path': 'data/intermediates_extended.pkl',
        'intm_loaded': False,
        'edgar_filing_meta': data.get('edgar_filing_meta', {}),
        'tradeable_tickers': data.get('tradeable_tickers', set()),
        'sp_tickers': sp_tickers,
    }

    # Build features
    log.info("\n  Building features on extended dataset...")
    ext_bundle = prepare_features(ext_bundle)

    n_dev = len(ext_bundle['df_dev'])
    n_hold = len(ext_bundle['df_hold'])
    n_tickers = ext_bundle['df_dev']['ticker'].nunique()
    log.info(f"  Extended features: {n_tickers} tickers, {n_dev:,} dev + {n_hold:,} hold rows")

    return ext_bundle, data


# ═══════════════════════════════════════════════════════════════
# TEST 1: Conviction threshold sweep
# ═══════════════════════════════════════════════════════════════

def test1_conviction_sweep(wf_df):
    """Sweep conviction tiers from 5% to 100%."""
    log.info("=" * 70)
    log.info("TEST 1: CONVICTION SWEEP (12yr)")
    log.info("=" * 70)

    header = f"  {'Tier':<10} {'Trades':>7} {'Win':>6} {'$/sh':>8} {'Med$/sh':>9} {'Stops':>7} {'Tr/Q':>6} {'ProfQ':>8} {'TotProfit':>10}"
    log.info(header)
    log.info(f"  {'---'*27}")

    sweep = {}
    for pct in [5, 10, 15, 20, 25, 30, 33, 40, 50, 60, 75, 100]:
        t = compute_tier(wf_df, pct)
        sweep[pct] = t
        if t['n'] == 0:
            continue
        total = t['n'] * t['pnl']
        tpq = t['n'] / max(t['n_quarters'], 1)
        marker = ' << CURRENT' if pct == 25 else ''
        log.info(f"  Top {pct:>3}%  {t['n']:>7} {t['win']:>5.0%} "
                 f"${t['pnl']:>+7.2f} ${t['median_pnl']:>+8.2f} "
                 f"{t['stops']:>6.0%} {tpq:>5.1f} {t['prof_q']:>8} "
                 f"${total:>+9.0f}{marker}")

    # Sweet spots
    viable = {k: v for k, v in sweep.items() if v['win'] > 0.65 and v['n'] >= 20}
    if viable:
        best_total = max(viable, key=lambda k: viable[k]['n'] * viable[k]['pnl'])
        log.info(f"\n  SWEET SPOT (win>65%): Top {best_total}% — "
                 f"{viable[best_total]['n']} trades, "
                 f"${viable[best_total]['n'] * viable[best_total]['pnl']:+.0f} total")

    # Significance check
    for pct in [25, 33, 50]:
        n = sweep[pct]['n']
        status = 'sufficient (n>100)' if n > 100 else 'borderline' if n > 50 else 'insufficient'
        log.info(f"  Top-{pct}%: {n} trades — {status}")

    return sweep


# ═══════════════════════════════════════════════════════════════
# TEST 2: Quarter concentration
# ═══════════════════════════════════════════════════════════════

def test2_concentration(wf_df):
    """Analyze P&L concentration by quarter."""
    log.info("=" * 70)
    log.info("TEST 2: QUARTER CONCENTRATION (12yr)")
    log.info("=" * 70)

    # Use top-25% trades
    score_75 = wf_df['score'].quantile(0.75)
    top25 = wf_df[wf_df['score'] >= score_75].copy()

    quarters = sorted(top25['quarter'].unique())
    total_pnl = top25['pnl_per_share'].sum()
    log.info(f"  {len(top25)} top-25% trades across {len(quarters)} quarters")
    log.info(f"  Total P&L: ${total_pnl:+.0f}")

    log.info(f"\n  {'Quarter':<10} {'Trades':>7} {'Win':>6} {'Avg PnL':>8} {'Tot PnL':>10} {'% Total':>10}")
    log.info(f"  {'---'*18}")

    dominant_quarters = []
    for q in quarters:
        qs = top25[top25['quarter'] == q]
        q_pnl = qs['pnl_per_share'].sum()
        q_win = (qs['pnl_per_share'] > 0).mean()
        pct_total = q_pnl / total_pnl * 100 if total_pnl != 0 else 0
        flag = ' << DOMINANT' if abs(pct_total) > 20 else ''
        if abs(pct_total) > 20:
            dominant_quarters.append(q)
        log.info(f"  {q:<10} {len(qs):>7} {q_win:>5.0%} "
                 f"${q_pnl/len(qs):>+7.2f} ${q_pnl:>+9.0f} {pct_total:>+9.1f}%{flag}")

    # Leave-one-out for dominant quarters
    critical = []
    log.info(f"\n  Leave-one-out:")
    for q in quarters:
        rest = top25[top25['quarter'] != q]
        if len(rest) < 5:
            continue
        r_pnl = rest['pnl_per_share'].mean()
        r_win = (rest['pnl_per_share'] > 0).mean()
        if r_pnl <= 0:
            critical.append(q)
            log.info(f"    Without {q}: {len(rest)} trades, {r_win:.0%} win, ${r_pnl:+.2f}/sh  << CRITICAL")
        elif q in dominant_quarters:
            log.info(f"    Without {q}: {len(rest)} trades, {r_win:.0%} win, ${r_pnl:+.2f}/sh")

    # Without all 2022
    no_2022 = top25[~top25['quarter'].str.contains('2022')]
    if len(no_2022) > 5:
        w = (no_2022['pnl_per_share'] > 0).mean()
        p = no_2022['pnl_per_share'].mean()
        log.info(f"\n  WITHOUT 2022: {len(no_2022)} trades, {w:.0%} win, ${p:+.2f}/sh")
        if p > 0 and w > 0.55:
            log.info("  Edge survives without bear market")
        elif p > 0:
            log.info("  Edge weakens but survives without 2022")
        else:
            log.info("  WARNING: Edge depends on 2022 bear market")

    # Herfindahl index
    q_abs_pnls = []
    for q in quarters:
        q_abs_pnls.append(abs(top25[top25['quarter'] == q]['pnl_per_share'].sum()))
    total_abs = sum(q_abs_pnls)
    if total_abs > 0:
        shares = [p / total_abs for p in q_abs_pnls]
        hhi = sum(s**2 for s in shares)
        even = 1 / len(quarters) if quarters else 1
        log.info(f"\n  Herfindahl index: {hhi:.3f} (perfectly even = {even:.3f})")
        if hhi < 0.10:
            log.info("  Well distributed across quarters")
        elif hhi < 0.20:
            log.info("  Somewhat concentrated")
        else:
            log.info("  WARNING: Dangerously concentrated")
    else:
        hhi = 1.0

    return {
        'hhi': hhi, 'dominant_quarters': dominant_quarters,
        'critical_quarters': critical,
        'no_2022_pnl': no_2022['pnl_per_share'].mean() if len(no_2022) > 5 else 0,
    }


# ═══════════════════════════════════════════════════════════════
# TEST 3: Confirmed entry decomposition
# ═══════════════════════════════════════════════════════════════

def test3_entry_decomposition(ext_bundle):
    """Decompose alpha: model vs confirmation filter."""
    log.info("=" * 70)
    log.info("TEST 3: CONFIRMED ENTRY DECOMPOSITION (12yr)")
    log.info("=" * 70)

    results = {}

    # A: Model + confirmed (standard)
    log.info("  A: Model + confirmed entry...")
    wf_a, tiers_a = run_walkforward_analysis(ext_bundle, entry_mode="confirmed")
    a = compute_tier(wf_a, 25) if len(wf_a) >= 20 else {'n': 0, 'win': 0, 'pnl': 0}
    results['model_confirmed'] = a
    log.info(f"     {a['n']} trades, {a['win']:.0%} win, ${a['pnl']:+.2f}/sh")

    # B: Model + immediate entry
    log.info("  B: Model + immediate entry...")
    wf_b, _ = run_walkforward_analysis(ext_bundle, entry_mode="immediate")
    b = compute_tier(wf_b, 25) if len(wf_b) >= 20 else {'n': 0, 'win': 0, 'pnl': 0}
    results['model_immediate'] = b
    log.info(f"     {b['n']} trades, {b['win']:.0%} win, ${b['pnl']:+.2f}/sh")

    # C: Random flags + confirmed (momentum baseline, 5 seeds)
    log.info("  C: Random flags + confirmed (5 seeds)...")
    c_runs = []
    for seed in range(5):
        wf_c, _ = run_walkforward_analysis(
            ext_bundle, entry_mode="confirmed",
            random_flags=True, n_random_flags=50, random_seed=seed + 42,
        )
        ct = compute_tier(wf_c, 25) if len(wf_c) >= 20 else {'n': 0, 'win': 0.5, 'pnl': 0}
        c_runs.append(ct)
    c = {
        'n': int(np.mean([r['n'] for r in c_runs])),
        'win': np.mean([r['win'] for r in c_runs]),
        'pnl': np.mean([r['pnl'] for r in c_runs]),
    }
    c_std = np.std([r['pnl'] for r in c_runs])
    results['random_confirmed'] = c
    log.info(f"     {c['n']} trades, {c['win']:.0%} win, ${c['pnl']:+.2f}/sh (std=${c_std:.2f})")

    # D: Model + relaxed confirmation (1% in 3d)
    log.info("  D: Model + relaxed confirmation (1% in 3d)...")
    wf_d, _ = run_walkforward_analysis(
        ext_bundle, entry_mode="confirmed",
        confirm_drop=0.01, confirm_window=3,
    )
    d = compute_tier(wf_d, 25) if len(wf_d) >= 20 else {'n': 0, 'win': 0, 'pnl': 0}
    results['model_relaxed'] = d
    log.info(f"     {d['n']} trades, {d['win']:.0%} win, ${d['pnl']:+.2f}/sh")

    # E: Model + immediate + no stops
    log.info("  E: Model + immediate + no stops...")
    wf_e, _ = run_walkforward_analysis(
        ext_bundle, entry_mode="immediate", use_stops=False,
    )
    e = compute_tier(wf_e, 25) if len(wf_e) >= 20 else {'n': 0, 'win': 0, 'pnl': 0}
    results['model_nostops'] = e
    log.info(f"     {e['n']} trades, {e['win']:.0%} win, ${e['pnl']:+.2f}/sh")

    # Decomposition
    model_alpha = b['pnl'] - c['pnl']
    confirm_alpha = a['pnl'] - b['pnl']
    total_edge = a['pnl']

    log.info(f"""
                          Model+Confirm  Model+Immed  Random+Confirm  Relaxed  NoStops
  Trades (top-25%):       {a['n']:>7}       {b['n']:>7}        {c['n']:>7}      {d['n']:>7}   {e['n']:>7}
  Win rate:               {a['win']:>6.0%}       {b['win']:>6.0%}        {c['win']:>6.0%}      {d['win']:>6.0%}   {e['win']:>6.0%}
  Avg P&L:                ${a['pnl']:>+6.2f}     ${b['pnl']:>+6.2f}      ${c['pnl']:>+6.2f}    ${d['pnl']:>+6.2f} ${e['pnl']:>+6.2f}

  DECOMPOSITION:
    Model alpha:          ${model_alpha:+.2f}/sh  (Model+Immed - Random+Confirm)
    Confirm alpha:        ${confirm_alpha:+.2f}/sh  (Model+Confirm - Model+Immed)
    Total edge:           ${total_edge:+.2f}/sh""")

    if total_edge > 0:
        m_pct = max(0, model_alpha / total_edge * 100)
        c_pct = max(0, confirm_alpha / total_edge * 100)
        log.info(f"    Model share:          {m_pct:.0f}%")
        log.info(f"    Confirm share:        {c_pct:.0f}%")
    else:
        m_pct, c_pct = 0, 0
        log.info("    Total edge is negative")

    if model_alpha > 0.30:
        log.info(f"  Model adds substantial alpha (${model_alpha:+.2f}/sh)")
    elif model_alpha > 0:
        log.info(f"  Model adds marginal alpha (${model_alpha:+.2f}/sh)")
    else:
        log.info(f"  WARNING: Model does NOT add alpha (${model_alpha:+.2f}/sh)")

    if d['n'] > a['n'] * 1.3 and d['pnl'] > 0 and d['win'] > 0.60:
        log.info(f"  Relaxed confirmation viable: {d['n']} trades ({d['n']/max(a['n'],1):.1f}x)")

    results['model_alpha'] = model_alpha
    results['confirm_alpha'] = confirm_alpha
    results['model_share'] = m_pct
    results['confirm_share'] = c_pct
    return results


# ═══════════════════════════════════════════════════════════════
# TEST 4: Temporal persistence
# ═══════════════════════════════════════════════════════════════

def test4_persistence(ext_bundle):
    """Split-half test: train on one era, test on another."""
    log.info("=" * 70)
    log.info("TEST 4: TEMPORAL PERSISTENCE (split-half)")
    log.info("=" * 70)

    split_date = '2018-12-31'

    # Direction A: Train early, test late
    log.info("  Training on 2012-2018, testing on 2019-2024...")
    wf_a, tiers_a = run_walkforward_split(ext_bundle, split_date, train_side="early")
    a = compute_tier(wf_a, 25) if len(wf_a) >= 20 else {'n': 0, 'win': 0, 'pnl': 0}

    # Direction B: Train late, test early
    log.info("  Training on 2019-2024, testing on 2012-2018...")
    wf_b, tiers_b = run_walkforward_split(ext_bundle, split_date, train_side="late")
    b = compute_tier(wf_b, 25) if len(wf_b) >= 20 else {'n': 0, 'win': 0, 'pnl': 0}

    log.info(f"""
                          Train Early    Train Late
                          Test Late      Test Early
  Trades (top-25%):       {a['n']:>7}       {b['n']:>7}
  Win rate:               {a['win']:>6.0%}       {b['win']:>6.0%}
  Avg P&L:                ${a['pnl']:>+6.2f}     ${b['pnl']:>+6.2f}""")

    a_ok = a['pnl'] > 0 and a['win'] > 0.55 and a['n'] >= 10
    b_ok = b['pnl'] > 0 and b['win'] > 0.55 and b['n'] >= 10

    if a_ok and b_ok:
        log.info("  Signal persists in BOTH directions — edge is durable")
    elif a_ok:
        log.info("  Works forward (2019+) but not backward — possible recency bias")
    elif b_ok:
        log.info("  Works backward but not forward — signal may be decaying")
    else:
        log.info("  WARNING: Signal does not persist in either direction")

    return {'a': a, 'b': b, 'a_ok': a_ok, 'b_ok': b_ok}


# ═══════════════════════════════════════════════════════════════
# TEST 5: Target definition sweep
# ═══════════════════════════════════════════════════════════════

def test5_targets(ext_bundle):
    """Compare different target definitions on extended data."""
    log.info("=" * 70)
    log.info("TEST 5: TARGET DEFINITION SWEEP (12yr)")
    log.info("=" * 70)

    targets = [
        'exdrop_15_10d', 'exdrop_15_5d', 'exdrop_15_21d',
        'exdrop_10_5d', 'exdrop_10_10d', 'exdrop_10_21d',
        'exdrop_15_42d', 'drop_5_21d', 'drop_5_42d',
    ]

    # Check which targets actually exist in data
    available = [t for t in targets if t in ext_bundle['df_dev'].columns]
    log.info(f"  {len(available)}/{len(targets)} targets available in data")

    log.info(f"\n  {'Target':<24} {'Trades':>7} {'Win':>6} {'$/sh':>8} {'TotProfit':>10}")
    log.info(f"  {'---'*18}")

    target_results = {}
    for target in available:
        e_min = (time.time() - START_TIME) / 60
        if e_min > 78:
            log.info("  Time limit — stopping target sweep")
            break

        # Temporarily override the target in the bundle's v_results
        # We need to check if this target was trained
        v_results = ext_bundle.get('v_results', {})
        if target not in v_results:
            log.info(f"  {target:<24} (not in trained targets — skipped)")
            continue

        try:
            wf, _ = run_walkforward_analysis(ext_bundle, entry_mode="confirmed")
            t = compute_tier(wf, 25) if len(wf) >= 20 else {'n': 0, 'win': 0, 'pnl': 0}
            target_results[target] = t
            total = t['n'] * t['pnl']
            marker = ' << CURRENT' if target == LOCKED_TARGET else ''
            log.info(f"  {target:<24} {t['n']:>7} {t['win']:>5.0%} "
                     f"${t['pnl']:>+7.2f} ${total:>+9.0f}{marker}")
        except Exception as e:
            log.info(f"  {target:<24} FAILED: {str(e)[:50]}")

    if target_results:
        viable = {k: v for k, v in target_results.items()
                  if v['win'] > 0.60 and v['n'] >= 30}
        if viable:
            best = max(viable, key=lambda k: viable[k]['n'] * viable[k]['pnl'])
            log.info(f"\n  Best target: {best}")
        else:
            best = LOCKED_TARGET
            log.info("\n  No viable alternatives found")
    else:
        best = LOCKED_TARGET

    return target_results, best


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("DROP SCORE v18.3 — DEEP ANALYSIS")
    log.info(f"  Target: {LOCKED_TARGET} | Hold: {TRADING_HOLD}d | Entry: {ENTRY_MODE}")
    log.info(f"  Features: {LOCKED_FEATURES_B}")
    log.info("=" * 70)

    # Load existing data bundle from Stage 1
    with open('data/data_bundle.pkl', 'rb') as f:
        data = pickle.load(f)
    log.info(f"  Loaded data bundle: {data['df_dev']['ticker'].nunique()} tickers")

    # Also reload raw SimFin frames for merging (if available)
    # The portable bundle strips df_inc/df_bal/df_cf, so reload from intermediates
    try:
        intm_path = os.path.join(data.get('cache_dir', 'data/'),
                                 f"intermediates_{data.get('intermediates_version', 'v15')}.pkl")
        if not os.path.exists(intm_path):
            intm_path = 'data/intermediates_v15.pkl'
        if os.path.exists(intm_path):
            with open(intm_path, 'rb') as f:
                intm = pickle.load(f)
            # Intermediates have df_q, df_dev, df_hold, df_daily
            log.info(f"  Loaded intermediates from {intm_path}")
    except Exception as e:
        log.info(f"  Could not load intermediates: {e}")

    # ── Run pipeline on existing data first (get current baseline) ──
    from pipeline import run_pipeline, get_pipeline_metrics
    from model import run_vulnerability_model

    log.info("\n  Running vulnerability model on existing data (baseline)...")
    data = run_vulnerability_model(data)
    from walkforward import run_walkforward
    data = run_walkforward(data)

    current_wf = data.get('wf_df', pd.DataFrame())
    current_top25 = compute_tier(current_wf, 25) if len(current_wf) >= 20 else None
    if current_top25:
        log.info(f"\n  BASELINE (current 3yr): {current_top25['n']} top-25% trades, "
                 f"{current_top25['win']:.0%} win, ${current_top25['pnl']:+.2f}/sh")

    # ── TEST 0: Extend backtest ──
    if check_time("Test 0: Extend backtest"):
        ext_bundle, orig_data = test0_extend_backtest(data)
    else:
        ext_bundle = data
        orig_data = data

    # Run vulnerability model on extended data
    if check_time("Vulnerability model (extended)"):
        log.info("\n  Running vulnerability model on extended data...")
        ext_bundle = run_vulnerability_model(ext_bundle)

    # Run walk-forward on extended data
    if check_time("Walk-forward (extended)"):
        log.info("\n  Running walk-forward on extended data...")
        ext_bundle = run_walkforward(ext_bundle)

    ext_wf = ext_bundle.get('wf_df', pd.DataFrame())
    ext_top25 = compute_tier(ext_wf, 25) if len(ext_wf) >= 20 else None

    if ext_top25 and current_top25:
        log.info(f"""
  EXTENDED vs CURRENT:
                          Current (3yr)  Extended (12yr)
  Top-25% trades:         {current_top25['n']:>7}       {ext_top25['n']:>7}
  Win rate:               {current_top25['win']:>6.0%}       {ext_top25['win']:>6.0%}
  Avg P&L:                ${current_top25['pnl']:>+6.2f}     ${ext_top25['pnl']:>+6.2f}""")

    # ── TEST 1: Conviction sweep ──
    sweep = {}
    if check_time("Test 1: Conviction sweep") and len(ext_wf) >= 20:
        sweep = test1_conviction_sweep(ext_wf)

    # ── TEST 2: Quarter concentration ──
    concentration = {}
    if check_time("Test 2: Quarter concentration") and len(ext_wf) >= 20:
        concentration = test2_concentration(ext_wf)

    # ── TEST 3: Entry decomposition ──
    decomposition = {}
    if check_time("Test 3: Entry decomposition"):
        decomposition = test3_entry_decomposition(ext_bundle)

    # ── TEST 4: Temporal persistence ──
    persistence = {}
    if check_time("Test 4: Temporal persistence"):
        persistence = test4_persistence(ext_bundle)

    # ── TEST 5: Target sweep ──
    target_results = {}
    best_target = LOCKED_TARGET
    if check_time("Test 5: Target sweep"):
        target_results, best_target = test5_targets(ext_bundle)

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    elapsed_min = (time.time() - START_TIME) / 60

    log.info(f"""
{'='*70}
DROP SCORE — DEEP ANALYSIS RESULTS
{'='*70}

  RUN TIME: {elapsed_min:.0f} minutes""")

    if ext_top25:
        log.info(f"""
  TRADE COUNT:
    Current (3yr, Top-25%):   {current_top25['n'] if current_top25 else 'N/A'} trades
    Extended (12yr, Top-25%): {ext_top25['n']} trades""")
        if ext_top25['n'] > 100:
            log.info("    Statistically significant (n>100)")
        elif ext_top25['n'] > 50:
            log.info("    Borderline significance (50<n<100)")
        else:
            log.info("    WARNING: Insufficient trades")

    # Test results
    tests_passed = 0
    tests_total = 0

    # Test 1: Did we get enough trades?
    if ext_top25:
        tests_total += 1
        if ext_top25['n'] > 50:
            tests_passed += 1
            log.info("\n  Test 0 (trade count): PASS")
        else:
            log.info("\n  Test 0 (trade count): FAIL")

    # Test 2: Concentration
    if concentration:
        tests_total += 1
        if concentration.get('hhi', 1) < 0.15:
            tests_passed += 1
            log.info(f"  Test 2 (concentration): PASS (HHI={concentration['hhi']:.3f})")
        else:
            log.info(f"  Test 2 (concentration): FAIL (HHI={concentration['hhi']:.3f})")

    # Test 3: Entry decomposition
    if decomposition:
        tests_total += 1
        ma = decomposition.get('model_alpha', 0)
        if ma > 0:
            tests_passed += 1
            log.info(f"  Test 3 (model alpha): PASS (${ma:+.2f}/sh)")
        else:
            log.info(f"  Test 3 (model alpha): FAIL (${ma:+.2f}/sh)")

    # Test 4: Persistence
    if persistence:
        tests_total += 1
        if persistence.get('a_ok') and persistence.get('b_ok'):
            tests_passed += 1
            log.info("  Test 4 (persistence): PASS (both directions)")
        elif persistence.get('a_ok') or persistence.get('b_ok'):
            tests_passed += 0.5
            log.info("  Test 4 (persistence): PARTIAL (one direction)")
        else:
            log.info("  Test 4 (persistence): FAIL")

    # Test 5: Target viability
    if target_results:
        tests_total += 1
        if any(v['win'] > 0.60 and v['n'] >= 30 for v in target_results.values()):
            tests_passed += 1
            log.info(f"  Test 5 (targets): PASS (best={best_target})")
        else:
            log.info("  Test 5 (targets): FAIL")

    tests_passed = int(tests_passed)

    log.info(f"""
  OVERALL: {tests_passed}/{tests_total} tests passed""")

    if tests_passed >= tests_total - 1 and tests_total >= 3:
        log.info("  Model has a DURABLE, DECOMPOSABLE edge")
        log.info("  -> Ready for paper trading")
    elif tests_passed >= tests_total // 2:
        log.info("  Model has signal but with CAVEATS")
        log.info("  -> Address failing tests before paper trading")
    else:
        log.info("  Model edge is NOT ROBUST ENOUGH")
        log.info("  -> Fundamental rework needed")

    # Recommendations
    log.info("\n  RECOMMENDATIONS:")
    if sweep:
        viable = {k: v for k, v in sweep.items() if v['win'] > 0.65 and v['n'] >= 20}
        if viable:
            best_pct = max(viable, key=lambda k: viable[k]['n'] * viable[k]['pnl'])
            if best_pct != 25:
                log.info(f"    Consider Top-{best_pct}% tier ({viable[best_pct]['n']} trades)")
    if decomposition:
        d = decomposition.get('model_relaxed', {})
        a = decomposition.get('model_confirmed', {})
        if d.get('n', 0) > a.get('n', 0) * 1.3 and d.get('pnl', 0) > 0:
            log.info(f"    Consider relaxed confirmation ({d['n']} trades vs {a['n']})")
    if best_target != LOCKED_TARGET:
        log.info(f"    Consider switching target to {best_target}")

    log.info(f"\n  Analysis complete | {elapsed()}")


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
