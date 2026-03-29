"""
Walk-forward backtest locked to TRADING_TARGET configuration,
confirmation entry, rule-based filters, sector cap, regime filter,
volume-based borrow costs, and conviction tiers.
"""
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from config import (
    log, VOL_FLOOR, SECTOR_CAP, REGIME_SPY_MAX, REGIME_VIX_MIN,
    SLIPPAGE, STOP_LOSS, PROFIT_TARGET, TRAILING_STOP,
    SKIP_RET5D_DOWN, SKIP_VOL_PCT, ENTRY_DELAY,
    TRADING_TARGET, TRADING_HOLD, ENTRY_MODE,
    CONFIRMATION_DROP, CONFIRMATION_WINDOW,
    BORROW_RATE_EASY, BORROW_RATE_HARD,
)
from utils import clean_X, elapsed, to_scalar, ensure_series


def _process_quarters(data_bundle, wf_tgt, xgb_override=None):
    """Train WF-internal models per quarter and return scored picks.

    Args:
        xgb_override: dict of XGB params to override defaults (for A/B comparison).
                      Keys: n_estimators, max_depth, learning_rate, subsample, etc.
    """
    df_dev = data_bundle['df_dev']
    fcols_q = data_bundle['fcols_q']
    fill_meds_q = data_bundle['fill_meds_q']
    K = data_bundle['K']
    spy_close = data_bundle['spy_close']
    vix_series = data_bundle['vix_series']

    quarters = sorted(df_dev['report_date'].dt.to_period('Q').unique())
    scored_quarters = []

    for qi in range(3, len(quarters)):
        test_q = quarters[qi]
        train_end = test_q.start_time
        wf_train = df_dev[df_dev['report_date'] < train_end]
        wf_test = df_dev[
            (df_dev['report_date'] >= test_q.start_time)
            & (df_dev['report_date'] <= test_q.end_time)
        ]
        if len(wf_train) < 200 or len(wf_test) < 20:
            continue
        if wf_tgt not in wf_train.columns or wf_train[wf_tgt].sum() < 30:
            continue

        # Regime filter at quarter start
        skip = None
        if spy_close is not None:
            sa = ensure_series(spy_close.loc[:test_q.start_time + pd.Timedelta(days=10)])
            if len(sa) > 21:
                s21 = (to_scalar(sa.iloc[-1]) - to_scalar(sa.iloc[-22])) / to_scalar(sa.iloc[-22])
                if s21 > REGIME_SPY_MAX:
                    skip = f"SPY +{s21:.1%}"
        if vix_series is not None and not skip:
            va = ensure_series(vix_series.loc[:test_q.start_time + pd.Timedelta(days=10)])
            if len(va) > 0 and to_scalar(va.iloc[-1]) < REGIME_VIX_MIN:
                skip = f"VIX={to_scalar(va.iloc[-1]):.0f}"
        if skip:
            continue

        # WF-internal feature selection + model training
        vd_wf = wf_train.dropna(subset=[wf_tgt])
        X_wf = clean_X(vd_wf, fcols_q, fill_meds_q)
        y_wf = vd_wf[wf_tgt].fillna(0).astype(int)
        sw_wf = (len(y_wf) - y_wf.sum()) / max(y_wf.sum(), 1)
        sel_wf = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.05,
            scale_pos_weight=sw_wf, subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=42, verbosity=0,
        )
        sel_wf.fit(X_wf, y_wf)
        wf_feats = pd.Series(
            sel_wf.feature_importances_, index=fcols_q
        ).nlargest(K).index.tolist()
        # Default params (can be overridden for A/B comparison)
        xgb_params = dict(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
        )
        if xgb_override:
            xgb_params.update(xgb_override)
        mdl_wf = xgb.XGBClassifier(
            scale_pos_weight=sw_wf,
            eval_metric='logloss', random_state=42, verbosity=0,
            **xgb_params,
        )
        mdl_wf.fit(X_wf[wf_feats], y_wf)
        X_test = clean_X(wf_test, fcols_q, fill_meds_q)
        try:
            scores = mdl_wf.predict_proba(X_test[wf_feats])[:, 1]
        except Exception:
            continue
        wf_test = wf_test.copy()
        wf_test['wf_score'] = scores

        # Sector cap
        picks = wf_test.nlargest(min(200, len(wf_test)), 'wf_score')
        picks = picks[picks['avg_vol'] >= VOL_FLOOR]
        if 'sector' in picks.columns:
            mps = max(3, int(len(picks) * SECTOR_CAP))
            capped = []
            sc = Counter()
            for _, row in picks.iterrows():
                sec = row.get('sector', 'Other')
                if sc[sec] < mps:
                    capped.append(row)
                    sc[sec] += 1
            picks = pd.DataFrame(capped)

        scored_quarters.append((test_q, picks))

    return scored_quarters


def _generate_trades(scored_quarters, hold_days, price_dict,
                     use_confirmation=False, tradeable_tickers=None,
                     use_stops=True, confirm_drop=None, confirm_window=None):
    """Generate trades from scored picks with specific hold period and entry mode.

    Borrow costs are volume-based:
        avg_vol >= 1M  -> BORROW_RATE_EASY (3% annual)
        avg_vol < 1M   -> BORROW_RATE_HARD (6% annual)

    Args:
        use_stops: If False, hold for full period without stop-loss/trailing/profit-target.
        confirm_drop: Override CONFIRMATION_DROP (e.g. 0.01 for relaxed).
        confirm_window: Override CONFIRMATION_WINDOW (e.g. 3 for relaxed).
    """
    _conf_drop = confirm_drop if confirm_drop is not None else CONFIRMATION_DROP
    _conf_window = confirm_window if confirm_window is not None else CONFIRMATION_WINDOW
    all_trades = []

    for test_q, picks in scored_quarters:
        for _, row in picks.iterrows():
            tk = row.get('ticker', '')
            pr = row.get('price', np.nan)
            if pd.isna(pr) or pr <= 0 or tk not in price_dict:
                continue
            if tradeable_tickers is not None and tk not in tradeable_tickers:
                continue

            pxd2 = price_dict[tk]
            px2 = ensure_series(pxd2['Close'] if 'Close' in pxd2.columns else pxd2.iloc[:, 0])
            vi = px2.index[px2.index >= row['report_date']]
            if len(vi) == 0:
                continue
            si = px2.index.get_loc(vi[0])

            if use_confirmation:
                signal_price = to_scalar(px2.iloc[si])
                if signal_price <= 0:
                    continue
                confirm_si = None
                for ci in range(si + 1, min(si + _conf_window + 1, len(px2))):
                    day_price = to_scalar(px2.iloc[ci])
                    if (day_price - signal_price) / signal_price <= -_conf_drop:
                        confirm_si = ci
                        break
                if confirm_si is None:
                    continue
                si_entry = confirm_si
            else:
                si_entry = si + ENTRY_DELAY

            if si_entry >= len(px2):
                continue
            entry_pr = to_scalar(px2.iloc[si_entry])
            if entry_pr <= 0:
                continue

            # Rule filters on entry day
            if si_entry >= 5:
                r5d = (entry_pr - to_scalar(px2.iloc[si_entry - 5])) / to_scalar(px2.iloc[si_entry - 5])
                if r5d < SKIP_RET5D_DOWN:
                    continue
            if si_entry >= 60:
                dr_stk = px2.pct_change()
                v5d = to_scalar(dr_stk.iloc[si_entry - 4:si_entry + 1].std() * np.sqrt(252))
                vh = dr_stk.iloc[si_entry - 59:si_entry + 1].rolling(5).std() * np.sqrt(252)
                v90 = to_scalar(vh.quantile(SKIP_VOL_PCT))
                if pd.notna(v5d) and pd.notna(v90) and v5d > v90:
                    continue

            entry_p = entry_pr * (1 + SLIPPAGE)
            best_p = entry_p
            path = []
            for di in range(1, min(hold_days + 1, len(px2) - si_entry)):
                path.append(to_scalar(px2.iloc[si_entry + di]))
            if not path:
                continue

            stopped = False
            profit_taken = False
            exit_p = entry_p
            exit_day = len(path)
            for di, day_p in enumerate(path, 1):
                best_p = min(best_p, day_p)
                if use_stops:
                    if (day_p - entry_p) / entry_p > STOP_LOSS:
                        exit_p = entry_p * (1 + STOP_LOSS) * (1 + SLIPPAGE)
                        stopped = True
                        exit_day = di
                        break
                    if (entry_p - day_p) / entry_p >= PROFIT_TARGET:
                        exit_p = day_p * (1 + SLIPPAGE)
                        profit_taken = True
                        exit_day = di
                        break
                    if (day_p < entry_p * 0.97
                            and best_p < entry_p
                            and (day_p - best_p) / best_p > TRAILING_STOP):
                        exit_p = day_p * (1 + SLIPPAGE)
                        stopped = True
                        exit_day = di
                        break
            if not stopped and not profit_taken:
                exit_p = path[-1] * (1 + SLIPPAGE)

            # Volume-based borrow cost
            avg_vol = row.get('avg_vol', 0)
            if pd.notna(avg_vol) and avg_vol >= 1_000_000:
                br = BORROW_RATE_EASY
            else:
                br = BORROW_RATE_HARD
            borrow = entry_p * br * (exit_day / 252)
            pnl_raw = entry_p - exit_p
            pnl = pnl_raw - borrow
            pnl_pct = pnl / entry_p
            entry_date = px2.index[si_entry]
            exit_date = px2.index[min(si_entry + exit_day, len(px2) - 1)]
            all_trades.append({
                'quarter': str(test_q), 'ticker': tk,
                'sector': row.get('sector', 'Other'),
                'entry_price': entry_p, 'exit_price': exit_p,
                'pnl_raw': pnl_raw, 'pnl_per_share': pnl, 'pnl_pct': pnl_pct,
                'borrow_cost': borrow, 'borrow_rate': br,
                'score': row['wf_score'], 'stopped': stopped,
                'profit_taken': profit_taken,
                'exit_days': exit_day, 'entry_date': entry_date,
                'exit_date': exit_date,
            })

    return all_trades


def _tier_stats(wf_df: pd.DataFrame) -> Dict[str, dict]:
    """Compute conviction tier statistics for Top 10/25/50% and Full."""
    if len(wf_df) < 20:
        return {}
    tiers = {}
    for tier_name, lo_pct in [('Top 10%', 0.90), ('Top 25%', 0.75),
                               ('Top 50%', 0.50), ('Full', 0.0)]:
        lo_v = wf_df['score'].quantile(lo_pct)
        sub = wf_df[wf_df['score'] >= lo_v]
        if len(sub) < 5:
            continue
        wr = (sub['pnl_per_share'] > 0).mean()
        avg = sub['pnl_per_share'].mean()
        avg_raw = sub['pnl_raw'].mean() if 'pnl_raw' in sub.columns else avg
        avg_pct = sub['pnl_pct'].mean() * 100
        sr = sub['stopped'].mean()
        avg_borrow = sub['borrow_cost'].mean() if 'borrow_cost' in sub.columns else 0
        nq = wf_df['quarter'].nunique()
        tpq = len(sub) / nq if nq > 0 else 0
        tiers[tier_name] = {
            'tier': tier_name, 'n': len(sub),
            'win': wr, 'avg_pnl': avg, 'avg_raw': avg_raw,
            'avg_pct': avg_pct, 'avg_borrow': avg_borrow,
            'stop_rate': sr, 'trades_per_q': tpq,
        }
    return tiers


def run_walkforward(data_bundle: dict) -> dict:
    """Run walk-forward locked to TRADING_TARGET / TRADING_HOLD / ENTRY_MODE."""
    t0 = time.time()
    log.info("=" * 70)
    log.info(f"WALK-FORWARD: {TRADING_TARGET} / {TRADING_HOLD}d hold / {ENTRY_MODE} entry")
    log.info("=" * 70)

    v_results = data_bundle['v_results']
    price_dict = data_bundle['price_dict']
    tradeable_tickers = data_bundle.get('tradeable_tickers')

    wf_tgt = TRADING_TARGET
    if wf_tgt not in v_results:
        # Fallback: use whatever target Pareto selected for this universe
        best_fallback = max(v_results, key=lambda k: v_results[k]['mauc'])
        log.info(f"\n  {wf_tgt} not in trained targets, using best: {best_fallback}")
        wf_tgt = best_fallback

    auc = v_results[wf_tgt]['mauc']
    use_conf = (ENTRY_MODE == "confirmed")
    log.info(f"\n  Target: {wf_tgt} (AUC={auc:.3f})")
    log.info(f"  Hold: {TRADING_HOLD}d | Entry: {ENTRY_MODE}"
          f"{f' ({CONFIRMATION_DROP:.0%} in {CONFIRMATION_WINDOW}d)' if use_conf else ''}")
    log.info(f"  Borrow: {BORROW_RATE_EASY:.0%} (easy, vol>=1M) / "
          f"{BORROW_RATE_HARD:.0%} (hard, vol<1M)")

    # Train models once
    scored_quarters = _process_quarters(data_bundle, wf_tgt)

    # Generate trades
    trades = _generate_trades(
        scored_quarters, TRADING_HOLD, price_dict,
        use_confirmation=use_conf,
        tradeable_tickers=tradeable_tickers,
    )
    wf_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    if len(wf_df) < 10:
        log.info(f"\n  Only {len(wf_df)} trades — too few for analysis")
        data_bundle.update(
            all_wf_trades=trades, wf_df=wf_df,
            wf_top=wf_df.copy(), wf_comparison=[],
        )
        return data_bundle

    # Tier analysis
    tiers = _tier_stats(wf_df)
    log.info(f"\n  {'Tier':<10} {'n':>4} {'Win':>5} {'Raw$/sh':>9} {'Borw$/sh':>9} "
          f"{'Net$/sh':>9} {'Net%':>7} {'Stops':>6}")
    log.info(f"  {'-'*60}")
    for name in ['Top 10%', 'Top 25%', 'Top 50%', 'Full']:
        t = tiers.get(name)
        if t:
            log.info(f"  {t['tier']:<10} {t['n']:>4} {t['win']:>5.0%} "
                  f"${t['avg_raw']:>+7.2f} ${t['avg_borrow']:>7.2f} "
                  f"${t['avg_pnl']:>+7.2f} {t['avg_pct']:>+6.1f}% "
                  f"{t['stop_rate']:>5.0%}")

    # Top-25% subset for equity curve
    if len(wf_df) > 20:
        score_25 = wf_df['score'].quantile(0.75)
        wf_top = wf_df[wf_df['score'] >= score_25].copy()
    else:
        wf_top = wf_df.copy()

    # Per-quarter breakdown
    if len(wf_top) > 0:
        log.info(f"\n  TOP 25% BY QUARTER ({len(wf_top)} trades):")
        for q in sorted(wf_top['quarter'].unique()):
            qs = wf_top[wf_top['quarter'] == q]
            if len(qs) >= 2:
                raw_avg = qs['pnl_raw'].mean() if 'pnl_raw' in qs.columns else qs['pnl_per_share'].mean()
                borr_avg = qs['borrow_cost'].mean() if 'borrow_cost' in qs.columns else 0
                log.info(f"    {q}: n={len(qs):>3} win={(qs['pnl_per_share']>0).mean():.0%} "
                      f"raw=${raw_avg:+.2f} borrow=${borr_avg:.2f} "
                      f"net=${qs['pnl_per_share'].mean():+.2f}/sh")

    # Profitable quarters
    q_pnl = wf_top.groupby('quarter')['pnl_per_share'].sum()
    prof_q = (q_pnl > 0).sum()
    total_q = len(q_pnl)
    log.info(f"\n  Profitable quarters: {prof_q}/{total_q}")
    log.info(f"  Trades: {len(wf_df)} total, {len(wf_top)} top-25%")
    log.info(f"  [{time.time()-t0:.0f}s] {elapsed()}")
    log.info("")

    # Build comparison row for main.py head-to-head
    t25 = tiers.get('Top 25%', {})
    full = tiers.get('Full', {})
    comparison = [{
        'target': wf_tgt, 'hold': TRADING_HOLD,
        'entry': ENTRY_MODE,
        'n': len(wf_df),
        'top25_win': t25.get('win', np.nan),
        'top25_pnl': t25.get('avg_pnl', np.nan),
        'full_win': full.get('win', np.nan),
        'full_pnl': full.get('avg_pnl', np.nan),
        'stops': full.get('stop_rate', np.nan),
    }]

    data_bundle.update(
        all_wf_trades=trades,
        wf_df=wf_df,
        wf_top=wf_top,
        wf_comparison=comparison,
    )
    return data_bundle


def run_walkforward_analysis(data_bundle: dict, entry_mode: str = "confirmed",
                             use_stops: bool = True, random_flags: bool = False,
                             n_random_flags: int = 50, random_seed: int = 42,
                             confirm_drop: Optional[float] = None,
                             confirm_window: Optional[int] = None,
                             xgb_override: Optional[dict] = None,
                             ) -> Tuple[pd.DataFrame, dict]:
    """Flexible walk-forward for analysis variants. Returns (wf_df, tiers).

    Does NOT modify data_bundle.

    Args:
        entry_mode: "confirmed", "immediate", or "confirmed_relaxed".
        use_stops: If False, hold full period without stops.
        random_flags: If True, assign random scores instead of model scores.
        n_random_flags: Number of random picks per quarter (when random_flags=True).
        random_seed: Seed for random flag generation.
        confirm_drop: Override confirmation drop threshold.
        confirm_window: Override confirmation window days.
        xgb_override: Override XGB params for A/B testing.
    """
    import random as _rand

    wf_tgt = TRADING_TARGET
    v_results = data_bundle['v_results']
    price_dict = data_bundle['price_dict']
    tradeable_tickers = data_bundle.get('tradeable_tickers')

    if wf_tgt not in v_results:
        best_fallback = max(v_results, key=lambda k: v_results[k]['mauc'])
        wf_tgt = best_fallback

    # Entry mode logic
    use_conf = entry_mode in ("confirmed", "confirmed_relaxed")
    if entry_mode == "confirmed_relaxed":
        confirm_drop = confirm_drop or 0.01
        confirm_window = confirm_window or 3

    # Train models and score quarters
    scored_quarters = _process_quarters(data_bundle, wf_tgt, xgb_override=xgb_override)

    # Random flags: replace model scores with random scores
    if random_flags:
        rng = _rand.Random(random_seed)
        randomized = []
        for test_q, picks in scored_quarters:
            picks = picks.copy()
            # Assign random scores and take n_random_flags
            picks['wf_score'] = [rng.random() for _ in range(len(picks))]
            picks = picks.nlargest(min(n_random_flags, len(picks)), 'wf_score')
            randomized.append((test_q, picks))
        scored_quarters = randomized

    trades = _generate_trades(
        scored_quarters, TRADING_HOLD, price_dict,
        use_confirmation=use_conf,
        tradeable_tickers=tradeable_tickers,
        use_stops=use_stops,
        confirm_drop=confirm_drop,
        confirm_window=confirm_window,
    )
    wf_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    tiers = _tier_stats(wf_df) if len(wf_df) >= 20 else {}
    return wf_df, tiers


def run_walkforward_ab(data_bundle: dict, xgb_override: dict,
                       label: str = "Bayesian") -> Tuple[pd.DataFrame, dict]:
    """Run walk-forward with custom XGB params for A/B comparison.

    Returns (wf_df, tiers) without modifying data_bundle.
    """
    t0 = time.time()
    wf_tgt = TRADING_TARGET
    v_results = data_bundle['v_results']
    price_dict = data_bundle['price_dict']
    tradeable_tickers = data_bundle.get('tradeable_tickers')

    if wf_tgt not in v_results:
        best_fallback = max(v_results, key=lambda k: v_results[k]['mauc'])
        wf_tgt = best_fallback

    use_conf = (ENTRY_MODE == "confirmed")

    scored_quarters = _process_quarters(data_bundle, wf_tgt, xgb_override=xgb_override)

    trades = _generate_trades(
        scored_quarters, TRADING_HOLD, price_dict,
        use_confirmation=use_conf,
        tradeable_tickers=tradeable_tickers,
    )
    wf_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    tiers = _tier_stats(wf_df) if len(wf_df) >= 20 else {}

    log.info(f"  [{label}] {len(wf_df)} trades, {time.time()-t0:.0f}s")
    return wf_df, tiers


def run_walkforward_split(data_bundle: dict, split_date: str,
                          train_side: str = "early") -> Tuple[pd.DataFrame, dict]:
    """Single train/test split for temporal persistence testing.

    Trains one model on all data from one side of split_date,
    scores the other side, then generates trades on the test period.

    Args:
        split_date: Date string (e.g. '2018-12-31') to split on.
        train_side: 'early' trains on pre-split, tests on post-split.
                    'late' trains on post-split, tests on pre-split.

    Returns (wf_df, tiers).
    """
    df_dev = data_bundle['df_dev']
    fcols_q = data_bundle['fcols_q']
    fill_meds_q = data_bundle['fill_meds_q']
    K = data_bundle['K']
    price_dict = data_bundle['price_dict']
    tradeable_tickers = data_bundle.get('tradeable_tickers')

    wf_tgt = TRADING_TARGET
    v_results = data_bundle.get('v_results', {})
    if wf_tgt not in v_results and v_results:
        wf_tgt = max(v_results, key=lambda k: v_results[k]['mauc'])

    split_ts = pd.Timestamp(split_date)
    if train_side == "early":
        train_df = df_dev[df_dev['report_date'] <= split_ts]
        test_df = df_dev[df_dev['report_date'] > split_ts]
    else:
        train_df = df_dev[df_dev['report_date'] > split_ts]
        test_df = df_dev[df_dev['report_date'] <= split_ts]

    log.info(f"  Split train={train_side}: {len(train_df)} train, {len(test_df)} test rows")

    if wf_tgt not in train_df.columns or train_df[wf_tgt].sum() < 30:
        log.warning(f"  Target {wf_tgt} has insufficient events in train set")
        return pd.DataFrame(), {}

    # Train single model on all training data
    vd = train_df.dropna(subset=[wf_tgt])
    X_train = clean_X(vd, fcols_q, fill_meds_q)
    y_train = vd[wf_tgt].fillna(0).astype(int)
    sw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    # Feature selection
    sel = xgb.XGBClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.05,
        scale_pos_weight=sw, subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, verbosity=0,
    )
    sel.fit(X_train, y_train)
    feats = pd.Series(
        sel.feature_importances_, index=fcols_q
    ).nlargest(K).index.tolist()

    # Full model
    mdl = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=sw, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='logloss', random_state=42, verbosity=0,
    )
    mdl.fit(X_train[feats], y_train)

    # Score test data by quarter
    X_test = clean_X(test_df, fcols_q, fill_meds_q)
    try:
        scores = mdl.predict_proba(X_test[feats])[:, 1]
    except Exception:
        return pd.DataFrame(), {}

    test_df = test_df.copy()
    test_df['wf_score'] = scores

    # Group into quarters and generate trades
    quarters = sorted(test_df['report_date'].dt.to_period('Q').unique())
    scored_quarters = []
    for q in quarters:
        q_data = test_df[
            (test_df['report_date'] >= q.start_time)
            & (test_df['report_date'] <= q.end_time)
        ]
        if len(q_data) < 5:
            continue
        picks = q_data.nlargest(min(200, len(q_data)), 'wf_score')
        picks = picks[picks['avg_vol'] >= VOL_FLOOR]
        if 'sector' in picks.columns:
            mps = max(3, int(len(picks) * SECTOR_CAP))
            capped = []
            sc = Counter()
            for _, row in picks.iterrows():
                sec = row.get('sector', 'Other')
                if sc[sec] < mps:
                    capped.append(row)
                    sc[sec] += 1
            picks = pd.DataFrame(capped)
        scored_quarters.append((q, picks))

    use_conf = (ENTRY_MODE == "confirmed")
    trades = _generate_trades(
        scored_quarters, TRADING_HOLD, price_dict,
        use_confirmation=use_conf,
        tradeable_tickers=tradeable_tickers,
    )
    wf_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    tiers = _tier_stats(wf_df) if len(wf_df) >= 20 else {}

    log.info(f"  Split result: {len(wf_df)} trades")
    return wf_df, tiers
