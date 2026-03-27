"""
Walk-forward backtest with matched target/hold periods,
confirmation entry, rule-based filters, sector cap, regime filter,
and conviction tiers.
"""
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter
from tqdm import tqdm

from config import (
    VOL_FLOOR, SECTOR_CAP, REGIME_SPY_MAX, REGIME_VIX_MIN,
    SLIPPAGE, STOP_LOSS, PROFIT_TARGET, TRAILING_STOP,
    SKIP_RET5D_DOWN, SKIP_VOL_PCT, ENTRY_DELAY, FORCE_TARGET,
)
from utils import clean_X, elapsed, to_scalar, ensure_series


# Walk-forward configurations: (target, hold_days)
WF_CONFIGS = [
    ('voladj_2sig_63d', 63),
    ('exdrop_15_10d', 21),
    ('voladj_2sig_42d', 42),
]


def _process_quarters(data_bundle, wf_tgt):
    """Train WF-internal models per quarter and return scored picks.

    This is the expensive part (model training). Called once per target,
    then _generate_trades runs twice (immediate + confirmed) reusing results.
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
        mdl_wf = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=sw_wf, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric='logloss', random_state=42, verbosity=0,
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
                     use_confirmation=False, tradeable_tickers=None):
    """Generate trades from scored picks with specific hold period and entry mode.

    Args:
        scored_quarters: list of (quarter, picks_df) from _process_quarters
        hold_days: max hold period in trading days (matched to target window)
        price_dict: ticker -> price DataFrame
        use_confirmation: if True, wait for 2% drop within 5 days before entering
        tradeable_tickers: if set, only generate trades for these tickers
    """
    all_trades = []

    for test_q, picks in scored_quarters:
        for _, row in picks.iterrows():
            tk = row.get('ticker', '')
            pr = row.get('price', np.nan)
            if pd.isna(pr) or pr <= 0 or tk not in price_dict:
                continue
            # Only trade currently-listed stocks in walk-forward
            if tradeable_tickers is not None and tk not in tradeable_tickers:
                continue

            pxd2 = price_dict[tk]
            px2 = ensure_series(pxd2['Close'] if 'Close' in pxd2.columns else pxd2.iloc[:, 0])
            vi = px2.index[px2.index >= row['report_date']]
            if len(vi) == 0:
                continue
            si = px2.index.get_loc(vi[0])

            if use_confirmation:
                # Wait up to 5 trading days for a 2% decline from signal day's close
                signal_price = to_scalar(px2.iloc[si])
                if signal_price <= 0:
                    continue
                confirm_si = None
                for ci in range(si + 1, min(si + 6, len(px2))):
                    day_price = to_scalar(px2.iloc[ci])
                    if (day_price - signal_price) / signal_price <= -0.02:
                        confirm_si = ci
                        break
                if confirm_si is None:
                    continue  # Not confirmed within 5 days, skip
                si_entry = confirm_si
            else:
                # Standard 1-day entry delay
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

            mcap = row.get('market_cap', np.nan)
            br = (0.02 if pd.notna(mcap) and mcap > 10e9
                  else 0.04 if pd.notna(mcap) and mcap > 2e9
                  else 0.08)
            borrow = entry_p * br * (exit_day / 252)
            pnl = (entry_p - exit_p) - borrow
            pnl_pct = pnl / entry_p
            entry_date = px2.index[si_entry]
            exit_date = px2.index[min(si_entry + exit_day, len(px2) - 1)]
            all_trades.append({
                'quarter': str(test_q), 'ticker': tk,
                'sector': row.get('sector', 'Other'),
                'entry_price': entry_p, 'exit_price': exit_p,
                'pnl_per_share': pnl, 'pnl_pct': pnl_pct,
                'score': row['wf_score'], 'stopped': stopped,
                'profit_taken': profit_taken,
                'exit_days': exit_day, 'entry_date': entry_date,
                'exit_date': exit_date, 'borrow_rate': br,
            })

    return all_trades


def _tier_stats(wf_df):
    """Compute conviction tier statistics."""
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
        avg_pct = sub['pnl_pct'].mean() * 100
        sr = sub['stopped'].mean()
        nq = wf_df['quarter'].nunique()
        tpq = len(sub) / nq if nq > 0 else 0
        tiers[tier_name] = {
            'tier': tier_name, 'n': len(sub),
            'win': wr, 'avg_pnl': avg, 'avg_pct': avg_pct,
            'stop_rate': sr, 'trades_per_q': tpq,
        }
    return tiers


def run_walkforward(data_bundle):
    """Run walk-forward comparison across 3 target/hold configs x 2 entry modes."""
    t0 = time.time()
    print("=" * 70)
    print("WALK-FORWARD (matched target/hold, immediate + confirmed entry)")
    print("=" * 70)

    v_results = data_bundle['v_results']
    price_dict = data_bundle['price_dict']
    tradeable_tickers = data_bundle.get('tradeable_tickers')

    comparison = []
    # Store results keyed by (target, entry_mode)
    all_results = {}

    for tgt, hold in WF_CONFIGS:
        if tgt not in v_results:
            print(f"\n  {tgt}/{hold}d: SKIPPED (target not trained)")
            continue

        auc = v_results[tgt]['mauc']
        print(f"\n  {tgt}/{hold}d (AUC={auc:.3f}):")

        # Train models once per target (expensive)
        scored_quarters = _process_quarters(data_bundle, tgt)

        for entry_mode, use_conf in [('Immediate', False), ('Confirmed', True)]:
            trades = _generate_trades(
                scored_quarters, hold, price_dict,
                use_confirmation=use_conf,
                tradeable_tickers=tradeable_tickers,
            )
            wf_df = pd.DataFrame(trades) if trades else pd.DataFrame()
            all_results[(tgt, entry_mode)] = (trades, wf_df, hold)

            if len(wf_df) < 10:
                print(f"    {entry_mode}: {len(wf_df)} trades (too few)")
                comparison.append({
                    'target': tgt, 'hold': hold, 'entry': entry_mode,
                    'n': len(wf_df),
                    'top25_win': np.nan, 'top25_pnl': np.nan,
                    'full_win': np.nan, 'full_pnl': np.nan,
                    'stops': np.nan,
                })
                continue

            tiers = _tier_stats(wf_df)
            t25 = tiers.get('Top 25%', {})
            full = tiers.get('Full', {})

            row = {
                'target': tgt, 'hold': hold, 'entry': entry_mode,
                'n': len(wf_df),
                'top25_win': t25.get('win', np.nan),
                'top25_pnl': t25.get('avg_pnl', np.nan),
                'full_win': full.get('win', np.nan),
                'full_pnl': full.get('avg_pnl', np.nan),
                'stops': full.get('stop_rate', np.nan),
            }
            comparison.append(row)

            # Print tier details
            for name in ['Top 10%', 'Top 25%', 'Top 50%', 'Full']:
                t = tiers.get(name)
                if t:
                    print(f"    {entry_mode:<11} {t['tier']:<10} n={t['n']:>3} "
                          f"win={t['win']:.0%} avg=${t['avg_pnl']:+.2f}/sh "
                          f"({t['avg_pct']:+.1f}%) stop={t['stop_rate']:.0%}")

    # ── Comparison table ──
    print(f"\n  {'=' * 85}")
    print(f"  COMPARISON TABLE:")
    hdr = (f"  {'TARGET/HOLD':<28} {'Entry':<12} {'n':>4} "
           f"{'Top25%Win':>9} {'Top25%P&L':>10} "
           f"{'FullWin':>8} {'FullP&L':>9} {'Stops':>6}")
    print(hdr)
    print(f"  {'-' * 85}")
    for row in comparison:
        tgt_hold = f"{row['target']}/{row['hold']}d"
        t25w = f"{row['top25_win']:.0%}" if pd.notna(row['top25_win']) else 'N/A'
        t25p = f"${row['top25_pnl']:+.2f}" if pd.notna(row['top25_pnl']) else 'N/A'
        fw = f"{row['full_win']:.0%}" if pd.notna(row['full_win']) else 'N/A'
        fp = f"${row['full_pnl']:+.2f}" if pd.notna(row['full_pnl']) else 'N/A'
        st = f"{row['stops']:.0%}" if pd.notna(row['stops']) else 'N/A'
        print(f"  {tgt_hold:<28} {row['entry']:<12} {row['n']:>4} "
              f"{t25w:>9} {t25p:>10} {fw:>8} {fp:>9} {st:>6}")
    print(f"  {'=' * 85}")

    # ── Select best configuration ──
    # Primary: Confirmed entry where Top25% Win > Full Win AND Top25% P&L > 0
    best_key = None
    best_score = -999
    for row in comparison:
        if row['entry'] != 'Confirmed':
            continue
        t25w = row.get('top25_win', 0)
        fw = row.get('full_win', 0)
        t25p = row.get('top25_pnl', 0)
        if pd.isna(t25w) or pd.isna(fw) or pd.isna(t25p):
            continue
        if t25w > fw and t25p > 0:
            score = t25p  # Higher P&L = better
            if score > best_score:
                best_score = score
                best_key = (row['target'], 'Confirmed')

    # Fallback: best confirmed by Top25% P&L
    if best_key is None:
        for row in comparison:
            if row['entry'] != 'Confirmed' or row['n'] < 10:
                continue
            t25p = row.get('top25_pnl', -999)
            if pd.notna(t25p) and t25p > best_score:
                best_score = t25p
                best_key = (row['target'], 'Confirmed')

    # Ultimate fallback: best immediate
    if best_key is None:
        for row in comparison:
            if row['n'] < 10:
                continue
            t25p = row.get('top25_pnl', -999)
            if pd.notna(t25p) and t25p > best_score:
                best_score = t25p
                best_key = (row['target'], row['entry'])

    # Extract best config's trades
    if best_key and best_key in all_results:
        best_trades, best_wf_df, best_hold = all_results[best_key]
        print(f"\n  SELECTED: {best_key[0]}/{best_hold}d ({best_key[1]})")
    else:
        best_trades = []
        best_wf_df = pd.DataFrame()
        print(f"\n  WARNING: No valid configuration found")

    # Top-25% subset for equity curve
    if len(best_wf_df) > 20:
        score_25 = best_wf_df['score'].quantile(0.75)
        wf_top = best_wf_df[best_wf_df['score'] >= score_25].copy()
    else:
        wf_top = best_wf_df.copy()

    # Print per-quarter for selected config
    if len(wf_top) > 0:
        print(f"\n  DEFAULT (Top 25%): {len(wf_top)} trades")
        print(f"  BY QUARTER:")
        for q in sorted(wf_top['quarter'].unique()):
            qs = wf_top[wf_top['quarter'] == q]
            if len(qs) >= 2:
                print(f"    {q}: n={len(qs):>3} win={(qs['pnl_per_share']>0).mean():.0%} "
                      f"avg=${qs['pnl_per_share'].mean():+.2f}/sh "
                      f"({qs['pnl_pct'].mean()*100:+.1f}%)")

    print(f"\n  Trades: {len(best_wf_df)} total, {len(wf_top)} top-25%")
    print(f"  [{time.time()-t0:.0f}s] {elapsed()}")
    print()

    data_bundle.update(
        all_wf_trades=best_trades,
        wf_df=best_wf_df,
        wf_top=wf_top,
        wf_comparison=comparison,
    )
    return data_bundle
