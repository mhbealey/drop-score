"""
Walk-forward backtest with 1-day entry delay, rule-based filters,
sector cap, regime filter, and conviction tiers.
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


def run_walkforward(data_bundle):
    """Execute walk-forward backtest. Returns updated data_bundle with trade results."""
    t0 = time.time()
    print("=" * 70)
    print("WALK-FORWARD (1-day entry delay, rule-based filters)")
    print("=" * 70)

    df_dev = data_bundle['df_dev']
    fcols_q = data_bundle['fcols_q']
    fill_meds_q = data_bundle['fill_meds_q']
    v_results = data_bundle['v_results']
    best_v_t = data_bundle['best_v_t']
    K = data_bundle['K']
    price_dict = data_bundle['price_dict']
    spy_close = data_bundle['spy_close']
    vix_series = data_bundle['vix_series']

    quarters = sorted(df_dev['report_date'].dt.to_period('Q').unique())
    # Use FORCE_TARGET if set and available, otherwise fall back to best/exdrop logic
    if FORCE_TARGET and FORCE_TARGET in v_results:
        wf_tgt = FORCE_TARGET
        print(f"  Target (forced): {wf_tgt} (AUC={v_results[wf_tgt]['mauc']:.3f})")
    else:
        wf_tgt = best_v_t
        ex_cands = [t for t in v_results if t.startswith('exdrop_')]
        if ex_cands:
            bex = max(ex_cands, key=lambda k: v_results[k]['mauc'])
            if v_results[bex]['mauc'] > 0.70:
                wf_tgt = bex
        if FORCE_TARGET:
            print(f"  WARNING: FORCE_TARGET={FORCE_TARGET} not available, "
                  f"using {wf_tgt}")
        print(f"  Target: {wf_tgt} (AUC={v_results[wf_tgt]['mauc']:.3f})")

    all_wf_trades = []
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
            print(f"    {test_q}: SKIP ({skip})")
            continue

        # WF-internal feature selection
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
        except:
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

        # Simulate each trade
        for _, row in picks.iterrows():
            tk = row.get('ticker', '')
            pr = row.get('price', np.nan)
            if pd.isna(pr) or pr <= 0 or tk not in price_dict:
                continue
            pxd2 = price_dict[tk]
            px2 = ensure_series(pxd2['Close'] if 'Close' in pxd2.columns else pxd2.iloc[:, 0])
            vi = px2.index[px2.index >= row['report_date']]
            if len(vi) == 0:
                continue
            si = px2.index.get_loc(vi[0])
            # 1-DAY ENTRY DELAY
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
            for di in range(1, min(22, len(px2) - si_entry)):
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
            pnl_pct = pnl / entry_p  # Return on capital
            entry_date = px2.index[si_entry]
            exit_date = px2.index[min(si_entry + exit_day, len(px2) - 1)]
            all_wf_trades.append({
                'quarter': str(test_q), 'ticker': tk,
                'sector': row.get('sector', 'Other'),
                'entry_price': entry_p, 'exit_price': exit_p,
                'pnl_per_share': pnl, 'pnl_pct': pnl_pct,
                'score': row['wf_score'], 'stopped': stopped,
                'profit_taken': profit_taken,
                'exit_days': exit_day, 'entry_date': entry_date,
                'exit_date': exit_date, 'borrow_rate': br,
            })

    wf_df = pd.DataFrame(all_wf_trades) if all_wf_trades else pd.DataFrame()
    print(f"\n  Trades: {len(wf_df)}")

    # CONVICTION TIERS
    wf_top = wf_df.copy()
    if len(wf_df) > 20:
        print(f"\n  CONVICTION TIERS:")
        for tier_name, lo_pct, hi_pct in [
            ('Top 10%', 0.90, 1.0), ('Top 25%', 0.75, 1.0),
            ('Top 50%', 0.50, 1.0), ('Full', 0.0, 1.0),
        ]:
            lo_v = wf_df['score'].quantile(lo_pct)
            sub = wf_df[wf_df['score'] >= lo_v]
            if len(sub) < 5:
                continue
            wr = (sub['pnl_per_share'] > 0).mean()
            avg = sub['pnl_per_share'].mean()
            avg_pct = sub['pnl_pct'].mean() * 100
            sr = sub['stopped'].mean()
            tpq = len(sub) / wf_df['quarter'].nunique()
            print(f"    {tier_name:<10} n={len(sub):>3} win={wr:.0%} "
                  f"avg=${avg:+.2f}/sh ({avg_pct:+.1f}%) stop={sr:.0%} ~{tpq:.0f}/Q")

        # Default tier for equity curve: top 25%
        score_25 = wf_df['score'].quantile(0.75)
        wf_top = wf_df[wf_df['score'] >= score_25].copy()
        print(f"\n  DEFAULT (Top 25%): {len(wf_top)} trades")
        print(f"  BY QUARTER:")
        for q in sorted(wf_top['quarter'].unique()):
            qs = wf_top[wf_top['quarter'] == q]
            if len(qs) >= 2:
                print(f"    {q}: n={len(qs):>3} win={(qs['pnl_per_share']>0).mean():.0%} "
                      f"avg=${qs['pnl_per_share'].mean():+.2f}/sh "
                      f"({qs['pnl_pct'].mean()*100:+.1f}%)")

    print(f"  [{time.time()-t0:.0f}s] {elapsed()}")
    print()

    data_bundle.update(
        all_wf_trades=all_wf_trades,
        wf_df=wf_df,
        wf_top=wf_top,
    )
    return data_bundle
