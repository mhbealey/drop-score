"""
Equity curve simulation with sequential position filling, dynamic regime check,
tiered borrow costs, 2% max loss cap. Paper trade scenarios. Diagnostics.
"""
import os, time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from config import (
    SLIPPAGE, STOP_LOSS, VOL_FLOOR, log,
)
from utils import elapsed, to_scalar, ensure_series


def run_equity_sim(trades_df: pd.DataFrame, start_bal: int, pos_size: int,
                   max_pos: int, label: str,
                   spy_regime_pct: Optional[float] = None,
                   spy_close: Optional[pd.Series] = None,
                   price_dict: Optional[dict] = None) -> Optional[dict]:
    """Sequential equity sim. Returns dict of results or None if too few trades."""
    if len(trades_df) < 5:
        return None
    ws = trades_df.sort_values(
        ['entry_date', 'score'], ascending=[True, False]
    ).reset_index(drop=True)
    cash = float(start_bal)
    positions = []
    trade_log = []
    daily_eq = []
    if spy_close is not None:
        dates = sorted(spy_close.index[
            (spy_close.index >= ws['entry_date'].min())
            & (spy_close.index <= ws['exit_date'].max())
        ])
    else:
        dates = sorted(set(ws['entry_date'].tolist() + ws['exit_date'].tolist()))
    taken = set()
    regime_off_until = None

    for dt in dates:
        # Dynamic regime check
        if spy_regime_pct is not None and spy_close is not None:
            sp10 = ensure_series(spy_close.loc[:dt])
            if len(sp10) > 10:
                spy_10d = (
                    (to_scalar(sp10.iloc[-1]) - to_scalar(sp10.iloc[-11]))
                    / to_scalar(sp10.iloc[-11])
                    if to_scalar(sp10.iloc[-11]) > 0 else 0
                )
                if spy_10d > spy_regime_pct:
                    # Close all open positions at today's price
                    for pos in positions:
                        tk_p = price_dict.get(pos['ticker']) if price_dict else None
                        if tk_p is not None:
                            px_t = ensure_series(tk_p['Close'] if 'Close' in tk_p.columns
                                    else tk_p.iloc[:, 0])
                            cp = px_t.asof(dt)
                            if pd.notna(cp):
                                early_pnl = (pos['entry_p'] - to_scalar(cp)) * pos['shares']
                                cash += pos['entry_val'] + early_pnl
                                trade_log.append({**pos, 'pnl': early_pnl, 'early_close': True})
                            else:
                                cash += pos['entry_val'] + pos['pnl']
                                trade_log.append(pos)
                        else:
                            cash += pos['entry_val'] + pos['pnl']
                            trade_log.append(pos)
                    positions = []
                    # Wait until SPY 5d turns negative
                    regime_off_until = 'wait_neg'
                    continue
            if regime_off_until == 'wait_neg' and len(sp10) > 5:
                sp5v = to_scalar(sp10.iloc[-6])
                spy_5d = (to_scalar(sp10.iloc[-1]) - sp5v) / sp5v if sp5v > 0 else 0
                if spy_5d < 0:
                    regime_off_until = None
                else:
                    # Still waiting -- compute daily equity with no positions
                    daily_eq.append({'date': dt, 'equity': cash, 'n_pos': 0})
                    continue

        # Close exited positions
        still_open = []
        for pos in positions:
            if dt >= pos['exit_date']:
                cash += pos['entry_val'] + pos['pnl']
                trade_log.append(pos)
            else:
                still_open.append(pos)
        positions = still_open
        # Fill slots
        if len(positions) < max_pos and regime_off_until is None:
            open_tk = {p['ticker'] for p in positions}
            for tidx in range(len(ws)):
                if tidx in taken or len(positions) >= max_pos:
                    continue
                t_row = ws.iloc[tidx]
                ed = t_row['entry_date']
                if ed > dt or (dt - ed).days > 30:
                    continue
                if t_row['ticker'] in open_tk:
                    continue
                # 2% max loss cap
                max_shares_by_risk = (
                    int(start_bal * 0.02 / (t_row['entry_price'] * STOP_LOSS))
                    if t_row['entry_price'] > 0 else 0
                )
                shares = (
                    min(int(pos_size / t_row['entry_price']), max_shares_by_risk)
                    if t_row['entry_price'] > 0 else 0
                )
                if shares == 0:
                    continue
                if cash < shares * t_row['entry_price'] * 0.5:
                    continue
                ev = shares * t_row['entry_price']
                pv = shares * t_row['pnl_per_share']
                cash -= ev
                positions.append({
                    'ticker': t_row['ticker'],
                    'entry_p': t_row['entry_price'],
                    'shares': shares,
                    'entry_date': ed,
                    'exit_date': t_row['exit_date'],
                    'entry_val': ev,
                    'pnl': pv,
                    'sector': t_row['sector'],
                    'stopped': t_row['stopped'],
                    'profit_taken': t_row['profit_taken'],
                })
                taken.add(tidx)
                open_tk.add(t_row['ticker'])
        # MTM
        pv = 0
        for pos in positions:
            tk_p = price_dict.get(pos['ticker']) if price_dict else None
            if tk_p is not None:
                px_t = ensure_series(tk_p['Close'] if 'Close' in tk_p.columns
                        else tk_p.iloc[:, 0])
                cp = px_t.asof(dt)
                if pd.notna(cp):
                    pv += (pos['entry_p'] - to_scalar(cp)) * pos['shares']
        teq = cash + sum(p['entry_val'] for p in positions) + pv
        daily_eq.append({'date': dt, 'equity': teq, 'n_pos': len(positions)})

    # Close remaining
    for pos in positions:
        trade_log.append(pos)
    if not daily_eq:
        return None
    eq = pd.DataFrame(daily_eq)
    eq['date'] = pd.to_datetime(eq['date'])
    eq = eq.sort_values('date')
    peak = eq['equity'].cummax()
    dd = eq['equity'] - peak
    max_dd = dd.min()
    max_dd_pct = max_dd / peak.max() if peak.max() > 0 else 0
    yrs = max((eq['date'].iloc[-1] - eq['date'].iloc[0]).days / 365, 0.5)
    ann = (eq['equity'].iloc[-1] / start_bal - 1) / yrs
    calmar = ann / abs(max_dd_pct) if max_dd < 0 else np.nan
    return {
        'label': label, 'start': start_bal,
        'end': to_scalar(eq['equity'].iloc[-1]),
        'ret': (eq['equity'].iloc[-1] - start_bal) / start_bal,
        'ann': ann, 'max_dd': max_dd, 'max_dd_pct': max_dd_pct,
        'calmar': calmar, 'trades': len(trade_log),
        'avg_pos': eq['n_pos'].mean(), 'eq_df': eq,
        'trade_log': trade_log,
    }


def run_equity_scenarios(data_bundle: dict) -> dict:
    """Run all equity scenarios, ramp-up, diagnostics, and plots."""
    t0 = time.time()
    log.info("=" * 70)
    log.info("EQUITY CURVES + DYNAMIC REGIME")
    log.info("=" * 70)

    wf_top = data_bundle['wf_top']
    wf_df = data_bundle['wf_df']
    all_wf_trades = data_bundle['all_wf_trades']
    spy_close = data_bundle['spy_close']
    price_dict = data_bundle['price_dict']
    cache_dir = data_bundle['cache_dir']

    # Dynamic regime diagnostic: when would each threshold have fired?
    if spy_close is not None and len(wf_top) > 0:
        log.info("  REGIME DIAGNOSTIC (when would triggers fire?):")
        bad_qs = [
            q for q in wf_top['quarter'].unique()
            if wf_top[wf_top['quarter'] == q]['pnl_per_share'].mean() < -0.5
        ]
        for q in bad_qs:
            try:
                qp = pd.Period(q)
                spy_q = ensure_series(spy_close.loc[qp.start_time:qp.end_time])
                for thr_pct in [0.03, 0.04, 0.05]:
                    for i in range(10, len(spy_q)):
                        r10 = ((to_scalar(spy_q.iloc[i]) - to_scalar(spy_q.iloc[i - 10]))
                               / to_scalar(spy_q.iloc[i - 10]))
                        if r10 > thr_pct:
                            day_num = i
                            log.info(f"    {q} @ {thr_pct:.0%}: triggers day {day_num} "
                                  f"({spy_q.index[i].date()})")
                            break
                    else:
                        log.info(f"    {q} @ {thr_pct:.0%}: never triggers")
            except Exception:
                pass

    # Run 3 account scenarios x medium regime check
    log.info(f"\n  ACCOUNT SCENARIOS (top-25% trades, +4% regime check):")
    scenarios = [
        ('Conservative', 10000, 2000, 3),
        ('Moderate',     25000, 5000, 5),
        ('Aggressive',   50000, 8000, 5),
    ]
    eq_results = {}
    for lbl, bal, ps, mp in scenarios:
        r = run_equity_sim(wf_top, bal, ps, mp, lbl,
                           spy_regime_pct=0.04,
                           spy_close=spy_close,
                           price_dict=price_dict)
        if r:
            eq_results[lbl] = r
            log.info(f"    {lbl:<14} ${bal:>6,}\u2192${r['end']:>8,.0f} ({r['ret']:+.0%}) "
                  f"DD=${r['max_dd']:>7,.0f} ({r['max_dd_pct']:.0%}) "
                  f"Cal={r['calmar']:.2f} trades={r['trades']}")

    # Ramp-up scenario
    log.info(f"\n  RAMP-UP ($5K start, scale 25% after profitable Q):")
    if len(wf_top) > 10:
        ws_ramp = wf_top.sort_values('entry_date').reset_index(drop=True)
        ramp_bal = 5000
        ramp_ps = 1000
        ramp_mp = 3
        ramp_cap = 5000
        ramp_eq = []
        ramp_trades = 0
        # Run quarter by quarter
        for q in sorted(wf_top['quarter'].unique()):
            q_trades = wf_top[wf_top['quarter'] == q].sort_values('score', ascending=False)
            q_pnl = 0
            q_count = 0
            if ramp_bal < 100:
                ramp_eq.append({'quarter': q, 'balance': ramp_bal, 'q_pnl': 0, 'ps': ramp_ps})
                continue
            for _, row in q_trades.iterrows():
                if q_count >= ramp_mp * 4:
                    break  # Max trades per quarter
                shares = (
                    min(int(ramp_ps / row['entry_price']),
                        int(ramp_bal * 0.02 / (row['entry_price'] * STOP_LOSS)))
                    if row['entry_price'] > 0 else 0
                )
                if shares == 0:
                    continue
                trade_pnl = shares * row['pnl_per_share']
                ramp_bal += trade_pnl
                q_pnl += trade_pnl
                q_count += 1
                ramp_trades += 1
            ramp_eq.append({'quarter': q, 'balance': ramp_bal, 'q_pnl': q_pnl, 'ps': ramp_ps})
            if q_pnl > 0:
                ramp_ps = min(int(ramp_ps * 1.25), ramp_cap)  # Scale up after win
            # Hold flat after loss (don't decrease)
        ramp_df = pd.DataFrame(ramp_eq)
        if len(ramp_df) > 0:
            log.info(f"    Start: $5,000 | End: ${ramp_bal:,.0f} ({(ramp_bal-5000)/5000:+.0%})")
            log.info(f"    Position size: $1,000 \u2192 ${ramp_ps:,}")
            log.info(f"    Trades: {ramp_trades}")
            for _, r in ramp_df.iterrows():
                log.info(f"      {r['quarter']}: ${r['balance']:>8,.0f} "
                      f"(Q P&L ${r['q_pnl']:+,.0f}, size ${r['ps']:,})")

    # Best scenario equity curve plot
    best_sc = eq_results.get('Moderate')
    if best_sc and 'eq_df' in best_sc:
        eq = best_sc['eq_df']
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(eq['date'], eq['equity'], color='blue', linewidth=1)
        ax.axhline(best_sc['start'], color='gray', linestyle='--', alpha=0.5)
        pk = eq['equity'].cummax()
        ax.fill_between(eq['date'], eq['equity'], pk, alpha=0.1, color='red')
        ax.set_title(f"Moderate Scenario ($25K, top-25%, +4% regime)")
        ax.set_ylabel('Account ($)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(cache_dir, 'equity_v18.png'), dpi=150)
        except Exception:
            pass
        plt.show()

    log.info(f"  [{time.time()-t0:.0f}s] {elapsed()}")
    log.info("")

    data_bundle['eq_results'] = eq_results
    data_bundle['best_sc'] = best_sc

    _run_diagnostics(data_bundle, wf_top, all_wf_trades, best_sc, spy_close, price_dict)

    return data_bundle


def _run_diagnostics(data_bundle, wf_top, all_wf_trades, best_sc, spy_close, price_dict):
    """Post-equity diagnostics: tech-only, staged entry, SPY corr, earnings proximity."""
    t0 = time.time()
    log.info("=" * 70)
    log.info("DIAGNOSTICS")
    log.info("=" * 70)

    if len(wf_top) > 10 and 'sector' in wf_top.columns:
        tech = wf_top[wf_top['sector'] == 'Technology']
        if len(tech) > 10:
            r_tech = run_equity_sim(tech, 25000, 5000, 5, 'Tech-only',
                                    spy_regime_pct=0.04,
                                    spy_close=spy_close,
                                    price_dict=price_dict)
            if r_tech:
                log.info(f"\n  TECH-ONLY: {len(tech)} trades \u2192 "
                      f"${r_tech['end']:,.0f} ({r_tech['ret']:+.0%}) "
                      f"DD={r_tech['max_dd_pct']:.0%} Cal={r_tech['calmar']:.2f}")

    _staged_entry_diagnostic(all_wf_trades, price_dict)
    _spy_correlation_diagnostic(best_sc, spy_close)
    try:
        _earnings_proximity_diagnostic(data_bundle, wf_top)
    except (KeyError, Exception) as e:
        log.info(f"\n  EARNINGS PROXIMITY: skipped ({e})")

    log.info(f"  [{time.time()-t0:.0f}s] {elapsed()}")
    log.info("")


def _staged_entry_diagnostic(all_wf_trades, price_dict):
    """Compare immediate vs confirmed entry P&L."""
    log.info(f"\n  STAGED ENTRY (enter after 2% drop in 5d):")
    if len(all_wf_trades) <= 20:
        return
    immediate_pnl = []
    confirmed_pnl = []
    confirmed_count = 0
    missed = 0
    for t in all_wf_trades:
        immediate_pnl.append(t['pnl_per_share'])
        tk = t['ticker']
        if tk not in price_dict:
            continue
        pxd2 = price_dict[tk]
        px2 = ensure_series(pxd2['Close'] if 'Close' in pxd2.columns else pxd2.iloc[:, 0])
        ed = t['entry_date']
        if ed not in px2.index:
            continue
        si = px2.index.get_loc(ed)
        confirmed = False
        for di in range(1, min(6, len(px2) - si)):
            if (to_scalar(px2.iloc[si + di]) - to_scalar(px2.iloc[si])) / to_scalar(px2.iloc[si]) <= -0.02:
                confirmed = True
                break
        if confirmed:
            confirmed_pnl.append(t['pnl_per_share'])
            confirmed_count += 1
        else:
            missed += 1
    if confirmed_count >= 10:
        ia = np.array(immediate_pnl)
        ca = np.array(confirmed_pnl)
        log.info(f"    Immediate: n={len(ia)} win={(ia>0).mean():.0%} avg=${ia.mean():+.2f}/sh")
        log.info(f"    Confirmed: n={len(ca)} win={(ca>0).mean():.0%} avg=${ca.mean():+.2f}/sh (missed {missed})")


def _spy_correlation_diagnostic(best_sc, spy_close):
    """Monthly correlation between equity curve and SPY."""
    if not best_sc or 'eq_df' not in best_sc or spy_close is None:
        return
    eq = best_sc['eq_df'].copy()
    eq['month'] = eq['date'].dt.to_period('M')
    eq_monthly = eq.groupby('month')['equity'].agg(['first', 'last'])
    eq_monthly['ret'] = (eq_monthly['last'] - eq_monthly['first']) / eq_monthly['first']
    spy_monthly = spy_close.resample('ME').agg(['first', 'last'])
    spy_monthly.columns = ['first', 'last']
    spy_monthly['ret'] = (spy_monthly['last'] - spy_monthly['first']) / spy_monthly['first']
    spy_monthly.index = spy_monthly.index.to_period('M')
    common = eq_monthly.index.intersection(spy_monthly.index)
    if len(common) > 5:
        corr = np.corrcoef(
            eq_monthly.loc[common, 'ret'].values,
            spy_monthly.loc[common, 'ret'].values,
        )[0, 1]
        log.info(f"\n  SPY MONTHLY CORRELATION: {corr:.2f}")
        if corr < -0.3:
            log.info(f"    \u2192 Useful portfolio hedge")
        elif abs(corr) < 0.2:
            log.info(f"    \u2192 Market-neutral")
        else:
            log.info(f"    \u2192 Correlated with market (unexpected for short strategy)")


def _earnings_proximity_diagnostic(data_bundle, wf_top):
    """Earnings proximity analysis for walk-forward trades."""
    cache = data_bundle['cache']
    cache_path = data_bundle['cache_path']
    log.info(f"\n  EARNINGS PROXIMITY:")
    earn_dates = cache.get('earnings_dates', {})
    if not earn_dates and len(wf_top) > 0:
        wf_tickers = list(wf_top['ticker'].unique())[:100]
        log.info(f"    Downloading earnings dates for {len(wf_tickers)} tickers...")
        for tk in wf_tickers:
            try:
                tkr = yf.Ticker(tk)
                cal = tkr.calendar
                if cal is not None and not cal.empty:
                    if 'Earnings Date' in cal.index:
                        ed_val = cal.loc['Earnings Date']
                        if hasattr(ed_val, 'tolist'):
                            earn_dates[tk] = ed_val.tolist()
                        else:
                            earn_dates[tk] = [ed_val]
                ehist = tkr.earnings_dates
                if ehist is not None and len(ehist) > 0:
                    earn_dates[tk] = sorted(ehist.index.tolist())
            except Exception:
                pass
        cache['earnings_dates'] = earn_dates
        from data import save_cache
        save_cache(cache, cache_path)
        log.info(f"    Got dates for {len(earn_dates)} tickers")

    if earn_dates and len(wf_top) > 10:
        near_earn = []
        far_earn = []
        for _, row in wf_top.iterrows():
            tk = row['ticker']
            ed = row['entry_date']
            if tk not in earn_dates:
                continue
            try:
                tk_earns = [pd.Timestamp(e) for e in earn_dates[tk]]
                tk_earns = [e for e in tk_earns if e >= ed]
            except Exception:
                continue
            if not tk_earns:
                continue
            days_to_earn = (tk_earns[0] - ed).days
            if days_to_earn <= 15:
                near_earn.append(row)
            elif days_to_earn > 30:
                far_earn.append(row)
        if len(near_earn) >= 5 and len(far_earn) >= 5:
            ne = pd.DataFrame(near_earn)
            fe = pd.DataFrame(far_earn)
            log.info(f"    Near earnings (\u226415d): n={len(ne)} "
                  f"win={(ne['pnl_per_share']>0).mean():.0%} "
                  f"avg=${ne['pnl_per_share'].mean():+.2f}/sh "
                  f"stop={ne['stopped'].mean():.0%}")
            log.info(f"    Far from earnings (>30d): n={len(fe)} "
                  f"win={(fe['pnl_per_share']>0).mean():.0%} "
                  f"avg=${fe['pnl_per_share'].mean():+.2f}/sh "
                  f"stop={fe['stopped'].mean():.0%}")
        else:
            log.info(f"    Insufficient earnings data (near={len(near_earn)}, far={len(far_earn)})")
    else:
        log.info(f"    No earnings dates available")
