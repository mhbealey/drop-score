"""
Feature engineering, cached intermediates loading,
voladj outcome recompute if missing, target selection.
"""
import sys, time, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    FWD_WINDOWS, DROP_THRESH, EXCESS_THRESH,
    MIN_EVENTS, MAX_BASE_RATE, MIN_K, HOLDOUT_MO,
)
from utils import elapsed, get_col, to_scalar, ensure_series


def recompute_outcomes(df, price_dict, spy_close):
    """Recompute forward-return and vol-adjusted outcomes for a dataframe."""
    chunks = []
    for tk, grp in tqdm(df.groupby('ticker'), desc=f"  Outcomes"):
        c = _outcomes(grp, price_dict, spy_close)
        if len(c) > 0:
            chunks.append(c)
    if chunks:
        odf = pd.concat(chunks, ignore_index=True).set_index('_idx')
        return df.join(odf, how='inner')
    return df


def _outcomes(grp, pd_dict, spy_c):
    tk = grp['ticker'].iloc[0]
    if tk not in pd_dict:
        return pd.DataFrame()
    pxd = pd_dict[tk]
    px = ensure_series(pxd['Close'] if 'Close' in pxd.columns else pxd.iloc[:, 0])
    if len(px) < 63:
        return pd.DataFrame()
    dr_s = px.pct_change()
    v30_s = dr_s.rolling(30).std() * np.sqrt(252)
    results = []
    for idx, row in grp.iterrows():
        rd = row['report_date']
        oc = {'_idx': idx}
        vi = px.index[px.index >= rd]
        if len(vi) == 0:
            results.append(oc)
            continue
        si = px.index.get_loc(vi[0])
        sp_ = to_scalar(px.iloc[si])
        if sp_ <= 0:
            results.append(oc)
            continue
        cur_vol = max(
            to_scalar(v30_s.iloc[si]) if si < len(v30_s) and pd.notna(v30_s.iloc[si]) else 0.3,
            0.05,
        )
        for w in FWD_WINDOWS:
            ei = si + w
            if ei >= len(px):
                continue
            ep = to_scalar(px.iloc[ei])
            er = (ep - sp_) / sp_
            oc[f'ret_{w}d_fwd'] = er
            if spy_c is not None:
                try:
                    ss = to_scalar(spy_c.asof(px.index[si]))
                    se = to_scalar(spy_c.asof(px.index[ei]))
                    if pd.notna(ss) and pd.notna(se) and ss > 0:
                        oc[f'excess_{w}d'] = er - (se - ss) / ss
                except Exception:
                    pass
            ex = oc.get(f'excess_{w}d', np.nan)
            for t in EXCESS_THRESH:
                if pd.notna(ex):
                    oc[f'exdrop_{int(t*100)}_{w}d'] = 1 if ex <= -t else 0
            for t in DROP_THRESH:
                oc[f'drop_{int(t*100)}_{w}d'] = 1 if er <= -t else 0
            dv = cur_vol / np.sqrt(252) * np.sqrt(w)
            for sig in [1.0, 1.5, 2.0]:
                oc[f'voladj_{sig:.0f}sig_{w}d'] = 1 if er <= -(sig * dv) else 0
        results.append(oc)
    return pd.DataFrame(results)


def _safe_div(a, b, default=np.nan):
    """Safe division avoiding divide-by-zero."""
    if b is None or b == 0 or np.isnan(b):
        return default
    return a / b


def _safe(v, default=np.nan):
    """Return default if v is NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return float(v)


def build_quarterly_row(tk, idx, df_inc, df_bal, df_cf, price_dict, spy_close,
                        sector_map, sector_etf_ret):
    """Build one quarterly feature row for a ticker at a given report index."""
    row = {'ticker': tk}

    # Get report date
    try:
        rd = idx[1] if isinstance(idx, tuple) else idx
        row['report_date'] = pd.Timestamp(rd)
    except Exception:
        return None

    # === Income statement ===
    rev = get_col(df_inc, idx, 'Revenue')
    gp = get_col(df_inc, idx, 'Gross Profit')
    oi = get_col(df_inc, idx, 'Operating Income (Loss)')
    ni = get_col(df_inc, idx, 'Net Income')
    ie = get_col(df_inc, idx, 'Interest Expense, Net')

    # === Balance sheet ===
    ta = get_col(df_bal, idx, 'Total Assets')
    te = get_col(df_bal, idx, 'Total Equity')
    tl = get_col(df_bal, idx, 'Total Liabilities')
    td = get_col(df_bal, idx, 'Total Debt')
    tca = get_col(df_bal, idx, 'Total Current Assets')
    tcl = get_col(df_bal, idx, 'Total Current Liabilities')
    cash = get_col(df_bal, idx, 'Cash, Cash Equivalents & Short Term Investments')
    shares = get_col(df_bal, idx, 'Shares (Diluted)')
    shares_basic = get_col(df_bal, idx, 'Shares (Basic)')

    # === Cash flow ===
    cfo = get_col(df_cf, idx, 'Net Cash from Operating Activities')
    capex = get_col(df_cf, idx, 'Change in Fixed Assets & Intangibles')

    # === Profitability ratios ===
    row['gross_margin'] = _safe_div(gp, rev)
    row['operating_margin'] = _safe_div(oi, rev)
    row['net_margin'] = _safe_div(ni, rev)
    row['roe'] = _safe_div(ni * 4, te) if not np.isnan(_safe(ni)) else np.nan  # annualized
    row['roa'] = _safe_div(ni * 4, ta) if not np.isnan(_safe(ni)) else np.nan

    # === Leverage ===
    row['debt_to_equity'] = _safe_div(td, te)
    row['debt_to_assets'] = _safe_div(td, ta)
    row['current_ratio'] = _safe_div(tca, tcl)
    row['interest_coverage'] = _safe_div(oi, abs(ie)) if not np.isnan(_safe(ie)) and ie != 0 else np.nan
    row['liab_to_assets'] = _safe_div(tl, ta)
    row['cash_to_assets'] = _safe_div(cash, ta)
    row['cash_to_debt'] = _safe_div(cash, td)
    row['net_debt'] = _safe(td, 0) - _safe(cash, 0)
    row['net_debt_to_equity'] = _safe_div(row['net_debt'], te)

    # === Cash flow ===
    fcf = _safe(cfo, 0) - abs(_safe(capex, 0))
    row['fcf'] = fcf
    row['fcf_margin'] = _safe_div(fcf, rev)
    row['cfo_to_revenue'] = _safe_div(cfo, rev)
    row['capex_to_revenue'] = _safe_div(abs(_safe(capex, 0)), rev)
    row['fcf_to_debt'] = _safe_div(fcf, td)
    row['accruals'] = _safe_div(_safe(ni, 0) - _safe(cfo, 0), ta)

    # === Per-share metrics ===
    sh = _safe(shares, _safe(shares_basic, np.nan))
    row['eps'] = _safe_div(ni, sh)
    row['bvps'] = _safe_div(te, sh)
    row['rev_per_share'] = _safe_div(rev, sh)
    row['fcf_per_share'] = _safe_div(fcf, sh)

    # === Share dilution ===
    row['shares'] = sh
    row['dilution'] = _safe_div(
        _safe(shares, 0) - _safe(shares_basic, 0),
        _safe(shares_basic, 1)
    )

    # === Size ===
    row['log_assets'] = np.log1p(_safe(ta, 0))
    row['log_revenue'] = np.log1p(_safe(rev, 0))

    # === Price-based features (at report date) ===
    _add_price_features(row, tk, price_dict, spy_close, ni, te, rev, fcf, sh)

    # === Sector ===
    row['sector'] = sector_map.get(tk, 'Other')

    # === Sector-relative momentum ===
    _add_sector_relative(row, tk, price_dict, sector_etf_ret)

    return row


def _add_price_features(row, tk, price_dict, spy_close, ni, te, rev, fcf, sh):
    """Add price-based features to a quarterly row."""
    if tk not in price_dict:
        return
    pxd = price_dict[tk]
    px = ensure_series(pxd['Close'] if 'Close' in pxd.columns else pxd.iloc[:, 0])
    vol_col = ensure_series(pxd['Volume']) if 'Volume' in pxd.columns else None
    rd = row['report_date']
    vi = px.index[px.index >= rd]
    if len(vi) == 0:
        return
    si = px.index.get_loc(vi[0])
    price = to_scalar(px.iloc[si])
    row['price'] = price
    row['market_cap'] = price * _safe(sh, 0)

    if vol_col is not None and si >= 30:
        avg_vol = to_scalar(vol_col.iloc[max(0, si-30):si].mean())
        row['avg_vol'] = avg_vol
        row['dollar_volume'] = avg_vol * price

    dr = px.pct_change()
    for lb, lbl in [(5, '5d'), (21, '21d'), (63, '63d'), (126, '126d'), (252, '252d')]:
        if si >= lb:
            row[f'mom_{lbl}'] = to_scalar(px.iloc[si] / px.iloc[si - lb] - 1)

    if si >= 30:
        row['vol_30d'] = to_scalar(dr.iloc[max(0, si-30):si].std() * np.sqrt(252))
    if si >= 60:
        row['vol_60d'] = to_scalar(dr.iloc[max(0, si-60):si].std() * np.sqrt(252))
    if si >= 252:
        row['vol_252d'] = to_scalar(dr.iloc[max(0, si-252):si].std() * np.sqrt(252))

    if si >= 60:
        v30 = dr.iloc[max(0, si-30):si].std()
        v60 = dr.iloc[max(0, si-60):si].std()
        row['vol_ratio_30_60'] = _safe_div(to_scalar(v30), to_scalar(v60))

    if si >= 63:
        window = px.iloc[max(0, si-63):si+1]
        running_max = window.cummax()
        dd = (window - running_max) / running_max
        row['max_dd_63d'] = to_scalar(dd.min())

    if si >= 252:
        h52 = to_scalar(px.iloc[max(0, si-252):si+1].max())
        l52 = to_scalar(px.iloc[max(0, si-252):si+1].min())
        row['dist_52w_high'] = (price - h52) / h52 if h52 > 0 else np.nan
        row['dist_52w_low'] = (price - l52) / l52 if l52 > 0 else np.nan

    if vol_col is not None and si >= 60:
        v30_avg = to_scalar(vol_col.iloc[max(0, si-30):si].mean())
        v60_avg = to_scalar(vol_col.iloc[max(0, si-60):si].mean())
        row['vol_trend'] = _safe_div(v30_avg, v60_avg)

    if price > 0 and sh > 0:
        row['earnings_yield'] = _safe_div(ni * 4, price * sh)
        row['book_to_price'] = _safe_div(te, price * sh)
        row['sales_to_price'] = _safe_div(rev * 4, price * sh)
        row['fcf_yield'] = _safe_div(fcf * 4, price * sh)

    if spy_close is not None and si >= 63:
        try:
            stk_ret = dr.iloc[max(0, si-252):si]
            spy_s = ensure_series(spy_close)
            spy_ret_aligned = spy_s.pct_change().reindex(stk_ret.index)
            valid = stk_ret.notna() & spy_ret_aligned.notna()
            if valid.sum() >= 60:
                cov = np.cov(stk_ret[valid].values, spy_ret_aligned[valid].values)
                row['beta'] = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
        except Exception:
            pass


def _add_sector_relative(row, tk, price_dict, sector_etf_ret):
    """Add sector-relative momentum to a quarterly row."""
    sec = row.get('sector', 'Other')
    if sec not in sector_etf_ret or tk not in price_dict:
        return
    try:
        pxd = price_dict[tk]
        px = ensure_series(pxd['Close'] if 'Close' in pxd.columns else pxd.iloc[:, 0])
        rd = row['report_date']
        vi = px.index[px.index >= rd]
        if len(vi) > 0:
            si = px.index.get_loc(vi[0])
            sec_ret = ensure_series(sector_etf_ret.get(sec))
            if sec_ret is not None and si >= 63:
                stk_63 = to_scalar(px.iloc[si] / px.iloc[si-63] - 1)
                sec_idx = sec_ret.index[sec_ret.index <= px.index[si]]
                if len(sec_idx) >= 63:
                    sec_63 = to_scalar((1 + sec_ret.loc[sec_idx[-63:]]).prod() - 1)
                    row['sector_rel_mom_63d'] = stk_63 - sec_63
    except Exception:
        pass


def build_features_from_scratch(data_bundle):
    """Build quarterly features from SimFin + price data when no cached intermediates exist."""
    print("FEATURES: building from scratch...")

    df_inc = data_bundle['df_inc']
    df_bal = data_bundle['df_bal']
    df_cf = data_bundle['df_cf']
    price_dict = data_bundle['price_dict']
    spy_close = data_bundle['spy_close']
    sector_map = data_bundle['sector_map']
    sector_etf_ret = data_bundle['sector_etf_ret']
    universe = data_bundle['universe']
    intermediates_path = data_bundle['intermediates_path']

    # Build quarterly feature rows
    rows = []
    for tk in tqdm(universe, desc="  Building features"):
        if tk not in df_inc.index.get_level_values('Ticker'):
            continue
        tk_inc = df_inc.loc[tk]
        for report_date in tk_inc.index:
            idx = (tk, report_date)
            try:
                r = build_quarterly_row(
                    tk, idx, df_inc, df_bal, df_cf, price_dict,
                    spy_close, sector_map, sector_etf_ret,
                )
                if r is not None:
                    rows.append(r)
            except Exception:
                continue

    df_q = pd.DataFrame(rows)
    print(f"  Raw quarterly rows: {len(df_q):,}")

    if len(df_q) == 0:
        print("  ERROR: No features built")
        sys.exit(1)

    # === YoY growth features (need groupby ticker, sorted by date) ===
    df_q = df_q.sort_values(['ticker', 'report_date']).reset_index(drop=True)
    for col, name in [('gross_margin', 'gm'), ('operating_margin', 'om'),
                       ('net_margin', 'nm'), ('eps', 'eps'), ('rev_per_share', 'rps')]:
        if col in df_q.columns:
            df_q[f'{name}_chg_4q'] = df_q.groupby('ticker')[col].transform(
                lambda x: x - x.shift(4)
            )

    # Revenue growth YoY
    if 'rev_per_share' in df_q.columns:
        df_q['rev_growth_yoy'] = df_q.groupby('ticker')['rev_per_share'].transform(
            lambda x: x / x.shift(4) - 1
        )

    # Earnings growth YoY
    if 'eps' in df_q.columns:
        df_q['eps_growth_yoy'] = df_q.groupby('ticker')['eps'].transform(
            lambda x: x / x.shift(4) - 1
        )

    # Sequential (QoQ) changes
    for col in ['gross_margin', 'operating_margin', 'net_margin', 'current_ratio',
                'debt_to_equity', 'accruals']:
        if col in df_q.columns:
            df_q[f'{col}_qoq'] = df_q.groupby('ticker')[col].transform(
                lambda x: x - x.shift(1)
            )

    # Share dilution YoY
    if 'shares' in df_q.columns:
        df_q['share_chg_yoy'] = df_q.groupby('ticker')['shares'].transform(
            lambda x: x / x.shift(4) - 1
        )

    # === Interaction features ===
    if 'mom_63d' in df_q.columns and 'accruals' in df_q.columns:
        df_q['mom_x_accruals'] = df_q['mom_63d'] * df_q['accruals']
    if 'mom_63d' in df_q.columns and 'vol_30d' in df_q.columns:
        df_q['mom_x_vol'] = df_q['mom_63d'] * df_q['vol_30d']
    if 'earnings_yield' in df_q.columns and 'mom_63d' in df_q.columns:
        df_q['ey_x_mom'] = df_q['earnings_yield'] * df_q['mom_63d']
    if 'fcf_yield' in df_q.columns and 'debt_to_equity' in df_q.columns:
        df_q['fcf_x_leverage'] = df_q['fcf_yield'] * df_q['debt_to_equity']

    # === Clean infinities ===
    numeric_cols = df_q.select_dtypes(include=[np.number]).columns
    df_q[numeric_cols] = df_q[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # === Clip extreme values ===
    for col in numeric_cols:
        if col in ('price', 'market_cap', 'avg_vol', 'dollar_volume', 'shares',
                   'fcf', 'net_debt'):
            continue
        q01 = df_q[col].quantile(0.01)
        q99 = df_q[col].quantile(0.99)
        df_q[col] = df_q[col].clip(q01, q99)

    # === Filter: need price and report_date ===
    df_q = df_q.dropna(subset=['price', 'report_date']).reset_index(drop=True)
    print(f"  After filtering: {len(df_q):,} rows, {df_q['ticker'].nunique()} tickers")

    # === Dev / Holdout split ===
    cutoff = df_q['report_date'].max() - pd.DateOffset(months=HOLDOUT_MO)
    df_dev = df_q[df_q['report_date'] < cutoff].copy().reset_index(drop=True)
    df_hold = df_q[df_q['report_date'] >= cutoff].copy().reset_index(drop=True)
    print(f"  Dev: {len(df_dev):,} | Hold: {len(df_hold):,} (cutoff: {cutoff.date()})")

    # === Compute outcomes ===
    print("  Computing outcomes...")
    df_dev = recompute_outcomes(df_dev, price_dict, spy_close)
    df_hold = recompute_outcomes(df_hold, price_dict, spy_close)

    # === Build daily dataframe (stub for compatibility) ===
    df_daily = pd.DataFrame()

    # === Save intermediates ===
    try:
        with open(intermediates_path, 'wb') as f:
            pickle.dump({
                'df_q': df_q, 'df_dev': df_dev,
                'df_hold': df_hold, 'df_daily': df_daily,
            }, f)
        print(f"  Saved intermediates to {intermediates_path}")
    except Exception as e:
        print(f"  Warning: could not save intermediates: {e}")

    return df_q, df_dev, df_hold, df_daily


def prepare_features(data_bundle):
    """
    Process cached intermediates or build from scratch.
    Identify feature / outcome columns, recompute voladj outcomes if missing,
    select targets. Returns updated data_bundle with feature metadata.
    """
    t0 = time.time()
    meta = {'ticker', 'report_date', 'price', 'avg_vol', 'market_cap', 'sector',
            'vuln_score', 'wf_score'}
    opfx = ('ret_', 'excess_', 'drop_', 'exdrop_', 'fwd_ret', 'voladj_')

    intm_loaded = data_bundle['intm_loaded']
    df_dev = data_bundle['df_dev']
    df_hold = data_bundle['df_hold']
    df_q = data_bundle['df_q']
    df_daily = data_bundle['df_daily']
    price_dict = data_bundle['price_dict']
    spy_close = data_bundle['spy_close']
    intermediates_path = data_bundle['intermediates_path']

    if not intm_loaded:
        # Build features from scratch
        df_q, df_dev, df_hold, df_daily = build_features_from_scratch(data_bundle)
        data_bundle['df_q'] = df_q
        data_bundle['df_dev'] = df_dev
        data_bundle['df_hold'] = df_hold
        data_bundle['df_daily'] = df_daily
        data_bundle['intm_loaded'] = True

    print("FEATURES: processing...")
    ocols = set(c for c in df_dev.columns if any(c.startswith(p) for p in opfx))
    fcols_q = sorted([
        c for c in df_dev.columns
        if c not in meta and c not in ocols
        and df_dev[c].dtype in ('float64', 'int64', 'float32', 'int32')
    ])
    fill_meds_q = df_dev[fcols_q].median()
    has_voladj = any(c.startswith('voladj_') for c in df_dev.columns)
    if not has_voladj:
        print("  Recomputing outcomes (voladj)...")
        old_oc = [c for c in df_dev.columns
                  if any(c.startswith(p) for p in opfx) and c not in fcols_q]
        df_dev.drop(columns=old_oc, inplace=True, errors='ignore')
        df_hold.drop(columns=old_oc, inplace=True, errors='ignore')
        for lbl, dft_name in [('Dev', 'df_dev'), ('Hold', 'df_hold')]:
            dft = df_dev if dft_name == 'df_dev' else df_hold
            chunks = []
            for tk, grp in tqdm(dft.groupby('ticker'), desc=f"  {lbl}"):
                c = _outcomes(grp, price_dict, spy_close)
                if len(c) > 0:
                    chunks.append(c)
            if chunks:
                odf = pd.concat(chunks, ignore_index=True).set_index('_idx')
                if lbl == 'Dev':
                    df_dev = df_dev.join(odf, how='inner')
                else:
                    df_hold = df_hold.join(odf, how='inner')
        ocols = set(c for c in df_dev.columns if any(c.startswith(p) for p in opfx))
        fcols_q = sorted([
            c for c in df_dev.columns
            if c not in meta and c not in ocols
            and df_dev[c].dtype in ('float64', 'int64', 'float32', 'int32')
        ])
        fill_meds_q = df_dev[fcols_q].median()
        try:
            with open(intermediates_path, 'wb') as f:
                pickle.dump({
                    'df_q': df_q, 'df_dev': df_dev,
                    'df_hold': df_hold, 'df_daily': df_daily,
                }, f)
            print("  Cached voladj")
        except Exception:
            pass

    # Target selection
    all_tgt = [
        c for c in df_dev.columns
        if (c.startswith('exdrop_') or c.startswith('drop_') or c.startswith('voladj_'))
        and df_dev[c].sum() >= MIN_EVENTS
    ]
    tgt_rates = {c: df_dev[c].mean() for c in all_tgt}
    all_viable = [c for c in all_tgt if tgt_rates[c] <= MAX_BASE_RATE]
    ex_tgts = sorted([c for c in all_viable if c.startswith('exdrop_')])
    va_tgts = sorted([c for c in all_viable if c.startswith('voladj_')])
    raw_tgts = sorted(
        [c for c in all_viable if c.startswith('drop_')],
        key=lambda c: df_dev[c].sum(), reverse=True,
    )[:3]
    tcols = list(set(ex_tgts + va_tgts + raw_tgts))
    print(f"  Dev: {len(df_dev):,} | Hold: {len(df_hold):,} | "
          f"Training {len(tcols)} targets ({len(ex_tgts)} ex, {len(va_tgts)} va, {len(raw_tgts)} raw)")
    print(f"  [{time.time()-t0:.0f}s] {elapsed()}")
    print()

    data_bundle.update(
        df_dev=df_dev, df_hold=df_hold,
        fcols_q=fcols_q, fill_meds_q=fill_meds_q,
        tcols=tcols, tgt_rates=tgt_rates,
        ex_tgts=ex_tgts, va_tgts=va_tgts, raw_tgts=raw_tgts,
    )
    return data_bundle
