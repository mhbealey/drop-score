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
    MIN_EVENTS, MAX_BASE_RATE, MIN_K,
)
from utils import elapsed


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
    px = pxd['Close'] if 'Close' in pxd.columns else pxd.iloc[:, 0]
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
        sp_ = float(px.iloc[si])
        if sp_ <= 0:
            results.append(oc)
            continue
        cur_vol = max(
            float(v30_s.iloc[si]) if si < len(v30_s) and pd.notna(v30_s.iloc[si]) else 0.3,
            0.05,
        )
        for w in FWD_WINDOWS:
            ei = si + w
            if ei >= len(px):
                continue
            ep = float(px.iloc[ei])
            er = (ep - sp_) / sp_
            oc[f'ret_{w}d_fwd'] = er
            if spy_c is not None:
                try:
                    ss = float(spy_c.asof(px.index[si]))
                    se = float(spy_c.asof(px.index[ei]))
                    if pd.notna(ss) and pd.notna(se) and ss > 0:
                        oc[f'excess_{w}d'] = er - (se - ss) / ss
                except:
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


def prepare_features(data_bundle):
    """
    Process cached intermediates: identify feature / outcome columns,
    recompute voladj outcomes if missing, select targets.
    Returns updated data_bundle with feature metadata.
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

    if intm_loaded:
        print("FEATURES: cached \u2705")
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
                print("  \u2705 Cached voladj to Drive")
            except:
                pass
    else:
        print("  \u26a0\ufe0f  No intermediates \u2014 run v15/v16 first")
        sys.exit(1)

    # Target selection: most relevant targets
    all_tgt = [
        c for c in df_dev.columns
        if (c.startswith('exdrop_') or c.startswith('drop_') or c.startswith('voladj_'))
        and df_dev[c].sum() >= MIN_EVENTS
    ]
    tgt_rates = {c: df_dev[c].mean() for c in all_tgt}
    all_viable = [c for c in all_tgt if tgt_rates[c] <= MAX_BASE_RATE]
    # Keep: all excess, all voladj, top 3 raw drops by events
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
