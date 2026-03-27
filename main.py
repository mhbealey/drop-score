"""
DROP SCORE v18 — LOCKED CONFIG + DUAL UNIVERSE + BORROW COSTS
Locked: exdrop_15_10d / 21d hold / confirmed entry.
Multi-source price waterfall. S&P index vs full universe comparison.
Volume-based borrow costs. Training/tradeable universe split.
"""
import subprocess, sys, os, time, warnings, random, pickle
from datetime import datetime

# ── Install dependencies if needed ──
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["simfin", "yfinance", "xgboost", "lightgbm", "scikit-learn",
            "matplotlib", "tqdm", "lxml", "html5lib"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        install(pkg)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

# ── Tee stdout to results/ ──
os.makedirs('results', exist_ok=True)
_log_path = os.path.join('results', f"run_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.txt")

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_log_file = open(_log_path, 'w')
sys.stdout = _Tee(sys.__stdout__, _log_file)

# ── Project modules ──
from config import (
    N_BOOT, N_FOLDS, VOL_FLOOR, t_start,
    TRADING_TARGET, TRADING_HOLD, ENTRY_MODE,
    BORROW_RATE_EASY, BORROW_RATE_HARD,
    UNIVERSE_MODE, SECTOR_ETFS,
)
from utils import elapsed, clean_X, to_scalar, ensure_series
from data import load_all_data, get_sp_index_tickers
from features import prepare_features
from model import run_vulnerability_model, run_model
from walkforward import run_walkforward
from equity import run_equity_scenarios

# ═══════════════════════════════════════════════════════════════
# BANNER
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("DROP SCORE v18 — LOCKED CONFIG + DUAL UNIVERSE + BORROW COSTS")
print(f"  Target: {TRADING_TARGET} | Hold: {TRADING_HOLD}d | Entry: {ENTRY_MODE}")
print(f"  Borrow: {BORROW_RATE_EASY:.0%}/{BORROW_RATE_HARD:.0%} | Universe: {UNIVERSE_MODE}")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════
data = load_all_data()

# ═══════════════════════════════════════════════════════════════
# 2. FEATURES (computed once on full dataset)
# ═══════════════════════════════════════════════════════════════
data = prepare_features(data)


# ═══════════════════════════════════════════════════════════════
# PIPELINE: model → walk-forward → equity (reusable per universe)
# ═══════════════════════════════════════════════════════════════

def run_pipeline(data_bundle, label, ticker_subset=None):
    """Run model + walk-forward + equity for a universe subset.

    If ticker_subset is given, filters df_dev/df_hold to those tickers
    and recomputes fill medians. Features are NOT recomputed.
    """
    bundle = dict(data_bundle)  # shallow copy

    if ticker_subset is not None:
        subset = set(ticker_subset)
        df_dev = bundle['df_dev']
        df_hold = bundle['df_hold']
        bundle['df_dev'] = df_dev[df_dev['ticker'].isin(subset)].copy()
        bundle['df_hold'] = df_hold[df_hold['ticker'].isin(subset)].copy()
        # Recompute fill medians for the subset
        fcols_q = bundle['fcols_q']
        bundle['fill_meds_q'] = bundle['df_dev'][fcols_q].median()
        # Also restrict tradeable tickers
        if bundle.get('tradeable_tickers'):
            bundle['tradeable_tickers'] = bundle['tradeable_tickers'] & subset

    print("=" * 70)
    print(f"PIPELINE: {label}")
    n_dev = len(bundle['df_dev'])
    n_hold = len(bundle['df_hold'])
    n_tickers = bundle['df_dev']['ticker'].nunique()
    print(f"  {n_tickers} tickers | {n_dev:,} dev + {n_hold:,} holdout rows")
    print("=" * 70)

    bundle = run_vulnerability_model(bundle)
    bundle = run_walkforward(bundle)
    bundle = run_equity_scenarios(bundle)

    return bundle


def holdout_eval(bundle, label):
    """Evaluate holdout on TRADING_TARGET and voladj best."""
    print(f"\n  HOLDOUT ({label}):")
    best_v_t = bundle['best_v_t']
    best_v_r = bundle['best_v_r']
    df_hold = bundle['df_hold']
    v_results = bundle['v_results']

    results = {}

    # Evaluate on both the Pareto-best target and TRADING_TARGET
    targets_to_eval = [best_v_t]
    if TRADING_TARGET != best_v_t and TRADING_TARGET in v_results:
        targets_to_eval.append(TRADING_TARGET)

    # Also evaluate best voladj target
    va_tgts = [t for t in v_results if t.startswith('voladj_')]
    best_va = max(va_tgts, key=lambda k: v_results[k]['mauc']) if va_tgts else None
    if best_va and best_va not in targets_to_eval:
        targets_to_eval.append(best_va)

    for tgt in targets_to_eval:
        ho_auc = np.nan
        try:
            if tgt not in df_hold.columns:
                print(f"    {tgt}: column not in holdout set")
                continue
            yh = df_hold[tgt].fillna(0).astype(int)
            n_pos = int(yh.sum())
            n_neg = int((yh == 0).sum())
            n_nan = int(df_hold[tgt].isna().sum())
            print(f"    {tgt}: events pos={n_pos} neg={n_neg} nan={n_nan}")
            if n_pos < 20:
                print(f"      \u26a0\ufe0f Too few events for reliable holdout AUC on {tgt}")
            if n_pos < 10 or 'vuln_score' not in df_hold.columns:
                continue
            hp = df_hold['vuln_score'].values
            valid = ~np.isnan(hp)
            if valid.sum() >= 20 and yh[valid].sum() >= 5:
                ho_auc = roc_auc_score(yh[valid], hp[valid])
                htk = df_hold['ticker'].values[valid]
                utk = np.unique(htk)
                tidx2 = {t: np.where(htk == t)[0] for t in utk}
                hb = []
                for _ in range(N_BOOT):
                    bt2 = np.random.choice(utk, len(utk), replace=True)
                    idx = np.concatenate([tidx2[t] for t in bt2])
                    if (len(idx) > 0
                            and yh.values[valid][idx].sum() > 0
                            and yh.values[valid][idx].sum() < len(idx)):
                        try:
                            hb.append(roc_auc_score(yh.values[valid][idx], hp[valid][idx]))
                        except:
                            pass
                hcl = np.percentile(hb, 2.5) if hb else ho_auc
                hch = np.percentile(hb, 97.5) if hb else ho_auc
                dev_auc = v_results[tgt]['mauc']
                marker = " <-- TRADING" if tgt == TRADING_TARGET else ""
                print(f"    {tgt}: Dev={dev_auc:.3f} Hold={ho_auc:.3f} "
                      f"[{hcl:.3f},{hch:.3f}]{marker}")
                results[tgt] = ho_auc
        except Exception as e:
            print(f"    {tgt}: error \u2014 {e}")

    return results


# ═══════════════════════════════════════════════════════════════
# 3. RUN PIPELINE(S)
# ═══════════════════════════════════════════════════════════════

pipeline_results = {}

if UNIVERSE_MODE in ("full", "both"):
    # Universe A: full SimFin universe
    print("\n" + "#" * 70)
    print("# UNIVERSE A: Full SimFin")
    print("#" * 70)
    result_a = run_pipeline(data, "Full SimFin")
    ho_a = holdout_eval(result_a, "Full SimFin")
    pipeline_results['Full SimFin'] = result_a
    t_a = time.time() - t_start

if UNIVERSE_MODE in ("sp_index", "both"):
    # Time budget: skip Universe B if >25 min already
    elapsed_min = (time.time() - t_start) / 60
    if UNIVERSE_MODE == "both" and elapsed_min > 25:
        print(f"\n  TIME BUDGET: {elapsed_min:.0f} min elapsed, skipping S&P index pipeline")
    else:
        print("\n" + "#" * 70)
        print("# UNIVERSE B: S&P 400+600 Index")
        print("#" * 70)
        sp_tickers = get_sp_index_tickers()
        if sp_tickers:
            # Filter to tickers we have data for
            all_tickers = set(data['df_dev']['ticker'].unique())
            sp_overlap = sp_tickers & all_tickers
            print(f"  S&P index overlap with our data: {len(sp_overlap)}/{len(sp_tickers)}")
            if len(sp_overlap) >= 50:
                result_b = run_pipeline(data, "S&P 400+600", ticker_subset=sp_overlap)
                ho_b = holdout_eval(result_b, "S&P 400+600")
                pipeline_results['S&P 400+600'] = result_b
            else:
                print(f"  Too few overlapping tickers ({len(sp_overlap)}), skipping")
        else:
            print("  Could not get S&P index tickers, skipping")

# If only one mode requested
if UNIVERSE_MODE == "full" and 'Full SimFin' not in pipeline_results:
    result_a = run_pipeline(data, "Full SimFin")
    ho_a = holdout_eval(result_a, "Full SimFin")
    pipeline_results['Full SimFin'] = result_a

if UNIVERSE_MODE == "sp_index" and 'S&P 400+600' not in pipeline_results:
    # Already handled above
    pass


# ═══════════════════════════════════════════════════════════════
# VOLADJ COLLAPSE DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════

# Run on the primary (full) universe result
_diag_label = 'Full SimFin' if 'Full SimFin' in pipeline_results else (
    list(pipeline_results.keys())[0] if pipeline_results else None
)
if _diag_label:
    _diag = pipeline_results[_diag_label]
    _df_dev = _diag['df_dev']
    _df_hold = _diag['df_hold']
    _fcols_q = _diag['fcols_q']
    _fill_meds = _diag['fill_meds_q']
    _v_results = _diag['v_results']
    _K = _diag['K']
    _price_dict = data['price_dict']

    _va_tgt = 'voladj_2sig_63d'
    if _va_tgt in _v_results:
        print("\n" + "=" * 70)
        print("VOLADJ COLLAPSE DIAGNOSTIC")
        print("=" * 70)
        _va_dev_auc = _v_results[_va_tgt]['mauc']

        # ── Test 1: Truncation check ──
        _full_window = 0
        _truncated = 0
        for _idx, _row in _df_hold.iterrows():
            _tk = _row['ticker']
            if _tk not in _price_dict:
                _truncated += 1
                continue
            _pxd = _price_dict[_tk]
            _px = ensure_series(
                _pxd['Close'] if 'Close' in _pxd.columns else _pxd.iloc[:, 0]
            )
            _vi = _px.index[_px.index >= _row['report_date']]
            if len(_vi) == 0:
                _truncated += 1
                continue
            _si = _px.index.get_loc(_vi[0])
            if _si + 63 < len(_px):
                _full_window += 1
            else:
                _truncated += 1
        print(f"\n  Test 1 -- Truncation: {_full_window} full 63d window, {_truncated} truncated")
        if _truncated > len(_df_hold) * 0.20:
            print(f"    \u26a0\ufe0f {_truncated/len(_df_hold):.0%} truncated"
                  f" -- this likely explains the low holdout AUC")

        # ── Test 2: Redefine voladj using 252-day trailing vol ──
        print(f"\n  Test 2 -- 252d vol threshold:")
        for _lbl, _dft in [('Dev', _df_dev), ('Hold', _df_hold)]:
            _new_labels = []
            for _idx, _row in _dft.iterrows():
                _tk = _row['ticker']
                if _tk not in _price_dict:
                    _new_labels.append(np.nan)
                    continue
                _pxd = _price_dict[_tk]
                _px = ensure_series(
                    _pxd['Close'] if 'Close' in _pxd.columns else _pxd.iloc[:, 0]
                )
                _vi = _px.index[_px.index >= _row['report_date']]
                if len(_vi) == 0:
                    _new_labels.append(np.nan)
                    continue
                _si = _px.index.get_loc(_vi[0])
                if _si >= 252:
                    _dr = _px.pct_change()
                    _vol_252 = float(_dr.iloc[_si - 251:_si + 1].std() * np.sqrt(252))
                else:
                    _vol_252 = 0.3
                _vol_252 = max(_vol_252, 0.05)
                _period_vol = (_vol_252 / np.sqrt(252)) * np.sqrt(63)
                _threshold = -2.0 * _period_vol
                if _si + 63 < len(_px):
                    _ret = (float(_px.iloc[_si + 63]) - float(_px.iloc[_si])) / float(_px.iloc[_si])
                    _new_labels.append(1 if _ret <= _threshold else 0)
                else:
                    _new_labels.append(np.nan)
            _dft['voladj_2sig_63d_v252'] = _new_labels

        _r252 = run_model(_df_dev, _fcols_q, 'voladj_2sig_63d_v252', _fill_meds, N_FOLDS, 100, _K)
        if _r252:
            print(f"    Dev AUC: {_r252['mauc']:.3f}")
            _lf252 = _r252['folds'][-1]
            _Xh252 = clean_X(_df_hold, _fcols_q, _fill_meds)
            try:
                _hp252 = _lf252['model'].predict_proba(_Xh252[_lf252['feats']])[:, 1]
                _yh252 = _df_hold['voladj_2sig_63d_v252'].fillna(0).astype(int)
                _valid252 = ~_df_hold['voladj_2sig_63d_v252'].isna()
                if _valid252.sum() > 50 and _yh252[_valid252].sum() >= 5:
                    _ho252 = roc_auc_score(_yh252[_valid252], _hp252[_valid252])
                    _n_pos252 = int(_yh252[_valid252].sum())
                    _n_neg252 = int((~_yh252[_valid252].astype(bool)).sum())
                    print(f"    Holdout AUC: {_ho252:.3f}")
                    print(f"    Events: {_n_pos252} pos, {_n_neg252} neg")
                    if _ho252 > 0.55:
                        print(f"    \u2705 252d vol decoupling FIXES the holdout"
                              f" -- original collapse was vol-regime dependent")
                    else:
                        print(f"    \u274c Still collapsed"
                              f" -- problem is deeper than vol definition")
                else:
                    print(f"    Holdout: too few valid samples or events")
            except Exception as _e:
                print(f"    Holdout scoring failed: {_e}")
        else:
            print(f"    run_model returned None for v252 target")

        # ── Test 3: Pure fundamental model on holdout ──
        _fund_exclude = {
            'vol_30d', 'vol_60d', 'vol_90d', 'ret_5d', 'ret_21d', 'ret_63d',
            'dd_from_high', 'gap_count_30d', 'down_days_30d', 'death_cross',
            'excess_ret_21d', 'excess_ret_63d', 'sector_excess_21d',
            'sector_excess_63d', 'consec_down_days', 'gap_down_today',
            'gap_downs_5d', 'spy_corr_60d', 'roa_x_vol', 'margin_x_vol',
            'margin_trend_x_vol',
        }
        _fund_only = [f for f in _fcols_q if f not in _fund_exclude]
        if len(_fund_only) >= 5:
            print(f"\n  Test 3 -- Fund-only model ({len(_fund_only)} features):")
            _fund_meds = _df_dev[_fund_only].median()
            _r_fund = run_model(
                _df_dev, _fund_only, _va_tgt, _fund_meds,
                N_FOLDS, 100, min(_K, len(_fund_only)),
            )
            if _r_fund:
                print(f"    Dev AUC: {_r_fund['mauc']:.3f}")
                _lf_f = _r_fund['folds'][-1]
                _Xh_f = clean_X(_df_hold, _fund_only, _fund_meds)
                try:
                    _hp_f = _lf_f['model'].predict_proba(_Xh_f[_lf_f['feats']])[:, 1]
                    _yh_f = _df_hold[_va_tgt].fillna(0).astype(int)
                    _ho_f = roc_auc_score(_yh_f, _hp_f)
                    print(f"    Holdout AUC: {_ho_f:.3f}")
                    if _ho_f > 0.55:
                        print(f"    \u2705 Fundamental signal survives"
                              f" -- vol features caused the collapse")
                    else:
                        print(f"    \u274c Fundamentals also collapse"
                              f" -- signal is regime-specific")
                except Exception as _e:
                    print(f"    Holdout: failed ({_e})")

        # ── Summary ──
        _ho252_val = _ho252 if '_ho252' in locals() else None
        _ho_f_val = _ho_f if '_ho_f' in locals() else None
        _r_fund_val = _r_fund if '_r_fund' in locals() else None
        print(f"\n  VOLADJ SUMMARY:")
        print(f"    Original:  Dev={_va_dev_auc:.3f}")
        if _r252:
            _s252 = f"    252d-vol:  Dev={_r252['mauc']:.3f}"
            if _ho252_val is not None:
                _s252 += f" Hold={_ho252_val:.3f}"
            print(_s252)
        if len(_fund_only) >= 5 and _r_fund_val:
            _sf = f"    Fund-only: Dev={_r_fund_val['mauc']:.3f}"
            if _ho_f_val is not None:
                _sf += f" Hold={_ho_f_val:.3f}"
            print(_sf)
        print(f"    Truncation: {_truncated}/{len(_df_hold)}"
              f" ({_truncated/len(_df_hold):.0%}) holdout rows lack full 63d window")
        print()


# ═══════════════════════════════════════════════════════════════
# 4. HEAD-TO-HEAD COMPARISON (if both universes ran)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("HEAD-TO-HEAD COMPARISON")
print("=" * 70)

if len(pipeline_results) >= 2:
    print(f"\n  {'Universe':<20} {'Tickers':>8} {'DevAUC':>8} {'WF Trades':>10} "
          f"{'Top25%Win':>10} {'Top25%P&L':>10} {'Stops':>6}")
    print(f"  {'-'*75}")
    for lbl, res in pipeline_results.items():
        n_tk = res['df_dev']['ticker'].nunique()
        dev_auc = res['v_results'].get(TRADING_TARGET, {}).get('mauc', np.nan)
        wf = res.get('wf_df', pd.DataFrame())
        wf_top = res.get('wf_top', pd.DataFrame())
        n_wf = len(wf)
        if len(wf_top) > 0:
            t25w = (wf_top['pnl_per_share'] > 0).mean()
            t25p = wf_top['pnl_per_share'].mean()
            stops = wf['stopped'].mean() if len(wf) > 0 else np.nan
        else:
            t25w = t25p = stops = np.nan
        t25w_s = f"{t25w:.0%}" if pd.notna(t25w) else "N/A"
        t25p_s = f"${t25p:+.2f}" if pd.notna(t25p) else "N/A"
        stops_s = f"{stops:.0%}" if pd.notna(stops) else "N/A"
        dev_s = f"{dev_auc:.3f}" if pd.notna(dev_auc) else "N/A"
        print(f"  {lbl:<20} {n_tk:>8} {dev_s:>8} {n_wf:>10} "
              f"{t25w_s:>10} {t25p_s:>10} {stops_s:>6}")
    print(f"  {'-'*75}")
elif len(pipeline_results) == 1:
    lbl, res = list(pipeline_results.items())[0]
    print(f"\n  Single universe: {lbl}")
    wf_top = res.get('wf_top', pd.DataFrame())
    if len(wf_top) > 0:
        print(f"  Top 25%: win={(wf_top['pnl_per_share']>0).mean():.0%} "
              f"avg=${wf_top['pnl_per_share'].mean():+.2f}/sh")
else:
    print("\n  No pipeline results available")


# ═══════════════════════════════════════════════════════════════
# 5. FULL RESULTS (from primary universe)
# ═══════════════════════════════════════════════════════════════

# Use Full SimFin as primary, fall back to whatever ran
primary_label = 'Full SimFin' if 'Full SimFin' in pipeline_results else list(pipeline_results.keys())[0] if pipeline_results else None

if primary_label:
    res = pipeline_results[primary_label]
    best_v_t = res['best_v_t']
    best_v_r = res['best_v_r']
    df_dev = res['df_dev']
    df_hold = res['df_hold']
    df_q = data.get('df_q', df_dev)  # df_q from original data load
    topf_v = res['topf_v']
    v_results = res['v_results']
    wf_df = res['wf_df']
    wf_top = res['wf_top']
    eq_results = res['eq_results']
    cache_dir = data['cache_dir']
    fcols_q = res['fcols_q']
    tgt_rates = res['tgt_rates']
    all_wf_trades = res['all_wf_trades']
    K = res['K']

    ba = best_v_r['mauc']
    cl = best_v_r['clo']

    # Holdout AUC (already computed above, recompute for final display)
    ho_auc = np.nan
    try:
        if (TRADING_TARGET in df_hold.columns
                and df_hold[TRADING_TARGET].sum() >= 10
                and 'vuln_score' in df_hold.columns):
            yh = df_hold[TRADING_TARGET].fillna(0).astype(int)
            hp = df_hold['vuln_score'].values
            valid = ~np.isnan(hp)
            if valid.sum() >= 20 and yh[valid].sum() >= 5:
                ho_auc = roc_auc_score(yh[valid], hp[valid])
    except:
        pass

    # Also get voladj AUC
    va_tgts_r = [t for t in v_results if t.startswith('voladj_')]
    bva = max(va_tgts_r, key=lambda k: v_results[k]['mauc']) if va_tgts_r else None
    va_auc = v_results[bva]['mauc'] if bva else 0

    print("\n" + "\u2588" * 70)
    print("  DROP SCORE v18 \u2014 FULL RESULTS")
    print("\u2588" * 70)
    print(f"""
DATA: {df_q['ticker'].nunique() if 'ticker' in df_q.columns else df_dev['ticker'].nunique()} stocks | {len(df_dev):,} dev + {len(df_hold):,} hold
VULN: {best_v_t} Dev={ba:.3f} [{cl:.3f},{best_v_r['chi']:.3f}]
TRADING: {TRADING_TARGET} Hold={ho_auc:.3f} (holdout on trading target)
Features: {topf_v}
VOL-ADJUSTED: {bva or 'N/A'} AUC={va_auc:.3f}
""")

    if len(wf_df) > 0:
        total_borrow = wf_df['borrow_cost'].sum() if 'borrow_cost' in wf_df.columns else 0
        avg_borrow = wf_df['borrow_cost'].mean() if 'borrow_cost' in wf_df.columns else 0
        print(f"""WALK-FORWARD: {len(wf_df)} trades ({ENTRY_MODE} entry, {TRADING_HOLD}d hold, vol floor {VOL_FLOOR:,})
  Full:    win={(wf_df['pnl_per_share']>0).mean():.0%} avg=${wf_df['pnl_per_share'].mean():+.2f}/sh ({wf_df['pnl_pct'].mean()*100:+.1f}%) stops={wf_df['stopped'].mean():.0%}
  Top 25%: win={(wf_top['pnl_per_share']>0).mean():.0%} avg=${wf_top['pnl_per_share'].mean():+.2f}/sh ({wf_top['pnl_pct'].mean()*100:+.1f}%) stops={wf_top['stopped'].mean():.0%}
  Quarters: {(wf_top.groupby('quarter')['pnl_per_share'].sum()>0).sum()}/{wf_top['quarter'].nunique()} profitable
  Borrow: avg ${avg_borrow:.2f}/sh total ${total_borrow:,.0f}
""")

    # Print scenarios
    for lbl, r in eq_results.items():
        print(f"  {lbl}: ${r['start']:,}\u2192${r['end']:,.0f} ({r['ret']:+.0%}) "
              f"DD={r['max_dd_pct']:.0%} Calmar={r['calmar']:.2f}")

    signal_icon = '\u2705' if ba >= 0.70 else '\u26a0\ufe0f'
    alpha_icon = '\u2705' if len(wf_top) > 0 and wf_top['pnl_per_share'].mean() > 0 else '\u26a0\ufe0f'
    print(f"""
VERDICT:
  Signal: {signal_icon} Dev={ba:.3f} Hold={ho_auc:.3f} VolAdj={va_auc:.3f}
  Alpha: {alpha_icon}
  Conviction: Top 25% outperforms \u2192 trade only highest-scored picks
  Config: {TRADING_TARGET} / {TRADING_HOLD}d / {ENTRY_MODE}
""")

    # Top 20 trades inline
    if len(wf_top) > 0:
        print("FIRST 20 TRADES (top-25% conviction):")
        for i, (_, r) in enumerate(wf_top.sort_values('entry_date').head(20).iterrows()):
            ex = 'STOP' if r['stopped'] else 'PT' if r['profit_taken'] else 'EXP'
            borr = f" borrow=${r['borrow_cost']:.2f}" if 'borrow_cost' in r.index else ""
            print(f"  {r['ticker']:<6} {str(r['entry_date'])[:10]} "
                  f"${r['entry_price']:>6.0f}\u2192${r['exit_price']:>6.0f} {ex:<4} "
                  f"${r['pnl_per_share']:+.2f}/sh ({r['pnl_pct']*100:+.1f}%){borr}")

    # Save to Drive
    with open(os.path.join(cache_dir, 'v18_results.pkl'), 'wb') as f:
        pickle.dump({
            'best_v_t': best_v_t, 'best_v_r': best_v_r, 'topf_v': topf_v, 'K': K,
            'ho_auc': ho_auc, 'v_results': v_results, 'wf_trades': all_wf_trades,
            'wf_top_trades': wf_top.to_dict('records') if len(wf_top) > 0 else [],
            'eq_results': {
                k: {kk: vv for kk, vv in v.items() if kk != 'eq_df' and kk != 'trade_log'}
                for k, v in eq_results.items()
            },
            'fcols_q': fcols_q, 'tgt_rates': tgt_rates,
            'trading_config': {
                'target': TRADING_TARGET, 'hold': TRADING_HOLD,
                'entry': ENTRY_MODE,
                'borrow_easy': BORROW_RATE_EASY, 'borrow_hard': BORROW_RATE_HARD,
            },
            'universes': list(pipeline_results.keys()),
        }, f)

print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")
print("Drop Score v18 complete.")

# Close log
print(f"\nLog saved to {_log_path}")
sys.stdout = sys.__stdout__
_log_file.close()
print(f"Log saved to {_log_path}")
