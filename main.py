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
            "matplotlib", "tqdm", "optuna", "scipy"]:
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
    UNIVERSE_A_FEATURES, UNIVERSE_B_FEATURES,
)
from utils import elapsed, clean_X, to_scalar, ensure_series
from data import load_all_data, get_sp_index_tickers
from features import prepare_features
from model import run_vulnerability_model, run_model, run_bayesian_optimization, run_bootstrap_ci
from walkforward import run_walkforward
from equity import run_equity_scenarios
from edgar import run_feature_qa

# ═══════════════════════════════════════════════════════════════
# BANNER
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("DROP SCORE v18.3 — LOCKED CONFIG + SEPARATED UNIVERSES + BORROW COSTS")
print(f"  Target: {TRADING_TARGET} | Hold: {TRADING_HOLD}d | Entry: {ENTRY_MODE}")
print(f"  Borrow: {BORROW_RATE_EASY:.0%}/{BORROW_RATE_HARD:.0%} | Universe: {UNIVERSE_MODE}")
print(f"  Universe A features: {UNIVERSE_A_FEATURES}")
print(f"  Universe B features: {UNIVERSE_B_FEATURES}")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════
data = load_all_data()

# ═══════════════════════════════════════════════════════════════
# 2. FEATURES (computed once on full dataset)
# ═══════════════════════════════════════════════════════════════
data = prepare_features(data)

# ── Feature QA (Phase 4) ──
edgar_tickers = data.get('edgar_tickers', set())
if edgar_tickers:
    run_feature_qa(
        data['df_dev'], data['df_hold'],
        edgar_tickers, data['fcols_q'],
    )


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
                        except Exception:
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
    # Universe A: SimFin-ONLY tickers (no EDGAR contamination)
    print("\n" + "#" * 70)
    print("# UNIVERSE A: Full SimFin (SimFin-only, locked features)")
    print("#" * 70)
    simfin_universe = set(data.get('simfin_universe', []))
    all_tickers_a = set(data['df_dev']['ticker'].unique())
    simfin_only = simfin_universe & all_tickers_a
    print(f"  SimFin universe: {len(simfin_universe)} tickers")
    print(f"  Overlap with feature data: {len(simfin_only)}")
    data_a = dict(data)
    data_a['locked_features'] = UNIVERSE_A_FEATURES
    result_a = run_pipeline(data_a, "Full SimFin", ticker_subset=simfin_only)
    ho_a = holdout_eval(result_a, "Full SimFin")
    pipeline_results['Full SimFin'] = result_a
    t_a = time.time() - t_start

if UNIVERSE_MODE in ("sp_index", "both"):
    # Time budget: skip Universe B if >25 min already
    elapsed_min = (time.time() - t_start) / 60
    if UNIVERSE_MODE == "both" and elapsed_min > 40:
        print(f"\n  TIME BUDGET: {elapsed_min:.0f} min elapsed (>40), skipping S&P index pipeline")
    else:
        print("\n" + "#" * 70)
        print("# UNIVERSE B: S&P 400+600 Index (SimFin + EDGAR, locked features)")
        print("#" * 70)
        sp_tickers = get_sp_index_tickers()
        if sp_tickers:
            # Filter to tickers we have data for
            all_tickers = set(data['df_dev']['ticker'].unique())
            sp_overlap = sp_tickers & all_tickers
            # Also check overlap with price cache
            price_overlap = sp_tickers & set(data['price_dict'].keys())
            print(f"  S&P index: {len(sp_tickers)} tickers")
            print(f"  Overlap with feature data: {len(sp_overlap)}/{len(sp_tickers)}")
            print(f"  Overlap with price cache: {len(price_overlap)}/{len(sp_tickers)}")
            if len(sp_overlap) < 200:
                print(f"  WARNING: low overlap ({len(sp_overlap)} tickers), running anyway")
            data_b = dict(data)
            data_b['locked_features'] = UNIVERSE_B_FEATURES
            result_b = run_pipeline(data_b, "S&P 400+600", ticker_subset=sp_overlap)
            ho_b = holdout_eval(result_b, "S&P 400+600")
            pipeline_results['S&P 400+600'] = result_b
        else:
            print("  Could not get S&P index tickers, skipping")


# ═══════════════════════════════════════════════════════════════
# BAYESIAN OPTIMIZATION + BOOTSTRAP CIs (Phase 6)
# ═══════════════════════════════════════════════════════════════
elapsed_min = (time.time() - t_start) / 60
print(f"\n  Time check: {elapsed_min:.0f} min elapsed")

_primary = 'Full SimFin' if 'Full SimFin' in pipeline_results else (
    list(pipeline_results.keys())[0] if pipeline_results else None
)

if _primary and elapsed_min < 80:
    _res = pipeline_results[_primary]
    # Bayesian optimization
    _baseline_auc = _res.get('v_results', {}).get(
        _res.get('best_v_t', ''), {}).get('mauc', 0.5)
    _res = run_bayesian_optimization(_res, n_trials=30, timeout=1200)
    pipeline_results[_primary] = _res

    # If Optuna found better params (>0.01 AUC gain), retrain with them
    _optuna_bp = _res.get('optuna_best_params')
    _optuna_auc = _res.get('optuna_study')
    if _optuna_bp and _optuna_auc:
        _opt_val = _optuna_auc.best_value
        if _opt_val - _baseline_auc > 0.01:
            print(f"  Optuna improved AUC by {_opt_val - _baseline_auc:+.3f} — "
                  f"params stored for next run")
        else:
            print(f"  Optuna improvement marginal ({_opt_val - _baseline_auc:+.3f}) — "
                  f"keeping default params")
            _res.pop('optuna_best_params', None)
            pipeline_results[_primary] = _res

    # Bootstrap CIs on walk-forward top 25%
    _wf_top = _res.get('wf_top', pd.DataFrame())
    _boot = run_bootstrap_ci(_wf_top, n_boot=1000)
    pipeline_results[_primary]['bootstrap_ci'] = _boot
elif elapsed_min >= 80:
    print("  TIME BUDGET: >80 min, skipping Bayesian optimization")


# ═══════════════════════════════════════════════════════════════
# VOLADJ COLLAPSE DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════

# Run on the primary (full) universe result
_diag_label = 'Full SimFin' if 'Full SimFin' in pipeline_results else (
    list(pipeline_results.keys())[0] if pipeline_results else None
)
if _diag_label:
    try:
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
            print(f"\n  Test 1 -- Truncation: {_full_window} full 63d window, "
                  f"{_truncated} truncated")
            if _truncated > len(_df_hold) * 0.20:
                print(f"    \u26a0\ufe0f {_truncated/len(_df_hold):.0%} truncated"
                      f" -- this likely explains the low holdout AUC")

            # ── Test 2: Redefine voladj using 252-day trailing vol ──
            print(f"\n  Test 2 -- 252d vol threshold:")
            _ho252 = None
            for _lbl, _dft in [('Dev', _df_dev), ('Hold', _df_hold)]:
                _new_labels = []
                for _idx, _row in _dft.iterrows():
                    _tk = _row['ticker']
                    if _tk not in _price_dict:
                        _new_labels.append(np.nan)
                        continue
                    _pxd = _price_dict[_tk]
                    _px = ensure_series(
                        _pxd['Close'] if 'Close' in _pxd.columns
                        else _pxd.iloc[:, 0]
                    )
                    _vi = _px.index[_px.index >= _row['report_date']]
                    if len(_vi) == 0:
                        _new_labels.append(np.nan)
                        continue
                    _si = _px.index.get_loc(_vi[0])
                    if _si >= 252:
                        _dr = _px.pct_change()
                        _vol_252 = float(
                            _dr.iloc[_si - 251:_si + 1].std() * np.sqrt(252)
                        )
                    else:
                        _vol_252 = 0.3
                    _vol_252 = max(_vol_252, 0.05)
                    _period_vol = (_vol_252 / np.sqrt(252)) * np.sqrt(63)
                    _threshold = -2.0 * _period_vol
                    if _si + 63 < len(_px):
                        _ret = (float(_px.iloc[_si + 63])
                                - float(_px.iloc[_si])) / float(_px.iloc[_si])
                        _new_labels.append(1 if _ret <= _threshold else 0)
                    else:
                        _new_labels.append(np.nan)
                _dft['voladj_2sig_63d_v252'] = _new_labels

            _r252 = run_model(
                _df_dev, _fcols_q, 'voladj_2sig_63d_v252',
                _fill_meds, N_FOLDS, 100, _K,
            )
            if _r252:
                print(f"    Dev AUC: {_r252['mauc']:.3f}")
                _lf252 = _r252['folds'][-1]
                _Xh252 = clean_X(_df_hold, _fcols_q, _fill_meds)
                try:
                    _hp252 = _lf252['model'].predict_proba(
                        _Xh252[_lf252['feats']]
                    )[:, 1]
                    _yh252 = _df_hold['voladj_2sig_63d_v252'].fillna(0).astype(int)
                    _valid252 = ~_df_hold['voladj_2sig_63d_v252'].isna()
                    if _valid252.sum() > 50 and _yh252[_valid252].sum() >= 5:
                        _ho252 = roc_auc_score(
                            _yh252[_valid252], _hp252[_valid252]
                        )
                        print(f"    Holdout AUC: {_ho252:.3f} "
                              f"(pos={int(_yh252[_valid252].sum())} "
                              f"neg={int((~_yh252[_valid252].astype(bool)).sum())})")
                        if _ho252 > 0.55:
                            print(f"    \u2705 252d vol FIXES holdout")
                        else:
                            print(f"    \u274c Still collapsed")
                    else:
                        print(f"    Holdout: too few valid samples")
                except Exception as _e:
                    print(f"    Holdout scoring failed: {str(_e)[:100]}")
            else:
                print(f"    run_model returned None for v252 target")

            # ── Test 3: Pure fundamental model on holdout ──
            _price_kw = [
                'vol_', 'ret_', 'dd_from', 'rsi', 'gap_count', 'beta',
                'vix', 'death_cross', 'down_days', 'gap_down', 'spy_corr',
                'excess_ret', 'sector_excess', 'consec_down', '_x_vol',
            ]
            _fund_only = [
                f for f in _fcols_q if not any(kw in f for kw in _price_kw)
            ]
            _ho_f = None
            _r_fund = None
            if len(_fund_only) >= 5:
                print(f"\n  Test 3 -- Fund-only model "
                      f"({len(_fund_only)} features):")
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
                        _hp_f = _lf_f['model'].predict_proba(
                            _Xh_f[_lf_f['feats']]
                        )[:, 1]
                        _yh_f = _df_hold[_va_tgt].fillna(0).astype(int)
                        _ho_f = roc_auc_score(_yh_f, _hp_f)
                        print(f"    Holdout AUC: {_ho_f:.3f}")
                        if _ho_f > 0.55:
                            print(f"    \u2705 Fundamental signal survives")
                        else:
                            print(f"    \u274c Fundamentals also collapse")
                    except Exception as _e:
                        print(f"    Holdout: failed ({str(_e)[:100]})")

            # ── Summary ──
            print(f"\n  VOLADJ COLLAPSE DIAGNOSTIC SUMMARY:")
            print(f"    Test 1: {_full_window} full window, "
                  f"{_truncated} truncated")
            _s2 = f"    Test 2: 252d vol -- Dev={_r252['mauc']:.3f}" if _r252 else "    Test 2: failed"
            if _ho252 is not None:
                _s2 += f" Hold={_ho252:.3f}"
            print(_s2)
            _s3 = f"    Test 3: Fund-only -- Dev={_r_fund['mauc']:.3f}" if _r_fund else "    Test 3: failed"
            if _ho_f is not None:
                _s3 += f" Hold={_ho_f:.3f}"
            print(_s3)
            print()
        else:
            print(f"\n  VOLADJ DIAGNOSTIC: {_va_tgt} not in trained targets, skipped")
    except Exception as _diag_err:
        print(f"\n  VOLADJ DIAGNOSTIC ERROR: {str(_diag_err)[:200]}")
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════
# 4. V18 vs V18.3 COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("V18 vs V18.3 COMPARISON")
print("=" * 70)

def _get_metrics(res, tier='top25'):
    """Extract metrics from a pipeline result."""
    n_tk = res['df_dev']['ticker'].nunique()
    dev_auc = res['v_results'].get(TRADING_TARGET, {}).get('mauc', np.nan)
    wf = res.get('wf_df', pd.DataFrame())
    wf_top = res.get('wf_top', pd.DataFrame())

    # Holdout AUC
    hold_auc = np.nan
    df_hold = res.get('df_hold', pd.DataFrame())
    if (TRADING_TARGET in df_hold.columns
            and df_hold[TRADING_TARGET].sum() >= 10
            and 'vuln_score' in df_hold.columns):
        try:
            yh = df_hold[TRADING_TARGET].fillna(0).astype(int)
            hp = df_hold['vuln_score'].values
            valid = ~np.isnan(hp)
            if valid.sum() >= 20 and yh[valid].sum() >= 5:
                hold_auc = roc_auc_score(yh[valid], hp[valid])
        except Exception:
            pass

    # Top 25% metrics
    if len(wf_top) > 0:
        t25_trades = len(wf_top)
        t25_win = (wf_top['pnl_per_share'] > 0).mean()
        t25_pnl = wf_top['pnl_per_share'].mean()
    else:
        t25_trades = t25_win = t25_pnl = np.nan

    # Top 10% metrics
    if len(wf) > 0 and 'wf_score' in wf.columns:
        t10_cutoff = wf['wf_score'].quantile(0.90)
        wf_t10 = wf[wf['wf_score'] >= t10_cutoff]
        if len(wf_t10) > 0:
            t10_trades = len(wf_t10)
            t10_win = (wf_t10['pnl_per_share'] > 0).mean()
            t10_pnl = wf_t10['pnl_per_share'].mean()
        else:
            t10_trades = t10_win = t10_pnl = np.nan
    else:
        t10_trades = t10_win = t10_pnl = np.nan

    return {
        'tickers': n_tk, 'dev_auc': dev_auc, 'hold_auc': hold_auc,
        't25_trades': t25_trades, 't25_win': t25_win, 't25_pnl': t25_pnl,
        't10_trades': t10_trades, 't10_win': t10_win, 't10_pnl': t10_pnl,
    }

def _fmt(val, fmt_str='.3f'):
    if pd.isna(val): return 'N/A'
    return f"{val:{fmt_str}}"

# V18 benchmarks
v18_a = {'tickers': 960, 'dev_auc': 0.791, 'hold_auc': 0.777, 't25_trades': 75, 't25_win': 0.71, 't25_pnl': 1.98}
v18_b = {'tickers': 231, 'dev_auc': 0.725, 'hold_auc': 0.718, 't10_trades': 35, 't10_win': 0.86, 't10_pnl': 3.06, 't25_trades': 75, 't25_win': 0.69, 't25_pnl': 1.14}

res_a = _get_metrics(pipeline_results['Full SimFin']) if 'Full SimFin' in pipeline_results else {}
res_b = _get_metrics(pipeline_results['S&P 400+600']) if 'S&P 400+600' in pipeline_results else {}

print(f"\n  {'':30s} {'V18':>12s} {'V18.3':>12s}")
print(f"  {'-'*56}")
if res_a:
    print(f"  {'Universe A tickers:':<30s} {v18_a['tickers']:>12} {res_a['tickers']:>12}")
    print(f"  {'Universe A Dev AUC:':<30s} {v18_a['dev_auc']:>12.3f} {_fmt(res_a['dev_auc']):>12s}")
    print(f"  {'Universe A Hold AUC:':<30s} {v18_a['hold_auc']:>12.3f} {_fmt(res_a['hold_auc']):>12s}")
    if pd.notna(res_a.get('t25_win')):
        print(f"  {'Universe A Top-25%:':<30s} {'71%/$1.98':>12s} "
              f"{res_a['t25_win']:.0%}/${_fmt(res_a['t25_pnl'], '.2f'):>12s}")
if res_b:
    print(f"  {'Universe B tickers:':<30s} {v18_b['tickers']:>12} {res_b['tickers']:>12}")
    print(f"  {'Universe B Dev AUC:':<30s} {v18_b['dev_auc']:>12.3f} {_fmt(res_b['dev_auc']):>12s}")
    print(f"  {'Universe B Hold AUC:':<30s} {v18_b['hold_auc']:>12.3f} {_fmt(res_b['hold_auc']):>12s}")
    if pd.notna(res_b.get('t10_win')):
        print(f"  {'Universe B Top-10%:':<30s} {'86%/$3.06':>12s} "
              f"{res_b['t10_win']:.0%}/${_fmt(res_b['t10_pnl'], '.2f'):>12s}")
    if pd.notna(res_b.get('t25_win')):
        print(f"  {'Universe B Top-25%:':<30s} {'69%/$1.14':>12s} "
              f"{res_b['t25_win']:.0%}/${_fmt(res_b['t25_pnl'], '.2f'):>12s}")
print(f"  {'-'*56}")

# V18 reproduction QA
if res_a:
    _a_dev = res_a.get('dev_auc', 0)
    _a_hold = res_a.get('hold_auc', 0)
    _a_t25 = res_a.get('t25_trades', 0)
    if pd.notna(_a_dev) and abs(_a_dev - 0.791) >= 0.05:
        print(f"  WARNING: Universe A dev AUC shifted: {_a_dev:.3f} (expected ~0.791)")
    if pd.notna(_a_hold) and abs(_a_hold - 0.777) >= 0.10:
        print(f"  WARNING: Universe A holdout AUC shifted: {_a_hold:.3f} (expected ~0.777)")
    if pd.notna(_a_t25) and _a_t25 < 50:
        print(f"  WARNING: Universe A trade count low: {_a_t25} (expected ~75)")

# Also print standard head-to-head
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
    except Exception:
        pass

    # Also get voladj AUC
    va_tgts_r = [t for t in v_results if t.startswith('voladj_')]
    bva = max(va_tgts_r, key=lambda k: v_results[k]['mauc']) if va_tgts_r else None
    va_auc = v_results[bva]['mauc'] if bva else 0

    print("\n" + "\u2588" * 70)
    print("  DROP SCORE v18.3 \u2014 FULL RESULTS")
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

# ═══════════════════════════════════════════════════════════════
# QA CHECKLIST (Phase 7)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("QA CHECKLIST")
print("=" * 70)

_sp_tk = data.get('sp_tickers', set())
_edgar_tk = data.get('edgar_tickers', set())

_checks = {}
# EDGAR tickers in feature pipeline
if primary_label:
    _b_tickers = pipeline_results.get('S&P 400+600', {}).get('df_dev', pd.DataFrame())
    if hasattr(_b_tickers, 'ticker'):
        _b_count = _b_tickers['ticker'].nunique()
    else:
        _b_count = 0
    _checks["EDGAR tickers in feature pipeline"] = _b_count > 300
    _checks["S&P coverage > 400"] = len(_sp_tk & set(data.get('universe', []))) > 400

    # Inf check
    _total_inf = 0
    for _fc in data.get('fcols_q', []):
        if _fc in data['df_dev'].columns:
            _total_inf += np.isinf(data['df_dev'][_fc]).sum()
    _checks["No inf in features"] = _total_inf == 0

    # Duplicate check
    if 'ticker' in data['df_dev'].columns and 'report_date' in data['df_dev'].columns:
        _n_dupes = data['df_dev'].duplicated(subset=['ticker', 'report_date']).sum()
        _checks["No duplicate rows"] = _n_dupes == 0

    # WF includes EDGAR-sourced trades
    _wf_df = pipeline_results.get(_primary, {}).get('wf_df', pd.DataFrame())
    if len(_wf_df) > 0 and 'ticker' in _wf_df.columns:
        _n_edgar_trades = len(_wf_df[_wf_df['ticker'].isin(_edgar_tk)])
        _checks["WF includes EDGAR-sourced trades"] = _n_edgar_trades > 0
    else:
        _checks["WF includes EDGAR-sourced trades"] = False

    # Margin null rate check (Phase 7 key metrics)
    for _feat_name in ['gross_margin', 'operating_margin']:
        if _feat_name in data['df_dev'].columns:
            _null_rate = data['df_dev'][_feat_name].isna().mean()
            _checks[f"{_feat_name} null < 30%"] = _null_rate < 0.30

    # Cash flow null rate
    for _feat_name in ['cfo_to_revenue', 'fcf_margin']:
        if _feat_name in data['df_dev'].columns:
            _null_rate = data['df_dev'][_feat_name].isna().mean()
            _checks[f"{_feat_name} null < 40%"] = _null_rate < 0.40

    # Bayesian opt
    _checks["Bayesian opt completed"] = 'optuna_study' in pipeline_results.get(_primary, {})

    # Bootstrap CIs
    _boot_ci = pipeline_results.get(_primary, {}).get('bootstrap_ci', {})
    _checks["Bootstrap CIs computed"] = len(_boot_ci) > 0

for _check, _passed in _checks.items():
    _icon = 'PASS' if _passed else 'FAIL'
    print(f"  [{_icon}] {_check}")

print("=" * 70)

print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")
print("Drop Score v18.3 complete.")

# Close log
print(f"\nLog saved to {_log_path}")
sys.stdout = sys.__stdout__
_log_file.close()
print(f"Log saved to {_log_path}")
