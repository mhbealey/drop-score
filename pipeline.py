"""
Shared pipeline functions used by run_validate.py and run_model.py.
Extracted from main.py -- same logic, no model/feature/config changes.
"""
import os, sys, time, warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import (
    N_BOOT, TRADING_TARGET, TRADING_HOLD, ENTRY_MODE, log,
)
from model import run_vulnerability_model, run_model, run_bayesian_optimization, run_bootstrap_ci
from walkforward import run_walkforward
from equity import run_equity_scenarios


class Tee:
    """Tee stdout to both console and a log file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def setup_logging(prefix: str = 'run') -> Tuple[str, Any]:
    """Set up stdout tee to results/ log file. Returns (log_path, log_file)."""
    os.makedirs('results', exist_ok=True)
    log_path = os.path.join('results', f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    log_file = open(log_path, 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)
    return log_path, log_file


def teardown_logging(log_file: Any, log_path: str) -> None:
    """Restore stdout and close log file."""
    print(f"\nLog saved to {log_path}")
    sys.stdout = sys.__stdout__
    log_file.close()


def run_pipeline(data_bundle: dict, label: str,
                 ticker_subset: Optional[set] = None,
                 locked_features: Optional[List[str]] = None) -> dict:
    """Run model + walk-forward + equity for a universe subset.

    If ticker_subset is given, filters df_dev/df_hold to those tickers
    and recomputes fill medians. Features are NOT recomputed.
    If locked_features is given, passes to vulnerability model to skip Pareto.
    """
    bundle = dict(data_bundle)  # shallow copy

    if locked_features:
        bundle['locked_features'] = locked_features

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

    log.info("=" * 70)
    log.info(f"PIPELINE: {label}")
    n_dev = len(bundle['df_dev'])
    n_hold = len(bundle['df_hold'])
    n_tickers = bundle['df_dev']['ticker'].nunique()
    log.info(f"  {n_tickers} tickers | {n_dev:,} dev + {n_hold:,} holdout rows")
    log.info("=" * 70)

    bundle = run_vulnerability_model(bundle)
    bundle = run_walkforward(bundle)
    bundle = run_equity_scenarios(bundle)

    return bundle


def holdout_eval(bundle: dict, label: str = "") -> Dict[str, float]:
    """Evaluate holdout on TRADING_TARGET. Returns dict of AUCs."""
    if label:
        log.info(f"\n  HOLDOUT ({label}):")
    best_v_t = bundle.get('best_v_t', TRADING_TARGET)
    v_results = bundle.get('v_results', {})
    df_hold = bundle['df_hold']

    results = {}

    targets_to_eval = [best_v_t]
    if TRADING_TARGET != best_v_t and TRADING_TARGET in v_results:
        targets_to_eval.append(TRADING_TARGET)

    va_tgts = [t for t in v_results if t.startswith('voladj_')]
    best_va = max(va_tgts, key=lambda k: v_results[k]['mauc']) if va_tgts else None
    if best_va and best_va not in targets_to_eval:
        targets_to_eval.append(best_va)

    for tgt in targets_to_eval:
        try:
            if tgt not in df_hold.columns:
                continue
            yh = df_hold[tgt].fillna(0).astype(int)
            n_pos = int(yh.sum())
            if n_pos < 10 or 'vuln_score' not in df_hold.columns:
                continue
            hp = df_hold['vuln_score'].values
            valid = ~np.isnan(hp)
            if valid.sum() >= 20 and yh[valid].sum() >= 5:
                ho_auc = roc_auc_score(yh[valid], hp[valid])
                dev_auc = v_results[tgt]['mauc']
                marker = " <-- TRADING" if tgt == TRADING_TARGET else ""
                if label:
                    log.info(f"    {tgt}: Dev={dev_auc:.3f} Hold={ho_auc:.3f}{marker}")
                results[tgt] = ho_auc
        except Exception as e:
            if label:
                log.info(f"    {tgt}: error — {e}")

    return results


def get_pipeline_metrics(result: dict) -> dict:
    """Extract standard metrics from a pipeline result for comparison."""
    dev_auc = result.get('v_results', {}).get(TRADING_TARGET, {}).get('mauc', np.nan)

    # Holdout AUC
    hold_auc = np.nan
    df_hold = result.get('df_hold', pd.DataFrame())
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

    wf = result.get('wf_df', pd.DataFrame())
    wf_top = result.get('wf_top', pd.DataFrame())

    m = {
        'n_tickers': result['df_dev']['ticker'].nunique(),
        'dev_auc': dev_auc,
        'hold_auc': hold_auc,
    }

    # Top 25%
    if len(wf_top) > 0:
        m['top25_trades'] = len(wf_top)
        m['top25_win'] = (wf_top['pnl_per_share'] > 0).mean()
        m['top25_pnl'] = wf_top['pnl_per_share'].mean()
    else:
        m['top25_trades'] = 0
        m['top25_win'] = np.nan
        m['top25_pnl'] = np.nan

    # Top 10%
    if len(wf) > 0 and 'wf_score' in wf.columns:
        t10_cutoff = wf['wf_score'].quantile(0.90)
        wf_t10 = wf[wf['wf_score'] >= t10_cutoff]
        if len(wf_t10) > 0:
            m['top10_trades'] = len(wf_t10)
            m['top10_win'] = (wf_t10['pnl_per_share'] > 0).mean()
            m['top10_pnl'] = wf_t10['pnl_per_share'].mean()
        else:
            m['top10_trades'] = 0
            m['top10_win'] = np.nan
            m['top10_pnl'] = np.nan
    else:
        m['top10_trades'] = 0
        m['top10_win'] = np.nan
        m['top10_pnl'] = np.nan

    return m
