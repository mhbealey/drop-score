"""
Stage 2: Run Universe A (SimFin-only) with locked v18 config.
Compare against v18 benchmarks. Gate for Stage 3.
"""
import os, sys, time, pickle, warnings, random

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

from pipeline import setup_logging, teardown_logging, run_pipeline, holdout_eval, get_pipeline_metrics
_log_path, _log_file = setup_logging('validate')

from config import (
    t_start, log, TRADING_TARGET, TRADING_HOLD, ENTRY_MODE,
    UNIVERSE_A_FEATURES, BENCHMARKS,
)
from utils import elapsed

# v18 pinned benchmarks from config.BenchmarkGates
V18 = {
    'tickers': BENCHMARKS.v18_tickers,
    'dev_auc': BENCHMARKS.v18_dev_auc,
    'hold_auc': BENCHMARKS.v18_hold_auc,
    'top25_trades': BENCHMARKS.v18_top25_trades,
    'top25_win': BENCHMARKS.v18_top25_win,
    'top25_pnl': BENCHMARKS.v18_top25_pnl,
}


def main():
    log.info("=" * 70)
    log.info("DROP SCORE v18.3 — STAGE 2: VALIDATE (Universe A)")
    log.info(f"  Target: {TRADING_TARGET} (LOCKED)")
    log.info(f"  Features: {UNIVERSE_A_FEATURES} (LOCKED)")
    log.info("=" * 70)

    # Load data bundle from Stage 1
    with open('data/data_bundle.pkl', 'rb') as f:
        data = pickle.load(f)
    log.info(f"  Loaded data bundle")

    # Universe A: SimFin-fundamental tickers only (no EDGAR)
    # simfin_universe = tickers from build_universe() on pure SimFin data
    # (>=8 quarters, non-financial, recent). Excludes EDGAR-only tickers.
    # v18 had ~960 after price/feature pipeline filtering.
    simfin_universe = set(data.get('simfin_universe', []))
    edgar_tickers = data.get('edgar_tickers', set())
    all_tickers = set(data['df_dev']['ticker'].unique())

    # Exclude any EDGAR-only tickers that leaked into the universe
    simfin_only = (simfin_universe - edgar_tickers) & all_tickers
    log.info(f"  SimFin fundamental universe: {len(simfin_universe)}")
    log.info(f"  EDGAR tickers excluded: {len(edgar_tickers & simfin_universe)}")
    log.info(f"  Universe A (in dev set): {len(simfin_only)} tickers (v18 had 960)")
    assert 500 < len(simfin_only) < 5000, \
        f"Universe A unexpected size: {len(simfin_only)}"

    # Run pipeline with locked features
    result = run_pipeline(
        data, "Full SimFin (Universe A)",
        ticker_subset=simfin_only,
        locked_features=UNIVERSE_A_FEATURES,
    )

    # Holdout evaluation
    ho = holdout_eval(result, "Full SimFin")

    # Extract metrics
    m = get_pipeline_metrics(result)

    # ── Comparison table ──
    def _f(v, fmt='.3f'):
        return f"{v:{fmt}}" if pd.notna(v) else "N/A"

    log.info(f"\n  {'='*56}")
    log.info(f"  UNIVERSE A: V18 vs V18.3")
    log.info(f"  {'':28s} {'V18':>12s} {'V18.3':>12s}")
    log.info(f"  {'-'*56}")
    log.info(f"  {'Tickers:':<28s} {V18['tickers']:>12} {m['n_tickers']:>12}")
    log.info(f"  {'Dev AUC:':<28s} {V18['dev_auc']:>12.3f} {_f(m['dev_auc']):>12s}")
    log.info(f"  {'Hold AUC:':<28s} {V18['hold_auc']:>12.3f} {_f(m['hold_auc']):>12s}")
    log.info(f"  {'Top-25% trades:':<28s} {V18['top25_trades']:>12} {m['top25_trades']:>12}")
    v18_win = f"{V18['top25_win']:.0%}"
    v183_win = _f(m['top25_win'], '.0%')
    v18_pnl = f"${V18['top25_pnl']:+.2f}"
    v183_pnl = '$' + _f(m['top25_pnl'], '+.2f')
    log.info(f"  {'Top-25% win rate:':<28s} {v18_win:>12s} {v183_win:>12s}")
    log.info(f"  {'Top-25% P&L:':<28s} {v18_pnl:>12s} {v183_pnl:>12s}")
    log.info(f"  {'='*56}")

    # ── Benchmark gates ──
    log.info(f"\n  BENCHMARK GATES:")
    gates_passed = True

    checks = [
        ('Dev AUC', m['dev_auc'], BENCHMARKS.min_dev_auc),
        ('Hold AUC', m['hold_auc'], BENCHMARKS.min_hold_auc),
        ('Trade count', m['top25_trades'], BENCHMARKS.min_top25_trades),
    ]

    for name, actual, minimum in checks:
        if pd.isna(actual) or actual < minimum:
            log.warning(f"    FAIL: {name} = {_f(actual)} (gate: >= {minimum:.3f})")
            gates_passed = False
        else:
            log.info(f"    PASS: {name} = {_f(actual)} (gate: >= {minimum:.3f})")

    # ── Save results for Stage 3 comparison ──
    with open('data/validate_results.pkl', 'wb') as f:
        pickle.dump(m, f)
    log.info(f"\n  Validation results saved to data/validate_results.pkl")

    if not gates_passed:
        log.warning(f"\n  VALIDATION FAILED — Model job will not run")
        log.warning(f"  Check for: feature changes, data pipeline changes, config changes")
        log.warning(f"  | {elapsed()}")
        sys.exit(1)

    log.info(f"\n  Validation PASSED — Model job cleared to run | {elapsed()}")


if __name__ == '__main__':
    main()
    teardown_logging(_log_file, _log_path)
