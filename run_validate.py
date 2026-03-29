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
    t_start, TRADING_TARGET, TRADING_HOLD, ENTRY_MODE,
    UNIVERSE_A_FEATURES,
)
from utils import elapsed

# ═══════════════════════════════════════════════════════════════
# v18 pinned benchmarks — DO NOT CHANGE
# ═══════════════════════════════════════════════════════════════
V18 = {
    'tickers': 960,
    'dev_auc': 0.791,
    'hold_auc': 0.777,
    'top25_trades': 75,
    'top25_win': 0.71,
    'top25_pnl': 1.98,
}


def main():
    print("=" * 70)
    print("DROP SCORE v18.3 — STAGE 2: VALIDATE (Universe A)")
    print(f"  Target: {TRADING_TARGET} (LOCKED)")
    print(f"  Features: {UNIVERSE_A_FEATURES} (LOCKED)")
    print("=" * 70)

    # Load data bundle from Stage 1
    with open('data/data_bundle.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded data bundle")

    # Universe A: SimFin-only tickers
    simfin_universe = set(data.get('simfin_universe', []))
    all_tickers = set(data['df_dev']['ticker'].unique())
    simfin_only = simfin_universe & all_tickers
    print(f"  SimFin universe: {len(simfin_universe)} -> {len(simfin_only)} in dev set")

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

    print(f"\n  {'='*56}")
    print(f"  UNIVERSE A: V18 vs V18.3")
    print(f"  {'':28s} {'V18':>12s} {'V18.3':>12s}")
    print(f"  {'-'*56}")
    print(f"  {'Tickers:':<28s} {V18['tickers']:>12} {m['n_tickers']:>12}")
    print(f"  {'Dev AUC:':<28s} {V18['dev_auc']:>12.3f} {_f(m['dev_auc']):>12s}")
    print(f"  {'Hold AUC:':<28s} {V18['hold_auc']:>12.3f} {_f(m['hold_auc']):>12s}")
    print(f"  {'Top-25% trades:':<28s} {V18['top25_trades']:>12} {m['top25_trades']:>12}")
    v18_win = f"{V18['top25_win']:.0%}"
    v183_win = _f(m['top25_win'], '.0%')
    v18_pnl = f"${V18['top25_pnl']:+.2f}"
    v183_pnl = '$' + _f(m['top25_pnl'], '+.2f')
    print(f"  {'Top-25% win rate:':<28s} {v18_win:>12s} {v183_win:>12s}")
    print(f"  {'Top-25% P&L:':<28s} {v18_pnl:>12s} {v183_pnl:>12s}")
    print(f"  {'='*56}")

    # ── Benchmark gates ──
    print(f"\n  BENCHMARK GATES:")
    gates_passed = True

    checks = [
        ('Dev AUC', m['dev_auc'], V18['dev_auc'] - 0.05),
        ('Hold AUC', m['hold_auc'], V18['hold_auc'] - 0.10),
        ('Trade count', m['top25_trades'], V18['top25_trades'] * 0.5),
    ]

    for name, actual, minimum in checks:
        if pd.isna(actual) or actual < minimum:
            print(f"    FAIL: {name} = {_f(actual)} (gate: >= {minimum:.3f})")
            gates_passed = False
        else:
            print(f"    PASS: {name} = {_f(actual)} (gate: >= {minimum:.3f})")

    # ── Save results for Stage 3 comparison ──
    with open('data/validate_results.pkl', 'wb') as f:
        pickle.dump(m, f)
    print(f"\n  Validation results saved to data/validate_results.pkl")

    if not gates_passed:
        print(f"\n  VALIDATION FAILED — Model job will not run")
        print(f"  Check for: feature changes, data pipeline changes, config changes")
        print(f"  | {elapsed()}")
        sys.exit(1)

    print(f"\n  Validation PASSED — Model job cleared to run | {elapsed()}")


if __name__ == '__main__':
    main()
    teardown_logging(_log_file, _log_path)
