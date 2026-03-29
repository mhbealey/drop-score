"""
Stage 1: Fetch and cache all data. Build features. Save bundle for downstream jobs.
No model training — just data pipeline + feature engineering.
"""
import os, sys, time, pickle, warnings, random

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

from pipeline import setup_logging, teardown_logging
_log_path, _log_file = setup_logging('data')

from config import (
    t_start, log, TRADING_TARGET, TRADING_HOLD, ENTRY_MODE,
    UNIVERSE_A_FEATURES, UNIVERSE_B_FEATURES,
)
from utils import elapsed
from data import load_all_data
from features import prepare_features
from edgar import run_feature_qa, edgar_field_diagnostic


def main():
    log.info("=" * 70)
    log.info("DROP SCORE v18.3 — STAGE 1: DATA PIPELINE")
    log.info(f"  Target: {TRADING_TARGET} | Hold: {TRADING_HOLD}d | Entry: {ENTRY_MODE}")
    log.info(f"  Universe A features: {UNIVERSE_A_FEATURES}")
    log.info(f"  Universe B features: {UNIVERSE_B_FEATURES}")
    log.info("=" * 70)

    # Load all data sources (SimFin + EDGAR + prices)
    data = load_all_data()

    # Build features (or load from intermediates cache)
    data = prepare_features(data)

    # Feature QA on EDGAR tickers
    edgar_tickers = data.get('edgar_tickers', set())
    if edgar_tickers:
        run_feature_qa(
            data['df_dev'], data['df_hold'],
            edgar_tickers, data['fcols_q'],
        )

    # Coverage report
    simfin_universe = set(data.get('simfin_universe', []))
    all_dev_tickers = set(data['df_dev']['ticker'].unique())
    sp_tickers = data.get('sp_tickers', set())
    sp_in_dev = sp_tickers & all_dev_tickers
    price_tickers = set(data['price_dict'].keys()) - {'SPY', '^VIX'}

    log.info(f"\n  {'='*50}")
    log.info(f"  DATA COVERAGE")
    log.info(f"    SimFin universe:      {len(simfin_universe)} tickers")
    log.info(f"    EDGAR tickers:        {len(edgar_tickers)} tickers "
          f"({len(edgar_tickers - simfin_universe)} new)")
    log.info(f"    Combined dev set:     {all_dev_tickers.__len__()} tickers, "
          f"{len(data['df_dev']):,} rows")
    log.info(f"    Holdout set:          {len(data['df_hold']):,} rows")
    log.info(f"    S&P in dev:           {len(sp_in_dev)}/{len(sp_tickers)}")
    log.info(f"    Price dict:           {len(price_tickers)} tickers")
    log.info(f"    Features:             {len(data['fcols_q'])} columns")
    log.info(f"    Targets:              {len(data.get('tcols', []))} targets")
    log.info(f"  {'='*50}\n")

    # Save portable bundle (strip cache/file refs not needed downstream)
    strip_keys = {
        'cache', 'cache_path', 'intermediates_path',
        'intm_loaded', 'df_inc', 'df_bal', 'df_cf', 'unavail', 'df_daily',
    }
    portable = {k: v for k, v in data.items() if k not in strip_keys}

    # equity.py needs cache_dir — ensure it's set
    portable['cache_dir'] = data.get('cache_dir', 'data/')

    bundle_path = 'data/data_bundle.pkl'
    with open(bundle_path, 'wb') as f:
        pickle.dump(portable, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(bundle_path) / 1e6
    log.info(f"  Data bundle saved: {bundle_path} ({size_mb:.0f} MB)")

    # Assertions — fail fast if coverage is broken
    assert len(simfin_universe) > 900, \
        f"SimFin coverage too low: {len(simfin_universe)}"
    assert len(sp_in_dev) > 200, \
        f"S&P tradeable too low: {len(sp_in_dev)}"
    assert len(price_tickers) > 1000, \
        f"Price coverage too low: {len(price_tickers)}"
    assert len(data['df_dev']) > 10000, \
        f"Dev set too small: {len(data['df_dev'])}"

    log.info(f"  Data pipeline complete | {elapsed()}")


if __name__ == '__main__':
    main()
    teardown_logging(_log_file, _log_path)
