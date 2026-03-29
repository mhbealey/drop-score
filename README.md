# Drop Score v18.3

Quantitative short-selling model for US equities. Identifies stocks likely to
experience excess dividend-date drops using fundamental and price features,
then backtests via walk-forward simulation with confirmed entries and
volume-based borrow costs.

## Architecture

```
main.py              # Fallback runner (local dev — runs all 3 stages)
run_data.py          # Stage 1: data pipeline + feature engineering
run_validate.py      # Stage 2: Universe A validation against v18 benchmarks
run_model.py         # Stage 3: Universe B model + Bayesian opt + diagnostics
```

### Core Modules

| Module | Purpose |
|---|---|
| `config.py` | All parameters, thresholds, feature lists, XBRL mappings |
| `data.py` | SimFin + EDGAR fundamentals, multi-source price waterfall |
| `edgar.py` | SEC EDGAR XBRL parsing, CIK lookup, gap-fill for SimFin |
| `features.py` | Quarterly feature engineering, outcome computation |
| `model.py` | XGB+LGB ensemble, Pareto optimisation, Bayesian tuning |
| `walkforward.py` | Walk-forward backtest with regime filters and borrow costs |
| `equity.py` | Equity curve simulation, account scenarios, diagnostics |
| `pipeline.py` | Shared orchestration (Tee logging, run_pipeline, holdout) |
| `display.py` | Output formatting utilities |
| `utils.py` | Shared helpers (elapsed, clean_X, to_scalar, etc.) |

## Configuration

All model parameters live in `config.py` as frozen dataclasses:

- **TradingConfig** -- target, hold days, entry mode, slippage, stops
- **RegimeConfig** -- SPY/VIX regime filters
- **FeatureConfig** -- Universe A and B feature lists (locked)
- **ModelConfig** -- XGBoost/LightGBM training parameters
- **TargetConfig** -- Forward-return windows and drop thresholds
- **DataConfig** -- API keys, caching, intermediates version
- **EquityConfig** -- Account scenarios
- **BenchmarkGates** -- Minimum thresholds for validation pass

Backward-compatible flat aliases (`TRADING_TARGET`, `STOP_LOSS`, etc.) are
provided at module level for incremental migration.

## Pipeline

The CI pipeline (`.github/workflows/run.yml`) runs 3 jobs with artifact passing:

1. **Data** (`run_data.py`) -- Fetch SimFin + EDGAR fundamentals, download prices,
   build features, save `data/data_bundle.pkl`
2. **Validate** (`run_validate.py`) -- Run Universe A (SimFin-only) against v18
   benchmarks. Gates Stage 3.
3. **Model** (`run_model.py`) -- Run Universe B (S&P 400+600), Bayesian
   optimization, bootstrap CIs, equity simulation, diagnostics.

For local development, `main.py` runs all three stages sequentially.

## Requirements

```
simfin yfinance xgboost lightgbm scikit-learn
matplotlib tqdm pandas numpy gdown optuna scipy
```

## Key Design Decisions

- **Confirmed entry**: Waits for a 2% drop within 5 days of signal before
  entering, reducing false positive rate
- **Volume-based borrow costs**: 3% annual for liquid stocks (vol >= 1M),
  6% for illiquid
- **EDGAR gap-fill**: Supplements SimFin with SEC XBRL data for S&P 400/600
  tickers missing from SimFin
- **Walk-forward**: Per-quarter model retraining with regime filters (SPY
  momentum, VIX floor) to avoid trading in bull markets
- **Dual universe**: Universe A (SimFin-only) validates against v18 benchmarks;
  Universe B (S&P 400+600) is the production model
