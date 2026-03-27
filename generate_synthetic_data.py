"""
Generate synthetic financial data for the Drop Score v17 pipeline.

Creates realistic-looking fake data so the pipeline can run end-to-end
when real SimFin / yFinance cache files are unavailable.

Outputs:
  data/drop_score_cache.pkl   -- SimFin-style fundamentals + prices + sector map
  data/v13_intermediates.pkl  -- quarterly features, dev/hold splits, daily SPY
"""

import os
import random
import pickle
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_TICKERS = 200
N_QUARTERS = 20
N_YEARS_PRICES = 5
SECTORS = ['Technology', 'Healthcare', 'Energy', 'Industrials',
           'Consumer', 'Utilities', 'Other']

FWD_WINDOWS = [5, 10, 21, 42, 63]
DROP_THRESH = [0.05, 0.10, 0.15, 0.20, 0.25]
EXCESS_THRESH = [0.05, 0.10, 0.15]

SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI',
               'XLY', 'XLP', 'XLB', 'XLU', 'XLRE', 'XLC']

FUNDAMENTAL_FEATURES = [
    'roa', 'roe', 'gross_margin', 'op_margin', 'net_margin',
    'debt_to_equity', 'current_ratio', 'quick_ratio',
    'asset_turnover', 'inventory_turnover',
    'revenue_growth_yoy', 'earnings_growth_yoy',
    'margin_trend', 'accruals_ratio', 'cash_flow_ratio',
    'capex_to_rev', 'interest_coverage', 'debt_to_assets',
    'equity_ratio', 'working_capital_ratio',
    'fcf_yield', 'earnings_quality', 'revenue_accel', 'margin_accel',
    'debt_change', 'shares_change', 'cash_burn_rate',
    'altman_z', 'piotroski_f', 'beneish_m', 'sloan_ratio',
    'days_sales_outstanding', 'days_payable', 'cash_conversion',
    'capex_growth', 'sga_to_rev', 'rd_to_rev',
    'tangible_book', 'goodwill_ratio', 'intangible_ratio',
    'tax_rate_eff', 'dividend_payout', 'buyback_yield',
    'insider_ownership_proxy', 'float_ratio',
    'revenue_surprise_proxy', 'earnings_surprise_proxy',
    'guidance_proxy', 'analyst_revision_proxy',
    'sector_relative_margin', 'sector_relative_growth',
]

PRICE_FEATURES = [
    'vol_30d', 'vol_60d', 'vol_90d',
    'ret_5d', 'ret_21d', 'ret_63d',
    'dd_from_high', 'gap_count_30d', 'down_days_30d',
    'death_cross', 'excess_ret_21d', 'excess_ret_63d',
    'sector_excess_21d', 'sector_excess_63d',
    'consec_down_days', 'gap_down_today', 'gap_downs_5d',
    'spy_corr_60d',
]

INTERACTION_FEATURES = ['roa_x_vol', 'margin_x_vol', 'margin_trend_x_vol']

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Helper: generate a GBM price series
# ---------------------------------------------------------------------------

def _gbm_prices(n_days, start_price, mu=0.0002, sigma=0.02):
    """Geometric Brownian Motion daily prices."""
    dt = 1.0
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn(n_days)
    log_returns[0] = 0.0
    prices = start_price * np.exp(np.cumsum(log_returns))
    return prices


# ---------------------------------------------------------------------------
# 1. Build trading-day calendar and price data
# ---------------------------------------------------------------------------

def _make_trading_days():
    """Business-day calendar spanning ~5 years back from today."""
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=N_YEARS_PRICES)
    return pd.bdate_range(start, end)


def _generate_prices(trading_days):
    """Return {ticker: DataFrame(Close, Volume)} for all tickers + benchmarks."""
    n_days = len(trading_days)
    price_dict = {}

    # SPY
    spy_px = _gbm_prices(n_days, 400.0, mu=0.0003, sigma=0.012)
    spy_vol = np.random.lognormal(mean=np.log(8e7), sigma=0.3, size=n_days).astype(int)
    price_dict['SPY'] = pd.DataFrame(
        {'Close': spy_px, 'Volume': spy_vol}, index=trading_days
    )

    # VIX (mean-reverting around 18)
    vix = np.empty(n_days)
    vix[0] = 18.0
    for i in range(1, n_days):
        vix[i] = max(9.0, vix[i - 1] + 0.05 * (18.0 - vix[i - 1]) + 1.5 * np.random.randn())
    price_dict['^VIX'] = pd.DataFrame({'Close': vix}, index=trading_days)

    # Sector ETFs
    for etf in SECTOR_ETFS:
        px = _gbm_prices(n_days, np.random.uniform(30, 150), mu=0.0002, sigma=0.015)
        vol = np.random.lognormal(mean=np.log(2e7), sigma=0.4, size=n_days).astype(int)
        price_dict[etf] = pd.DataFrame({'Close': px, 'Volume': vol}, index=trading_days)

    # Synthetic tickers
    for i in range(N_TICKERS):
        tk = f'SYN{i:03d}'
        start_px = np.random.uniform(10, 300)
        sigma = np.random.uniform(0.015, 0.045)
        mu = np.random.uniform(-0.0002, 0.0004)
        px = _gbm_prices(n_days, start_px, mu=mu, sigma=sigma)
        vol = np.random.lognormal(mean=np.log(2e6), sigma=0.5, size=n_days).astype(int)
        price_dict[tk] = pd.DataFrame({'Close': px, 'Volume': vol}, index=trading_days)

    return price_dict


# ---------------------------------------------------------------------------
# 2. Build quarterly feature dataset
# ---------------------------------------------------------------------------

def _generate_feature_row(ticker, report_date, sector, price, avg_vol, market_cap):
    """Generate one row of ~70 features for a single quarter-report."""
    row = {
        'ticker': ticker,
        'report_date': report_date,
        'price': price,
        'avg_vol': avg_vol,
        'market_cap': market_cap,
        'sector': sector,
    }

    # ── Fundamental features ──
    row['roa'] = np.random.normal(0.03, 0.06)
    row['roe'] = np.random.normal(0.10, 0.15)
    row['gross_margin'] = np.clip(np.random.normal(0.40, 0.15), 0.0, 0.95)
    row['op_margin'] = np.clip(np.random.normal(0.12, 0.12), -0.5, 0.6)
    row['net_margin'] = np.clip(np.random.normal(0.08, 0.10), -0.5, 0.5)
    row['debt_to_equity'] = max(0.0, np.random.normal(0.8, 0.6))
    row['current_ratio'] = max(0.2, np.random.normal(1.8, 0.8))
    row['quick_ratio'] = max(0.1, np.random.normal(1.2, 0.7))
    row['asset_turnover'] = max(0.05, np.random.normal(0.7, 0.3))
    row['inventory_turnover'] = max(0.5, np.random.normal(8.0, 4.0))
    row['revenue_growth_yoy'] = np.random.normal(0.05, 0.15)
    row['earnings_growth_yoy'] = np.random.normal(0.05, 0.25)
    row['margin_trend'] = np.random.normal(0.0, 0.03)
    row['accruals_ratio'] = np.random.normal(0.0, 0.05)
    row['cash_flow_ratio'] = max(0.0, np.random.normal(1.1, 0.4))
    row['capex_to_rev'] = max(0.0, np.random.normal(0.06, 0.04))
    row['interest_coverage'] = np.random.normal(8.0, 6.0)
    row['debt_to_assets'] = np.clip(np.random.normal(0.30, 0.15), 0.0, 0.95)
    row['equity_ratio'] = np.clip(np.random.normal(0.45, 0.15), 0.0, 1.0)
    row['working_capital_ratio'] = np.random.normal(0.15, 0.10)
    row['fcf_yield'] = np.random.normal(0.04, 0.05)
    row['earnings_quality'] = np.random.normal(0.8, 0.3)
    row['revenue_accel'] = np.random.normal(0.0, 0.05)
    row['margin_accel'] = np.random.normal(0.0, 0.02)
    row['debt_change'] = np.random.normal(0.0, 0.10)
    row['shares_change'] = np.random.normal(0.0, 0.03)
    row['cash_burn_rate'] = np.random.normal(0.0, 0.05)
    row['altman_z'] = np.random.normal(3.0, 1.5)
    row['piotroski_f'] = np.clip(np.random.normal(5.0, 2.0), 0, 9).round()
    row['beneish_m'] = np.random.normal(-2.2, 0.8)
    row['sloan_ratio'] = np.random.normal(0.0, 0.05)
    row['days_sales_outstanding'] = max(5.0, np.random.normal(45.0, 20.0))
    row['days_payable'] = max(5.0, np.random.normal(40.0, 18.0))
    row['cash_conversion'] = np.random.normal(0.9, 0.3)
    row['capex_growth'] = np.random.normal(0.05, 0.15)
    row['sga_to_rev'] = max(0.0, np.random.normal(0.20, 0.10))
    row['rd_to_rev'] = max(0.0, np.random.normal(0.05, 0.05))
    row['tangible_book'] = max(0.0, np.random.normal(15.0, 10.0))
    row['goodwill_ratio'] = np.clip(np.random.normal(0.10, 0.08), 0.0, 0.6)
    row['intangible_ratio'] = np.clip(np.random.normal(0.08, 0.06), 0.0, 0.5)
    row['tax_rate_eff'] = np.clip(np.random.normal(0.21, 0.08), 0.0, 0.5)
    row['dividend_payout'] = max(0.0, np.random.normal(0.25, 0.20))
    row['buyback_yield'] = np.random.normal(0.01, 0.02)
    row['insider_ownership_proxy'] = np.clip(np.random.normal(0.05, 0.04), 0.0, 0.5)
    row['float_ratio'] = np.clip(np.random.normal(0.85, 0.10), 0.3, 1.0)
    row['revenue_surprise_proxy'] = np.random.normal(0.01, 0.04)
    row['earnings_surprise_proxy'] = np.random.normal(0.02, 0.08)
    row['guidance_proxy'] = np.random.normal(0.0, 0.03)
    row['analyst_revision_proxy'] = np.random.normal(0.0, 0.02)
    row['sector_relative_margin'] = np.random.normal(0.0, 0.05)
    row['sector_relative_growth'] = np.random.normal(0.0, 0.08)

    # ── Price features ──
    row['vol_30d'] = max(0.05, np.random.normal(0.30, 0.12))
    row['vol_60d'] = max(0.05, np.random.normal(0.30, 0.11))
    row['vol_90d'] = max(0.05, np.random.normal(0.30, 0.10))
    row['ret_5d'] = np.random.normal(0.0, 0.04)
    row['ret_21d'] = np.random.normal(0.0, 0.08)
    row['ret_63d'] = np.random.normal(0.0, 0.14)
    row['dd_from_high'] = -abs(np.random.normal(0.10, 0.10))
    row['gap_count_30d'] = max(0, int(np.random.normal(2, 2)))
    row['down_days_30d'] = max(0, min(30, int(np.random.normal(13, 3))))
    row['death_cross'] = int(np.random.random() < 0.15)
    row['excess_ret_21d'] = np.random.normal(0.0, 0.06)
    row['excess_ret_63d'] = np.random.normal(0.0, 0.10)
    row['sector_excess_21d'] = np.random.normal(0.0, 0.05)
    row['sector_excess_63d'] = np.random.normal(0.0, 0.08)
    row['consec_down_days'] = max(0, int(np.random.exponential(1.5)))
    row['gap_down_today'] = int(np.random.random() < 0.08)
    row['gap_downs_5d'] = max(0, int(np.random.normal(0.4, 0.6)))
    row['spy_corr_60d'] = np.clip(np.random.normal(0.5, 0.25), -0.5, 1.0)

    # ── Interaction features ──
    row['roa_x_vol'] = row['roa'] * row['vol_30d']
    row['margin_x_vol'] = row['op_margin'] * row['vol_30d']
    row['margin_trend_x_vol'] = row['margin_trend'] * row['vol_30d']

    return row


def _build_quarterly_features(price_dict, trading_days):
    """Build the full quarterly feature DataFrame (df_q)."""
    tickers = [f'SYN{i:03d}' for i in range(N_TICKERS)]
    sector_map = {tk: SECTORS[i % len(SECTORS)] for i, tk in enumerate(tickers)}

    # Quarter-start dates spread across the history
    end_date = trading_days[-1]
    start_date = trading_days[0] + pd.DateOffset(months=6)  # leave room for lookback
    quarter_starts = pd.date_range(start_date, end_date, freq='QS')
    # Keep only the last N_QUARTERS
    quarter_starts = quarter_starts[-N_QUARTERS:] if len(quarter_starts) >= N_QUARTERS else quarter_starts

    rows = []
    for tk in tickers:
        px_df = price_dict[tk]
        px = px_df['Close']
        vol = px_df['Volume']
        sector = sector_map[tk]

        # Each ticker reports in a random subset of quarters
        n_reports = random.randint(8, N_QUARTERS)
        n_reports = min(n_reports, len(quarter_starts))
        report_qs = sorted(random.sample(list(quarter_starts), n_reports))

        for qd in report_qs:
            # Offset the actual report date by 30-60 days past quarter start
            offset_days = random.randint(30, 60)
            report_date = qd + pd.Timedelta(days=offset_days)
            # Snap to the nearest trading day at or before
            valid = px.index[px.index <= report_date]
            if len(valid) < 63:
                continue
            report_date = valid[-1]

            price_val = float(px.loc[report_date])
            avg_vol_val = float(vol.loc[:report_date].iloc[-21:].mean())
            market_cap = price_val * np.random.uniform(5e7, 5e10)

            row = _generate_feature_row(
                tk, report_date, sector, price_val, avg_vol_val, market_cap
            )
            rows.append(row)

    df_q = pd.DataFrame(rows)
    df_q['report_date'] = pd.to_datetime(df_q['report_date'])
    df_q.sort_values(['ticker', 'report_date'], inplace=True)
    df_q.reset_index(drop=True, inplace=True)
    return df_q, sector_map


# ---------------------------------------------------------------------------
# 3. Compute forward returns and outcome labels
# ---------------------------------------------------------------------------

def _compute_outcomes(df_q, price_dict):
    """Attach real forward-return columns and binary outcome labels."""
    spy_px = price_dict['SPY']['Close']

    outcome_rows = []
    for idx, row in df_q.iterrows():
        tk = row['ticker']
        rd = row['report_date']
        oc = {}

        px = price_dict[tk]['Close']
        dr_s = px.pct_change()
        v30_s = dr_s.rolling(30).std() * np.sqrt(252)

        vi = px.index[px.index >= rd]
        if len(vi) == 0:
            outcome_rows.append(oc)
            continue
        si = px.index.get_loc(vi[0])
        sp_ = float(px.iloc[si])
        if sp_ <= 0:
            outcome_rows.append(oc)
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

            # Excess return vs SPY
            try:
                ss = float(spy_px.asof(px.index[si]))
                se = float(spy_px.asof(px.index[ei]))
                if pd.notna(ss) and pd.notna(se) and ss > 0:
                    oc[f'excess_{w}d'] = er - (se - ss) / ss
            except Exception:
                pass

            ex = oc.get(f'excess_{w}d', np.nan)

            # Binary excess-drop labels
            for t in EXCESS_THRESH:
                if pd.notna(ex):
                    oc[f'exdrop_{int(t * 100)}_{w}d'] = 1 if ex <= -t else 0

            # Binary absolute-drop labels
            for t in DROP_THRESH:
                oc[f'drop_{int(t * 100)}_{w}d'] = 1 if er <= -t else 0

            # Vol-adjusted drop labels
            dv = cur_vol / np.sqrt(252) * np.sqrt(w)
            for sig in [1.0, 1.5, 2.0]:
                oc[f'voladj_{sig:.0f}sig_{w}d'] = 1 if er <= -(sig * dv) else 0

        outcome_rows.append(oc)

    outcome_df = pd.DataFrame(outcome_rows, index=df_q.index)
    df_q = pd.concat([df_q, outcome_df], axis=1)
    return df_q


# ---------------------------------------------------------------------------
# 4. Dev / holdout split
# ---------------------------------------------------------------------------

def _split_dev_hold(df_q, dev_frac=0.85):
    """Split by report_date: first 85% of dates -> dev, last 15% -> holdout."""
    dates_sorted = df_q['report_date'].sort_values().unique()
    cutoff_idx = int(len(dates_sorted) * dev_frac)
    cutoff_date = dates_sorted[cutoff_idx]

    df_dev = df_q[df_q['report_date'] < cutoff_date].copy().reset_index(drop=True)
    df_hold = df_q[df_q['report_date'] >= cutoff_date].copy().reset_index(drop=True)
    return df_dev, df_hold


# ---------------------------------------------------------------------------
# 5. Build df_daily (SPY daily close)
# ---------------------------------------------------------------------------

def _build_df_daily(price_dict):
    spy = price_dict['SPY']
    df_daily = pd.DataFrame({'date': spy.index, 'spy_close': spy['Close'].values})
    df_daily.reset_index(drop=True, inplace=True)
    return df_daily


# ---------------------------------------------------------------------------
# 6. Build SimFin-style DataFrames (multi-indexed by Ticker, Report Date)
# ---------------------------------------------------------------------------

def _build_simfin_frames(sector_map, trading_days):
    """Create df_inc, df_bal, df_cf with MultiIndex (Ticker, Report Date)."""
    tickers = [f'SYN{i:03d}' for i in range(N_TICKERS)]

    end_date = trading_days[-1]
    start_date = trading_days[0] + pd.DateOffset(months=6)
    quarter_starts = pd.date_range(start_date, end_date, freq='QS')
    quarter_starts = quarter_starts[-N_QUARTERS:] if len(quarter_starts) >= N_QUARTERS else quarter_starts

    inc_rows, bal_rows, cf_rows = [], [], []

    for tk in tickers:
        n_reports = random.randint(8, N_QUARTERS)
        n_reports = min(n_reports, len(quarter_starts))
        report_qs = sorted(random.sample(list(quarter_starts), n_reports))

        for qd in report_qs:
            report_date = qd + pd.Timedelta(days=random.randint(30, 60))

            revenue = np.random.uniform(1e8, 1e10)
            gross_profit = revenue * np.random.uniform(0.2, 0.7)
            op_income = gross_profit * np.random.uniform(0.1, 0.6)
            net_income = op_income * np.random.uniform(0.5, 0.9)

            inc_rows.append({
                'Ticker': tk,
                'Report Date': report_date,
                'Revenue': revenue,
                'Gross Profit': gross_profit,
                'Operating Income (Loss)': op_income,
                'Net Income': net_income,
                'Shares (Diluted)': np.random.uniform(5e7, 5e9),
                'Shares (Basic)': np.random.uniform(5e7, 5e9),
            })

            total_assets = np.random.uniform(5e8, 5e11)
            total_equity = total_assets * np.random.uniform(0.2, 0.6)
            total_liab = total_assets - total_equity
            cur_assets = total_assets * np.random.uniform(0.15, 0.5)
            cur_liab = total_liab * np.random.uniform(0.2, 0.5)
            cash = cur_assets * np.random.uniform(0.1, 0.5)
            total_debt = total_liab * np.random.uniform(0.3, 0.7)

            bal_rows.append({
                'Ticker': tk,
                'Report Date': report_date,
                'Total Assets': total_assets,
                'Total Equity': total_equity,
                'Total Liabilities': total_liab,
                'Total Current Assets': cur_assets,
                'Total Current Liabilities': cur_liab,
                'Cash, Cash Equivalents & Short Term Investments': cash,
                'Total Debt': total_debt,
            })

            net_cash_ops = net_income * np.random.uniform(0.8, 1.5)
            capex = -abs(revenue * np.random.uniform(0.02, 0.10))
            interest_exp = total_debt * np.random.uniform(0.01, 0.06) / 4

            cf_rows.append({
                'Ticker': tk,
                'Report Date': report_date,
                'Net Cash from Operating Activities': net_cash_ops,
                'Change in Fixed Assets & Intangibles': capex,
                'Interest Expense, Net': interest_exp,
            })

    df_inc = pd.DataFrame(inc_rows).set_index(['Ticker', 'Report Date'])
    df_bal = pd.DataFrame(bal_rows).set_index(['Ticker', 'Report Date'])
    df_cf = pd.DataFrame(cf_rows).set_index(['Ticker', 'Report Date'])

    return df_inc, df_bal, df_cf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating synthetic data for Drop Score v17 ...")

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 1. Trading days and prices
    print("  [1/6] Trading-day calendar ...")
    trading_days = _make_trading_days()
    print(f"        {len(trading_days)} business days")

    print("  [2/6] Price series ...")
    price_dict = _generate_prices(trading_days)
    print(f"        {len(price_dict)} tickers")

    # 2. Quarterly features
    print("  [3/6] Quarterly feature rows ...")
    df_q, sector_map = _build_quarterly_features(price_dict, trading_days)
    print(f"        {len(df_q):,} rows, {len(df_q.columns)} columns (pre-outcome)")

    # 3. Outcomes
    print("  [4/6] Forward returns & outcome labels ...")
    df_q = _compute_outcomes(df_q, price_dict)
    print(f"        {len(df_q.columns)} total columns")

    # 4. Split
    print("  [5/6] Dev / holdout split ...")
    df_dev, df_hold = _split_dev_hold(df_q, dev_frac=0.85)
    print(f"        Dev: {len(df_dev):,}  Holdout: {len(df_hold):,}")

    # 5. df_daily
    df_daily = _build_df_daily(price_dict)

    # 6. SimFin-style frames
    print("  [6/6] SimFin-style fundamentals ...")
    df_inc, df_bal, df_cf = _build_simfin_frames(sector_map, trading_days)
    print(f"        inc={len(df_inc):,}  bal={len(df_bal):,}  cf={len(df_cf):,}")

    # ── Save v13_intermediates.pkl ──
    intm_path = os.path.join(data_dir, 'v13_intermediates.pkl')
    with open(intm_path, 'wb') as f:
        pickle.dump({
            'df_q': df_q,
            'df_dev': df_dev,
            'df_hold': df_hold,
            'df_daily': df_daily,
        }, f)
    print(f"  Saved {intm_path}")

    # ── Save drop_score_cache.pkl ──
    cache_path = os.path.join(data_dir, 'drop_score_cache.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'df_inc': df_inc,
            'df_bal': df_bal,
            'df_cf': df_cf,
            'simfin_ts': time.time(),
            'prices': price_dict,
            'sector_map': sector_map,
            'unavailable_tickers': set(),
        }, f)
    print(f"  Saved {cache_path}")

    print("Done.")


if __name__ == '__main__':
    main()
