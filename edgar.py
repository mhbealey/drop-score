"""
EDGAR fundamental data: XBRL field mapping, CIK lookup,
quarterly fact extraction, gap-fill for SimFin-missing tickers.

SEC EDGAR provides free, unlimited access to all US public company filings.
Rate limit: 10 req/sec with proper User-Agent header.
"""
import os, time, json, pickle
import urllib.request
import urllib.error
import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════
# SEC User-Agent (required by SEC EDGAR)
# ═══════════════════════════════════════════════════════════════

SEC_HEADERS = {'User-Agent': 'DropScore michael@dropscore.dev'}
SEC_RATE_DELAY = 0.12  # 10 req/sec


# ═══════════════════════════════════════════════════════════════
# XBRL → SimFin field mapping
# ═══════════════════════════════════════════════════════════════

XBRL_TO_FIELD = {
    'Revenue': [
        'Revenues',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
        'SalesRevenueNet',
        'SalesRevenueGoodsNet',
        'SalesRevenueServicesNet',
    ],
    'Gross Profit': [
        'GrossProfit',
    ],
    'Operating Income (Loss)': [
        'OperatingIncomeLoss',
    ],
    'Net Income': [
        'NetIncomeLoss',
        'ProfitLoss',
        'NetIncomeLossAvailableToCommonStockholdersBasic',
    ],
    'Interest Expense, Net': [
        'InterestExpense',
        'InterestExpenseDebt',
    ],
    'Total Assets': [
        'Assets',
    ],
    'Total Equity': [
        'StockholdersEquity',
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
    ],
    'Total Liabilities': [
        'Liabilities',
    ],
    'Total Debt': [
        'LongTermDebt',
        'LongTermDebtAndCapitalLeaseObligations',
        'DebtAndCapitalLeaseObligations',
    ],
    'Total Current Assets': [
        'AssetsCurrent',
    ],
    'Total Current Liabilities': [
        'LiabilitiesCurrent',
    ],
    'Cash, Cash Equivalents & Short Term Investments': [
        'CashAndCashEquivalentsAtCarryingValue',
        'CashCashEquivalentsAndShortTermInvestments',
    ],
    'Net Cash from Operating Activities': [
        'NetCashProvidedByUsedInOperatingActivities',
        'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',
    ],
    'Change in Fixed Assets & Intangibles': [
        'PaymentsToAcquirePropertyPlantAndEquipment',
        'PaymentsToAcquireProductiveAssets',
    ],
    'Shares (Diluted)': [
        'WeightedAverageNumberOfDilutedSharesOutstanding',
    ],
    'Shares (Basic)': [
        'WeightedAverageNumberOfShareOutstandingBasicAndDiluted',
        'CommonStockSharesOutstanding',
    ],
}

# Balance sheet fields are point-in-time (any filing period is ok).
# Income/cash flow fields need quarterly duration (~90 days).
BALANCE_SHEET_FIELDS = {
    'Total Assets', 'Total Liabilities', 'Total Equity',
    'Total Current Assets', 'Total Current Liabilities',
    'Cash, Cash Equivalents & Short Term Investments',
    'Total Debt',
    'Shares (Diluted)', 'Shares (Basic)',
}

# SimFin core fields (must match for build_quarterly_row compatibility)
SIMFIN_INCOME_FIELDS = [
    'Revenue', 'Gross Profit', 'Operating Income (Loss)',
    'Net Income', 'Interest Expense, Net',
]
SIMFIN_BALANCE_FIELDS = [
    'Total Assets', 'Total Equity', 'Total Liabilities', 'Total Debt',
    'Total Current Assets', 'Total Current Liabilities',
    'Cash, Cash Equivalents & Short Term Investments',
    'Shares (Diluted)', 'Shares (Basic)',
]
SIMFIN_CASHFLOW_FIELDS = [
    'Net Cash from Operating Activities',
    'Change in Fixed Assets & Intangibles',
]

ALL_CORE_FIELDS = set(SIMFIN_INCOME_FIELDS + SIMFIN_BALANCE_FIELDS + SIMFIN_CASHFLOW_FIELDS)


# ═══════════════════════════════════════════════════════════════
# CIK Lookup
# ═══════════════════════════════════════════════════════════════

def load_cik_map(cache_dir='data/'):
    """Load or download SEC ticker→CIK mapping. Refresh if >30 days old."""
    path = os.path.join(cache_dir, 'sec_cik_map.json')
    if os.path.exists(path):
        age_days = (time.time() - os.path.getmtime(path)) / 86400
        if age_days < 30:
            with open(path) as f:
                return json.load(f)

    print("  Downloading SEC CIK map...")
    url = 'https://www.sec.gov/files/company_tickers.json'
    req = urllib.request.Request(url, headers=SEC_HEADERS)
    resp = urllib.request.urlopen(req, timeout=30)
    raw = json.loads(resp.read())

    cik_map = {}
    for entry in raw.values():
        ticker = entry['ticker'].upper().replace('.', '-')
        cik = str(entry['cik_str']).zfill(10)
        cik_map[ticker] = cik

    os.makedirs(cache_dir, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(cik_map, f)
    print(f"  CIK map: {len(cik_map)} tickers")
    return cik_map


# ═══════════════════════════════════════════════════════════════
# EDGAR Fact Fetching
# ═══════════════════════════════════════════════════════════════

def _fetch_company_facts(cik):
    """Fetch all XBRL facts for a company from EDGAR."""
    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'
    req = urllib.request.Request(url, headers=SEC_HEADERS)
    resp = urllib.request.urlopen(req, timeout=15)
    return json.loads(resp.read())


# ═══════════════════════════════════════════════════════════════
# Quarterly Parsing
# ═══════════════════════════════════════════════════════════════

def parse_edgar_facts(facts_json, ticker):
    """Extract quarterly financial data from EDGAR company facts.

    Returns dict: {(ticker, end_date_str): {field: value, 'filed': date_str}}

    Key logic:
    - Balance sheet items: point-in-time, take value at period end
    - Income/cash flow items: only take quarterly duration (60-115 days)
    - 10-Q and 10-K filings only
    - First matching XBRL tag wins per field (no double-counting)
    """
    us_gaap = facts_json.get('facts', {}).get('us-gaap', {})
    if not us_gaap:
        return {}

    quarterly_data = {}  # {(ticker, end_date): {field: value, 'filed': str}}

    for simfin_field, xbrl_tags in XBRL_TO_FIELD.items():
        is_bs = simfin_field in BALANCE_SHEET_FIELDS

        for tag_name in xbrl_tags:
            if tag_name not in us_gaap:
                continue

            units = us_gaap[tag_name].get('units', {})
            for unit_type in ['USD', 'shares']:
                if unit_type not in units:
                    continue

                for entry in units[unit_type]:
                    form = entry.get('form', '')
                    if form not in ('10-Q', '10-K'):
                        continue

                    end = entry.get('end', '')
                    start = entry.get('start', '')
                    filed = entry.get('filed', '')
                    val = entry.get('val')

                    if not end or val is None:
                        continue

                    key = (ticker, end)

                    if is_bs:
                        if key not in quarterly_data:
                            quarterly_data[key] = {'filed': filed}
                        if simfin_field not in quarterly_data[key]:
                            quarterly_data[key][simfin_field] = val
                    else:
                        if not start:
                            continue
                        try:
                            days = (pd.to_datetime(end) - pd.to_datetime(start)).days
                        except Exception:
                            continue
                        if 60 <= days <= 115:
                            if key not in quarterly_data:
                                quarterly_data[key] = {'filed': filed}
                            if simfin_field not in quarterly_data[key]:
                                quarterly_data[key][simfin_field] = val

            break  # First matching tag wins, don't double-count

    return quarterly_data


# ═══════════════════════════════════════════════════════════════
# Build SimFin-compatible DataFrames from EDGAR data
# ═══════════════════════════════════════════════════════════════

def _edgar_to_simfin_frames(all_quarterly_data):
    """Convert parsed EDGAR data into 3 SimFin-compatible MultiIndex DataFrames.

    CRITICAL: Every (ticker, date) row is added to ALL THREE frames,
    even if some fields are NaN. This ensures tickers pass build_universe's
    intersection filter (requires presence in all 3 frames).
    """
    inc_rows = []
    bal_rows = []
    cf_rows = []

    for (ticker, end_date), fields in all_quarterly_data.items():
        try:
            rd = pd.Timestamp(end_date)
        except Exception:
            continue

        # Check if this row has ANY useful data
        has_any = any(f in fields for f in ALL_CORE_FIELDS)
        if not has_any:
            continue

        # Always create a row in ALL three frames
        inc_row = {'Ticker': ticker, 'Report Date': rd}
        for f in SIMFIN_INCOME_FIELDS:
            inc_row[f] = fields.get(f, np.nan)
        inc_rows.append(inc_row)

        bal_row = {'Ticker': ticker, 'Report Date': rd}
        for f in SIMFIN_BALANCE_FIELDS:
            bal_row[f] = fields.get(f, np.nan)
        bal_rows.append(bal_row)

        cf_row = {'Ticker': ticker, 'Report Date': rd}
        for f in SIMFIN_CASHFLOW_FIELDS:
            cf_row[f] = fields.get(f, np.nan)
        cf_rows.append(cf_row)

    def _to_multiindex(rows):
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.set_index(['Ticker', 'Report Date']).sort_index()
        df = df[~df.index.duplicated(keep='first')]
        return df

    return _to_multiindex(inc_rows), _to_multiindex(bal_rows), _to_multiindex(cf_rows)


# ═══════════════════════════════════════════════════════════════
# Filing metadata
# ═══════════════════════════════════════════════════════════════

def extract_filing_metadata(all_quarterly_data):
    """Extract filing dates and compute filing delay days."""
    metadata = {}
    for (ticker, end_date), fields in all_quarterly_data.items():
        filed = fields.get('filed', '')
        if not filed or not end_date:
            continue
        try:
            delay = (pd.to_datetime(filed) - pd.to_datetime(end_date)).days
        except Exception:
            continue
        metadata[(ticker, end_date)] = {
            'filed': filed,
            'filing_delay_days': delay,
        }
    return metadata


# ═══════════════════════════════════════════════════════════════
# Main EDGAR pipeline
# ═══════════════════════════════════════════════════════════════

def load_edgar_cache(cache_dir='data/'):
    """Load cached EDGAR quarterly data."""
    path = os.path.join(cache_dir, 'edgar_fundamentals.pkl')
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def save_edgar_cache(all_data, cache_dir='data/'):
    """Save EDGAR quarterly data to pickle cache."""
    path = os.path.join(cache_dir, 'edgar_fundamentals.pkl')
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(all_data, f)


def fetch_edgar_fundamentals(tickers_to_fetch, cik_map, cache_dir='data/'):
    """Fetch EDGAR data for a list of tickers.

    Respects SEC rate limits (10 req/sec). Saves progress every 100 tickers.
    """
    all_data = load_edgar_cache(cache_dir)
    already_fetched = {k[0] for k in all_data}

    to_fetch = [t for t in tickers_to_fetch
                if t in cik_map and t not in already_fetched]

    if not to_fetch:
        print(f"  EDGAR: {len(already_fetched)} cached, 0 to fetch")
        return all_data

    print(f"  EDGAR: {len(already_fetched)} cached, {len(to_fetch)} to fetch")

    fetched_count = 0
    failed_count = 0

    for i, ticker in enumerate(to_fetch):
        cik = cik_map[ticker]
        try:
            facts_json = _fetch_company_facts(cik)
            quarterly = parse_edgar_facts(facts_json, ticker)

            if quarterly:
                all_data.update(quarterly)
                fetched_count += 1
            else:
                failed_count += 1

            time.sleep(SEC_RATE_DELAY)

            if (i + 1) % 100 == 0:
                save_edgar_cache(all_data, cache_dir)
                print(f"    EDGAR progress: {i+1}/{len(to_fetch)} "
                      f"(+{fetched_count} ok, {failed_count} empty)")

        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"    Rate limited at {i}, sleeping 120s...")
                time.sleep(120)
                try:
                    facts_json = _fetch_company_facts(cik)
                    quarterly = parse_edgar_facts(facts_json, ticker)
                    if quarterly:
                        all_data.update(quarterly)
                        fetched_count += 1
                except Exception:
                    failed_count += 1
            else:
                failed_count += 1
            time.sleep(SEC_RATE_DELAY)
        except Exception:
            failed_count += 1
            time.sleep(SEC_RATE_DELAY)

    save_edgar_cache(all_data, cache_dir)
    unique_tickers = len({k[0] for k in all_data})
    print(f"    EDGAR done: +{fetched_count} tickers ({failed_count} empty/failed)")
    print(f"    EDGAR cache total: {unique_tickers} tickers, "
          f"{len(all_data)} quarter-rows")

    return all_data


def merge_edgar_into_simfin(df_inc, df_bal, df_cf, edgar_data, sector_map):
    """Merge EDGAR data into SimFin DataFrames.

    - EDGAR-only tickers: add as new rows to ALL three frames
    - SimFin tickers with sparse data: fill NaN fields only (SimFin priority)

    Returns updated (df_inc, df_bal, df_cf, edgar_filing_metadata).
    """
    if not edgar_data:
        return df_inc, df_bal, df_cf, {}

    e_inc, e_bal, e_cf = _edgar_to_simfin_frames(edgar_data)
    filing_meta = extract_filing_metadata(edgar_data)

    simfin_tickers = set(df_inc.index.get_level_values('Ticker').unique())

    edgar_tickers = set()
    for edf in [e_inc, e_bal, e_cf]:
        if len(edf) > 0:
            edgar_tickers |= set(edf.index.get_level_values('Ticker').unique())

    new_tickers = edgar_tickers - simfin_tickers
    overlap_tickers = edgar_tickers & simfin_tickers

    n_simfin = len(simfin_tickers)
    n_new = len(new_tickers)
    n_overlap = len(overlap_tickers)

    # Add EDGAR-only tickers to all 3 frames
    def _append_new(df_simfin, df_edgar):
        if len(df_edgar) == 0:
            return df_simfin
        new_rows = df_edgar[
            df_edgar.index.get_level_values('Ticker').isin(new_tickers)
        ].copy()
        if len(new_rows) == 0:
            return df_simfin
        # Ensure columns match
        for col in df_simfin.columns:
            if col not in new_rows.columns:
                new_rows[col] = np.nan
        # Only keep columns that exist in simfin frame
        common_cols = [c for c in df_simfin.columns if c in new_rows.columns]
        new_rows = new_rows[common_cols]
        return pd.concat([df_simfin, new_rows]).sort_index()

    df_inc = _append_new(df_inc, e_inc)
    df_bal = _append_new(df_bal, e_bal)
    df_cf = _append_new(df_cf, e_cf)

    # Gap-fill: for overlapping tickers, fill NaN in SimFin with EDGAR values
    filled_count = 0
    for df_simfin, df_edgar in [(df_inc, e_inc), (df_bal, e_bal), (df_cf, e_cf)]:
        if len(df_edgar) == 0:
            continue
        overlap_idx = df_simfin.index.intersection(df_edgar.index)
        for idx in overlap_idx:
            for col in df_simfin.columns:
                if col not in df_edgar.columns:
                    continue
                try:
                    sim_val = df_simfin.loc[idx, col]
                    if pd.isna(sim_val):
                        edgar_val = df_edgar.loc[idx, col]
                        if pd.notna(edgar_val):
                            df_simfin.loc[idx, col] = edgar_val
                            filled_count += 1
                except Exception:
                    continue

    # Add EDGAR tickers to sector_map
    for tk in new_tickers:
        if tk not in sector_map:
            sector_map[tk] = 'Other'

    total_tickers = len(set(df_inc.index.get_level_values('Ticker').unique()))
    total_quarters = len(df_inc)

    # Diagnostic: check how many EDGAR tickers appear in all 3 frames
    inc_tickers = set(df_inc.index.get_level_values('Ticker').unique())
    bal_tickers = set(df_bal.index.get_level_values('Ticker').unique())
    cf_tickers = set(df_cf.index.get_level_values('Ticker').unique())
    in_all_3 = inc_tickers & bal_tickers & cf_tickers
    edgar_in_all_3 = new_tickers & in_all_3

    print(f"\n  {'='*50}")
    print(f"  FUNDAMENTAL COVERAGE")
    print(f"    SimFin:        {n_simfin} tickers")
    print(f"    EDGAR new:    +{n_new} tickers ({len(edgar_in_all_3)} in all 3 frames)")
    print(f"    Gap-fill:     +{filled_count} fields in {n_overlap} overlap tickers")
    print(f"    Combined:      {total_tickers} tickers ({total_quarters} quarters)")
    print(f"  {'='*50}\n")

    return df_inc, df_bal, df_cf, filing_meta


def get_edgar_sector_map(cik_map, tickers, cache_dir='data/'):
    """SIC-based sector mapping from SEC for EDGAR-only tickers."""
    path = os.path.join(cache_dir, 'sec_sic_map.json')
    if os.path.exists(path):
        age_days = (time.time() - os.path.getmtime(path)) / 86400
        if age_days < 30:
            with open(path) as f:
                return json.load(f)

    sic_map = {}
    try:
        url = 'https://www.sec.gov/files/company_tickers_exchange.json'
        req = urllib.request.Request(url, headers=SEC_HEADERS)
        resp = urllib.request.urlopen(req, timeout=30)
        raw = json.loads(resp.read())
        fields = raw.get('fields', [])
        data_rows = raw.get('data', [])

        ticker_idx = fields.index('ticker') if 'ticker' in fields else None
        sic_idx = fields.index('sic') if 'sic' in fields else None

        if ticker_idx is not None and sic_idx is not None:
            for row in data_rows:
                tk = str(row[ticker_idx]).upper().replace('.', '-')
                sic = row[sic_idx]
                if sic:
                    sic_map[tk] = _sic_to_sector(int(sic))

        os.makedirs(cache_dir, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(sic_map, f)
    except Exception:
        pass

    return sic_map


def _sic_to_sector(sic):
    """Map SIC code to sector name matching our sector categories."""
    if 3570 <= sic <= 3579 or 3660 <= sic <= 3699 or 7370 <= sic <= 7379:
        return 'Technology'
    elif 2830 <= sic <= 2836 or 3841 <= sic <= 3851 or 8000 <= sic <= 8099:
        return 'Healthcare'
    elif 6000 <= sic <= 6999:
        return 'Financial'
    elif 1300 <= sic <= 1389 or 2900 <= sic <= 2999 or 4900 <= sic <= 4949:
        return 'Energy'
    elif 3400 <= sic <= 3569 or 3580 <= sic <= 3659 or 3700 <= sic <= 3799:
        return 'Industrials'
    elif 5000 <= sic <= 5199 or 5200 <= sic <= 5999 or 2000 <= sic <= 2111:
        return 'Consumer'
    elif 4800 <= sic <= 4899 or 4950 <= sic <= 4999:
        return 'Utilities'
    else:
        return 'Other'


# ═══════════════════════════════════════════════════════════════
# Data QA
# ═══════════════════════════════════════════════════════════════

def run_data_qa(df_inc, df_bal, df_cf, edgar_data, sp_tickers):
    """Comprehensive data quality audit after merge."""
    print("\n  ═══ DATA QA ═══")

    all_tickers = set(df_inc.index.get_level_values('Ticker').unique())
    edgar_tickers = {k[0] for k in edgar_data} if edgar_data else set()
    simfin_only = all_tickers - edgar_tickers
    edgar_only = edgar_tickers - (all_tickers - edgar_tickers)
    overlap = all_tickers & edgar_tickers
    sp_covered = all_tickers & sp_tickers if sp_tickers else set()

    print(f"  Total tickers with fundamentals: {len(all_tickers)}")
    print(f"  SimFin-only: {len(simfin_only)}")
    print(f"  EDGAR-only: {len(edgar_only & all_tickers)}")
    print(f"  Both sources: {len(overlap)}")
    print(f"  S&P coverage: {len(sp_covered)}/{len(sp_tickers)}")

    # Temporal coverage
    for label, tickers in [('SimFin', simfin_only), ('EDGAR', edgar_only & all_tickers)]:
        if not tickers:
            continue
        mask = df_inc.index.get_level_values('Ticker').isin(tickers)
        if mask.sum() == 0:
            continue
        sub = df_inc[mask]
        quarters_per = sub.groupby('Ticker').size()
        dates = sub.index.get_level_values('Report Date')
        print(f"  {label}: median {quarters_per.median():.0f} quarters/ticker, "
              f"range {quarters_per.min()}-{quarters_per.max()}")
        print(f"  {label}: date range {dates.min().date()} to {dates.max().date()}")

    # Field completeness
    print(f"\n  Field completeness:")
    for df_frame, fields, label in [
        (df_inc, SIMFIN_INCOME_FIELDS, 'Income'),
        (df_bal, SIMFIN_BALANCE_FIELDS, 'Balance'),
        (df_cf, SIMFIN_CASHFLOW_FIELDS, 'CashFlow'),
    ]:
        for col in fields:
            if col not in df_frame.columns:
                print(f"    {col}: MISSING from {label} frame")
                continue
            pct_null = df_frame[col].isna().mean()
            status = 'OK' if pct_null < 0.3 else 'WARN' if pct_null < 0.6 else 'HIGH'
            print(f"    {col}: {pct_null:.0%} null [{status}]")

    # Duplicate check
    for label, df_frame in [('Income', df_inc), ('Balance', df_bal), ('CashFlow', df_cf)]:
        dupes = df_frame.index.duplicated().sum()
        if dupes > 0:
            print(f"  WARNING: {dupes} duplicate (ticker, date) rows in {label}")

    print(f"  ═══ END DATA QA ═══\n")

    return {
        'total': len(all_tickers),
        'simfin_only': len(simfin_only),
        'edgar_only': len(edgar_only & all_tickers),
        'overlap': len(overlap),
        'sp_covered': len(sp_covered),
        'sp_total': len(sp_tickers) if sp_tickers else 0,
    }


def run_feature_qa(df_dev, df_hold, edgar_tickers, fcols_q):
    """Feature quality audit after feature engineering."""
    print("\n  ═══ FEATURE QA ═══")

    # Track source
    if edgar_tickers:
        dev_edgar = df_dev[df_dev['ticker'].isin(edgar_tickers)]
        dev_simfin = df_dev[~df_dev['ticker'].isin(edgar_tickers)]
    else:
        dev_edgar = pd.DataFrame()
        dev_simfin = df_dev

    print(f"  Feature rows: SimFin={len(dev_simfin):,} EDGAR={len(dev_edgar):,}")

    # Feature completeness by source
    warn_feats = []
    for feat in fcols_q:
        sf_null = dev_simfin[feat].isna().mean() if len(dev_simfin) > 0 else 0
        ed_null = dev_edgar[feat].isna().mean() if len(dev_edgar) > 0 else 1.0
        if ed_null > 0.5 and sf_null < 0.3:
            warn_feats.append((feat, sf_null, ed_null))

    if warn_feats:
        print(f"  Features with high EDGAR null rate:")
        for feat, sf_n, ed_n in warn_feats[:10]:
            print(f"    {feat}: SimFin {sf_n:.0%} null, EDGAR {ed_n:.0%} null")
    else:
        print(f"  All features have acceptable null rates")

    # Inf/NaN audit
    total_inf = 0
    for feat in fcols_q:
        if feat in df_dev.columns:
            n_inf = np.isinf(df_dev[feat]).sum()
            total_inf += n_inf
    print(f"  Total inf values across features: {total_inf}")

    # Target distribution by source
    target_cols = [c for c in df_dev.columns if c.startswith('exdrop_')]
    for tgt in target_cols[:3]:
        sf_rate = dev_simfin[tgt].mean() if tgt in dev_simfin.columns and len(dev_simfin) > 0 else 0
        ed_rate = dev_edgar[tgt].mean() if tgt in dev_edgar.columns and len(dev_edgar) > 0 else 0
        print(f"  {tgt}: SimFin rate={sf_rate:.3f}, EDGAR rate={ed_rate:.3f}")

    print(f"  ═══ END FEATURE QA ═══\n")

    return {
        'n_simfin_rows': len(dev_simfin),
        'n_edgar_rows': len(dev_edgar),
        'n_warn_feats': len(warn_feats),
        'total_inf': total_inf,
    }
