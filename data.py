"""
Data loading: SimFin + EDGAR fundamentals, multi-source price waterfall,
sector mapping, pickle caching, universe construction,
training/tradeable split, S&P index tickers.
"""
import os, time, pickle, random, json
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

import simfin as sf
import yfinance as yf
import pandas as pd
import numpy as np

# Monkey-patch: strip deprecated date_parser kwarg so simfin works with pandas 2.x+
import simfin.load
_original_read_csv = pd.read_csv
def _patched_read_csv(*args, **kwargs):
    kwargs.pop('date_parser', None)
    return _original_read_csv(*args, **kwargs)
pd.read_csv = _patched_read_csv

from config import (
    SIMFIN_KEY, FMP_KEY, SECTOR_ETFS, FORCE_RECOMPUTE, VOL_FLOOR, log,
)
from utils import strip_tz, elapsed
from edgar import (
    load_cik_map, fetch_edgar_fundamentals, merge_edgar_into_simfin,
    get_edgar_sector_map, load_edgar_cache, run_data_qa,
    edgar_field_diagnostic,
)


# ═══════════════════════════════════════════════════════════════
# Cache helpers
# ═══════════════════════════════════════════════════════════════

def setup_cache_dir() -> str:
    """Mount Google Drive if available, else use local data/ dir."""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        cache_dir = '/content/drive/MyDrive/drop_score/'
        os.makedirs(cache_dir, exist_ok=True)
        log.info("  Drive mounted")
    except Exception:
        cache_dir = 'data/'
        os.makedirs(cache_dir, exist_ok=True)
        log.info("  Using local data/")
    return cache_dir


def load_cache(cache_dir: str) -> Tuple[dict, str]:
    """Load pickle cache from disk. Returns (cache_dict, cache_path)."""
    cache_path = os.path.join(cache_dir, 'drop_score_cache.pkl')
    cache = {}
    for cf in [cache_path, os.path.join(cache_dir, 'v11_cache.pkl')]:
        if os.path.exists(cf):
            try:
                with open(cf, 'rb') as f:
                    cache = pickle.load(f)
                log.info(f"  Cache: {len(cache.get('prices', {}))} prices")
                break
            except Exception:
                continue
    return cache, cache_path


def save_cache(cache: dict, cache_path: str) -> None:
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
    except Exception:
        pass


def load_intermediates(cache_dir, force_recompute=False):
    """Load cached feature-engineering intermediates if available."""
    intermediates_path = os.path.join(cache_dir, 'v15_intermediates.pkl')
    intm_loaded = False
    df_q = df_dev = df_hold = df_daily = None
    if not force_recompute and os.path.exists(intermediates_path):
        try:
            with open(intermediates_path, 'rb') as f:
                intm = pickle.load(f)
            df_q = intm['df_q']
            df_dev = intm['df_dev']
            df_hold = intm['df_hold']
            df_daily = intm['df_daily']
            intm_loaded = True
            log.info(f"  Intermediates: {len(df_q):,}q, {len(df_daily):,}d")
        except Exception as e:
            log.warning(f"  Intermediates error: {e}")
            intm_loaded = False
    return intm_loaded, df_q, df_dev, df_hold, df_daily, intermediates_path


# ═══════════════════════════════════════════════════════════════
# SimFin fundamentals
# ═══════════════════════════════════════════════════════════════

def load_simfin(cache, cache_path):
    """Download or load cached SimFin quarterly statements."""
    if 'simfin_ts' in cache and (time.time() - cache['simfin_ts']) < 7 * 86400:
        df_inc = cache['df_inc']
        df_bal = cache['df_bal']
        df_cf = cache['df_cf']
        log.info("  SimFin: cached")
    else:
        sf.set_api_key(SIMFIN_KEY)
        sf.set_data_dir('~/simfin_data/')
        df_inc = sf.load_income(variant='quarterly', market='us')
        df_bal = sf.load_balance(variant='quarterly', market='us')
        df_cf = sf.load_cashflow(variant='quarterly', market='us')
        cache['df_inc'] = df_inc
        cache['df_bal'] = df_bal
        cache['df_cf'] = df_cf
        cache['simfin_ts'] = time.time()
        save_cache(cache, cache_path)
    return df_inc, df_bal, df_cf


def load_sector_map(cache, cache_path):
    """Build sector mapping from SimFin companies data."""
    sector_map = cache.get('sector_map', {})
    if not sector_map:
        try:
            sf.set_api_key(SIMFIN_KEY)
            sf.set_data_dir('~/simfin_data/')
            df_co = sf.load_companies(market='us')
            if df_co is not None and 'IndustryId' in df_co.columns:
                for tk in df_co.index:
                    try:
                        iid = int(df_co.loc[tk, 'IndustryId'])
                        if iid in range(101000, 104000):
                            sector_map[tk] = 'Technology'
                        elif iid in range(104000, 105000):
                            sector_map[tk] = 'Healthcare'
                        elif iid in range(105000, 106000):
                            sector_map[tk] = 'Financial'
                        elif iid in range(106000, 107000):
                            sector_map[tk] = 'Energy'
                        elif iid in range(107000, 108000):
                            sector_map[tk] = 'Industrials'
                        elif iid in range(108000, 110000):
                            sector_map[tk] = 'Consumer'
                        elif iid in range(110000, 112000):
                            sector_map[tk] = 'Utilities'
                        else:
                            sector_map[tk] = 'Other'
                    except Exception:
                        sector_map[tk] = 'Other'
        except Exception:
            pass
        cache['sector_map'] = sector_map
        save_cache(cache, cache_path)
    return sector_map


# ═══════════════════════════════════════════════════════════════
# Universe construction
# ═══════════════════════════════════════════════════════════════

def build_universe(df_inc: pd.DataFrame, df_bal: pd.DataFrame,
                   df_cf: pd.DataFrame, sector_map: Dict[str, str],
                   min_quarters: int = 8) -> List[str]:
    """Build filtered stock universe (exclude Financials only)."""
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=4)
    u_raw = set(df_inc.groupby('Ticker').size()[lambda x: x >= min_quarters].index)
    for ds in [df_bal, df_cf]:
        u_raw &= set(ds.groupby('Ticker').size()[lambda x: x >= min_quarters].index)
    universe_all = sorted(
        tk for tk in u_raw
        if df_inc.loc[tk].index.max() >= cutoff
        if tk in df_inc.index.get_level_values('Ticker')
    )
    universe = [tk for tk in universe_all if sector_map.get(tk) not in ('Financial',)]
    n_fin = len(universe_all) - len(universe)
    log.info(f"  Universe: {len(universe_all)} raw -> {len(universe)} (excl {n_fin} financial)")
    return universe


def classify_tickers(price_dict: Dict[str, pd.DataFrame],
                     vol_floor: int = VOL_FLOOR,
                     months_threshold: int = 6) -> Tuple[Set[str], Set[str]]:
    """Classify tickers as tradeable vs delisted-with-history.

    Returns (tradeable, delisted_with_history).
    tradeable: recent prices (within months_threshold) AND avg volume >= vol_floor
    delisted_with_history: has prices but not tradeable (kept for training)
    """
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=months_threshold)
    tradeable = set()
    delisted_with_history = set()
    for tk, pxd in price_dict.items():
        if tk in ('SPY', '^VIX') or tk in SECTOR_ETFS:
            continue
        if len(pxd) == 0:
            continue
        last_date = pxd.index.max()
        if last_date < cutoff:
            delisted_with_history.add(tk)
            continue
        # Check volume in most recent 30 trading days
        if 'Volume' in pxd.columns and len(pxd) >= 30:
            avg_vol = pxd['Volume'].iloc[-30:].mean()
            if hasattr(avg_vol, 'item'):
                avg_vol = avg_vol.item()
            if avg_vol < vol_floor:
                delisted_with_history.add(tk)
                continue
        tradeable.add(tk)
    return tradeable, delisted_with_history


# ═══════════════════════════════════════════════════════════════
# Multi-source price waterfall
# ═══════════════════════════════════════════════════════════════

def _yf_download_with_retry(tickers, max_retries=3, base_delay=5, **kwargs):
    """Download from yFinance with retry + exponential backoff for rate limits."""
    for attempt in range(max_retries):
        try:
            data = yf.download(tickers, **kwargs)
            return data
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = (
                'ratelimit' in err_str
                or '429' in err_str
                or 'too many requests' in err_str
            )
            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (3 ** attempt) + random.uniform(0, 2)
                log.warning(f"    Rate limited (attempt {attempt+1}/{max_retries}), "
                            f"waiting {delay:.0f}s...")
                time.sleep(delay)
                continue
            raise
    return None


def _add_price(price_dict, tk, df, min_rows=60):
    """Add a ticker's price data to the dict if it has enough rows."""
    if df is None or len(df) < min_rows:
        return False
    df = df.copy()
    df.index = strip_tz(df.index)
    if 'Close' in df.columns and 'Volume' in df.columns:
        price_dict[tk] = df[['Close', 'Volume']].dropna()
    elif 'Close' in df.columns:
        price_dict[tk] = df[['Close']].dropna()
    else:
        return False
    return True


def download_all_prices(universe, cache, cache_path):
    """Multi-source price waterfall. Returns (price_dict, unavail_set, simfin_price_tickers).

    Sources tried in order:
    1. Local cache (instant)
    2. SimFin daily share prices (bulk)
    3. yFinance batch (chunks of 500)
    4. FMP historical (cap 200 requests)
    5. yFinance individual (cap 100)
    """
    price_dict = cache.get('prices', {})
    unavail = cache.get('unavailable_tickers', set())
    counts = {'cache': 0, 'simfin': 0, 'yf_batch': 0, 'fmp': 0, 'yf_individual': 0}
    simfin_price_tickers = set()  # Track which tickers have SimFin prices

    # Clear stale unavail set (retry after 7 days)
    unavail_ts = cache.get('unavail_ts', 0)
    if time.time() - unavail_ts > 7 * 86400 and unavail:
        log.info(f"  Clearing {len(unavail)} stale unavailable tickers (>7d old)")
        unavail = set()
        cache['unavailable_tickers'] = unavail
        cache['unavail_ts'] = time.time()
        save_cache(cache, cache_path)

    # Ensure benchmark / ETF prices
    for sym in list(SECTOR_ETFS.keys()) + ['SPY', '^VIX']:
        if sym not in price_dict:
            try:
                d = _yf_download_with_retry(sym, period="5y", progress=False)
                if d is not None and len(d) > 100:
                    _add_price(price_dict, sym, d, min_rows=100)
            except Exception:
                pass

    # ── Always load SimFin price ticker list (needed for Universe A filtering) ──
    sp = None
    try:
        sf.set_api_key(SIMFIN_KEY)
        sf.set_data_dir('~/simfin_data/')
        sp = sf.load_shareprices(market='us', variant='daily')
        if sp is not None and len(sp) > 0:
            simfin_price_tickers = set(sp.index.get_level_values('Ticker'))
            log.info(f"  SimFin price universe: {len(simfin_price_tickers)} tickers")
    except Exception as e:
        log.warning(f"  SimFin price ticker list: {e}")

    # What do we still need?
    all_need = [tk for tk in universe if tk not in price_dict and tk not in unavail]
    cached_count = len([tk for tk in universe if tk in price_dict])
    counts['cache'] = cached_count

    if not all_need:
        skipped = len([tk for tk in universe if tk in unavail])
        log.info(f"  Cache: {cached_count} prices | {skipped} unavailable | "
                 f"{len(all_need)} to download")
        return price_dict, unavail, simfin_price_tickers

    log.info(f"  Need prices for {len(all_need)} tickers ({cached_count} cached, "
             f"{len(unavail)} known-unavailable)...")

    # ── Source 2: SimFin daily share prices (use already-loaded sp) ──
    need = [tk for tk in all_need if tk not in price_dict]
    if need and sp is not None and len(sp) > 0:
        try:
            sf_tickers = simfin_price_tickers
            for tk in need:
                if tk in sf_tickers:
                    try:
                        tkd = sp.loc[tk]
                        rename = {}
                        for col in tkd.columns:
                            cl = col.lower()
                            if 'close' in cl:
                                rename[col] = 'Close'
                            elif 'volume' in cl:
                                rename[col] = 'Volume'
                        if rename:
                            tkd = tkd.rename(columns=rename)
                        if _add_price(price_dict, tk, tkd):
                            counts['simfin'] += 1
                    except Exception:
                        pass
            log.info(f"    SimFin prices: +{counts['simfin']}")
            cache['prices'] = price_dict
            save_cache(cache, cache_path)
        except Exception as e:
            log.warning(f"    SimFin prices: {e}")

    # ── Source 3: yFinance batch (chunks of 500) ──
    need = [tk for tk in all_need if tk not in price_dict]
    if need:
        before = len(price_dict)
        for bs in range(0, len(need), 500):
            batch = need[bs:bs + 500]
            try:
                data = _yf_download_with_retry(
                    batch, period="5y", group_by="ticker",
                    threads=True, progress=False,
                )
                if data is not None:
                    if len(batch) == 1:
                        tk = batch[0]
                        _add_price(price_dict, tk, data)
                    else:
                        for tk in batch:
                            try:
                                tkd = data[tk][['Close', 'Volume']].dropna()
                                _add_price(price_dict, tk, tkd)
                            except Exception:
                                pass
            except Exception as e:
                err_str = str(e).lower()
                if 'no data' not in err_str:
                    log.warning(f"    yFinance batch error: {e}")
            time.sleep(5)
        counts['yf_batch'] = len(price_dict) - before - counts['simfin']
        if counts['yf_batch'] > 0:
            log.info(f"    yFinance batch: +{counts['yf_batch']}")
        cache['prices'] = price_dict
        save_cache(cache, cache_path)

    # ── Source 4: FMP historical (cap 200) ──
    need = [tk for tk in all_need if tk not in price_dict]
    fmp_key = FMP_KEY
    if fmp_key and need:
        fmp_cap = min(200, len(need))
        for tk in need[:fmp_cap]:
            try:
                url = (f"https://financialmodelingprep.com/api/v3/"
                       f"historical-price-full/{tk}?timeseries=1260&apikey={fmp_key}")
                resp = urllib.request.urlopen(url, timeout=10)
                raw = json.loads(resp.read())
                hist = raw.get('historical', [])
                if len(hist) > 60:
                    df = pd.DataFrame(hist)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date').sort_index()
                    df = df.rename(columns={'close': 'Close', 'volume': 'Volume'})
                    if 'Close' in df.columns:
                        cols = ['Close']
                        if 'Volume' in df.columns:
                            cols.append('Volume')
                        if _add_price(price_dict, tk, df[cols]):
                            counts['fmp'] += 1
            except Exception:
                pass
            time.sleep(0.3)
        if counts['fmp'] > 0:
            log.info(f"    FMP historical: +{counts['fmp']}")
        cache['prices'] = price_dict
        save_cache(cache, cache_path)

    # ── Source 5: yFinance individual (cap 100) ──
    need = [tk for tk in all_need if tk not in price_dict]
    if need:
        yf_ind_cap = min(100, len(need))
        for tk in need[:yf_ind_cap]:
            try:
                d = _yf_download_with_retry(tk, period="5y", progress=False)
                if d is not None and _add_price(price_dict, tk, d):
                    counts['yf_individual'] += 1
            except Exception:
                pass
            time.sleep(1)
        if counts['yf_individual'] > 0:
            log.info(f"    yFinance individual: +{counts['yf_individual']}")
        cache['prices'] = price_dict
        save_cache(cache, cache_path)

    # ── Update skip list with tickers that failed ALL sources ──
    still_missing = {tk for tk in all_need if tk not in price_dict}
    if still_missing:
        unavail.update(still_missing)
        cache['unavailable_tickers'] = unavail
        cache['unavail_ts'] = time.time()
        save_cache(cache, cache_path)

    # ── Summary ──
    total_new = counts['simfin'] + counts['yf_batch'] + counts['fmp'] + counts['yf_individual']
    total = len([tk for tk in universe if tk in price_dict])
    log.info(f"\n  {'='*45}")
    log.info(f"  PRICE DOWNLOAD SUMMARY")
    log.info(f"    Cache hit:            {counts['cache']:>5}")
    log.info(f"    SimFin prices:        {counts['simfin']:>5}")
    log.info(f"    yFinance batch:       {counts['yf_batch']:>5}")
    log.info(f"    FMP historical:       {counts['fmp']:>5}")
    log.info(f"    yFinance individual:  {counts['yf_individual']:>5}")
    log.info(f"    Total:              {total:>5,}")
    log.info(f"    Still missing:        {len(still_missing):>5}")
    log.info(f"  {'='*45}")

    return price_dict, unavail, simfin_price_tickers


# ═══════════════════════════════════════════════════════════════
# S&P Index tickers
# ═══════════════════════════════════════════════════════════════

def get_sp_index_tickers() -> Set[str]:
    """S&P 400+600 tickers from static CSV files."""
    tickers = set()
    for f in ['data/sp400_tickers.csv', 'data/sp600_tickers.csv']:
        if os.path.exists(f):
            with open(f) as fh:
                tickers.update(line.strip().replace('.', '-') for line in fh if line.strip())
    log.info(f"  S&P index: {len(tickers)} tickers from static files")
    return tickers


# ═══════════════════════════════════════════════════════════════
# Benchmarks
# ═══════════════════════════════════════════════════════════════

def _to_series(df, col='Close'):
    """Extract a single column as a guaranteed pd.Series (not DataFrame)."""
    s = df[col] if col in df.columns else df.iloc[:, 0]
    if isinstance(s, pd.DataFrame):
        s = s.squeeze(axis=1)
    return s


def derive_benchmarks(price_dict: Dict[str, pd.DataFrame]) -> Tuple[
        Optional[pd.Series], Optional[pd.Series],
        Optional[pd.Series], Dict[str, pd.Series]]:
    """Extract SPY, VIX, and sector ETF return series.

    Returns (spy_close, spy_ret, vix_series, sector_etf_ret).
    """
    spy_close = spy_ret = vix_series = None
    if 'SPY' in price_dict:
        spy_close = _to_series(price_dict['SPY'])
        spy_ret = spy_close.pct_change()
    if '^VIX' in price_dict:
        vix_series = _to_series(price_dict['^VIX'])
    sector_etf_ret = {}
    for etf, sec in SECTOR_ETFS.items():
        if etf in price_dict:
            px = _to_series(price_dict[etf])
            sector_etf_ret[sec] = px.pct_change()
            sector_etf_ret[sec + '_close'] = px
    return spy_close, spy_ret, vix_series, sector_etf_ret


# ═══════════════════════════════════════════════════════════════
# Top-level data loader
# ═══════════════════════════════════════════════════════════════

def load_all_data() -> dict:
    """Top-level entry point: load everything and return a data bundle dict."""
    t0 = time.time()
    log.info("DATA...")

    cache_dir = setup_cache_dir()
    cache, cache_path = load_cache(cache_dir)
    intm_loaded, df_q, df_dev, df_hold, df_daily, intermediates_path = load_intermediates(
        cache_dir, FORCE_RECOMPUTE
    )
    df_inc, df_bal, df_cf = load_simfin(cache, cache_path)
    sector_map = load_sector_map(cache, cache_path)

    # ── Track SimFin-only tickers BEFORE EDGAR merge ──
    simfin_tickers_all = set(df_inc.index.get_level_values('Ticker').unique())
    simfin_universe = build_universe(df_inc, df_bal, df_cf, sector_map)
    simfin_universe_set = set(simfin_universe)  # Pure SimFin universe (pre-EDGAR)
    # Capture SimFin's max report date BEFORE EDGAR merge (for holdout cutoff)
    simfin_max_date = df_inc.index.get_level_values('Report Date').max()
    sp_tickers = get_sp_index_tickers()

    simfin_tickers_with_data = set(
        df_inc.groupby('Ticker').size()[lambda x: x >= 4].index
    )

    # Phase 2: Expanded EDGAR coverage
    # 1) ALL S&P 400+600 tickers not in SimFin
    # 2) ALL SimFin tickers with insufficient data (<8 quarters)
    # 3) SimFin tickers with >50% NaN in critical fields
    needs_fundamentals = set()

    # S&P tickers missing from SimFin
    sp_missing = sp_tickers - simfin_tickers_with_data
    needs_fundamentals |= sp_missing

    # SimFin tickers with sparse data
    all_simfin_tickers = set(df_inc.index.get_level_values('Ticker').unique())
    sparse_simfin = set(
        df_inc.groupby('Ticker').size()[lambda x: (x >= 1) & (x < 8)].index
    )
    needs_fundamentals |= sparse_simfin

    # SimFin tickers with high NaN rate in critical fields
    for col in ['Revenue', 'Total Assets', 'Net Income']:
        if col in df_inc.columns:
            null_rate = df_inc.groupby('Ticker')[col].apply(lambda x: x.isna().mean())
            high_null = set(null_rate[null_rate > 0.5].index)
            needs_fundamentals |= high_null

    # Any universe ticker without SimFin data
    needs_fundamentals |= set(simfin_universe) - simfin_tickers_with_data

    log.info(f"  EDGAR candidates: {len(needs_fundamentals)} tickers "
             f"({len(sp_missing)} S&P missing, {len(sparse_simfin)} sparse)")

    edgar_data = {}
    edgar_filing_meta = {}
    edgar_tickers = set()
    try:
        cik_map = load_cik_map(cache_dir)
        sic_sectors = get_edgar_sector_map(cik_map, needs_fundamentals, cache_dir)
        for tk, sec in sic_sectors.items():
            if tk not in sector_map:
                sector_map[tk] = sec

        edgar_data = fetch_edgar_fundamentals(
            sorted(needs_fundamentals), cik_map, cache_dir
        )
        edgar_tickers = {k[0] for k in edgar_data}

        # EDGAR field diagnostic (shows null rates after re-parse)
        edgar_field_diagnostic(edgar_data)

        df_inc, df_bal, df_cf, edgar_filing_meta = merge_edgar_into_simfin(
            df_inc, df_bal, df_cf, edgar_data, sector_map
        )

        # Run data QA
        run_data_qa(df_inc, df_bal, df_cf, edgar_data, sp_tickers)

    except Exception as e:
        log.warning(f"  EDGAR failed: {e}")
        import traceback
        traceback.print_exc()

    # Rebuild universe with EDGAR-expanded data (lower min to 4)
    universe = build_universe(df_inc, df_bal, df_cf, sector_map, min_quarters=4)

    # Diagnostic: S&P coverage after EDGAR
    universe_set = set(universe)
    sp_in_universe = sp_tickers & universe_set
    log.info(f"  S&P in universe after EDGAR: {len(sp_in_universe)}/{len(sp_tickers)}")

    price_dict, unavail, simfin_price_tickers = download_all_prices(universe, cache, cache_path)
    log.info(f"  SimFin price tickers: {len(simfin_price_tickers)}")

    training_universe = sorted(
        set(universe) & set(price_dict.keys())
        - {'SPY', '^VIX'}
        - set(SECTOR_ETFS.keys())
    )

    tradeable, delisted_with_history = classify_tickers(price_dict)
    tradeable_tickers = set(training_universe) & tradeable

    spy_close, spy_ret, vix_series, sector_etf_ret = derive_benchmarks(price_dict)

    log.info(f"  Training: {len(training_universe)} stocks | "
             f"Tradeable: {len(tradeable_tickers)} for walk-forward "
             f"({len(delisted_with_history)} delisted with history)")
    log.info(f"  {elapsed()}")
    log.info("")

    return dict(
        cache_dir=cache_dir, cache=cache, cache_path=cache_path,
        intermediates_path=intermediates_path,
        intm_loaded=intm_loaded,
        df_q=df_q, df_dev=df_dev, df_hold=df_hold, df_daily=df_daily,
        df_inc=df_inc, df_bal=df_bal, df_cf=df_cf,
        sector_map=sector_map,
        universe=training_universe,
        simfin_universe=sorted(simfin_universe_set),
        simfin_price_tickers=simfin_price_tickers,
        simfin_max_date=simfin_max_date,
        tradeable_tickers=tradeable_tickers,
        price_dict=price_dict, unavail=unavail,
        spy_close=spy_close, spy_ret=spy_ret,
        vix_series=vix_series, sector_etf_ret=sector_etf_ret,
        edgar_filing_meta=edgar_filing_meta,
        edgar_tickers=edgar_tickers,
        sp_tickers=sp_tickers,
    )
