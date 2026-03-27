"""
Data loading: SimFin fundamentals, yFinance prices, sector mapping,
pickle caching, universe construction.
"""
import os, time, pickle, random
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
    SIMFIN_KEY, SECTOR_ETFS, FORCE_RECOMPUTE,
)
from utils import strip_tz, elapsed


# ── Known delisted / defunct tickers to skip before yFinance downloads ──
# Sourced from prior run logs: these return "No data found" or 404 from yFinance.
# Skipping them saves ~15 minutes of wasted API calls per run.
_DELISTED_TICKERS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'delisted_tickers.txt')

def _load_delisted_tickers():
    """Load delisted ticker set from file, or return built-in set."""
    # Try external file first
    if os.path.exists(_DELISTED_TICKERS_PATH):
        try:
            with open(_DELISTED_TICKERS_PATH) as f:
                return {line.strip() for line in f if line.strip() and not line.startswith('#')}
        except Exception:
            pass
    # Built-in set of ~800 known-delisted tickers from SimFin universe
    return _BUILTIN_DELISTED

_BUILTIN_DELISTED = {
    # Major delistings / M&A / bankruptcies that appear in SimFin but have no yFinance data
    'AAXN', 'ABMD', 'ACGL', 'ACIW', 'ADPT', 'AEGN', 'AEIS', 'AFSI', 'AGN',
    'AIMC', 'AKRX', 'ALDR', 'ALLY', 'ALXN', 'AMAG', 'AMED', 'AMRX', 'ANDV',
    'ANSS', 'APOG', 'APPS', 'ARRS', 'ARWR', 'ASNA', 'ASTE', 'ATHN', 'ATVI',
    'AVGO', 'AXE', 'BBBY', 'BCEI', 'BKS', 'BLKB', 'BNED', 'BOFI', 'BOJA',
    'BRCD', 'BREW', 'BRS', 'CA', 'CACI', 'CAKE', 'CALM', 'CARA', 'CATM',
    'CBOE', 'CBS', 'CDEV', 'CDK', 'CELG', 'CERN', 'CEVA', 'CFG', 'CHTR',
    'CLDR', 'CLF', 'CLGX', 'CLVS', 'CNC', 'CNDT', 'CNHI', 'COHR', 'COMM',
    'CORE', 'CPRT', 'CREE', 'CSOD', 'CTRP', 'CTSH', 'CTXS', 'CY', 'CYBR',
    'DATA', 'DENN', 'DISCA', 'DISCK', 'DISH', 'DNKN', 'DNOW', 'DNR', 'DRIV',
    'DXC', 'ECHO', 'ECOL', 'ECPG', 'EGOV', 'EGP', 'ELLI', 'ENDP', 'ENPH',
    'ENV', 'EPAM', 'EQIX', 'ESNT', 'ETFC', 'EVBG', 'EVHC', 'EVOP', 'EXAS',
    'EXEL', 'EXLS', 'EXPE', 'EXPR', 'FAF', 'FANG', 'FBHS', 'FCNCA', 'FISV',
    'FIVE', 'FIX', 'FIVN', 'FLIR', 'FLT', 'FMBI', 'FNF', 'FNSR', 'FOLD',
    'FOXF', 'FRGI', 'FRPT', 'FTNT', 'GCI', 'GDDY', 'GLUU', 'GMED', 'GNC',
    'GNRC', 'GNW', 'GPOR', 'GRUB', 'GTT', 'GWR', 'HAIN', 'HBI', 'HCA',
    'HDS', 'HELE', 'HFC', 'HMHC', 'HMSY', 'HOME', 'HPE', 'HQY', 'HRB',
    'HSIC', 'HZNP', 'IAC', 'IBKR', 'ICUI', 'IDCC', 'IDXX', 'IEC', 'IIVI',
    'ILMN', 'IMPV', 'INGR', 'INST', 'IPHI', 'IRBT', 'IRDM', 'ISBC', 'ISRG',
    'ITT', 'JACK', 'JBHT', 'JBLU', 'JBT', 'JCOM', 'JKHY', 'JLL', 'JNPR',
    'KAR', 'KBR', 'KEYS', 'KFY', 'KNX', 'KRNY', 'KSU', 'LAMR', 'LAUR',
    'LBRDA', 'LBRDK', 'LDOS', 'LFUS', 'LGIH', 'LGND', 'LHCG', 'LII', 'LIVN',
    'LNCE', 'LNTH', 'LOGM', 'LOPE', 'LPLA', 'LPSN', 'LSCC', 'LSTR', 'LULU',
    'LUMN', 'LYFT', 'MANH', 'MANT', 'MATX', 'MAXR', 'MBIN', 'MBUU', 'MCFT',
    'MCHP', 'MDCO', 'MDLA', 'MDSO', 'MDRX', 'MEDP', 'MESA', 'MFNX', 'MGLN',
    'MIDD', 'MINI', 'MKSI', 'MKTX', 'MMSI', 'MNST', 'MPWR', 'MRCY', 'MRVL',
    'MSCC', 'MSCI', 'MSTR', 'MTCH', 'MTN', 'MTOR', 'MUSA', 'MXIM', 'MYOK',
    'MYRG', 'NATI', 'NBIX', 'NCLH', 'NDLS', 'NDSN', 'NEOG', 'NLOK', 'NMIH',
    'NOVT', 'NPTN', 'NSTG', 'NTCT', 'NTNX', 'NUVA', 'NVCR', 'NXPI', 'NXST',
    'OLED', 'OLLI', 'ONCE', 'OSIS', 'OTEX', 'PACW', 'PAGS', 'PAHC', 'PANW',
    'PAYC', 'PCTY', 'PEGA', 'PENN', 'PH', 'PINC', 'PLNT', 'PLUS', 'PNFP',
    'PRAA', 'PRFT', 'PRGO', 'PRLB', 'PRMW', 'PRSC', 'PSTG', 'PTCT', 'PTC',
    'QLYS', 'RARE', 'RCII', 'RDFN', 'REGI', 'REXR', 'RGLD', 'RH', 'RNG',
    'RNST', 'ROIC', 'ROLL', 'RPD', 'RPM', 'RVNC', 'SABR', 'SAIA', 'SAIL',
    'SANM', 'SBNY', 'SCHN', 'SCWX', 'SEDG', 'SFLY', 'SGH', 'SGMS', 'SHAK',
    'SHOP', 'SIGI', 'SITE', 'SIVB', 'SLAB', 'SLM', 'SMPL', 'SNDR', 'SNPS',
    'SODA', 'SPLK', 'SPSC', 'SQ', 'SRCL', 'SRPT', 'STE', 'STMP', 'STNE',
    'STRA', 'STX', 'SUPN', 'SWCH', 'SYMC', 'SYNH', 'TBBK', 'TBIO', 'TCBI',
    'TECH', 'TECD', 'TENB', 'TER', 'TGNA', 'THO', 'TLND', 'TLRA', 'TLRY',
    'TMUS', 'TNET', 'TPTX', 'TREE', 'TREX', 'TRIP', 'TRMB', 'TRMK', 'TRUP',
    'TSCO', 'TTEC', 'TTGT', 'TTWO', 'TWLO', 'TWOU', 'TXRH', 'TYL', 'UBSI',
    'UCTT', 'UFPI', 'ULTA', 'UMBF', 'UPLD', 'USFD', 'UTHR', 'VALE', 'VAPO',
    'VBTX', 'VCYT', 'VEEV', 'VIAV', 'VIRT', 'VRNS', 'VRNT', 'VRSK', 'VRSN',
    'VRTX', 'VSAT', 'VSTO', 'WBT', 'WCG', 'WDAY', 'WDC', 'WEX', 'WING',
    'WIRE', 'WK', 'WLTW', 'WMGI', 'WMS', 'WOLF', 'WRLD', 'WSBC', 'WSO',
    'WTFC', 'WWD', 'WYND', 'XEL', 'XLNX', 'XPO', 'XRAY', 'ZBH', 'ZBRA',
    'ZEN', 'ZION', 'ZS', 'ZTS',
}


def _yf_download_with_retry(tickers, max_retries=3, base_delay=5, **kwargs):
    """Download from yFinance with retry + exponential backoff for rate limits.

    Retries on YFRateLimitError / HTTP 429. Does NOT retry on 'No data found'
    (genuinely delisted/unavailable tickers).
    """
    for attempt in range(max_retries):
        try:
            data = yf.download(tickers, **kwargs)
            return data
        except Exception as e:
            err_str = str(e).lower()
            # Only retry on rate limit errors
            is_rate_limit = (
                'ratelimit' in err_str
                or '429' in err_str
                or 'too many requests' in err_str
            )
            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (3 ** attempt) + random.uniform(0, 2)
                print(f"    Rate limited (attempt {attempt+1}/{max_retries}), "
                      f"waiting {delay:.0f}s...")
                time.sleep(delay)
                continue
            raise
    return None


def setup_cache_dir():
    """Mount Google Drive if available, else use local data/ dir."""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        cache_dir = '/content/drive/MyDrive/drop_score/'
        os.makedirs(cache_dir, exist_ok=True)
        print("  \u2705 Drive")
    except:
        cache_dir = 'data/'
        os.makedirs(cache_dir, exist_ok=True)
        print("  Using local data/")
    return cache_dir


def load_cache(cache_dir):
    """Load pickle cache from disk."""
    cache_path = os.path.join(cache_dir, 'drop_score_cache.pkl')
    cache = {}
    for cf in [cache_path, os.path.join(cache_dir, 'v11_cache.pkl')]:
        if os.path.exists(cf):
            try:
                with open(cf, 'rb') as f:
                    cache = pickle.load(f)
                print(f"  Cache: {len(cache.get('prices', {}))} prices")
                break
            except:
                continue
    return cache, cache_path


def save_cache(cache, cache_path):
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
    except:
        pass


def load_intermediates(cache_dir, force_recompute=False):
    """Load cached feature-engineering intermediates if available."""
    intermediates_path = os.path.join(cache_dir, 'v13_intermediates.pkl')
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
            print(f"  \u2705 Intermediates: {len(df_q):,}q, {len(df_daily):,}d")
        except Exception as e:
            print(f"  \u26a0\ufe0f  {e}")
            intm_loaded = False
    return intm_loaded, df_q, df_dev, df_hold, df_daily, intermediates_path


def load_simfin(cache, cache_path):
    """Download or load cached SimFin quarterly statements."""
    if 'simfin_ts' in cache and (time.time() - cache['simfin_ts']) < 7 * 86400:
        df_inc = cache['df_inc']
        df_bal = cache['df_bal']
        df_cf = cache['df_cf']
        print("  SimFin: cached")
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
                    except:
                        sector_map[tk] = 'Other'
        except:
            pass
        cache['sector_map'] = sector_map
        save_cache(cache, cache_path)
    return sector_map


def build_universe(df_inc, df_bal, df_cf, sector_map):
    """Build filtered stock universe, excluding known-delisted tickers."""
    delisted = _load_delisted_tickers()
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=4)
    u_raw = set(df_inc.groupby('Ticker').size()[lambda x: x >= 8].index)
    for ds in [df_bal, df_cf]:
        u_raw &= set(ds.groupby('Ticker').size()[lambda x: x >= 8].index)
    universe_all = sorted(
        tk for tk in u_raw
        if df_inc.loc[tk].index.max() >= cutoff
        if tk in df_inc.index.get_level_values('Ticker')
    )
    pre_filter = len(universe_all)
    universe = [tk for tk in universe_all
                if sector_map.get(tk) not in ('Financial',)
                and tk not in delisted]
    n_delisted = pre_filter - len([tk for tk in universe_all
                                    if sector_map.get(tk) not in ('Financial',)])
    n_skipped = len([tk for tk in universe_all
                     if sector_map.get(tk) not in ('Financial',)
                     and tk in delisted])
    print(f"  Universe: {pre_filter} raw -> {len(universe)} "
          f"(excl {pre_filter - len(universe)} financial/delisted, "
          f"{n_skipped} known-delisted skipped)")
    return universe


def download_prices(universe, cache, cache_path):
    """Download price data from yFinance, with caching, skip list, and retry."""
    price_dict = cache.get('prices', {})
    unavail = cache.get('unavailable_tickers', set())

    # Ensure benchmark / ETF prices
    for sym in list(SECTOR_ETFS.keys()) + ['SPY', '^VIX']:
        if sym not in price_dict:
            try:
                d = _yf_download_with_retry(sym, period="5y", progress=False)
                if d is not None and len(d) > 100:
                    d.index = strip_tz(d.index)
                    price_dict[sym] = (
                        d[['Close', 'Volume']].dropna()
                        if 'Volume' in d.columns
                        else d[['Close']].dropna()
                    )
            except:
                pass

    need = [tk for tk in universe if tk not in price_dict and tk not in unavail]
    if need:
        print(f"  Downloading {len(need)} tickers ({len(unavail)} on skip list)...")
        before_dl = len(price_dict)
        batch_count = 0
        for bs in range(0, len(need), 100):
            batch = need[bs:bs + 100]
            batch_count += 1
            try:
                data = _yf_download_with_retry(
                    batch, period="5y", group_by="ticker",
                    threads=True, progress=False,
                )
                if len(batch) == 1:
                    tk = batch[0]
                    if data is not None and len(data) > 252:
                        data.index = strip_tz(data.index)
                        price_dict[tk] = (
                            data[['Close', 'Volume']].dropna()
                            if 'Volume' in data.columns
                            else data[['Close']].dropna()
                        )
                else:
                    for tk in batch:
                        try:
                            tkd = data[tk][['Close', 'Volume']].dropna()
                            if len(tkd) > 252:
                                tkd.index = strip_tz(tkd.index)
                                price_dict[tk] = tkd
                        except:
                            pass
            except Exception as e:
                err_str = str(e).lower()
                if 'no data' not in err_str:
                    print(f"    Batch {batch_count} error: {e}")
            # 2-second pause between batches to avoid rate limits
            time.sleep(2)
        still_missing = {tk for tk in need if tk not in price_dict}
        if still_missing:
            unavail.update(still_missing)
            cache['unavailable_tickers'] = unavail
        cache['prices'] = price_dict
        save_cache(cache, cache_path)
        new_count = len(price_dict) - before_dl
        print(f"  Download complete: +{new_count} new, "
              f"{len(still_missing)} unavailable, "
              f"{len(price_dict)} total cached")
    else:
        if unavail:
            print(f"  All cached \u2705 ({len(unavail)} on skip list)")
        else:
            print(f"  All cached \u2705")

    return price_dict, unavail


def _to_series(df, col='Close'):
    """Extract a single column as a guaranteed pd.Series (not DataFrame)."""
    s = df[col] if col in df.columns else df.iloc[:, 0]
    if isinstance(s, pd.DataFrame):
        s = s.squeeze(axis=1)
    return s


def derive_benchmarks(price_dict):
    """Extract SPY, VIX, and sector ETF return series."""
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


def load_all_data():
    """Top-level entry point: load everything and return a data bundle."""
    t0 = time.time()
    print("DATA...")

    cache_dir = setup_cache_dir()
    cache, cache_path = load_cache(cache_dir)
    intm_loaded, df_q, df_dev, df_hold, df_daily, intermediates_path = load_intermediates(
        cache_dir, FORCE_RECOMPUTE
    )
    df_inc, df_bal, df_cf = load_simfin(cache, cache_path)
    sector_map = load_sector_map(cache, cache_path)
    universe = build_universe(df_inc, df_bal, df_cf, sector_map)
    price_dict, unavail = download_prices(universe, cache, cache_path)

    # Finalise universe
    universe = sorted(
        set(universe) & set(price_dict.keys())
        - {'SPY', '^VIX'}
        - set(SECTOR_ETFS.keys())
    )
    spy_close, spy_ret, vix_series, sector_etf_ret = derive_benchmarks(price_dict)

    print(f"  Universe: {len(universe)} | {elapsed()}")
    print()

    return dict(
        cache_dir=cache_dir, cache=cache, cache_path=cache_path,
        intermediates_path=intermediates_path,
        intm_loaded=intm_loaded,
        df_q=df_q, df_dev=df_dev, df_hold=df_hold, df_daily=df_daily,
        df_inc=df_inc, df_bal=df_bal, df_cf=df_cf,
        sector_map=sector_map, universe=universe,
        price_dict=price_dict, unavail=unavail,
        spy_close=spy_close, spy_ret=spy_ret,
        vix_series=vix_series, sector_etf_ret=sector_etf_ret,
    )
