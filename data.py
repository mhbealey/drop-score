"""
Data loading: SimFin fundamentals, yFinance prices, sector mapping,
pickle caching, universe construction.
"""
import os, time, pickle
import simfin as sf
import yfinance as yf
import pandas as pd
import numpy as np

from config import (
    SIMFIN_KEY, SECTOR_ETFS, FORCE_RECOMPUTE,
)
from utils import strip_tz, elapsed


def setup_cache_dir():
    """Mount Google Drive if available, else use local dir."""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        cache_dir = '/content/drive/MyDrive/drop_score/'
        os.makedirs(cache_dir, exist_ok=True)
        print("  \u2705 Drive")
    except:
        cache_dir = './'
        print("  \u26a0\ufe0f  No Drive")
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
    """Build filtered stock universe."""
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=4)
    u_raw = set(df_inc.groupby('Ticker').size()[lambda x: x >= 8].index)
    for ds in [df_bal, df_cf]:
        u_raw &= set(ds.groupby('Ticker').size()[lambda x: x >= 8].index)
    universe_all = sorted(
        tk for tk in u_raw
        if df_inc.loc[tk].index.max() >= cutoff
        if tk in df_inc.index.get_level_values('Ticker')
    )
    universe = [tk for tk in universe_all if sector_map.get(tk) not in ('Financial',)]
    return universe


def download_prices(universe, cache, cache_path):
    """Download price data from yFinance, with caching and skip list."""
    price_dict = cache.get('prices', {})
    unavail = cache.get('unavailable_tickers', set())

    # Ensure benchmark / ETF prices
    for sym in list(SECTOR_ETFS.keys()) + ['SPY', '^VIX']:
        if sym not in price_dict:
            try:
                d = yf.download(sym, period="5y", progress=False)
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
        print(f"  Downloading {len(need)} ({len(unavail)} skipped)...")
        before_dl = len(price_dict)
        for bs in range(0, len(need), 100):
            batch = need[bs:bs + 100]
            try:
                data = yf.download(batch, period="5y", group_by="ticker",
                                   threads=True, progress=False)
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
            except:
                pass
            time.sleep(1 if min(bs + 100, len(need)) < 1000 else 3)
        still_missing = {tk for tk in need if tk not in price_dict}
        if still_missing:
            unavail.update(still_missing)
            cache['unavailable_tickers'] = unavail
        cache['prices'] = price_dict
        save_cache(cache, cache_path)
        print(f"  +{len(price_dict) - before_dl} new, {len(still_missing)} unavail")
    else:
        if unavail:
            print(f"  All cached \u2705 ({len(unavail)} on skip list)")
        else:
            print(f"  All cached \u2705")

    return price_dict, unavail


def derive_benchmarks(price_dict):
    """Extract SPY, VIX, and sector ETF return series."""
    spy_close = spy_ret = vix_series = None
    if 'SPY' in price_dict:
        sd = price_dict['SPY']
        spy_close = sd['Close'] if 'Close' in sd.columns else sd.iloc[:, 0]
        spy_ret = spy_close.pct_change()
    if '^VIX' in price_dict:
        vd = price_dict['^VIX']
        vix_series = vd['Close'] if 'Close' in vd.columns else vd.iloc[:, 0]
    sector_etf_ret = {}
    for etf, sec in SECTOR_ETFS.items():
        if etf in price_dict:
            px = (
                price_dict[etf]['Close']
                if 'Close' in price_dict[etf].columns
                else price_dict[etf].iloc[:, 0]
            )
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
